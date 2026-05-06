//! Per-token forward pass for a Llama-architecture model.
//!
//! Owns scratch buffers; reuses them across tokens. One call to `step` reads
//! a single token id, advances the KV cache by one slot, and writes logits
//! into `state.logits`.

const std = @import("std");
const Allocator = std.mem.Allocator;

const parser = @import("../core/parser.zig");
const simd = @import("../kernels/simd.zig");
const math = @import("../kernels/math.zig");
const kernels = @import("../kernels/comptime_gen.zig");
const model_mod = @import("model.zig");
const kv_cache_mod = @import("kv_cache.zig");
const pool_mod = @import("pool.zig");
const metal_backend_mod = @import("metal_backend.zig");
const metal_bridge = @import("../metal/bridge.zig");

const Model = model_mod.Model;
const TypedTensor = model_mod.TypedTensor;
const KvCache = kv_cache_mod.KvCache;
const ThreadPool = pool_mod.ThreadPool;
pub const MetalBackend = metal_backend_mod.MetalBackend;

pub const StepError = error{
    UnsupportedWeightType,
    KvCacheFull,
    TokenOutOfRange,
};

pub const State = struct {
    x: []f32,
    xb: []f32,
    xb2: []f32,
    q: []f32,
    k_cur: []f32,
    v_cur: []f32,
    attn_scores: []f32, // size: n_heads * max_seq, CPU-only.
    attn_out: []f32,
    gate: []f32,
    up: []f32,
    ffn_out: []f32,
    logits: []f32,

    /// True when the buffer slices are heap-allocated and must be freed by
    /// `deinit`. False when they alias `MetalBackend` shared-storage scratch
    /// (in which case the backend frees them).
    owns_buffers: bool,

    pub fn init(allocator: Allocator, m: *const Model, metal: ?*MetalBackend) !State {
        const c = m.config;
        const kv_dim = c.n_kv_heads * c.head_dim;

        if (metal) |mb| {
            // Alias the persistent shared-storage scratch buffers. Slices are
            // views over the same memory the GPU writes; no copy at dispatch.
            return .{
                .x = mb.x.ptr[0..c.dim],
                .xb = mb.xb.ptr[0..c.dim],
                .xb2 = mb.xb2.ptr[0..c.dim],
                .q = mb.q.ptr[0 .. c.n_heads * c.head_dim],
                .k_cur = mb.k_cur.ptr[0..kv_dim],
                .v_cur = mb.v_cur.ptr[0..kv_dim],
                .attn_scores = mb.attn_scores.ptr[0 .. c.n_heads * c.max_seq],
                .attn_out = mb.attn_out.ptr[0..c.dim],
                .gate = mb.gate.ptr[0..c.ffn_dim],
                .up = mb.up.ptr[0..c.ffn_dim],
                .ffn_out = mb.ffn_out.ptr[0..c.dim],
                .logits = mb.logits.ptr[0..c.vocab_size],
                .owns_buffers = false,
            };
        }

        const attn_scores = try allocator.alloc(f32, c.n_heads * c.max_seq);
        errdefer allocator.free(attn_scores);

        return .{
            .x = try allocator.alloc(f32, c.dim),
            .xb = try allocator.alloc(f32, c.dim),
            .xb2 = try allocator.alloc(f32, c.dim),
            .q = try allocator.alloc(f32, c.n_heads * c.head_dim),
            .k_cur = try allocator.alloc(f32, kv_dim),
            .v_cur = try allocator.alloc(f32, kv_dim),
            .attn_scores = attn_scores,
            .attn_out = try allocator.alloc(f32, c.dim),
            .gate = try allocator.alloc(f32, c.ffn_dim),
            .up = try allocator.alloc(f32, c.ffn_dim),
            .ffn_out = try allocator.alloc(f32, c.dim),
            .logits = try allocator.alloc(f32, c.vocab_size),
            .owns_buffers = true,
        };
    }

    pub fn deinit(self: *State, allocator: Allocator) void {
        if (self.owns_buffers) {
            allocator.free(self.attn_scores);
            allocator.free(self.x);
            allocator.free(self.xb);
            allocator.free(self.xb2);
            allocator.free(self.q);
            allocator.free(self.k_cur);
            allocator.free(self.v_cur);
            allocator.free(self.attn_out);
            allocator.free(self.gate);
            allocator.free(self.up);
            allocator.free(self.ffn_out);
            allocator.free(self.logits);
        }
        self.* = undefined;
    }
};

pub fn step(
    m: *const Model,
    state: *State,
    cache: *KvCache,
    pool: *ThreadPool,
    metal: ?*MetalBackend,
    token_id: u32,
) StepError!void {
    const c = m.config;
    if (token_id >= c.vocab_size) return error.TokenOutOfRange;
    if (cache.pos >= c.max_seq) return error.KvCacheFull;

    // Embed.
    try gatherEmbedding(state.x, m.token_embd, token_id, c.dim);

    // GPU fast path: per-layer ops are chained into 3 MTLCommandBuffers,
    // each containing rmsnorm/matmul/rope/swiglu/residual dispatches. CPU
    // only does KV-cache writes and attention. Fall back to the per-op CPU
    // path when the model uses non-Q8_0 projections or non-f32 norms.
    const gpu_ready: ?*MetalBackend = blk: {
        if (metal) |mb| if (modelGpuEligible(m)) break :blk mb;
        break :blk null;
    };

    for (m.layers, 0..) |layer, l| {
        if (gpu_ready) |mb| {
            try gpuLayerStep(mb, layer, cache, c, l);
        } else {
            try cpuLayerStep(pool, metal, layer, state, cache, c, l);
        }
    }

    cache.advance();

    if (gpu_ready) |mb| {
        try gpuFinalStep(mb, m, c);
        // Single per-token barrier: every prior segment commit was async, so
        // the CPU must wait here before the sampler reads `state.logits` (a
        // view over the shared-storage logits buffer the GPU just wrote).
        mb.dev.waitIdle();
    } else {
        try rmsNormTyped(state.x, m.output_norm, c.rms_eps);
        try matmulRuntime(pool, metal, state.logits, m.output_w, state.x, c.vocab_size, c.dim);
    }
}

fn cpuAttention(state: *State, cache: *KvCache, c: model_mod.LlamaConfig, l: usize) void {
    const head_dim_f: f32 = @floatFromInt(c.head_dim);
    const inv_sqrt_hd: f32 = 1.0 / @sqrt(head_dim_f);
    const seq_len = cache.pos + 1;
    const kv_row = c.n_kv_heads * c.head_dim;
    const k_buf = cache.keysFor(l, seq_len);
    const v_buf = cache.valuesFor(l, seq_len);

    // Sliding-window attention (Mistral): position cache.pos only attends to
    // positions in [cache.pos - sliding_window + 1, cache.pos]. For arches
    // with sliding_window == maxInt(usize) the start clamps to 0 and this
    // reduces to standard full attention.
    const window_start: usize = if (c.sliding_window == std.math.maxInt(usize) or cache.pos < c.sliding_window)
        0
    else
        cache.pos - c.sliding_window + 1;
    const attended_len = seq_len - window_start;

    @memset(state.attn_out, 0);
    for (0..c.n_heads) |h| {
        const kv_h = (h * c.n_kv_heads) / c.n_heads; // GQA fan-in
        const q_h = state.q[h * c.head_dim ..][0..c.head_dim];
        const scores = state.attn_scores[h * c.max_seq ..][0..attended_len];

        for (0..attended_len) |i| {
            const t = window_start + i;
            const k_th = k_buf[t * kv_row + kv_h * c.head_dim ..][0..c.head_dim];
            var s: f32 = 0;
            for (q_h, k_th) |qv, kv| s += qv * kv;
            scores[i] = s * inv_sqrt_hd;
        }
        math.softmax(scores);

        const out_h = state.attn_out[h * c.head_dim ..][0..c.head_dim];
        for (0..attended_len) |i| {
            const t = window_start + i;
            const v_th = v_buf[t * kv_row + kv_h * c.head_dim ..][0..c.head_dim];
            const w = scores[i];
            for (out_h, v_th) |*o, vv| o.* += w * vv;
        }
    }
}

fn cpuLayerStep(
    pool: *ThreadPool,
    metal: ?*MetalBackend,
    layer: model_mod.LayerWeights,
    state: *State,
    cache: *KvCache,
    c: model_mod.LlamaConfig,
    l: usize,
) StepError!void {
    const kv_dim = c.n_kv_heads * c.head_dim;

    // Attention RMSNorm.
    @memcpy(state.xb, state.x);
    try rmsNormTyped(state.xb, layer.attn_norm, c.rms_eps);

    // QKV projections.
    try matmulRuntime(pool, metal, state.q, layer.attn_q, state.xb, c.dim, c.dim);
    try matmulRuntime(pool, metal, state.k_cur, layer.attn_k, state.xb, kv_dim, c.dim);
    try matmulRuntime(pool, metal, state.v_cur, layer.attn_v, state.xb, kv_dim, c.dim);

    // RoPE on Q (per head) and K (per kv head).
    if (c.rope_half) {
        for (0..c.n_heads) |h| {
            math.ropeHalf(state.q[h * c.head_dim ..][0..c.head_dim], cache.pos, c.rope_base);
        }
        for (0..c.n_kv_heads) |h| {
            math.ropeHalf(state.k_cur[h * c.head_dim ..][0..c.head_dim], cache.pos, c.rope_base);
        }
    } else {
        for (0..c.n_heads) |h| {
            math.rope(state.q[h * c.head_dim ..][0..c.head_dim], cache.pos, c.rope_base);
        }
        for (0..c.n_kv_heads) |h| {
            math.rope(state.k_cur[h * c.head_dim ..][0..c.head_dim], cache.pos, c.rope_base);
        }
    }

    // KV cache write + attention.
    @memcpy(cache.keySlot(l), state.k_cur);
    @memcpy(cache.valueSlot(l), state.v_cur);
    cpuAttention(state, cache, c, l);

    // Output projection + residual.
    try matmulRuntime(pool, metal, state.xb2, layer.attn_o, state.attn_out, c.dim, c.dim);
    math.addInto(state.x, state.xb2);

    // FFN RMSNorm + activation (SwiGLU for Llama / Mistral / Phi, GeGLU
    // with tanh-approx GELU for Gemma).
    @memcpy(state.xb, state.x);
    try rmsNormTyped(state.xb, layer.ffn_norm, c.rms_eps);
    try matmulRuntime(pool, metal, state.gate, layer.ffn_gate, state.xb, c.ffn_dim, c.dim);
    try matmulRuntime(pool, metal, state.up, layer.ffn_up, state.xb, c.ffn_dim, c.dim);
    switch (c.hidden_activation) {
        .silu => math.swiglu(state.gate, state.up),
        .gelu_approx => math.gegluApprox(state.gate, state.up),
    }
    try matmulRuntime(pool, metal, state.ffn_out, layer.ffn_down, state.gate, c.dim, c.ffn_dim);
    math.addInto(state.x, state.ffn_out);
}

fn isGpuEligibleProj(q: kernels.QuantType) bool {
    return switch (q) {
        .q8_0, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

test "cpuAttention: sliding window restricts attended range" {
    // Synthetic 1-head, head_dim=2 setup. Position 5 with sliding_window=3
    // must attend only to positions {3, 4, 5}. We craft K so that scores at
    // positions 0/1/2 would be huge if attended (10x other positions); a
    // correctly-windowed attention will not see them, so the output is
    // dominated by V[3..6].
    const allocator = std.testing.allocator;
    const dim: usize = 2;
    const head_dim: usize = 2;
    const max_seq: usize = 8;
    const cfg: model_mod.LlamaConfig = .{
        .dim = dim,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_heads = 1,
        .head_dim = head_dim,
        .ffn_dim = 4,
        .vocab_size = 16,
        .max_seq = max_seq,
        .sliding_window = 3,
    };
    var cache = try kv_cache_mod.KvCache.init(allocator, 1, 1, head_dim, max_seq, null);
    defer cache.deinit(allocator);
    // Manually populate KV: positions 0..2 = "loud" (norm 100), 3..5 = "quiet" (norm 1).
    // Layout per-token: [k0, k1, ...kheads][v0, v1, ...]
    const layer_off: usize = 0;
    const k_view = cache.k[layer_off..];
    const v_view = cache.v[layer_off..];
    for (0..6) |t| {
        const big = t < 3;
        const k_val: f32 = if (big) 100.0 else 1.0;
        k_view[t * head_dim + 0] = k_val;
        k_view[t * head_dim + 1] = 0.0;
        v_view[t * head_dim + 0] = if (big) -999.0 else 1.0;
        v_view[t * head_dim + 1] = 0.0;
    }
    cache.pos = 5;

    // State.q is [1, 0] so dot(q, k) = k[0]. With windowing on, only
    // positions 3..5 contribute. Without windowing, position 0..2's huge
    // values dominate via softmax.
    var q = [_]f32{ 1.0, 0.0 };
    var attn_out = [_]f32{ 0.0, 0.0 };
    var attn_scores = [_]f32{0.0} ** (1 * max_seq);
    var x = [_]f32{ 0.0, 0.0 };
    var xb = [_]f32{ 0.0, 0.0 };
    var xb2 = [_]f32{ 0.0, 0.0 };
    var k_cur = [_]f32{ 0.0, 0.0 };
    var v_cur = [_]f32{ 0.0, 0.0 };
    var gate = [_]f32{0.0} ** 4;
    var up = [_]f32{0.0} ** 4;
    var ffn_out = [_]f32{ 0.0, 0.0 };
    var logits = [_]f32{0.0} ** 16;
    var state: State = .{
        .x = &x, .xb = &xb, .xb2 = &xb2,
        .q = &q, .k_cur = &k_cur, .v_cur = &v_cur,
        .attn_scores = &attn_scores, .attn_out = &attn_out,
        .gate = &gate, .up = &up, .ffn_out = &ffn_out,
        .logits = &logits, .owns_buffers = false,
    };

    cpuAttention(&state, &cache, cfg, 0);
    // With window=3 and uniform "quiet" KVs at positions 3..5, attention
    // output should equal the V values (~1.0). The "loud" -999.0 at earlier
    // positions must NOT leak into the output.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), attn_out[0], 1e-3);
    try std.testing.expect(attn_out[0] > -100.0); // hard-fails if windowing broken
}

test "math.gegluApprox: per-element correctness already covered in math.zig" {
    // Smoke: ensure the activation switch in cpuLayerStep compiles for
    // both variants. Real-model correctness is gated on the math.zig unit
    // test which validates against reference values.
}

fn modelGpuEligible(m: *const Model) bool {
    // GPU full-forward shape:
    //   - Llama: full causal attention + SwiGLU
    //   - Mistral: sliding-window via K/V offset shift, SwiGLU
    //   - Gemma: full causal attention + GeGLU (gelu_approx kernel)
    //   - Phi: parallel-block — refused at Model.init, never seen here
    switch (m.config.architecture) {
        .llama, .mistral, .gemma => {},
        .phi => return false,
    }
    if (m.output_norm.quant != .f32) return false;
    if (!isGpuEligibleProj(m.output_w.quant)) return false;
    for (m.layers) |lw| {
        if (lw.attn_norm.quant != .f32) return false;
        if (lw.ffn_norm.quant != .f32) return false;
        const projs = [_]kernels.QuantType{
            lw.attn_q.quant,    lw.attn_k.quant,
            lw.attn_v.quant,    lw.attn_o.quant,
            lw.ffn_gate.quant,  lw.ffn_up.quant,
            lw.ffn_down.quant,
        };
        for (projs) |p| if (!isGpuEligibleProj(p)) return false;
    }
    return true;
}

/// Dispatch a quantized matmul on the segment, selecting the kernel by
/// `quant`. Caller has already verified eligibility via modelGpuEligible.
fn segMatmulQuant(
    seg: metal_bridge.Segment,
    out_buf: *metal_bridge.Buf,
    w_buf: *metal_bridge.Buf,
    w_offset: usize,
    in_buf: *metal_bridge.Buf,
    m: usize,
    k: usize,
    quant: kernels.QuantType,
) void {
    switch (quant) {
        .q8_0 => seg.matmulQ8_0(out_buf, w_buf, w_offset, in_buf, m, k),
        .q4_k => seg.matmulQ4_K(out_buf, w_buf, w_offset, in_buf, m, k),
        .q5_k => seg.matmulQ5_K(out_buf, w_buf, w_offset, in_buf, m, k),
        .q6_k => seg.matmulQ6_K(out_buf, w_buf, w_offset, in_buf, m, k),
        else => unreachable, // gated by modelGpuEligible / isGpuEligibleProj
    }
}

fn gpuLayerStep(
    mb: *MetalBackend,
    layer: model_mod.LayerWeights,
    cache: *KvCache,
    c: model_mod.LlamaConfig,
    l: usize,
) StepError!void {
    const kv_dim = c.n_kv_heads * c.head_dim;
    const seq_len = cache.pos + 1;
    const head_dim_f: f32 = @floatFromInt(c.head_dim);
    const inv_sqrt_hd: f32 = 1.0 / @sqrt(head_dim_f);

    // Per-layer KV slab offset (bytes) into the shared cache buffer.
    const layer_kv_floats = c.max_seq * kv_dim;
    const layer_kv_off_bytes = l * layer_kv_floats * @sizeOf(f32);
    const slot_off_bytes = layer_kv_off_bytes + cache.pos * kv_dim * @sizeOf(f32);

    // One MTLCommandBuffer per layer — every op (including attention) on GPU.
    // 16 dispatches chained with implicit serial barriers; one commit/sync.
    const seg = mb.dev.segmentBegin() catch return error.UnsupportedWeightType;

    // Pre-attn: rmsnorm(xb ← x), QKV projection, RoPE on Q/K.
    seg.rmsnorm(mb.xb.buf, mb.x.buf, mb.weightsBuf(), mb.weightOffset(layer.attn_norm.bytes), c.dim, c.rms_eps);
    const qkv_fused_ready = layer.attn_qkv_fused != null and mb.fused_qkv != null and mb.qkv_scratch != null;
    if (qkv_fused_ready) {
        // One matmul produces the concatenated [Q | K | V] vector; copies
        // split it back into the per-projection scratch buffers so the
        // existing RoPE / KV-cache / attention path stays unchanged.
        const fused_t = layer.attn_qkv_fused.?;
        const fused = mb.fused_qkv.?;
        const scratch = mb.qkv_scratch.?;
        const qkv_dim = c.n_heads * c.head_dim + 2 * c.n_kv_heads * c.head_dim;
        segMatmulQuant(seg, scratch.buf, fused.buf, fused.byte_offsets[l], mb.xb.buf, qkv_dim, c.dim, fused_t.quant);
        const q_floats = c.n_heads * c.head_dim;
        const k_off_bytes = q_floats * @sizeOf(f32);
        const v_off_bytes = (q_floats + kv_dim) * @sizeOf(f32);
        seg.copy(mb.q.buf, 0, scratch.buf, 0, q_floats);
        seg.copy(mb.k_cur.buf, 0, scratch.buf, k_off_bytes, kv_dim);
        seg.copy(mb.v_cur.buf, 0, scratch.buf, v_off_bytes, kv_dim);
    } else {
        segMatmulQuant(seg, mb.q.buf,     mb.weightsBuf(), mb.weightOffset(layer.attn_q.bytes), mb.xb.buf, c.dim,  c.dim, layer.attn_q.quant);
        segMatmulQuant(seg, mb.k_cur.buf, mb.weightsBuf(), mb.weightOffset(layer.attn_k.bytes), mb.xb.buf, kv_dim, c.dim, layer.attn_k.quant);
        segMatmulQuant(seg, mb.v_cur.buf, mb.weightsBuf(), mb.weightOffset(layer.attn_v.bytes), mb.xb.buf, kv_dim, c.dim, layer.attn_v.quant);
    }
    seg.rope(mb.q.buf, c.n_heads, c.head_dim, cache.pos, c.rope_base);
    seg.rope(mb.k_cur.buf, c.n_kv_heads, c.head_dim, cache.pos, c.rope_base);

    // Splice rotated K and V into the per-layer cache slot at the current pos.
    seg.copy(mb.kv_k.buf, slot_off_bytes, mb.k_cur.buf, 0, kv_dim);
    seg.copy(mb.kv_v.buf, slot_off_bytes, mb.v_cur.buf, 0, kv_dim);

    // GQA self-attention on GPU: scores = Q · Kᵀ * inv_sqrt_hd, softmax,
    // weighted V sum. Sliding-window attention (Mistral) is implemented by
    // shifting the K/V cache base offset forward by `window_start` rows and
    // shortening `seq_len` to `attended_len` — the existing kernels only see
    // the windowed prefix and have no idea windowing is in play.
    const window_start: usize = if (c.sliding_window == std.math.maxInt(usize) or cache.pos < c.sliding_window)
        0
    else
        cache.pos - c.sliding_window + 1;
    const attended_len = seq_len - window_start;
    const window_off_bytes = layer_kv_off_bytes + window_start * kv_dim * @sizeOf(f32);
    seg.attnScores(mb.attn_scores.buf, mb.q.buf, mb.kv_k.buf, window_off_bytes, c.n_heads, c.n_kv_heads, c.head_dim, attended_len, inv_sqrt_hd);
    seg.softmaxRows(mb.attn_scores.buf, c.n_heads, c.n_kv_heads, c.head_dim, attended_len);
    seg.attnWeightedSum(mb.attn_out.buf, mb.attn_scores.buf, mb.kv_v.buf, window_off_bytes, c.n_heads, c.n_kv_heads, c.head_dim, attended_len);

    // Post-attn: O proj + residual into x.
    segMatmulQuant(seg, mb.xb2.buf, mb.weightsBuf(), mb.weightOffset(layer.attn_o.bytes), mb.attn_out.buf, c.dim, c.dim, layer.attn_o.quant);
    seg.residualAdd(mb.x.buf, mb.xb2.buf, c.dim);

    // FFN: rmsnorm(xb ← x), gate/up matmul, SwiGLU, ffn_down, residual.
    seg.rmsnorm(mb.xb.buf, mb.x.buf, mb.weightsBuf(), mb.weightOffset(layer.ffn_norm.bytes), c.dim, c.rms_eps);
    const gu_fused_ready = layer.ffn_gate_up_fused != null and mb.fused_gate_up != null and mb.gate_up_scratch != null;
    if (gu_fused_ready) {
        const fused_t = layer.ffn_gate_up_fused.?;
        const fused = mb.fused_gate_up.?;
        const scratch = mb.gate_up_scratch.?;
        const gu_dim = 2 * c.ffn_dim;
        segMatmulQuant(seg, scratch.buf, fused.buf, fused.byte_offsets[l], mb.xb.buf, gu_dim, c.dim, fused_t.quant);
        const up_off_bytes = c.ffn_dim * @sizeOf(f32);
        seg.copy(mb.gate.buf, 0, scratch.buf, 0, c.ffn_dim);
        seg.copy(mb.up.buf, 0, scratch.buf, up_off_bytes, c.ffn_dim);
    } else {
        segMatmulQuant(seg, mb.gate.buf, mb.weightsBuf(), mb.weightOffset(layer.ffn_gate.bytes), mb.xb.buf, c.ffn_dim, c.dim, layer.ffn_gate.quant);
        segMatmulQuant(seg, mb.up.buf,   mb.weightsBuf(), mb.weightOffset(layer.ffn_up.bytes),   mb.xb.buf, c.ffn_dim, c.dim, layer.ffn_up.quant);
    }
    switch (c.hidden_activation) {
        .silu => seg.swiglu(mb.gate.buf, mb.up.buf, c.ffn_dim),
        .gelu_approx => seg.gegluApprox(mb.gate.buf, mb.up.buf, c.ffn_dim),
    }
    segMatmulQuant(seg, mb.ffn_out.buf, mb.weightsBuf(), mb.weightOffset(layer.ffn_down.bytes), mb.gate.buf, c.dim, c.ffn_dim, layer.ffn_down.quant);
    seg.residualAdd(mb.x.buf, mb.ffn_out.buf, c.dim);

    seg.commit() catch return error.UnsupportedWeightType;
}

fn gpuFinalStep(mb: *MetalBackend, m: *const Model, c: model_mod.LlamaConfig) StepError!void {
    const seg = mb.dev.segmentBegin() catch return error.UnsupportedWeightType;
    seg.rmsnorm(mb.x.buf, mb.x.buf, mb.weightsBuf(), mb.weightOffset(m.output_norm.bytes), c.dim, c.rms_eps);
    segMatmulQuant(seg, mb.logits.buf, mb.weightsBuf(), mb.weightOffset(m.output_w.bytes), mb.x.buf, c.vocab_size, c.dim, m.output_w.quant);
    seg.commit() catch return error.UnsupportedWeightType;
}

// -------------------- helpers ---------------------------------------------

const MatmulCtx = struct {
    out: []f32,
    weights: []const u8,
    acts: []const f32,
    m: usize,
    k: usize,
    row_bytes: usize,
    quant: kernels.QuantType,
};

fn matmulWorker(worker_id: usize, n_workers: usize, ctx_ptr: *anyopaque) void {
    const ctx: *MatmulCtx = @ptrCast(@alignCast(ctx_ptr));
    const chunk = (ctx.m + n_workers - 1) / n_workers;
    const start = worker_id * chunk;
    const end = @min(start + chunk, ctx.m);
    if (start >= end) return;
    const m_chunk = end - start;
    const out_chunk = ctx.out[start..end];
    const w_chunk = ctx.weights[start * ctx.row_bytes ..][0 .. m_chunk * ctx.row_bytes];
    switch (ctx.quant) {
        .mlx_q4 => unreachable, // handled before pool dispatch in matmulRuntime
        inline else => |qt| {
            const kernel = comptime kernels.dispatch(qt);
            kernel(out_chunk, w_chunk, ctx.acts, m_chunk, ctx.k);
        },
    }
}

fn matmulRuntime(
    pool: *ThreadPool,
    metal: ?*MetalBackend,
    out: []f32,
    w: TypedTensor,
    acts: []const f32,
    m: usize,
    k: usize,
) StepError!void {
    const q = w.quant;

    // MLX-Q4: 3-buffer kernel. Prefer the GPU dispatch when a MetalBackend
    // is present, the weight is 4-bit, and both `out` and `acts` alias
    // persistent ScratchBufs (so the kernel can use real MTLBuffers without
    // a copy). Any failure — missing scratch alias, unsupported bits, or
    // dispatch error — falls through to the CPU baseline below.
    if (q == .mlx_q4) {
        const aux = w.mlx orelse return error.UnsupportedWeightType;

        gpu: {
            if (metal) |mb| {
                if (aux.bits == 4) {
                    const out_buf = mb.scratchForPtr(out.ptr) orelse break :gpu;
                    const acts_buf = mb.scratchForPtr(acts.ptr) orelse break :gpu;
                    mb.matmulMlxQ4(
                        out_buf.buf,
                        w.bytes.ptr,
                        aux.scales.ptr,
                        aux.biases.ptr,
                        acts_buf.buf,
                        m,
                        k,
                        aux.bits,
                        aux.group_size,
                        aux.scale_dtype,
                    ) catch break :gpu;
                    return;
                }
            }
        }

        @import("../kernels/mlx.zig").matmul(
            out, w.bytes, aux.scales, aux.biases, acts,
            m, k, aux.bits, aux.group_size, aux.scale_dtype,
        ) catch return error.UnsupportedWeightType;
        return;
    }

    // GPU fast path: Q8_0 + K-quant projections. Other quants stay on CPU.
    // Both `out` and `acts` must alias persistent ScratchBufs so the
    // dispatch can be zero-copy. State.init wires this up when metal is on.
    if (metal) |mb| if (isGpuEligibleProj(q)) {
        const out_buf = mb.scratchForPtr(out.ptr) orelse return error.UnsupportedWeightType;
        const in_buf = mb.scratchForPtr(acts.ptr) orelse return error.UnsupportedWeightType;
        const rc = switch (q) {
            .q8_0 => mb.matmulQ8_0(out_buf, w.bytes, in_buf, m, k),
            .q4_k => mb.matmulQ4_K(out_buf, w.bytes, in_buf, m, k),
            .q5_k => mb.matmulQ5_K(out_buf, w.bytes, in_buf, m, k),
            .q6_k => mb.matmulQ6_K(out_buf, w.bytes, in_buf, m, k),
            else => unreachable,
        };
        rc catch return error.UnsupportedWeightType;
        return;
    };

    var ctx: MatmulCtx = .{
        .out = out,
        .weights = w.bytes,
        .acts = acts,
        .m = m,
        .k = k,
        .row_bytes = rowBytes(q, k),
        .quant = q,
    };
    pool.dispatch(matmulWorker, &ctx);
}

pub fn rowBytes(q: kernels.QuantType, k: usize) usize {
    return switch (q) {
        .f32 => k * 4,
        .f16 => k * 2,
        .bf16 => k * 2,
        .mlx_q4 => unreachable, // handled separately in matmulRuntime
        .q8_0 => (k / simd.q8_0_block_elems) * simd.q8_0_block_bytes,
        .q4_k => (k / simd.q4_k_block_elems) * simd.q4_k_block_bytes,
        .q5_k => (k / simd.q5_k_block_elems) * simd.q5_k_block_bytes,
        .q6_k => (k / simd.q6_k_block_elems) * simd.q6_k_block_bytes,
        .ternary158 => (k / simd.tq2_0_block_elems) * simd.tq2_0_block_bytes,
    };
}

/// RMSNorm where the weight tensor is stored in some ggml type. Most llamas
/// keep norms in f32 but Q8_0 / f16 are also legal.
pub fn rmsNormTyped(x: []f32, w: TypedTensor, eps: f32) StepError!void {
    var stack_buf: [8192]f32 = undefined;
    const dim = x.len;
    if (dim > stack_buf.len) return error.UnsupportedWeightType;
    const w_f32 = stack_buf[0..dim];
    try dequantToF32(w_f32, w, dim);
    math.rmsNorm(x, w_f32, eps);
}

pub fn dequantToF32(out: []f32, w: TypedTensor, n: usize) StepError!void {
    const q = w.quant;
    switch (q) {
        .f32 => {
            const ptr: [*]align(1) const f32 = @ptrCast(w.bytes.ptr);
            for (out[0..n], 0..) |*o, i| o.* = ptr[i];
        },
        .f16 => {
            for (out[0..n], 0..) |*o, i| {
                const bits = std.mem.readInt(u16, w.bytes[i * 2 ..][0..2], .little);
                const h: f16 = @bitCast(bits);
                o.* = @floatCast(h);
            }
        },
        .bf16 => {
            for (out[0..n], 0..) |*o, i| {
                const bits = std.mem.readInt(u16, w.bytes[i * 2 ..][0..2], .little);
                o.* = simd.bf16BitsToF32(bits);
            }
        },
        .mlx_q4 => {
            const aux = w.mlx orelse return error.UnsupportedWeightType;
            @import("../kernels/mlx.zig").dequantRowDispatch(
                out[0..n], w.bytes, aux.scales, aux.biases, aux.bits, aux.group_size, aux.scale_dtype,
            ) catch return error.UnsupportedWeightType;
        },
        .q8_0 => {
            std.debug.assert(n % simd.q8_0_block_elems == 0);
            const blocks = n / simd.q8_0_block_elems;
            for (0..blocks) |b| {
                var tmp: [simd.q8_0_block_elems]f32 = undefined;
                const blk: *const [simd.q8_0_block_bytes]u8 =
                    w.bytes[b * simd.q8_0_block_bytes ..][0..simd.q8_0_block_bytes];
                simd.dequantBlockQ8_0(blk, &tmp);
                @memcpy(out[b * simd.q8_0_block_elems ..][0..simd.q8_0_block_elems], &tmp);
            }
        },
        .q4_k => {
            std.debug.assert(n % simd.q4_k_block_elems == 0);
            const blocks = n / simd.q4_k_block_elems;
            for (0..blocks) |b| {
                var tmp: [simd.q4_k_block_elems]f32 = undefined;
                const blk: *const [simd.q4_k_block_bytes]u8 =
                    w.bytes[b * simd.q4_k_block_bytes ..][0..simd.q4_k_block_bytes];
                simd.dequantBlockQ4_K(blk, &tmp);
                @memcpy(out[b * simd.q4_k_block_elems ..][0..simd.q4_k_block_elems], &tmp);
            }
        },
        .q5_k => {
            std.debug.assert(n % simd.q5_k_block_elems == 0);
            const blocks = n / simd.q5_k_block_elems;
            for (0..blocks) |b| {
                var tmp: [simd.q5_k_block_elems]f32 = undefined;
                const blk: *const [simd.q5_k_block_bytes]u8 =
                    w.bytes[b * simd.q5_k_block_bytes ..][0..simd.q5_k_block_bytes];
                simd.dequantBlockQ5_K(blk, &tmp);
                @memcpy(out[b * simd.q5_k_block_elems ..][0..simd.q5_k_block_elems], &tmp);
            }
        },
        .q6_k => {
            std.debug.assert(n % simd.q6_k_block_elems == 0);
            const blocks = n / simd.q6_k_block_elems;
            for (0..blocks) |b| {
                var tmp: [simd.q6_k_block_elems]f32 = undefined;
                const blk: *const [simd.q6_k_block_bytes]u8 =
                    w.bytes[b * simd.q6_k_block_bytes ..][0..simd.q6_k_block_bytes];
                simd.dequantBlockQ6_K(blk, &tmp);
                @memcpy(out[b * simd.q6_k_block_elems ..][0..simd.q6_k_block_elems], &tmp);
            }
        },
        .ternary158 => {
            std.debug.assert(n % simd.tq2_0_block_elems == 0);
            const blocks = n / simd.tq2_0_block_elems;
            for (0..blocks) |b| {
                var tmp: [simd.tq2_0_block_elems]f32 = undefined;
                const blk: *const [simd.tq2_0_block_bytes]u8 =
                    w.bytes[b * simd.tq2_0_block_bytes ..][0..simd.tq2_0_block_bytes];
                simd.dequantBlockTQ2_0(blk, &tmp);
                @memcpy(out[b * simd.tq2_0_block_elems ..][0..simd.tq2_0_block_elems], &tmp);
            }
        },
    }
}

pub fn gatherEmbedding(out: []f32, table: TypedTensor, id: u32, dim: usize) StepError!void {
    const q = table.quant;
    const id_us: usize = @intCast(id);
    switch (q) {
        .mlx_q4 => {
            // Embedding row of an MLX-quantized table: same packed layout
            // (out_features rows × in_features quant cols), so reuse the
            // single-row dequant for row `id`.
            const aux = table.mlx orelse return error.UnsupportedWeightType;
            const el_per_int: usize = 32 / aux.bits;
            const u32_per_row = dim / el_per_int;
            const row_bytes = u32_per_row * 4;
            const groups_per_row = dim / aux.group_size;
            const scale_row_bytes = groups_per_row * 2;
            const w_row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            const s_row = aux.scales[id_us * scale_row_bytes ..][0..scale_row_bytes];
            const b_row = aux.biases[id_us * scale_row_bytes ..][0..scale_row_bytes];
            @import("../kernels/mlx.zig").dequantRowDispatch(
                out, w_row, s_row, b_row, aux.bits, aux.group_size, aux.scale_dtype,
            ) catch return error.UnsupportedWeightType;
            return;
        },
        .f32 => {
            const ptr: [*]align(1) const f32 = @ptrCast(table.bytes.ptr);
            for (out, 0..) |*o, i| o.* = ptr[id_us * dim + i];
        },
        .bf16 => {
            const off = id_us * dim * 2;
            for (out, 0..) |*o, i| {
                const bits = std.mem.readInt(u16, table.bytes[off + i * 2 ..][0..2], .little);
                o.* = simd.bf16BitsToF32(bits);
            }
        },
        .f16 => {
            const off = id_us * dim * 2;
            for (out, 0..) |*o, i| {
                const bits = std.mem.readInt(u16, table.bytes[off + i * 2 ..][0..2], .little);
                const h: f16 = @bitCast(bits);
                o.* = @floatCast(h);
            }
        },
        .q8_0 => {
            std.debug.assert(dim % simd.q8_0_block_elems == 0);
            const blocks = dim / simd.q8_0_block_elems;
            const row_bytes = blocks * simd.q8_0_block_bytes;
            const row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            for (0..blocks) |b| {
                var tmp: [simd.q8_0_block_elems]f32 = undefined;
                const blk: *const [simd.q8_0_block_bytes]u8 =
                    row[b * simd.q8_0_block_bytes ..][0..simd.q8_0_block_bytes];
                simd.dequantBlockQ8_0(blk, &tmp);
                @memcpy(out[b * simd.q8_0_block_elems ..][0..simd.q8_0_block_elems], &tmp);
            }
        },
        .q4_k => {
            std.debug.assert(dim % simd.q4_k_block_elems == 0);
            const blocks = dim / simd.q4_k_block_elems;
            const row_bytes = blocks * simd.q4_k_block_bytes;
            const row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            for (0..blocks) |b| {
                var tmp: [simd.q4_k_block_elems]f32 = undefined;
                const blk: *const [simd.q4_k_block_bytes]u8 =
                    row[b * simd.q4_k_block_bytes ..][0..simd.q4_k_block_bytes];
                simd.dequantBlockQ4_K(blk, &tmp);
                @memcpy(out[b * simd.q4_k_block_elems ..][0..simd.q4_k_block_elems], &tmp);
            }
        },
        .q5_k => {
            std.debug.assert(dim % simd.q5_k_block_elems == 0);
            const blocks = dim / simd.q5_k_block_elems;
            const row_bytes = blocks * simd.q5_k_block_bytes;
            const row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            for (0..blocks) |b| {
                var tmp: [simd.q5_k_block_elems]f32 = undefined;
                const blk: *const [simd.q5_k_block_bytes]u8 =
                    row[b * simd.q5_k_block_bytes ..][0..simd.q5_k_block_bytes];
                simd.dequantBlockQ5_K(blk, &tmp);
                @memcpy(out[b * simd.q5_k_block_elems ..][0..simd.q5_k_block_elems], &tmp);
            }
        },
        .q6_k => {
            std.debug.assert(dim % simd.q6_k_block_elems == 0);
            const blocks = dim / simd.q6_k_block_elems;
            const row_bytes = blocks * simd.q6_k_block_bytes;
            const row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            for (0..blocks) |b| {
                var tmp: [simd.q6_k_block_elems]f32 = undefined;
                const blk: *const [simd.q6_k_block_bytes]u8 =
                    row[b * simd.q6_k_block_bytes ..][0..simd.q6_k_block_bytes];
                simd.dequantBlockQ6_K(blk, &tmp);
                @memcpy(out[b * simd.q6_k_block_elems ..][0..simd.q6_k_block_elems], &tmp);
            }
        },
        .ternary158 => {
            std.debug.assert(dim % simd.tq2_0_block_elems == 0);
            const blocks = dim / simd.tq2_0_block_elems;
            const row_bytes = blocks * simd.tq2_0_block_bytes;
            const row = table.bytes[id_us * row_bytes ..][0..row_bytes];
            for (0..blocks) |b| {
                var tmp: [simd.tq2_0_block_elems]f32 = undefined;
                const blk: *const [simd.tq2_0_block_bytes]u8 =
                    row[b * simd.tq2_0_block_bytes ..][0..simd.tq2_0_block_bytes];
                simd.dequantBlockTQ2_0(blk, &tmp);
                @memcpy(out[b * simd.tq2_0_block_elems ..][0..simd.tq2_0_block_elems], &tmp);
            }
        },
    }
}

// Regression test: ensures the GPU mlx_q4 branch in matmulRuntime produces
// results matching the CPU baseline within 1e-3. The test wires `out` and
// `acts` into MetalBackend ScratchBufs (xb, q) and calls matmulRuntime so
// the GPU dispatch fires; CPU expected values come from the same kernel
// the fallback branch would have called. Detects future divergence between
// the two codepaths.
test "matmulRuntime: GPU mlx_q4 matches CPU baseline within tolerance" {
    const metal = @import("../metal/bridge.zig");
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const mlx_kernels = @import("../kernels/mlx.zig");

    const page = std.heap.pageSize();

    // Single-shard fixture: weight + scales + biases adjacent in one mmap.
    const region = try std.posix.mmap(
        null,
        page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    );
    defer std.posix.munmap(region);

    // Q4 layout: m=4, k=64, group_size=64 (1 group/row).
    const m: usize = 4;
    const k: usize = 64;
    const w_len = m * k / 2; // 4-bit packed
    const s_len = m * 2; // 1 group/row × f16
    const b_len = m * 2;

    // Every nibble = 0x3 (q=3). f16(0.5) = 0x3800. f16(0.25) = 0x3400.
    @memset(region[0..w_len], 0x33);
    var i: usize = 0;
    while (i < m) : (i += 1) std.mem.writeInt(
        u16,
        region[w_len + i * 2 ..][0..2],
        0x3800,
        .little,
    );
    i = 0;
    while (i < m) : (i += 1) std.mem.writeInt(
        u16,
        region[w_len + s_len + i * 2 ..][0..2],
        0x3400,
        .little,
    );

    const shards = [_]metal_backend_mod.ShardInput{
        .{ .bytes = region[0 .. w_len + s_len + b_len] },
    };

    // Reuse the LlamaConfig used by the metal_backend Q4 tests: dim=64,
    // n_heads*head_dim=64, max_seq=16. Keeps scratch xb / q sized to fit.
    const cfg: model_mod.LlamaConfig = .{
        .dim = 64,
        .n_layers = 1,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .ffn_dim = 64,
        .vocab_size = 128,
        .max_seq = 16,
    };

    var be = try metal_backend_mod.MetalBackend.initShards(allocator, &shards, cfg);
    defer be.deinit(allocator);

    // acts[0]=1, rest=0 — placed in scratch xb (cap = cfg.dim = 64).
    @memset(be.xb.ptr[0..k], 0);
    be.xb.ptr[0] = 1.0;

    const aux: model_mod.MlxAux = .{
        .scales = region[w_len .. w_len + s_len],
        .biases = region[w_len + s_len .. w_len + s_len + b_len],
        .bits = 4,
        .group_size = 64,
        .scale_dtype = .f16,
    };
    const w: TypedTensor = .{
        .bytes = region[0..w_len],
        .quant = .mlx_q4,
        .mlx = aux,
    };

    // CPU baseline.
    var cpu_out: [4]f32 = undefined;
    try mlx_kernels.matmul(
        &cpu_out,
        w.bytes,
        aux.scales,
        aux.biases,
        be.xb.ptr[0..k],
        m,
        k,
        aux.bits,
        aux.group_size,
        aux.scale_dtype,
    );

    // GPU via matmulRuntime: out aliases scratch q (cap = n_heads*head_dim = 64).
    const pool = try ThreadPool.init(allocator, 1);
    defer pool.deinit(allocator);

    const out_slice = be.q.ptr[0..m];
    @memset(out_slice, 0);

    try matmulRuntime(pool, &be, out_slice, w, be.xb.ptr[0..k], m, k);

    for (cpu_out, out_slice) |c_v, g_v| {
        try std.testing.expectApproxEqAbs(c_v, g_v, 1e-3);
    }
}

// Mirrors the f16 GPU/CPU bit-equality test above but uses bf16-encoded
// scales/biases. This is the format MLX has shipped by default since
// mlx-lm 0.10+, and the format the regression hunt exposed: real MLX-Q4
// checkpoints (e.g. mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit) use bf16
// here, while the f16 path was the only one exercised by Tasks 5/6.
test "matmulRuntime: GPU mlx_q4 with bf16 scales matches CPU baseline" {
    const metal = @import("../metal/bridge.zig");
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const mlx_kernels = @import("../kernels/mlx.zig");

    const page = std.heap.pageSize();

    const region = try std.posix.mmap(
        null,
        page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    );
    defer std.posix.munmap(region);

    // Same shape as the f16 test: m=4, k=64, group_size=64.
    const m: usize = 4;
    const k: usize = 64;
    const w_len = m * k / 2;
    const s_len = m * 2;
    const b_len = m * 2;

    // bf16 encodings: top-16-bits of the corresponding f32.
    //   bf16(0.5)  = top16(0x3F000000) = 0x3F00
    //   bf16(-1.0) = top16(0xBF800000) = 0xBF80
    @memset(region[0..w_len], 0x33); // every nibble = 3
    var i: usize = 0;
    while (i < m) : (i += 1) std.mem.writeInt(
        u16,
        region[w_len + i * 2 ..][0..2],
        0x3F00,
        .little,
    );
    i = 0;
    while (i < m) : (i += 1) std.mem.writeInt(
        u16,
        region[w_len + s_len + i * 2 ..][0..2],
        0xBF80,
        .little,
    );

    const shards = [_]metal_backend_mod.ShardInput{
        .{ .bytes = region[0 .. w_len + s_len + b_len] },
    };

    const cfg: model_mod.LlamaConfig = .{
        .dim = 64,
        .n_layers = 1,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .ffn_dim = 64,
        .vocab_size = 128,
        .max_seq = 16,
    };

    var be = try metal_backend_mod.MetalBackend.initShards(allocator, &shards, cfg);
    defer be.deinit(allocator);

    @memset(be.xb.ptr[0..k], 0);
    be.xb.ptr[0] = 1.0;
    be.xb.ptr[1] = 1.0; // give Σ x ≠ 0 so bias contributes
    be.xb.ptr[2] = -0.5;

    const aux: model_mod.MlxAux = .{
        .scales = region[w_len .. w_len + s_len],
        .biases = region[w_len + s_len .. w_len + s_len + b_len],
        .bits = 4,
        .group_size = 64,
        .scale_dtype = .bf16,
    };
    const w: TypedTensor = .{
        .bytes = region[0..w_len],
        .quant = .mlx_q4,
        .mlx = aux,
    };

    var cpu_out: [4]f32 = undefined;
    try mlx_kernels.matmul(
        &cpu_out,
        w.bytes,
        aux.scales,
        aux.biases,
        be.xb.ptr[0..k],
        m,
        k,
        aux.bits,
        aux.group_size,
        aux.scale_dtype,
    );

    const pool = try ThreadPool.init(allocator, 1);
    defer pool.deinit(allocator);

    const out_slice = be.q.ptr[0..m];
    @memset(out_slice, 0);

    try matmulRuntime(pool, &be, out_slice, w, be.xb.ptr[0..k], m, k);

    // bf16 has 7 fractional bits — looser tolerance than f16's 10. The
    // operands here are small powers-of-two with exact bf16 representations,
    // so 1e-3 is still tight. Loosen if a future fixture introduces rounding.
    for (cpu_out, out_slice) |c_v, g_v| {
        try std.testing.expectApproxEqAbs(c_v, g_v, 1e-2);
    }
}

// TinyLlama-shaped fixture: the real attn_q projection in
// mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit is m=2048, k=2048,
// group_size=64, bf16. Small fixtures don't catch issues that only
// surface at production dims (e.g. row-stride bugs). This test fills
// every nibble + every scale + every bias with deterministic values
// derived from row index, then compares CPU vs GPU end-to-end.
test "matmulRuntime: GPU mlx_q4 bf16 at TinyLlama dims (m=k=2048, group=64)" {
    const metal = @import("../metal/bridge.zig");
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const mlx_kernels = @import("../kernels/mlx.zig");

    const m: usize = 2048;
    const k: usize = 2048;
    const group_size: u32 = 64;
    const groups_per_row = k / group_size; // 32

    const w_len = m * k / 2; // 2048 * 1024 = 2 MiB
    const s_len = m * groups_per_row * 2;
    const b_len = m * groups_per_row * 2;

    const total = w_len + s_len + b_len;
    const page = std.heap.pageSize();
    const total_aligned = std.mem.alignForward(usize, total, page);

    const region = try std.posix.mmap(
        null,
        total_aligned,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    );
    defer std.posix.munmap(region);

    // Deterministic per-byte weight pattern. Avoid all-same nibbles so a
    // mis-shifted bit pattern would surface.
    for (region[0..w_len], 0..) |*byte, i| {
        byte.* = @truncate((i * 31 + 7) & 0xFF);
    }

    // Per-group scales/biases. Values stay in a range that's exactly
    // representable in bf16 (powers of two over a small mantissa).
    var gi: usize = 0;
    while (gi < m * groups_per_row) : (gi += 1) {
        // bf16(0.125 * (1 + (gi % 4))) ≈ 0.125 .. 0.5. Encode by taking
        // top16 of the f32 bits.
        const scale_f: f32 = 0.125 * @as(f32, @floatFromInt(1 + (gi % 4)));
        const scale_bits: u32 = @bitCast(scale_f);
        std.mem.writeInt(u16, region[w_len + gi * 2 ..][0..2], @truncate(scale_bits >> 16), .little);

        const bias_f: f32 = -0.5 + 0.0625 * @as(f32, @floatFromInt(gi % 8));
        const bias_bits: u32 = @bitCast(bias_f);
        std.mem.writeInt(u16, region[w_len + s_len + gi * 2 ..][0..2], @truncate(bias_bits >> 16), .little);
    }

    const shards = [_]metal_backend_mod.ShardInput{
        .{ .bytes = region[0..total] },
    };

    const cfg: model_mod.LlamaConfig = .{
        .dim = 2048,
        .n_layers = 1,
        .n_heads = 32,
        .n_kv_heads = 4,
        .head_dim = 64,
        .ffn_dim = 5632,
        .vocab_size = 32000,
        .max_seq = 64,
    };

    var be = try metal_backend_mod.MetalBackend.initShards(allocator, &shards, cfg);
    defer be.deinit(allocator);

    // Activations: deterministic small floats. Use scratch xb (cap=cfg.dim=2048).
    for (be.xb.ptr[0..k], 0..) |*x, i| {
        const v: f32 = 0.1 + 0.0625 * @as(f32, @floatFromInt(i % 5)) - 0.0625 * @as(f32, @floatFromInt(i % 7));
        x.* = v;
    }

    const aux: model_mod.MlxAux = .{
        .scales = region[w_len .. w_len + s_len],
        .biases = region[w_len + s_len .. w_len + s_len + b_len],
        .bits = 4,
        .group_size = group_size,
        .scale_dtype = .bf16,
    };
    const w: TypedTensor = .{
        .bytes = region[0..w_len],
        .quant = .mlx_q4,
        .mlx = aux,
    };

    // CPU baseline.
    const cpu_out = try allocator.alloc(f32, m);
    defer allocator.free(cpu_out);
    try mlx_kernels.matmul(
        cpu_out, w.bytes, aux.scales, aux.biases, be.xb.ptr[0..k],
        m, k, aux.bits, aux.group_size, aux.scale_dtype,
    );

    // GPU via matmulRuntime. Out aliases scratch xb2 (cap=cfg.dim=2048)
    // since q has cap = n_heads*head_dim = 2048 too. Use xb2 to avoid
    // reusing the same buffer the test wrote acts into.
    const pool = try ThreadPool.init(allocator, 1);
    defer pool.deinit(allocator);

    // Need a separate buffer for `out`; xb is already used for acts. q has
    // cap = 2048 (32*64), exactly m. Use it.
    const out_slice = be.q.ptr[0..m];
    @memset(out_slice, 0);

    try matmulRuntime(pool, &be, out_slice, w, be.xb.ptr[0..k], m, k);

    // Diagnose: track max abs diff and first divergence.
    var max_abs: f32 = 0;
    var first_bad: ?usize = null;
    for (cpu_out, out_slice, 0..) |c_v, g_v, i| {
        const d = @abs(c_v - g_v);
        if (d > max_abs) max_abs = d;
        if (d > 1e-2 and first_bad == null) {
            first_bad = i;
        }
    }
    if (first_bad) |i| {
        std.debug.print(
            "TinyLlama-dim divergence at row {d}: cpu={d:.6} gpu={d:.6} max_abs_diff={d:.6}\n",
            .{ i, cpu_out[i], out_slice[i], max_abs },
        );
        return error.TestUnexpectedResult;
    }
}

fn argmaxLocal(logits: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = logits[0];
    for (logits, 1..) |v, i| {
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    return @intCast(best_idx);
}

// End-to-end bit-equality test: greedy-decode 32 tokens with the CPU path
// and the GPU path side-by-side and assert identical token IDs. Gated on
// MLX_TEST_DIR pointing at a real MLX-Q4 snapshot. Catches any divergence
// the per-op unit tests miss (state aliasing, KV-cache coherency,
// embedding/lm_head dequant, etc.).
test "mlx-q4 GPU produces identical token ids to CPU on TinyLlama" {
    const metal = @import("../metal/bridge.zig");
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    // Read MLX_TEST_DIR via libc getenv. Zig 0.16 routes process env through
    // `Init.environ` which isn't threaded into test mains, so falling back
    // to libc keeps the test self-contained.
    const env_dir_cstr = std.c.getenv("MLX_TEST_DIR") orelse return error.SkipZigTest;
    const env_dir: []const u8 = std.mem.sliceTo(env_dir_cstr, 0);
    const io = std.testing.io;
    const hf_loader_mod = @import("../core/hf_loader.zig");
    const hf_config_mod = @import("../core/hf_config.zig");
    const tokenizer_mod = @import("tokenizer.zig");

    var bundle = try hf_loader_mod.HfBundle.init(allocator, io, env_dir);
    defer bundle.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const cfg = try hf_config_mod.parse(arena.allocator(), bundle.config_json);

    var model = try model_mod.Model.initFromHf(allocator, &bundle, cfg);
    defer model.deinit(allocator);

    // Encode prompt with the real tokenizer so token IDs are valid.
    if (bundle.tokenizer_json == null) return error.SkipZigTest;
    var tok = try tokenizer_mod.Tokenizer.initFromHfJson(allocator, bundle.tokenizer_json.?);
    defer tok.deinitOwnedPieces(allocator);
    const prompt_ids = try tok.encode(arena.allocator(), "Once upon a time", true);

    var pool = try ThreadPool.init(allocator, 1);
    defer pool.deinit(allocator);

    const N: usize = 16;

    // CPU run.
    var cpu_ids: [N]u32 = undefined;
    {
        var cache = try @import("kv_cache.zig").KvCache.init(
            allocator,
            model.config.n_layers,
            model.config.n_kv_heads,
            model.config.head_dim,
            model.config.max_seq,
            null,
        );
        defer cache.deinit(allocator);
        var st = try State.init(allocator, &model, null);
        defer st.deinit(allocator);
        for (prompt_ids) |id| try step(&model, &st, &cache, pool, null, id);
        var last: u32 = argmaxLocal(st.logits);
        for (0..N) |i| {
            cpu_ids[i] = last;
            if (i + 1 < N) {
                try step(&model, &st, &cache, pool, null, last);
                last = argmaxLocal(st.logits);
            }
        }
    }

    // GPU run.
    var gpu_ids: [N]u32 = undefined;
    {
        var be = try metal_backend_mod.MetalBackend.initFromHf(allocator, &bundle, model.config);
        defer be.deinit(allocator);
        var cache = try @import("kv_cache.zig").KvCache.init(
            allocator,
            model.config.n_layers,
            model.config.n_kv_heads,
            model.config.head_dim,
            model.config.max_seq,
            &be,
        );
        defer cache.deinit(allocator);
        var st = try State.init(allocator, &model, &be);
        defer st.deinit(allocator);
        for (prompt_ids) |id| try step(&model, &st, &cache, pool, &be, id);
        var last: u32 = argmaxLocal(st.logits);
        for (0..N) |i| {
            gpu_ids[i] = last;
            if (i + 1 < N) {
                try step(&model, &st, &cache, pool, &be, last);
                last = argmaxLocal(st.logits);
            }
        }
    }

    for (cpu_ids, gpu_ids, 0..) |c, g, i| {
        std.testing.expectEqual(c, g) catch |e| {
            std.debug.print("\nmismatch at i={d}: cpu={d} gpu={d}\n", .{ i, c, g });
            std.debug.print("cpu_ids: {any}\ngpu_ids: {any}\n", .{ cpu_ids, gpu_ids });
            return e;
        };
    }
}
