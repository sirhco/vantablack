//! Batched prompt prefill.
//!
//! `forward.step` processes one token per call; for an N-token prompt that's
//! N sequential per-token forwards, each of which re-reads every weight row
//! once. Prefill fuses the prompt into a single B-wide CPU pass: the
//! activation tensor becomes [B, dim], every projection matmul reads each
//! weight row once and fans it out across the B activation vectors, and
//! per-token operations (RoPE, KV-cache write, causal attention) run inside
//! a small inner B-loop with the heavy work amortized.
//!
//! Scope:
//!   * CPU only. The GPU forward path keeps shipping per-token; once a
//!     batched MSL kernel exists for Q8_0 / Q4_K / Q5_K / Q6_K the same
//!     pipeline gets a GPU specialization.
//!   * Bypasses `MetalBackend` even when `-Dmetal=true`. After prefill the
//!     caller resumes per-token `forward.step` (which can be GPU-backed) for
//!     decode.
//!   * Final RMSNorm + LM head run only on the last batch position so the
//!     vocab×dim matmul cost matches the per-token loop.
//!
//! Correctness invariant: byte-equal to `for (tokens) |id| forward.step(...)`
//! followed by `cache.advance()`. Verified against the standard
//! "Once upon a time" prompt on TinyLlama Q8_0.

const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../kernels/math.zig");
const simd = @import("../kernels/simd.zig");
const kernels = @import("../kernels/comptime_gen.zig");
const model_mod = @import("model.zig");
const kv_cache_mod = @import("kv_cache.zig");
const pool_mod = @import("pool.zig");
const forward = @import("forward.zig");

const Model = model_mod.Model;
const TypedTensor = model_mod.TypedTensor;
const KvCache = kv_cache_mod.KvCache;
const ThreadPool = pool_mod.ThreadPool;

pub const PrefillError = error{
    KvCacheFull,
    TokenOutOfRange,
    UnsupportedWeightType,
    DimTooLarge,
} || Allocator.Error;

/// Heap-allocated B-wide scratch buffers. Sized once per prefill call. All
/// per-batch slices of width `dim`, `kv_dim`, etc. are laid out contiguously
/// — `slice[b * width ..][0..width]` selects batch element `b`.
pub const Scratch = struct {
    allocator: Allocator,
    B: usize,
    x: []f32,
    xb: []f32,
    xb2: []f32,
    q: []f32,
    k_cur: []f32,
    v_cur: []f32,
    attn_out: []f32,
    gate: []f32,
    up: []f32,
    ffn_out: []f32,
    /// Per-batch attention scores: [B, n_heads * max_seq].
    attn_scores: []f32,

    pub fn init(allocator: Allocator, m: *const Model, B: usize) !Scratch {
        const c = m.config;
        const kv_dim = c.n_kv_heads * c.head_dim;
        const dim = c.dim;
        const ffn = c.ffn_dim;
        const q_dim = c.n_heads * c.head_dim;

        return .{
            .allocator = allocator,
            .B = B,
            .x = try allocator.alloc(f32, B * dim),
            .xb = try allocator.alloc(f32, B * dim),
            .xb2 = try allocator.alloc(f32, B * dim),
            .q = try allocator.alloc(f32, B * q_dim),
            .k_cur = try allocator.alloc(f32, B * kv_dim),
            .v_cur = try allocator.alloc(f32, B * kv_dim),
            .attn_out = try allocator.alloc(f32, B * dim),
            .gate = try allocator.alloc(f32, B * ffn),
            .up = try allocator.alloc(f32, B * ffn),
            .ffn_out = try allocator.alloc(f32, B * dim),
            .attn_scores = try allocator.alloc(f32, B * c.n_heads * c.max_seq),
        };
    }

    pub fn deinit(self: *Scratch) void {
        const a = self.allocator;
        a.free(self.x);
        a.free(self.xb);
        a.free(self.xb2);
        a.free(self.q);
        a.free(self.k_cur);
        a.free(self.v_cur);
        a.free(self.attn_out);
        a.free(self.gate);
        a.free(self.up);
        a.free(self.ffn_out);
        a.free(self.attn_scores);
        self.* = undefined;
    }
};

/// Run a B-wide forward over `tokens`, advancing `cache.pos` by B and
/// writing the LAST token's logits into `state.logits`. Caller can then
/// sample from `state.logits` and continue per-token with `forward.step`.
pub fn prefillCpu(
    allocator: Allocator,
    pool: *ThreadPool,
    m: *const Model,
    state: *forward.State,
    cache: *KvCache,
    tokens: []const u32,
) PrefillError!void {
    const c = m.config;
    const B = tokens.len;
    if (B == 0) return;
    if (cache.pos + B > c.max_seq) return error.KvCacheFull;
    for (tokens) |id| {
        if (id >= c.vocab_size) return error.TokenOutOfRange;
    }

    var s = try Scratch.init(allocator, m, B);
    defer s.deinit();

    // Embed all B tokens.
    for (tokens, 0..) |id, b| {
        const dst = s.x[b * c.dim ..][0..c.dim];
        try forward.gatherEmbedding(dst, m.token_embd, id, c.dim);
    }

    // Per-layer batched forward.
    for (m.layers, 0..) |layer, l| {
        try prefillLayer(pool, layer, &s, cache, c, l);
    }

    cache.pos += B;

    // Final RMSNorm + LM head only on the last batch position.
    const last_x = s.x[(B - 1) * c.dim ..][0..c.dim];
    try forward.rmsNormTyped(last_x, m.output_norm, c.rms_eps);
    try matmulCpu(pool, state.logits, m.output_w, last_x, c.vocab_size, c.dim);
}

fn prefillLayer(
    pool: *ThreadPool,
    layer: model_mod.LayerWeights,
    s: *Scratch,
    cache: *KvCache,
    c: model_mod.LlamaConfig,
    l: usize,
) PrefillError!void {
    const B = s.B;
    const kv_dim = c.n_kv_heads * c.head_dim;
    const q_dim = c.n_heads * c.head_dim;

    // Attention RMSNorm per batch position.
    for (0..B) |b| {
        const x_b = s.x[b * c.dim ..][0..c.dim];
        const xb_b = s.xb[b * c.dim ..][0..c.dim];
        @memcpy(xb_b, x_b);
        try forward.rmsNormTyped(xb_b, layer.attn_norm, c.rms_eps);
    }

    // Batched Q, K, V projections.
    try matmulBatched(pool, s.q, layer.attn_q, s.xb, c.dim, c.dim, B);
    try matmulBatched(pool, s.k_cur, layer.attn_k, s.xb, kv_dim, c.dim, B);
    try matmulBatched(pool, s.v_cur, layer.attn_v, s.xb, kv_dim, c.dim, B);

    // RoPE on Q and K per (batch, head); KV cache write at slot pos+b.
    const layer_kv_off = l * c.max_seq * kv_dim;
    for (0..B) |b| {
        const pos_b = cache.pos + b;
        const q_b = s.q[b * q_dim ..][0..q_dim];
        const k_b = s.k_cur[b * kv_dim ..][0..kv_dim];
        const v_b = s.v_cur[b * kv_dim ..][0..kv_dim];
        if (c.rope_half) {
            for (0..c.n_heads) |h| math.ropeHalf(q_b[h * c.head_dim ..][0..c.head_dim], pos_b, c.rope_base);
            for (0..c.n_kv_heads) |h| math.ropeHalf(k_b[h * c.head_dim ..][0..c.head_dim], pos_b, c.rope_base);
        } else {
            for (0..c.n_heads) |h| math.rope(q_b[h * c.head_dim ..][0..c.head_dim], pos_b, c.rope_base);
            for (0..c.n_kv_heads) |h| math.rope(k_b[h * c.head_dim ..][0..c.head_dim], pos_b, c.rope_base);
        }
        const slot_off = layer_kv_off + pos_b * kv_dim;
        @memcpy(cache.k[slot_off..][0..kv_dim], k_b);
        @memcpy(cache.v[slot_off..][0..kv_dim], v_b);
    }

    // Causal attention per batch position. Token at batch position b sees
    // KV positions [0, cache.pos + b].
    for (0..B) |b| {
        attentionCausal(s, cache, c, l, b);
    }

    // Output projection + residual.
    try matmulBatched(pool, s.xb2, layer.attn_o, s.attn_out, c.dim, c.dim, B);
    for (0..B) |b| {
        const xb2_b = s.xb2[b * c.dim ..][0..c.dim];
        const x_b = s.x[b * c.dim ..][0..c.dim];
        math.addInto(x_b, xb2_b);
    }

    // FFN RMSNorm per batch position.
    for (0..B) |b| {
        const x_b = s.x[b * c.dim ..][0..c.dim];
        const xb_b = s.xb[b * c.dim ..][0..c.dim];
        @memcpy(xb_b, x_b);
        try forward.rmsNormTyped(xb_b, layer.ffn_norm, c.rms_eps);
    }

    // Batched gate / up matmul, per-batch activation, batched ffn_down.
    try matmulBatched(pool, s.gate, layer.ffn_gate, s.xb, c.ffn_dim, c.dim, B);
    try matmulBatched(pool, s.up, layer.ffn_up, s.xb, c.ffn_dim, c.dim, B);
    for (0..B) |b| {
        const gate_b = s.gate[b * c.ffn_dim ..][0..c.ffn_dim];
        const up_b = s.up[b * c.ffn_dim ..][0..c.ffn_dim];
        switch (c.hidden_activation) {
            .silu => math.swiglu(gate_b, up_b),
            .gelu_approx => math.gegluApprox(gate_b, up_b),
        }
    }
    try matmulBatched(pool, s.ffn_out, layer.ffn_down, s.gate, c.dim, c.ffn_dim, B);
    for (0..B) |b| {
        const ffn_b = s.ffn_out[b * c.dim ..][0..c.dim];
        const x_b = s.x[b * c.dim ..][0..c.dim];
        math.addInto(x_b, ffn_b);
    }
}

/// Causal attention for batch position `b`. Token attends to KV positions
/// [0, cache.pos + b]. Reads from `s.q` / `s.attn_scores`, writes to
/// `s.attn_out[b]`.
fn attentionCausal(
    s: *Scratch,
    cache: *KvCache,
    c: model_mod.LlamaConfig,
    l: usize,
    b: usize,
) void {
    const head_dim_f: f32 = @floatFromInt(c.head_dim);
    const inv_sqrt_hd: f32 = 1.0 / @sqrt(head_dim_f);
    const cur_pos = cache.pos + b;
    const seq_len = cur_pos + 1;
    const kv_row = c.n_kv_heads * c.head_dim;
    const k_buf = cache.keysFor(l, seq_len);
    const v_buf = cache.valuesFor(l, seq_len);

    // Sliding-window: same logic as forward.cpuAttention. Mistral uses
    // sliding_window = 4096; non-windowed arches set maxInt(usize).
    const window_start: usize = if (c.sliding_window == std.math.maxInt(usize) or cur_pos < c.sliding_window)
        0
    else
        cur_pos - c.sliding_window + 1;
    const attended_len = seq_len - window_start;

    const q_b = s.q[b * (c.n_heads * c.head_dim) ..][0 .. c.n_heads * c.head_dim];
    const out_b = s.attn_out[b * c.dim ..][0..c.dim];
    const scores_b = s.attn_scores[b * (c.n_heads * c.max_seq) ..][0 .. c.n_heads * c.max_seq];

    @memset(out_b, 0);
    for (0..c.n_heads) |h| {
        const kv_h = (h * c.n_kv_heads) / c.n_heads;
        const q_h = q_b[h * c.head_dim ..][0..c.head_dim];
        const scores = scores_b[h * c.max_seq ..][0..attended_len];

        for (0..attended_len) |i| {
            const t = window_start + i;
            const k_th = k_buf[t * kv_row + kv_h * c.head_dim ..][0..c.head_dim];
            var dot: f32 = 0;
            for (q_h, k_th) |qv, kv| dot += qv * kv;
            scores[i] = dot * inv_sqrt_hd;
        }
        math.softmax(scores);

        const out_h = out_b[h * c.head_dim ..][0..c.head_dim];
        for (0..attended_len) |i| {
            const t = window_start + i;
            const v_th = v_buf[t * kv_row + kv_h * c.head_dim ..][0..c.head_dim];
            const w = scores[i];
            for (out_h, v_th) |*o, vv| o.* += w * vv;
        }
    }
}

// -------------------- batched matmul (CPU) --------------------------------
//
// out [B, m] = acts [B, k] @ Wᵀ [k, m]
//
// Inverted loop: for each output row of W, dequantize once into a
// thread-local f32 buffer, then run B inner-product loops against the B
// activation rows. Reads each weight row once across the whole batch — the
// lever the prefill pipeline pulls.
//
// Parallelism: rows are striped across the persistent thread pool. Each
// worker keeps its own dequant scratch on the stack (capped at MAX_K).

const MAX_K: usize = 8192; // Covers TinyLlama / Llama-3-8B / Mistral-7B widths.

const BatchedCtx = struct {
    out: []f32,
    w: TypedTensor,
    acts: []const f32,
    m: usize,
    k: usize,
    B: usize,
    err: ?PrefillError,
};

fn matmulBatched(
    pool: *ThreadPool,
    out: []f32,
    w: TypedTensor,
    acts: []const f32,
    m: usize,
    k: usize,
    B: usize,
) PrefillError!void {
    if (k > MAX_K) return error.DimTooLarge;
    var ctx: BatchedCtx = .{
        .out = out,
        .w = w,
        .acts = acts,
        .m = m,
        .k = k,
        .B = B,
        .err = null,
    };
    pool.dispatch(matmulBatchedWorker, &ctx);
    if (ctx.err) |e| return e;
}

fn matmulBatchedWorker(worker_id: usize, n_workers: usize, ctx_ptr: *anyopaque) void {
    const ctx: *BatchedCtx = @ptrCast(@alignCast(ctx_ptr));
    const chunk = (ctx.m + n_workers - 1) / n_workers;
    const start = worker_id * chunk;
    const end = @min(start + chunk, ctx.m);
    if (start >= end) return;

    var w_dq: [MAX_K]f32 = undefined;
    const w_dq_slice = w_dq[0..ctx.k];
    const row_bytes = forward.rowBytes(ctx.w.quant, ctx.k);

    var row: usize = start;
    while (row < end) : (row += 1) {
        const row_data = ctx.w.bytes[row * row_bytes ..][0..row_bytes];
        const synth: TypedTensor = .{
            .bytes = row_data,
            .quant = ctx.w.quant,
            .ggml_type = ctx.w.ggml_type,
            .mlx = ctx.w.mlx,
        };
        forward.dequantToF32(w_dq_slice, synth, ctx.k) catch |e| {
            ctx.err = e;
            return;
        };
        var b: usize = 0;
        while (b < ctx.B) : (b += 1) {
            const a = ctx.acts[b * ctx.k ..][0..ctx.k];
            ctx.out[b * ctx.m + row] = simd.dot_f32(w_dq_slice, a);
        }
    }
}

// Single-batch CPU matmul for the LM head pass. Mirrors `matmulBatched`
// but without the B inner loop — kept here to avoid threading a metal
// dependency through prefill (the existing forward.matmulRuntime takes
// metal as an argument).
fn matmulCpu(
    pool: *ThreadPool,
    out: []f32,
    w: TypedTensor,
    acts: []const f32,
    m: usize,
    k: usize,
) PrefillError!void {
    return matmulBatched(pool, out, w, acts, m, k, 1);
}

// -------------------- tests -----------------------------------------------

test "prefillCpu API surface compiles" {
    // Real end-to-end prefill needs a model — covered by the manual
    // byte-equality smoke test against TinyLlama. This compile-only check
    // catches signature drift.
    const f: *const @TypeOf(prefillCpu) = &prefillCpu;
    _ = f;
}
