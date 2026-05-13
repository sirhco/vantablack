//! Gemma 4 `.litertlm` Model binding (Phase 19c partial).
//!
//! Holds per-layer weight pointers + architecture metadata extracted
//! from a parsed `.litertlm` bundle. Does NOT yet run inference —
//! that's the future `runtime/forward_gemma4.zig`. This module's
//! contract: given a bundle, return a populated Gemma4Model whose
//! pointers can drive a forward pass.
//!
//! Architecture deduced empirically from
//! `litert-community/gemma-4-E2B-it-litert-lm` (see
//! `tests/golden/gemma4-e2b-architecture-notes.md`). Generalises to
//! Gemma 4 E4B / Gemma 3n by virtue of the same TFLite layout + naming
//! convention.

const std = @import("std");

const litertlm = @import("../core/litertlm.zig");
const tflite = @import("../core/tflite.zig");
const scan = @import("../core/gemma_layer_scan.zig");

pub const Error = error{
    NoDecoderSection,
    LayerCountZero,
    OutOfMemory,
} || litertlm.Error || tflite.Error;

/// Architecture parameters for a Gemma 4 model. Most fields are
/// derived from tensor shapes (the model rarely ships these as
/// explicit metadata).
pub const Gemma4Config = struct {
    /// Hidden state dimension (token embedding width).
    hidden: u32,
    /// Number of decoder layers.
    n_layers: u32,
    /// Number of query attention heads.
    n_q_heads: u32,
    /// Number of key/value attention heads (GQA when < n_q_heads).
    n_kv_heads: u32,
    /// Head dimension (n_q_heads * head_dim == Q projection rows).
    head_dim: u32,
    /// Per-layer FFN inner dimension. May vary across layers
    /// (MatFormer-style). Indexed by layer.
    ffn_dim_per_layer: []u32,
    /// Vocab size (rows of the embedding table).
    vocab_size: u32,
    /// Per-layer-embedding width. 256 in Gemma 4 E2B.
    ple_dim: u32,

    pub fn deinit(self: *Gemma4Config, allocator: std.mem.Allocator) void {
        allocator.free(self.ffn_dim_per_layer);
        self.* = undefined;
    }
};

/// One transformer block's weight pointers. Borrowed views into the
/// underlying mmap; not owned. Some slots are null in KV-shared layers
/// (Gemma 3n / 4 local↔global attention).
pub const Gemma4Layer = struct {
    /// Q projection weight. INT4 per-row, shape [n_q_heads*head_dim, hidden].
    q: *const tflite.Tensor,
    /// K projection weight. INT4 per-row, shape [n_kv_heads*head_dim, hidden].
    /// Null when this layer reuses K from a neighbouring full-attention layer.
    k: ?*const tflite.Tensor,
    v: ?*const tflite.Tensor,
    /// Attention output projection. shape [hidden, n_q_heads*head_dim].
    attn_o: *const tflite.Tensor,
    /// MLP gate / up / down (SwiGLU): shapes [ffn, hidden], [ffn, hidden], [hidden, ffn].
    mlp_gate: *const tflite.Tensor,
    mlp_up: *const tflite.Tensor,
    mlp_down: *const tflite.Tensor,
    /// Per-Layer Embedding gate + projection (Gemma-specific PLE mechanism).
    /// Shapes [ple_dim, hidden] and [hidden, ple_dim].
    ple_gate: *const tflite.Tensor,
    ple_proj: *const tflite.Tensor,
    /// Pre-computed RoPE inv-freq table (FLOAT32, shape [1, 1, head_dim]).
    /// Null on layers that share a global RoPE table.
    rope_inv_freq: ?*const tflite.Tensor,
    /// Learned RMSNorm scales. Five per layer in Gemma 4:
    /// pre_attention_norm, post_attention_norm, pre_ffw_norm,
    /// post_ffw_norm, post_per_layer_input_norm. Each shape [hidden]
    /// FLOAT32. Null when discovery couldn't pin a tensor.
    norm_pre_attn: ?*const tflite.Tensor = null,
    norm_post_attn: ?*const tflite.Tensor = null,
    norm_pre_ffw: ?*const tflite.Tensor = null,
    norm_post_ffw: ?*const tflite.Tensor = null,
    norm_post_per_layer: ?*const tflite.Tensor = null,
    /// Q/K-norm scales applied after projection, before RoPE.
    /// Shape [n_kv_heads * head_dim] (256 in Gemma 4 E2B). K-norm
    /// applies elementwise on the K vector; Q-norm broadcasts over
    /// the n_q_per_kv groups.
    norm_query: ?*const tflite.Tensor = null,
    norm_key: ?*const tflite.Tensor = null,
    /// Per-layer scalar that multiplies the residual stream after
    /// all sub-block residuals (attn + MLP + PLE) before passing to
    /// the next layer. FP32 [1,1,1] = 4 bytes total. Gemma 4's
    /// `_maybe_apply_skip_scale` mechanism.
    skip_scale: ?*const tflite.Tensor = null,
};

pub const Gemma4Model = struct {
    /// Architecture parameters.
    config: Gemma4Config,
    /// Per-layer weights. `layers.len == config.n_layers`.
    layers: []Gemma4Layer,
    /// Text embedding lookup table (INT2 per-row, shape [vocab, hidden]).
    /// Null if no embedder section was found. Used for token → hidden.
    embedder: ?*const tflite.Tensor = null,
    /// Decode-time embedder ("LM head"). Same shape as `embedder` but
    /// lives in the decoder section and may carry distinct per-row
    /// quantization scales tuned for hidden → vocab projection.
    /// Falls back to `embedder` (weight tying) when absent.
    lm_head: ?*const tflite.Tensor = null,
    /// Per-layer embedder lookup tables (Gemma's PLE mechanism).
    /// 35 INT4 tensors shape [vocab, ple_dim], indexed by layer.
    /// Slot is null if a layer's per-layer embed couldn't be located.
    per_layer_embedder: []?*const tflite.Tensor = &.{},
    /// TFLite Model holding the per_layer_embedder shards (section 3).
    /// Kept alive so the borrowed pointers stay valid.
    per_layer_tfl: ?tflite.Model = null,
    /// Underlying TFLite Model for the decoder section. Kept alive so
    /// the borrowed tensor pointers stay valid.
    decoder_tfl: tflite.Model,
    /// Underlying TFLite Model for the embedder section (separate from
    /// decoder_tfl — embedder lives in its own section). Kept alive
    /// to anchor the `embedder` pointer.
    embedder_tfl: ?tflite.Model = null,
    /// Index in `bundle.sections` of the decoder TFLite section.
    decoder_section_idx: usize,
    /// Allocator that owns this struct's growing arrays.
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Gemma4Model) void {
        self.config.deinit(self.allocator);
        self.allocator.free(self.layers);
        if (self.per_layer_embedder.len > 0) self.allocator.free(self.per_layer_embedder);
        self.decoder_tfl.deinit();
        if (self.embedder_tfl) |*tfl| tfl.deinit();
        if (self.per_layer_tfl) |*tfl| tfl.deinit();
        self.* = undefined;
    }

    /// Per-layer KV cache. Each layer's slot holds [max_seq * kv_dim_l]
    /// FP32 floats for both K and V. KV-shared layers re-use the
    /// previous full-attention layer's slot — they don't allocate own
    /// storage.
    pub const KvCache = struct {
        k_per_layer: [][]f32, // one slice per layer; alias for shared layers
        v_per_layer: [][]f32,
        owned: []bool, // true if slot is heap-owned (vs aliased)
        parent_of_shared: []usize, // for diagnostic
        max_seq: usize,
        pos: usize,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *KvCache) void {
            for (self.k_per_layer, self.owned) |slot, own| {
                if (own) self.allocator.free(slot);
            }
            for (self.v_per_layer, self.owned) |slot, own| {
                if (own) self.allocator.free(slot);
            }
            self.allocator.free(self.k_per_layer);
            self.allocator.free(self.v_per_layer);
            self.allocator.free(self.owned);
            self.allocator.free(self.parent_of_shared);
            self.* = undefined;
        }
    };

    pub fn initKvCache(self: *const Gemma4Model, allocator: std.mem.Allocator, max_seq: usize) !KvCache {
        const n = self.layers.len;
        const k_slots = try allocator.alloc([]f32, n);
        errdefer allocator.free(k_slots);
        const v_slots = try allocator.alloc([]f32, n);
        errdefer allocator.free(v_slots);
        const owned = try allocator.alloc(bool, n);
        errdefer allocator.free(owned);
        const parents = try allocator.alloc(usize, n);
        errdefer allocator.free(parents);

        const head_dim: usize = self.config.head_dim;
        _ = head_dim;
        // Gemma 4 E2B KV-share routing (verified via TFLite op-graph
        // trace of section 10): shared layers 15..34 alias EITHER layer
        // 13 (sliding-window-local parent) or layer 14 (global-attention
        // parent). The cycle is 4 local + 1 global, where the global
        // slot is every 5 layers starting at 14, i.e. layers {14, 19,
        // 24, 29, 34}. Shared layer L picks layer 14 when
        // (L - 14) % 5 == 0 else layer 13.
        const LOCAL_PARENT: usize = 13;
        const GLOBAL_PARENT: usize = 14;
        var last_owner: usize = 0;
        for (self.layers, 0..) |layer, i| {
            parents[i] = i;
            if (layer.k) |k_t| {
                const kv_dim_l: usize = @as(usize, @intCast(k_t.shape[0]));
                k_slots[i] = try allocator.alloc(f32, max_seq * kv_dim_l);
                v_slots[i] = try allocator.alloc(f32, max_seq * kv_dim_l);
                owned[i] = true;
                last_owner = i;
            } else {
                const cycle_parent: usize = if (i >= GLOBAL_PARENT and ((i - GLOBAL_PARENT) % 5) == 0)
                    GLOBAL_PARENT
                else
                    LOCAL_PARENT;
                const parent: usize = if (cycle_parent < self.layers.len and owned[cycle_parent])
                    cycle_parent
                else
                    last_owner;
                k_slots[i] = k_slots[parent];
                v_slots[i] = v_slots[parent];
                owned[i] = false;
                parents[i] = parent;
            }
        }
        return .{
            .k_per_layer = k_slots,
            .v_per_layer = v_slots,
            .owned = owned,
            .parent_of_shared = parents,
            .max_seq = max_seq,
            .pos = 0,
            .allocator = allocator,
        };
    }

    /// Generic per-row GEMV against a borrowed weight tensor. Dispatches
    /// on tensor dtype: INT4 (2/byte), INT2 (4/byte) — the two packings
    /// Gemma 4 E2B uses for matmul weights. INT8 has its own helper
    /// (`projectInt8Generic`) because the storage isn't bit-packed.
    fn projectQuant(t: *const tflite.Tensor, x: []const f32, y: []f32) !void {
        if (t.shape.len < 2) return error.ShapeMismatch;
        const m: usize = @intCast(t.shape[0]);
        const k: usize = @intCast(t.shape[1]);
        if (y.len != m or x.len != k) return error.ShapeMismatch;
        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len < m or zps.len < m) return error.ShapeMismatch;
        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        switch (t.dtype) {
            .int4 => try tflite_int4.gemvInt4PerRow(t.data, scales, zps, x, y, m, k),
            .int2 => try tflite_int4.gemvInt2PerRow(t.data, scales, zps, x, y, m, k),
            else => return error.UnsupportedQuant,
        }
    }

    /// Q projection. `q_out.len == n_q_heads * head_dim`.
    pub fn projectQ(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, q_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        try projectQuant(self.layers[layer_idx].q, hidden_in, q_out);
    }

    /// K projection. Returns error.NoOwnK on layers that share K from
    /// a neighbouring full-attention layer (15..34 in Gemma 4 E2B).
    /// `k_out.len == n_kv_heads * head_dim`.
    pub fn projectK(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, k_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        const t = self.layers[layer_idx].k orelse return error.NoOwnK;
        try projectQuant(t, hidden_in, k_out);
    }

    pub fn projectV(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, v_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        const t = self.layers[layer_idx].v orelse return error.NoOwnV;
        try projectQuant(t, hidden_in, v_out);
    }

    /// Output projection. `attn_in.len == n_q_heads * head_dim`,
    /// `hidden_out.len == config.hidden`.
    pub fn projectAttnO(self: *const Gemma4Model, attn_in: []const f32, layer_idx: usize, hidden_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        try projectQuant(self.layers[layer_idx].attn_o, attn_in, hidden_out);
    }

    /// MLP gate projection. `hidden_in.len == hidden`,
    /// `gate_out.len == ffn_dim_per_layer[layer_idx]`.
    pub fn projectMlpGate(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, gate_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        try projectQuant(self.layers[layer_idx].mlp_gate, hidden_in, gate_out);
    }

    pub fn projectMlpUp(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, up_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        try projectQuant(self.layers[layer_idx].mlp_up, hidden_in, up_out);
    }

    /// MLP down projection. `gated_in.len == ffn`, `hidden_out.len == hidden`.
    pub fn projectMlpDown(self: *const Gemma4Model, gated_in: []const f32, layer_idx: usize, hidden_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        try projectQuant(self.layers[layer_idx].mlp_down, gated_in, hidden_out);
    }

    /// Per-Layer Embedding gate projection. INT8 (not INT4) per-row.
    /// `hidden_in.len == hidden`, `gate_out.len == ple_dim`.
    pub fn projectPleGate(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, gate_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        const t = self.layers[layer_idx].ple_gate;
        try projectInt8Generic(t, hidden_in, gate_out);
    }

    /// Per-Layer Embedding projection back into hidden width.
    /// `ple_in.len == ple_dim`, `hidden_out.len == hidden`.
    pub fn projectPleProj(self: *const Gemma4Model, ple_in: []const f32, layer_idx: usize, hidden_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        const t = self.layers[layer_idx].ple_proj;
        try projectInt8Generic(t, ple_in, hidden_out);
    }

    /// Single-token layer-0 forward — minimum-viable end-to-end forward
    /// step on the real Gemma 4 weights. No KV cache (only attends to
    /// the current token), no per-layer norms (TBD — likely fused into
    /// matmul scales), no Q/K-norm, no PLE residual yet. Position 0.
    ///
    /// Pipeline:
    ///   hidden = embed(token_id)
    ///   q = projectQ(hidden); k = projectK(hidden); v = projectV(hidden)
    ///   rope(q, pos=0) ; rope(k, pos=0)
    ///   attn_in = self-attend(q, k, v)   // seq_len = 1
    ///   hidden += projectAttnO(attn_in)
    ///   gate = projectMlpGate(hidden); up = projectMlpUp(hidden)
    ///   swiglu(gate, up)
    ///   hidden += projectMlpDown(gate)
    ///
    /// Returns the updated hidden state (caller-owned buffer).
    pub fn forwardLayer0SingleToken(
        self: *const Gemma4Model,
        allocator: std.mem.Allocator,
        token_id: u32,
        hidden_out: []f32,
    ) !void {
        const math = @import("../kernels/math.zig");
        if (hidden_out.len != self.config.hidden) return error.ShapeMismatch;

        const q_dim: usize = @as(usize, self.config.n_q_heads) * self.config.head_dim;
        const kv_dim: usize = @as(usize, self.config.n_kv_heads) * self.config.head_dim;
        const ffn_dim: usize = self.config.ffn_dim_per_layer[0];

        // Stage 1: embed lookup.
        try self.lookupEmbedding(token_id, hidden_out);

        // Stage 2: Q/K/V projections.
        const q = try allocator.alloc(f32, q_dim);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, kv_dim);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, kv_dim);
        defer allocator.free(v);
        try self.projectQ(hidden_out, 0, q);
        try self.projectK(hidden_out, 0, k);
        try self.projectV(hidden_out, 0, v);

        // Stage 3: RoPE on Q + K (position 0, head-by-head).
        // Uses the Llama-style interleaved-pair RoPE for v1. Gemma 4
        // actually uses neox half-rotation — swap to `ropeHalf` once
        // verified.
        const head_dim: usize = self.config.head_dim;
        // Gemma 2/3/4 use 1e6 for the RoPE base (Llama default is 1e4).
        // At pos=0 this is a no-op; matters once KV history exists.
        const rope_base: f32 = 1_000_000.0; // standard Llama default; Gemma 4 may differ
        var h: usize = 0;
        while (h < self.config.n_q_heads) : (h += 1) {
            math.ropeHalf(q[h * head_dim ..][0..head_dim], 0, rope_base);
        }
        h = 0;
        while (h < self.config.n_kv_heads) : (h += 1) {
            math.ropeHalf(k[h * head_dim ..][0..head_dim], 0, rope_base);
        }

        // Stage 4: self-attention against this token only (seq=1).
        // For seq=1, the attention output is just V weighted by softmax
        // over a single score, which is always 1 → output = V (GQA
        // expansion).
        const attn_in = try allocator.alloc(f32, q_dim);
        defer allocator.free(attn_in);
        // Each Q head reads from V head `(h * n_kv_heads) / n_q_heads`.
        var qh: usize = 0;
        while (qh < self.config.n_q_heads) : (qh += 1) {
            const kv_h = (qh * self.config.n_kv_heads) / self.config.n_q_heads;
            const v_src = v[kv_h * head_dim ..][0..head_dim];
            const out_h = attn_in[qh * head_dim ..][0..head_dim];
            @memcpy(out_h, v_src);
        }

        // Stage 5: attn.o projection + residual.
        const attn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(attn_out);
        try self.projectAttnO(attn_in, 0, attn_out);
        math.addInto(hidden_out, attn_out);

        // Stage 6: MLP gate + up + SwiGLU.
        const gate = try allocator.alloc(f32, ffn_dim);
        defer allocator.free(gate);
        const up = try allocator.alloc(f32, ffn_dim);
        defer allocator.free(up);
        try self.projectMlpGate(hidden_out, 0, gate);
        try self.projectMlpUp(hidden_out, 0, up);
        math.swiglu(gate, up); // gate = silu(gate) * up

        // Stage 7: MLP down + residual.
        const ffn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(ffn_out);
        try self.projectMlpDown(gate, 0, ffn_out);
        math.addInto(hidden_out, ffn_out);
    }

    /// Run a single forward pass through every layer at position 0
    /// with no KV history. K/V are projected on full-attention layers
    /// (0..14 for Gemma 4 E2B) and reused on KV-shared layers
    /// (15..34). Returns the post-stack hidden state.
    ///
    /// Still missing for bit-equal reference: per-layer RMSNorm
    /// scales, Q/K-norm, PLE residual mix, neox-style RoPE convention,
    /// final post-stack norm. This is the minimum-viable multi-layer
    /// path — kernels exist and chain correctly; the gaps are glue
    /// + missing weight discovery.
    pub fn forwardSingleToken(
        self: *const Gemma4Model,
        allocator: std.mem.Allocator,
        token_id: u32,
        hidden_out: []f32,
    ) !void {
        const math = @import("../kernels/math.zig");
        if (hidden_out.len != self.config.hidden) return error.ShapeMismatch;

        const head_dim: usize = self.config.head_dim;
        // Gemma 2/3/4 use 1e6 for the RoPE base (Llama default is 1e4).
        // At pos=0 this is a no-op; matters once KV history exists.
        const rope_base: f32 = 1_000_000.0;

        try self.lookupEmbedding(token_id, hidden_out);

        // Per-layer scratch buffers. Q-output / KV-output / FFN dims
        // all vary per layer (MatFormer style); alloc once at the
        // largest seen across the stack.
        var max_q: usize = 0;
        var max_kv: usize = 0;
        var max_ffn: usize = 0;
        for (self.layers, 0..) |layer, i| {
            const q_sz: usize = @intCast(layer.q.shape[0]);
            if (q_sz > max_q) max_q = q_sz;
            if (layer.k) |k_t| {
                const kv_sz: usize = @intCast(k_t.shape[0]);
                if (kv_sz > max_kv) max_kv = kv_sz;
            }
            if (self.config.ffn_dim_per_layer[i] > max_ffn) max_ffn = self.config.ffn_dim_per_layer[i];
        }
        if (max_kv == 0) max_kv = max_q; // safety: no full-attn layer found

        const q = try allocator.alloc(f32, max_q);
        defer allocator.free(q);
        const cached_k = try allocator.alloc(f32, max_kv);
        defer allocator.free(cached_k);
        const cached_v = try allocator.alloc(f32, max_kv);
        defer allocator.free(cached_v);
        const attn_in = try allocator.alloc(f32, max_q);
        defer allocator.free(attn_in);
        const attn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(attn_out);
        const gate = try allocator.alloc(f32, max_ffn);
        defer allocator.free(gate);
        const up = try allocator.alloc(f32, max_ffn);
        defer allocator.free(up);
        const ffn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(ffn_out);

        // Initialise cached_k / cached_v as zero so a hypothetical
        // KV-shared layer at index 0 wouldn't read garbage. In Gemma
        // 4 E2B layer 0 owns its K/V so this never matters; future
        // models might differ.
        @memset(cached_k, 0);
        @memset(cached_v, 0);

        const rms_eps: f32 = 1.0e-6;
        // Scratch for the pre-projection normalised hidden state. Keeps
        // the residual stream in `hidden_out` untouched while
        // projecting Q/K/V/MLP off a normalised copy.
        const h_norm = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(h_norm);

        var layer_idx: usize = 0;
        while (layer_idx < self.layers.len) : (layer_idx += 1) {
            const layer = self.layers[layer_idx];
            // Per-layer attention shapes.
            const q_dim_l: usize = @intCast(layer.q.shape[0]);
            const n_q_l: usize = q_dim_l / head_dim;
            const n_kv_l: usize = if (layer.k) |k_t|
                @as(usize, @intCast(k_t.shape[0])) / head_dim
            else
                self.config.n_kv_heads;

            // Pre-attention norm (learned scale if available).
            @memcpy(h_norm, hidden_out);
            if (layer.norm_pre_attn) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(h_norm, w_f32, rms_eps);
            } else {
                math.rmsNormUnit(h_norm, rms_eps);
            }

            const q_slice = q[0..q_dim_l];
            try self.projectQ(h_norm, layer_idx, q_slice);
            if (layer.k != null) {
                const k_slice = cached_k[0 .. n_kv_l * head_dim];
                const v_slice = cached_v[0 .. n_kv_l * head_dim];
                try self.projectK(h_norm, layer_idx, k_slice);
                try self.projectV(h_norm, layer_idx, v_slice);
            }

            // Q-norm + K-norm (Gemma 4) applied after projection, before
            // RoPE. K-norm matches K's element count exactly; Q-norm
            // scale [n_kv_heads * head_dim] is broadcast across the
            // n_q_per_kv groups so each kv-head's q-fan-out shares the
            // same per-feature scale.
            if (layer.norm_query) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..@intCast(w_t.shape[0])];
                const group_size: usize = w_f32.len; // n_kv * head_dim
                const groups: usize = q_dim_l / group_size;
                var g: usize = 0;
                while (g < groups) : (g += 1) {
                    math.rmsNormGemma(q_slice[g * group_size ..][0..group_size], w_f32, rms_eps);
                }
            }
            if (layer.k != null) {
                if (layer.norm_key) |w_t| {
                    const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..@intCast(w_t.shape[0])];
                    const k_slice2 = cached_k[0 .. n_kv_l * head_dim];
                    math.rmsNormGemma(k_slice2, w_f32, rms_eps);
                }
            }

            // RoPE on Q + K (K only when freshly projected — once
            // cached, it's already rotated for pos 0).
            var hh: usize = 0;
            while (hh < n_q_l) : (hh += 1) {
                math.ropeHalf(q_slice[hh * head_dim ..][0..head_dim], 0, rope_base);
            }
            if (layer.k != null) {
                hh = 0;
                while (hh < n_kv_l) : (hh += 1) {
                    math.ropeHalf(cached_k[hh * head_dim ..][0..head_dim], 0, rope_base);
                }
            }
            // Seq=1 attention: each Q head's output equals its mapped V head.
            const attn_in_slice = attn_in[0..q_dim_l];
            var qh: usize = 0;
            while (qh < n_q_l) : (qh += 1) {
                const kv_h = (qh * n_kv_l) / n_q_l;
                const v_src = cached_v[kv_h * head_dim ..][0..head_dim];
                @memcpy(attn_in_slice[qh * head_dim ..][0..head_dim], v_src);
            }
            try self.projectAttnO(attn_in_slice, layer_idx, attn_out);
            // Post-attention norm damps the attention output before
            // the residual add — Gemma 2/3/4 sandwich norm pattern.
            if (layer.norm_post_attn) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(attn_out, w_f32, rms_eps);
            }
            math.addInto(hidden_out, attn_out);

            // Pre-MLP norm (learned scale if available — pre_ffw_norm).
            @memcpy(h_norm, hidden_out);
            if (layer.norm_pre_ffw) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(h_norm, w_f32, rms_eps);
            } else {
                math.rmsNormUnit(h_norm, rms_eps);
            }

            // MLP block (uses per-layer FFN dim).
            const ffn = self.config.ffn_dim_per_layer[layer_idx];
            const gate_slice = gate[0..ffn];
            const up_slice = up[0..ffn];
            try self.projectMlpGate(h_norm, layer_idx, gate_slice);
            try self.projectMlpUp(h_norm, layer_idx, up_slice);
            math.swiglu(gate_slice, up_slice);
            try self.projectMlpDown(gate_slice, layer_idx, ffn_out);
            // Post-FFW norm damps the MLP output before residual add.
            if (layer.norm_post_ffw) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(ffn_out, w_f32, rms_eps);
            }
            math.addInto(hidden_out, ffn_out);

            // PLE residual mix (Gemma's per-layer-embedding contribution).
            // Skipped when no per-layer-embedder shard for this layer.
            if (layer_idx < self.per_layer_embedder.len and self.per_layer_embedder[layer_idx] != null) {
                ple_scratch_residual_add(self, allocator, token_id, layer_idx, hidden_out) catch {};
            }

            // Per-layer skip-scale scalar damps the residual stream
            // before the next layer. Single FP32 scalar.
            if (layer.skip_scale) |s_t| {
                const s: f32 = @as([*]const f32, @ptrCast(@alignCast(s_t.data.ptr)))[0];
                var ii: usize = 0;
                while (ii < hidden_out.len) : (ii += 1) hidden_out[ii] *= s;
            }
        }
    }

    /// Run the PLE residual block for `layer_idx`:
    ///   ple_in[ple_dim] = lookupPerLayerEmbedding(token, layer)
    ///   gate_out[ple_dim] = projectPleGate(hidden_out)
    ///   gated[ple_dim] = ple_in * gelu(gate_out)        // approximate
    ///   residual[hidden] = projectPleProj(gated)
    ///   residual = post_per_layer_input_norm(residual)   // when available
    ///   hidden_out += residual
    fn ple_scratch_residual_add(
        self: *const Gemma4Model,
        allocator: std.mem.Allocator,
        token_id: u32,
        layer_idx: usize,
        hidden_out: []f32,
    ) !void {
        const math = @import("../kernels/math.zig");
        const ple_dim: usize = self.config.ple_dim;
        const ple_in = try allocator.alloc(f32, ple_dim);
        defer allocator.free(ple_in);
        const gate_out = try allocator.alloc(f32, ple_dim);
        defer allocator.free(gate_out);
        const residual = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(residual);

        try self.lookupPerLayerEmbedding(token_id, layer_idx, ple_in);
        try self.projectPleGate(hidden_out, layer_idx, gate_out);

        // GELU activation (tanh approximation):
        //   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // TFLite builtin 150 = GELU (verified against upstream schema
        // `tensorflow/compiler/mlir/lite/schema/schema.fbs`).
        var i: usize = 0;
        while (i < ple_dim) : (i += 1) {
            const x = gate_out[i];
            const t = 0.7978845608 * (x + 0.044715 * x * x * x);
            const tanh_t = std.math.tanh(t);
            const g = 0.5 * x * (1.0 + tanh_t);
            ple_in[i] *= g;
        }
        try self.projectPleProj(ple_in, layer_idx, residual);

        // post_per_layer_input_norm damps the PLE residual before add.
        const layer = self.layers[layer_idx];
        if (layer.norm_post_per_layer) |w_t| {
            const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
            math.rmsNormGemma(residual, w_f32, 1.0e-6);
        }
        math.addInto(hidden_out, residual);
    }

    /// Single-step forward at the given absolute position, with KV
    /// cache. Updates `cache.pos` to `position + 1` on return.
    pub fn step(
        self: *const Gemma4Model,
        allocator: std.mem.Allocator,
        token_id: u32,
        position: usize,
        cache: *KvCache,
        hidden_out: []f32,
    ) !void {
        const math = @import("../kernels/math.zig");
        if (hidden_out.len != self.config.hidden) return error.ShapeMismatch;
        if (position >= cache.max_seq) return error.KvCacheFull;

        const head_dim: usize = self.config.head_dim;
        const rope_base: f32 = 1_000_000.0;
        const rms_eps: f32 = 1.0e-6;
        const seq_len: usize = position + 1;
        const inv_sqrt_hd: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        try self.lookupEmbedding(token_id, hidden_out);

        // Determine scratch buffer sizes (max across all layers).
        var max_q: usize = 0;
        var max_kv: usize = 0;
        var max_ffn: usize = 0;
        for (self.layers, 0..) |layer, i| {
            const q_sz: usize = @intCast(layer.q.shape[0]);
            if (q_sz > max_q) max_q = q_sz;
            if (layer.k) |kt| {
                const kvs: usize = @intCast(kt.shape[0]);
                if (kvs > max_kv) max_kv = kvs;
            }
            if (self.config.ffn_dim_per_layer[i] > max_ffn) max_ffn = self.config.ffn_dim_per_layer[i];
        }
        if (max_kv == 0) max_kv = max_q;

        const q = try allocator.alloc(f32, max_q);
        defer allocator.free(q);
        const k_curr = try allocator.alloc(f32, max_kv);
        defer allocator.free(k_curr);
        const v_curr = try allocator.alloc(f32, max_kv);
        defer allocator.free(v_curr);
        const attn_in = try allocator.alloc(f32, max_q);
        defer allocator.free(attn_in);
        const attn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(attn_out);
        const gate = try allocator.alloc(f32, max_ffn);
        defer allocator.free(gate);
        const up = try allocator.alloc(f32, max_ffn);
        defer allocator.free(up);
        const ffn_out = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(ffn_out);
        const h_norm = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(h_norm);
        const scores = try allocator.alloc(f32, seq_len);
        defer allocator.free(scores);

        var layer_idx: usize = 0;
        while (layer_idx < self.layers.len) : (layer_idx += 1) {
            const layer = self.layers[layer_idx];
            const q_dim_l: usize = @intCast(layer.q.shape[0]);
            const n_q_l: usize = q_dim_l / head_dim;
            const n_kv_l: usize = if (layer.k) |k_t|
                @as(usize, @intCast(k_t.shape[0])) / head_dim
            else
                @as(usize, @intCast(self.layers[cache.parent_of_shared[layer_idx]].k.?.shape[0])) / head_dim;
            const kv_dim_l: usize = n_kv_l * head_dim;

            // Pre-attention norm.
            @memcpy(h_norm, hidden_out);
            if (layer.norm_pre_attn) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(h_norm, w_f32, rms_eps);
            } else {
                math.rmsNormUnit(h_norm, rms_eps);
            }

            // Project Q, K, V (K/V only on owner layers — write into cache slot).
            const q_slice = q[0..q_dim_l];
            try self.projectQ(h_norm, layer_idx, q_slice);

            if (layer.k != null) {
                const k_slot = cache.k_per_layer[layer_idx][position * kv_dim_l ..][0..kv_dim_l];
                const v_slot = cache.v_per_layer[layer_idx][position * kv_dim_l ..][0..kv_dim_l];
                try self.projectK(h_norm, layer_idx, k_slot);
                try self.projectV(h_norm, layer_idx, v_slot);
                @memcpy(k_curr[0..kv_dim_l], k_slot);
                @memcpy(v_curr[0..kv_dim_l], v_slot);

                // Q/K-norm + RoPE on the fresh K vector before caching.
                // Actually applied above to k_slot already (need to redo).
                // For simplicity apply norms+rope to k_curr and copy back.
                if (layer.norm_key) |w_t| {
                    const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..@intCast(w_t.shape[0])];
                    math.rmsNormGemma(k_curr[0..kv_dim_l], w_f32, rms_eps);
                }
                var hh: usize = 0;
                while (hh < n_kv_l) : (hh += 1) {
                    math.ropeHalf(k_curr[hh * head_dim ..][0..head_dim], position, rope_base);
                }
                @memcpy(k_slot, k_curr[0..kv_dim_l]);
            }
            // Q-norm + RoPE.
            if (layer.norm_query) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..@intCast(w_t.shape[0])];
                const group_size: usize = w_f32.len;
                const groups: usize = q_dim_l / group_size;
                var g: usize = 0;
                while (g < groups) : (g += 1) {
                    math.rmsNormGemma(q_slice[g * group_size ..][0..group_size], w_f32, rms_eps);
                }
            }
            var hh: usize = 0;
            while (hh < n_q_l) : (hh += 1) {
                math.ropeHalf(q_slice[hh * head_dim ..][0..head_dim], position, rope_base);
            }

            // Real attention over the cached prefix.
            const k_cache = cache.k_per_layer[layer_idx];
            const v_cache = cache.v_per_layer[layer_idx];
            const attn_in_slice = attn_in[0..q_dim_l];
            @memset(attn_in_slice, 0);
            var qh: usize = 0;
            while (qh < n_q_l) : (qh += 1) {
                const kv_h = (qh * n_kv_l) / n_q_l;
                const q_h = q_slice[qh * head_dim ..][0..head_dim];

                // scores[t] = (q_h · k_cache[t, kv_h]) * inv_sqrt_hd
                var t: usize = 0;
                while (t < seq_len) : (t += 1) {
                    const k_row = k_cache[t * kv_dim_l + kv_h * head_dim ..][0..head_dim];
                    var s: f32 = 0;
                    for (q_h, k_row) |qv, kv| s += qv * kv;
                    scores[t] = s * inv_sqrt_hd;
                }
                math.softmax(scores);

                const out_h = attn_in_slice[qh * head_dim ..][0..head_dim];
                t = 0;
                while (t < seq_len) : (t += 1) {
                    const v_row = v_cache[t * kv_dim_l + kv_h * head_dim ..][0..head_dim];
                    const w = scores[t];
                    for (out_h, v_row) |*o, vv| o.* += w * vv;
                }
            }

            try self.projectAttnO(attn_in_slice, layer_idx, attn_out);
            if (layer.norm_post_attn) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(attn_out, w_f32, rms_eps);
            }
            math.addInto(hidden_out, attn_out);

            // Pre-MLP norm + MLP block.
            @memcpy(h_norm, hidden_out);
            if (layer.norm_pre_ffw) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(h_norm, w_f32, rms_eps);
            } else {
                math.rmsNormUnit(h_norm, rms_eps);
            }
            const ffn = self.config.ffn_dim_per_layer[layer_idx];
            const gate_slice = gate[0..ffn];
            const up_slice = up[0..ffn];
            try self.projectMlpGate(h_norm, layer_idx, gate_slice);
            try self.projectMlpUp(h_norm, layer_idx, up_slice);
            math.swiglu(gate_slice, up_slice);
            try self.projectMlpDown(gate_slice, layer_idx, ffn_out);
            if (layer.norm_post_ffw) |w_t| {
                const w_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_t.data.ptr)))[0..self.config.hidden];
                math.rmsNormGemma(ffn_out, w_f32, rms_eps);
            }
            math.addInto(hidden_out, ffn_out);

            // PLE residual.
            if (layer_idx < self.per_layer_embedder.len and self.per_layer_embedder[layer_idx] != null) {
                ple_scratch_residual_add(self, allocator, token_id, layer_idx, hidden_out) catch {};
            }

            // Skip-scale.
            if (layer.skip_scale) |s_t| {
                const s: f32 = @as([*]const f32, @ptrCast(@alignCast(s_t.data.ptr)))[0];
                var ii: usize = 0;
                while (ii < hidden_out.len) : (ii += 1) hidden_out[ii] *= s;
            }
        }

        cache.pos = position + 1;
    }

    /// LM head: computes vocab-size logits from the post-stack hidden
    /// state. Uses `lm_head` (Gemma 4's dedicated decode-time embedder
    /// living inside the decoder section) when present; falls back to
    /// the section-2 `embedder` table (weight tying) otherwise.
    ///
    /// `hidden.len == config.hidden`; `logits.len == config.vocab_size`.
    pub fn lmHead(self: *const Gemma4Model, hidden: []const f32, logits: []f32) !void {
        const t = self.lm_head orelse self.embedder orelse return error.NoEmbedder;
        if (hidden.len != self.config.hidden) return error.ShapeMismatch;
        if (logits.len != self.config.vocab_size) return error.ShapeMismatch;
        const m: usize = @intCast(t.shape[0]); // vocab
        const k: usize = @intCast(t.shape[1]); // hidden
        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len < m or zps.len < m) return error.ShapeMismatch;
        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        try tflite_int4.gemvInt2PerRow(t.data, scales, zps, hidden, logits, m, k);
    }

    /// Generate `n_new` tokens greedily starting from `prompt_ids`.
    /// Caller-owned output buffer; populates with newly-generated ids.
    /// Returns the number actually written (could be less if cache fills).
    pub fn generate(
        self: *const Gemma4Model,
        allocator: std.mem.Allocator,
        prompt_ids: []const u32,
        n_new: usize,
        max_seq: usize,
        out_tokens: []u32,
    ) !usize {
        const math = @import("../kernels/math.zig");
        var cache = try self.initKvCache(allocator, max_seq);
        defer cache.deinit();

        const hidden = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(hidden);
        const logits = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(logits);

        // Prefill the prompt; keep the last hidden state to predict next.
        var pos: usize = 0;
        for (prompt_ids) |tid| {
            if (pos >= max_seq) break;
            try self.step(allocator, tid, pos, &cache, hidden);
            pos += 1;
        }
        if (pos == 0) return 0;

        var produced: usize = 0;
        while (produced < n_new and produced < out_tokens.len and pos < max_seq) : (produced += 1) {
            // hidden currently holds the post-stack state for the last
            // fed token. RMSNorm + LM head argmax → next.
            const h_copy = try allocator.alloc(f32, self.config.hidden);
            defer allocator.free(h_copy);
            @memcpy(h_copy, hidden);
            math.rmsNormUnit(h_copy, 1.0e-6);
            try self.lmHead(h_copy, logits);
            var best: u32 = 0;
            var best_v: f32 = logits[0];
            var i: usize = 1;
            while (i < logits.len) : (i += 1) {
                if (logits[i] > best_v) {
                    best_v = logits[i];
                    best = @intCast(i);
                }
            }
            out_tokens[produced] = best;
            // Feed the new token for the next round.
            try self.step(allocator, best, pos, &cache, hidden);
            pos += 1;
        }
        return produced;
    }

    /// End-to-end: token_id → next-token argmax. Runs forwardSingleToken,
    /// final RMSNorm on the residual stream, LM head via embedder
    /// weight-tying, then argmax.
    pub fn predictNext(self: *const Gemma4Model, allocator: std.mem.Allocator, token_id: u32) !u32 {
        const math = @import("../kernels/math.zig");
        const hidden = try allocator.alloc(f32, self.config.hidden);
        defer allocator.free(hidden);
        try self.forwardSingleToken(allocator, token_id, hidden);
        // Final RMSNorm before LM head — every transformer applies one.
        math.rmsNormUnit(hidden, 1.0e-6);

        const logits = try allocator.alloc(f32, self.config.vocab_size);
        defer allocator.free(logits);
        try self.lmHead(hidden, logits);

        var best: u32 = 0;
        var best_v: f32 = logits[0];
        var i: usize = 1;
        while (i < logits.len) : (i += 1) {
            if (logits[i] > best_v) {
                best_v = logits[i];
                best = @intCast(i);
            }
        }
        return best;
    }

    fn projectInt8Generic(t: *const tflite.Tensor, x: []const f32, y: []f32) !void {
        if (t.shape.len < 2) return error.ShapeMismatch;
        const m: usize = @intCast(t.shape[0]);
        const k: usize = @intCast(t.shape[1]);
        if (y.len != m or x.len != k) return error.ShapeMismatch;
        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len < m or zps.len < m) return error.ShapeMismatch;
        // PLE tensors are INT8; reinterpret the raw bytes.
        const weights: []const i8 = @as([*]const i8, @ptrCast(t.data.ptr))[0..t.data.len];
        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        try tflite_int4.gemvInt8PerRow(weights, scales, zps, x, y, m, k);
    }

    /// Decode one row of the INT4 per-layer embedder for `layer_idx`.
    /// `out.len == config.ple_dim` (256 for Gemma 4 E2B).
    pub fn lookupPerLayerEmbedding(
        self: *const Gemma4Model,
        token_id: u32,
        layer_idx: usize,
        out: []f32,
    ) !void {
        if (layer_idx >= self.per_layer_embedder.len) return error.LayerOutOfRange;
        const t = self.per_layer_embedder[layer_idx] orelse return error.NoPerLayerEmbedder;
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (out.len != self.config.ple_dim) return error.ShapeMismatch;

        const k = self.config.ple_dim;
        const row_bytes_per_token = (k + 1) / 2; // INT4 = 2 values per byte
        const row_off_bytes = @as(usize, token_id) * row_bytes_per_token;
        const row_bytes = t.data[row_off_bytes .. row_off_bytes + row_bytes_per_token];

        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len <= token_id or zps.len <= token_id) return error.ShapeMismatch;
        const scale = scales[token_id];
        const zp = zps[token_id];
        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        var col: usize = 0;
        while (col < k) : (col += 1) {
            const v: i32 = tflite_int4.unpackInt4(row_bytes, col);
            const centered: i32 = v - @as(i32, @intCast(zp));
            out[col] = @as(f32, @floatFromInt(centered)) * scale;
        }
    }

    /// Decode one row of the INT2 text embedder into FP32. `token_id`
    /// must be < `config.vocab_size`; `out.len` must equal
    /// `config.hidden`.
    pub fn lookupEmbedding(self: *const Gemma4Model, token_id: u32, out: []f32) !void {
        const t = self.embedder orelse return error.NoEmbedder;
        if (token_id >= self.config.vocab_size) return error.TokenOutOfRange;
        if (out.len != self.config.hidden) return error.ShapeMismatch;

        const k = self.config.hidden;
        const row_bytes_per_token = (k + 3) / 4; // INT2 = 4 values per byte
        const row_off_bytes = @as(usize, token_id) * row_bytes_per_token;
        const row_bytes = t.data[row_off_bytes .. row_off_bytes + row_bytes_per_token];

        // Per-axis quantization: scales array is `vocab` entries when
        // quantized_dimension == 0 (per-row). Each row has own scale + zp.
        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len <= token_id or zps.len <= token_id) return error.ShapeMismatch;

        const scale = scales[token_id];
        const zp = zps[token_id];

        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        var col: usize = 0;
        while (col < k) : (col += 1) {
            const v: i32 = tflite_int4.unpackInt2(row_bytes, col);
            const centered: i32 = v - @as(i32, @intCast(zp));
            out[col] = @as(f32, @floatFromInt(centered)) * scale;
        }
    }
};

/// Build a Gemma4Model from a parsed `.litertlm` bundle. The bundle
/// must outlive the returned Gemma4Model — tensor pointers borrow
/// from the underlying mmap.
pub fn initFromLitertlm(
    allocator: std.mem.Allocator,
    bundle: *const litertlm.Bundle,
) Error!Gemma4Model {
    // Find the decoder section: the TFLiteModel with the most layers
    // (35 for Gemma 4 E2B, found in section 10).
    var best_section: ?usize = null;
    var best_n_layers: usize = 0;

    for (bundle.sections, 0..) |s, sec_idx| {
        if (s.data_type != .tflite_model) continue;
        const bytes = bundle.sectionBytes(s) catch continue;
        var tfl = tflite.Model.init(allocator, bytes) catch continue;
        defer tfl.deinit();
        const probe = scan.scanLayers(allocator, tfl.tensors) catch continue;
        defer allocator.free(probe);
        if (probe.len > best_n_layers) {
            best_n_layers = probe.len;
            best_section = sec_idx;
        }
    }

    const sec_idx = best_section orelse return error.NoDecoderSection;
    if (best_n_layers == 0) return error.LayerCountZero;

    // Re-open the winning section and KEEP it alive for the model's
    // lifetime — layer pointers reference its tensor storage.
    const decoder_section = bundle.sections[sec_idx];
    const decoder_bytes = try bundle.sectionBytes(decoder_section);
    var decoder_tfl = try tflite.Model.init(allocator, decoder_bytes);
    errdefer decoder_tfl.deinit();
    const buckets = try scan.scanLayers(allocator, decoder_tfl.tensors);
    defer allocator.free(buckets);

    // Translate buckets into Gemma4Layer pointers. Drop layers missing
    // a Q projection (shouldn't happen — Q exists on every layer per
    // our 35/35 scan).
    const layers = try allocator.alloc(Gemma4Layer, buckets.len);
    errdefer allocator.free(layers);
    var resolved: usize = 0;
    for (buckets) |b| {
        const q = b.q_proj orelse continue;
        const ao = b.attn_o orelse continue;
        const mg = b.mlp_gate orelse continue;
        const mu = b.mlp_up orelse continue;
        const md = b.mlp_down orelse continue;
        const pg = b.ple_gate orelse continue;
        const pp = b.ple_proj orelse continue;
        layers[resolved] = .{
            .q = q,
            .k = b.k_proj, // optional — KV-shared layers leave this null
            .v = b.v_proj,
            .attn_o = ao,
            .mlp_gate = mg,
            .mlp_up = mu,
            .mlp_down = md,
            .ple_gate = pg,
            .ple_proj = pp,
            .rope_inv_freq = b.rope_inv_freq,
        };
        resolved += 1;
    }
    // Trim trailing space if some layers didn't resolve.
    const final_layers = if (resolved < layers.len)
        try allocator.realloc(layers, resolved)
    else
        layers;

    // Architecture inference from layer 0 shapes.
    const l0 = final_layers[0];
    const hidden: u32 = blk: {
        // Q weight shape is [n_q_heads*head_dim, hidden]. Hidden is the
        // second axis.
        if (l0.q.shape.len >= 2) break :blk @intCast(l0.q.shape[1]);
        return error.LayerCountZero; // unreachable in practice
    };
    const q_out: u32 = @intCast(l0.q.shape[0]);
    const kv_out: u32 = blk: {
        if (l0.k) |k| break :blk @intCast(k.shape[0]);
        // KV-shared on layer 0 (unexpected for Gemma 4 E2B) — fall back
        // to scanning later layers.
        for (final_layers) |layer| if (layer.k) |k| break :blk @intCast(k.shape[0]);
        break :blk q_out; // last-resort guess
    };

    // RoPE inv_freq table shape is [1, 1, head_dim].
    const head_dim: u32 = blk: {
        for (final_layers) |layer| {
            if (layer.rope_inv_freq) |rf| if (rf.shape.len >= 1) {
                break :blk @intCast(rf.shape[rf.shape.len - 1]);
            };
        }
        // Fallback: assume head_dim divides Q output evenly. Gemma 4
        // E2B uses head_dim=128.
        break :blk 128;
    };
    const n_q_heads: u32 = q_out / head_dim;
    const n_kv_heads: u32 = kv_out / head_dim;

    // PLE width from PLE projection shape [hidden, ple_dim].
    const ple_dim: u32 = if (l0.ple_proj.shape.len >= 2)
        @intCast(l0.ple_proj.shape[1])
    else
        256;

    // Per-layer FFN inner dim from mlp_gate shape [ffn, hidden].
    const ffn_dims = try allocator.alloc(u32, final_layers.len);
    errdefer allocator.free(ffn_dims);
    for (final_layers, 0..) |layer, i| {
        ffn_dims[i] = if (layer.mlp_gate.shape.len >= 1)
            @intCast(layer.mlp_gate.shape[0])
        else
            0;
    }

    // Discover the LM head: `decode_softmax/.../embedder.decode/composite`
    // lives inside the decoder section itself. Distinct from the
    // section-2 `lookup_embedding_table`. Both are [vocab, hidden]
    // INT2 but may carry different per-row scales.
    var lm_head_tensor: ?*const tflite.Tensor = null;
    {
        var best_size: usize = 0;
        for (decoder_tfl.tensors) |*t| {
            if (t.data.len == 0) continue;
            if (std.mem.indexOf(u8, t.name, "embedder.decode") == null) continue;
            if (t.data.len > best_size) {
                best_size = t.data.len;
                lm_head_tensor = t;
            }
        }
    }

    // Find the text embedder. Look for a tensor whose name contains
    // "embedder.lookup_embedding_table" (Gemma 4 convention) with the
    // largest data — the actual lookup table, not metadata. Lives in a
    // different TFLite section than the decoder (section 2 in E2B).
    var embedder_section_idx: ?usize = null;
    var embedder_tfl_opt: ?tflite.Model = null;
    var embedder_tensor: ?*const tflite.Tensor = null;
    var vocab_size: u32 = 0;

    for (bundle.sections, 0..) |s, sec_i| {
        if (s.data_type != .tflite_model) continue;
        if (sec_i == sec_idx) continue; // decoder section, not embedder
        const bytes = bundle.sectionBytes(s) catch continue;
        var tfl = tflite.Model.init(allocator, bytes) catch continue;
        var best_tensor: ?*const tflite.Tensor = null;
        var best_size: usize = 0;
        for (tfl.tensors) |*t| {
            if (t.data.len == 0) continue;
            if (std.mem.indexOf(u8, t.name, "embedder.lookup_embedding_table") == null) continue;
            if (std.mem.indexOf(u8, t.name, "per_layer") != null) continue; // skip the per-layer embedder
            if (t.data.len > best_size) {
                best_size = t.data.len;
                best_tensor = t;
            }
        }
        if (best_tensor) |t| {
            embedder_section_idx = sec_i;
            embedder_tensor = t;
            if (t.shape.len >= 2) vocab_size = @intCast(t.shape[0]);
            embedder_tfl_opt = tfl;
            break;
        }
        tfl.deinit();
    }

    // Discover the 35 per-layer-embedder lookup tables. They live in
    // a section separate from both decoder + embedder (section 3 in
    // E2B, 1.2 GB total). Naming: `per_layer_embedder.lookup_embedding_table/composite[N]`
    // where N is the layer index (suffix "" means layer 0).
    const n_layers_usize: usize = final_layers.len;
    const ple_table = try allocator.alloc(?*const tflite.Tensor, n_layers_usize);
    errdefer allocator.free(ple_table);
    @memset(ple_table, null);
    var per_layer_tfl_opt: ?tflite.Model = null;

    for (bundle.sections, 0..) |s, sec_i| {
        if (s.data_type != .tflite_model) continue;
        if (sec_i == sec_idx) continue;
        if (embedder_section_idx) |e| if (sec_i == e) continue;
        const bytes = bundle.sectionBytes(s) catch continue;
        var tfl = tflite.Model.init(allocator, bytes) catch continue;
        var matched_any = false;
        for (tfl.tensors) |*t| {
            if (t.data.len == 0) continue;
            const prefix = "per_layer_embedder.lookup_embedding_table/composite";
            const pos = std.mem.indexOf(u8, t.name, prefix) orelse continue;
            const after = t.name[pos + prefix.len ..];
            // Suffix may be empty (layer 0), an integer (layer N), or
            // include trailing path components. Trim at first '/'.
            const num_end = std.mem.indexOfScalar(u8, after, '/') orelse after.len;
            const num_str = after[0..num_end];
            const layer_n: u32 = if (num_str.len == 0)
                0
            else
                std.fmt.parseInt(u32, num_str, 10) catch continue;
            if (layer_n >= n_layers_usize) continue;
            // Prefer larger tensors when slot already filled.
            if (ple_table[layer_n]) |existing| {
                if (t.data.len <= existing.data.len) continue;
            }
            ple_table[layer_n] = t;
            matched_any = true;
        }
        if (matched_any) {
            per_layer_tfl_opt = tfl;
            break;
        }
        tfl.deinit();
    }

    // Discover learned RMSNorm scales. Pattern: each `*_norm/composite`
    // tensor in section 10 is produced by an op consuming the residual
    // stream + a FP32 [hidden] scale weight. Walk the ops, find norm
    // outputs by name, and pin the scale input by elimination (the
    // input that's a constant tensor with FP32 [hidden] data).
    {
        const NormRole = enum { pre_attn, post_attn, pre_ffw, post_ffw, post_per_layer, query, key, none };
        for (decoder_tfl.operators) |op| {
            if (op.outputs.len != 1 or op.inputs.len < 2) continue;
            const out_idx_i: i32 = op.outputs[0];
            if (out_idx_i < 0) continue;
            const out_idx: usize = @intCast(out_idx_i);
            if (out_idx >= decoder_tfl.tensors.len) continue;
            const out_t = &decoder_tfl.tensors[out_idx];
            // Identify (layer, role) from the output tensor's name.
            const name = out_t.name;
            const norm_marker = "_norm/composite";
            if (std.mem.indexOf(u8, name, norm_marker) == null) continue;
            const layer_n_opt = scan.parseLayerIndex(name);
            if (layer_n_opt == null) continue;
            const layer_n = layer_n_opt.?;
            if (layer_n >= final_layers.len) continue;

            const role: NormRole = if (std.mem.indexOf(u8, name, "pre_attention_norm") != null)
                .pre_attn
            else if (std.mem.indexOf(u8, name, "post_attention_norm") != null)
                .post_attn
            else if (std.mem.indexOf(u8, name, "pre_ffw_norm") != null)
                .pre_ffw
            else if (std.mem.indexOf(u8, name, "post_ffw_norm") != null)
                .post_ffw
            else if (std.mem.indexOf(u8, name, "post_per_layer_input_norm") != null)
                .post_per_layer
            else if (std.mem.indexOf(u8, name, "query_norm") != null)
                .query
            else if (std.mem.indexOf(u8, name, "key_norm") != null)
                .key
            else
                .none;
            if (role == .none) continue;

            // Block norms have shape [hidden]; Q/K norms vary across
            // layers (256 in standard layers, 512 in wider-attention
            // layer 4, ...). For block norms enforce the exact dim
            // check; for Q/K accept any FP32 1D tensor with non-empty
            // data (the op-output name already disambiguates the role).
            const expected_dim_opt: ?u32 = switch (role) {
                .query, .key => null,
                else => @as(u32, @intCast(hidden)),
            };
            var scale_tensor: ?*const tflite.Tensor = null;
            for (op.inputs) |in_idx_i| {
                if (in_idx_i < 0) continue;
                const in_idx: usize = @intCast(in_idx_i);
                if (in_idx >= decoder_tfl.tensors.len) continue;
                const in_t = &decoder_tfl.tensors[in_idx];
                if (in_t.dtype != .float32) continue;
                if (in_t.data.len == 0) continue;
                if (in_t.shape.len != 1) continue;
                if (expected_dim_opt) |d| if (in_t.shape[0] != @as(i32, @intCast(d))) continue;
                scale_tensor = in_t;
                break;
            }
            if (scale_tensor) |st| {
                const layer = &final_layers[layer_n];
                switch (role) {
                    .pre_attn => if (layer.norm_pre_attn == null) {
                        layer.norm_pre_attn = st;
                    },
                    .post_attn => if (layer.norm_post_attn == null) {
                        layer.norm_post_attn = st;
                    },
                    .pre_ffw => if (layer.norm_pre_ffw == null) {
                        layer.norm_pre_ffw = st;
                    },
                    .post_ffw => if (layer.norm_post_ffw == null) {
                        layer.norm_post_ffw = st;
                    },
                    .post_per_layer => if (layer.norm_post_per_layer == null) {
                        layer.norm_post_per_layer = st;
                    },
                    .query => if (layer.norm_query == null) {
                        layer.norm_query = st;
                    },
                    .key => if (layer.norm_key == null) {
                        layer.norm_key = st;
                    },
                    .none => unreachable,
                }
            }
        }
    }

    // Discover per-layer `_maybe_apply_skip_scale` scalars. FP32
    // shape [1,1,1] data=4 bytes. Name contains
    // `layer_N._maybe_apply_skip_scale/broadcast_in_dim`.
    for (decoder_tfl.tensors) |*t| {
        if (t.dtype != .float32) continue;
        if (t.data.len != 4) continue;
        const marker = "_maybe_apply_skip_scale/broadcast_in_dim";
        if (std.mem.indexOf(u8, t.name, marker) == null) continue;
        const layer_n_opt = scan.parseLayerIndex(t.name);
        if (layer_n_opt == null) continue;
        const ln = layer_n_opt.?;
        if (ln >= final_layers.len) continue;
        if (final_layers[ln].skip_scale != null) continue;
        final_layers[ln].skip_scale = t;
    }

    return .{
        .config = .{
            .hidden = hidden,
            .n_layers = @intCast(final_layers.len),
            .n_q_heads = n_q_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .ffn_dim_per_layer = ffn_dims,
            .vocab_size = vocab_size,
            .ple_dim = ple_dim,
        },
        .layers = final_layers,
        .embedder = embedder_tensor,
        .lm_head = lm_head_tensor,
        .per_layer_embedder = ple_table,
        .decoder_tfl = decoder_tfl,
        .embedder_tfl = embedder_tfl_opt,
        .per_layer_tfl = per_layer_tfl_opt,
        .decoder_section_idx = sec_idx,
        .allocator = allocator,
    };
}


// -- tests ----------------------------------------------------------------

test "Gemma4Config.deinit frees ffn_dim_per_layer" {
    const gpa = std.testing.allocator;
    const dims = try gpa.alloc(u32, 4);
    @memcpy(dims, &[_]u32{ 6144, 6144, 12288, 12288 });
    var cfg: Gemma4Config = .{
        .hidden = 1536,
        .n_layers = 4,
        .n_q_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 128,
        .ffn_dim_per_layer = dims,
        .vocab_size = 262144,
        .ple_dim = 256,
    };
    cfg.deinit(gpa);
    // No assertion needed — `testing.allocator` panics on leak.
}
