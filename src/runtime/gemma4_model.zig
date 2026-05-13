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
};

pub const Gemma4Model = struct {
    /// Architecture parameters.
    config: Gemma4Config,
    /// Per-layer weights. `layers.len == config.n_layers`.
    layers: []Gemma4Layer,
    /// Text embedding lookup table (INT2 per-row, shape [vocab, hidden]).
    /// Null if no embedder section was found.
    embedder: ?*const tflite.Tensor = null,
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
        self.decoder_tfl.deinit();
        if (self.embedder_tfl) |*tfl| tfl.deinit();
        self.* = undefined;
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
        const rope_base: f32 = 10000.0; // standard Llama default; Gemma 4 may differ
        var h: usize = 0;
        while (h < self.config.n_q_heads) : (h += 1) {
            math.rope(q[h * head_dim ..][0..head_dim], 0, rope_base);
        }
        h = 0;
        while (h < self.config.n_kv_heads) : (h += 1) {
            math.rope(k[h * head_dim ..][0..head_dim], 0, rope_base);
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
        const rope_base: f32 = 10000.0;

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

            // Pre-attention norm (unit RMSNorm — no learned scale yet).
            @memcpy(h_norm, hidden_out);
            math.rmsNormUnit(h_norm, rms_eps);

            const q_slice = q[0..q_dim_l];
            try self.projectQ(h_norm, layer_idx, q_slice);
            if (layer.k != null) {
                const k_slice = cached_k[0 .. n_kv_l * head_dim];
                const v_slice = cached_v[0 .. n_kv_l * head_dim];
                try self.projectK(h_norm, layer_idx, k_slice);
                try self.projectV(h_norm, layer_idx, v_slice);
            }
            // RoPE on Q + K (K only when freshly projected — once
            // cached, it's already rotated for pos 0).
            var hh: usize = 0;
            while (hh < n_q_l) : (hh += 1) {
                math.rope(q_slice[hh * head_dim ..][0..head_dim], 0, rope_base);
            }
            if (layer.k != null) {
                hh = 0;
                while (hh < n_kv_l) : (hh += 1) {
                    math.rope(cached_k[hh * head_dim ..][0..head_dim], 0, rope_base);
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
            math.addInto(hidden_out, attn_out);

            // Pre-MLP norm.
            @memcpy(h_norm, hidden_out);
            math.rmsNormUnit(h_norm, rms_eps);

            // MLP block (uses per-layer FFN dim).
            const ffn = self.config.ffn_dim_per_layer[layer_idx];
            const gate_slice = gate[0..ffn];
            const up_slice = up[0..ffn];
            try self.projectMlpGate(h_norm, layer_idx, gate_slice);
            try self.projectMlpUp(h_norm, layer_idx, up_slice);
            math.swiglu(gate_slice, up_slice);
            try self.projectMlpDown(gate_slice, layer_idx, ffn_out);
            math.addInto(hidden_out, ffn_out);
        }
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
        .decoder_tfl = decoder_tfl,
        .embedder_tfl = embedder_tfl_opt,
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
