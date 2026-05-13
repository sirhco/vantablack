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

    /// Project hidden state through layer `layer_idx`'s Q matrix.
    /// `hidden_in.len == config.hidden`; `q_out.len == n_q_heads * head_dim`.
    /// Uses the per-row INT4 fused GEMV kernel against the borrowed Q
    /// tensor. Per-axis quantization scales include any pre-projection
    /// RMSNorm fold (TFLite optimization).
    pub fn projectQ(self: *const Gemma4Model, hidden_in: []const f32, layer_idx: usize, q_out: []f32) !void {
        if (layer_idx >= self.layers.len) return error.LayerOutOfRange;
        if (hidden_in.len != self.config.hidden) return error.ShapeMismatch;
        const layer = self.layers[layer_idx];
        const t = layer.q;
        const m: usize = @intCast(t.shape[0]); // n_q_heads * head_dim
        const k: usize = @intCast(t.shape[1]); // hidden
        if (q_out.len != m) return error.ShapeMismatch;
        if (k != hidden_in.len) return error.ShapeMismatch;

        const scales = tflite.scalesAsF32(t.scales);
        const zps = tflite.zeroPointsAsI64(t.zero_points);
        if (scales.len < m or zps.len < m) return error.ShapeMismatch;

        const tflite_int4 = @import("../kernels/tflite_int4.zig");
        try tflite_int4.gemvInt4PerRow(t.data, scales, zps, hidden_in, q_out, m, k);
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
