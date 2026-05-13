//! Tensor-name → semantic-role mapper for Gemma 4 `.litertlm` decoder
//! sections.
//!
//! Pattern observed on `litert-community/gemma-4-E2B-it-litert-lm`
//! section 10 (the decoder graph):
//!
//!   LanguageModel.decode_graph
//!     /LanguageModel.transformer_stack
//!     /transformer.transformer
//!     /layer_{N}
//!     /{layer_{N}.pre_qkv | layer_{N}.post_qkv | layer_{N}.update_cache | attn.dot_product_attention}
//!     /<role-specific subtree>
//!
//! This module groups tensors by `layer_{N}` and classifies each weight
//! by walking the path components. Used by the `scan-layers` CLI to
//! verify the mapping is consistent across all layers before building a
//! proper `Model` struct (Phase 19c proper).

const std = @import("std");
const tflite = @import("tflite.zig");

pub const Role = enum {
    /// MLP gate projection (SwiGLU): `gating_einsum1` shape ~[ffn, hidden].
    mlp_gate,
    /// MLP up projection (SwiGLU): `gating_einsum2`.
    mlp_up,
    /// MLP down projection: `linear` shape ~[hidden, ffn].
    mlp_down,
    /// Attention output projection (after attention reduce):
    /// `attn_vec_einsum` shape ~[hidden, n_q_heads * head_dim].
    attn_o,
    /// Q projection: `q_einsum` shape ~[n_q_heads * head_dim, hidden].
    q_proj,
    /// K projection: `k_einsum` shape ~[n_kv_heads * head_dim, hidden].
    k_proj,
    /// V projection: `v_einsum` shape ~[n_kv_heads * head_dim, hidden].
    v_proj,
    /// Per-layer embedding gate (Gemma's PLE mechanism).
    ple_gate,
    /// Per-layer embedding projection.
    ple_proj,
    /// Precomputed RoPE inv-freq table (FLOAT32 shape ~[1,1,head_dim]).
    rope_inv_freq,
    /// Any tensor we couldn't classify.
    unknown,

    pub fn name(self: Role) []const u8 {
        return switch (self) {
            .mlp_gate => "mlp.gate",
            .mlp_up => "mlp.up",
            .mlp_down => "mlp.down",
            .attn_o => "attn.o",
            .q_proj => "q",
            .k_proj => "k",
            .v_proj => "v",
            .ple_gate => "ple.gate",
            .ple_proj => "ple.proj",
            .rope_inv_freq => "rope.inv_freq",
            .unknown => "?",
        };
    }
};

/// Classify a single tensor by its full name. Returns `unknown` when no
/// pattern matches. Patterns are matched as substring contains — order
/// matters; more specific patterns first.
///
/// Gemma 4 E2B note: RMSNorm scale weights are stored with `data=0`
/// (computed inline / shared globally), so `scan-layers` doesn't see
/// them as per-layer weights. Re-add a `norm` role only if a future
/// model has per-layer norm scales with non-empty data.
pub fn classify(name: []const u8) Role {
    const C = struct {
        fn has(s: []const u8, needle: []const u8) bool {
            return std.mem.indexOf(u8, s, needle) != null;
        }
    };
    // Most specific first.
    if (C.has(name, "per_layer_embedding_gate")) return .ple_gate;
    if (C.has(name, "per_layer_embedding_projection")) return .ple_proj;
    if (C.has(name, "ple_gate_activation")) return .ple_gate;
    if (C.has(name, "gating_einsum1")) return .mlp_gate;
    if (C.has(name, "gating_einsum2")) return .mlp_up;
    if (C.has(name, "mlp/linear")) return .mlp_down;
    if (C.has(name, "q_einsum")) return .q_proj;
    if (C.has(name, "k_einsum")) return .k_proj;
    if (C.has(name, "v_einsum")) return .v_proj;
    if (C.has(name, "attn_vec_einsum") or C.has(name, "post_qkv/attn")) return .attn_o;
    if (C.has(name, "maybe_rope")) return .rope_inv_freq;
    return .unknown;
}

/// Per-layer aggregate: which weight tensors landed for this layer,
/// keyed by role. Empty slots = absent / not yet seen.
pub const LayerBucket = struct {
    layer_idx: u32,
    mlp_gate: ?*const tflite.Tensor = null,
    mlp_up: ?*const tflite.Tensor = null,
    mlp_down: ?*const tflite.Tensor = null,
    attn_o: ?*const tflite.Tensor = null,
    q_proj: ?*const tflite.Tensor = null,
    k_proj: ?*const tflite.Tensor = null,
    v_proj: ?*const tflite.Tensor = null,
    ple_gate: ?*const tflite.Tensor = null,
    ple_proj: ?*const tflite.Tensor = null,
    rope_inv_freq: ?*const tflite.Tensor = null,
    unknown_count: u32 = 0,
};

/// Extract the `layer_{N}` index from a tensor name. Returns null if
/// the name doesn't contain a layer marker.
pub fn parseLayerIndex(name: []const u8) ?u32 {
    const prefix = "layer_";
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, name, pos, prefix)) |found| {
        const start = found + prefix.len;
        if (start >= name.len) break;
        // Parse digits until non-digit.
        var end = start;
        while (end < name.len and std.ascii.isDigit(name[end])) end += 1;
        if (end == start) {
            pos = found + 1;
            continue;
        }
        // Reject if followed by an alpha (so "layer_norm" won't parse
        // as layer 0). Allow non-digit non-alpha (e.g. `/`, `.`).
        if (end < name.len and (std.ascii.isAlphabetic(name[end]) or name[end] == '_')) {
            pos = found + 1;
            continue;
        }
        return std.fmt.parseInt(u32, name[start..end], 10) catch {
            pos = found + 1;
            continue;
        };
    }
    return null;
}

/// Build per-layer buckets from a TFLite model's tensors. Skips
/// tensors with empty data (activations) and tensors with no layer
/// marker. Returned slice is allocator-owned.
pub fn scanLayers(allocator: std.mem.Allocator, tensors: []const tflite.Tensor) ![]LayerBucket {
    // First pass: max layer index.
    var max_layer: i32 = -1;
    for (tensors) |t| {
        if (t.data.len == 0) continue;
        const li = parseLayerIndex(t.name) orelse continue;
        if (@as(i32, @intCast(li)) > max_layer) max_layer = @intCast(li);
    }
    if (max_layer < 0) return allocator.alloc(LayerBucket, 0);

    const n_layers: usize = @intCast(max_layer + 1);
    const buckets = try allocator.alloc(LayerBucket, n_layers);
    errdefer allocator.free(buckets);
    for (buckets, 0..) |*b, i| b.* = .{ .layer_idx = @intCast(i) };

    // Second pass: classify + assign. Prefer larger tensors when role
    // already populated (so a real weight wins over a small constant
    // accidentally matching the pattern).
    for (tensors) |*t| {
        if (t.data.len == 0) continue;
        const li = parseLayerIndex(t.name) orelse continue;
        if (li >= n_layers) continue;
        const role = classify(t.name);
        const bucket = &buckets[li];
        switch (role) {
            .mlp_gate => preferLarger(&bucket.mlp_gate, t),
            .mlp_up => preferLarger(&bucket.mlp_up, t),
            .mlp_down => preferLarger(&bucket.mlp_down, t),
            .attn_o => preferLarger(&bucket.attn_o, t),
            .q_proj => preferLarger(&bucket.q_proj, t),
            .k_proj => preferLarger(&bucket.k_proj, t),
            .v_proj => preferLarger(&bucket.v_proj, t),
            .ple_gate => preferLarger(&bucket.ple_gate, t),
            .ple_proj => preferLarger(&bucket.ple_proj, t),
            .rope_inv_freq => preferLarger(&bucket.rope_inv_freq, t),
            .unknown => bucket.unknown_count += 1,
        }
    }
    return buckets;
}

fn preferLarger(slot: *?*const tflite.Tensor, candidate: *const tflite.Tensor) void {
    if (slot.*) |existing| {
        if (candidate.data.len > existing.data.len) slot.* = candidate;
    } else {
        slot.* = candidate;
    }
}

// -- tests ----------------------------------------------------------------

test "parseLayerIndex picks first numeric layer marker" {
    try std.testing.expectEqual(@as(?u32, 24), parseLayerIndex("transformer/layer_24/mlp/linear"));
    try std.testing.expectEqual(@as(?u32, 0), parseLayerIndex("layer_0/qkv"));
    try std.testing.expectEqual(@as(?u32, null), parseLayerIndex("layer_norm/scale"));
    try std.testing.expectEqual(@as(?u32, null), parseLayerIndex("transformer/embed"));
}

test "classify picks correct roles for canonical Gemma 4 paths" {
    try std.testing.expectEqual(Role.mlp_gate, classify("layer_3/post_qkv/mlp/gating_einsum1/dot_general"));
    try std.testing.expectEqual(Role.mlp_up, classify("layer_3/post_qkv/mlp/gating_einsum2/dot_general"));
    try std.testing.expectEqual(Role.mlp_down, classify("layer_3/post_qkv/mlp/linear/dot_general"));
    try std.testing.expectEqual(Role.q_proj, classify("layer_3/pre_qkv/attn.pre_qkv/attn._pre_attention_fn/q_einsum/dot_general"));
    try std.testing.expectEqual(Role.k_proj, classify("layer_3/pre_qkv/attn.pre_qkv/attn._pre_attention_fn/k_einsum/dot_general"));
    try std.testing.expectEqual(Role.v_proj, classify("layer_3/pre_qkv/attn.pre_qkv/attn._pre_attention_fn/v_einsum/dot_general"));
    try std.testing.expectEqual(Role.attn_o, classify("layer_3/post_qkv/attn.post_qkv/attn_vec_einsum/dot_general"));
    try std.testing.expectEqual(Role.ple_gate, classify("layer_3/post_qkv/per_layer_embedding_gate"));
    try std.testing.expectEqual(Role.ple_proj, classify("layer_3/post_qkv/per_layer_embedding_projection"));
    try std.testing.expectEqual(Role.rope_inv_freq, classify("layer_3/pre_qkv/attn.pre_qkv/attn._pre_attention_fn/maybe_rope/div"));
}
