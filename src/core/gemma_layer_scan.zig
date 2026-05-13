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
    /// Attention output projection (after attention reduce).
    attn_o,
    /// QKV pre-projection (combined Q+K+V matmul; pre_qkv block).
    qkv_proj,
    /// Per-layer embedding gate (Gemma's PLE mechanism).
    ple_gate,
    /// Per-layer embedding projection.
    ple_proj,
    /// RMS / layer norm scale (any of: pre-attn, post-attn, pre-ffw, post-ffw, post-per-layer-input).
    norm,
    /// Any tensor we couldn't classify.
    unknown,

    pub fn name(self: Role) []const u8 {
        return switch (self) {
            .mlp_gate => "mlp.gate",
            .mlp_up => "mlp.up",
            .mlp_down => "mlp.down",
            .attn_o => "attn.o",
            .qkv_proj => "qkv",
            .ple_gate => "ple.gate",
            .ple_proj => "ple.proj",
            .norm => "norm",
            .unknown => "?",
        };
    }
};

/// Classify a single tensor by its full name. Returns `unknown` when no
/// pattern matches. Patterns are matched as substring contains — order
/// matters; more specific patterns first.
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
    if (C.has(name, "pre_qkv/attn") or C.has(name, "qkv_fn") or C.has(name, "qkv_proj")) return .qkv_proj;
    if (C.has(name, "post_qkv/attn")) return .attn_o;
    if (C.has(name, "_norm/scale") or C.has(name, "_norm/mul") or
        C.has(name, "attention_norm") or C.has(name, "ffw_norm") or
        C.has(name, "per_layer_input_norm")) return .norm;
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
    qkv_proj: ?*const tflite.Tensor = null,
    ple_gate: ?*const tflite.Tensor = null,
    ple_proj: ?*const tflite.Tensor = null,
    norm_count: u32 = 0,
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
            .qkv_proj => preferLarger(&bucket.qkv_proj, t),
            .ple_gate => preferLarger(&bucket.ple_gate, t),
            .ple_proj => preferLarger(&bucket.ple_proj, t),
            .norm => bucket.norm_count += 1,
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
    try std.testing.expectEqual(Role.ple_gate, classify("layer_3/post_qkv/per_layer_embedding_gate"));
    try std.testing.expectEqual(Role.ple_proj, classify("layer_3/post_qkv/per_layer_embedding_projection"));
    try std.testing.expectEqual(Role.norm, classify("layer_3/post_qkv/post_attention_norm/mul"));
}
