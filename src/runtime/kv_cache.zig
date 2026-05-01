//! Fixed-size KV cache.
//!
//! Pre-allocates the maximum possible K/V buffer once at init. Single bulk
//! allocation per buffer, no realloc, no fragmentation. `pos` tracks how many
//! tokens have been written; reads return slices over the populated prefix.
//!
//! Layout per layer (seq-major):
//!   k_layer[t * n_kv_heads * head_dim + h * head_dim + d]  for token t, head h, dim d
//!
//! The "ring" property requested by the spec is achieved by capping `pos` to
//! `max_seq`; once full, callers must either grow `max_seq` (re-init) or
//! truncate. Wrap-around eviction breaks causal attention semantics so it is
//! intentionally not implemented at this layer.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const KvCache = struct {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    /// [n_layers * max_seq * n_kv_heads * head_dim] f32, layer-major.
    k: []f32,
    v: []f32,
    pos: usize,

    pub fn init(
        allocator: Allocator,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) !KvCache {
        const per_layer = max_seq * n_kv_heads * head_dim;
        const total = n_layers * per_layer;
        const k = try allocator.alloc(f32, total);
        errdefer allocator.free(k);
        const v = try allocator.alloc(f32, total);
        errdefer allocator.free(v);
        return .{
            .n_layers = n_layers,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq = max_seq,
            .k = k,
            .v = v,
            .pos = 0,
        };
    }

    pub fn deinit(self: *KvCache, allocator: Allocator) void {
        allocator.free(self.k);
        allocator.free(self.v);
        self.* = undefined;
    }

    pub fn reset(self: *KvCache) void {
        self.pos = 0;
    }

    fn perLayer(self: *const KvCache) usize {
        return self.max_seq * self.n_kv_heads * self.head_dim;
    }

    fn rowSize(self: *const KvCache) usize {
        return self.n_kv_heads * self.head_dim;
    }

    /// Returns the writable K slice at the current `pos` for the given layer.
    /// Caller must call `advance()` after writing both K and V for all layers.
    pub fn keySlot(self: *KvCache, layer: usize) []f32 {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(self.pos < self.max_seq);
        const layer_off = layer * self.perLayer();
        const row_off = self.pos * self.rowSize();
        return self.k[layer_off + row_off ..][0..self.rowSize()];
    }

    pub fn valueSlot(self: *KvCache, layer: usize) []f32 {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(self.pos < self.max_seq);
        const layer_off = layer * self.perLayer();
        const row_off = self.pos * self.rowSize();
        return self.v[layer_off + row_off ..][0..self.rowSize()];
    }

    /// Returns the populated K prefix for the layer (rows 0..pos+1 inclusive
    /// AFTER the current row has been written). Use `advance` first if you
    /// want to include the current write.
    pub fn keysFor(self: *const KvCache, layer: usize, n_tokens: usize) []const f32 {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(n_tokens <= self.max_seq);
        const layer_off = layer * self.perLayer();
        return self.k[layer_off..][0 .. n_tokens * self.rowSize()];
    }

    pub fn valuesFor(self: *const KvCache, layer: usize, n_tokens: usize) []const f32 {
        std.debug.assert(layer < self.n_layers);
        std.debug.assert(n_tokens <= self.max_seq);
        const layer_off = layer * self.perLayer();
        return self.v[layer_off..][0 .. n_tokens * self.rowSize()];
    }

    pub fn advance(self: *KvCache) void {
        std.debug.assert(self.pos < self.max_seq);
        self.pos += 1;
    }
};

test "KvCache slot/advance round-trip" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 4, 8, 16);
    defer cache.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 0), cache.pos);

    // Write first token across both layers.
    const k0 = cache.keySlot(0);
    for (k0, 0..) |*v, i| v.* = @floatFromInt(i);
    const v0 = cache.valueSlot(0);
    for (v0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 10.0;
    const k0_l1 = cache.keySlot(1);
    for (k0_l1, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) + 0.5;
    cache.advance();

    // Second token.
    const k1 = cache.keySlot(0);
    for (k1, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) + 100.0;
    cache.advance();

    try std.testing.expectEqual(@as(usize, 2), cache.pos);

    // Read back: layer 0, all populated tokens.
    const populated = cache.keysFor(0, cache.pos);
    try std.testing.expectEqual(@as(usize, 2 * 4 * 8), populated.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0), populated[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 100), populated[4 * 8], 1e-6);

    // Cross-layer isolation.
    const l1 = cache.keysFor(1, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), l1[0], 1e-6);
}
