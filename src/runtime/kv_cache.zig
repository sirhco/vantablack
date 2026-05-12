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

const metal_backend_mod = @import("metal_backend.zig");

pub const KvCache = struct {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    /// [n_layers * max_seq * n_kv_heads * head_dim] f32, layer-major.
    k: []f32,
    v: []f32,
    pos: usize,
    /// True when k/v are gpa-allocated; false when they alias GPU shared
    /// MTLBuffers owned by `MetalBackend`.
    owns_buffers: bool,

    pub fn init(
        allocator: Allocator,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        metal: ?*metal_backend_mod.MetalBackend,
    ) !KvCache {
        const per_layer = max_seq * n_kv_heads * head_dim;
        const total = n_layers * per_layer;

        if (metal) |mb| {
            std.debug.assert(mb.kv_k.cap >= total);
            std.debug.assert(mb.kv_v.cap >= total);
            return .{
                .n_layers = n_layers,
                .n_kv_heads = n_kv_heads,
                .head_dim = head_dim,
                .max_seq = max_seq,
                .k = mb.kv_k.ptr[0..total],
                .v = mb.kv_v.ptr[0..total],
                .pos = 0,
                .owns_buffers = false,
            };
        }

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
            .owns_buffers = true,
        };
    }

    pub fn deinit(self: *KvCache, allocator: Allocator) void {
        if (self.owns_buffers) {
            allocator.free(self.k);
            allocator.free(self.v);
        }
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

    /// Reduce `max_seq` to `new_max_seq` and release the unused tail.
    /// The first `pos * rowSize()` elements of every layer are preserved;
    /// anything past `pos` is discarded (it was unwritten or stale).
    ///
    /// Heap-backed: try in-place `Allocator.resize`; fall back to alloc+
    /// memcpy+free if the allocator can't shrink in place.
    ///
    /// Metal-backed: the underlying MTLBuffer is not shrunk (that would
    /// need a new device allocation). The slice view is narrowed and
    /// `max_seq` is reduced so future reads cap at the new bound. The
    /// unused tail of the MTLBuffer continues to occupy GPU memory but
    /// the live cache footprint at the API surface is now smaller.
    pub fn shrinkToFit(self: *KvCache, allocator: Allocator, new_max_seq: usize) !void {
        if (new_max_seq >= self.max_seq) return; // no-op
        if (new_max_seq < self.pos) return error.NewMaxSeqBelowPos;
        const old_per_layer = self.perLayer();
        const new_per_layer = new_max_seq * self.rowSize();
        const live_per_layer = self.pos * self.rowSize();

        // Compact: move each layer's live prefix to its new (smaller)
        // offset. Layer 0 stays put — its data is already at offset 0.
        // Destination always precedes source so a forward copy is safe
        // even for overlapping ranges.
        var layer: usize = 1;
        while (layer < self.n_layers) : (layer += 1) {
            const old_off = layer * old_per_layer;
            const new_off = layer * new_per_layer;
            if (live_per_layer > 0) {
                std.mem.copyForwards(f32, self.k[new_off..][0..live_per_layer], self.k[old_off..][0..live_per_layer]);
                std.mem.copyForwards(f32, self.v[new_off..][0..live_per_layer], self.v[old_off..][0..live_per_layer]);
            }
        }

        const new_total = self.n_layers * new_per_layer;

        if (self.owns_buffers) {
            // Try shrink-in-place first.
            if (allocator.resize(self.k, new_total)) {
                self.k = self.k[0..new_total];
            } else {
                const new_k = try allocator.alloc(f32, new_total);
                @memcpy(new_k, self.k[0..new_total]);
                allocator.free(self.k);
                self.k = new_k;
            }
            if (allocator.resize(self.v, new_total)) {
                self.v = self.v[0..new_total];
            } else {
                const new_v = try allocator.alloc(f32, new_total);
                @memcpy(new_v, self.v[0..new_total]);
                allocator.free(self.v);
                self.v = new_v;
            }
        } else {
            // Metal-backed: narrow the slice over the same MTLBuffer.
            self.k = self.k[0..new_total];
            self.v = self.v[0..new_total];
        }

        self.max_seq = new_max_seq;
    }
};

test "KvCache slot/advance round-trip" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 4, 8, 16, null);
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

test "KvCache.shrinkToFit preserves live data + caps further writes" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 2, 4, 8, 64, null);
    defer cache.deinit(gpa);

    // Advance to pos=10 with marker values so we can verify preservation.
    for (0..10) |t| {
        for (cache.keySlot(0), 0..) |*v, i| v.* = @as(f32, @floatFromInt(t * 1000 + i));
        for (cache.valueSlot(0), 0..) |*v, i| v.* = @as(f32, @floatFromInt(t * 1000 + i)) + 0.5;
        for (cache.keySlot(1), 0..) |*v, i| v.* = @as(f32, @floatFromInt(t * 1000 + i)) + 10000.0;
        for (cache.valueSlot(1), 0..) |*v, i| v.* = @as(f32, @floatFromInt(t * 1000 + i)) + 10000.5;
        cache.advance();
    }
    try std.testing.expectEqual(@as(usize, 10), cache.pos);

    // Shrink from 64 → 32. Layer-1 data must be compacted to the new offset.
    try cache.shrinkToFit(gpa, 32);
    try std.testing.expectEqual(@as(usize, 32), cache.max_seq);
    try std.testing.expectEqual(@as(usize, 10), cache.pos);

    // Verify live data survives across both layers.
    const l0_keys = cache.keysFor(0, 10);
    try std.testing.expectApproxEqAbs(@as(f32, 0), l0_keys[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9000), l0_keys[9 * 4 * 8], 1e-6);

    const l1_keys = cache.keysFor(1, 10);
    try std.testing.expectApproxEqAbs(@as(f32, 10000), l1_keys[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 19000), l1_keys[9 * 4 * 8], 1e-6);

    // Continue writing in the smaller cache — 22 more tokens fit (pos=10 → 32).
    for (10..32) |t| {
        for (cache.keySlot(0), 0..) |*v, i| v.* = @as(f32, @floatFromInt(t * 1000 + i));
        cache.advance();
    }
    try std.testing.expectEqual(@as(usize, 32), cache.pos);
}

test "KvCache.shrinkToFit refuses to drop populated entries" {
    const gpa = std.testing.allocator;
    var cache = try KvCache.init(gpa, 1, 2, 4, 32, null);
    defer cache.deinit(gpa);
    for (0..20) |_| {
        for (cache.keySlot(0)) |*v| v.* = 0;
        cache.advance();
    }
    // pos=20 — shrinking below 20 must error rather than truncate.
    try std.testing.expectError(error.NewMaxSeqBelowPos, cache.shrinkToFit(gpa, 10));
    // No-op when target >= current.
    try cache.shrinkToFit(gpa, 64);
    try std.testing.expectEqual(@as(usize, 32), cache.max_seq);
}
