//! Per-process Metal backend state.
//!
//! Owns the MTLDevice + persistent state buffers used by the GPU forward
//! path. State scratch (x, xb, q, k_cur, ...) and the LM-head logits live in
//! shared-storage MTLBuffers so the CPU can read intermediate values when
//! attention runs CPU-side, and the GPU can read activations without
//! per-call memcpy.
//!
//! Weights are wrapped no-copy from the mmap'd region; norm weights are
//! addressed via per-call buffer offsets.

const std = @import("std");
const Allocator = std.mem.Allocator;

const metal = @import("../metal/bridge.zig");
const model_mod = @import("model.zig");
const mapper_mod = @import("../core/mapper.zig");

pub const InitError = error{
    MetalUnavailable,
    MetalAllocFailed,
};

pub const ScratchBuf = struct {
    buf: *metal.Buf,
    ptr: [*]f32,
    cap: usize,
};

/// One page-aligned mmap region to register with the GPU as a no-copy
/// MTLBuffer. Multi-shard HF/MLX bundles supply one of these per shard.
pub const ShardInput = struct {
    bytes: []const u8,
};

/// One wrapped MTLBuffer plus the host-pointer extents that resolve to it.
pub const WeightShard = struct {
    buf: *metal.Buf,
    base: [*]const u8,
    len: usize,
};

/// Result of `resolveWeight`: which shard a host pointer belongs to and the
/// byte offset within that shard's wrapped buffer.
pub const WeightLoc = struct {
    buf: *metal.Buf,
    offset: usize,
};

pub const MetalBackend = struct {
    dev: metal.Device,

    // One wrapped MTLBuffer per source mmap region. GGUF puts one shard
    // here; HF / MLX bundles can have many.
    weight_shards: []WeightShard,

    // Per-token persistent state. All shared-storage MTLBuffers; both CPU
    // and GPU read/write them (Apple Silicon unified memory means same
    // physical pages — no DMA, no copy).
    x: ScratchBuf,
    xb: ScratchBuf,
    xb2: ScratchBuf,
    q: ScratchBuf,
    k_cur: ScratchBuf,
    v_cur: ScratchBuf,
    attn_out: ScratchBuf,
    gate: ScratchBuf,
    up: ScratchBuf,
    ffn_out: ScratchBuf,
    logits: ScratchBuf,

    // KV cache slab and attention scores scratch — owned here so the GPU
    // attention kernels can read/write them directly. CPU keeps a host-slice
    // alias via `KvCache` / `State.attn_scores`.
    kv_k: ScratchBuf,         // n_layers * max_seq * n_kv_heads * head_dim
    kv_v: ScratchBuf,         // n_layers * max_seq * n_kv_heads * head_dim
    attn_scores: ScratchBuf,  // n_heads * max_seq

    /// GGUF call-site convenience: wraps the mapper's single mmap region as
    /// a one-shard backend.
    pub fn init(
        allocator: Allocator,
        mapper: *const mapper_mod.ModelMapper,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        const shards = [_]ShardInput{.{ .bytes = mapper.mapped }};
        return initShards(allocator, &shards, cfg);
    }

    /// Multi-shard entry point. Each `ShardInput.bytes` must be a
    /// page-aligned mmap region; lengths are rounded up to a page multiple
    /// before being handed to Metal (mmap zero-fills the trailing partial
    /// page, so the rounded length stays inside the kernel mapping).
    pub fn initShards(
        allocator: Allocator,
        shards: []const ShardInput,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        if (!metal.metal_enabled) return error.MetalUnavailable;
        if (shards.len == 0) return InitError.MetalUnavailable;

        var dev = metal.Device.init() catch return error.MetalUnavailable;
        errdefer dev.deinit();

        const weight_shards = allocator.alloc(WeightShard, shards.len) catch
            return error.MetalAllocFailed;
        errdefer allocator.free(weight_shards);

        // Number of shards successfully wrapped; on error path we release
        // exactly that many.
        var wrapped: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < wrapped) : (i += 1) dev.release(weight_shards[i].buf);
        }

        const page = std.heap.pageSize();
        for (shards, 0..) |sh, i| {
            // Round the wrap length up to a page multiple. mmap implicitly
            // zero-fills the trailing partial page, so the rounded length is
            // safe to hand to Metal. We can't index `sh.bytes[0..len_aligned]`
            // because the caller-supplied slice has `len == file size` (not
            // page-rounded) — build the wider slice from the raw pointer.
            const len_aligned = std.mem.alignForward(usize, sh.bytes.len, page);
            const wide: []const u8 = sh.bytes.ptr[0..len_aligned];
            const buf = dev.wrap(wide) catch
                return error.MetalAllocFailed;
            weight_shards[i] = .{
                .buf = buf,
                .base = sh.bytes.ptr,
                .len = len_aligned,
            };
            wrapped = i + 1;
        }

        const kv_dim = cfg.n_kv_heads * cfg.head_dim;
        const x = try allocScratch(dev, cfg.dim);
        errdefer dev.release(x.buf);
        const xb = try allocScratch(dev, cfg.dim);
        errdefer dev.release(xb.buf);
        const xb2 = try allocScratch(dev, cfg.dim);
        errdefer dev.release(xb2.buf);
        const q = try allocScratch(dev, cfg.n_heads * cfg.head_dim);
        errdefer dev.release(q.buf);
        const k_cur = try allocScratch(dev, kv_dim);
        errdefer dev.release(k_cur.buf);
        const v_cur = try allocScratch(dev, kv_dim);
        errdefer dev.release(v_cur.buf);
        const attn_out = try allocScratch(dev, cfg.dim);
        errdefer dev.release(attn_out.buf);
        const gate = try allocScratch(dev, cfg.ffn_dim);
        errdefer dev.release(gate.buf);
        const up = try allocScratch(dev, cfg.ffn_dim);
        errdefer dev.release(up.buf);
        const ffn_out = try allocScratch(dev, cfg.dim);
        errdefer dev.release(ffn_out.buf);
        const logits = try allocScratch(dev, cfg.vocab_size);
        errdefer dev.release(logits.buf);

        const kv_total = cfg.n_layers * cfg.max_seq * kv_dim;
        const kv_k = try allocScratch(dev, kv_total);
        errdefer dev.release(kv_k.buf);
        const kv_v = try allocScratch(dev, kv_total);
        errdefer dev.release(kv_v.buf);
        const attn_scores = try allocScratch(dev, cfg.n_heads * cfg.max_seq);
        errdefer dev.release(attn_scores.buf);

        return .{
            .dev = dev,
            .weight_shards = weight_shards,
            .x = x,
            .xb = xb,
            .xb2 = xb2,
            .q = q,
            .k_cur = k_cur,
            .v_cur = v_cur,
            .attn_out = attn_out,
            .gate = gate,
            .up = up,
            .ffn_out = ffn_out,
            .logits = logits,
            .kv_k = kv_k,
            .kv_v = kv_v,
            .attn_scores = attn_scores,
        };
    }

    pub fn deinit(self: *MetalBackend, allocator: Allocator) void {
        self.dev.release(self.attn_scores.buf);
        self.dev.release(self.kv_v.buf);
        self.dev.release(self.kv_k.buf);
        self.dev.release(self.logits.buf);
        self.dev.release(self.ffn_out.buf);
        self.dev.release(self.up.buf);
        self.dev.release(self.gate.buf);
        self.dev.release(self.attn_out.buf);
        self.dev.release(self.v_cur.buf);
        self.dev.release(self.k_cur.buf);
        self.dev.release(self.q.buf);
        self.dev.release(self.xb2.buf);
        self.dev.release(self.xb.buf);
        self.dev.release(self.x.buf);
        for (self.weight_shards) |sh| self.dev.release(sh.buf);
        allocator.free(self.weight_shards);
        self.dev.deinit();
        self.* = undefined;
    }

    /// Linear-scan the registered shards and return the one containing
    /// `host_ptr`, or null if it lies outside every shard.
    pub fn resolveWeight(self: *const MetalBackend, host_ptr: [*]const u8) ?WeightLoc {
        const p = @intFromPtr(host_ptr);
        for (self.weight_shards) |sh| {
            const base = @intFromPtr(sh.base);
            if (p >= base and p < base + sh.len) {
                return .{ .buf = sh.buf, .offset = p - base };
            }
        }
        return null;
    }

    /// Legacy single-shard accessor for the GGUF forward path. Asserts there
    /// is at least one shard registered.
    pub fn weightsBuf(self: *const MetalBackend) *metal.Buf {
        std.debug.assert(self.weight_shards.len >= 1);
        return self.weight_shards[0].buf;
    }

    /// Byte offset of `weight_bytes.ptr` inside the FIRST wrapped weight
    /// buffer. GGUF-only — multi-shard callers must use `resolveWeight`.
    pub fn weightOffset(self: *const MetalBackend, weight_bytes: []const u8) usize {
        std.debug.assert(self.weight_shards.len >= 1);
        const sh = self.weight_shards[0];
        const w_ptr_int = @intFromPtr(weight_bytes.ptr);
        const base_int = @intFromPtr(sh.base);
        std.debug.assert(w_ptr_int >= base_int);
        const off = w_ptr_int - base_int;
        std.debug.assert(off + weight_bytes.len <= sh.len);
        return off;
    }

    /// Stand-alone Q8_0 matmul entry point: still used for the LM-head pass
    /// where the input lives in a scratch buffer rather than the per-token
    /// persistent state.
    pub fn matmulQ8_0(
        self: *MetalBackend,
        out_buf: *ScratchBuf,
        weight_bytes: []const u8,
        in_buf: *const ScratchBuf,
        m: usize,
        k: usize,
    ) !void {
        std.debug.assert(m <= out_buf.cap);
        std.debug.assert(k <= in_buf.cap);
        const off = self.weightOffset(weight_bytes);
        try self.dev.matmulQ8_0(out_buf.buf, self.weightsBuf(), off, in_buf.buf, m, k);
    }

    /// Returns the ScratchBuf whose shared-storage pointer matches `p`, or
    /// null if `p` was allocated outside the backend (e.g. in CPU-only mode).
    /// Lets `forward.zig` map a plain `[]f32` view back to its GPU buffer
    /// without threading an explicit handle through every call site.
    pub fn scratchForPtr(self: *MetalBackend, p: [*]const f32) ?*ScratchBuf {
        const fields = .{ "x", "xb", "xb2", "q", "k_cur", "v_cur", "attn_out", "gate", "up", "ffn_out", "logits" };
        inline for (fields) |name| {
            if (@field(self, name).ptr == p) return &@field(self, name);
        }
        return null;
    }
};

fn allocScratch(dev: metal.Device, n: usize) !ScratchBuf {
    var raw: ?*anyopaque = null;
    const buf = dev.alloc(n * @sizeOf(f32), &raw) catch return error.MetalAllocFailed;
    return .{
        .buf = buf,
        .ptr = @ptrCast(@alignCast(raw.?)),
        .cap = n,
    };
}

fn fakeCfg() model_mod.LlamaConfig {
    return .{
        .dim = 32,
        .n_layers = 1,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 8,
        .ffn_dim = 64,
        .vocab_size = 128,
        .max_seq = 16,
    };
}

test "MetalBackend.resolveWeight: two shards, pointer routes to correct buffer" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Two page-aligned dummy shards.
    const page = std.heap.pageSize();
    const shard0 = try std.posix.mmap(null, page, .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard0);
    const shard1 = try std.posix.mmap(null, page, .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard1);
    shard0[0] = 0xAA;
    shard1[0] = 0xBB;

    const shards = [_]ShardInput{
        .{ .bytes = shard0 },
        .{ .bytes = shard1 },
    };

    var be = try MetalBackend.initShards(allocator, &shards, fakeCfg());
    defer be.deinit(allocator);

    const loc0 = be.resolveWeight(shard0.ptr).?;
    const loc1 = be.resolveWeight(shard1.ptr + 64).?;
    try std.testing.expectEqual(@as(usize, 0), loc0.offset);
    try std.testing.expectEqual(@as(usize, 64), loc1.offset);
    try std.testing.expect(loc0.buf != loc1.buf);

    try std.testing.expect(be.resolveWeight(@as([*]const u8, @ptrFromInt(0xdead0000))) == null);
}

test "MetalBackend.resolveWeight: shard with non-page-aligned length covers trailing partial page" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const page = std.heap.pageSize();

    // mmap two pages, but advertise a length of (page + 7) — i.e. a partial
    // trailing page, exactly the layout real safetensors / GGUF files have.
    const region = try std.posix.mmap(null, 2 * page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(region);

    const shard_len = page + 7;
    const shards = [_]ShardInput{ .{ .bytes = region[0..shard_len] } };

    var be = try MetalBackend.initShards(allocator, &shards, fakeCfg());
    defer be.deinit(allocator);

    // Pointer 3 bytes into the trailing partial page — alignBackward would
    // have truncated this off and resolveWeight would return null.
    const probe = region.ptr + page + 3;
    const loc = be.resolveWeight(probe) orelse return error.TestFailed;
    try std.testing.expectEqual(@as(usize, page + 3), loc.offset);
}
