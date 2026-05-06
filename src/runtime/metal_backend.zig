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
const build_options = @import("build_options");

const metal = @import("../metal/bridge.zig");
const model_mod = @import("model.zig");
const mapper_mod = @import("../core/mapper.zig");
const hf_loader_mod = @import("../core/hf_loader.zig");
const safetensors = @import("../core/safetensors.zig");
const mlx_kernels = @import("../kernels/mlx.zig");

pub const InitError = error{
    MetalUnavailable,
    MetalAllocFailed,
};

pub const MatmulError = error{
    UnsupportedBits,
    WeightNotMapped,
    MetalDispatchFailed,
};

pub const ScratchBuf = struct {
    buf: *metal.Buf,
    ptr: [*]f32,
    cap: usize,
};

/// Per-layer fused-weight pack. `buf` is one shared-storage MTLBuffer
/// containing every layer's fused tensor concatenated; `byte_offsets[l]` is
/// where layer `l`'s tensor starts inside that buffer.
pub const FusedPack = struct {
    buf: *metal.Buf,
    byte_offsets: []usize, // len == n_layers
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

    // Optional fused-weight packs + matmul output scratch. Populated by
    // `attachFusedWeights` when `-Dweight_fusion=true`; null otherwise.
    fused_qkv: ?FusedPack = null,
    fused_gate_up: ?FusedPack = null,
    qkv_scratch: ?ScratchBuf = null,        // (n_heads + 2*n_kv_heads) * head_dim
    gate_up_scratch: ?ScratchBuf = null,    // 2 * ffn_dim
    fused_alloc: ?Allocator = null,         // remembers which allocator owns offset arrays

    pub fn init(
        allocator: Allocator,
        mapper: *const mapper_mod.ModelMapper,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        const shards = [_]ShardInput{.{ .bytes = mapper.mapped }};
        return initShards(allocator, &shards, cfg);
    }

    /// HF / MLX call-site convenience: builds a ShardInput per safetensors
    /// shard in `bundle` and delegates to `initShards`. Used by the multi-
    /// shard runner to attach Metal acceleration without each call site
    /// having to know the bundle's internal shape.
    pub fn initFromHf(
        allocator: Allocator,
        bundle: *const hf_loader_mod.HfBundle,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        const ins = allocator.alloc(ShardInput, bundle.shards.len) catch
            return error.MetalAllocFailed;
        defer allocator.free(ins);
        for (bundle.shards, 0..) |sh, i| ins[i] = .{ .bytes = sh.mapped };
        return initShards(allocator, ins, cfg);
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

    pub fn deinit(self: *MetalBackend) void {
        if (self.fused_qkv) |p| {
            self.dev.release(p.buf);
            if (self.fused_alloc) |a| a.free(p.byte_offsets);
        }
        if (self.fused_gate_up) |p| {
            self.dev.release(p.buf);
            if (self.fused_alloc) |a| a.free(p.byte_offsets);
        }
        if (self.qkv_scratch) |s| self.dev.release(s.buf);
        if (self.gate_up_scratch) |s| self.dev.release(s.buf);
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

    /// Move heap-allocated fused weight bytes from `model` into shared-storage
    /// MTLBuffers and free the heap copies. After this call,
    /// `model.layers[l].attn_qkv_fused.?.bytes` aliases GPU memory directly,
    /// and `model.owned_buffers` is empty.
    pub fn attachFusedWeights(
        self: *MetalBackend,
        allocator: Allocator,
        model: *model_mod.Model,
    ) !void {
        if (!build_options.weight_fusion) return;
        const cfg = model.config;

        // Pack a single MTLBuffer containing every layer's fused tensor
        // concatenated. Returns null if no layer has the fused field
        // populated.
        const PackKind = enum { qkv, gate_up };
        const Helpers = struct {
            fn fusedTensor(lw: *const model_mod.LayerWeights, kind: PackKind) ?model_mod.TypedTensor {
                return switch (kind) {
                    .qkv => lw.attn_qkv_fused,
                    .gate_up => lw.ffn_gate_up_fused,
                };
            }
            fn setFusedTensor(lw: *model_mod.LayerWeights, kind: PackKind, t: model_mod.TypedTensor) void {
                switch (kind) {
                    .qkv => lw.attn_qkv_fused = t,
                    .gate_up => lw.ffn_gate_up_fused = t,
                }
            }
            fn buildPack(
                dev: metal.Device,
                alloc: Allocator,
                layers: []model_mod.LayerWeights,
                kind: PackKind,
            ) !?FusedPack {
                var total: usize = 0;
                for (layers) |lw| {
                    if (fusedTensor(&lw, kind)) |t| total += t.bytes.len;
                }
                if (total == 0) return null;

                var raw: ?*anyopaque = null;
                const buf = dev.alloc(total, &raw) catch return error.MetalAllocFailed;
                errdefer dev.release(buf);
                const dst: [*]u8 = @ptrCast(@alignCast(raw.?));

                const offsets = try alloc.alloc(usize, layers.len);
                errdefer alloc.free(offsets);

                var off: usize = 0;
                for (layers, 0..) |*lw, l| {
                    if (fusedTensor(lw, kind)) |t| {
                        offsets[l] = off;
                        @memcpy(dst[off..][0..t.bytes.len], t.bytes);
                        // Re-point the model's view at GPU shared memory and
                        // drop the heap copy below.
                        var new_t = t;
                        new_t.bytes = dst[off..][0..t.bytes.len];
                        setFusedTensor(lw, kind, new_t);
                        off += t.bytes.len;
                    } else {
                        offsets[l] = 0;
                    }
                }
                return .{ .buf = buf, .byte_offsets = offsets };
            }
        };

        self.fused_qkv = try Helpers.buildPack(self.dev, allocator, model.layers, .qkv);
        errdefer if (self.fused_qkv) |p| {
            self.dev.release(p.buf);
            allocator.free(p.byte_offsets);
        };
        self.fused_gate_up = try Helpers.buildPack(self.dev, allocator, model.layers, .gate_up);
        errdefer if (self.fused_gate_up) |p| {
            self.dev.release(p.buf);
            allocator.free(p.byte_offsets);
        };
        self.fused_alloc = allocator;

        if (self.fused_qkv != null) {
            const qkv_dim = cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim;
            self.qkv_scratch = try allocScratch(self.dev, qkv_dim);
        }
        if (self.fused_gate_up != null) {
            self.gate_up_scratch = try allocScratch(self.dev, 2 * cfg.ffn_dim);
        }

        // The MTLBuffer copies are the only copies we need now — drop the
        // heap originals so we don't keep paying their memory cost.
        for (model.owned_buffers.items) |b| allocator.free(b);
        model.owned_buffers.clearRetainingCapacity();
    }

    /// Byte offset of `weight_bytes.ptr` inside the wrapped weight buffer.
    /// Caller must ensure the slice is within the mmap region.
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

    /// MLX-Q4 matmul host helper: resolves raw weight/scales/biases pointers
    /// (which point into mmap'd shard regions) to (buf, offset) tuples via
    /// `resolveWeight`, then dispatches the 4-buffer Metal kernel via
    /// `Device.matmulMlxQ4`. Plan C only handles 4-bit on GPU; other bit
    /// widths are rejected with `error.UnsupportedBits` so the caller can
    /// fall back to CPU.
    pub fn matmulMlxQ4(
        self: *MetalBackend,
        out: *metal.Buf,
        weight_ptr: [*]const u8,
        scales_ptr: [*]const u8,
        biases_ptr: [*]const u8,
        acts: *metal.Buf,
        m: usize,
        k: usize,
        bits: u32,
        group_size: u32,
        scale_dtype: mlx_kernels.ScaleDtype,
    ) MatmulError!void {
        if (bits != 4) return error.UnsupportedBits;

        const w = self.resolveWeight(weight_ptr) orelse return error.WeightNotMapped;
        const s = self.resolveWeight(scales_ptr) orelse return error.WeightNotMapped;
        const b = self.resolveWeight(biases_ptr) orelse return error.WeightNotMapped;

        try self.dev.matmulMlxQ4(
            out,
            w.buf, w.offset,
            s.buf, s.offset,
            b.buf, b.offset,
            acts, 0,
            m, k,
            group_size,
            scale_dtype == .bf16,
        );
    }

    pub fn matmulQ4_K(self: *MetalBackend, out_buf: *ScratchBuf, weight_bytes: []const u8, in_buf: *const ScratchBuf, m: usize, k: usize) !void {
        std.debug.assert(m <= out_buf.cap);
        std.debug.assert(k <= in_buf.cap);
        try self.dev.matmulQ4_K(out_buf.buf, self.weights_buf, self.weightOffset(weight_bytes), in_buf.buf, m, k);
    }
    pub fn matmulQ5_K(self: *MetalBackend, out_buf: *ScratchBuf, weight_bytes: []const u8, in_buf: *const ScratchBuf, m: usize, k: usize) !void {
        std.debug.assert(m <= out_buf.cap);
        std.debug.assert(k <= in_buf.cap);
        try self.dev.matmulQ5_K(out_buf.buf, self.weights_buf, self.weightOffset(weight_bytes), in_buf.buf, m, k);
    }
    pub fn matmulQ6_K(self: *MetalBackend, out_buf: *ScratchBuf, weight_bytes: []const u8, in_buf: *const ScratchBuf, m: usize, k: usize) !void {
        std.debug.assert(m <= out_buf.cap);
        std.debug.assert(k <= in_buf.cap);
        try self.dev.matmulQ6_K(out_buf.buf, self.weights_buf, self.weightOffset(weight_bytes), in_buf.buf, m, k);
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

test "MetalBackend.initFromHf: wraps every HF shard's mmap" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const page = std.heap.pageSize();

    // Build a minimal HfBundle stub with two page-aligned anonymous mmap
    // regions standing in for safetensors shards. We don't need real
    // catalogs because initFromHf only reads `shards[i].mapped`.
    const region0 = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(region0);
    const region1 = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(region1);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const empty_cat: safetensors.Catalog = .{ .data_segment_start = 0, .descs = &.{} };
    const Shard = hf_loader_mod.Shard;
    var shards = try arena.allocator().alloc(Shard, 2);
    shards[0] = .{ .mapped = region0, .catalog = empty_cat };
    shards[1] = .{ .mapped = region1, .catalog = empty_cat };

    const bundle: hf_loader_mod.HfBundle = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .shards = shards,
        .config_json = "",
        .tokenizer_json = null,
        .tokenizer_model = null,
    };
    // Don't call bundle.deinit() — that would munmap our test regions and
    // free `shards`, both of which are owned by this test's arena and defers.
    var bundle_arena = bundle.arena;
    defer bundle_arena.deinit();

    var be = try MetalBackend.initFromHf(allocator, &bundle, fakeCfg());
    defer be.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), be.weight_shards.len);
    // Spot-check that pointers from shard 1 resolve to a different MTLBuffer.
    const loc0 = be.resolveWeight(region0.ptr).?;
    const loc1 = be.resolveWeight(region1.ptr).?;
    try std.testing.expect(loc0.buf != loc1.buf);
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

fn fakeCfgQ4() model_mod.LlamaConfig {
    // Wide enough that scratch xb (cap=dim) holds k=64 floats and
    // scratch q (cap=n_heads*head_dim) holds m=4 floats.
    return .{
        .dim = 64,
        .n_layers = 1,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .ffn_dim = 64,
        .vocab_size = 128,
        .max_seq = 16,
    };
}

test "MetalBackend.matmulMlxQ4: routes weight + scales + biases across two shards" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const page = std.heap.pageSize();

    // Two anonymous mmap shards. Weight in shard 0, scales+biases in shard 1.
    const shard0_region = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard0_region);
    const shard1_region = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(shard1_region);

    // Q4 layout: m=4, k=64, group_size=64 (1 group/row).
    const m: usize = 4;
    const k: usize = 64;
    const group_size: u32 = 64;

    // Weight: m * k / 2 bytes packed; every nibble = 0x1 (q=1, dequant = (1-8) = -7).
    const w_bytes_len = m * k / 2;
    @memset(shard0_region[0..w_bytes_len], 0x11);

    // Scales: m * 1 group/row × 2 bytes = m*2 bytes. f16(1.0) = 0x3C00.
    const s_bytes_len = m * 2;
    var i: usize = 0;
    while (i < m) : (i += 1) std.mem.writeInt(u16, shard1_region[i * 2 ..][0..2], 0x3C00, .little);

    // Biases: same shape as scales, placed AFTER scales in shard 1. The
    // kernel computes `scale * Σ q*x + bias * Σ x` (MLX affine-block
    // dequant — bias absorbs the -zero_point*scale offset rather than
    // the kernel subtracting 8). To produce dequant = -7 with q=1 and
    // scale=1.0, set bias = -8.0. f16(-8.0) = 0xC800.
    const b_bytes_len = m * 2;
    var bi: usize = 0;
    while (bi < m) : (bi += 1) std.mem.writeInt(
        u16,
        shard1_region[s_bytes_len + bi * 2 ..][0..2],
        0xC800,
        .little,
    );

    const shards = [_]ShardInput{
        .{ .bytes = shard0_region[0..w_bytes_len] },
        .{ .bytes = shard1_region[0 .. s_bytes_len + b_bytes_len] },
    };

    var be = try MetalBackend.initShards(allocator, &shards, fakeCfgQ4());
    defer be.deinit(allocator);

    // acts (k=64 f32 slots) lives in scratch xb (cap = cfg.dim = 64).
    @memset(be.xb.ptr[0..k], 0);
    be.xb.ptr[0] = 1.0;

    // out (m=4 f32 slots) lives in scratch q (cap = n_heads*head_dim = 64).
    @memset(be.q.ptr[0..m], 0);

    const w_ptr: [*]const u8 = shard0_region.ptr;
    const s_ptr: [*]const u8 = shard1_region.ptr;
    const b_ptr: [*]const u8 = shard1_region.ptr + s_bytes_len;

    try be.matmulMlxQ4(
        be.q.buf, w_ptr, s_ptr, b_ptr, be.xb.buf,
        m, k, 4, group_size, mlx_kernels.ScaleDtype.f16,
    );

    // Expected: every nibble = 1, scale = 1.0, bias = -8.0, acts[0] = 1.0.
    // Per-row: scale * Σ q*x + bias * Σ x = 1.0*1.0 + (-8.0)*1.0 = -7.0.
    for (be.q.ptr[0..m]) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, -7.0), v, 1e-3);
    }
}

test "MetalBackend.matmulMlxQ4: rejects bits != 4 with UnsupportedBits" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const page = std.heap.pageSize();
    const region = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(region);

    const shards = [_]ShardInput{.{ .bytes = region }};
    var be = try MetalBackend.initShards(allocator, &shards, fakeCfgQ4());
    defer be.deinit(allocator);

    const err = be.matmulMlxQ4(
        be.q.buf, region.ptr, region.ptr, region.ptr, be.xb.buf,
        4, 64, 8, 64, mlx_kernels.ScaleDtype.f16,
    );
    try std.testing.expectError(error.UnsupportedBits, err);
}

test "MetalBackend.matmulMlxQ4: returns WeightNotMapped for stray pointer" {
    if (!metal.metal_enabled) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const page = std.heap.pageSize();
    const region = try std.posix.mmap(null, page,
        .{ .READ = true, .WRITE = true },
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(region);

    const shards = [_]ShardInput{.{ .bytes = region }};
    var be = try MetalBackend.initShards(allocator, &shards, fakeCfgQ4());
    defer be.deinit(allocator);

    const stray: [*]const u8 = @ptrFromInt(0xdead0000);
    const err = be.matmulMlxQ4(
        be.q.buf, stray, region.ptr, region.ptr, be.xb.buf,
        4, 64, 4, 64, mlx_kernels.ScaleDtype.f16,
    );
    try std.testing.expectError(error.WeightNotMapped, err);
}
