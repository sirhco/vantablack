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

pub const InitError = error{
    MetalUnavailable,
    MetalAllocFailed,
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

    // Whole-mmap weight buffer (no-copy wrap).
    weights_buf: *metal.Buf,
    weights_base: [*]const u8,
    weights_len: usize,

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
        _ = allocator;
        if (!metal.metal_enabled) return error.MetalUnavailable;

        var dev = metal.Device.init() catch return error.MetalUnavailable;
        errdefer dev.deinit();

        const page = std.heap.pageSize();
        const len_aligned = std.mem.alignForward(usize, mapper.mapped.len, page);
        const weights_buf = dev.wrap(@as([*]const u8, @ptrCast(mapper.mapped.ptr))[0..len_aligned]) catch
            return error.MetalAllocFailed;
        errdefer dev.release(weights_buf);

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
            .weights_buf = weights_buf,
            .weights_base = @ptrCast(mapper.mapped.ptr),
            .weights_len = len_aligned,
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
        self.dev.release(self.weights_buf);
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
        const w_ptr_int = @intFromPtr(weight_bytes.ptr);
        const base_int = @intFromPtr(self.weights_base);
        std.debug.assert(w_ptr_int >= base_int);
        const off = w_ptr_int - base_int;
        std.debug.assert(off + weight_bytes.len <= self.weights_len);
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
        try self.dev.matmulQ8_0(out_buf.buf, self.weights_buf, off, in_buf.buf, m, k);
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
