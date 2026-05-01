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
