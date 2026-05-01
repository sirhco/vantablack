//! Per-process Metal backend state.
//!
//! Owns: the MTLDevice, a no-copy MTLBuffer wrapping the entire mmap'd weight
//! region, and pre-allocated scratch buffers for activations + output. Init
//! cost is amortized across all subsequent matmul calls.
//!
//! Currently only Q8_0 matmul has a GPU kernel. Other ops fall back to the
//! CPU thread pool. See `runtime/forward.zig`.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

const metal = @import("../metal/bridge.zig");
const model_mod = @import("model.zig");
const mapper_mod = @import("../core/mapper.zig");

pub const InitError = error{
    MetalUnavailable,
    MetalAllocFailed,
};

pub const MetalBackend = struct {
    dev: metal.Device,
    weights_buf: *metal.Buf,
    weights_base: [*]const u8,
    weights_len: usize,

    /// Reusable shared MTLBuffer for activations input. Sized to fit the
    /// largest k of any matmul in the model (`max(dim, ffn_dim, kv_dim)`).
    acts_buf: *metal.Buf,
    acts_ptr: [*]f32,
    acts_cap: usize,

    /// Reusable shared MTLBuffer for matmul output. Sized to fit the largest
    /// m of any matmul (`max(dim, ffn_dim, vocab_size)`).
    out_buf: *metal.Buf,
    out_ptr: [*]f32,
    out_cap: usize,

    pub fn init(
        allocator: Allocator,
        mapper: *const mapper_mod.ModelMapper,
        cfg: model_mod.LlamaConfig,
    ) InitError!MetalBackend {
        _ = allocator;
        if (!metal.metal_enabled) return error.MetalUnavailable;

        var dev = metal.Device.init() catch return error.MetalUnavailable;
        errdefer dev.deinit();

        // Wrap the entire mmap'd weight region. mmap maps full pages so the
        // allocation is already page-aligned; round len up to page boundary.
        const page = std.heap.pageSize();
        const len_aligned = std.mem.alignForward(usize, mapper.mapped.len, page);
        const weights_buf = dev.wrap(@as([*]const u8, @ptrCast(mapper.mapped.ptr))[0..len_aligned]) catch
            return error.MetalAllocFailed;
        errdefer dev.release(weights_buf);

        const max_k = @max(cfg.dim, @max(cfg.ffn_dim, cfg.n_kv_heads * cfg.head_dim));
        const max_m = @max(cfg.vocab_size, @max(cfg.dim, cfg.ffn_dim));

        var acts_ptr_raw: ?*anyopaque = null;
        const acts_buf = dev.alloc(max_k * @sizeOf(f32), &acts_ptr_raw) catch
            return error.MetalAllocFailed;
        errdefer dev.release(acts_buf);

        var out_ptr_raw: ?*anyopaque = null;
        const out_buf = dev.alloc(max_m * @sizeOf(f32), &out_ptr_raw) catch
            return error.MetalAllocFailed;
        errdefer dev.release(out_buf);

        return .{
            .dev = dev,
            .weights_buf = weights_buf,
            .weights_base = @ptrCast(mapper.mapped.ptr),
            .weights_len = len_aligned,
            .acts_buf = acts_buf,
            .acts_ptr = @as([*]f32, @ptrCast(@alignCast(acts_ptr_raw.?))),
            .acts_cap = max_k,
            .out_buf = out_buf,
            .out_ptr = @as([*]f32, @ptrCast(@alignCast(out_ptr_raw.?))),
            .out_cap = max_m,
        };
    }

    pub fn deinit(self: *MetalBackend) void {
        self.dev.release(self.out_buf);
        self.dev.release(self.acts_buf);
        self.dev.release(self.weights_buf);
        self.dev.deinit();
        self.* = undefined;
    }

    /// Q8_0 matmul: out = W @ acts using the GPU. Input/output are host f32
    /// slices; this fn handles the in/out memcpy through the shared scratch
    /// buffers.
    pub fn matmulQ8_0(
        self: *MetalBackend,
        out: []f32,
        weight_bytes: []const u8,
        acts: []const f32,
        m: usize,
        k: usize,
    ) !void {
        std.debug.assert(out.len >= m);
        std.debug.assert(acts.len >= k);
        std.debug.assert(k <= self.acts_cap);
        std.debug.assert(m <= self.out_cap);

        // Compute byte offset into the wrapped mmap region.
        const w_ptr_int = @intFromPtr(weight_bytes.ptr);
        const base_int = @intFromPtr(self.weights_base);
        std.debug.assert(w_ptr_int >= base_int);
        const w_offset = w_ptr_int - base_int;
        std.debug.assert(w_offset + weight_bytes.len <= self.weights_len);

        // Stage acts into shared buffer, dispatch, read back.
        @memcpy(self.acts_ptr[0..k], acts);
        try self.dev.matmulQ8_0(self.out_buf, self.weights_buf, w_offset, self.acts_buf, m, k);
        @memcpy(out[0..m], self.out_ptr[0..m]);
    }
};
