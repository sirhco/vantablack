//! Zig wrapper around the Objective-C Metal bridge in `bridge.m`.
//!
//! Only compiled when `-Dmetal=true` is set on the build command line.
//! When the flag is off, `metal_enabled` is false and consumers fall back
//! to the CPU path.

const std = @import("std");
const build_options = @import("build_options");

pub const metal_enabled = build_options.metal;

pub const Ctx = opaque {};
pub const Buf = opaque {};
pub const Seg = opaque {};

const c_api = if (metal_enabled) struct {
    extern "c" fn vtb_metal_init() ?*Ctx;
    extern "c" fn vtb_metal_destroy(ctx: ?*Ctx) void;
    extern "c" fn vtb_metal_alloc(ctx: *Ctx, bytes: usize, ptr_out: ?*?*anyopaque) ?*Buf;
    extern "c" fn vtb_metal_wrap(ctx: *Ctx, ptr: [*]const u8, len: usize) ?*Buf;
    extern "c" fn vtb_metal_release(buf: ?*Buf) void;
    extern "c" fn vtb_metal_matmul_q8_0(
        ctx: *Ctx,
        out_buf: *Buf,
        w_buf: *Buf,
        w_offset: usize,
        acts_buf: *Buf,
        m: usize,
        k: usize,
    ) c_int;
    extern "c" fn vtb_metal_segment_begin(ctx: *Ctx) ?*Seg;
    extern "c" fn vtb_metal_segment_commit(seg: *Seg) c_int;
    extern "c" fn vtb_metal_wait_idle(ctx: *Ctx) void;
    extern "c" fn vtb_metal_segment_matmul_q8_0(seg: *Seg, out_buf: *Buf, w_buf: *Buf, w_offset: usize, acts_buf: *Buf, m: usize, k: usize) void;
    extern "c" fn vtb_metal_segment_rmsnorm(seg: *Seg, out_buf: *Buf, in_buf: *Buf, weight_buf: *Buf, weight_offset: usize, n: usize, eps: f32) void;
    extern "c" fn vtb_metal_segment_rope(seg: *Seg, x_buf: *Buf, n_heads: usize, head_dim: usize, pos: usize, base: f32) void;
    extern "c" fn vtb_metal_segment_swiglu(seg: *Seg, gate_buf: *Buf, up_buf: *Buf, n: usize) void;
    extern "c" fn vtb_metal_segment_residual_add(seg: *Seg, a_buf: *Buf, b_buf: *Buf, n: usize) void;
    extern "c" fn vtb_metal_segment_copy(seg: *Seg, dst_buf: *Buf, dst_offset_bytes: usize, src_buf: *Buf, src_offset_bytes: usize, n_floats: usize) void;
    extern "c" fn vtb_metal_segment_attn_scores(seg: *Seg, scores_buf: *Buf, q_buf: *Buf, k_cache_buf: *Buf, k_offset: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize, inv_sqrt_hd: f32) void;
    extern "c" fn vtb_metal_segment_softmax_rows(seg: *Seg, scores_buf: *Buf, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) void;
    extern "c" fn vtb_metal_segment_attn_weighted_sum(seg: *Seg, out_buf: *Buf, scores_buf: *Buf, v_cache_buf: *Buf, v_offset: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) void;
} else struct {
    fn vtb_metal_init() ?*Ctx {
        return null;
    }
    fn vtb_metal_destroy(_: ?*Ctx) void {}
    fn vtb_metal_alloc(_: *Ctx, _: usize, _: ?*?*anyopaque) ?*Buf {
        return null;
    }
    fn vtb_metal_wrap(_: *Ctx, _: [*]const u8, _: usize) ?*Buf {
        return null;
    }
    fn vtb_metal_release(_: ?*Buf) void {}
    fn vtb_metal_matmul_q8_0(_: *Ctx, _: *Buf, _: *Buf, _: usize, _: *Buf, _: usize, _: usize) c_int {
        return 1;
    }
    fn vtb_metal_segment_begin(_: *Ctx) ?*Seg {
        return null;
    }
    fn vtb_metal_segment_commit(_: *Seg) c_int {
        return 1;
    }
    fn vtb_metal_wait_idle(_: *Ctx) void {}
    fn vtb_metal_segment_matmul_q8_0(_: *Seg, _: *Buf, _: *Buf, _: usize, _: *Buf, _: usize, _: usize) void {}
    fn vtb_metal_segment_rmsnorm(_: *Seg, _: *Buf, _: *Buf, _: *Buf, _: usize, _: usize, _: f32) void {}
    fn vtb_metal_segment_rope(_: *Seg, _: *Buf, _: usize, _: usize, _: usize, _: f32) void {}
    fn vtb_metal_segment_swiglu(_: *Seg, _: *Buf, _: *Buf, _: usize) void {}
    fn vtb_metal_segment_residual_add(_: *Seg, _: *Buf, _: *Buf, _: usize) void {}
    fn vtb_metal_segment_copy(_: *Seg, _: *Buf, _: usize, _: *Buf, _: usize, _: usize) void {}
    fn vtb_metal_segment_attn_scores(_: *Seg, _: *Buf, _: *Buf, _: *Buf, _: usize, _: usize, _: usize, _: usize, _: usize, _: f32) void {}
    fn vtb_metal_segment_softmax_rows(_: *Seg, _: *Buf, _: usize, _: usize, _: usize, _: usize) void {}
    fn vtb_metal_segment_attn_weighted_sum(_: *Seg, _: *Buf, _: *Buf, _: *Buf, _: usize, _: usize, _: usize, _: usize, _: usize) void {}
};

const vtb_metal_init = c_api.vtb_metal_init;
const vtb_metal_destroy = c_api.vtb_metal_destroy;
const vtb_metal_alloc = c_api.vtb_metal_alloc;
const vtb_metal_wrap = c_api.vtb_metal_wrap;
const vtb_metal_release = c_api.vtb_metal_release;
const vtb_metal_matmul_q8_0 = c_api.vtb_metal_matmul_q8_0;
const vtb_metal_segment_begin = c_api.vtb_metal_segment_begin;
const vtb_metal_segment_commit = c_api.vtb_metal_segment_commit;
const vtb_metal_wait_idle = c_api.vtb_metal_wait_idle;
const vtb_metal_segment_matmul_q8_0 = c_api.vtb_metal_segment_matmul_q8_0;
const vtb_metal_segment_rmsnorm = c_api.vtb_metal_segment_rmsnorm;
const vtb_metal_segment_rope = c_api.vtb_metal_segment_rope;
const vtb_metal_segment_swiglu = c_api.vtb_metal_segment_swiglu;
const vtb_metal_segment_residual_add = c_api.vtb_metal_segment_residual_add;
const vtb_metal_segment_copy = c_api.vtb_metal_segment_copy;
const vtb_metal_segment_attn_scores = c_api.vtb_metal_segment_attn_scores;
const vtb_metal_segment_softmax_rows = c_api.vtb_metal_segment_softmax_rows;
const vtb_metal_segment_attn_weighted_sum = c_api.vtb_metal_segment_attn_weighted_sum;

pub const InitError = error{MetalUnavailable};
pub const AllocError = error{MetalAllocFailed};
pub const DispatchError = error{MetalDispatchFailed};

pub const Device = struct {
    handle: *Ctx,

    pub fn init() InitError!Device {
        if (!metal_enabled) return error.MetalUnavailable;
        const ctx = vtb_metal_init() orelse return error.MetalUnavailable;
        return .{ .handle = ctx };
    }

    pub fn deinit(self: *Device) void {
        vtb_metal_destroy(self.handle);
        self.* = undefined;
    }

    /// Allocate a shared-storage buffer. `ptr_out` (if non-null) receives the
    /// CPU-accessible pointer.
    pub fn alloc(self: Device, bytes: usize, ptr_out: ?*?*anyopaque) AllocError!*Buf {
        return vtb_metal_alloc(self.handle, bytes, ptr_out) orelse error.MetalAllocFailed;
    }

    /// Wrap an existing host pointer (mmap'd region) as a no-copy shared
    /// MTLBuffer. The pointer must be page-aligned and `len` a page multiple.
    pub fn wrap(self: Device, bytes: []const u8) AllocError!*Buf {
        return vtb_metal_wrap(self.handle, bytes.ptr, bytes.len) orelse error.MetalAllocFailed;
    }

    pub fn release(self: Device, buf: *Buf) void {
        _ = self;
        vtb_metal_release(buf);
    }

    pub fn matmulQ8_0(
        self: Device,
        out_buf: *Buf,
        w_buf: *Buf,
        w_offset: usize,
        acts_buf: *Buf,
        m: usize,
        k: usize,
    ) DispatchError!void {
        const rc = vtb_metal_matmul_q8_0(self.handle, out_buf, w_buf, w_offset, acts_buf, m, k);
        if (rc != 0) return error.MetalDispatchFailed;
    }

    pub fn segmentBegin(self: Device) DispatchError!Segment {
        const seg = vtb_metal_segment_begin(self.handle) orelse return error.MetalDispatchFailed;
        return .{ .handle = seg };
    }

    /// Block until every previously-committed command buffer has finished.
    /// Must be called before the CPU reads any shared-storage buffer that an
    /// async segment wrote.
    pub fn waitIdle(self: Device) void {
        vtb_metal_wait_idle(self.handle);
    }
};

pub const Segment = struct {
    handle: *Seg,

    pub fn commit(self: Segment) DispatchError!void {
        const rc = vtb_metal_segment_commit(self.handle);
        if (rc != 0) return error.MetalDispatchFailed;
    }

    pub fn matmulQ8_0(self: Segment, out_buf: *Buf, w_buf: *Buf, w_offset: usize, acts_buf: *Buf, m: usize, k: usize) void {
        vtb_metal_segment_matmul_q8_0(self.handle, out_buf, w_buf, w_offset, acts_buf, m, k);
    }
    pub fn rmsnorm(self: Segment, out_buf: *Buf, in_buf: *Buf, weight_buf: *Buf, weight_offset: usize, n: usize, eps: f32) void {
        vtb_metal_segment_rmsnorm(self.handle, out_buf, in_buf, weight_buf, weight_offset, n, eps);
    }
    pub fn rope(self: Segment, x_buf: *Buf, n_heads: usize, head_dim: usize, pos: usize, base: f32) void {
        vtb_metal_segment_rope(self.handle, x_buf, n_heads, head_dim, pos, base);
    }
    pub fn swiglu(self: Segment, gate_buf: *Buf, up_buf: *Buf, n: usize) void {
        vtb_metal_segment_swiglu(self.handle, gate_buf, up_buf, n);
    }
    pub fn residualAdd(self: Segment, a_buf: *Buf, b_buf: *Buf, n: usize) void {
        vtb_metal_segment_residual_add(self.handle, a_buf, b_buf, n);
    }
    pub fn copy(self: Segment, dst_buf: *Buf, dst_offset_bytes: usize, src_buf: *Buf, src_offset_bytes: usize, n_floats: usize) void {
        vtb_metal_segment_copy(self.handle, dst_buf, dst_offset_bytes, src_buf, src_offset_bytes, n_floats);
    }
    pub fn attnScores(self: Segment, scores_buf: *Buf, q_buf: *Buf, k_cache_buf: *Buf, k_offset: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize, inv_sqrt_hd: f32) void {
        vtb_metal_segment_attn_scores(self.handle, scores_buf, q_buf, k_cache_buf, k_offset, n_heads, n_kv_heads, head_dim, seq_len, inv_sqrt_hd);
    }
    pub fn softmaxRows(self: Segment, scores_buf: *Buf, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) void {
        vtb_metal_segment_softmax_rows(self.handle, scores_buf, n_heads, n_kv_heads, head_dim, seq_len);
    }
    pub fn attnWeightedSum(self: Segment, out_buf: *Buf, scores_buf: *Buf, v_cache_buf: *Buf, v_offset: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) void {
        vtb_metal_segment_attn_weighted_sum(self.handle, out_buf, scores_buf, v_cache_buf, v_offset, n_heads, n_kv_heads, head_dim, seq_len);
    }
};
