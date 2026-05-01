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
};

const vtb_metal_init = c_api.vtb_metal_init;
const vtb_metal_destroy = c_api.vtb_metal_destroy;
const vtb_metal_alloc = c_api.vtb_metal_alloc;
const vtb_metal_wrap = c_api.vtb_metal_wrap;
const vtb_metal_release = c_api.vtb_metal_release;
const vtb_metal_matmul_q8_0 = c_api.vtb_metal_matmul_q8_0;

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
};
