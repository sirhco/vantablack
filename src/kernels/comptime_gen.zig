//! Comptime kernel dispatcher.
//!
//! `dispatch(comptime q)` returns a specialized kernel function pointer at
//! compile time. The future inference inner loop will call:
//!
//!     switch (layer.quant) {
//!         inline .q8_0, .q4_k, .ternary158 => |q| {
//!             const kernel = comptime dispatch(q);
//!             kernel(out, weights, acts, m, k);
//!         },
//!         else => unreachable,
//!     }
//!
//! `inline` + `comptime dispatch(q)` emits one direct call per quant variant.
//! No runtime function-pointer indirection in the per-element inner loop.

const simd = @import("simd.zig");

pub const QuantType = enum {
    f32,
    f16,
    bf16,
    q8_0,
    q4_k,
    q5_k,
    q6_k,
    ternary158,
    /// MLX 4-bit affine block quant. Tensor is split across 3 buffers
    /// (`weight` packed u32 + `scales` + `biases`), so the standard
    /// `(out, weights, acts, m, k)` matmul signature does not fit.
    /// `forward.zig` routes this variant to `kernels/mlx.zig` directly
    /// rather than going through `dispatch()`.
    mlx_q4,
};

pub const Kernel = *const fn (
    out: []f32,
    weights: []const u8,
    acts: []const f32,
    m: usize,
    k: usize,
) void;

pub fn dispatch(comptime q: QuantType) Kernel {
    return switch (q) {
        .f32 => &simd.matmul_f32,
        .f16 => &simd.matmul_f16,
        .bf16 => &simd.matmul_bf16,
        .q8_0 => &simd.matmul_q8_0,
        .q4_k => &simd.matmul_q4_k,
        .q5_k => &simd.matmul_q5_k,
        .q6_k => &simd.matmul_q6_k,
        .ternary158 => &simd.matmul_ternary158,
        // MLX-Q4 doesn't fit the single-buffer Kernel signature; callers
        // must dispatch via `kernels/mlx.zig` directly.
        .mlx_q4 => @compileError("mlx_q4 cannot be dispatched via the single-buffer Kernel signature; call kernels.mlx.matmulQ4 directly"),
    };
}

test "dispatch returns Kernel type for every dispatchable variant" {
    const std = @import("std");
    inline for (@typeInfo(QuantType).@"enum".fields) |field| {
        const v = @field(QuantType, field.name);
        if (v == .mlx_q4) continue;
        const k = comptime dispatch(v);
        try std.testing.expect(@TypeOf(k) == Kernel);
    }
}
