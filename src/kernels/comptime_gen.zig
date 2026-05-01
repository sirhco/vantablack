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
    q8_0,
    q4_k,
    q5_k,
    q6_k,
    ternary158,
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
        .q8_0 => &simd.matmul_q8_0,
        .q4_k => &simd.matmul_q4_k,
        .q5_k => &simd.matmul_q5_k,
        .q6_k => &simd.matmul_q6_k,
        .ternary158 => &simd.matmul_ternary158,
    };
}

test "dispatch returns Kernel type for every variant" {
    const std = @import("std");
    inline for (@typeInfo(QuantType).@"enum".fields) |field| {
        const k = comptime dispatch(@field(QuantType, field.name));
        try std.testing.expect(@TypeOf(k) == Kernel);
    }
}
