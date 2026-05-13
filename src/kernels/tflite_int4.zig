//! TFLite per-axis INT4 dequantization (CPU).
//!
//! Decodes a tensor stored as packed signed 4-bit integers with one
//! `(scale, zero_point)` pair per row along the quantization axis. This
//! is the layout Google's `litert-lm` uses for the bulk of Gemma 4
//! E2B's MLP and attention weight tensors (see
//! `tests/golden/gemma4-e2b-architecture-notes.md`).
//!
//! Storage layout for a weight tensor with logical shape `[M, K]` and
//! `quantization_dimension = 0` (per-row scales):
//!
//!   packed[i, j]  = 4-bit signed value, two values per byte:
//!                     packed_bytes[(i * K + j) / 2]
//!                       low nibble  = even j  (j % 2 == 0)
//!                       high nibble = odd  j  (j % 2 == 1)
//!   scale[i]      : f32, one entry per row
//!   zero_point[i] : i64 (TFLite uses i64 in the proto; runtime narrows
//!                    to whatever fits the encoded range — for INT4 the
//!                    practical range is [-8, 7])
//!
//! Dequant formula (matches `TFLite` reference, see
//! `tensorflow/lite/kernels/internal/reference/dequantize.h`):
//!
//!   fp32[i, j] = (int4[i, j] - zero_point[i]) * scale[i]
//!
//! For `quantization_dimension = 1` (per-column scales), the same
//! formula applies with axes swapped — call `dequantizeInt4PerCol`
//! instead.

const std = @import("std");

pub const Error = error{
    ShapeMismatch,
    UnsupportedAxis,
};

/// Sign-extend a 4-bit unsigned nibble to i8.
inline fn signExtendInt4(nib: u8) i8 {
    // nib & 0x0F is in [0, 15]; values 8..15 represent negative.
    const v: i8 = @intCast(nib & 0x0F);
    return if (v >= 8) v - 16 else v;
}

/// Unpack one signed INT4 value from a packed byte buffer.
/// Even indices use the low nibble, odd indices use the high nibble.
pub inline fn unpackInt4(packed_bytes: []const u8, idx: usize) i8 {
    const byte = packed_bytes[idx / 2];
    const nib: u8 = if (idx % 2 == 0) byte & 0x0F else (byte >> 4) & 0x0F;
    return signExtendInt4(nib);
}

/// Pack two signed INT4 values into one byte. Test helper.
pub inline fn packInt4Pair(low: i8, high: i8) u8 {
    const lo: u8 = @as(u8, @bitCast(low)) & 0x0F;
    const hi: u8 = @as(u8, @bitCast(high)) & 0x0F;
    return lo | (hi << 4);
}

// ---------------------------------------------------------------------------
// INT2 — extreme quant used by Gemma 4 E2B's text embedder + many MLP layers.
// Signed 2-bit values, range [-2, 1], packed 4 per byte.
// ---------------------------------------------------------------------------

inline fn signExtendInt2(nib: u8) i8 {
    const v: i8 = @intCast(nib & 0x03);
    return if (v >= 2) v - 4 else v;
}

/// Unpack one signed INT2 value from a packed byte buffer.
/// 4 values per byte: bits 0-1 first, 2-3 second, 4-5 third, 6-7 fourth.
pub inline fn unpackInt2(packed_bytes: []const u8, idx: usize) i8 {
    const byte = packed_bytes[idx / 4];
    const shift: u3 = @intCast((idx % 4) * 2);
    return signExtendInt2(byte >> shift);
}

/// Pack 4 signed INT2 values into one byte. Test helper.
pub inline fn packInt2Quad(a: i8, b: i8, c: i8, d: i8) u8 {
    const av: u8 = @as(u8, @bitCast(a)) & 0x03;
    const bv: u8 = @as(u8, @bitCast(b)) & 0x03;
    const cv: u8 = @as(u8, @bitCast(c)) & 0x03;
    const dv: u8 = @as(u8, @bitCast(d)) & 0x03;
    return av | (bv << 2) | (cv << 4) | (dv << 6);
}

/// Dequantize a per-row INT2 tensor to FP32. Same shape contract as
/// `dequantizeInt4PerRow`. INT2 storage requires `(m*k + 3) / 4` bytes.
pub fn dequantizeInt2PerRow(
    packed_bytes: []const u8,
    scales: []const f32,
    zero_points: []const i64,
    m: usize,
    k: usize,
    out: []f32,
) Error!void {
    if (out.len != m * k) return error.ShapeMismatch;
    if (scales.len < m or zero_points.len < m) return error.ShapeMismatch;
    if (packed_bytes.len * 4 < m * k) return error.ShapeMismatch;

    var row: usize = 0;
    while (row < m) : (row += 1) {
        const scale = scales[row];
        const zp = zero_points[row];
        const row_off = row * k;
        var col: usize = 0;
        while (col < k) : (col += 1) {
            const v: i32 = unpackInt2(packed_bytes, row_off + col);
            const centered: i32 = v - @as(i32, @intCast(zp));
            out[row_off + col] = @as(f32, @floatFromInt(centered)) * scale;
        }
    }
}

/// Dequantize a per-row INT4 tensor (`quantization_dimension = 0`) to
/// FP32. Output buffer must be exactly `m * k` elements.
pub fn dequantizeInt4PerRow(
    packed_bytes: []const u8,
    scales: []const f32,
    zero_points: []const i64,
    m: usize,
    k: usize,
    out: []f32,
) Error!void {
    if (out.len != m * k) return error.ShapeMismatch;
    if (scales.len < m or zero_points.len < m) return error.ShapeMismatch;
    if (packed_bytes.len * 2 < m * k) return error.ShapeMismatch;

    var row: usize = 0;
    while (row < m) : (row += 1) {
        const scale = scales[row];
        const zp = zero_points[row];
        const row_off = row * k;
        var col: usize = 0;
        while (col < k) : (col += 1) {
            const v: i32 = unpackInt4(packed_bytes, row_off + col);
            const centered: i32 = v - @as(i32, @intCast(zp));
            out[row_off + col] = @as(f32, @floatFromInt(centered)) * scale;
        }
    }
}

/// Dequantize a per-column INT4 tensor (`quantization_dimension = 1`).
/// Same shape contract as `dequantizeInt4PerRow`; scales/zero_points
/// have `k` entries instead of `m`.
pub fn dequantizeInt4PerCol(
    packed_bytes: []const u8,
    scales: []const f32,
    zero_points: []const i64,
    m: usize,
    k: usize,
    out: []f32,
) Error!void {
    if (out.len != m * k) return error.ShapeMismatch;
    if (scales.len < k or zero_points.len < k) return error.ShapeMismatch;
    if (packed_bytes.len * 2 < m * k) return error.ShapeMismatch;

    var row: usize = 0;
    while (row < m) : (row += 1) {
        const row_off = row * k;
        var col: usize = 0;
        while (col < k) : (col += 1) {
            const v: i32 = unpackInt4(packed_bytes, row_off + col);
            const centered: i32 = v - @as(i32, @intCast(zero_points[col]));
            out[row_off + col] = @as(f32, @floatFromInt(centered)) * scales[col];
        }
    }
}

// -- tests ----------------------------------------------------------------

test "unpackInt4 round-trip across full range" {
    var i: i8 = -8;
    while (i <= 7) : (i += 1) {
        var buf = [_]u8{0};
        buf[0] = packInt4Pair(i, 0);
        try std.testing.expectEqual(i, unpackInt4(&buf, 0));
        buf[0] = packInt4Pair(0, i);
        try std.testing.expectEqual(i, unpackInt4(&buf, 1));
    }
}

test "dequantizeInt4PerRow basic 2x4 tensor" {
    // Two rows of 4 cols = 8 INT4 values = 4 bytes.
    // Row 0: [1, -1, 2, -2]  scale=1.0  zp=0  → fp32 [1, -1, 2, -2]
    // Row 1: [3, 3, 3, 3]    scale=0.5  zp=1  → fp32 [(3-1)*0.5]*4 = [1,1,1,1]
    var packed_bytes: [4]u8 = .{
        packInt4Pair(1, -1),
        packInt4Pair(2, -2),
        packInt4Pair(3, 3),
        packInt4Pair(3, 3),
    };
    const scales = [_]f32{ 1.0, 0.5 };
    const zps = [_]i64{ 0, 1 };
    var out: [8]f32 = undefined;
    try dequantizeInt4PerRow(&packed_bytes, &scales, &zps, 2, 4, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[7], 1e-6);
}

test "dequantizeInt4PerCol uses per-column scale" {
    // 2x4 tensor, per-column scales:
    //   col scales = [1.0, 2.0, 1.0, 2.0]
    //   col zps    = [0, 0, 0, 0]
    //   all packed values = 1
    var packed_bytes: [4]u8 = .{
        packInt4Pair(1, 1),
        packInt4Pair(1, 1),
        packInt4Pair(1, 1),
        packInt4Pair(1, 1),
    };
    const scales = [_]f32{ 1.0, 2.0, 1.0, 2.0 };
    const zps = [_]i64{ 0, 0, 0, 0 };
    var out: [8]f32 = undefined;
    try dequantizeInt4PerCol(&packed_bytes, &scales, &zps, 2, 4, &out);
    // Each cell should be (1 - 0) * scales[col]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[7], 1e-6);
}

test "dequantizeInt4PerRow rejects size mismatch" {
    var packed_bytes: [2]u8 = .{ 0, 0 };
    const scales = [_]f32{1.0};
    const zps = [_]i64{0};
    var out: [4]f32 = undefined;
    // packed is 4 INT4 values (2 bytes), m=2 k=4 wants 8 INT4 → error.
    try std.testing.expectError(error.ShapeMismatch, dequantizeInt4PerRow(&packed_bytes, &scales, &zps, 2, 4, &out));
}

test "unpackInt2 round-trip across [-2, 1]" {
    var i: i8 = -2;
    while (i <= 1) : (i += 1) {
        var buf = [_]u8{0};
        buf[0] = packInt2Quad(i, 0, 0, 0);
        try std.testing.expectEqual(i, unpackInt2(&buf, 0));
        buf[0] = packInt2Quad(0, i, 0, 0);
        try std.testing.expectEqual(i, unpackInt2(&buf, 1));
        buf[0] = packInt2Quad(0, 0, i, 0);
        try std.testing.expectEqual(i, unpackInt2(&buf, 2));
        buf[0] = packInt2Quad(0, 0, 0, i);
        try std.testing.expectEqual(i, unpackInt2(&buf, 3));
    }
}

test "dequantizeInt2PerRow basic 1x8 tensor" {
    // Single row of 8 INT2 values = 2 bytes.
    // Values: [-2, -1, 0, 1, -2, -1, 0, 1]
    // scale=2.0, zp=0  →  fp32 = v * 2
    var packed_bytes: [2]u8 = .{
        packInt2Quad(-2, -1, 0, 1),
        packInt2Quad(-2, -1, 0, 1),
    };
    const scales = [_]f32{2.0};
    const zps = [_]i64{0};
    var out: [8]f32 = undefined;
    try dequantizeInt2PerRow(&packed_bytes, &scales, &zps, 1, 8, &out);
    const expect = [_]f32{ -4, -2, 0, 2, -4, -2, 0, 2 };
    for (expect, out) |e, a| try std.testing.expectApproxEqAbs(e, a, 1e-6);
}

test "dequantizeInt4PerRow handles negative zero_point + full int4 range" {
    // Row with all 8 representable values, scale=1, zp=-1.
    //   stored:  [-8, -7, -6, ..., 6, 7]  (16 values, but row is 4 long)
    // Use values [-8, -7, 6, 7] in row 0, scale=1, zp=-1.
    //   fp32 = (v - (-1)) * 1 = v + 1
    var packed_bytes: [2]u8 = .{
        packInt4Pair(-8, -7),
        packInt4Pair(6, 7),
    };
    const scales = [_]f32{1.0};
    const zps = [_]i64{-1};
    var out: [4]f32 = undefined;
    try dequantizeInt4PerRow(&packed_bytes, &scales, &zps, 1, 4, &out);
    try std.testing.expectApproxEqAbs(@as(f32, -7.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), out[3], 1e-6);
}
