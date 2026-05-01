//! MLX-format quantized kernels (CPU).
//!
//! Layout (matches `mlx_lm.utils.quantize` output):
//!
//!   For a linear layer with `out_features` rows and `in_features` columns,
//!   `bits` ∈ {2, 3, 4, 6, 8} and group size `G` (typically 64 or 128):
//!
//!     <name>.weight : U32, shape [out_features, in_features * bits / 32]
//!         Each U32 packs `32 / bits` quantized values, low bits first.
//!     <name>.scales : F16/BF16, shape [out_features, in_features / G]
//!     <name>.biases : F16/BF16, shape [out_features, in_features / G]
//!
//!   Reconstructed weight: w[r, c] = q[r, c] * scales[r, c / G] + biases[r, c / G]
//!   where q[r, c] is the unsigned bits-bit value extracted from the packed
//!   U32 word at column index c.
//!
//! The MLX library always packs along the in_features (columns) axis. Group
//! boundaries align with G; G must divide in_features. The kernels here
//! assume that and assert at runtime.
//!
//! Only 4-bit + 8-bit are implemented in this first cut — those are the
//! formats MLX ships by default. Adding 2/3/6-bit is a matter of changing
//! the bit-extraction loop.

const std = @import("std");

pub const QuantBits = enum(u32) { q4 = 4, q8 = 8 };

/// Whether `scales` and `biases` are stored as IEEE-754 binary16 (F16) or
/// truncated f32 (BF16). MLX defaults to BF16; older mlx-lm versions emit F16.
pub const ScaleDtype = enum { f16, bf16 };

inline fn f16ToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

inline fn bf16ToF32(bits: u16) f32 {
    const w: u32 = @as(u32, bits) << 16;
    return @bitCast(w);
}

inline fn loadScale(bytes: []const u8, idx: usize, dtype: ScaleDtype) f32 {
    const off = idx * 2;
    const bits = std.mem.readInt(u16, bytes[off..][0..2], .little);
    return switch (dtype) {
        .f16 => f16ToF32(bits),
        .bf16 => bf16ToF32(bits),
    };
}

inline fn loadU32(bytes: []const u8, idx: usize) u32 {
    const off = idx * 4;
    return std.mem.readInt(u32, bytes[off..][0..4], .little);
}

/// Dequantize one row of an MLX-Q4 weight tensor into a scratch buffer.
/// `out.len` must equal `in_features`. `weight_row`/`scales_row`/`biases_row`
/// are the slices for that row only.
pub fn dequantRowQ4(
    out: []f32,
    weight_row: []const u8, // packed u32, in_features * 4 / 32 = in_features/8 u32s
    scales_row: []const u8, // f16/bf16, in_features / G entries
    biases_row: []const u8, // f16/bf16, in_features / G entries
    group_size: usize,
    dtype: ScaleDtype,
) void {
    const k = out.len;
    std.debug.assert(k % group_size == 0);
    std.debug.assert(k % 8 == 0);

    const n_groups = k / group_size;
    var col: usize = 0;
    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        const scale = loadScale(scales_row, g, dtype);
        const bias = loadScale(biases_row, g, dtype);
        const cols_left = group_size;
        var c: usize = 0;
        while (c < cols_left) : (c += 8) {
            const word = loadU32(weight_row, (col + c) / 8);
            // 8 nibbles, low nibble first.
            inline for (0..8) |i| {
                const q: u32 = (word >> @as(u5, i * 4)) & 0xF;
                out[col + c + i] = @as(f32, @floatFromInt(q)) * scale + bias;
            }
        }
        col += group_size;
    }
}

/// MLX-Q4 GEMV: `out = W @ x` for one matmul call, computed row-by-row.
/// Uses the dequantized-per-group inner loop directly without materializing
/// an intermediate dense fp32 weight. CPU baseline; Metal kernel TBD.
pub fn matmulQ4(
    out: []f32,
    weight: []const u8,
    scales: []const u8,
    biases: []const u8,
    acts: []const f32,
    m: usize,
    k: usize,
    group_size: usize,
    dtype: ScaleDtype,
) void {
    std.debug.assert(out.len >= m);
    std.debug.assert(acts.len >= k);
    std.debug.assert(k % group_size == 0);
    std.debug.assert(k % 8 == 0);

    const u32_per_row = k / 8;
    const groups_per_row = k / group_size;
    const weight_row_bytes = u32_per_row * 4;
    const scale_row_bytes = groups_per_row * 2; // 16-bit (f16 or bf16)

    var r: usize = 0;
    while (r < m) : (r += 1) {
        const w_row = weight[r * weight_row_bytes ..][0..weight_row_bytes];
        const s_row = scales[r * scale_row_bytes ..][0..scale_row_bytes];
        const b_row = biases[r * scale_row_bytes ..][0..scale_row_bytes];

        var total: f32 = 0.0;
        var col: usize = 0;
        var g: usize = 0;
        while (g < groups_per_row) : (g += 1) {
            const scale = loadScale(s_row, g, dtype);
            const bias = loadScale(b_row, g, dtype);
            // Σ_{c in group} (q*scale + bias) * x[c]
            //   = scale * Σ q*x[c] + bias * Σ x[c]
            var q_dot: f32 = 0;
            var x_sum: f32 = 0;
            const start = col;
            const end = col + group_size;
            var c = start;
            while (c < end) : (c += 8) {
                const word = loadU32(w_row, c / 8);
                inline for (0..8) |i| {
                    const q: u32 = (word >> @as(u5, i * 4)) & 0xF;
                    const xv = acts[c + i];
                    q_dot += @as(f32, @floatFromInt(q)) * xv;
                    x_sum += xv;
                }
            }
            total += scale * q_dot + bias * x_sum;
            col = end;
        }
        out[r] = total;
    }
}

// -- tests ----------------------------------------------------------------

test "dequantRowQ4: identity scale + zero bias recovers nibbles" {
    const k: usize = 16;
    const group_size: usize = 16;

    // Scales = 1.0 (f16 1.0 = 0x3C00), biases = 0.0.
    const scales: [2]u8 = .{ 0x00, 0x3C };
    const biases: [2]u8 = .{ 0x00, 0x00 };

    // Pack nibbles 0..7 into word0 and 8..15 into word1, low-nibble first.
    var weight: [8]u8 = undefined;
    var w0: u32 = 0;
    var w1: u32 = 0;
    inline for (0..8) |i| {
        w0 |= @as(u32, i) << @as(u5, i * 4);
        w1 |= @as(u32, i + 8) << @as(u5, i * 4);
    }
    std.mem.writeInt(u32, weight[0..4], w0, .little);
    std.mem.writeInt(u32, weight[4..8], w1, .little);

    var out: [k]f32 = undefined;
    dequantRowQ4(&out, &weight, &scales, &biases, group_size, .f16);

    inline for (0..16) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(i)), out[i], 1e-6);
    }
}

test "matmulQ4: scalar reference matches direct dot product" {
    // 1 output row, k = 16, group_size = 16.
    const m: usize = 1;
    const k: usize = 16;
    const group_size: usize = 16;

    // Same weight as previous test: nibbles 0..15.
    var weight: [8]u8 = undefined;
    var w0: u32 = 0;
    var w1: u32 = 0;
    inline for (0..8) |i| {
        w0 |= @as(u32, i) << @as(u5, i * 4);
        w1 |= @as(u32, i + 8) << @as(u5, i * 4);
    }
    std.mem.writeInt(u32, weight[0..4], w0, .little);
    std.mem.writeInt(u32, weight[4..8], w1, .little);

    // scale = 0.5, bias = 0.25.
    const scales: [2]u8 = .{ 0x00, 0x38 }; // 0x3800 = f16 0.5
    const biases: [2]u8 = .{ 0x00, 0x34 }; // 0x3400 = f16 0.25

    // Activations: all 1.0 → reference = Σ (q*0.5 + 0.25) = 0.5*Σq + 0.25*16
    //                                  = 0.5 * (0+1+…+15) + 4.0
    //                                  = 0.5 * 120 + 4 = 64
    var acts: [k]f32 = undefined;
    for (&acts) |*a| a.* = 1.0;

    var out: [m]f32 = undefined;
    matmulQ4(&out, &weight, &scales, &biases, &acts, m, k, group_size, .f16);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), out[0], 1e-4);

    // Activations: alternating +1, -1 → Σ x = 0, Σ q*x = -8
    //   q = 0,1,2,…,15; x = +1 at even idx, -1 at odd.
    //   Σ q*x = 0 - 1 + 2 - 3 + … + 14 - 15 = -8
    //   total = 0.5 * -8 + 0.25 * 0 = -4.0
    for (&acts, 0..) |*a, i| a.* = if (i % 2 == 0) 1.0 else -1.0;
    matmulQ4(&out, &weight, &scales, &biases, &acts, m, k, group_size, .f16);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), out[0], 1e-4);
}
