//! SIMD kernels.
//!
//! Each `matmul_*` computes `out[i] = sum_j weights[i,j] * acts[j]` for a
//! row-major weight matrix of shape `[m, k]`. Weights are stored in the
//! quantization-specific block layout documented at each kernel.

const std = @import("std");
const builtin = @import("builtin");

pub const lane_width: usize = 8;
const Vec = @Vector(lane_width, f32);

pub fn dot_f32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lane_width <= a.len) : (i += lane_width) {
        const va: Vec = a[i..][0..lane_width].*;
        const vb: Vec = b[i..][0..lane_width].*;
        acc += va * vb;
    }

    var sum: f32 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

fn f16BitsToF32(bits: u16) f32 {
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

// -------------------- F32 (row-major dense) ------------------------------
//
// Weights laid out as `m * k` contiguous f32 in row-major order.

pub fn matmul_f32(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    std.debug.assert(weights.len >= m * k * @sizeOf(f32));

    const w_ptr: [*]align(1) const f32 = @ptrCast(weights.ptr);
    for (0..m) |i| {
        const row = w_ptr[i * k ..][0..k];
        // dot_f32 only requires same length, not alignment of either side.
        var acc: Vec = @splat(0.0);
        var j: usize = 0;
        while (j + lane_width <= k) : (j += lane_width) {
            // Unaligned load via vector cast on slice.
            const wv: Vec = blk: {
                var tmp: [lane_width]f32 = undefined;
                inline for (0..lane_width) |li| tmp[li] = row[j + li];
                break :blk tmp;
            };
            const av: Vec = acts[j..][0..lane_width].*;
            acc += wv * av;
        }
        var sum: f32 = @reduce(.Add, acc);
        while (j < k) : (j += 1) sum += row[j] * acts[j];
        out[i] = sum;
    }
}

// -------------------- F16 (row-major dense) ------------------------------
//
// Weights stored as f16 little-endian, m × k.

pub fn matmul_f16(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    std.debug.assert(weights.len >= m * k * 2);

    const row_bytes = k * 2;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var acc: f32 = 0;
        for (0..k) |j| {
            const bits = std.mem.readInt(u16, row[j * 2 ..][0..2], .little);
            acc += f16BitsToF32(bits) * acts[j];
        }
        out[i] = acc;
    }
}

/// BF16 → f32: just zero-extend the 16-bit half into the high half of the f32
/// bit pattern. No exponent/mantissa mangling required.
pub inline fn bf16BitsToF32(bits: u16) f32 {
    const w: u32 = @as(u32, bits) << 16;
    return @bitCast(w);
}

pub fn matmul_bf16(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    std.debug.assert(weights.len >= m * k * 2);

    const row_bytes = k * 2;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var acc: f32 = 0;
        for (0..k) |j| {
            const bits = std.mem.readInt(u16, row[j * 2 ..][0..2], .little);
            acc += bf16BitsToF32(bits) * acts[j];
        }
        out[i] = acc;
    }
}

// -------------------- Q8_0 ----------------------------------------------
//
// block_q8_0 = { f16 d; i8 qs[32]; }   // 34 bytes per 32 weights
// dequant: w[j] = d * qs[j]

pub const q8_0_block_bytes: usize = 34;
pub const q8_0_block_elems: usize = 32;

pub fn dequantBlockQ8_0(block: *const [q8_0_block_bytes]u8, out: *[q8_0_block_elems]f32) void {
    const scale = f16BitsToF32(std.mem.readInt(u16, block[0..2], .little));
    const qs_ptr: *const [q8_0_block_elems]i8 = @ptrCast(block[2..34]);
    for (out, qs_ptr) |*dst, q| dst.* = scale * @as(f32, @floatFromInt(q));
}

pub fn matmul_q8_0(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(k % q8_0_block_elems == 0);
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    const blocks_per_row = k / q8_0_block_elems;
    const row_bytes = blocks_per_row * q8_0_block_bytes;
    std.debug.assert(weights.len >= m * row_bytes);

    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var total: f32 = 0;
        for (0..blocks_per_row) |b| {
            const blk: *const [q8_0_block_bytes]u8 = row[b * q8_0_block_bytes ..][0..q8_0_block_bytes];
            const scale = f16BitsToF32(std.mem.readInt(u16, blk[0..2], .little));

            var block_dot: f32 = 0;
            // Process 32 weights as 4 chunks of 8 to fit @Vector(8, f32) lanes.
            inline for (0..4) |chunk| {
                var qs_arr: [lane_width]i8 = undefined;
                inline for (0..lane_width) |li| {
                    qs_arr[li] = @bitCast(blk[2 + chunk * lane_width + li]);
                }
                const qs_vec: @Vector(lane_width, i8) = qs_arr;
                const qs_f: Vec = @floatFromInt(qs_vec);
                const x_vec: Vec = acts[b * q8_0_block_elems + chunk * lane_width ..][0..lane_width].*;
                block_dot += @reduce(.Add, qs_f * x_vec);
            }
            total += scale * block_dot;
        }
        out[i] = total;
    }
}

// -------------------- Q4_K ----------------------------------------------
//
// block_q4_K (144 bytes per 256 weights, 8 sub-blocks of 32):
//   bytes [0..2]    : f16 d           super-block scale for sub-scales
//   bytes [2..4]    : f16 dmin        super-block scale for sub-mins
//   bytes [4..16]   : u8 scales[12]   packed 6-bit (scale, min) per sub-block
//   bytes [16..144] : u8 qs[128]      4-bit weights, 2 per byte
//
// Sub-blocks 2j and 2j+1 share qs[j*32 .. (j+1)*32]:
//   sub_block 2j:   low nibble  (q & 0xF) for l in 0..32
//   sub_block 2j+1: high nibble (q >> 4)  for l in 0..32
//
// Per sub-block s: get_scale_min_k4(s, scales, &sc, &mn);
//   sub_scale = d * sc;  sub_min = dmin * mn;
//   weight = sub_scale * q4 - sub_min

pub const q4_k_block_bytes: usize = 144;
pub const q4_k_block_elems: usize = 256;

inline fn q4kScaleMin(s: usize, scales: *const [12]u8) struct { sc: u8, mn: u8 } {
    if (s < 4) {
        return .{ .sc = scales[s] & 0x3F, .mn = scales[s + 4] & 0x3F };
    }
    return .{
        .sc = (scales[s + 4] & 0x0F) | ((scales[s - 4] >> 6) << 4),
        .mn = (scales[s + 4] >> 4) | ((scales[s] >> 6) << 4),
    };
}

pub fn dequantBlockQ4_K(block: *const [q4_k_block_bytes]u8, out: *[q4_k_block_elems]f32) void {
    const d = f16BitsToF32(std.mem.readInt(u16, block[0..2], .little));
    const dmin = f16BitsToF32(std.mem.readInt(u16, block[2..4], .little));
    const scales: *const [12]u8 = block[4..16];
    const qs: *const [128]u8 = block[16..144];

    var y_idx: usize = 0;
    var s: usize = 0;
    while (s < 8) : (s += 2) {
        const a = q4kScaleMin(s, scales);
        const b = q4kScaleMin(s + 1, scales);
        const d1 = d * @as(f32, @floatFromInt(a.sc));
        const m1 = dmin * @as(f32, @floatFromInt(a.mn));
        const d2 = d * @as(f32, @floatFromInt(b.sc));
        const m2 = dmin * @as(f32, @floatFromInt(b.mn));
        const q = qs[(s / 2) * 32 ..][0..32];
        for (0..32) |l| {
            out[y_idx + l] = d1 * @as(f32, @floatFromInt(q[l] & 0x0F)) - m1;
            out[y_idx + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
        }
        y_idx += 64;
    }
}

pub fn matmul_q4_k(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(k % q4_k_block_elems == 0);
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    const blocks_per_row = k / q4_k_block_elems;
    const row_bytes = blocks_per_row * q4_k_block_bytes;
    std.debug.assert(weights.len >= m * row_bytes);

    var dq: [q4_k_block_elems]f32 = undefined;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var total: f32 = 0;
        for (0..blocks_per_row) |b| {
            const blk: *const [q4_k_block_bytes]u8 = row[b * q4_k_block_bytes ..][0..q4_k_block_bytes];
            dequantBlockQ4_K(blk, &dq);
            const x = acts[b * q4_k_block_elems ..][0..q4_k_block_elems];
            // Vector dot of two len-256 f32 slices.
            var acc: Vec = @splat(0.0);
            var j: usize = 0;
            while (j + lane_width <= q4_k_block_elems) : (j += lane_width) {
                const wv: Vec = dq[j..][0..lane_width].*;
                const av: Vec = x[j..][0..lane_width].*;
                acc += wv * av;
            }
            total += @reduce(.Add, acc);
        }
        out[i] = total;
    }
}

// -------------------- Q5_K ----------------------------------------------
//
// block_q5_K (176 bytes per 256 weights):
//   bytes [0..2]    : f16 d           super-block scale for sub-scales
//   bytes [2..4]    : f16 dmin        super-block scale for sub-mins
//   bytes [4..16]   : u8 scales[12]   packed 6-bit (scale, min) per sub-block (same layout as Q4_K)
//   bytes [16..48]  : u8 qh[32]       1-bit hi nibble per weight, bit s of qh[l] serves sub-block s
//   bytes [48..176] : u8 qs[128]      4-bit lo nibble, 2 weights per byte
//
// 5-bit weight = (lo4 | (hi1 << 4)). Then weight = sub_scale * w5 - sub_min
// (same dequant formula as Q4_K).

pub const q5_k_block_bytes: usize = 176;
pub const q5_k_block_elems: usize = 256;

pub fn dequantBlockQ5_K(block: *const [q5_k_block_bytes]u8, out: *[q5_k_block_elems]f32) void {
    const d = f16BitsToF32(std.mem.readInt(u16, block[0..2], .little));
    const dmin = f16BitsToF32(std.mem.readInt(u16, block[2..4], .little));
    const scales: *const [12]u8 = block[4..16];
    const qh: *const [32]u8 = block[16..48];
    const qs: *const [128]u8 = block[48..176];

    var y_idx: usize = 0;
    var s: usize = 0;
    while (s < 8) : (s += 2) {
        const a = q4kScaleMin(s, scales);
        const b = q4kScaleMin(s + 1, scales);
        const d1 = d * @as(f32, @floatFromInt(a.sc));
        const m1 = dmin * @as(f32, @floatFromInt(a.mn));
        const d2 = d * @as(f32, @floatFromInt(b.sc));
        const m2 = dmin * @as(f32, @floatFromInt(b.mn));
        const q = qs[(s / 2) * 32 ..][0..32];
        const mask_lo: u8 = @as(u8, 1) << @intCast(s);
        const mask_hi: u8 = @as(u8, 1) << @intCast(s + 1);
        for (0..32) |l| {
            const w0: u8 = (q[l] & 0x0F) + (if ((qh[l] & mask_lo) != 0) @as(u8, 16) else 0);
            const w1: u8 = (q[l] >> 4) + (if ((qh[l] & mask_hi) != 0) @as(u8, 16) else 0);
            out[y_idx + l] = d1 * @as(f32, @floatFromInt(w0)) - m1;
            out[y_idx + 32 + l] = d2 * @as(f32, @floatFromInt(w1)) - m2;
        }
        y_idx += 64;
    }
}

pub fn matmul_q5_k(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(k % q5_k_block_elems == 0);
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    const blocks_per_row = k / q5_k_block_elems;
    const row_bytes = blocks_per_row * q5_k_block_bytes;
    std.debug.assert(weights.len >= m * row_bytes);

    var dq: [q5_k_block_elems]f32 = undefined;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var total: f32 = 0;
        for (0..blocks_per_row) |b| {
            const blk: *const [q5_k_block_bytes]u8 =
                row[b * q5_k_block_bytes ..][0..q5_k_block_bytes];
            dequantBlockQ5_K(blk, &dq);
            const x = acts[b * q5_k_block_elems ..][0..q5_k_block_elems];
            var acc: Vec = @splat(0.0);
            var j: usize = 0;
            while (j + lane_width <= q5_k_block_elems) : (j += lane_width) {
                const wv: Vec = dq[j..][0..lane_width].*;
                const av: Vec = x[j..][0..lane_width].*;
                acc += wv * av;
            }
            total += @reduce(.Add, acc);
        }
        out[i] = total;
    }
}

// -------------------- Q6_K ----------------------------------------------
//
// block_q6_K (210 bytes per 256 weights):
//   bytes [0..128]   : ql[128]    quants, low 4 bits, 2 weights per byte
//   bytes [128..192] : qh[64]     quants, high 2 bits, 4 weights per byte
//   bytes [192..208] : sc[16]     i8 sub-block scales (1 byte per 16 elems)
//   bytes [208..210] : f16 d      super-block scale
//
// Per the ggml reference (ggml-quants.c `dequantize_row_q6_K`), each
// super-block of 256 weights is processed in two halves of 128. For each l in
// 0..32 within a half, four weights (positions l, l+32, l+64, l+96) are
// reconstructed from one byte of `ql` plus 2 bits of `qh`, biased by -32, then
// scaled by `d * sc[is + n]` for n ∈ {0,2,4,6} and `is = l/16`.

pub const q6_k_block_bytes: usize = 210;
pub const q6_k_block_elems: usize = 256;

pub fn dequantBlockQ6_K(block: *const [q6_k_block_bytes]u8, out: *[q6_k_block_elems]f32) void {
    const d = f16BitsToF32(std.mem.readInt(u16, block[208..210], .little));
    const ql_all: *const [128]u8 = block[0..128];
    const qh_all: *const [64]u8 = block[128..192];
    const sc_all: *const [16]i8 = @ptrCast(block[192..208]);

    var y_off: usize = 0;
    var ql_off: usize = 0;
    var qh_off: usize = 0;
    var sc_off: usize = 0;
    var half: usize = 0;
    while (half < 2) : (half += 1) {
        for (0..32) |l| {
            const is = l / 16;
            const ql_l = ql_all[ql_off + l];
            const ql_l32 = ql_all[ql_off + l + 32];
            const qh_l = qh_all[qh_off + l];

            const q1: i32 = @as(i32, (ql_l & 0x0F) | (((qh_l >> 0) & 0x3) << 4)) - 32;
            const q2: i32 = @as(i32, (ql_l32 & 0x0F) | (((qh_l >> 2) & 0x3) << 4)) - 32;
            const q3: i32 = @as(i32, (ql_l >> 4) | (((qh_l >> 4) & 0x3) << 4)) - 32;
            const q4: i32 = @as(i32, (ql_l32 >> 4) | (((qh_l >> 6) & 0x3) << 4)) - 32;

            const s0: f32 = @floatFromInt(sc_all[sc_off + is + 0]);
            const s2: f32 = @floatFromInt(sc_all[sc_off + is + 2]);
            const s4: f32 = @floatFromInt(sc_all[sc_off + is + 4]);
            const s6: f32 = @floatFromInt(sc_all[sc_off + is + 6]);

            out[y_off + l + 0] = d * s0 * @as(f32, @floatFromInt(q1));
            out[y_off + l + 32] = d * s2 * @as(f32, @floatFromInt(q2));
            out[y_off + l + 64] = d * s4 * @as(f32, @floatFromInt(q3));
            out[y_off + l + 96] = d * s6 * @as(f32, @floatFromInt(q4));
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}

pub fn matmul_q6_k(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(k % q6_k_block_elems == 0);
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    const blocks_per_row = k / q6_k_block_elems;
    const row_bytes = blocks_per_row * q6_k_block_bytes;
    std.debug.assert(weights.len >= m * row_bytes);

    var dq: [q6_k_block_elems]f32 = undefined;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var total: f32 = 0;
        for (0..blocks_per_row) |b| {
            const blk: *const [q6_k_block_bytes]u8 =
                row[b * q6_k_block_bytes ..][0..q6_k_block_bytes];
            dequantBlockQ6_K(blk, &dq);
            const x = acts[b * q6_k_block_elems ..][0..q6_k_block_elems];
            var acc: Vec = @splat(0.0);
            var j: usize = 0;
            while (j + lane_width <= q6_k_block_elems) : (j += lane_width) {
                const wv: Vec = dq[j..][0..lane_width].*;
                const av: Vec = x[j..][0..lane_width].*;
                acc += wv * av;
            }
            total += @reduce(.Add, acc);
        }
        out[i] = total;
    }
}

// -------------------- TQ2_0 (1.58-bit ternary) ---------------------------
//
// block_tq2_0 (66 bytes per 256 weights):
//   bytes [0..64]   : qs[64]   2-bit weights, 4 packed per byte
//   bytes [64..66]  : f16 d    block scale
//
// Each 2-bit field stores {0, 1, 2}. Decode: ternary value = (raw - 1) ∈ {-1, 0, +1}.
// Final weight = scale * ternary.
//
// Layout within byte (matches ggml TQ2_0): four sub-groups of 64 weights each,
// where sub-group g (g in 0..4) takes bit positions [2g .. 2g+2] across the
// first 16 bytes of qs, then advances to the next 16 bytes for the next group.
// Concretely: weight at logical index `g*64 + l` lives in `qs[(g*16) + (l % 16)]`
// at bit offset `2 * (l / 16)`. (See ggml-quants.c `dequantize_row_tq2_0`.)

pub const tq2_0_block_bytes: usize = 66;
pub const tq2_0_block_elems: usize = 256;

pub fn dequantBlockTQ2_0(block: *const [tq2_0_block_bytes]u8, out: *[tq2_0_block_elems]f32) void {
    const scale = f16BitsToF32(std.mem.readInt(u16, block[64..66], .little));
    const qs: *const [64]u8 = block[0..64];

    // Outer loop over the four 64-element sub-groups (matches ggml layout).
    inline for (0..4) |g| {
        const shift: u3 = @intCast(2 * g);
        const base = g * 16;
        for (0..64) |l| {
            const byte = qs[base + (l % 16)];
            const raw: u8 = (byte >> shift) & 0x03;
            const tern: f32 = @as(f32, @floatFromInt(@as(i32, raw))) - 1.0;
            out[g * 64 + l] = scale * tern;
        }
    }
}

pub fn matmul_ternary158(out: []f32, weights: []const u8, acts: []const f32, m: usize, k: usize) void {
    std.debug.assert(k % tq2_0_block_elems == 0);
    std.debug.assert(out.len == m);
    std.debug.assert(acts.len == k);
    const blocks_per_row = k / tq2_0_block_elems;
    const row_bytes = blocks_per_row * tq2_0_block_bytes;
    std.debug.assert(weights.len >= m * row_bytes);

    var dq: [tq2_0_block_elems]f32 = undefined;
    for (0..m) |i| {
        const row = weights[i * row_bytes ..][0..row_bytes];
        var total: f32 = 0;
        for (0..blocks_per_row) |b| {
            const blk: *const [tq2_0_block_bytes]u8 = row[b * tq2_0_block_bytes ..][0..tq2_0_block_bytes];
            dequantBlockTQ2_0(blk, &dq);
            const x = acts[b * tq2_0_block_elems ..][0..tq2_0_block_elems];
            var acc: Vec = @splat(0.0);
            var j: usize = 0;
            while (j + lane_width <= tq2_0_block_elems) : (j += lane_width) {
                const wv: Vec = dq[j..][0..lane_width].*;
                const av: Vec = x[j..][0..lane_width].*;
                acc += wv * av;
            }
            total += @reduce(.Add, acc);
        }
        out[i] = total;
    }
}

// -- tests ----------------------------------------------------------------

test "dot_f32 vector path matches scalar reference" {
    var a: [19]f32 = undefined;
    var b: [19]f32 = undefined;
    var expected: f32 = 0.0;
    for (&a, &b, 0..) |*ai, *bi, idx| {
        ai.* = @floatFromInt(idx + 1);
        bi.* = @floatFromInt((idx + 1) * 2);
        expected += ai.* * bi.*;
    }
    try std.testing.expectApproxEqAbs(expected, dot_f32(&a, &b), 1e-4);
}

test "matmul_f32 matches scalar reference" {
    const m: usize = 5;
    const k: usize = 17;
    var w: [m * k]f32 = undefined;
    var x: [k]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @floatFromInt(i);
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.5;

    var got: [m]f32 = undefined;
    const w_bytes = std.mem.sliceAsBytes(&w);
    matmul_f32(&got, w_bytes, &x, m, k);

    var expected: [m]f32 = undefined;
    for (0..m) |i| {
        var s: f32 = 0;
        for (0..k) |j| s += w[i * k + j] * x[j];
        expected[i] = s;
    }
    for (got, expected) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-3);
}

test "matmul_f16 matches scalar reference" {
    const m: usize = 3;
    const k: usize = 8;
    var w_f16: [m * k]f16 = undefined;
    var x: [k]f32 = undefined;
    for (&w_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.25);
    for (&x, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var got: [m]f32 = undefined;
    matmul_f16(&got, std.mem.sliceAsBytes(&w_f16), &x, m, k);

    var expected: [m]f32 = undefined;
    for (0..m) |i| {
        var s: f32 = 0;
        for (0..k) |j| s += @as(f32, @floatCast(w_f16[i * k + j])) * x[j];
        expected[i] = s;
    }
    for (got, expected) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-3);
}

test "Q8_0 dequant round-trips" {
    var block: [q8_0_block_bytes]u8 = undefined;
    const scale: f16 = 0.125;
    std.mem.writeInt(u16, block[0..2], @bitCast(scale), .little);
    const qs: [q8_0_block_elems]i8 = .{ -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 100, -100, 50, -50, 25, -25, 12, -12, 6, -6, 3, -3, 1, -1, 0, 0, 0 };
    for (qs, 0..) |q, i| block[2 + i] = @bitCast(q);

    var out: [q8_0_block_elems]f32 = undefined;
    dequantBlockQ8_0(&block, &out);
    for (qs, out) |q, o| {
        const expected: f32 = @as(f32, @floatCast(scale)) * @as(f32, @floatFromInt(q));
        try std.testing.expectApproxEqAbs(expected, o, 1e-4);
    }
}

test "Q4_K dequant + matmul self-consistent" {
    var block: [q4_k_block_bytes]u8 = .{0} ** q4_k_block_bytes;
    const d: f16 = 0.0078125; // 1/128
    const dmin: f16 = 0.0625; // 1/16
    std.mem.writeInt(u16, block[0..2], @bitCast(d), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(dmin), .little);

    // Set scales[0..4] = sc bytes (low 6 bits used by sub-blocks 0..3)
    // Set scales[4..8] = mn bytes (low 6 bits used by sub-blocks 0..3)
    // Set scales[8..12] = mixed bytes for sub-blocks 4..7 (low4=sc, hi4=mn)
    //  combined with scales[0..4] high2 bits = sc[s+4] hi2; scales[4..8] high2 = mn[s+4] hi2.
    for (0..4) |s| {
        block[4 + s] = @intCast((s + 1) * 5); // sc[s] = 5,10,15,20 (≤63)
        block[4 + 4 + s] = @intCast(s + 2); // mn[s] = 2,3,4,5
    }
    // sub-blocks 4..7: low4 of scales[8+s] = sc[s+4] low4; hi4 = mn[s+4] low4
    // hi2 of sc[s+4] = (scales[s] >> 6) — currently 0 (since sc bytes ≤ 63)
    for (0..4) |s| {
        const sc_lo: u8 = @intCast((s + 1) * 3); // 3,6,9,12 (≤15 since fits low4)
        const mn_lo: u8 = @intCast(s + 1); // 1,2,3,4
        block[4 + 8 + s] = sc_lo | (mn_lo << 4);
    }
    // qs[i] sequence for verification
    for (0..128) |i| block[16 + i] = @intCast(((i * 7) % 16) | (((i * 11) % 16) << 4));

    var dq: [q4_k_block_elems]f32 = undefined;
    dequantBlockQ4_K(&block, &dq);

    // Spot-check sub-block 0, weight 0: should equal d*sc[0] * (qs[0] & 0xF) - dmin*mn[0]
    const sc0: u8 = block[4] & 0x3F;
    const mn0: u8 = block[8] & 0x3F;
    const d_f: f32 = @floatCast(d);
    const dmin_f: f32 = @floatCast(dmin);
    const expected0 = d_f * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(block[16] & 0x0F)) -
        dmin_f * @as(f32, @floatFromInt(mn0));
    try std.testing.expectApproxEqAbs(expected0, dq[0], 1e-4);

    // Sub-block 1, weight 0: high nibble of qs[0]
    const expected_sb1 = d_f * @as(f32, @floatFromInt(block[5] & 0x3F)) * @as(f32, @floatFromInt(block[16] >> 4)) -
        dmin_f * @as(f32, @floatFromInt(block[9] & 0x3F));
    try std.testing.expectApproxEqAbs(expected_sb1, dq[32], 1e-4);

    // Now: matmul against an activation vector, compare to dot of dequantized.
    var x: [q4_k_block_elems]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    var got: [1]f32 = undefined;
    matmul_q4_k(&got, &block, &x, 1, q4_k_block_elems);
    var expected: f32 = 0;
    for (dq, x) |w, xi| expected += w * xi;
    try std.testing.expectApproxEqAbs(expected, got[0], 1e-3);
}

test "Q5_K dequant + matmul self-consistent" {
    var block: [q5_k_block_bytes]u8 = .{0} ** q5_k_block_bytes;
    const d: f16 = 0.00390625; // 1/256
    const dmin: f16 = 0.03125; // 1/32
    std.mem.writeInt(u16, block[0..2], @bitCast(d), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(dmin), .little);
    // Same packed-scale layout as Q4_K (sub-blocks 0..3 in low 6 bits).
    for (0..4) |s| {
        block[4 + s] = @intCast((s + 1) * 4); // 4,8,12,16
        block[4 + 4 + s] = @intCast(s + 1); // 1,2,3,4
    }
    for (0..4) |s| {
        const sc_lo: u8 = @intCast((s + 1) * 2); // 2,4,6,8
        const mn_lo: u8 = @intCast(s + 1); // 1,2,3,4
        block[4 + 8 + s] = sc_lo | (mn_lo << 4);
    }
    // qh: bit s of qh[l] sets the 5th bit for sub-block s, weight l.
    for (0..32) |l| block[16 + l] = @intCast((l * 31) % 256);
    // qs: low 4 + high 4 nibbles of all sub-block pairs.
    for (0..128) |i| block[48 + i] = @intCast(((i * 13) % 16) | (((i * 7) % 16) << 4));

    var dq: [q5_k_block_elems]f32 = undefined;
    dequantBlockQ5_K(&block, &dq);

    // Spot-check: sub-block 0, weight 0.
    const sc0: u8 = block[4] & 0x3F;
    const mn0: u8 = block[8] & 0x3F;
    const d_f: f32 = @floatCast(d);
    const dmin_f: f32 = @floatCast(dmin);
    const lo0: u8 = block[48] & 0x0F;
    const hi0: u8 = if ((block[16] & 0x01) != 0) 16 else 0;
    const w5_0: u8 = lo0 + hi0;
    const expected0 = d_f * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(w5_0)) -
        dmin_f * @as(f32, @floatFromInt(mn0));
    try std.testing.expectApproxEqAbs(expected0, dq[0], 1e-4);

    // Matmul vs dot-of-dequantized.
    var x: [q5_k_block_elems]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.1 - 0.4;
    var got: [1]f32 = undefined;
    matmul_q5_k(&got, &block, &x, 1, q5_k_block_elems);
    var expected: f32 = 0;
    for (dq, x) |w, xi| expected += w * xi;
    try std.testing.expectApproxEqAbs(expected, got[0], 1e-3);
}

test "Q6_K dequant + matmul self-consistent" {
    var block: [q6_k_block_bytes]u8 = .{0} ** q6_k_block_bytes;
    const d: f16 = 0.001953125; // 1/512
    std.mem.writeInt(u16, block[208..210], @bitCast(d), .little);
    // ql + qh patterns; sc as i8.
    for (0..128) |i| block[i] = @intCast((i * 17) % 256);
    for (0..64) |i| block[128 + i] = @intCast((i * 19) % 256);
    for (0..16) |i| {
        const sv: i8 = @intCast(@as(i32, @intCast(i)) - 8); // -8..7
        block[192 + i] = @bitCast(sv);
    }

    var dq: [q6_k_block_elems]f32 = undefined;
    dequantBlockQ6_K(&block, &dq);

    // Spot-check: position 0 (half=0, l=0, n=0).
    // q1 = ((ql[0] & 0xF) | ((qh[0] & 0x3) << 4)) - 32
    const ql0: u8 = block[0];
    const qh0: u8 = block[128];
    const q1: i32 = @as(i32, (ql0 & 0x0F) | (((qh0 >> 0) & 0x3) << 4)) - 32;
    const sc0: i8 = @bitCast(block[192]);
    const d_f: f32 = @floatCast(d);
    const expected0 = d_f * @as(f32, @floatFromInt(sc0)) * @as(f32, @floatFromInt(q1));
    try std.testing.expectApproxEqAbs(expected0, dq[0], 1e-4);

    // Matmul vs dot-of-dequantized.
    var x: [q6_k_block_elems]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.25;
    var got: [1]f32 = undefined;
    matmul_q6_k(&got, &block, &x, 1, q6_k_block_elems);
    var expected: f32 = 0;
    for (dq, x) |w, xi| expected += w * xi;
    try std.testing.expectApproxEqAbs(expected, got[0], 1e-3);
}

test "TQ2_0 dequant + matmul self-consistent" {
    var block: [tq2_0_block_bytes]u8 = .{0} ** tq2_0_block_bytes;
    const scale: f16 = 0.25;
    std.mem.writeInt(u16, block[64..66], @bitCast(scale), .little);
    // Pattern: byte i has trits (i%3, (i+1)%3, (i+2)%3, i%3) packed.
    for (0..64) |i| {
        const a: u8 = @intCast(i % 3);
        const b: u8 = @intCast((i + 1) % 3);
        const c: u8 = @intCast((i + 2) % 3);
        const d: u8 = @intCast(i % 3);
        block[i] = a | (b << 2) | (c << 4) | (d << 6);
    }

    var dq: [tq2_0_block_elems]f32 = undefined;
    dequantBlockTQ2_0(&block, &dq);

    // Spot check: weight at logical index `g*64 + l` uses byte qs[g*16 + l%16],
    // shift = 2*g. raw = ((byte >> shift) & 0x3); tern = raw - 1; weight = scale * tern.
    const scale_f: f32 = @floatCast(scale);
    inline for (.{ 0, 64, 128, 192 }, .{ 0, 1, 2, 3 }) |idx, g| {
        const l = idx - g * 64;
        const byte = block[g * 16 + (l % 16)];
        const raw: u8 = (byte >> @as(u3, @intCast(2 * g))) & 0x3;
        const expected = scale_f * (@as(f32, @floatFromInt(raw)) - 1.0);
        try std.testing.expectApproxEqAbs(expected, dq[idx], 1e-5);
    }

    var x: [tq2_0_block_elems]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.2 - 0.1;
    var got: [1]f32 = undefined;
    matmul_ternary158(&got, &block, &x, 1, tq2_0_block_elems);
    var expected: f32 = 0;
    for (dq, x) |w, xi| expected += w * xi;
    try std.testing.expectApproxEqAbs(expected, got[0], 1e-3);
}

test "matmul_q8_0 matches scalar reference" {
    const m: usize = 4;
    const k: usize = 64; // 2 blocks per row
    const blocks_per_row = k / q8_0_block_elems;
    const row_bytes = blocks_per_row * q8_0_block_bytes;
    var weights: [m * row_bytes]u8 = undefined;

    // Fill: scale per block = 0.0625 * (block_idx + 1); qs[j] = ((i*97 + b*31 + j) % 17) - 8
    var ref_w: [m * k]f32 = undefined;
    for (0..m) |i| {
        for (0..blocks_per_row) |b| {
            const blk_off = i * row_bytes + b * q8_0_block_bytes;
            const scale: f16 = @floatCast(0.0625 * @as(f32, @floatFromInt(b + 1)));
            std.mem.writeInt(u16, weights[blk_off..][0..2], @bitCast(scale), .little);
            for (0..q8_0_block_elems) |j| {
                const v: i8 = @intCast(@as(i32, @intCast((i * 97 + b * 31 + j) % 17)) - 8);
                weights[blk_off + 2 + j] = @bitCast(v);
                ref_w[i * k + b * q8_0_block_elems + j] =
                    @as(f32, @floatCast(scale)) * @as(f32, @floatFromInt(v));
            }
        }
    }

    var x: [k]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.1 - 0.5;

    var got: [m]f32 = undefined;
    matmul_q8_0(&got, &weights, &x, m, k);

    var expected: [m]f32 = undefined;
    for (0..m) |i| {
        var s: f32 = 0;
        for (0..k) |j| s += ref_w[i * k + j] * x[j];
        expected[i] = s;
    }
    for (got, expected) |g, e| try std.testing.expectApproxEqAbs(e, g, 1e-3);
}
