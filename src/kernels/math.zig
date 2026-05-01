//! Transformer math primitives.

const std = @import("std");
const simd = @import("simd.zig");

const lane = simd.lane_width;
const Vec = @Vector(lane, f32);

/// In-place RMSNorm: x[i] = (x[i] / rms(x)) * weight[i]
/// rms(x) = sqrt(mean(x^2) + eps)
pub fn rmsNorm(x: []f32, weight: []const f32, eps: f32) void {
    std.debug.assert(x.len == weight.len);

    var ss_acc: Vec = @splat(0.0);
    var i: usize = 0;
    while (i + lane <= x.len) : (i += lane) {
        const v: Vec = x[i..][0..lane].*;
        ss_acc += v * v;
    }
    var ss: f32 = @reduce(.Add, ss_acc);
    while (i < x.len) : (i += 1) ss += x[i] * x[i];

    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);

    i = 0;
    while (i + lane <= x.len) : (i += lane) {
        const v: Vec = x[i..][0..lane].*;
        const w: Vec = weight[i..][0..lane].*;
        x[i..][0..lane].* = v * w * @as(Vec, @splat(inv));
    }
    while (i < x.len) : (i += 1) x[i] = x[i] * inv * weight[i];
}

/// In-place RoPE on `x` of length `head_dim`, for the given absolute position.
/// Standard interleaved-pair rotation: (x[2i], x[2i+1]) ← rotate by theta_i*pos
/// where theta_i = 1 / base^(2i/head_dim).
pub fn rope(x: []f32, pos: usize, base: f32) void {
    std.debug.assert(x.len % 2 == 0);
    const head_dim = x.len;
    const pos_f: f32 = @floatFromInt(pos);
    var i: usize = 0;
    while (i < head_dim) : (i += 2) {
        const exp: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(head_dim));
        const inv_freq = 1.0 / std.math.pow(f32, base, exp);
        const theta = pos_f * inv_freq;
        const c = @cos(theta);
        const s = @sin(theta);
        const x0 = x[i];
        const x1 = x[i + 1];
        x[i] = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

/// In-place numerically-stable softmax.
pub fn softmax(x: []f32) void {
    if (x.len == 0) return;
    var max_v: f32 = x[0];
    for (x[1..]) |v| {
        if (v > max_v) max_v = v;
    }
    var sum: f32 = 0;
    for (x) |*v| {
        const e = @exp(v.* - max_v);
        v.* = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (x) |*v| v.* *= inv;
}

/// SwiGLU: out[i] = silu(gate[i]) * up[i]   where silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
/// Result written into `gate` (so caller passes the gate proj output as the first slice).
pub fn swiglu(gate: []f32, up: []const f32) void {
    std.debug.assert(gate.len == up.len);
    for (gate, up) |*g, u| {
        const z = g.*;
        const silu = z / (1.0 + @exp(-z));
        g.* = silu * u;
    }
}

/// Add `b` into `a` element-wise (a += b).
pub fn addInto(a: []f32, b: []const f32) void {
    std.debug.assert(a.len == b.len);
    var i: usize = 0;
    while (i + lane <= a.len) : (i += lane) {
        const av: Vec = a[i..][0..lane].*;
        const bv: Vec = b[i..][0..lane].*;
        a[i..][0..lane].* = av + bv;
    }
    while (i < a.len) : (i += 1) a[i] += b[i];
}

/// Argmax index. Returns 0 for empty input.
pub fn argmax(x: []const f32) usize {
    if (x.len == 0) return 0;
    var best_i: usize = 0;
    var best_v: f32 = x[0];
    for (x[1..], 1..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    return best_i;
}

// -- tests ----------------------------------------------------------------

test "rmsNorm: identity weight zeros eps" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var ref = x;
    var ss: f32 = 0;
    for (ref) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(ref.len)));
    for (&ref) |*v| v.* *= inv;
    rmsNorm(&x, &w, 0.0);
    for (x, ref) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
}

test "rope: zero position is identity" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    rope(&x, 0, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), x[1], 1e-5);
}

test "softmax sums to 1" {
    var x = [_]f32{ 0.5, 1.5, -0.5, 2.0 };
    softmax(&x);
    var sum: f32 = 0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "swiglu against scalar reference" {
    var g = [_]f32{ 0.5, -1.0, 2.0, 0.0 };
    const u = [_]f32{ 1.0, 2.0, 0.5, 4.0 };
    var ref = [_]f32{0} ** 4;
    for (&ref, g, u) |*r, gi, ui| r.* = (gi / (1.0 + @exp(-gi))) * ui;
    swiglu(&g, &u);
    for (g, ref) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-5);
}

test "argmax" {
    try std.testing.expectEqual(@as(usize, 2), argmax(&[_]f32{ 0.1, 0.5, 1.5, 0.3 }));
}
