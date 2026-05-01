//! Token sampling: greedy / temperature / top-k / top-p.
//!
//! Pipeline (when temp > 0): scale logits by 1/T → optional top-k mask →
//! softmax → optional top-p nucleus mask → renormalize → sample by CDF.
//! When temp == 0, falls through to argmax regardless of other settings.

const std = @import("std");
const Allocator = std.mem.Allocator;

const math = @import("../kernels/math.zig");

pub const Config = struct {
    /// 0.0 = greedy argmax. Must be ≥ 0.
    temperature: f32 = 0.0,
    /// 0 = disabled. Otherwise keep only the top-k highest logits.
    top_k: usize = 0,
    /// 0.0 or 1.0 = disabled. Otherwise nucleus filter to cumulative probability mass `p`.
    top_p: f32 = 0.0,
    /// PRNG seed. Same seed + same logits = same output.
    seed: u64 = 0,
};

pub const Sampler = struct {
    cfg: Config,
    prng: std.Random.DefaultPrng,
    /// Reusable scratch [vocab_size]: index buffer for partial sort.
    indices: []usize,

    pub fn init(allocator: Allocator, cfg: Config, vocab_size: usize) !Sampler {
        const indices = try allocator.alloc(usize, vocab_size);
        return .{
            .cfg = cfg,
            .prng = std.Random.DefaultPrng.init(cfg.seed),
            .indices = indices,
        };
    }

    pub fn deinit(self: *Sampler, allocator: Allocator) void {
        allocator.free(self.indices);
        self.* = undefined;
    }

    /// `logits` is mutated in place (scaled / softmaxed). Returns selected token id.
    pub fn sample(self: *Sampler, logits: []f32) u32 {
        if (self.cfg.temperature == 0.0) {
            return @intCast(math.argmax(logits));
        }

        // Scale by 1/T.
        const inv_t = 1.0 / self.cfg.temperature;
        for (logits) |*l| l.* *= inv_t;

        // Top-k mask: keep highest-k logits, others → -inf.
        if (self.cfg.top_k > 0 and self.cfg.top_k < logits.len) {
            for (self.indices, 0..) |*idx, i| idx.* = i;
            // Partial sort: pick the k-th largest threshold.
            const k = self.cfg.top_k;
            std.mem.sort(usize, self.indices, logits, lessByLogitsDesc);
            const threshold = logits[self.indices[k - 1]];
            for (logits) |*l| {
                if (l.* < threshold) l.* = -std.math.inf(f32);
            }
        }

        math.softmax(logits);

        // Top-p (nucleus) mask: keep smallest set whose cumulative prob ≥ p.
        if (self.cfg.top_p > 0.0 and self.cfg.top_p < 1.0) {
            for (self.indices, 0..) |*idx, i| idx.* = i;
            std.mem.sort(usize, self.indices, logits, lessByLogitsDesc);
            var cumulative: f32 = 0;
            var cutoff: usize = self.indices.len;
            for (self.indices, 0..) |idx, rank| {
                cumulative += logits[idx];
                if (cumulative >= self.cfg.top_p) {
                    cutoff = rank + 1;
                    break;
                }
            }
            // Zero out everything past the cutoff.
            for (self.indices[cutoff..]) |idx| logits[idx] = 0;
            // Renormalize.
            var sum: f32 = 0;
            for (logits) |l| sum += l;
            if (sum > 0) {
                const inv = 1.0 / sum;
                for (logits) |*l| l.* *= inv;
            }
        }

        // Sample from the CDF.
        const u = self.prng.random().float(f32);
        var acc: f32 = 0;
        for (logits, 0..) |p, i| {
            acc += p;
            if (u <= acc) return @intCast(i);
        }
        // Fallback: last nonzero (edge case if probs sum to <1 due to fp).
        var last: usize = 0;
        for (logits, 0..) |p, i| if (p > 0) {
            last = i;
        };
        return @intCast(last);
    }
};

fn lessByLogitsDesc(logits: []const f32, a: usize, b: usize) bool {
    return logits[a] > logits[b];
}

// -- tests ----------------------------------------------------------------

test "greedy: temp=0 picks argmax" {
    const gpa = std.testing.allocator;
    var s = try Sampler.init(gpa, .{ .temperature = 0.0 }, 5);
    defer s.deinit(gpa);
    var logits = [_]f32{ 0.1, 0.5, 2.0, 0.3, 1.2 };
    try std.testing.expectEqual(@as(u32, 2), s.sample(&logits));
}

test "deterministic with seed" {
    const gpa = std.testing.allocator;
    var s1 = try Sampler.init(gpa, .{ .temperature = 1.0, .seed = 42 }, 5);
    defer s1.deinit(gpa);
    var s2 = try Sampler.init(gpa, .{ .temperature = 1.0, .seed = 42 }, 5);
    defer s2.deinit(gpa);
    var l1 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var l2 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectEqual(s1.sample(&l1), s2.sample(&l2));
}

test "top-k=1 acts as greedy" {
    const gpa = std.testing.allocator;
    var s = try Sampler.init(gpa, .{ .temperature = 1.0, .top_k = 1, .seed = 0 }, 5);
    defer s.deinit(gpa);
    var logits = [_]f32{ 0.1, 0.5, 2.0, 0.3, 1.2 };
    try std.testing.expectEqual(@as(u32, 2), s.sample(&logits));
}
