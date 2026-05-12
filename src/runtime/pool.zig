//! Minimal persistent thread pool.
//!
//! N-1 worker threads spin on a shared atomic `epoch` counter. The main
//! thread participates as worker 0. Each `dispatch` call increments the
//! epoch, runs worker 0 in-line, then spins until all workers report done
//! for that epoch.
//!
//! Designed for many small jobs (one per matmul) where pthread_create per
//! job would dominate runtime. Workers are spawned once at `init` and
//! parked on a spin until destroyed.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

pub const JobFn = *const fn (worker_id: usize, n_workers: usize, ctx: *anyopaque) void;

const Job = struct {
    fn_ptr: JobFn,
    ctx: *anyopaque,
};

pub const ThreadPool = struct {
    threads: []std.Thread,
    n_workers: usize, // includes main thread (worker_id 0)

    /// Active worker count, mutable. `n_active <= n_workers`. Workers
    /// with id >= n_active still spin on the epoch counter and bump
    /// `completed` each epoch (so dispatch's wait target stays stable),
    /// but skip calling the job function. Used for thermal-pressure
    /// throttling — see `setActiveWorkersClamped`.
    n_active: std.atomic.Value(usize),

    epoch: std.atomic.Value(u64),
    completed: std.atomic.Value(u64),
    job: std.atomic.Value(?*Job) align(64),
    shutdown: std.atomic.Value(bool),

    /// `requested == 0` triggers autodetect: defaults to roughly the number of
    /// performance cores. On Apple Silicon this avoids burning the
    /// efficiency cores (which are ~5x slower per watt at matmul) and cuts
    /// total CPU usage roughly in half versus pinning every logical CPU.
    /// Caller can pass an explicit count to override.
    pub fn init(allocator: Allocator, requested: usize) !*ThreadPool {
        const n_workers = blk: {
            if (builtin.single_threaded) break :blk 1;
            if (requested != 0) break :blk @max(requested, 1);
            const total = std.Thread.getCpuCount() catch 1;
            // Heuristic: assume ~half are perf cores. M1/M2/M3 typically have
            // 4 efficiency cores so this maps cleanly. Hosts with no e-cores
            // (Intel, AMD) get capped at half their logical CPUs which favours
            // power+thermal headroom for other work; override with --threads.
            break :blk @max(1, (total + 1) / 2);
        };

        const self = try allocator.create(ThreadPool);
        errdefer allocator.destroy(self);
        self.* = .{
            .threads = &.{},
            .n_workers = n_workers,
            .n_active = .init(n_workers),
            .epoch = .init(0),
            .completed = .init(0),
            .job = .init(null),
            .shutdown = .init(false),
        };

        if (!builtin.single_threaded and n_workers > 1) {
            self.threads = try allocator.alloc(std.Thread, n_workers - 1);
            errdefer allocator.free(self.threads);
            for (self.threads, 1..) |*t, worker_id| {
                t.* = try std.Thread.spawn(.{}, workerLoop, .{ self, worker_id });
            }
        }
        return self;
    }

    pub fn deinit(self: *ThreadPool, allocator: Allocator) void {
        if (!builtin.single_threaded) {
            self.shutdown.store(true, .release);
            _ = self.epoch.fetchAdd(1, .release);
            for (self.threads) |t| t.join();
        }
        if (self.threads.len > 0) allocator.free(self.threads);
        allocator.destroy(self);
    }

    /// Run `job_fn(worker_id, active_workers, ctx)` once per active
    /// worker (0..active_workers). Worker 0 runs on the calling thread.
    /// `active_workers` reflects the current `n_active` snapshot — see
    /// `setActiveWorkersClamped` for the thermal-throttling hook.
    pub fn dispatch(self: *ThreadPool, job_fn: JobFn, ctx: *anyopaque) void {
        const active = self.n_active.load(.acquire);
        if (self.n_workers == 1 or active == 1) {
            job_fn(0, 1, ctx);
            return;
        }
        var job: Job = .{ .fn_ptr = job_fn, .ctx = ctx };
        self.job.store(&job, .release);
        self.completed.store(0, .release);
        _ = self.epoch.fetchAdd(1, .release);

        // Main does worker 0.
        job_fn(0, active, ctx);

        // Wait for n_workers-1 others. ALL spawned workers complete
        // every epoch (inactive ones just skip the job body), so the
        // wait target is the fixed pool size, not the active count.
        const target: u64 = self.n_workers - 1;
        while (self.completed.load(.acquire) < target) {
            std.atomic.spinLoopHint();
        }
        self.job.store(null, .release);
    }

    /// Total worker count established at init (includes main thread).
    /// Constant across the pool's lifetime.
    pub fn workerCount(self: *const ThreadPool) usize {
        return self.n_workers;
    }

    /// Throttle hook for thermal / memory-pressure callers. Clamps
    /// `target` to `[1, n_workers]` and stores it in the atomic
    /// `n_active`. Next `dispatch` call observes the new value; any
    /// in-flight dispatch finishes with the old value (no preemption).
    /// Safe to call from any thread.
    pub fn setActiveWorkersClamped(self: *ThreadPool, target: usize) void {
        const clamped = @max(1, @min(target, self.n_workers));
        self.n_active.store(clamped, .release);
    }

    /// Snapshot of the active worker count. Mainly for introspection
    /// and tests; the hot path reads `n_active` directly inside
    /// `dispatch`.
    pub fn activeWorkers(self: *const ThreadPool) usize {
        return self.n_active.load(.acquire);
    }
};

/// After this many spin iterations without seeing a new epoch, the worker
/// yields to the scheduler. Pure spinning would pin every CPU at ~100%
/// even when the pool is idle (server mode between requests). Yielding
/// preserves wake-up latency in the hot path while letting the OS reclaim
/// the CPU when no work is pending.
const idle_spin_threshold: u32 = 1024;

fn workerLoop(pool: *ThreadPool, worker_id: usize) void {
    var seen_epoch: u64 = 0;
    while (true) {
        // Spin briefly, then yield, until a new epoch arrives.
        var spins: u32 = 0;
        while (true) {
            const e = pool.epoch.load(.acquire);
            if (e != seen_epoch) {
                seen_epoch = e;
                break;
            }
            if (pool.shutdown.load(.acquire)) return;
            if (spins < idle_spin_threshold) {
                std.atomic.spinLoopHint();
                spins += 1;
            } else {
                std.Thread.yield() catch {};
            }
        }
        if (pool.shutdown.load(.acquire)) return;
        const job = pool.job.load(.acquire) orelse continue;
        const active = pool.n_active.load(.acquire);
        // Inactive workers (id >= active) still acknowledge the epoch
        // by bumping `completed`, so the main thread's wait target is
        // satisfied. They just don't execute the job body.
        if (worker_id < active) {
            job.fn_ptr(worker_id, active, job.ctx);
        }
        _ = pool.completed.fetchAdd(1, .release);
    }
}

// -- tests ----------------------------------------------------------------

const Sum = struct {
    inputs: []const u32,
    partials: []std.atomic.Value(u64),
};

fn sumWorker(worker_id: usize, n_workers: usize, ctx: *anyopaque) void {
    const s: *Sum = @ptrCast(@alignCast(ctx));
    const chunk = (s.inputs.len + n_workers - 1) / n_workers;
    const start = worker_id * chunk;
    const end = @min(start + chunk, s.inputs.len);
    var acc: u64 = 0;
    for (s.inputs[start..end]) |v| acc += v;
    s.partials[worker_id].store(acc, .release);
}

test "ThreadPool sums chunks" {
    const gpa = std.testing.allocator;
    var pool = try ThreadPool.init(gpa, 4);
    defer pool.deinit(gpa);

    var inputs: [1024]u32 = undefined;
    for (&inputs, 0..) |*v, i| v.* = @intCast(i + 1);

    var partials: [4]std.atomic.Value(u64) = .{ .init(0), .init(0), .init(0), .init(0) };
    var ctx: Sum = .{ .inputs = &inputs, .partials = &partials };
    pool.dispatch(sumWorker, &ctx);

    var total: u64 = 0;
    for (partials) |p| total += p.load(.acquire);
    try std.testing.expectEqual(@as(u64, 1024 * 1025 / 2), total);
}

const Counter = struct {
    flags: [8]std.atomic.Value(u8) = .{ .init(0), .init(0), .init(0), .init(0), .init(0), .init(0), .init(0), .init(0) },
};

fn markWorker(worker_id: usize, _: usize, ctx: *anyopaque) void {
    const c: *Counter = @ptrCast(@alignCast(ctx));
    c.flags[worker_id].store(1, .release);
}

test "ThreadPool.setActiveWorkersClamped restricts active worker set" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var pool = try ThreadPool.init(gpa, 4);
    defer pool.deinit(gpa);

    // Baseline: all 4 workers fire.
    var c1: Counter = .{};
    pool.dispatch(markWorker, &c1);
    try std.testing.expectEqual(@as(usize, 4), pool.activeWorkers());
    var hits1: usize = 0;
    for (&c1.flags) |*f| {
        if (f.load(.acquire) == 1) hits1 += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), hits1);

    // Throttle to 2 workers.
    pool.setActiveWorkersClamped(2);
    try std.testing.expectEqual(@as(usize, 2), pool.activeWorkers());
    var c2: Counter = .{};
    pool.dispatch(markWorker, &c2);
    var hits2: usize = 0;
    for (&c2.flags) |*f| {
        if (f.load(.acquire) == 1) hits2 += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), hits2);
    // Only workers 0 and 1 should have fired.
    try std.testing.expectEqual(@as(u8, 1), c2.flags[0].load(.acquire));
    try std.testing.expectEqual(@as(u8, 1), c2.flags[1].load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), c2.flags[2].load(.acquire));
    try std.testing.expectEqual(@as(u8, 0), c2.flags[3].load(.acquire));

    // Out-of-range targets clamp.
    pool.setActiveWorkersClamped(0);
    try std.testing.expectEqual(@as(usize, 1), pool.activeWorkers());
    pool.setActiveWorkersClamped(99);
    try std.testing.expectEqual(@as(usize, 4), pool.activeWorkers());
}
