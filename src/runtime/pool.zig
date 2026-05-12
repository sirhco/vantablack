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

    /// Run `job_fn(worker_id, n_workers, ctx)` once per worker (0..n_workers).
    /// Worker 0 runs on the calling thread.
    pub fn dispatch(self: *ThreadPool, job_fn: JobFn, ctx: *anyopaque) void {
        if (self.n_workers == 1) {
            job_fn(0, 1, ctx);
            return;
        }
        var job: Job = .{ .fn_ptr = job_fn, .ctx = ctx };
        self.job.store(&job, .release);
        self.completed.store(0, .release);
        _ = self.epoch.fetchAdd(1, .release);

        // Main does worker 0.
        job_fn(0, self.n_workers, ctx);

        // Wait for n_workers-1 others.
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

    /// Throttle hook for thermal / memory-pressure callers. Currently a
    /// no-op placeholder — true mid-flight worker count adjustment is
    /// landing with the pressure-hooks task. Documented here so callers
    /// can wire the API now without a future refactor.
    pub fn setActiveWorkersClamped(self: *ThreadPool, target: usize) void {
        _ = self;
        _ = target;
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
        job.fn_ptr(worker_id, pool.n_workers, job.ctx);
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
