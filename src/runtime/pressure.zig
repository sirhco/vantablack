//! Memory-pressure and thermal-throttling signal hub.
//!
//! Mobile and embedded hosts report two kinds of stress through the OS:
//! memory pressure (about-to-OOM) and thermal state (about-to-throttle).
//! This module gives the host a small C-ABI surface to forward those
//! signals into the inference engine. Subscribed sinks — Backends,
//! KvCache, ThreadPool — decide what to do about them.
//!
//! Wire-up pattern on iOS:
//!
//!   UIApplicationDidReceiveMemoryWarningNotification
//!       → vtb_pressure_signal_memory(.warning)
//!
//!   ProcessInfo.thermalStateDidChangeNotification
//!       → vtb_pressure_signal_thermal(<mapped state>)
//!
//! The Swift host wires the bridge. This module is platform-agnostic.

const std = @import("std");

const backend_mod = @import("backend.zig");

pub const PressureLevel = backend_mod.PressureLevel;
pub const ThermalState = backend_mod.ThermalState;

pub const MemoryHandler = *const fn (ctx: *anyopaque, level: PressureLevel) void;
pub const ThermalHandler = *const fn (ctx: *anyopaque, state: ThermalState) void;

pub const Sink = struct {
    ctx: *anyopaque,
    on_memory: ?MemoryHandler = null,
    on_thermal: ?ThermalHandler = null,
};

const max_sinks: usize = 8;

/// Process-wide registry. Tiny and stack-allocated — a real mobile host
/// has on the order of 1–4 sinks (the active Backend + KvCache +
/// ThreadPool). The fixed bound keeps the API pointer-free and survives
/// `static_library + nonzero TLS` constraints on iOS.
///
/// Concurrency: `std.atomic.Mutex` is lock-free trySpin — register and
/// signal are rare enough events that contention is effectively zero,
/// but the spin loop is correct in the rare overlap.
pub const Hub = struct {
    sinks: [max_sinks]?Sink = .{null} ** max_sinks,
    count: usize = 0,
    last_memory: PressureLevel = .normal,
    last_thermal: ThermalState = .nominal,
    mu: std.atomic.Mutex = .unlocked,

    fn lock(self: *Hub) void {
        while (!self.mu.tryLock()) std.atomic.spinLoopHint();
    }

    fn unlock(self: *Hub) void {
        self.mu.unlock();
    }

    pub fn register(self: *Hub, sink: Sink) error{HubFull}!void {
        self.lock();
        defer self.unlock();
        if (self.count >= max_sinks) return error.HubFull;
        self.sinks[self.count] = sink;
        self.count += 1;
    }

    pub fn unregister(self: *Hub, ctx: *anyopaque) void {
        self.lock();
        defer self.unlock();
        var i: usize = 0;
        while (i < self.count) : (i += 1) {
            const s = self.sinks[i] orelse continue;
            if (s.ctx == ctx) {
                // Swap-remove. Order is not load-bearing.
                self.sinks[i] = self.sinks[self.count - 1];
                self.sinks[self.count - 1] = null;
                self.count -= 1;
                return;
            }
        }
    }

    pub fn signalMemory(self: *Hub, level: PressureLevel) void {
        self.lock();
        const count = self.count;
        var snapshot: [max_sinks]?Sink = .{null} ** max_sinks;
        @memcpy(snapshot[0..count], self.sinks[0..count]);
        self.last_memory = level;
        self.unlock();

        // Notify outside the lock — handlers must not deadlock the hub.
        for (snapshot[0..count]) |maybe| {
            const s = maybe orelse continue;
            if (s.on_memory) |fp| fp(s.ctx, level);
        }
    }

    pub fn signalThermal(self: *Hub, state: ThermalState) void {
        self.lock();
        const count = self.count;
        var snapshot: [max_sinks]?Sink = .{null} ** max_sinks;
        @memcpy(snapshot[0..count], self.sinks[0..count]);
        self.last_thermal = state;
        self.mu.unlock();

        for (snapshot[0..count]) |maybe| {
            const s = maybe orelse continue;
            if (s.on_thermal) |fp| fp(s.ctx, state);
        }
    }

    pub fn lastMemory(self: *Hub) PressureLevel {
        self.lock();
        defer self.unlock();
        return self.last_memory;
    }

    pub fn lastThermal(self: *Hub) ThermalState {
        self.lock();
        defer self.unlock();
        return self.last_thermal;
    }
};

/// Convenience: build a Sink that fans straight into a Backend's vtable.
/// Lets callers register a Backend without writing a bespoke adapter.
pub fn backendSink(be: *backend_mod.Backend) Sink {
    return .{
        .ctx = be,
        .on_memory = backendOnMemory,
        .on_thermal = backendOnThermal,
    };
}

fn backendOnMemory(ctx: *anyopaque, level: PressureLevel) void {
    const be: *backend_mod.Backend = @ptrCast(@alignCast(ctx));
    be.onPressure(level);
}

fn backendOnThermal(ctx: *anyopaque, state: ThermalState) void {
    const be: *backend_mod.Backend = @ptrCast(@alignCast(ctx));
    be.onThermal(state);
}

// -- tests ----------------------------------------------------------------

const Counter = struct {
    memory_calls: usize = 0,
    thermal_calls: usize = 0,
    last_memory: PressureLevel = .normal,
    last_thermal: ThermalState = .nominal,

    fn onMemory(ctx: *anyopaque, level: PressureLevel) void {
        const self: *Counter = @ptrCast(@alignCast(ctx));
        self.memory_calls += 1;
        self.last_memory = level;
    }
    fn onThermal(ctx: *anyopaque, state: ThermalState) void {
        const self: *Counter = @ptrCast(@alignCast(ctx));
        self.thermal_calls += 1;
        self.last_thermal = state;
    }
};

test "Hub dispatches memory + thermal to registered sink" {
    var hub: Hub = .{};
    var c: Counter = .{};
    try hub.register(.{
        .ctx = @ptrCast(&c),
        .on_memory = Counter.onMemory,
        .on_thermal = Counter.onThermal,
    });

    hub.signalMemory(.warning);
    hub.signalThermal(.serious);

    try std.testing.expectEqual(@as(usize, 1), c.memory_calls);
    try std.testing.expectEqual(@as(usize, 1), c.thermal_calls);
    try std.testing.expectEqual(PressureLevel.warning, c.last_memory);
    try std.testing.expectEqual(ThermalState.serious, c.last_thermal);
    try std.testing.expectEqual(PressureLevel.warning, hub.lastMemory());
    try std.testing.expectEqual(ThermalState.serious, hub.lastThermal());
}

test "Hub.unregister stops further notifications" {
    var hub: Hub = .{};
    var c: Counter = .{};
    try hub.register(.{
        .ctx = @ptrCast(&c),
        .on_memory = Counter.onMemory,
    });
    hub.signalMemory(.warning);
    hub.unregister(@ptrCast(&c));
    hub.signalMemory(.critical);
    try std.testing.expectEqual(@as(usize, 1), c.memory_calls);
}

test "Hub.register HubFull when bound exceeded" {
    var hub: Hub = .{};
    var counters: [max_sinks]Counter = .{Counter{}} ** max_sinks;
    for (&counters) |*c| {
        try hub.register(.{ .ctx = @ptrCast(c), .on_memory = Counter.onMemory });
    }
    var overflow_c: Counter = .{};
    const rc = hub.register(.{ .ctx = @ptrCast(&overflow_c), .on_memory = Counter.onMemory });
    try std.testing.expectError(error.HubFull, rc);
}
