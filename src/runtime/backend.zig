//! Backend abstraction.
//!
//! Forward-pass primitives (per-layer matmul/attn/ffn + final logits) are
//! invoked through a vtable. The CPU and Metal impls live in
//! `backend_cpu.zig` and `backend_metal.zig`. Adding a new backend (Vulkan,
//! CoreML, NPU) is a matter of writing one file that satisfies this vtable
//! — no surgery in `forward.zig`.
//!
//! Granularity is "one layer" + "final projection," not per-op. That keeps
//! the abstraction thin and matches the existing `gpuLayerStep` /
//! `cpuLayerStep` shape. A finer per-op vtable is a future option if a
//! backend needs to share state across the layer boundary in a way the
//! coarse interface doesn't expose.

const std = @import("std");

const model_mod = @import("model.zig");
const kv_cache_mod = @import("kv_cache.zig");

const Model = model_mod.Model;
const Layer = model_mod.LayerWeights;
const LlamaConfig = model_mod.LlamaConfig;
const KvCache = kv_cache_mod.KvCache;

/// Opaque pointer to `forward.State`. The vtable doesn't depend on the
/// concrete shape; each backend casts it back via `@ptrCast`. This avoids
/// a circular import between `backend.zig` and `forward.zig`.
pub const StatePtr = *anyopaque;

pub const PressureLevel = enum(u8) {
    normal = 0,
    warning = 1,
    critical = 2,
};

pub const ThermalState = enum(u8) {
    nominal = 0,
    fair = 1,
    serious = 2,
    critical = 3,
};

pub const Capabilities = struct {
    /// True if Q/K/V and gate/up projections can be fused into one matmul.
    supports_fused_qkv: bool = false,
    /// True if Q4_K / Q5_K / Q6_K matmul kernels are available.
    supports_kquant_gpu: bool = false,
    /// True if MLX-format 4-bit affine block-quant matmul runs on this backend.
    supports_mlx_q4: bool = false,
    /// True when host and device share the same physical memory pages
    /// (Apple Silicon, iOS, integrated GPUs). False for discrete GPUs.
    unified_memory: bool = true,
    /// Backend-reported display name. Used in capability logs.
    name: []const u8 = "unknown",
};

pub const Error = error{
    BackendInitFailed,
    BackendDispatchFailed,
    BackendOutOfMemory,
    BackendUnsupportedQuant,
};

/// One per-token forward pass through the model is N `layer_step` calls
/// followed by one `final_step` and one `finalize_token`. Backends are
/// allowed to batch GPU work and only flush in `finalize_token`.
pub const VTable = struct {
    /// Run one transformer block. Reads state.x, writes state.x.
    layer_step: *const fn (
        ctx: *anyopaque,
        layer: *const Layer,
        cache: *KvCache,
        state: StatePtr,
        c: LlamaConfig,
        layer_idx: usize,
    ) anyerror!void,

    /// Run output_norm + LM head. Reads state.x, writes state.logits.
    final_step: *const fn (
        ctx: *anyopaque,
        m: *const Model,
        c: LlamaConfig,
        state: StatePtr,
    ) anyerror!void,

    /// Drain pending async work and make all writes visible to the host.
    /// CPU backends are no-ops; GPU backends wait_idle here.
    finalize_token: *const fn (ctx: *anyopaque) anyerror!void,

    capabilities: *const fn (ctx: *anyopaque) Capabilities,

    /// Notified by `runtime/pressure.zig` when the host OS reports memory
    /// pressure or thermal throttling. Backend reduces resource use
    /// (shrink scratch, reduce worker count, etc.) opportunistically.
    on_pressure: *const fn (ctx: *anyopaque, level: PressureLevel) void = noopPressure,
    on_thermal: *const fn (ctx: *anyopaque, state: ThermalState) void = noopThermal,
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub fn layerStep(
        self: Backend,
        layer: *const Layer,
        cache: *KvCache,
        state: StatePtr,
        c: LlamaConfig,
        layer_idx: usize,
    ) anyerror!void {
        return self.vtable.layer_step(self.ptr, layer, cache, state, c, layer_idx);
    }

    pub fn finalStep(self: Backend, m: *const Model, c: LlamaConfig, state: StatePtr) anyerror!void {
        return self.vtable.final_step(self.ptr, m, c, state);
    }

    pub fn finalizeToken(self: Backend) anyerror!void {
        return self.vtable.finalize_token(self.ptr);
    }

    pub fn capabilities(self: Backend) Capabilities {
        return self.vtable.capabilities(self.ptr);
    }

    pub fn onPressure(self: Backend, level: PressureLevel) void {
        self.vtable.on_pressure(self.ptr, level);
    }

    pub fn onThermal(self: Backend, state: ThermalState) void {
        self.vtable.on_thermal(self.ptr, state);
    }
};

fn noopPressure(_: *anyopaque, _: PressureLevel) void {}
fn noopThermal(_: *anyopaque, _: ThermalState) void {}

test "Backend struct is callable" {
    // Compile-time check that the vtable is well-formed.
    const T = struct {
        var dummy_caps: Capabilities = .{ .name = "test" };
        fn capabilities(_: *anyopaque) Capabilities {
            return dummy_caps;
        }
        fn layerStep(_: *anyopaque, _: *const Layer, _: *KvCache, _: StatePtr, _: LlamaConfig, _: usize) anyerror!void {}
        fn finalStep(_: *anyopaque, _: *const Model, _: LlamaConfig, _: StatePtr) anyerror!void {}
        fn finalizeToken(_: *anyopaque) anyerror!void {}
    };
    const vt: VTable = .{
        .layer_step = T.layerStep,
        .final_step = T.finalStep,
        .finalize_token = T.finalizeToken,
        .capabilities = T.capabilities,
    };
    var dummy: u8 = 0;
    const be: Backend = .{ .ptr = @ptrCast(&dummy), .vtable = &vt };
    const caps = be.capabilities();
    try std.testing.expectEqualStrings("test", caps.name);
}
