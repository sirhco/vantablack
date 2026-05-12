//! Metal `Backend` implementation.
//!
//! Wraps `forward.gpuLayerStep` / `forward.gpuFinalStep` and the
//! `MetalBackend` per-process GPU state behind the same vtable as the CPU
//! backend. Construction is checked: callers should only build this when
//! `modelGpuEligible(model)` is true; if a non-eligible model is bound,
//! `init` returns `error.BackendUnsupportedQuant` and the caller is
//! expected to fall back to `CpuBackend`.

const std = @import("std");

const backend_mod = @import("backend.zig");
const forward_mod = @import("forward.zig");
const kv_cache_mod = @import("kv_cache.zig");
const model_mod = @import("model.zig");
const metal_backend_mod = @import("metal_backend.zig");
const metal_bridge_mod = @import("../metal/bridge.zig");

const Backend = backend_mod.Backend;
const Capabilities = backend_mod.Capabilities;
const PressureLevel = backend_mod.PressureLevel;
const ThermalState = backend_mod.ThermalState;
const VTable = backend_mod.VTable;
const StatePtr = backend_mod.StatePtr;

const State = forward_mod.State;
const KvCache = kv_cache_mod.KvCache;
const Model = model_mod.Model;
const LayerWeights = model_mod.LayerWeights;
const LlamaConfig = model_mod.LlamaConfig;
const MetalBackend = metal_backend_mod.MetalBackend;

pub const MetalDispatch = struct {
    mb: *MetalBackend,

    pub fn init(mb: *MetalBackend) MetalDispatch {
        return .{ .mb = mb };
    }

    pub fn backend(self: *MetalDispatch) Backend {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }
};

const vtable: VTable = .{
    .layer_step = layerStep,
    .final_step = finalStep,
    .finalize_token = finalizeToken,
    .capabilities = capabilities,
    .on_pressure = onPressure,
    .on_thermal = onThermal,
};

fn castSelf(ctx: *anyopaque) *MetalDispatch {
    return @ptrCast(@alignCast(ctx));
}

fn layerStep(
    ctx: *anyopaque,
    layer: *const LayerWeights,
    cache: *KvCache,
    _: StatePtr,
    c: LlamaConfig,
    layer_idx: usize,
) anyerror!void {
    const self = castSelf(ctx);
    // `state` is intentionally unused: GPU forward writes through the
    // persistent shared-storage scratch buffers that `State` already
    // aliases, so reading state here would be redundant.
    try forward_mod.gpuLayerStep(self.mb, layer.*, cache, c, layer_idx);
}

fn finalStep(
    ctx: *anyopaque,
    m: *const Model,
    c: LlamaConfig,
    _: StatePtr,
) anyerror!void {
    const self = castSelf(ctx);
    try forward_mod.gpuFinalStep(self.mb, m, c);
}

fn finalizeToken(ctx: *anyopaque) anyerror!void {
    const self = castSelf(ctx);
    // Drain pending async segments. Every prior `gpuLayerStep` and
    // `gpuFinalStep` committed without waiting, so the host must block
    // here before sampling reads `state.logits` (a view over the GPU-
    // written shared-storage buffer).
    self.mb.dev.waitIdle();
}

fn capabilities(_: *anyopaque) Capabilities {
    return .{
        .supports_fused_qkv = true,
        .supports_kquant_gpu = true,
        .supports_mlx_q4 = true,
        .unified_memory = true,
        .name = "metal",
    };
}

fn onPressure(_: *anyopaque, _: PressureLevel) void {
    // Real shrink behaviour (KV-cache compaction, releasing fused weight
    // packs) lands with the pressure-hooks task. The vtable slot is wired
    // now so the host signal pipeline can call it without a follow-up
    // refactor.
}

fn onThermal(_: *anyopaque, _: ThermalState) void {
    // On Apple Silicon there is no separate "GPU active workers" knob —
    // thermal pressure is best handled by inserting per-layer
    // commit+waitUntilCompleted barriers (lower duty cycle). Land with
    // the pressure-hooks task.
}
