//! Public module index for the `vantablack` package.

pub const mapper = @import("core/mapper.zig");
pub const parser = @import("core/parser.zig");
pub const kernels = @import("kernels/comptime_gen.zig");
pub const simd = @import("kernels/simd.zig");
pub const math = @import("kernels/math.zig");
pub const kv_cache = @import("runtime/kv_cache.zig");
pub const model = @import("runtime/model.zig");
pub const forward = @import("runtime/forward.zig");
pub const tokenizer = @import("runtime/tokenizer.zig");
pub const sampler = @import("runtime/sampler.zig");

pub const Tokenizer = tokenizer.Tokenizer;
pub const Sampler = sampler.Sampler;
pub const SamplerConfig = sampler.Config;

pub const ModelMapper = mapper.ModelMapper;
pub const Catalog = parser.Catalog;
pub const TensorDesc = parser.TensorDesc;
pub const GgmlType = parser.GgmlType;
pub const MetaValue = parser.MetaValue;
pub const MetaKv = parser.MetaKv;

pub const QuantType = kernels.QuantType;
pub const Kernel = kernels.Kernel;
pub const dispatch = kernels.dispatch;

pub const KvCache = kv_cache.KvCache;
pub const LlamaConfig = model.LlamaConfig;
pub const Model = model.Model;
pub const State = forward.State;
pub const step = forward.step;

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
