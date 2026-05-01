//! Llama-architecture model binding.
//!
//! Reads `llama.*` metadata keys to build a typed config, then resolves the
//! per-layer weight tensors from the GGUF catalog into raw byte slices that
//! point directly into the mmap'd region. No weight bytes are copied.

const std = @import("std");
const Allocator = std.mem.Allocator;

const parser = @import("../core/parser.zig");
const mapper_mod = @import("../core/mapper.zig");
const ModelMapper = mapper_mod.ModelMapper;
const kernels = @import("../kernels/comptime_gen.zig");
const mlx_kernels = @import("../kernels/mlx.zig");
const hf_config_mod = @import("../core/hf_config.zig");
const hf_loader_mod = @import("../core/hf_loader.zig");
const safetensors = @import("../core/safetensors.zig");

pub const ConfigError = error{
    MissingMetadata,
    UnsupportedArchitecture,
    UnsupportedWeightType,
};

pub const ModelError = error{
    MissingTensor,
    UnexpectedTensorShape,
} || ConfigError || mapper_mod.TensorSliceError || Allocator.Error;

pub const LlamaConfig = struct {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_seq: usize,
    rope_base: f32 = 10000.0,
    rms_eps: f32 = 1e-5,
    /// True for HF / MLX checkpoints (neox / "half-rotation" RoPE convention);
    /// false for GGUF / llama.cpp (interleaved-pair convention). The two
    /// produce identical math after a head-dim permutation of weights, but
    /// since vantablack consumes both formats unchanged we apply the
    /// matching rotation at runtime.
    rope_half: bool = false,

    pub fn fromCatalog(catalog: parser.Catalog) ConfigError!LlamaConfig {
        const arch_v = findValue(catalog, "general.architecture") orelse return error.MissingMetadata;
        const arch = switch (arch_v) {
            .string => |s| s,
            else => return error.MissingMetadata,
        };
        if (!std.mem.eql(u8, arch, "llama")) return error.UnsupportedArchitecture;

        const dim = try metaUsize(catalog, "llama.embedding_length");
        const n_layers = try metaUsize(catalog, "llama.block_count");
        const n_heads = try metaUsize(catalog, "llama.attention.head_count");
        const ffn_dim = try metaUsize(catalog, "llama.feed_forward_length");
        const max_seq = try metaUsize(catalog, "llama.context_length");
        const n_kv_heads = metaUsize(catalog, "llama.attention.head_count_kv") catch n_heads;
        const rope_base = metaF32(catalog, "llama.rope.freq_base") catch 10000.0;
        const rms_eps = metaF32(catalog, "llama.attention.layer_norm_rms_epsilon") catch 1e-5;

        // Vocab size: prefer llama.vocab_size; fall back to tokenizer array len.
        const vocab_size = blk: {
            if (metaUsize(catalog, "llama.vocab_size")) |v| break :blk v else |_| {}
            if (findValue(catalog, "tokenizer.ggml.tokens")) |v| switch (v) {
                .array => |arr| break :blk @as(usize, @intCast(arr.count)),
                else => {},
            };
            return error.MissingMetadata;
        };

        if (n_heads == 0) return error.MissingMetadata;
        const head_dim = dim / n_heads;

        return .{
            .dim = dim,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .ffn_dim = ffn_dim,
            .vocab_size = vocab_size,
            .max_seq = max_seq,
            .rope_base = rope_base,
            .rms_eps = rms_eps,
        };
    }
};

/// MLX-style multi-buffer tensor extras. Present when `quant == .mlx_q4`.
pub const MlxAux = struct {
    scales: []const u8,
    biases: []const u8,
    bits: u32,
    group_size: u32,
    scale_dtype: mlx_kernels.ScaleDtype,
};

pub const TypedTensor = struct {
    /// Primary weight buffer. For MLX-Q4 this is the packed `weight` (U32);
    /// for everything else it's the dense / dequant-able payload.
    bytes: []const u8,
    quant: kernels.QuantType,
    /// Source-format raw type tag. Only meaningful for GGUF tensors; HF
    /// tensors set this to `.f32` as a placeholder. `quant` is what consumers
    /// should actually branch on.
    ggml_type: parser.GgmlType = .f32,
    mlx: ?MlxAux = null,
};

pub const LayerWeights = struct {
    attn_norm: TypedTensor,
    attn_q: TypedTensor,
    attn_k: TypedTensor,
    attn_v: TypedTensor,
    attn_o: TypedTensor,
    ffn_norm: TypedTensor,
    ffn_gate: TypedTensor,
    ffn_up: TypedTensor,
    ffn_down: TypedTensor,
};

pub const Model = struct {
    config: LlamaConfig,
    token_embd: TypedTensor,
    output_norm: TypedTensor,
    output_w: TypedTensor,
    layers: []LayerWeights,

    pub fn init(allocator: Allocator, m: *const ModelMapper) ModelError!Model {
        const config = try LlamaConfig.fromCatalog(m.catalog);

        const layers = try allocator.alloc(LayerWeights, config.n_layers);
        errdefer allocator.free(layers);

        var name_buf: [64]u8 = undefined;
        for (layers, 0..) |*lw, i| {
            lw.* = .{
                .attn_norm = try fetch(m, fmt(&name_buf, "blk.{d}.attn_norm.weight", .{i})),
                .attn_q = try fetch(m, fmt(&name_buf, "blk.{d}.attn_q.weight", .{i})),
                .attn_k = try fetch(m, fmt(&name_buf, "blk.{d}.attn_k.weight", .{i})),
                .attn_v = try fetch(m, fmt(&name_buf, "blk.{d}.attn_v.weight", .{i})),
                .attn_o = try fetch(m, fmt(&name_buf, "blk.{d}.attn_output.weight", .{i})),
                .ffn_norm = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_norm.weight", .{i})),
                .ffn_gate = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_gate.weight", .{i})),
                .ffn_up = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_up.weight", .{i})),
                .ffn_down = try fetch(m, fmt(&name_buf, "blk.{d}.ffn_down.weight", .{i})),
            };
        }

        const token_embd = try fetch(m, "token_embd.weight");
        const output_norm = try fetch(m, "output_norm.weight");
        // Some llamas tie output to token_embd (no `output.weight`); fall back.
        const output_w = fetch(m, "output.weight") catch token_embd;

        return .{
            .config = config,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .output_w = output_w,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Model, allocator: Allocator) void {
        allocator.free(self.layers);
        self.* = undefined;
    }

    /// Build a `Model` from a HuggingFace / MLX directory bundle. Tensor
    /// name remapping mirrors the mlx_lm + transformers conventions:
    ///
    ///   model.embed_tokens.weight                       → token_embd
    ///   model.norm.weight                                → output_norm
    ///   lm_head.weight (or tied to embed_tokens)         → output
    ///   model.layers.{i}.input_layernorm.weight          → attn_norm
    ///   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight → attn_{q,k,v,o}
    ///   model.layers.{i}.post_attention_layernorm.weight → ffn_norm
    ///   model.layers.{i}.mlp.{gate,up,down}_proj.weight  → ffn_{gate,up,down}
    ///
    /// MLX-quantized linears come as a 3-tensor group (`weight`, `scales`,
    /// `biases`); they're folded into a single `TypedTensor` with the
    /// scales/biases routed via the `mlx` aux pointer.
    pub fn initFromHf(
        allocator: Allocator,
        bundle: *const hf_loader_mod.HfBundle,
        cfg: hf_config_mod.HfConfig,
    ) ModelError!Model {
        const head_dim = cfg.head_dim orelse (cfg.hidden_size / cfg.num_attention_heads);

        const llama_cfg: LlamaConfig = .{
            .dim = cfg.hidden_size,
            .n_layers = cfg.num_hidden_layers,
            .n_heads = cfg.num_attention_heads,
            .n_kv_heads = cfg.num_key_value_heads,
            .head_dim = head_dim,
            .ffn_dim = cfg.intermediate_size,
            .vocab_size = cfg.vocab_size,
            .max_seq = cfg.max_position_embeddings,
            .rope_base = cfg.rope_theta,
            .rms_eps = cfg.rms_norm_eps,
            .rope_half = true,
        };

        const layers = try allocator.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer allocator.free(layers);

        var name_buf: [128]u8 = undefined;
        for (layers, 0..) |*lw, i| {
            lw.* = .{
                .attn_norm = try fetchHfNorm(bundle, fmt(&name_buf, "model.layers.{d}.input_layernorm.weight", .{i})),
                .attn_q = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.self_attn.q_proj", .{i})),
                .attn_k = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.self_attn.k_proj", .{i})),
                .attn_v = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.self_attn.v_proj", .{i})),
                .attn_o = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.self_attn.o_proj", .{i})),
                .ffn_norm = try fetchHfNorm(bundle, fmt(&name_buf, "model.layers.{d}.post_attention_layernorm.weight", .{i})),
                .ffn_gate = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.mlp.gate_proj", .{i})),
                .ffn_up = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.mlp.up_proj", .{i})),
                .ffn_down = try fetchHfLinear(bundle, cfg, fmt(&name_buf, "model.layers.{d}.mlp.down_proj", .{i})),
            };
        }

        // Embedding may be quantized too in some MLX variants.
        const token_embd = try fetchHfEmbedding(bundle, cfg, "model.embed_tokens");
        const output_norm = try fetchHfNorm(bundle, "model.norm.weight");
        const output_w = if (cfg.tie_word_embeddings)
            token_embd
        else
            fetchHfLinear(bundle, cfg, "lm_head") catch token_embd;

        return .{
            .config = llama_cfg,
            .token_embd = token_embd,
            .output_norm = output_norm,
            .output_w = output_w,
            .layers = layers,
        };
    }
};

fn quantOfHfDtype(d: safetensors.Dtype) ConfigError!kernels.QuantType {
    return switch (d) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        else => error.UnsupportedWeightType,
    };
}

fn fetchHfNorm(bundle: *const hf_loader_mod.HfBundle, name: []const u8) ModelError!TypedTensor {
    const loc = bundle.find(name) orelse return error.MissingTensor;
    const bytes = bundle.tensorBytes(name) orelse return error.MissingTensor;
    const q = try quantOfHfDtype(loc.desc.dtype);
    return .{ .bytes = bytes, .quant = q };
}

/// Fetch a (possibly MLX-quantized) linear-layer weight under the HF base
/// name `<base>` (no `.weight` suffix). Tries:
///   1. `<base>.weight` + `<base>.scales` + `<base>.biases` → MLX-Q4 if all
///      three exist and `cfg.quantization` says bits == 4.
///   2. `<base>.weight` alone → plain dense (F32 / F16 / BF16).
fn fetchHfLinear(
    bundle: *const hf_loader_mod.HfBundle,
    cfg: hf_config_mod.HfConfig,
    base: []const u8,
) ModelError!TypedTensor {
    var name_buf: [192]u8 = undefined;
    const w_name = std.fmt.bufPrint(&name_buf, "{s}.weight", .{base}) catch return error.UnsupportedWeightType;
    const w_loc = bundle.find(w_name) orelse return error.MissingTensor;
    const w_bytes = bundle.tensorBytes(w_name) orelse return error.MissingTensor;

    if (cfg.quantization) |qcfg| if (qcfg.bits == 2 or qcfg.bits == 3 or qcfg.bits == 4 or qcfg.bits == 5 or qcfg.bits == 6 or qcfg.bits == 8) {
        // Try scales/biases. mlx_lm leaves embedding (and sometimes lm_head)
        // unquantized, so missing scales just means "fall through to plain
        // dense" rather than a hard error.
        var s_buf: [192]u8 = undefined;
        var b_buf: [192]u8 = undefined;
        const s_name = std.fmt.bufPrint(&s_buf, "{s}.scales", .{base}) catch return error.UnsupportedWeightType;
        const b_name = std.fmt.bufPrint(&b_buf, "{s}.biases", .{base}) catch return error.UnsupportedWeightType;
        if (bundle.find(s_name)) |s_loc| {
            const s_bytes = bundle.tensorBytes(s_name) orelse return error.MissingTensor;
            const b_bytes = bundle.tensorBytes(b_name) orelse return error.MissingTensor;
            const scale_dt: mlx_kernels.ScaleDtype = switch (s_loc.desc.dtype) {
                .f16 => .f16,
                .bf16 => .bf16,
                else => return error.UnsupportedWeightType,
            };
            return .{
                .bytes = w_bytes,
                .quant = .mlx_q4,
                .mlx = .{
                    .scales = s_bytes,
                    .biases = b_bytes,
                    .bits = qcfg.bits,
                    .group_size = qcfg.group_size,
                    .scale_dtype = scale_dt,
                },
            };
        }
    };

    const q = try quantOfHfDtype(w_loc.desc.dtype);
    return .{ .bytes = w_bytes, .quant = q };
}

fn fetchHfEmbedding(
    bundle: *const hf_loader_mod.HfBundle,
    cfg: hf_config_mod.HfConfig,
    base: []const u8,
) ModelError!TypedTensor {
    return fetchHfLinear(bundle, cfg, base);
}

fn fmt(buf: []u8, comptime template: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, template, args) catch unreachable;
}

fn fetch(m: *const ModelMapper, name: []const u8) ModelError!TypedTensor {
    const desc = m.catalog.find(name) orelse return error.MissingTensor;
    const bytes = try m.tensorSliceFromDesc(desc);
    const q = try quantOf(desc.ggml_type);
    return .{ .bytes = bytes, .quant = q, .ggml_type = desc.ggml_type };
}

fn findValue(catalog: parser.Catalog, key: []const u8) ?parser.MetaValue {
    for (catalog.metadata) |kv| {
        if (std.mem.eql(u8, kv.key, key)) return kv.value;
    }
    return null;
}

fn metaUsize(catalog: parser.Catalog, key: []const u8) ConfigError!usize {
    const v = findValue(catalog, key) orelse return error.MissingMetadata;
    return switch (v) {
        .u8 => |x| x,
        .u16 => |x| x,
        .u32 => |x| x,
        .u64 => |x| @intCast(x),
        .i8 => |x| @intCast(x),
        .i16 => |x| @intCast(x),
        .i32 => |x| @intCast(x),
        .i64 => |x| @intCast(x),
        else => error.MissingMetadata,
    };
}

fn metaF32(catalog: parser.Catalog, key: []const u8) ConfigError!f32 {
    const v = findValue(catalog, key) orelse return error.MissingMetadata;
    return switch (v) {
        .f32 => |x| x,
        .f64 => |x| @floatCast(x),
        else => error.MissingMetadata,
    };
}

/// Map a GGML tensor type into the dispatcher's QuantType taxonomy.
pub fn quantOf(t: parser.GgmlType) ConfigError!@import("../kernels/comptime_gen.zig").QuantType {
    return switch (t) {
        .f32 => .f32,
        .f16 => .f16,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .tq2_0, .tq1_0 => .ternary158,
        else => error.UnsupportedWeightType,
    };
}
