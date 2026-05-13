const std = @import("std");
const Io = std.Io;

const vantablack = @import("vantablack");

const usage =
    \\usage:
    \\  vantablack <model.gguf>                              print tensor catalog
    \\  vantablack <model.gguf> generate <n> <id> [id...]    generate n tokens
    \\  vantablack <model.gguf> prompt <n> <text>            encode text, generate n more tokens
    \\  vantablack <model.gguf> chat <n> <user-message>      wrap in TinyLlama-Chat (zephyr) template
    \\  vantablack <model.gguf> serve [--host H] [--port P]  Ollama-compatible HTTP server (default 127.0.0.1:11434)
    \\  vantablack inspect <model.litertlm>                 print sections + metadata of a LiteRT-LM bundle
    \\  vantablack inspect-section <m.litertlm> <idx>       deep-dump tensors + operator graph of one TFLite section
    \\  vantablack scan-layers <model.litertlm>             classify decoder weights by layer + role (Phase 19c foundation)
    \\  vantablack trace-tensor <m.litertlm> <sec> <t_idx>  find ops that produce/consume a given TFLite tensor
    \\  vantablack gemma4-config <m.litertlm>               load a Gemma 4 .litertlm into Gemma4Model + print inferred config
    \\  vantablack gemma4-embed <m.litertlm> <token_id>     lookup embedding for token_id (INT2 dequant smoke test)
    \\  vantablack gemma4-step  <m.litertlm> <token_id>     embed + layer-0 Q projection forward smoke test
    \\  vantablack tokenize-litertlm <m.litertlm> <text>    encode text using HF tokenizer embedded in a .litertlm
    \\
    \\generate / prompt / chat accept sampler flags BEFORE the n value:
    \\  --temp <f>     temperature (0.0 = greedy)
    \\  --top-k <n>    top-k filter (0 = disabled)
    \\  --top-p <f>    nucleus filter (0 = disabled)
    \\  --seed <u64>   RNG seed (default 0)
    \\  --system <s>   system prompt for chat mode (default "You are a helpful assistant.")
    \\  --threads <n>  worker thread count (0 = autodetect, default ~half of cpu cores)
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_file_writer: Io.File.Writer = .init(.stderr(), io, &stderr_buffer);
    const err = &stderr_file_writer.interface;

    if (args.len < 2) {
        try err.writeAll(usage);
        try err.flush();
        return error.MissingPath;
    }

    // tokenize-litertlm <model.litertlm> <text>: extract the HF tokenizer
    // from a .litertlm bundle (zlib-decompressed HF_Tokenizer_Zlib
    // section), encode `text`, print token IDs. Lets a host validate
    // tokenization from a litertlm bundle without needing the full
    // forward path.
    if (std.mem.eql(u8, args[1], "tokenize-litertlm")) {
        if (args.len < 4) {
            try err.writeAll("usage: vantablack tokenize-litertlm <model.litertlm> <text>\n");
            try err.flush();
            return error.MissingArgs;
        }
        try runTokenizeLitertlm(gpa, arena, io, args[2], args[3], out, err);
        return;
    }

    // gemma4-step <model.litertlm> <token_id>: smoke test forward
    // progression — embed lookup + layer-0 Q projection. Prints
    // summary of each stage. Useful sanity check before wiring the
    // full per-token pipeline.
    if (std.mem.eql(u8, args[1], "gemma4-step")) {
        if (args.len < 4) {
            try err.writeAll("usage: vantablack gemma4-step <model.litertlm> <token_id>\n");
            try err.flush();
            return error.MissingArgs;
        }
        const tid = try std.fmt.parseInt(u32, args[3], 10);
        try runGemma4Step(gpa, arena, io, args[2], tid, out, err);
        return;
    }

    // gemma4-embed <model.litertlm> <token_id>: lookup the embedding
    // for `token_id`, print first 8 + last 4 components. Smoke-tests
    // the INT2 dequant path end-to-end against the real model.
    if (std.mem.eql(u8, args[1], "gemma4-embed")) {
        if (args.len < 4) {
            try err.writeAll("usage: vantablack gemma4-embed <model.litertlm> <token_id>\n");
            try err.flush();
            return error.MissingArgs;
        }
        const tid = try std.fmt.parseInt(u32, args[3], 10);
        try runGemma4Embed(gpa, arena, io, args[2], tid, out, err);
        return;
    }

    // gemma4-config <model.litertlm>: load a Gemma 4 .litertlm into the
    // Gemma4Model struct + print the inferred config. Proves the
    // bundle → Model binding works end-to-end without running
    // inference yet (Phase 19c integration test).
    if (std.mem.eql(u8, args[1], "gemma4-config")) {
        if (args.len < 3) {
            try err.writeAll("usage: vantablack gemma4-config <model.litertlm>\n");
            try err.flush();
            return error.MissingArgs;
        }
        try runGemma4Config(gpa, arena, io, args[2], out, err);
        return;
    }

    // trace-tensor <model.litertlm> <section_idx> <selector>: walks
    // the operator graph and prints every op that produces or consumes
    // the given tensor. Selector is either a numeric tensor index OR
    // a name substring (first match wins). Surfaces data-flow
    // neighbors so we can hand-verify e.g. KV-sharing wiring (which
    // layer's K cache does layer_20's attention read from?).
    if (std.mem.eql(u8, args[1], "trace-tensor")) {
        if (args.len < 5) {
            try err.writeAll("usage: vantablack trace-tensor <model.litertlm> <section_idx> <tensor_idx|name_substring>\n");
            try err.flush();
            return error.MissingArgs;
        }
        const sec_idx = try std.fmt.parseInt(usize, args[3], 10);
        try runTraceTensor(gpa, arena, io, args[2], sec_idx, args[4], out, err);
        return;
    }

    // scan-layers <model.litertlm>: name-pattern map of decoder weights.
    // Auto-selects the largest TFLiteModel section that contains a
    // `layer_N/` tensor prefix, groups weights by layer, classifies
    // each by role (mlp.gate/up/down, attn.o, qkv, ple.gate, ple.proj,
    // norms). Foundation for Phase 19c.
    if (std.mem.eql(u8, args[1], "scan-layers")) {
        if (args.len < 3) {
            try err.writeAll("usage: vantablack scan-layers <model.litertlm>\n");
            try err.flush();
            return error.MissingArgs;
        }
        try runScanLayers(gpa, arena, io, args[2], out, err);
        return;
    }

    // inspect-section <model.litertlm> <idx>: deep dump of a single
    // TFLite section. Lists every weight tensor (sorted by size) and
    // every operator with its input/output tensor indices. Lets a dev
    // hand-trace the forward graph to inform Phase 19c tensor mapping.
    if (std.mem.eql(u8, args[1], "inspect-section")) {
        if (args.len < 4) {
            try err.writeAll("usage: vantablack inspect-section <model.litertlm> <section_idx>\n");
            try err.flush();
            return error.MissingArgs;
        }
        const idx = try std.fmt.parseInt(usize, args[3], 10);
        try runInspectSection(gpa, arena, io, args[2], idx, out, err);
        return;
    }

    // inspect <model.litertlm>: parse Google's LiteRT-LM container header,
    // list sections + metadata. Inference is not yet supported on
    // .litertlm files — the TFLite tensor section parser is a follow-up
    // (Phase B/C). This subcommand lets callers see what's in a bundle
    // before deciding whether to convert to GGUF/MLX.
    if (std.mem.eql(u8, args[1], "inspect")) {
        if (args.len < 3) {
            try err.writeAll("usage: vantablack inspect <model.litertlm>\n");
            try err.flush();
            return error.MissingArgs;
        }
        try runInspect(gpa, arena, io, args[2], out, err);
        return;
    }

    // tokenize <tokenizer.json> <text>: load HF tokenizer.json, run encode,
    // print token IDs as space-separated ints. Useful for fixture parity
    // checks against HF `tokenizers` without needing a full model load.
    if (std.mem.eql(u8, args[1], "tokenize")) {
        if (args.len < 4) {
            try err.writeAll("usage: vantablack tokenize <tokenizer.json> <text>\n");
            try err.flush();
            return error.MissingArgs;
        }
        const tj_bytes = blk: {
            const tj_abs = if (std.fs.path.isAbsolute(args[2]))
                try arena.dupe(u8, args[2])
            else cwd_blk: {
                const cwd = try std.process.currentPathAlloc(io, arena);
                break :cwd_blk try std.fs.path.resolve(arena, &.{ cwd, args[2] });
            };
            var f = try Io.Dir.cwd().openFile(io, tj_abs, .{});
            defer f.close(io);
            const st = try f.stat(io);
            const buf = try arena.alloc(u8, st.size);
            _ = try f.readPositionalAll(io, buf, 0);
            break :blk buf;
        };
        var tok = try vantablack.Tokenizer.initFromHfJson(gpa, tj_bytes);
        defer tok.deinitOwnedPieces(gpa);
        try out.print("flavor={s} byte_split={s} vocab={d}\n", .{ @tagName(tok.flavor), @tagName(tok.byte_split), tok.pieces.len });
        const ids = try tok.encode(arena, args[3], false);
        for (ids, 0..) |id, i| {
            if (i > 0) try out.writeByte(' ');
            try out.print("{d}", .{id});
        }
        try out.writeByte('\n');
        try out.flush();
        return;
    }

    const input_path = args[1];
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd_path = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd_path, input_path });
    };

    // If the path points at a directory, treat it as a HuggingFace / MLX
    // model bundle (config.json + weights.NN.safetensors + tokenizer.json).
    // Otherwise fall through to the GGUF mmap loader. Directory mode supports
    // `prompt` and `chat` subcommands; `generate` (raw token IDs) and
    // `serve` remain GGUF-only for now.
    if (isDirectory(io, abs_path)) {
        try runHfCli(gpa, arena, io, abs_path, args, out, err);
        return;
    }

    var mapper = try vantablack.ModelMapper.init(gpa, io, abs_path);
    defer mapper.deinit();

    if (args.len >= 3 and std.mem.eql(u8, args[2], "serve")) {
        var host: []const u8 = "127.0.0.1";
        var port: u16 = 11434;
        var threads: usize = 0;
        var idx: usize = 3;
        while (idx < args.len) : (idx += 1) {
            const a = args[idx];
            if (std.mem.eql(u8, a, "--host")) {
                idx += 1;
                host = args[idx];
            } else if (std.mem.eql(u8, a, "--port")) {
                idx += 1;
                port = try std.fmt.parseInt(u16, args[idx], 10);
            } else if (std.mem.eql(u8, a, "--threads")) {
                idx += 1;
                threads = try std.fmt.parseInt(usize, args[idx], 10);
            } else break;
        }
        const model_name = std.fs.path.basename(abs_path);
        var srv = try vantablack.Server.init(gpa, io, &mapper, .{
            .host = host,
            .port = port,
            .model_name = model_name,
            .model_path = abs_path,
            .threads = threads,
        });
        defer srv.deinit();
        try srv.run();
        return;
    } else if (args.len >= 3 and (std.mem.eql(u8, args[2], "generate") or
        std.mem.eql(u8, args[2], "prompt") or
        std.mem.eql(u8, args[2], "chat")))
    {
        const Mode = enum { generate, prompt, chat };
        const mode: Mode = if (std.mem.eql(u8, args[2], "generate"))
            .generate
        else if (std.mem.eql(u8, args[2], "prompt"))
            .prompt
        else
            .chat;

        var cfg: vantablack.SamplerConfig = .{};
        var system_prompt: []const u8 = "You are a helpful assistant.";
        var threads: usize = 0;
        var idx: usize = 3;
        while (idx < args.len) {
            const a = args[idx];
            if (std.mem.eql(u8, a, "--temp")) {
                idx += 1;
                cfg.temperature = try std.fmt.parseFloat(f32, args[idx]);
            } else if (std.mem.eql(u8, a, "--top-k")) {
                idx += 1;
                cfg.top_k = try std.fmt.parseInt(usize, args[idx], 10);
            } else if (std.mem.eql(u8, a, "--top-p")) {
                idx += 1;
                cfg.top_p = try std.fmt.parseFloat(f32, args[idx]);
            } else if (std.mem.eql(u8, a, "--seed")) {
                idx += 1;
                cfg.seed = try std.fmt.parseInt(u64, args[idx], 10);
            } else if (std.mem.eql(u8, a, "--system")) {
                idx += 1;
                system_prompt = args[idx];
            } else if (std.mem.eql(u8, a, "--threads")) {
                idx += 1;
                threads = try std.fmt.parseInt(usize, args[idx], 10);
            } else break;
            idx += 1;
        }
        if (idx >= args.len) {
            try err.writeAll(usage);
            try err.flush();
            return error.MissingArgs;
        }
        const n_steps = try std.fmt.parseInt(u32, args[idx], 10);
        idx += 1;
        if (idx >= args.len) {
            try err.writeAll(usage);
            try err.flush();
            return error.MissingArgs;
        }

        const prompt_ids = switch (mode) {
            .generate => blk: {
                const ids = try arena.alloc(u32, args.len - idx);
                for (ids, args[idx..]) |*id, s| id.* = try std.fmt.parseInt(u32, s, 10);
                break :blk ids;
            },
            .prompt => blk: {
                var text_buf: std.ArrayList(u8) = .empty;
                defer text_buf.deinit(arena);
                for (args[idx..], 0..) |a, i| {
                    if (i != 0) try text_buf.append(arena, ' ');
                    try text_buf.appendSlice(arena, a);
                }
                var tok = try vantablack.Tokenizer.init(gpa, mapper.catalog);
                defer tok.deinit(gpa);
                const ids = try tok.encode(arena, text_buf.items, true);
                try out.print("[encoded {d} tokens]\n", .{ids.len});
                try out.flush();
                break :blk ids;
            },
            .chat => blk: {
                var user_buf: std.ArrayList(u8) = .empty;
                defer user_buf.deinit(arena);
                for (args[idx..], 0..) |a, i| {
                    if (i != 0) try user_buf.append(arena, ' ');
                    try user_buf.appendSlice(arena, a);
                }
                const wrapped = try vantablack.chat_template.formatLlamaChatSingle(
                    arena,
                    system_prompt,
                    user_buf.items,
                );
                var tok = try vantablack.Tokenizer.init(gpa, mapper.catalog);
                defer tok.deinit(gpa);
                const ids = try tok.encode(arena, wrapped, true);
                try out.print("[chat: encoded {d} tokens]\n", .{ids.len});
                try out.flush();
                break :blk ids;
            },
        };

        try generate(gpa, &mapper, n_steps, prompt_ids, cfg, threads, out);
    } else {
        try printCatalog(out, arena, mapper.catalog);
    }

    try out.flush();
}

fn generate(
    gpa: std.mem.Allocator,
    mapper: *const vantablack.ModelMapper,
    n_steps: u32,
    prompt_ids: []const u32,
    sampler_cfg: vantablack.SamplerConfig,
    threads: usize,
    out: *Io.Writer,
) !void {
    var model = try vantablack.Model.init(gpa, mapper);
    defer model.deinit(gpa);

    // Metal first so State can alias persistent shared-storage scratch.
    var maybe_metal: ?vantablack.MetalBackend = blk: {
        if (!vantablack.metal.metal_enabled) break :blk null;
        break :blk vantablack.MetalBackend.init(gpa, mapper, model.config) catch null;
    };
    defer if (maybe_metal) |*mb| mb.deinit(gpa);
    const metal_ptr: ?*vantablack.MetalBackend = if (maybe_metal != null) &maybe_metal.? else null;
    if (metal_ptr) |mb| mb.attachFusedWeights(gpa, &model) catch {};

    var state = try vantablack.forward.State.init(gpa, &model, metal_ptr);
    defer state.deinit(gpa);

    var cache = try vantablack.KvCache.init(
        gpa,
        model.config.n_layers,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.max_seq,
        metal_ptr,
    );
    defer cache.deinit(gpa);

    var tok = try vantablack.Tokenizer.init(gpa, mapper.catalog);
    defer tok.deinit(gpa);

    var sampler = try vantablack.Sampler.init(gpa, sampler_cfg, model.config.vocab_size);
    defer sampler.deinit(gpa);

    // When the GPU is doing the heavy matmuls, more CPU workers just spin
    // waiting on GPU sync. Default to 1 thread when Metal is active.
    const effective_threads = if (metal_ptr != null and threads == 0) @as(usize, 1) else threads;
    var pool = try vantablack.ThreadPool.init(gpa, effective_threads);
    defer pool.deinit(gpa);

    if (prompt_ids.len == 0) return;
    // Echo the prompt while it's being processed.
    for (prompt_ids) |id| try tok.decodeTo(out, id);
    try out.flush();
    // Prefill on CPU when there's no Metal backend — single B-wide pass
    // amortizes weight bandwidth across the whole prompt. With Metal, the
    // per-token GPU path is faster than a CPU prefill for typical prompt
    // sizes; revisit once a GPU batched matmul ships.
    if (metal_ptr == null) {
        try vantablack.prefill.prefillCpu(gpa, pool, &model, &state, &cache, prompt_ids);
    } else {
        for (prompt_ids) |id| {
            try vantablack.forward.step(&model, &state, &cache, pool, metal_ptr, id);
        }
    }

    var produced: u32 = 0;
    var next: u32 = sampler.sample(state.logits);
    while (produced < n_steps) : (produced += 1) {
        if (next == tok.eos) break;
        try tok.decodeTo(out, next);
        try out.flush();
        try vantablack.forward.step(&model, &state, &cache, pool, metal_ptr, next);
        next = sampler.sample(state.logits);
    }
    try out.writeByte('\n');
}

fn printCatalog(out: *Io.Writer, arena: std.mem.Allocator, catalog: vantablack.Catalog) !void {
    try out.print(
        "{d} tensors  alignment={d}  data_offset=0x{x}\n\n",
        .{ catalog.descs.len, catalog.alignment, catalog.data_segment_start },
    );

    const dim_strs = try arena.alloc([]const u8, catalog.descs.len);
    var name_w: usize = 4;
    var dim_w: usize = 5;
    for (catalog.descs, dim_strs) |d, *s| {
        s.* = try renderDims(arena, d.dims);
        if (d.name.len > name_w) name_w = d.name.len;
        if (s.*.len > dim_w) dim_w = s.*.len;
    }

    for (catalog.descs, dim_strs) |d, s| {
        try out.print("{s}", .{d.name});
        try out.splatByteAll(' ', name_w - d.name.len + 2);
        try out.print("{s}", .{s});
        try out.splatByteAll(' ', dim_w - s.len + 2);
        try out.print(
            "{s:<10}  offset=0x{x:0>10}  size={d}\n",
            .{ ggmlTypeName(d.ggml_type), catalog.data_segment_start + d.rel_offset, d.size_bytes },
        );
    }
}

fn renderDims(arena: std.mem.Allocator, dims: []const u64) ![]const u8 {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(arena);
    var w: std.Io.Writer.Allocating = .fromArrayList(arena, &list);
    try w.writer.writeByte('[');
    for (dims, 0..) |d, i| {
        if (i != 0) try w.writer.writeAll(", ");
        try w.writer.print("{d}", .{d});
    }
    try w.writer.writeByte(']');
    return w.toOwnedSlice();
}

fn ggmlTypeName(t: vantablack.GgmlType) []const u8 {
    return switch (t) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f64 => "F64",
        .q4_0 => "Q4_0",
        .q4_1 => "Q4_1",
        .q5_0 => "Q5_0",
        .q5_1 => "Q5_1",
        .q8_0 => "Q8_0",
        .q8_1 => "Q8_1",
        .q2_k => "Q2_K",
        .q3_k => "Q3_K",
        .q4_k => "Q4_K",
        .q5_k => "Q5_K",
        .q6_k => "Q6_K",
        .q8_k => "Q8_K",
        .iq2_xxs => "IQ2_XXS",
        .iq2_xs => "IQ2_XS",
        .iq3_xxs => "IQ3_XXS",
        .iq1_s => "IQ1_S",
        .iq4_nl => "IQ4_NL",
        .iq3_s => "IQ3_S",
        .iq2_s => "IQ2_S",
        .iq4_xs => "IQ4_XS",
        .iq1_m => "IQ1_M",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .tq1_0 => "TQ1_0",
        .tq2_0 => "TQ2_0",
        _ => "?",
    };
}

fn runTokenizeLitertlm(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    text: []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    _ = err;
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();

    const tok_json = try bundle.extractHfTokenizerJson(gpa);
    defer gpa.free(tok_json);

    var tok = try vantablack.Tokenizer.initFromHfJson(gpa, tok_json);
    defer tok.deinitOwnedPieces(gpa);

    try out.print("flavor={s} byte_split={s} vocab={d}\n", .{ @tagName(tok.flavor), @tagName(tok.byte_split), tok.pieces.len });
    const ids = try tok.encode(arena, text, false);
    for (ids, 0..) |id, i| {
        if (i > 0) try out.writeByte(' ');
        try out.print("{d}", .{id});
    }
    try out.writeByte('\n');
    try out.flush();
}

fn runGemma4Step(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    token_id: u32,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    _ = err;
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();
    var model = try vantablack.gemma4_model.initFromLitertlm(gpa, &bundle);
    defer model.deinit();

    const hidden = model.config.hidden;
    const q_out_dim: usize = @as(usize, model.config.n_q_heads) * model.config.head_dim;

    const h = try gpa.alloc(f32, hidden);
    defer gpa.free(h);
    const q = try gpa.alloc(f32, q_out_dim);
    defer gpa.free(q);

    try out.print("stage 1: embed lookup (token {d})\n", .{token_id});
    try model.lookupEmbedding(token_id, h);
    try summarize(out, "  hidden", h);

    try out.writeAll("stage 2: layer_0 Q projection\n");
    try model.projectQ(h, 0, q);
    try summarize(out, "  q_out", q);
    try out.flush();
}

fn summarize(out: *Io.Writer, label: []const u8, v: []const f32) !void {
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var mn: f32 = std.math.floatMax(f32);
    var mx: f32 = -std.math.floatMax(f32);
    for (v) |x| {
        sum += x;
        sum_sq += @as(f64, x) * @as(f64, x);
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    const n: f64 = @floatFromInt(v.len);
    try out.print("{s}[{d}]  min={d:.5}  max={d:.5}  mean={d:.6}  L2={d:.4}  first3=[{d:.5},{d:.5},{d:.5}]\n", .{
        label, v.len, mn, mx, sum / n, @sqrt(sum_sq), v[0], v[1], v[2],
    });
}

fn runGemma4Embed(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    token_id: u32,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    _ = err;
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();
    var model = try vantablack.gemma4_model.initFromLitertlm(gpa, &bundle);
    defer model.deinit();

    if (token_id >= model.config.vocab_size) {
        try out.print("token_id {d} >= vocab_size {d}\n", .{ token_id, model.config.vocab_size });
        try out.flush();
        return error.TokenOutOfRange;
    }

    const hidden = model.config.hidden;
    const buf = try gpa.alloc(f32, hidden);
    defer gpa.free(buf);
    try model.lookupEmbedding(token_id, buf);

    // Summary stats — embedding is huge (1536 floats), show edges + norm.
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var mn: f32 = std.math.floatMax(f32);
    var mx: f32 = -std.math.floatMax(f32);
    for (buf) |v| {
        sum += v;
        sum_sq += @as(f64, v) * @as(f64, v);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const mean: f64 = sum / @as(f64, @floatFromInt(hidden));
    const norm: f64 = @sqrt(sum_sq);

    try out.print("token_id {d} embedding (hidden={d}):\n  first 8: [", .{ token_id, hidden });
    for (buf[0..8], 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try out.print("{d:.5}", .{v});
    }
    try out.writeAll("]\n  last 4:  [");
    for (buf[hidden - 4 .. hidden], 0..) |v, i| {
        if (i > 0) try out.writeAll(", ");
        try out.print("{d:.5}", .{v});
    }
    try out.print("]\n  stats:   min={d:.5} max={d:.5} mean={d:.6} L2norm={d:.4}\n", .{ mn, mx, mean, norm });
    try out.flush();
}

fn runGemma4Config(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    _ = err;
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();

    var model = try vantablack.gemma4_model.initFromLitertlm(gpa, &bundle);
    defer model.deinit();

    try out.print("gemma4 config (decoder = section[{d}]):\n", .{model.decoder_section_idx});
    try out.print("  hidden:      {d}\n", .{model.config.hidden});
    try out.print("  n_layers:    {d}\n", .{model.config.n_layers});
    try out.print("  n_q_heads:   {d}\n", .{model.config.n_q_heads});
    try out.print("  n_kv_heads:  {d}\n", .{model.config.n_kv_heads});
    try out.print("  head_dim:    {d}\n", .{model.config.head_dim});
    try out.print("  ple_dim:     {d}\n", .{model.config.ple_dim});
    try out.print("  vocab_size:  {d}\n", .{model.config.vocab_size});
    if (model.embedder) |emb| {
        try out.print("  embedder:    {s} shape=[", .{emb.dtype.name()});
        for (emb.shape, 0..) |d, i| {
            if (i > 0) try out.writeByte(',');
            try out.print("{d}", .{d});
        }
        try out.print("] data={d} bytes\n", .{emb.data.len});
    } else {
        try out.writeAll("  embedder:    (not found)\n");
    }
    try out.writeAll("  ffn_dim per layer:\n    ");
    for (model.config.ffn_dim_per_layer, 0..) |d, i| {
        if (i > 0) try out.writeByte(' ');
        try out.print("{d}", .{d});
    }
    try out.writeByte('\n');

    var kv_owned: usize = 0;
    var kv_shared: usize = 0;
    var rope_owned: usize = 0;
    for (model.layers) |layer| {
        if (layer.k != null) kv_owned += 1 else kv_shared += 1;
        if (layer.rope_inv_freq != null) rope_owned += 1;
    }
    try out.print("\n  KV layout: {d} layers carry own K/V, {d} layers share from a neighbour\n", .{ kv_owned, kv_shared });
    try out.print("  RoPE: {d} layers ship own inv_freq table; remaining share globally\n", .{rope_owned});
    try out.flush();
}

fn runTraceTensor(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    section_idx: usize,
    selector: []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();
    if (section_idx >= bundle.sections.len) return error.IndexOutOfRange;
    const s = bundle.sections[section_idx];
    if (s.data_type != .tflite_model) {
        try err.print("section {d} is {s}, not TFLiteModel\n", .{ section_idx, s.data_type.name() });
        try err.flush();
        return error.WrongSectionType;
    }
    const bytes = try bundle.sectionBytes(s);
    var tfl = try vantablack.tflite.Model.init(gpa, bytes);
    defer tfl.deinit();

    // Selector: numeric index OR name substring.
    var target: i32 = -1;
    if (std.fmt.parseInt(i32, selector, 10)) |idx| {
        target = idx;
    } else |_| {
        // Substring search — first match wins.
        for (tfl.tensors, 0..) |tt, i| {
            if (std.mem.indexOf(u8, tt.name, selector) != null) {
                target = @intCast(i);
                break;
            }
        }
        if (target < 0) {
            try err.print("no tensor name contains '{s}'\n", .{selector});
            try err.flush();
            return error.NoMatchingTensor;
        }
    }
    if (target < 0 or @as(usize, @intCast(target)) >= tfl.tensors.len) return error.IndexOutOfRange;
    const t = &tfl.tensors[@intCast(target)];

    try out.print("tensor [{d}] {s}\n  shape=[", .{ target, t.dtype.name() });
    for (t.shape, 0..) |d, i| {
        if (i > 0) try out.writeByte(',');
        try out.print("{d}", .{d});
    }
    try out.print("] data={d} bytes\n  name: {s}\n\n", .{ t.data.len, t.name });

    // Walk operators. For each, look in inputs[] and outputs[].
    var n_producers: usize = 0;
    var n_consumers: usize = 0;
    try out.writeAll("producers (ops writing this tensor):\n");
    for (tfl.operators, 0..) |op, oi| {
        for (op.outputs) |out_t| {
            if (out_t == target) {
                n_producers += 1;
                try printOpRow(out, &tfl, oi, op);
            }
        }
    }
    if (n_producers == 0) try out.writeAll("  (none — likely a constant / input tensor)\n");

    try out.writeAll("\nconsumers (ops reading this tensor):\n");
    for (tfl.operators, 0..) |op, oi| {
        for (op.inputs) |in_t| {
            if (in_t == target) {
                n_consumers += 1;
                try printOpRow(out, &tfl, oi, op);
                break; // already shown; don't repeat if it appears twice in inputs
            }
        }
    }
    if (n_consumers == 0) try out.writeAll("  (none — likely a final output)\n");

    try out.print("\nsummary: {d} producer(s), {d} consumer(s)\n", .{ n_producers, n_consumers });
    try out.flush();
}

fn printOpRow(out: *Io.Writer, tfl: *const vantablack.tflite.Model, op_idx: usize, op: vantablack.tflite.Operator) !void {
    const op_name = if (op.opcode_index < tfl.operator_codes.len)
        vantablack.tflite.nameOfOp(tfl.operator_codes[op.opcode_index].builtin_code)
    else
        "?";
    try out.print("  [op {d:>4}] {s:<20} inputs=[", .{ op_idx, op_name });
    for (op.inputs, 0..) |t, j| {
        if (j > 0) try out.writeByte(',');
        try out.print("{d}", .{t});
    }
    try out.writeAll("] outputs=[");
    for (op.outputs, 0..) |t, j| {
        if (j > 0) try out.writeByte(',');
        try out.print("{d}", .{t});
    }
    try out.writeAll("]\n");
}

fn runScanLayers(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    _ = err;
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();

    // First pass: pick the TFLite section with the highest layer count.
    // Buckets store `*const Tensor` pointers into the section's tflite
    // Model — once we pick a winner we keep that Model alive for the
    // rest of this function (the loop's per-iteration deinit would
    // free the backing store and leave dangling pointers).
    var best_section: ?usize = null;
    var best_layer_count: usize = 0;
    for (bundle.sections, 0..) |s, sec_idx| {
        if (s.data_type != .tflite_model) continue;
        const bytes = try bundle.sectionBytes(s);
        var tfl = vantablack.tflite.Model.init(gpa, bytes) catch continue;
        defer tfl.deinit();
        // Probe-only: build buckets, count, throw away. Cheap because
        // we don't keep tensor pointers (the model is freed at end of
        // this iteration).
        const probe = vantablack.gemma_layer_scan.scanLayers(gpa, tfl.tensors) catch continue;
        defer gpa.free(probe);
        if (probe.len > best_layer_count) {
            best_layer_count = probe.len;
            best_section = sec_idx;
        }
    }

    if (best_section == null) {
        try out.writeAll("no decoder section with layer_N/ tensors found\n");
        try out.flush();
        return;
    }

    // Second pass: re-open the winning section and keep it alive.
    const decoder_section = bundle.sections[best_section.?];
    const decoder_bytes = try bundle.sectionBytes(decoder_section);
    var decoder_tfl = try vantablack.tflite.Model.init(gpa, decoder_bytes);
    defer decoder_tfl.deinit();
    const saved_buckets = try vantablack.gemma_layer_scan.scanLayers(gpa, decoder_tfl.tensors);
    defer gpa.free(saved_buckets);

    try out.print("decoder: section[{d}], {d} layers\n\n", .{ best_section.?, best_layer_count });

    // Coverage summary across all layers.
    var have_gate: usize = 0;
    var have_up: usize = 0;
    var have_down: usize = 0;
    var have_attn_o: usize = 0;
    var have_q: usize = 0;
    var have_k: usize = 0;
    var have_v: usize = 0;
    var have_ple_gate: usize = 0;
    var have_ple_proj: usize = 0;
    var have_rope: usize = 0;
    var total_unknown: usize = 0;
    for (saved_buckets) |b| {
        if (b.mlp_gate != null) have_gate += 1;
        if (b.mlp_up != null) have_up += 1;
        if (b.mlp_down != null) have_down += 1;
        if (b.attn_o != null) have_attn_o += 1;
        if (b.q_proj != null) have_q += 1;
        if (b.k_proj != null) have_k += 1;
        if (b.v_proj != null) have_v += 1;
        if (b.ple_gate != null) have_ple_gate += 1;
        if (b.ple_proj != null) have_ple_proj += 1;
        if (b.rope_inv_freq != null) have_rope += 1;
        total_unknown += b.unknown_count;
    }
    try out.print(
        "coverage (layers with role found / total):\n" ++
            "  mlp.gate    {d}/{d}\n" ++
            "  mlp.up      {d}/{d}\n" ++
            "  mlp.down    {d}/{d}\n" ++
            "  attn.o      {d}/{d}\n" ++
            "  q           {d}/{d}\n" ++
            "  k           {d}/{d}\n" ++
            "  v           {d}/{d}\n" ++
            "  ple.gate    {d}/{d}\n" ++
            "  ple.proj    {d}/{d}\n" ++
            "  rope.freqs  {d}/{d}\n" ++
            "  unknown     {d} across all layers\n\n",
        .{
            have_gate, best_layer_count, have_up, best_layer_count,
            have_down, best_layer_count, have_attn_o, best_layer_count,
            have_q, best_layer_count, have_k, best_layer_count, have_v, best_layer_count,
            have_ple_gate, best_layer_count, have_ple_proj, best_layer_count,
            have_rope, best_layer_count, total_unknown,
        },
    );

    // Per-layer breakdown with shapes.
    for (saved_buckets) |b| {
        try out.print("layer_{d:>2}:\n", .{b.layer_idx});
        try printSlot(out, "  q          ", b.q_proj);
        try printSlot(out, "  k          ", b.k_proj);
        try printSlot(out, "  v          ", b.v_proj);
        try printSlot(out, "  attn.o     ", b.attn_o);
        try printSlot(out, "  mlp.gate   ", b.mlp_gate);
        try printSlot(out, "  mlp.up     ", b.mlp_up);
        try printSlot(out, "  mlp.down   ", b.mlp_down);
        try printSlot(out, "  ple.gate   ", b.ple_gate);
        try printSlot(out, "  ple.proj   ", b.ple_proj);
        try printSlot(out, "  rope.freqs ", b.rope_inv_freq);
        try out.print("  unknown={d}\n", .{b.unknown_count});
    }
    try out.flush();
}

fn printSlot(out: *Io.Writer, label: []const u8, t: ?*const vantablack.tflite.Tensor) !void {
    try out.writeAll(label);
    if (t) |tt| {
        try out.print("{s:<8} shape=[", .{tt.dtype.name()});
        for (tt.shape, 0..) |d, i| {
            if (i > 0) try out.writeByte(',');
            try out.print("{d}", .{d});
        }
        try out.print("] {d} bytes\n", .{tt.data.len});
    } else {
        try out.writeAll("(absent)\n");
    }
}

fn runInspectSection(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    section_idx: usize,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(null, len_us, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(mapped);

    var bundle = try vantablack.litertlm.Bundle.init(gpa, mapped);
    defer bundle.deinit();

    if (section_idx >= bundle.sections.len) {
        try err.print("section index {d} out of range (bundle has {d} sections)\n", .{ section_idx, bundle.sections.len });
        try err.flush();
        return error.IndexOutOfRange;
    }
    const s = bundle.sections[section_idx];
    if (s.data_type != .tflite_model) {
        try err.print("section {d} is {s}, not TFLiteModel — nothing to deep-dump\n", .{ section_idx, s.data_type.name() });
        try err.flush();
        return error.WrongSectionType;
    }

    const bytes = try bundle.sectionBytes(s);
    var tfl = try vantablack.tflite.Model.init(gpa, bytes);
    defer tfl.deinit();

    const size_mb: f64 = @as(f64, @floatFromInt(s.size())) / (1024.0 * 1024.0);
    try out.print("section[{d}] TFLiteModel, {d:.2} MB, {d} tensors, {d} operators, {d} op_codes\n", .{
        section_idx, size_mb, tfl.tensors.len, tfl.operators.len, tfl.operator_codes.len,
    });

    try out.writeAll("\noperator_codes:\n");
    for (tfl.operator_codes, 0..) |oc, i| {
        const custom = if (oc.custom_code) |c| c else "";
        try out.print("  [{d}] builtin={d} ({s}) custom=\"{s}\" version={d}\n", .{
            i, oc.builtin_code, vantablack.tflite.nameOfOp(oc.builtin_code), custom, oc.version,
        });
    }

    try out.writeAll("\noperators (op #, opcode, inputs → outputs):\n");
    for (tfl.operators, 0..) |op, i| {
        const op_name = if (op.opcode_index < tfl.operator_codes.len)
            vantablack.tflite.nameOfOp(tfl.operator_codes[op.opcode_index].builtin_code)
        else
            "?";
        try out.print("  [{d:>4}] {s:<20} (idx={d}) inputs=[", .{ i, op_name, op.opcode_index });
        for (op.inputs, 0..) |t, j| {
            if (j > 0) try out.writeByte(',');
            try out.print("{d}", .{t});
        }
        try out.writeAll("] outputs=[");
        for (op.outputs, 0..) |t, j| {
            if (j > 0) try out.writeByte(',');
            try out.print("{d}", .{t});
        }
        try out.writeAll("]\n");
    }

    // Sort tensors by data size descending so the real weight blobs
    // surface ahead of trivial scalars.
    const ranked = try gpa.dupe(vantablack.tflite.Tensor, tfl.tensors);
    defer gpa.free(ranked);
    const SortCtx = struct {
        fn gt(_: void, a: vantablack.tflite.Tensor, b: vantablack.tflite.Tensor) bool {
            return a.data.len > b.data.len;
        }
    };
    std.sort.pdq(vantablack.tflite.Tensor, ranked, {}, SortCtx.gt);

    try out.writeAll("\ntensors (sorted by data size, all listed):\n");
    for (ranked, 0..) |t, i| {
        try out.print("  [{d:>4}] {s:<10} ", .{ i, t.dtype.name() });
        try out.writeAll(t.name);
        try out.writeAll(" shape=[");
        for (t.shape, 0..) |d, j| {
            if (j > 0) try out.writeByte(',');
            try out.print("{d}", .{d});
        }
        const quant = if (t.scales.len > 0) " quant" else "";
        try out.print("] data={d} bytes buffer={d}{s}\n", .{ t.data.len, t.buffer_index, quant });
    }

    try out.flush();
}

// ----- LiteRT-LM (.litertlm) container inspection -------------------------

fn runInspect(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    input_path: []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd, input_path });
    };

    // mmap the file. The litertlm header lives at the start and section
    // bodies are scattered through the rest; mmap is the cheapest way to
    // address them without paging the whole file into RAM.
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;
    if (len_us < 32) {
        try err.writeAll("file too small to be a .litertlm bundle\n");
        try err.flush();
        return error.FileTooSmall;
    }
    const mapped = try std.posix.mmap(
        null,
        len_us,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mapped);

    var bundle = vantablack.litertlm.Bundle.init(gpa, mapped) catch |e| {
        try err.print("failed to parse litertlm header: {s}\n", .{@errorName(e)});
        try err.flush();
        return e;
    };
    defer bundle.deinit();

    try out.print("litertlm bundle: {s}\n", .{abs_path});
    try out.print("  version: {d}.{d}.{d}\n", .{ bundle.version.major, bundle.version.minor, bundle.version.patch });
    try out.print("  size:    {d} bytes ({d:.2} MB)\n", .{ len_us, @as(f64, @floatFromInt(len_us)) / (1024.0 * 1024.0) });
    try out.print("  sections: {d}\n", .{bundle.sections.len});
    for (bundle.sections, 0..) |s, i| {
        const size_mb: f64 = @as(f64, @floatFromInt(s.size())) / (1024.0 * 1024.0);
        try out.print(
            "    [{d}] {s:<20} offset={d:<12} size={d} ({d:.2} MB)\n",
            .{ i, s.data_type.name(), s.begin, s.size(), size_mb },
        );
    }

    // LlmMetadataProto: small, decode and dump key fields.
    if (bundle.findSection(.llm_metadata_proto)) |s| {
        const bytes = try bundle.sectionBytes(s);
        var meta = try vantablack.llm_metadata.LlmMetadata.parse(arena, bytes);
        defer meta.deinit();
        try out.writeAll("  metadata:\n");
        try out.print("    max_num_tokens: {d}\n", .{meta.max_num_tokens});
        if (meta.llm_model_type_bytes) |mt| {
            if (try vantablack.llm_metadata.detectModelType(mt)) |tag| {
                try out.print("    model_type:     {s}\n", .{tag.name()});
            } else {
                try out.writeAll("    model_type:     <unset>\n");
            }
        }
        if (meta.jinja_prompt_template) |tpl| {
            const preview_len = @min(tpl.len, 80);
            try out.print(
                "    jinja_template: ({d} chars) {s}{s}\n",
                .{ tpl.len, tpl[0..preview_len], if (tpl.len > preview_len) "..." else "" },
            );
        }
        try out.print("    stop_tokens:    {d}\n", .{meta.stop_tokens_bytes.len});
        // Sampler defaults baked into the model. These are what
        // litert-lm CLI uses when --top-k / --temperature etc. are not
        // explicitly passed. Required for matching Phase 19e bit-equal.
        if (meta.sampler_params_bytes) |sb| {
            if (vantablack.llm_metadata.parseSamplerParameters(sb)) |sp| {
                try out.print(
                    "    sampler:        type={s} k={d} p={d:.4} temperature={d:.4} seed={?}\n",
                    .{ sp.type.name(), sp.k, sp.p, sp.temperature, sp.seed },
                );
            } else |_| {}
        }
    }

    // Tokenizer — decompress HF_Tokenizer_Zlib and surface vocab size +
    // flavor. SentencePiece sections are still left as raw byte count
    // (decode lives in `Tokenizer.init` which expects a GGUF catalog —
    // standalone SP decode is a follow-up).
    if (bundle.findSection(.hf_tokenizer_zlib)) |s| {
        try out.print("  tokenizer (HF, zlib): {d} compressed bytes\n", .{s.size()});
        if (bundle.extractHfTokenizerJson(gpa)) |json| {
            defer gpa.free(json);
            try out.print("    decompressed: {d} bytes\n", .{json.len});
            if (vantablack.Tokenizer.initFromHfJson(gpa, json)) |tok_owned| {
                var tok = tok_owned;
                defer tok.deinitOwnedPieces(gpa);
                try out.print(
                    "    flavor={s} byte_split={s} vocab={d}\n",
                    .{ @tagName(tok.flavor), @tagName(tok.byte_split), tok.pieces.len },
                );
            } else |e| {
                try out.print("    parse failed: {s}\n", .{@errorName(e)});
            }
        } else |e| {
            try out.print("    decompress failed: {s}\n", .{@errorName(e)});
        }
    } else if (bundle.findSection(.sp_tokenizer)) |s| {
        try out.print("  tokenizer (SentencePiece): {d} bytes\n", .{s.size()});
    }

    // TFLite Model sections — Phase B partial. Real Gemma 4 splits its
    // weights across multiple TFLite sections (embedder, decode loop,
    // KV cache layouts, etc.). Walk every TFLite section.
    var tfl_idx: usize = 0;
    for (bundle.sections, 0..) |s, sec_idx| {
        if (s.data_type != .tflite_model) continue;
        const tfl_bytes = try bundle.sectionBytes(s);
        var tfl_model = vantablack.tflite.Model.init(gpa, tfl_bytes) catch |e| {
            try out.print("  tflite_model[{d}] (section {d}): parse failed ({s})\n", .{ tfl_idx, sec_idx, @errorName(e) });
            tfl_idx += 1;
            continue;
        };
        defer tfl_model.deinit();
        const size_mb: f64 = @as(f64, @floatFromInt(s.size())) / (1024.0 * 1024.0);
        try out.print(
            "  tflite_model[{d}] (section {d}, {d:.2} MB): version={d}, tensors={d}, operators={d}, op_codes={d}\n",
            .{ tfl_idx, sec_idx, size_mb, tfl_model.version, tfl_model.tensors.len, tfl_model.operators.len, tfl_model.operator_codes.len },
        );

        // Op histogram — surfaces which BuiltinOperators the forward
        // graph uses. Critical for Phase 19c (mapping ops to vantablack's
        // Llama forward path).
        const hist = tfl_model.opHistogram(gpa) catch null;
        defer if (hist) |h| gpa.free(h);
        if (hist) |h| {
            try out.writeAll("    op histogram:\n");
            var printed: usize = 0;
            for (h) |row| {
                if (row.count == 0) continue;
                if (printed >= 10) {
                    try out.print("      ... ({d} more op types)\n", .{h.len - printed});
                    break;
                }
                try out.print("      {s:<20} ({d}) {d}\n", .{ vantablack.tflite.nameOfOp(row.builtin_code), row.builtin_code, row.count });
                printed += 1;
            }
        }

        // Top weight tensors by size — most informative for figuring out
        // which section holds which model component (embedder, attention,
        // FFN, etc.).
        var weight_count: usize = 0;
        var weight_bytes: usize = 0;
        for (tfl_model.tensors) |t| {
            if (t.data.len > 0) {
                weight_count += 1;
                weight_bytes += t.data.len;
            }
        }
        const weight_mb: f64 = @as(f64, @floatFromInt(weight_bytes)) / (1024.0 * 1024.0);
        try out.print("    weights: {d} tensors, {d:.2} MB\n", .{ weight_count, weight_mb });

        // Sort tensors by data size descending so the genuine weight
        // blobs surface (not index/scalar constants that happen to
        // appear first).
        const ranked = gpa.dupe(vantablack.tflite.Tensor, tfl_model.tensors) catch null;
        defer if (ranked) |r| gpa.free(r);
        if (ranked) |r| {
            const SortCtx = struct {
                fn gt(_: void, a: vantablack.tflite.Tensor, b: vantablack.tflite.Tensor) bool {
                    return a.data.len > b.data.len;
                }
            };
            std.sort.pdq(vantablack.tflite.Tensor, r, {}, SortCtx.gt);
            var shown: usize = 0;
            for (r) |t| {
                if (t.data.len == 0) break;
                if (shown == 0) try out.writeAll("    largest weight tensors (first 5 by size):\n");
                if (shown >= 5) break;
                shown += 1;
                try printTensorRow(out, t);
            }
        }
        tfl_idx += 1;
    }

    // Inference status — honest about what's missing.
    try out.writeAll("  inference: TFLite tensor → vantablack model mapping pending (Phase B/C/D — see README roadmap items 19b–d)\n");
    try out.flush();
}

fn printTensorRow(out: *Io.Writer, t: vantablack.tflite.Tensor) !void {
    const name_w: usize = 60;
    var name_buf: [name_w]u8 = undefined;
    const name = if (t.name.len <= name_w) t.name else blk: {
        @memcpy(name_buf[0 .. name_w - 3], t.name[0 .. name_w - 3]);
        @memcpy(name_buf[name_w - 3 ..][0..3], "...");
        break :blk name_buf[0..name_w];
    };
    try out.print("      {s:<60} {s:<10} shape=[", .{ name, t.dtype.name() });
    for (t.shape, 0..) |d, i| {
        if (i > 0) try out.writeByte(',');
        try out.print("{d}", .{d});
    }
    const quant_hint: []const u8 = if (t.scales.len > 0) " quantized" else "";
    try out.print("] data={d} bytes{s}\n", .{ t.data.len, quant_hint });
}

// ----- HuggingFace / MLX directory mode -----------------------------------

fn isDirectory(io: Io, abs_path: []const u8) bool {
    const dir = Io.Dir.openDirAbsolute(io, abs_path, .{ .iterate = false }) catch return false;
    var d = dir;
    d.close(io);
    return true;
}

fn runHfCli(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    dir_path: []const u8,
    args: []const []const u8,
    out: *Io.Writer,
    err: *Io.Writer,
) !void {
    if (args.len < 3 or !(std.mem.eql(u8, args[2], "prompt") or std.mem.eql(u8, args[2], "chat") or std.mem.eql(u8, args[2], "serve"))) {
        try err.writeAll(
            "MLX/HF directory mode supports:\n" ++
                "  vantablack <model_dir> prompt <n> <text>\n" ++
                "  vantablack <model_dir> chat   <n> <user-message>\n" ++
                "  vantablack <model_dir> serve  [--host H] [--port P] [--threads N]\n" ++
                "Sampler flags (--temp / --top-k / --top-p / --seed / --threads / --system) work as in GGUF mode.\n",
        );
        try err.flush();
        return error.MissingArgs;
    }
    if (std.mem.eql(u8, args[2], "serve")) {
        return runHfServe(gpa, arena, io, dir_path, args, err);
    }
    const is_chat = std.mem.eql(u8, args[2], "chat");

    var sampler_cfg: vantablack.SamplerConfig = .{};
    var system_prompt: []const u8 = "You are a helpful assistant.";
    var threads: usize = 0;
    var idx: usize = 3;
    while (idx < args.len) {
        const a = args[idx];
        if (std.mem.eql(u8, a, "--temp")) {
            idx += 1;
            sampler_cfg.temperature = try std.fmt.parseFloat(f32, args[idx]);
        } else if (std.mem.eql(u8, a, "--top-k")) {
            idx += 1;
            sampler_cfg.top_k = try std.fmt.parseInt(usize, args[idx], 10);
        } else if (std.mem.eql(u8, a, "--top-p")) {
            idx += 1;
            sampler_cfg.top_p = try std.fmt.parseFloat(f32, args[idx]);
        } else if (std.mem.eql(u8, a, "--seed")) {
            idx += 1;
            sampler_cfg.seed = try std.fmt.parseInt(u64, args[idx], 10);
        } else if (std.mem.eql(u8, a, "--system")) {
            idx += 1;
            system_prompt = args[idx];
        } else if (std.mem.eql(u8, a, "--threads")) {
            idx += 1;
            threads = try std.fmt.parseInt(usize, args[idx], 10);
        } else break;
        idx += 1;
    }

    if (idx + 1 >= args.len) {
        try err.writeAll("missing <n> or <text>\n");
        try err.flush();
        return error.MissingArgs;
    }
    const n_steps = try std.fmt.parseInt(u32, args[idx], 10);
    const user_text = args[idx + 1];

    var bundle = try vantablack.hf_loader.HfBundle.init(gpa, io, dir_path);
    defer bundle.deinit();

    const cfg = try vantablack.hf_config.parse(arena, bundle.config_json);

    var model = try vantablack.Model.initFromHf(gpa, &bundle, cfg);
    defer model.deinit(gpa);


    if (bundle.tokenizer_json == null) {
        try err.writeAll("error: tokenizer.json missing in directory; vantablack cannot encode without it\n");
        try err.flush();
        return error.MissingArgs;
    }
    var tok = try vantablack.Tokenizer.initFromHfJson(gpa, bundle.tokenizer_json.?);
    defer tok.deinitOwnedPieces(gpa);

    var sampler = try vantablack.Sampler.init(gpa, sampler_cfg, model.config.vocab_size);
    defer sampler.deinit(gpa);

    // Metal first so State can alias persistent shared-storage scratch into
    // the backend's MTLBuffers. The bundle's mmap'd shard memory must outlive
    // `maybe_metal` — `bundle.deinit()` runs after this defer because of LIFO
    // order, so that's already correct.
    var maybe_metal: ?vantablack.MetalBackend = blk: {
        if (!vantablack.metal.metal_enabled) break :blk null;
        break :blk vantablack.MetalBackend.initFromHf(gpa, &bundle, model.config) catch |e| {
            std.log.warn("MetalBackend.initFromHf failed: {s}; falling back to CPU", .{@errorName(e)});
            break :blk null;
        };
    };
    defer if (maybe_metal) |*mb| mb.deinit(gpa);
    const metal_ptr: ?*vantablack.MetalBackend = if (maybe_metal != null) &maybe_metal.? else null;

    var state = try vantablack.forward.State.init(gpa, &model, metal_ptr);
    defer state.deinit(gpa);

    var cache = try vantablack.KvCache.init(
        gpa,
        model.config.n_layers,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.max_seq,
        metal_ptr,
    );
    defer cache.deinit(gpa);

    // Metal handles matmuls; extra CPU workers just spin on GPU sync.
    const effective_threads = if (metal_ptr != null and threads == 0) @as(usize, 1) else threads;
    var pool = try vantablack.ThreadPool.init(gpa, effective_threads);
    defer pool.deinit(gpa);

    // Build prompt: chat mode wraps in TinyLlama zephyr template.
    const prompt_text: []const u8 = if (is_chat)
        try vantablack.chat_template.formatLlamaChatSingle(arena, system_prompt, user_text)
    else
        user_text;

    const prompt_ids = try tok.encode(arena, prompt_text, true);

    try out.print("[encoded {d} tokens]\n", .{prompt_ids.len});
    try out.flush();

    if (prompt_ids.len == 0) return;
    for (prompt_ids) |id| try tok.decodeTo(out, id);
    try out.flush();
    // HF path is CPU-only — always prefill.
    try vantablack.prefill.prefillCpu(gpa, pool, &model, &state, &cache, prompt_ids);

    var produced: u32 = 0;
    var next: u32 = sampler.sample(state.logits);
    while (produced < n_steps) : (produced += 1) {
        if (next == tok.eos) break;
        try tok.decodeTo(out, next);
        try out.flush();
        try vantablack.forward.step(&model, &state, &cache, pool, metal_ptr, next);
        next = sampler.sample(state.logits);
    }
    try out.writeByte('\n');
    try out.flush();
}

fn runHfServe(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: Io,
    dir_path: []const u8,
    args: []const []const u8,
    err: *Io.Writer,
) !void {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 11434;
    var threads: usize = 0;
    var idx: usize = 3;
    while (idx < args.len) : (idx += 1) {
        const a = args[idx];
        if (std.mem.eql(u8, a, "--host")) {
            idx += 1;
            host = args[idx];
        } else if (std.mem.eql(u8, a, "--port")) {
            idx += 1;
            port = try std.fmt.parseInt(u16, args[idx], 10);
        } else if (std.mem.eql(u8, a, "--threads")) {
            idx += 1;
            threads = try std.fmt.parseInt(usize, args[idx], 10);
        } else break;
    }

    // Heap-allocate the bundle so the Server can own it across its move
    // out of this stack frame; the bundle's mmap'd shard pointers are what
    // every TypedTensor inside Model points into.
    const bundle_ptr = try gpa.create(vantablack.hf_loader.HfBundle);
    errdefer gpa.destroy(bundle_ptr);
    bundle_ptr.* = try vantablack.hf_loader.HfBundle.init(gpa, io, dir_path);
    // Note: on success, ownership transfers into the Server.

    const cfg = try vantablack.hf_config.parse(arena, bundle_ptr.config_json);

    // Use the directory basename as the model name; clients address it via
    // POST /api/generate {"model": "<basename>"}.
    const model_name = std.fs.path.basename(dir_path);

    var srv = vantablack.Server.initFromHf(gpa, io, bundle_ptr, cfg, .{
        .host = host,
        .port = port,
        .model_name = model_name,
        .model_path = dir_path,
        .threads = threads,
    }) catch |e| {
        // initFromHf's errdefer already cleaned the bundle on its error
        // paths, so just surface and exit.
        try err.print("vantablack: serve init failed: {t}\n", .{e});
        try err.flush();
        return e;
    };
    defer srv.deinit();
    try srv.run();
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    try std.testing.fuzz({}, testOne, .{});
}

fn testOne(context: void, smith: *std.testing.Smith) !void {
    _ = context;
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    while (!smith.eos()) switch (smith.value(enum { add_data, dup_data })) {
        .add_data => {
            const slice = try list.addManyAsSlice(gpa, smith.value(u4));
            smith.bytes(slice);
        },
        .dup_data => {
            if (list.items.len == 0) continue;
            if (list.items.len > std.math.maxInt(u32)) return error.SkipZigTest;
            const len = smith.valueRangeAtMost(u32, 1, @min(32, list.items.len));
            const off = smith.valueRangeAtMost(u32, 0, @intCast(list.items.len - len));
            try list.appendSlice(gpa, list.items[off..][0..len]);
            try std.testing.expectEqualSlices(
                u8,
                list.items[off..][0..len],
                list.items[list.items.len - len ..],
            );
        },
    };
}
