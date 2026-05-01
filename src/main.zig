const std = @import("std");
const Io = std.Io;

const vantablack = @import("vantablack");

const usage =
    \\usage:
    \\  vantablack <model.gguf>                              print tensor catalog
    \\  vantablack <model.gguf> generate <n> <id> [id...]    generate n tokens
    \\                                                       after the given prompt ids
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

    const input_path = args[1];
    const abs_path = if (std.fs.path.isAbsolute(input_path))
        try arena.dupe(u8, input_path)
    else blk: {
        const cwd_path = try std.process.currentPathAlloc(io, arena);
        break :blk try std.fs.path.resolve(arena, &.{ cwd_path, input_path });
    };

    var mapper = try vantablack.ModelMapper.init(gpa, io, abs_path);
    defer mapper.deinit();

    if (args.len >= 3 and std.mem.eql(u8, args[2], "generate")) {
        if (args.len < 5) {
            try err.writeAll(usage);
            try err.flush();
            return error.MissingArgs;
        }
        const n_steps = try std.fmt.parseInt(u32, args[3], 10);
        const prompt_ids = try arena.alloc(u32, args.len - 4);
        for (prompt_ids, args[4..]) |*id, s| id.* = try std.fmt.parseInt(u32, s, 10);
        try generate(gpa, &mapper, n_steps, prompt_ids, out);
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
    out: *Io.Writer,
) !void {
    var model = try vantablack.Model.init(gpa, mapper);
    defer model.deinit(gpa);

    var state = try vantablack.forward.State.init(gpa, &model);
    defer state.deinit(gpa);

    var cache = try vantablack.KvCache.init(
        gpa,
        model.config.n_layers,
        model.config.n_kv_heads,
        model.config.head_dim,
        model.config.max_seq,
    );
    defer cache.deinit(gpa);

    var tok = try vantablack.Tokenizer.init(gpa, mapper.catalog);
    defer tok.deinit(gpa);

    // Feed the prompt one token at a time. Decode + print as we go for visibility.
    if (prompt_ids.len == 0) return;
    for (prompt_ids) |id| {
        try vantablack.forward.step(&model, &state, &cache, id);
        try tok.decodeTo(out, id);
    }
    try out.flush();

    // Generate n_steps tokens after the prompt, greedy argmax.
    var produced: u32 = 0;
    var next: u32 = @intCast(vantablack.math.argmax(state.logits));
    while (produced < n_steps) : (produced += 1) {
        if (next == tok.eos) break;
        try tok.decodeTo(out, next);
        try out.flush();
        try vantablack.forward.step(&model, &state, &cache, next);
        next = @intCast(vantablack.math.argmax(state.logits));
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
