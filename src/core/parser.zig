//! GGUF v2/v3 header + tensor descriptor parser.
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//! All numbers are little-endian. Strings are { length:u64, bytes:[length]u8 }
//! with no NUL terminator.
//!
//! Zero-copy: every `[]const u8` field returned in `Catalog` slices into the
//! caller-supplied mapped buffer, with the exception of `TensorDesc.dims`
//! (arena-copied because the file does not guarantee u64 alignment of the
//! `tensor_info` records).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const magic_le: u32 = 0x46554747; // 'G''G''U''F' little-endian

pub const ParseError = error{
    NotAGgufFile,
    UnsupportedGgufVersion,
    UnexpectedEndOfFile,
    UnknownMetadataValueType,
    ArrayNestingTooDeep,
    UnsupportedTensorBlockShape,
    TensorOutOfBounds,
} || Allocator.Error;

pub const max_array_depth: u32 = 4;

/// GGML quantization tag. Open enum: unknown future ggml_type values are
/// preserved as their raw u32 instead of failing the parse.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 31,
    tq2_0 = 32,
    _,
};

const BlockInfo = struct {
    elements: u64,
    bytes: u64,
};

/// Per-block sizing for the ggml_type values whose tensor sizes we need to
/// compute in Task 1. Source: `ggml-quants.h`. Returning `null` for an
/// unsupported type causes the parser to skip the size computation for that
/// tensor (size_bytes left at 0); the catalog still enumerates it.
pub fn blockInfo(t: GgmlType) ?BlockInfo {
    return switch (t) {
        .f32 => .{ .elements = 1, .bytes = 4 },
        .f16 => .{ .elements = 1, .bytes = 2 },
        .bf16 => .{ .elements = 1, .bytes = 2 },
        .f64 => .{ .elements = 1, .bytes = 8 },
        .i8 => .{ .elements = 1, .bytes = 1 },
        .i16 => .{ .elements = 1, .bytes = 2 },
        .i32 => .{ .elements = 1, .bytes = 4 },
        .i64 => .{ .elements = 1, .bytes = 8 },
        .q8_0 => .{ .elements = 32, .bytes = 34 },
        .q4_0 => .{ .elements = 32, .bytes = 18 },
        .q4_1 => .{ .elements = 32, .bytes = 20 },
        .q5_0 => .{ .elements = 32, .bytes = 22 },
        .q5_1 => .{ .elements = 32, .bytes = 24 },
        .q8_1 => .{ .elements = 32, .bytes = 36 },
        .q2_k => .{ .elements = 256, .bytes = 84 },
        .q3_k => .{ .elements = 256, .bytes = 110 },
        .q4_k => .{ .elements = 256, .bytes = 144 },
        .q5_k => .{ .elements = 256, .bytes = 176 },
        .q6_k => .{ .elements = 256, .bytes = 210 },
        .q8_k => .{ .elements = 256, .bytes = 292 },
        .tq1_0 => .{ .elements = 256, .bytes = 54 },
        .tq2_0 => .{ .elements = 256, .bytes = 66 },
        else => null,
    };
}

pub const MetaValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

pub const MetaArray = struct {
    elem_type: MetaValueType,
    count: u64,
    /// Raw slice of the array payload (not including the 12-byte header
    /// `{elem_type:u32, count:u64}`). Borrows from the mapped buffer; iterate
    /// by walking it with the same readers used by the top-level parser.
    raw: []const u8,
};

pub const MetaValue = union(MetaValueType) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    /// Borrows from mapped buffer.
    string: []const u8,
    array: MetaArray,
    u64: u64,
    i64: i64,
    f64: f64,
};

pub const MetaKv = struct {
    /// Borrows from mapped buffer.
    key: []const u8,
    value: MetaValue,
};

pub const TensorDesc = struct {
    /// Borrows from mapped buffer.
    name: []const u8,
    /// Arena-copied. GGUF does not guarantee 8-byte alignment of tensor_info
    /// records, so dims are read via `std.mem.readInt` rather than ptr-cast.
    dims: []const u64,
    ggml_type: GgmlType,
    /// Offset relative to `Catalog.data_segment_start`.
    rel_offset: u64,
    /// Computed from `dims` × `blockInfo(ggml_type)`. Zero if the ggml_type
    /// has no entry in `blockInfo` (unknown / future quant).
    size_bytes: u64,
};

pub const Catalog = struct {
    descs: []TensorDesc,
    metadata: []MetaKv,
    data_segment_start: u64,
    alignment: u32,

    pub fn find(self: Catalog, name: []const u8) ?*const TensorDesc {
        for (self.descs) |*d| {
            if (std.mem.eql(u8, d.name, name)) return d;
        }
        return null;
    }
};

const Cursor = struct {
    buf: []const u8,
    off: usize,

    fn ensure(self: *const Cursor, n: usize) ParseError!void {
        if (self.off + n > self.buf.len) return error.UnexpectedEndOfFile;
    }

    fn readInt(self: *Cursor, comptime T: type) ParseError!T {
        const n = @sizeOf(T);
        try self.ensure(n);
        const v = std.mem.readInt(T, self.buf[self.off..][0..n], .little);
        self.off += n;
        return v;
    }

    fn readFloat(self: *Cursor, comptime T: type) ParseError!T {
        const Bits = switch (T) {
            f32 => u32,
            f64 => u64,
            else => @compileError("unsupported float type"),
        };
        return @bitCast(try self.readInt(Bits));
    }

    fn readBool(self: *Cursor) ParseError!bool {
        const b = try self.readInt(u8);
        return b != 0;
    }

    fn readString(self: *Cursor) ParseError![]const u8 {
        const len = try self.readInt(u64);
        const len_us: usize = std.math.cast(usize, len) orelse return error.UnexpectedEndOfFile;
        try self.ensure(len_us);
        const s = self.buf[self.off..][0..len_us];
        self.off += len_us;
        return s;
    }

    fn skip(self: *Cursor, n: usize) ParseError!void {
        try self.ensure(n);
        self.off += n;
    }
};

fn readMetaValue(cur: *Cursor, vt: MetaValueType, depth: u32) ParseError!MetaValue {
    if (depth > max_array_depth) return error.ArrayNestingTooDeep;
    return switch (vt) {
        .u8 => .{ .u8 = try cur.readInt(u8) },
        .i8 => .{ .i8 = try cur.readInt(i8) },
        .u16 => .{ .u16 = try cur.readInt(u16) },
        .i16 => .{ .i16 = try cur.readInt(i16) },
        .u32 => .{ .u32 = try cur.readInt(u32) },
        .i32 => .{ .i32 = try cur.readInt(i32) },
        .f32 => .{ .f32 = try cur.readFloat(f32) },
        .bool => .{ .bool = try cur.readBool() },
        .string => .{ .string = try cur.readString() },
        .u64 => .{ .u64 = try cur.readInt(u64) },
        .i64 => .{ .i64 = try cur.readInt(i64) },
        .f64 => .{ .f64 = try cur.readFloat(f64) },
        .array => {
            const elem_type_raw = try cur.readInt(u32);
            const elem_type = tryEnumFromInt(MetaValueType, elem_type_raw) orelse
                return error.UnknownMetadataValueType;
            const count = try cur.readInt(u64);
            const start = cur.off;
            // Walk the array to compute its byte length without materializing
            // each element. Strings and nested arrays have variable size, so a
            // walk is unavoidable; we don't store the values, just advance.
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                _ = try readMetaValue(cur, elem_type, depth + 1);
            }
            return .{ .array = .{
                .elem_type = elem_type,
                .count = count,
                .raw = cur.buf[start..cur.off],
            } };
        },
    };
}

fn readMetaKv(cur: *Cursor) ParseError!MetaKv {
    const key = try cur.readString();
    const vt_raw = try cur.readInt(u32);
    const vt = tryEnumFromInt(MetaValueType, vt_raw) orelse
        return error.UnknownMetadataValueType;
    const value = try readMetaValue(cur, vt, 0);
    return .{ .key = key, .value = value };
}

fn tryEnumFromInt(comptime E: type, raw: anytype) ?E {
    inline for (@typeInfo(E).@"enum".fields) |f| {
        if (raw == f.value) return @enumFromInt(f.value);
    }
    return null;
}

fn alignUp(x: u64, alignment: u64) u64 {
    return (x + alignment - 1) & ~(alignment - 1);
}

/// Parse the GGUF header + metadata + tensor_info table from `mapped`.
/// `arena` is used for the small per-catalog allocations only (tensor and
/// metadata slice arrays plus arena-copied dim arrays). Strings and array
/// payloads continue to borrow from `mapped`.
pub fn parseHeader(arena: Allocator, mapped: []const u8) ParseError!Catalog {
    var cur: Cursor = .{ .buf = mapped, .off = 0 };

    const magic = try cur.readInt(u32);
    if (magic != magic_le) return error.NotAGgufFile;

    const version = try cur.readInt(u32);
    if (version != 2 and version != 3) return error.UnsupportedGgufVersion;

    const tensor_count = try cur.readInt(u64);
    const metadata_kv_count = try cur.readInt(u64);

    const tensor_count_us: usize = std.math.cast(usize, tensor_count) orelse
        return error.UnexpectedEndOfFile;
    const meta_count_us: usize = std.math.cast(usize, metadata_kv_count) orelse
        return error.UnexpectedEndOfFile;

    var metadata = try arena.alloc(MetaKv, meta_count_us);
    var alignment: u32 = 32;
    for (0..meta_count_us) |i| {
        const kv = try readMetaKv(&cur);
        metadata[i] = kv;
        if (std.mem.eql(u8, kv.key, "general.alignment")) {
            switch (kv.value) {
                .u32 => |v| alignment = v,
                else => {},
            }
        }
    }

    var descs = try arena.alloc(TensorDesc, tensor_count_us);
    for (0..tensor_count_us) |i| {
        const name = try cur.readString();
        const n_dims = try cur.readInt(u32);
        const dims = try arena.alloc(u64, n_dims);
        for (dims) |*d| d.* = try cur.readInt(u64);
        const ggml_raw = try cur.readInt(u32);
        const ggml_type: GgmlType = @enumFromInt(ggml_raw);
        const rel_offset = try cur.readInt(u64);

        var numel: u64 = 1;
        for (dims) |d| {
            const ov = @mulWithOverflow(numel, d);
            if (ov[1] != 0) return error.UnsupportedTensorBlockShape;
            numel = ov[0];
        }

        var size_bytes: u64 = 0;
        if (blockInfo(ggml_type)) |bi| {
            if (numel % bi.elements != 0) return error.UnsupportedTensorBlockShape;
            size_bytes = (numel / bi.elements) * bi.bytes;
        }

        descs[i] = .{
            .name = name,
            .dims = dims,
            .ggml_type = ggml_type,
            .rel_offset = rel_offset,
            .size_bytes = size_bytes,
        };
    }

    const data_segment_start = alignUp(cur.off, alignment);
    if (data_segment_start > mapped.len) return error.UnexpectedEndOfFile;

    return .{
        .descs = descs,
        .metadata = metadata,
        .data_segment_start = data_segment_start,
        .alignment = alignment,
    };
}

// -- tests ----------------------------------------------------------------

const Builder = struct {
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Builder, gpa: Allocator) void {
        self.list.deinit(gpa);
    }

    fn append(self: *Builder, gpa: Allocator, bytes: []const u8) !void {
        try self.list.appendSlice(gpa, bytes);
    }

    fn writeInt(self: *Builder, gpa: Allocator, comptime T: type, v: T) !void {
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        try self.append(gpa, &buf);
    }

    fn writeStr(self: *Builder, gpa: Allocator, s: []const u8) !void {
        try self.writeInt(gpa, u64, @intCast(s.len));
        try self.append(gpa, s);
    }
};

test "parseHeader: minimal valid v3 file with one F32 tensor" {
    const gpa = std.testing.allocator;
    var b: Builder = .{};
    defer b.deinit(gpa);

    // Header.
    try b.writeInt(gpa, u32, magic_le);
    try b.writeInt(gpa, u32, 3); // version
    try b.writeInt(gpa, u64, 1); // tensor_count
    try b.writeInt(gpa, u64, 2); // metadata_kv_count

    // metadata[0]: "general.architecture" = STRING "test"
    try b.writeStr(gpa, "general.architecture");
    try b.writeInt(gpa, u32, @intFromEnum(MetaValueType.string));
    try b.writeStr(gpa, "test");

    // metadata[1]: "general.alignment" = U32 32
    try b.writeStr(gpa, "general.alignment");
    try b.writeInt(gpa, u32, @intFromEnum(MetaValueType.u32));
    try b.writeInt(gpa, u32, 32);

    // tensor_info[0]: name="w" n_dims=2 dims=[4,4] ggml_type=F32 rel_offset=0
    try b.writeStr(gpa, "w");
    try b.writeInt(gpa, u32, 2);
    try b.writeInt(gpa, u64, 4);
    try b.writeInt(gpa, u64, 4);
    try b.writeInt(gpa, u32, @intFromEnum(GgmlType.f32));
    try b.writeInt(gpa, u64, 0);

    // Pad to alignment 32.
    const before_pad = b.list.items.len;
    const padded = alignUp(before_pad, 32);
    try b.list.appendNTimes(gpa, 0, padded - before_pad);

    // 64 bytes of tensor data (4×4 f32 zeros).
    try b.list.appendNTimes(gpa, 0, 64);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const cat = try parseHeader(arena.allocator(), b.list.items);
    try std.testing.expectEqual(@as(usize, 1), cat.descs.len);
    try std.testing.expectEqual(@as(usize, 2), cat.metadata.len);
    try std.testing.expectEqual(@as(u32, 32), cat.alignment);
    try std.testing.expectEqualStrings("w", cat.descs[0].name);
    try std.testing.expectEqual(@as(usize, 2), cat.descs[0].dims.len);
    try std.testing.expectEqual(@as(u64, 4), cat.descs[0].dims[0]);
    try std.testing.expectEqual(@as(u64, 4), cat.descs[0].dims[1]);
    try std.testing.expectEqual(GgmlType.f32, cat.descs[0].ggml_type);
    try std.testing.expectEqual(@as(u64, 64), cat.descs[0].size_bytes);
    try std.testing.expectEqual(padded, cat.data_segment_start);
    try std.testing.expectEqualStrings("test", cat.metadata[0].value.string);
    try std.testing.expectEqual(@as(u32, 32), cat.metadata[1].value.u32);
}

test "parseHeader: rejects bad magic" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var buf: [24]u8 = .{0} ** 24;
    std.mem.writeInt(u32, buf[0..4], 0xdeadbeef, .little);
    try std.testing.expectError(error.NotAGgufFile, parseHeader(arena.allocator(), &buf));
}

test "parseHeader: rejects unsupported version" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var buf: [24]u8 = .{0} ** 24;
    std.mem.writeInt(u32, buf[0..4], magic_le, .little);
    std.mem.writeInt(u32, buf[4..8], 1, .little);
    try std.testing.expectError(error.UnsupportedGgufVersion, parseHeader(arena.allocator(), &buf));
}

test "blockInfo: q4_k matches ggml-quants constants" {
    const bi = blockInfo(.q4_k).?;
    try std.testing.expectEqual(@as(u64, 256), bi.elements);
    try std.testing.expectEqual(@as(u64, 144), bi.bytes);
}
