//! Safetensors v1 parser, mmap-friendly.
//!
//! File layout:
//!   [0..8)  : little-endian u64 — JSON header byte length (N)
//!   [8..8+N): UTF-8 JSON object — tensor descriptors + optional `__metadata__`
//!   [8+N..) : raw tensor data; each tensor's `data_offsets` are relative to
//!             this byte (i.e. data section start, NOT file start).
//!
//! Only descriptor parsing happens up front; tensor bytes stay in the mmap.
//! `data_segment_start` lets `tensorSliceFromDesc` translate descriptor
//! offsets into mmap byte ranges without copies.
//!
//! Sharded checkpoints (`model.safetensors.index.json` + `model-NNNNN-of-MMMMM
//! .safetensors`) are not parsed by this module — see `core/hf_loader.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParseError = error{
    HeaderTruncated,
    HeaderTooLarge,
    BadJson,
    UnknownDtype,
    BadShape,
    BadOffsets,
    OutOfMemory,
};

pub const Dtype = enum {
    f32,
    f16,
    bf16,
    u8_,
    i8_,
    u16_,
    i16_,
    u32_,
    i32_,
    u64_,
    i64_,
    bool_,
};

pub fn dtypeBytes(d: Dtype) usize {
    return switch (d) {
        .f32, .u32_, .i32_ => 4,
        .f16, .bf16, .u16_, .i16_ => 2,
        .u8_, .i8_, .bool_ => 1,
        .u64_, .i64_ => 8,
    };
}

fn parseDtype(s: []const u8) ParseError!Dtype {
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F16")) return .f16;
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    if (std.mem.eql(u8, s, "U8")) return .u8_;
    if (std.mem.eql(u8, s, "I8")) return .i8_;
    if (std.mem.eql(u8, s, "U16")) return .u16_;
    if (std.mem.eql(u8, s, "I16")) return .i16_;
    if (std.mem.eql(u8, s, "U32")) return .u32_;
    if (std.mem.eql(u8, s, "I32")) return .i32_;
    if (std.mem.eql(u8, s, "U64")) return .u64_;
    if (std.mem.eql(u8, s, "I64")) return .i64_;
    if (std.mem.eql(u8, s, "BOOL")) return .bool_;
    return error.UnknownDtype;
}

pub const TensorDesc = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    /// Offset in bytes from the start of the file's data segment, NOT from
    /// the start of the file. Add `data_segment_start` to get the file
    /// offset / mmap offset.
    rel_offset_start: u64,
    rel_offset_end: u64,
};

pub const Catalog = struct {
    /// First byte of tensor-data region (= 8 + header_len).
    data_segment_start: u64,
    descs: []TensorDesc,

    pub fn find(self: *const Catalog, name: []const u8) ?*const TensorDesc {
        for (self.descs) |*d| {
            if (std.mem.eql(u8, d.name, name)) return d;
        }
        return null;
    }
};

/// Parse the header of a safetensors file. `mapped` must be the entire file
/// (or at least the first 8 + header_len bytes). All allocations land in
/// `arena`; the caller is responsible for the arena's lifetime.
pub fn parseHeader(arena: Allocator, mapped: []const u8) ParseError!Catalog {
    if (mapped.len < 8) return error.HeaderTruncated;
    const header_len = std.mem.readInt(u64, mapped[0..8], .little);
    if (header_len > 100 * 1024 * 1024) return error.HeaderTooLarge; // 100 MB sanity cap
    const header_end_u64 = @as(u64, 8) + header_len;
    if (header_end_u64 > mapped.len) return error.HeaderTruncated;
    const header_end: usize = @intCast(header_end_u64);
    const json_bytes = mapped[8..header_end];

    const parsed = std.json.parseFromSlice(std.json.Value, arena, json_bytes, .{}) catch return error.BadJson;
    // Note: parsed lives in arena; do not deinit before reading.
    const root = parsed.value;
    if (root != .object) return error.BadJson;

    var descs: std.ArrayList(TensorDesc) = .empty;
    try descs.ensureTotalCapacity(arena, root.object.count());

    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const v = kv.value_ptr.*;
        if (v != .object) return error.BadJson;

        const dtype_v = v.object.get("dtype") orelse return error.BadJson;
        if (dtype_v != .string) return error.BadJson;
        const dtype = try parseDtype(dtype_v.string);

        const shape_v = v.object.get("shape") orelse return error.BadJson;
        if (shape_v != .array) return error.BadShape;
        const dims = try arena.alloc(u64, shape_v.array.items.len);
        for (shape_v.array.items, 0..) |item, i| {
            if (item != .integer) return error.BadShape;
            const x = item.integer;
            if (x < 0) return error.BadShape;
            dims[i] = @intCast(x);
        }

        const off_v = v.object.get("data_offsets") orelse return error.BadOffsets;
        if (off_v != .array or off_v.array.items.len != 2) return error.BadOffsets;
        const off_start_v = off_v.array.items[0];
        const off_end_v = off_v.array.items[1];
        if (off_start_v != .integer or off_end_v != .integer) return error.BadOffsets;
        if (off_start_v.integer < 0 or off_end_v.integer < off_start_v.integer) return error.BadOffsets;
        const off_start: u64 = @intCast(off_start_v.integer);
        const off_end: u64 = @intCast(off_end_v.integer);

        try descs.append(arena, .{
            .name = try arena.dupe(u8, kv.key_ptr.*),
            .dtype = dtype,
            .shape = dims,
            .rel_offset_start = off_start,
            .rel_offset_end = off_end,
        });
    }

    return .{
        .data_segment_start = header_end_u64,
        .descs = try descs.toOwnedSlice(arena),
    };
}

// -- tests ----------------------------------------------------------------

test "parseHeader: tiny synthesized file with two tensors" {
    const gpa = std.testing.allocator;

    // Build a JSON header for two tensors:
    //   "x": F32 shape [4],   data_offsets [0, 16]
    //   "y": F16 shape [2,3], data_offsets [16, 28]
    const json =
        \\{"__metadata__":{"format":"pt"},
        \\"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
        \\"y":{"dtype":"F16","shape":[2,3],"data_offsets":[16,28]}}
    ;

    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);

    const json_len: u64 = json.len;
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, json_len, .little);
    try bytes.appendSlice(gpa, &hdr);
    try bytes.appendSlice(gpa, json);
    // 28 bytes of fake data.
    try bytes.appendNTimes(gpa, 0xCC, 28);

    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const catalog = try parseHeader(arena.allocator(), bytes.items);
    try std.testing.expectEqual(@as(usize, 2), catalog.descs.len);
    try std.testing.expectEqual(@as(u64, 8 + json_len), catalog.data_segment_start);

    const x = catalog.find("x") orelse return error.TensorNotFound;
    try std.testing.expectEqual(Dtype.f32, x.dtype);
    try std.testing.expectEqual(@as(usize, 1), x.shape.len);
    try std.testing.expectEqual(@as(u64, 4), x.shape[0]);
    try std.testing.expectEqual(@as(u64, 0), x.rel_offset_start);
    try std.testing.expectEqual(@as(u64, 16), x.rel_offset_end);

    const y = catalog.find("y") orelse return error.TensorNotFound;
    try std.testing.expectEqual(Dtype.f16, y.dtype);
    try std.testing.expectEqual(@as(usize, 2), y.shape.len);
    try std.testing.expectEqual(@as(u64, 2), y.shape[0]);
    try std.testing.expectEqual(@as(u64, 3), y.shape[1]);
}

test "parseHeader: rejects truncated header" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const tiny: [4]u8 = .{ 0, 0, 0, 0 };
    try std.testing.expectError(error.HeaderTruncated, parseHeader(arena.allocator(), &tiny));
}

test "parseHeader: rejects header_len > file size" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, 9999, .little);
    try std.testing.expectError(error.HeaderTruncated, parseHeader(arena.allocator(), &hdr));
}

test "parseHeader: rejects unknown dtype" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const json =
        \\{"x":{"dtype":"FLOAT99","shape":[1],"data_offsets":[0,4]}}
    ;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, json.len, .little);
    try bytes.appendSlice(gpa, &hdr);
    try bytes.appendSlice(gpa, json);
    try bytes.appendNTimes(gpa, 0, 4);

    try std.testing.expectError(error.UnknownDtype, parseHeader(arena.allocator(), bytes.items));
}
