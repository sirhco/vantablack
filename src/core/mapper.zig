//! `mmap`-backed model file owner.
//!
//! `ModelMapper.init` opens the file via `std.Io.Dir`, calls `std.posix.mmap`
//! directly on the underlying fd to map the whole file PROT_READ + MAP_PRIVATE,
//! then closes the fd (the kernel keeps the mapping alive). The catalog is
//! parsed in-place out of the mapped region into an arena.
//!
//! Lifetimes: `arena.deinit()` frees the catalog and dim arrays.
//! `std.posix.munmap` releases the mapping.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const parser = @import("parser.zig");

pub const min_header_bytes: usize = 24; // magic + version + tensor_count + kv_count

pub const InitError = error{
    FileEmpty,
    FileTooSmall,
    FileTooLarge,
} || Io.File.OpenError ||
    Io.File.LengthError ||
    std.posix.MMapError ||
    parser.ParseError;

pub const TensorSliceError = error{
    TensorNotFound,
    TensorOutOfBounds,
};

pub const ModelMapper = struct {
    arena: std.heap.ArenaAllocator,
    mapped: []align(std.heap.page_size_min) const u8,
    catalog: parser.Catalog,

    pub fn init(allocator: Allocator, io: Io, abs_path: []const u8) InitError!ModelMapper {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        const file = try Io.Dir.openFileAbsolute(io, abs_path, .{});
        defer file.close(io);

        const len = try file.length(io);
        if (len == 0) return error.FileEmpty;
        if (len < min_header_bytes) return error.FileTooSmall;
        const len_us: usize = std.math.cast(usize, len) orelse return error.FileTooLarge;

        const mapped = try std.posix.mmap(
            null,
            len_us,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        const catalog = try parser.parseHeader(arena.allocator(), mapped);

        return .{
            .arena = arena,
            .mapped = mapped,
            .catalog = catalog,
        };
    }

    pub fn deinit(self: *ModelMapper) void {
        std.posix.munmap(self.mapped);
        self.arena.deinit();
        self.* = undefined;
    }

    /// Returns a slice into the mapped region for the named tensor's raw bytes.
    /// Bounds-checked against the mapped region.
    pub fn tensorSlice(self: *const ModelMapper, name: []const u8) TensorSliceError![]const u8 {
        const desc = self.catalog.find(name) orelse return error.TensorNotFound;
        return self.tensorSliceFromDesc(desc);
    }

    pub fn tensorSliceFromDesc(self: *const ModelMapper, desc: *const parser.TensorDesc) TensorSliceError![]const u8 {
        const start_ov = @addWithOverflow(self.catalog.data_segment_start, desc.rel_offset);
        if (start_ov[1] != 0) return error.TensorOutOfBounds;
        const start = start_ov[0];
        const end_ov = @addWithOverflow(start, desc.size_bytes);
        if (end_ov[1] != 0) return error.TensorOutOfBounds;
        const end = end_ov[0];
        if (end > self.mapped.len) return error.TensorOutOfBounds;
        const start_us: usize = std.math.cast(usize, start) orelse return error.TensorOutOfBounds;
        const end_us: usize = std.math.cast(usize, end) orelse return error.TensorOutOfBounds;
        return self.mapped[start_us..end_us];
    }
};

// -- tests ----------------------------------------------------------------

test "ModelMapper round-trips a tiny synthesized GGUF file" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    // Synthesize the same minimal GGUF byte buffer used in parser tests, then
    // write to a tmpfile, mmap, and verify tensor slice.
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);

    const writeInt = struct {
        fn f(list: *std.ArrayList(u8), allocator: Allocator, comptime T: type, v: T) !void {
            var buf: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &buf, v, .little);
            try list.appendSlice(allocator, &buf);
        }
    }.f;
    const writeStr = struct {
        fn f(list: *std.ArrayList(u8), allocator: Allocator, s: []const u8) !void {
            var buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &buf, @intCast(s.len), .little);
            try list.appendSlice(allocator, &buf);
            try list.appendSlice(allocator, s);
        }
    }.f;

    try writeInt(&bytes, gpa, u32, parser.magic_le);
    try writeInt(&bytes, gpa, u32, 3);
    try writeInt(&bytes, gpa, u64, 1); // tensor_count
    try writeInt(&bytes, gpa, u64, 1); // metadata_kv_count
    try writeStr(&bytes, gpa, "general.alignment");
    try writeInt(&bytes, gpa, u32, @intFromEnum(parser.MetaValueType.u32));
    try writeInt(&bytes, gpa, u32, 32);
    try writeStr(&bytes, gpa, "w");
    try writeInt(&bytes, gpa, u32, 2);
    try writeInt(&bytes, gpa, u64, 4);
    try writeInt(&bytes, gpa, u64, 4);
    try writeInt(&bytes, gpa, u32, @intFromEnum(parser.GgmlType.f32));
    try writeInt(&bytes, gpa, u64, 0);

    const before_pad = bytes.items.len;
    const padded = (before_pad + 31) & ~@as(usize, 31);
    try bytes.appendNTimes(gpa, 0, padded - before_pad);
    try bytes.appendNTimes(gpa, 0xAB, 64); // distinguishable payload

    // Write to a tmpfile under cwd.
    const cwd = Io.Dir.cwd();
    const tmp_name = "vantablack-test.gguf.tmp";
    try cwd.writeFile(io, .{ .sub_path = tmp_name, .data = bytes.items });
    defer cwd.deleteFile(io, tmp_name) catch {};

    // Build an absolute path so openFileAbsolute is satisfied.
    const cwd_path = try std.process.currentPathAlloc(io, gpa);
    defer gpa.free(cwd_path);
    const abs_path = try std.fs.path.resolve(gpa, &.{ cwd_path, tmp_name });
    defer gpa.free(abs_path);

    var mapper = try ModelMapper.init(gpa, io, abs_path);
    defer mapper.deinit();

    const slice = try mapper.tensorSlice("w");
    try std.testing.expectEqual(@as(usize, 64), slice.len);
    for (slice) |b| try std.testing.expectEqual(@as(u8, 0xAB), b);

    try std.testing.expectError(error.TensorNotFound, mapper.tensorSlice("missing"));
}
