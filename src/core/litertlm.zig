//! `.litertlm` container reader (Phase A — header + section index).
//!
//! File layout (verified against `python/litert_lm_builder/litertlm_builder.py`
//! and `schema/core/litertlm_header_schema.fbs` in the upstream repo):
//!
//!   offset 0:  "LITERTLM" magic (8 bytes)
//!   offset 8:  major_version: u32 LE
//!   offset 12: minor_version: u32 LE
//!   offset 16: patch_version: u32 LE
//!   offset 20: padding (4 bytes)
//!   offset 24: header_end_offset: u64 LE
//!   offset 32: flatbuffer metadata (LiteRTLMMetaData root table)
//!   ...        padded to next 16 KB boundary
//!   offset N:  section 0 raw bytes
//!   ...
//!
//! Sections are stored at file offsets recorded in the flatbuffer
//! header's `SectionMetadata.objects`. Each `SectionObject` carries
//! `begin_offset` / `end_offset` (absolute file offsets) +
//! `data_type` (`AnySectionDataType` enum).
//!
//! This reader stops at indexing — it does NOT parse the inner TFLite
//! tensor data. That's Phase B/C. Today the bundle is useful for
//! extracting the tokenizer + chat-template metadata so vantablack can
//! at least describe what's in a `.litertlm` file.

const std = @import("std");

const flatbuffer = @import("flatbuffer.zig");

pub const Error = error{
    Truncated,
    BadMagic,
    UnsupportedMajorVersion,
    InvalidHeader,
    OutOfBounds,
    InvalidOffset,
    UnalignedOffset,
};

pub const magic: []const u8 = "LITERTLM";

/// Maximum major version this reader understands. Newer files MAY work
/// (additive minor version bumps are backward compatible per the
/// upstream SemVer rules) but a major bump means the layout changed.
pub const supported_major_version: u32 = 1;

pub const header_begin_byte_offset: u64 = 32;
pub const header_end_location_byte_offset: u64 = 24;

pub const SectionDataType = enum(u8) {
    none = 0,
    generic_binary_data = 1,
    deprecated = 2,
    tflite_model = 3,
    sp_tokenizer = 4,
    llm_metadata_proto = 5,
    hf_tokenizer_zlib = 6,
    tflite_weights = 7,
    _, // open enum — new types appear in minor bumps

    pub fn name(self: SectionDataType) []const u8 {
        return switch (self) {
            .none => "NONE",
            .generic_binary_data => "GenericBinaryData",
            .deprecated => "Deprecated",
            .tflite_model => "TFLiteModel",
            .sp_tokenizer => "SP_Tokenizer",
            .llm_metadata_proto => "LlmMetadataProto",
            .hf_tokenizer_zlib => "HF_Tokenizer_Zlib",
            .tflite_weights => "TFLiteWeights",
            _ => "Unknown",
        };
    }
};

pub const Section = struct {
    data_type: SectionDataType,
    begin: u64,
    end: u64,

    pub fn size(self: Section) u64 {
        return self.end - self.begin;
    }
};

pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,
};

/// Indexed view over a `.litertlm` file. `bytes` is the entire file
/// mapped/loaded into memory. `sections` is a heap-allocated array of
/// section descriptors; the bytes for each section live at
/// `bytes[s.begin..s.end]`.
pub const Bundle = struct {
    bytes: []const u8,
    version: Version,
    sections: []Section,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, bytes: []const u8) Error!Bundle {
        if (bytes.len < header_begin_byte_offset) return error.Truncated;
        if (!std.mem.eql(u8, bytes[0..8], magic)) return error.BadMagic;

        const major = std.mem.readInt(u32, bytes[8..12], .little);
        const minor = std.mem.readInt(u32, bytes[12..16], .little);
        const patch = std.mem.readInt(u32, bytes[16..20], .little);
        if (major > supported_major_version) return error.UnsupportedMajorVersion;

        const header_end = std.mem.readInt(u64, bytes[24..32], .little);
        if (header_end <= header_begin_byte_offset or header_end > bytes.len) {
            return error.InvalidHeader;
        }
        const fb_bytes = bytes[header_begin_byte_offset..header_end];

        const buf = flatbuffer.Buffer.init(fb_bytes);
        const root = buf.root() catch return error.InvalidHeader;
        // LiteRTLMMetaData.section_metadata (slot 1)
        const section_meta = root.readTable(1) catch return error.InvalidHeader;
        const section_meta_t = section_meta orelse return error.InvalidHeader;
        // SectionMetadata.objects (slot 0)
        const objects = section_meta_t.readVector(0) catch return error.InvalidHeader;
        const objects_v = objects orelse return error.InvalidHeader;

        const sections = allocator.alloc(Section, objects_v.len) catch return error.OutOfBounds;
        errdefer allocator.free(sections);

        var i: u32 = 0;
        while (i < objects_v.len) : (i += 1) {
            const obj = objects_v.tableAt(i) catch return error.InvalidHeader;
            // SectionObject fields:
            //   slot 0: items (vector of KeyValuePair) — skipped here
            //   slot 1: begin_offset (ulong)
            //   slot 2: end_offset (ulong)
            //   slot 3: data_type (ubyte enum)
            const begin = obj.readU64(1, 0) catch return error.InvalidHeader;
            const end = obj.readU64(2, 0) catch return error.InvalidHeader;
            const dt_u8 = obj.readU8(3, 0) catch return error.InvalidHeader;
            sections[i] = .{
                .data_type = @enumFromInt(dt_u8),
                .begin = begin,
                .end = end,
            };
        }

        return .{
            .bytes = bytes,
            .version = .{ .major = major, .minor = minor, .patch = patch },
            .sections = sections,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Bundle) void {
        self.allocator.free(self.sections);
        self.* = undefined;
    }

    /// Locate the first section of the given type. Returns null if absent.
    pub fn findSection(self: *const Bundle, data_type: SectionDataType) ?Section {
        for (self.sections) |s| {
            if (s.data_type == data_type) return s;
        }
        return null;
    }

    /// Bytes for `section`. Slices the underlying file mapping; no copy.
    pub fn sectionBytes(self: *const Bundle, section: Section) Error![]const u8 {
        if (section.end > self.bytes.len) return error.OutOfBounds;
        if (section.begin > section.end) return error.InvalidOffset;
        return self.bytes[section.begin..section.end];
    }

    /// Decompress the `HF_Tokenizer_Zlib` section into a freshly-allocated
    /// byte slice. Caller frees with `allocator.free`. Returns
    /// `error.MissingSection` if absent.
    ///
    /// Section payload layout (from `python/litert_lm_builder/litertlm_builder.py`):
    ///   [0..8]   uncompressed_size: u64 LE
    ///   [8..]    zlib-compressed tokenizer.json bytes
    pub fn extractHfTokenizerJson(self: *const Bundle, allocator: std.mem.Allocator) ![]u8 {
        const section = self.findSection(.hf_tokenizer_zlib) orelse return error.MissingSection;
        const raw = try self.sectionBytes(section);
        if (raw.len < 8) return error.InvalidHeader;
        const uncompressed_size = std.mem.readInt(u64, raw[0..8], .little);
        const compressed = raw[8..];

        var input_reader: std.Io.Reader = .fixed(compressed);
        var out: std.Io.Writer.Allocating = try .initCapacity(allocator, @intCast(uncompressed_size));
        errdefer out.deinit();

        var decompress: std.compress.flate.Decompress = .init(&input_reader, .zlib, &.{});
        _ = try decompress.reader.streamRemaining(&out.writer);

        return out.toOwnedSlice();
    }
};

// -- tests ----------------------------------------------------------------

test "Bundle rejects bad magic" {
    var bytes: [64]u8 = .{0} ** 64;
    @memcpy(bytes[0..8], "NOTITLM\x00");
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.BadMagic, Bundle.init(gpa, &bytes));
}

test "Bundle rejects truncated input" {
    var bytes: [4]u8 = .{ 0, 0, 0, 0 };
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.Truncated, Bundle.init(gpa, &bytes));
}

test "Bundle rejects future major version" {
    var bytes: [64]u8 = .{0} ** 64;
    @memcpy(bytes[0..8], magic);
    // Major version 9 — beyond supported.
    std.mem.writeInt(u32, bytes[8..12], 9, .little);
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.UnsupportedMajorVersion, Bundle.init(gpa, &bytes));
}

test "Bundle rejects header_end out of range" {
    var bytes: [64]u8 = .{0} ** 64;
    @memcpy(bytes[0..8], magic);
    std.mem.writeInt(u32, bytes[8..12], 1, .little);
    // header_end = 99999, file is only 64 bytes
    std.mem.writeInt(u64, bytes[24..32], 99999, .little);
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidHeader, Bundle.init(gpa, &bytes));
}

test "SectionDataType.name basic mapping" {
    try std.testing.expectEqualStrings("TFLiteModel", SectionDataType.tflite_model.name());
    try std.testing.expectEqualStrings("HF_Tokenizer_Zlib", SectionDataType.hf_tokenizer_zlib.name());
    try std.testing.expectEqualStrings("NONE", SectionDataType.none.name());
}
