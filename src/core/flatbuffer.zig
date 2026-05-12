//! Minimal read-only flatbuffer parser.
//!
//! Supports only what the `.litertlm` header schema needs:
//! - Root table dereference
//! - Scalar field read (u8/u32/u64/i32 via vtable offset)
//! - String field
//! - Vector of tables / scalars
//! - Optional fields (vtable slot == 0 → absent)
//!
//! No writer, no schema codegen, no unions. The litertlm `VData` union
//! is read by branching on a side-channel `value_type` enum field that
//! the schema places next to the union pointer — see `litertlm.zig`.
//!
//! References:
//! - https://flatbuffers.dev/internals.html
//! - Apache 2.0 reference impl (flatbuffers/include/flatbuffers/base.h)
//!
//! Layout primer:
//!   buf:
//!     [0..4]   root_uoffset → points to root table
//!     ...
//!     <table>:
//!       [0..4] soffset (signed) — relative offset BACKWARD to vtable
//!       [4..]  inline scalar fields per vtable
//!     <vtable>:
//!       [0..2] vtable_byte_size: u16
//!       [2..4] inline_byte_size: u16
//!       [4..]  field slots: u16 each, offset within table or 0 if absent
//!     <string>:
//!       [0..4] length: u32
//!       [4..]  utf8 bytes, terminator byte (not counted)
//!     <vector>:
//!       [0..4] length: u32
//!       [4..]  N elements (size depends on element type)

const std = @import("std");

pub const Error = error{
    OutOfBounds,
    InvalidOffset,
    UnalignedOffset,
};

/// View over a flatbuffer-encoded byte slice. `bytes` is the entire
/// buffer; offsets are absolute byte indices into it.
pub const Buffer = struct {
    bytes: []const u8,

    pub fn init(bytes: []const u8) Buffer {
        return .{ .bytes = bytes };
    }

    /// Root table offset. Caller passes `0` (the start of the buffer).
    /// The u32 at that location is the absolute offset of the root table.
    pub fn root(self: Buffer) Error!Table {
        const root_off = try self.readU32(0);
        return self.tableAt(root_off);
    }

    /// Read a u8 at absolute offset.
    pub fn readU8(self: Buffer, off: u32) Error!u8 {
        if (off >= self.bytes.len) return error.OutOfBounds;
        return self.bytes[off];
    }

    /// Read a little-endian u16 at absolute offset.
    pub fn readU16(self: Buffer, off: u32) Error!u16 {
        if (@as(usize, off) + 2 > self.bytes.len) return error.OutOfBounds;
        return std.mem.readInt(u16, self.bytes[off..][0..2], .little);
    }

    /// Read a little-endian u32 at absolute offset.
    pub fn readU32(self: Buffer, off: u32) Error!u32 {
        if (@as(usize, off) + 4 > self.bytes.len) return error.OutOfBounds;
        return std.mem.readInt(u32, self.bytes[off..][0..4], .little);
    }

    pub fn readI32(self: Buffer, off: u32) Error!i32 {
        if (@as(usize, off) + 4 > self.bytes.len) return error.OutOfBounds;
        return std.mem.readInt(i32, self.bytes[off..][0..4], .little);
    }

    pub fn readU64(self: Buffer, off: u32) Error!u64 {
        if (@as(usize, off) + 8 > self.bytes.len) return error.OutOfBounds;
        return std.mem.readInt(u64, self.bytes[off..][0..8], .little);
    }

    /// Build a Table view at the given absolute offset. The table's
    /// first 4 bytes are a signed offset BACKWARD to its vtable.
    pub fn tableAt(self: Buffer, off: u32) Error!Table {
        if (@as(usize, off) + 4 > self.bytes.len) return error.OutOfBounds;
        const soffset = try self.readI32(off);
        // soffset is the offset from the table to its vtable.
        // vtable_abs = table_abs - soffset (per flatbuffers spec).
        const ioff: i64 = @as(i64, off) - @as(i64, soffset);
        if (ioff < 0 or ioff > std.math.maxInt(u32)) return error.InvalidOffset;
        const vtable_off: u32 = @intCast(ioff);
        const vtable_size = try self.readU16(vtable_off);
        if (vtable_size < 4) return error.InvalidOffset;
        return .{ .buf = self, .table_off = off, .vtable_off = vtable_off, .vtable_size = vtable_size };
    }

    /// Read a length-prefixed byte slice at `off`. Strings + byte
    /// vectors share this layout.
    pub fn bytesAt(self: Buffer, off: u32) Error![]const u8 {
        const len = try self.readU32(off);
        const start: usize = @as(usize, off) + 4;
        const end = start + @as(usize, len);
        if (end > self.bytes.len) return error.OutOfBounds;
        return self.bytes[start..end];
    }

    /// Vector view at `off`. `elem_size` is the byte stride of each
    /// element (4 for uoffset-pointed tables/strings, 1 for u8, etc.).
    pub fn vectorAt(self: Buffer, off: u32) Error!Vector {
        const len = try self.readU32(off);
        return .{ .buf = self, .data_off = off + 4, .len = len };
    }
};

pub const Table = struct {
    buf: Buffer,
    /// Absolute offset of the table's first byte.
    table_off: u32,
    /// Absolute offset of the table's vtable.
    vtable_off: u32,
    /// Vtable byte size (includes the 4-byte header).
    vtable_size: u16,

    /// Returns the in-table byte offset of the field identified by
    /// `field_id`, or 0 if the field is absent (caller must treat 0 as
    /// "use the field's default value").
    ///
    /// `field_id` is the field's slot index in the vtable starting at
    /// 0 (NOT the schema-level @id attribute; for our hand-written
    /// schema mirror, vtable slot order matches schema declaration
    /// order).
    pub fn fieldOffset(self: Table, field_id: u32) Error!u32 {
        // First two u16 slots are vtable_size + inline_size. Field
        // slots start at byte offset 4 (vtable-relative).
        const slot_byte_off: u32 = 4 + field_id * 2;
        if (slot_byte_off + 2 > self.vtable_size) return 0; // absent
        const field_off_u16 = try self.buf.readU16(self.vtable_off + slot_byte_off);
        return field_off_u16;
    }

    /// Read a scalar u8 field. Returns `default` if absent.
    pub fn readU8(self: Table, field_id: u32, default: u8) Error!u8 {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return default;
        return self.buf.readU8(self.table_off + off);
    }

    pub fn readU32(self: Table, field_id: u32, default: u32) Error!u32 {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return default;
        return self.buf.readU32(self.table_off + off);
    }

    pub fn readU64(self: Table, field_id: u32, default: u64) Error!u64 {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return default;
        return self.buf.readU64(self.table_off + off);
    }

    /// Read a string field. Returns null if absent.
    pub fn readString(self: Table, field_id: u32) Error!?[]const u8 {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return null;
        // Field slot stores a uoffset relative to itself, pointing to the string.
        const slot_abs = self.table_off + off;
        const rel = try self.buf.readU32(slot_abs);
        return try self.buf.bytesAt(slot_abs + rel);
    }

    /// Read a table field. Returns null if absent.
    pub fn readTable(self: Table, field_id: u32) Error!?Table {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return null;
        const slot_abs = self.table_off + off;
        const rel = try self.buf.readU32(slot_abs);
        return try self.buf.tableAt(slot_abs + rel);
    }

    /// Read a vector field. Returns null if absent.
    pub fn readVector(self: Table, field_id: u32) Error!?Vector {
        const off = try self.fieldOffset(field_id);
        if (off == 0) return null;
        const slot_abs = self.table_off + off;
        const rel = try self.buf.readU32(slot_abs);
        return try self.buf.vectorAt(slot_abs + rel);
    }
};

pub const Vector = struct {
    buf: Buffer,
    /// Absolute offset of the first element.
    data_off: u32,
    /// Element count.
    len: u32,

    /// Read element `i` as a uoffset and resolve to a Table.
    pub fn tableAt(self: Vector, i: u32) Error!Table {
        if (i >= self.len) return error.OutOfBounds;
        const elem_abs = self.data_off + i * 4;
        const rel = try self.buf.readU32(elem_abs);
        return self.buf.tableAt(elem_abs + rel);
    }

    /// Read element `i` as a uoffset and resolve to a string.
    pub fn stringAt(self: Vector, i: u32) Error![]const u8 {
        if (i >= self.len) return error.OutOfBounds;
        const elem_abs = self.data_off + i * 4;
        const rel = try self.buf.readU32(elem_abs);
        return self.buf.bytesAt(elem_abs + rel);
    }

    /// Read element `i` as an inline u8.
    pub fn u8At(self: Vector, i: u32) Error!u8 {
        if (i >= self.len) return error.OutOfBounds;
        return self.buf.readU8(self.data_off + i);
    }
};

// -- tests ----------------------------------------------------------------

test "Buffer.readU32 little-endian" {
    var bytes = [_]u8{ 0x78, 0x56, 0x34, 0x12 };
    const b = Buffer.init(&bytes);
    try std.testing.expectEqual(@as(u32, 0x12345678), try b.readU32(0));
}

test "Buffer.bytesAt length-prefixed slice" {
    var bytes = [_]u8{ 0x05, 0, 0, 0, 'h', 'e', 'l', 'l', 'o', 0 };
    const b = Buffer.init(&bytes);
    const slice = try b.bytesAt(0);
    try std.testing.expectEqualStrings("hello", slice);
}

test "Buffer.tableAt vtable round-trip" {
    // Hand-built minimal flatbuffer:
    //   table at offset 16 has vtable at offset 4 (soffset=12)
    //   vtable size=6, inline size=8
    //   field 0 lives at table-offset 4 (a u32 = 0xDEADBEEF)
    //
    //   layout:
    //   off  bytes
    //   0    root uoffset = 16 (to table)
    //   4    vtable_size = 6
    //   6    inline_size = 8
    //   8    field 0 offset = 4
    //   10   padding (vtable end)
    //   16   soffset = 12 (table - vtable = 16 - 4 = 12)
    //   20   field 0 value = 0xDEADBEEF
    var bytes = [_]u8{
        16, 0, 0, 0, // 0:  root uoffset
        6, 0, // 4:  vtable_size = 6
        8, 0, // 6:  inline_size = 8
        4, 0, // 8:  field 0 vtable slot = table-offset 4
        0, 0, // 10: pad
        0, 0, 0, 0, // 12: pad to align table at 16
        12, 0, 0, 0, // 16: soffset = 12
        0xEF, 0xBE, 0xAD, 0xDE, // 20: field 0 value
    };
    const b = Buffer.init(&bytes);
    const root_t = try b.root();
    try std.testing.expectEqual(@as(u32, 16), root_t.table_off);
    try std.testing.expectEqual(@as(u32, 4), root_t.vtable_off);
    try std.testing.expectEqual(@as(u16, 6), root_t.vtable_size);
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), try root_t.readU32(0, 0));
}
