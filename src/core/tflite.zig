//! TFLite flatbuffer reader (Phase B partial).
//!
//! Enumerates tensors + buffers from the `TFLiteModel` section of a
//! `.litertlm` file (or a standalone `.tflite`). Stops at indexing:
//! caller gets tensor name, shape, dtype, quantization params, and a
//! slice into the raw weight buffer. No operator graph parsing yet.
//!
//! Schema (subset we use):
//!   table Model {
//!     version: uint;                // slot 0
//!     operator_codes: [OperatorCode];//slot 1
//!     subgraphs: [SubGraph];        // slot 2
//!     description: string;          // slot 3
//!     buffers: [Buffer];            // slot 4
//!     ...
//!   }
//!   table SubGraph {
//!     tensors: [Tensor];            // slot 0
//!     inputs: [int];                // slot 1
//!     outputs: [int];               // slot 2
//!     operators: [Operator];        // slot 3
//!     name: string;                 // slot 4
//!   }
//!   table Tensor {
//!     shape: [int];                 // slot 0
//!     type: TensorType (byte);      // slot 1
//!     buffer: uint;                 // slot 2
//!     name: string;                 // slot 3
//!     quantization: QuantParams;    // slot 4
//!     is_variable: bool;            // slot 5
//!     ...
//!   }
//!   table Buffer {
//!     data: [ubyte];                // slot 0
//!     offset: ulong;                // slot 1 (>2GB models)
//!     size: ulong;                  // slot 2
//!   }
//!   table QuantizationParameters {
//!     min: [float];                 // slot 0
//!     max: [float];                 // slot 1
//!     scale: [float];               // slot 2
//!     zero_point: [long];           // slot 3
//!     // details union: slots 4 + 5
//!     quantized_dimension: int;     // slot 6
//!   }
//!
//! Reference:
//!   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/schema/schema.fbs

const std = @import("std");

const flatbuffer = @import("flatbuffer.zig");

pub const Error = flatbuffer.Error || error{
    NoSubgraphs,
    TensorOutOfRange,
    BufferOutOfRange,
    OutOfMemory,
};

/// Mirror of TFLite `enum TensorType : byte`. Open enum — newer TFLite
/// builds add types (INT4, BFLOAT16, INT2, UINT4) and this reader
/// surfaces unknown values via the open variant.
pub const TensorType = enum(i8) {
    float32 = 0,
    float16 = 1,
    int32 = 2,
    uint8 = 3,
    int64 = 4,
    string = 5,
    bool_ = 6,
    int16 = 7,
    complex64 = 8,
    int8 = 9,
    float64 = 10,
    complex128 = 11,
    uint64 = 12,
    resource = 13,
    variant = 14,
    uint32 = 15,
    uint16 = 16,
    int4 = 17,
    bfloat16 = 18,
    int2 = 19,
    uint4 = 20,
    _,

    pub fn name(self: TensorType) []const u8 {
        return switch (self) {
            .float32 => "FLOAT32",
            .float16 => "FLOAT16",
            .int32 => "INT32",
            .uint8 => "UINT8",
            .int64 => "INT64",
            .string => "STRING",
            .bool_ => "BOOL",
            .int16 => "INT16",
            .complex64 => "COMPLEX64",
            .int8 => "INT8",
            .float64 => "FLOAT64",
            .complex128 => "COMPLEX128",
            .uint64 => "UINT64",
            .resource => "RESOURCE",
            .variant => "VARIANT",
            .uint32 => "UINT32",
            .uint16 => "UINT16",
            .int4 => "INT4",
            .bfloat16 => "BFLOAT16",
            .int2 => "INT2",
            .uint4 => "UINT4",
            _ => "UNKNOWN",
        };
    }

    /// Bytes-per-element for fixed-width types. Returns null for
    /// sub-byte (int4/int2/uint4) and variable-width types (string).
    pub fn elementBytes(self: TensorType) ?usize {
        return switch (self) {
            .float32, .int32, .uint32, .complex64 => 4,
            .float16, .int16, .uint16, .bfloat16 => 2,
            .uint8, .int8, .bool_ => 1,
            .int64, .uint64, .float64, .complex128 => 8,
            .int4, .int2, .uint4, .string, .resource, .variant => null,
            _ => null,
        };
    }
};

/// Lightweight tensor descriptor — enough to print or feed into a
/// future weight-binding layer. The `data` slice is borrowed from the
/// underlying flatbuffer (or an external buffer); not owned.
pub const Tensor = struct {
    name: []const u8,
    shape: []const i32,
    dtype: TensorType,
    buffer_index: u32,
    /// Scales array for quantized tensors (per-axis if >1 entry).
    /// Empty for non-quantized.
    scales: []const u8, // raw bytes — re-interpret via `f32At` on the vector
    /// Zero points (parallel to scales). Empty for non-quantized.
    zero_points: []const u8, // raw bytes — re-interpret via `i64At`
    quantized_dimension: i32,
    /// The raw weight bytes (may be empty for activation tensors).
    data: []const u8,
};

pub const Model = struct {
    /// TFLite schema version (typically 3).
    version: u32,
    /// Tensors from subgraph 0 (the "main" subgraph).
    tensors: []Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, bytes: []const u8) Error!Model {
        const buf = flatbuffer.Buffer.init(bytes);
        const model = try buf.root();

        const version = try model.readU32(0, 0);
        const subgraphs = try model.readVector(2);
        const buffers = try model.readVector(4);
        const sg_vec = subgraphs orelse return error.NoSubgraphs;
        if (sg_vec.len == 0) return error.NoSubgraphs;

        // Pre-resolve buffer slices once so each tensor can index in O(1).
        const buf_count = if (buffers) |bv| bv.len else 0;
        const buf_slices = try allocator.alloc([]const u8, buf_count);
        defer allocator.free(buf_slices);
        if (buffers) |bv| {
            var i: u32 = 0;
            while (i < bv.len) : (i += 1) {
                const buf_t = bv.tableAt(i) catch return error.BufferOutOfRange;
                const data_v = buf_t.readVector(0) catch null;
                if (data_v) |v| {
                    buf_slices[i] = v.bytes() catch &.{};
                } else {
                    buf_slices[i] = &.{};
                }
            }
        }

        const sg0 = sg_vec.tableAt(0) catch return error.NoSubgraphs;
        const tensors_v_opt = sg0.readVector(0) catch null;
        const tensors_v = tensors_v_opt orelse return error.NoSubgraphs;

        const tensors = try allocator.alloc(Tensor, tensors_v.len);
        errdefer allocator.free(tensors);

        var i: u32 = 0;
        while (i < tensors_v.len) : (i += 1) {
            const t = tensors_v.tableAt(i) catch return error.TensorOutOfRange;
            const name = (t.readString(3) catch null) orelse "";
            const dtype_raw = t.readU8(1, 0) catch 0;
            const buf_idx = t.readU32(2, 0) catch 0;

            // shape is [int] → store as borrowed []const i32 reinterpretation.
            const shape_v = t.readVector(0) catch null;
            const shape_bytes = if (shape_v) |v| (v.bytes() catch &.{}) else &.{};
            const shape_i32: []const i32 = @as([*]const i32, @ptrCast(@alignCast(shape_bytes.ptr)))[0 .. shape_bytes.len / 4];

            // Quantization params — keep raw bytes; caller re-interprets.
            var scales_bytes: []const u8 = &.{};
            var zp_bytes: []const u8 = &.{};
            var qdim: i32 = 0;
            if (t.readTable(4) catch null) |qp| {
                if (qp.readVector(2) catch null) |sv| scales_bytes = sv.bytes() catch &.{};
                if (qp.readVector(3) catch null) |zv| zp_bytes = zv.bytes() catch &.{};
                qdim = @bitCast(qp.readU32(6, 0) catch 0);
            }

            const data: []const u8 = if (buf_idx < buf_slices.len) buf_slices[buf_idx] else &.{};

            tensors[i] = .{
                .name = name,
                .shape = shape_i32,
                .dtype = @enumFromInt(@as(i8, @bitCast(dtype_raw))),
                .buffer_index = buf_idx,
                .scales = scales_bytes,
                .zero_points = zp_bytes,
                .quantized_dimension = qdim,
                .data = data,
            };
        }

        return .{ .version = version, .tensors = tensors, .allocator = allocator };
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    /// Locate a tensor by exact name match. Returns null if absent.
    pub fn findTensor(self: *const Model, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }
};

/// Decode a tensor's `scales` byte buffer as the float array TFLite
/// stores it as. Empty input ⇒ empty slice.
pub fn scalesAsF32(scales: []const u8) []const f32 {
    return @as([*]const f32, @ptrCast(@alignCast(scales.ptr)))[0 .. scales.len / 4];
}

/// Decode a tensor's `zero_points` byte buffer as i64 array.
pub fn zeroPointsAsI64(zp: []const u8) []const i64 {
    return @as([*]const i64, @ptrCast(@alignCast(zp.ptr)))[0 .. zp.len / 8];
}

// -- tests ----------------------------------------------------------------

test "TensorType.name + elementBytes round-trip" {
    try std.testing.expectEqualStrings("INT4", TensorType.int4.name());
    try std.testing.expectEqual(@as(?usize, 4), TensorType.float32.elementBytes());
    try std.testing.expectEqual(@as(?usize, null), TensorType.int4.elementBytes());
    try std.testing.expectEqual(@as(?usize, 2), TensorType.bfloat16.elementBytes());
}

test "scalesAsF32 reinterprets a 4-byte buffer" {
    // 1.5f32 = 0x3FC00000 LE bytes 00 00 C0 3F
    var bytes = [_]u8{ 0, 0, 0xC0, 0x3F };
    const scales = scalesAsF32(&bytes);
    try std.testing.expectEqual(@as(usize, 1), scales.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), scales[0], 1e-6);
}
