//! HuggingFace / MLX directory loader.
//!
//! Opens a model directory containing some combination of:
//!   * `config.json`              — architecture metadata
//!   * `weights.NN.safetensors`   — MLX naming
//!   * `model.safetensors`        — single-file HF naming
//!   * `model-NNNNN-of-MMMMM.safetensors` + `model.safetensors.index.json`
//!     — multi-shard HF naming
//!
//! mmap's every safetensors shard read-only, parses each header into a
//! `safetensors.Catalog`, and exposes a unified lookup that hides which
//! shard a tensor lives in. Non-tensor files (config.json, tokenizer.json)
//! are read fully into the arena.
//!
//! Lifetimes: `arena.deinit()` frees the catalogs and small JSON blobs;
//! `std.posix.munmap` releases each shard's mapping.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const safetensors = @import("safetensors.zig");

pub const LoadError = error{
    NoSafetensorsFiles,
    MissingConfig,
    OutOfMemory,
    StreamTooLong,
    Unexpected,
} ||
    Io.File.OpenError ||
    Io.File.LengthError ||
    Io.Dir.OpenError ||
    Io.Dir.Iterator.Error ||
    Io.File.ReadPositionalError ||
    std.posix.MMapError ||
    safetensors.ParseError;

pub const Shard = struct {
    /// Whole-file mmap, page-aligned.
    mapped: []align(std.heap.page_size_min) const u8,
    catalog: safetensors.Catalog,
};

/// Per-tensor lookup result: which shard, and the descriptor inside it.
pub const TensorLocation = struct {
    shard: *const Shard,
    desc: *const safetensors.TensorDesc,
};

pub const HfBundle = struct {
    arena: std.heap.ArenaAllocator,
    shards: []Shard,
    /// Raw bytes of `config.json`. Caller parses via `core/hf_config.zig`.
    config_json: []const u8,
    /// Raw bytes of `tokenizer.json` if the directory ships one (Llama-family
    /// MLX models always do). null otherwise.
    tokenizer_json: ?[]const u8 = null,
    /// Raw bytes of SentencePiece `tokenizer.model` if present alongside.
    tokenizer_model: ?[]const u8 = null,

    pub fn init(allocator: Allocator, io: Io, dir_path: []const u8) LoadError!HfBundle {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const aalloc = arena.allocator();

        var dir = try Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
        defer dir.close(io);

        var shard_paths: std.ArrayList([]const u8) = .empty;
        var config_path: ?[]const u8 = null;
        var tokenizer_json_path: ?[]const u8 = null;
        var tokenizer_model_path: ?[]const u8 = null;

        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            // HuggingFace's snapshot dirs are full of symlinks pointing into
            // `../../blobs/...`. Accept those alongside regular files.
            if (entry.kind != .file and entry.kind != .sym_link) continue;
            const name = entry.name;
            if (std.mem.endsWith(u8, name, ".safetensors")) {
                const full = try std.fs.path.join(aalloc, &.{ dir_path, name });
                try shard_paths.append(aalloc, full);
            } else if (std.mem.eql(u8, name, "config.json")) {
                config_path = try std.fs.path.join(aalloc, &.{ dir_path, name });
            } else if (std.mem.eql(u8, name, "tokenizer.json")) {
                tokenizer_json_path = try std.fs.path.join(aalloc, &.{ dir_path, name });
            } else if (std.mem.eql(u8, name, "tokenizer.model")) {
                tokenizer_model_path = try std.fs.path.join(aalloc, &.{ dir_path, name });
            }
        }

        if (config_path == null) return error.MissingConfig;
        if (shard_paths.items.len == 0) return error.NoSafetensorsFiles;

        // Sort shard names so MLX `weights.00`, `weights.01`, ... and HF
        // `model-00001-of-00002`, ... iterate in deterministic order.
        std.mem.sortUnstable([]const u8, shard_paths.items, {}, lessThanString);

        const config_json = try readFile(aalloc, io, config_path.?);
        const tokenizer_json = if (tokenizer_json_path) |p| try readFile(aalloc, io, p) else null;
        const tokenizer_model = if (tokenizer_model_path) |p| try readFile(aalloc, io, p) else null;

        const shards = try aalloc.alloc(Shard, shard_paths.items.len);
        var n_initialized: usize = 0;
        errdefer for (shards[0..n_initialized]) |s| std.posix.munmap(s.mapped);

        for (shard_paths.items, 0..) |path, i| {
            const file = try Io.Dir.openFileAbsolute(io, path, .{ .allow_directory = false });
            defer file.close(io);
            const len = try file.length(io);
            const len_us: usize = std.math.cast(usize, len) orelse return error.OutOfMemory;
            const mapped = try std.posix.mmap(
                null,
                len_us,
                .{ .READ = true },
                .{ .TYPE = .PRIVATE },
                file.handle,
                0,
            );
            // Ownership immediately moves into shards[i]; no errdefer needed.
            const cat = try safetensors.parseHeader(aalloc, mapped);
            shards[i] = .{ .mapped = mapped, .catalog = cat };
            n_initialized += 1;
        }

        return .{
            .arena = arena,
            .shards = shards,
            .config_json = config_json,
            .tokenizer_json = tokenizer_json,
            .tokenizer_model = tokenizer_model,
        };
    }

    pub fn deinit(self: *HfBundle) void {
        for (self.shards) |s| std.posix.munmap(s.mapped);
        self.arena.deinit();
        self.* = undefined;
    }

    /// O(n_shards × n_tensors_in_shard) tensor lookup. n_shards is small
    /// (1..~8) so this stays cheap even for sharded 70B checkpoints.
    pub fn find(self: *const HfBundle, name: []const u8) ?TensorLocation {
        for (self.shards) |*s| {
            if (s.catalog.find(name)) |d| return .{ .shard = s, .desc = d };
        }
        return null;
    }

    /// Returns a slice into the shard's mmap region for the named tensor's
    /// raw bytes. Bounds-checked.
    pub fn tensorBytes(self: *const HfBundle, name: []const u8) ?[]const u8 {
        const loc = self.find(name) orelse return null;
        const start = loc.shard.catalog.data_segment_start + loc.desc.rel_offset_start;
        const end = loc.shard.catalog.data_segment_start + loc.desc.rel_offset_end;
        if (end > loc.shard.mapped.len) return null;
        return loc.shard.mapped[@intCast(start)..@intCast(end)];
    }
};

fn lessThanString(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn readFile(allocator: Allocator, io: Io, abs_path: []const u8) ![]const u8 {
    const file = try Io.Dir.openFileAbsolute(io, abs_path, .{ .allow_directory = false });
    defer file.close(io);
    const len = try file.length(io);
    const len_us: usize = std.math.cast(usize, len) orelse return error.OutOfMemory;
    const buf = try allocator.alloc(u8, len_us);
    var off: usize = 0;
    while (off < len_us) {
        const buffers = [_][]u8{buf[off..]};
        const n = try file.readPositional(io, &buffers, @intCast(off));
        if (n == 0) break;
        off += n;
    }
    return buf;
}
