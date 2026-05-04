//! Ollama-compatible HTTP server.
//!
//! One Model + State + KvCache + Sampler + Tokenizer + ThreadPool live for
//! the process lifetime. Requests serialize through `inference_mu` so a
//! single inference path is shared by all callers (single-user code-gen
//! workload). Concurrent requests queue on the mutex.
//!
//! Endpoints (Ollama subset):
//!   GET  /              -> "vantablack ok"
//!   GET  /health        -> "vantablack ok"
//!   GET  /api/tags      -> JSON list with the one loaded model
//!   POST /api/generate  -> NDJSON token stream (raw prompt)
//!   POST /api/chat      -> NDJSON token stream (chat-template wrapped)

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const http = std.http;
const json = std.json;
const net = std.Io.net;

const parser = @import("../core/parser.zig");
const mapper_mod = @import("../core/mapper.zig");
const hf_loader_mod = @import("../core/hf_loader.zig");
const hf_config_mod = @import("../core/hf_config.zig");
const model_mod = @import("model.zig");
const forward_mod = @import("forward.zig");
const kv_cache_mod = @import("kv_cache.zig");
const sampler_mod = @import("sampler.zig");
const tokenizer_mod = @import("tokenizer.zig");
const pool_mod = @import("pool.zig");
const chat_template = @import("chat_template.zig");
const metal_backend_mod = @import("metal_backend.zig");
const metal_bridge = @import("../metal/bridge.zig");

const ModelMapper = mapper_mod.ModelMapper;
const Model = model_mod.Model;
const KvCache = kv_cache_mod.KvCache;
const Tokenizer = tokenizer_mod.Tokenizer;
const Sampler = sampler_mod.Sampler;
const ThreadPool = pool_mod.ThreadPool;
const MetalBackend = metal_backend_mod.MetalBackend;

const max_body_bytes: usize = 1 * 1024 * 1024; // 1 MB request body cap
const created_at_placeholder: []const u8 = "1970-01-01T00:00:00Z";

pub const Config = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 11434,
    /// Logical name reported by /api/tags and validated in request bodies.
    model_name: []const u8,
    /// Absolute path to the .gguf, used for /api/tags `size`.
    model_path: []const u8,
    /// Default sampling when the request omits `options`. Greedy by default
    /// to favour deterministic code generation.
    default_sampler: sampler_mod.Config = .{},
    default_num_predict: u32 = 256,
    /// Worker thread count. 0 = autodetect (~half of cpu cores).
    threads: usize = 0,
};

pub const Server = struct {
    cfg: Config,
    gpa: Allocator,
    io: Io,
    /// GGUF mode: borrowed mapper. HF/MLX mode: null.
    mapper: ?*const ModelMapper,
    /// HF/MLX mode: heap-owned bundle, freed by `deinit`. GGUF mode: null.
    bundle: ?*hf_loader_mod.HfBundle,
    model: Model,
    state: forward_mod.State,
    cache: KvCache,
    tok: Tokenizer,
    /// True when `tok.pieces` were `gpa.dupe`'d (HF tokenizer.json path);
    /// requires `Tokenizer.deinitOwnedPieces` instead of plain `deinit`.
    tok_owns_pieces: bool,
    pool: *ThreadPool,
    metal: ?MetalBackend,
    /// Mutex serializes the inference path. tryLock + spinLoopHint, same
    /// pattern as `runtime/pool.zig`.
    inference_mu: std.atomic.Value(u8),
    /// Stat'd once at init, returned in /api/tags.
    model_size_bytes: u64,

    fn metal_ptr(self: *Server) ?*MetalBackend {
        return if (self.metal != null) &self.metal.? else null;
    }

    pub fn init(
        gpa: Allocator,
        io: Io,
        mapper: *const ModelMapper,
        cfg: Config,
    ) !Server {
        var model = try Model.init(gpa, mapper);
        errdefer model.deinit(gpa);

        // Metal is built before State so State can alias the backend's
        // persistent shared-storage scratch buffers (zero-copy GPU dispatch).
        var maybe_metal: ?MetalBackend = blk: {
            if (!metal_bridge.metal_enabled) break :blk null;
            const mb = MetalBackend.init(gpa, mapper, model.config) catch break :blk null;
            break :blk mb;
        };
        errdefer if (maybe_metal) |*mb| mb.deinit(gpa);

        // Take a stable pointer into the local optional. Once this Server is
        // moved into the caller's slot the backend's Metal-heap pointers do
        // not change (they live in MTLBuffer storage, not in Server memory),
        // so the aliases held by State remain valid.
        const metal_ptr_init: ?*MetalBackend = if (maybe_metal != null) &maybe_metal.? else null;

        var state = try forward_mod.State.init(gpa, &model, metal_ptr_init);
        errdefer state.deinit(gpa);

        var cache = try KvCache.init(
            gpa,
            model.config.n_layers,
            model.config.n_kv_heads,
            model.config.head_dim,
            model.config.max_seq,
            metal_ptr_init,
        );
        errdefer cache.deinit(gpa);

        var tok = try Tokenizer.init(gpa, mapper.catalog);
        errdefer tok.deinit(gpa);

        // When Metal handles matmuls, CPU workers just spin on GPU sync.
        // Default to 1 thread in that case.
        const effective_threads = if (maybe_metal != null and cfg.threads == 0) @as(usize, 1) else cfg.threads;
        const pool = try ThreadPool.init(gpa, effective_threads);
        errdefer pool.deinit(gpa);

        // Best-effort stat for model size; ignore errors and report 0.
        const model_size: u64 = blk: {
            const file = Io.Dir.openFileAbsolute(io, cfg.model_path, .{ .allow_directory = false }) catch break :blk 0;
            defer file.close(io);
            break :blk file.length(io) catch 0;
        };

        return .{
            .cfg = cfg,
            .gpa = gpa,
            .io = io,
            .mapper = mapper,
            .bundle = null,
            .model = model,
            .state = state,
            .cache = cache,
            .tok = tok,
            .tok_owns_pieces = false,
            .pool = pool,
            .metal = maybe_metal,
            .inference_mu = .init(0),
            .model_size_bytes = model_size,
        };
    }

    /// Build a `Server` against a HuggingFace / MLX directory bundle. The
    /// `Server` takes ownership of `bundle_owned` and frees it via `deinit`.
    /// When built with `-Dmetal=true`, a per-shard `MetalBackend` is wired in
    /// so MLX-Q4 matmuls dispatch to the GPU. Failure is logged and the path
    /// silently falls back to CPU.
    pub fn initFromHf(
        gpa: Allocator,
        io: Io,
        bundle_owned: *hf_loader_mod.HfBundle,
        hf_cfg: hf_config_mod.HfConfig,
        cfg: Config,
    ) !Server {
        errdefer {
            bundle_owned.deinit();
            gpa.destroy(bundle_owned);
        }

        var model = try Model.initFromHf(gpa, bundle_owned, hf_cfg);
        errdefer model.deinit(gpa);

        // Metal is built before State so State can alias the backend's
        // persistent shared-storage scratch buffers (zero-copy GPU dispatch).
        // The bundle's mmap'd shard memory outlives this backend because the
        // Server owns `bundle_owned` and `deinit` frees the metal backend
        // before the bundle.
        var maybe_metal: ?MetalBackend = blk: {
            if (!metal_bridge.metal_enabled) break :blk null;
            const mb = MetalBackend.initFromHf(gpa, bundle_owned, model.config) catch |e| {
                std.log.warn("MetalBackend.initFromHf failed: {s}; falling back to CPU", .{@errorName(e)});
                break :blk null;
            };
            break :blk mb;
        };
        errdefer if (maybe_metal) |*mb| mb.deinit(gpa);

        const metal_ptr_init: ?*MetalBackend = if (maybe_metal != null) &maybe_metal.? else null;

        var state = try forward_mod.State.init(gpa, &model, metal_ptr_init);
        errdefer state.deinit(gpa);

        var cache = try KvCache.init(
            gpa,
            model.config.n_layers,
            model.config.n_kv_heads,
            model.config.head_dim,
            model.config.max_seq,
            metal_ptr_init,
        );
        errdefer cache.deinit(gpa);

        if (bundle_owned.tokenizer_json == null) return error.MissingTokenizer;
        var tok = try Tokenizer.initFromHfJson(gpa, bundle_owned.tokenizer_json.?);
        errdefer tok.deinitOwnedPieces(gpa);

        // When Metal handles matmuls, CPU workers just spin on GPU sync.
        const effective_threads = if (maybe_metal != null and cfg.threads == 0) @as(usize, 1) else cfg.threads;
        const pool = try ThreadPool.init(gpa, effective_threads);
        errdefer pool.deinit(gpa);

        // model_path may point at a directory; report 0 size if stat fails.
        const model_size: u64 = blk: {
            const file = Io.Dir.openFileAbsolute(io, cfg.model_path, .{ .allow_directory = true }) catch break :blk 0;
            defer file.close(io);
            break :blk file.length(io) catch 0;
        };

        return .{
            .cfg = cfg,
            .gpa = gpa,
            .io = io,
            .mapper = null,
            .bundle = bundle_owned,
            .model = model,
            .state = state,
            .cache = cache,
            .tok = tok,
            .tok_owns_pieces = true,
            .pool = pool,
            .metal = maybe_metal,
            .inference_mu = .init(0),
            .model_size_bytes = model_size,
        };
    }

    pub fn deinit(self: *Server) void {
        if (self.metal) |*mb| mb.deinit(self.gpa);
        self.pool.deinit(self.gpa);
        if (self.tok_owns_pieces) self.tok.deinitOwnedPieces(self.gpa) else self.tok.deinit(self.gpa);
        self.cache.deinit(self.gpa);
        self.state.deinit(self.gpa);
        self.model.deinit(self.gpa);
        if (self.bundle) |b| {
            b.deinit();
            self.gpa.destroy(b);
        }
        self.* = undefined;
    }

    pub fn run(self: *Server) !void {
        var addr = try net.IpAddress.parse(self.cfg.host, self.cfg.port);
        var listener = try addr.listen(self.io, .{ .reuse_address = true });
        defer listener.deinit(self.io);

        // Banner to stderr.
        var sb: [256]u8 = undefined;
        var sw: Io.File.Writer = .init(.stderr(), self.io, &sb);
        const sr = &sw.interface;
        sr.print("vantablack: serving '{s}' at http://{s}:{d}/\n", .{
            self.cfg.model_name, self.cfg.host, self.cfg.port,
        }) catch {};
        sr.flush() catch {};

        while (true) {
            const stream = listener.accept(self.io) catch continue;
            self.handleConnection(stream) catch |err| {
                self.logErr("connection: {t}", .{err});
            };
            stream.close(self.io);
        }
    }

    fn handleConnection(self: *Server, stream: net.Stream) !void {
        var read_buf: [16 * 1024]u8 = undefined;
        var write_buf: [16 * 1024]u8 = undefined;
        var sr = stream.reader(self.io, &read_buf);
        var sw = stream.writer(self.io, &write_buf);
        var http_server = http.Server.init(&sr.interface, &sw.interface);

        while (true) {
            var request = http_server.receiveHead() catch |err| switch (err) {
                error.HttpConnectionClosing => return,
                else => return err,
            };
            self.dispatch(&request) catch |err| {
                self.logErr("handler: {t}", .{err});
                return;
            };
            sw.interface.flush() catch return;
        }
    }

    fn dispatch(self: *Server, req: *http.Server.Request) !void {
        const m = req.head.method;
        const target = req.head.target;
        if (m == .GET and (eql(target, "/") or eql(target, "/health"))) {
            return req.respond("vantablack ok\n", .{
                .status = .ok,
                .extra_headers = &.{contentType("text/plain")},
            });
        }
        if (m == .GET and (eql(target, "/api/tags") or std.mem.startsWith(u8, target, "/api/tags?"))) {
            return self.handleTags(req);
        }
        if (m == .POST and eql(target, "/api/generate")) {
            return self.handleGenerate(req);
        }
        if (m == .POST and eql(target, "/api/chat")) {
            return self.handleChat(req);
        }
        return req.respond(
            "{\"error\":\"not found\"}\n",
            .{ .status = .not_found, .extra_headers = &.{contentType("application/json")} },
        );
    }

    fn handleTags(self: *Server, req: *http.Server.Request) !void {
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(a);
        var alloc_writer: Io.Writer.Allocating = .fromArrayList(a, &buf);

        const Tag = struct {
            name: []const u8,
            modified_at: []const u8,
            size: u64,
            digest: []const u8,
            details: struct {
                parent_model: []const u8 = "",
                format: []const u8 = "gguf",
                family: []const u8 = "llama",
                families: []const []const u8 = &.{"llama"},
                parameter_size: []const u8 = "",
                quantization_level: []const u8 = "",
            } = .{},
        };
        const Body = struct { models: []const Tag };
        const body: Body = .{ .models = &.{.{
            .name = self.cfg.model_name,
            .modified_at = created_at_placeholder,
            .size = self.model_size_bytes,
            .digest = "",
        }} };

        try json.Stringify.value(body, .{}, &alloc_writer.writer);
        try alloc_writer.writer.writeByte('\n');
        const out = alloc_writer.toOwnedSlice() catch return error.OutOfMemory;

        try req.respond(out, .{
            .status = .ok,
            .extra_headers = &.{contentType("application/json")},
        });
    }

    fn handleGenerate(self: *Server, req: *http.Server.Request) !void {
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const body_bytes = readBody(req, a) catch |err| {
            return self.respondError(req, .bad_request, "bad request body", err);
        };

        const Req = struct {
            model: []const u8,
            prompt: []const u8,
            stream: bool = true,
            options: ?Options = null,
        };
        const parsed = json.parseFromSlice(Req, a, body_bytes, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            return self.respondError(req, .bad_request, "invalid json", err);
        };
        defer parsed.deinit();
        const r = parsed.value;

        if (!eql(r.model, self.cfg.model_name)) {
            return req.respond(
                "{\"error\":\"model not found\"}\n",
                .{ .status = .not_found, .extra_headers = &.{contentType("application/json")} },
            );
        }

        try self.streamGenerate(req, a, r.prompt, r.options, r.stream, .raw);
    }

    fn handleChat(self: *Server, req: *http.Server.Request) !void {
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const body_bytes = readBody(req, a) catch |err| {
            return self.respondError(req, .bad_request, "bad request body", err);
        };

        const InMsg = struct {
            role: []const u8,
            content: []const u8,
        };
        const Req = struct {
            model: []const u8,
            messages: []const InMsg,
            stream: bool = true,
            options: ?Options = null,
        };
        const parsed = json.parseFromSlice(Req, a, body_bytes, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            return self.respondError(req, .bad_request, "invalid json", err);
        };
        defer parsed.deinit();
        const r = parsed.value;

        if (!eql(r.model, self.cfg.model_name)) {
            return req.respond(
                "{\"error\":\"model not found\"}\n",
                .{ .status = .not_found, .extra_headers = &.{contentType("application/json")} },
            );
        }

        // Convert Ollama messages → chat_template.Message[].
        const msgs = try a.alloc(chat_template.Message, r.messages.len);
        for (r.messages, msgs) |in, *out_msg| {
            const role: chat_template.Role = if (eql(in.role, "system"))
                .system
            else if (eql(in.role, "assistant"))
                .assistant
            else
                .user;
            out_msg.* = .{ .role = role, .content = in.content };
        }
        const prompt_text = try chat_template.formatLlamaChat(a, msgs);

        try self.streamGenerate(req, a, prompt_text, r.options, r.stream, .chat);
    }

    const StreamMode = enum { raw, chat };

    fn streamGenerate(
        self: *Server,
        req: *http.Server.Request,
        a: Allocator,
        prompt_text: []const u8,
        opts: ?Options,
        stream_mode: bool,
        mode: StreamMode,
    ) !void {
        // Acquire mutex (busy-wait).
        while (self.inference_mu.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
        defer self.inference_mu.store(0, .release);

        // Reset cache + sampler from per-request options.
        self.cache.reset();
        var cfg = self.cfg.default_sampler;
        var num_predict: u32 = self.cfg.default_num_predict;
        if (opts) |o| {
            if (o.temperature) |v| cfg.temperature = v;
            if (o.top_k) |v| cfg.top_k = v;
            if (o.top_p) |v| cfg.top_p = v;
            if (o.seed) |v| cfg.seed = v;
            if (o.num_predict) |v| num_predict = v;
        }

        var sampler = try Sampler.init(self.gpa, cfg, self.model.config.vocab_size);
        defer sampler.deinit(self.gpa);

        const prompt_ids = try self.tok.encode(a, prompt_text, true);

        // Feed prompt (no streaming back per Ollama convention).
        for (prompt_ids) |id| {
            try forward_mod.step(&self.model, &self.state, &self.cache, self.pool, self.metal_ptr(), id);
        }

        if (stream_mode) {
            try self.streamingResponse(req, a, num_predict, mode, &sampler);
        } else {
            try self.nonStreamingResponse(req, a, num_predict, mode, &sampler);
        }
    }

    fn streamingResponse(
        self: *Server,
        req: *http.Server.Request,
        a: Allocator,
        num_predict: u32,
        mode: StreamMode,
        sampler: *Sampler,
    ) !void {
        var resp_buf: [4096]u8 = undefined;
        var body_writer = try req.respondStreaming(&resp_buf, .{
            .respond_options = .{
                .status = .ok,
                .extra_headers = &.{contentType("application/x-ndjson")},
            },
        });
        const w = &body_writer.writer;

        var produced: u32 = 0;
        var next: u32 = sampler.sample(self.state.logits);
        while (produced < num_predict) : (produced += 1) {
            if (next == self.tok.eos) break;
            try writeChunk(w, self.cfg.model_name, mode, false, try decodeOne(self.tok, a, next));
            try body_writer.flush();
            try forward_mod.step(&self.model, &self.state, &self.cache, self.pool, self.metal_ptr(), next);
            next = sampler.sample(self.state.logits);
        }
        try writeChunk(w, self.cfg.model_name, mode, true, "");
        try body_writer.end();
    }

    fn nonStreamingResponse(
        self: *Server,
        req: *http.Server.Request,
        a: Allocator,
        num_predict: u32,
        mode: StreamMode,
        sampler: *Sampler,
    ) !void {
        var collected: std.ArrayList(u8) = .empty;
        defer collected.deinit(a);

        var produced: u32 = 0;
        var next: u32 = sampler.sample(self.state.logits);
        while (produced < num_predict) : (produced += 1) {
            if (next == self.tok.eos) break;
            const piece = try decodeOne(self.tok, a, next);
            try collected.appendSlice(a, piece);
            try forward_mod.step(&self.model, &self.state, &self.cache, self.pool, self.metal_ptr(), next);
            next = sampler.sample(self.state.logits);
        }

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(a);
        var aw: Io.Writer.Allocating = .fromArrayList(a, &buf);
        try writeChunk(&aw.writer, self.cfg.model_name, mode, true, collected.items);
        const out = aw.toOwnedSlice() catch return error.OutOfMemory;

        try req.respond(out, .{
            .status = .ok,
            .extra_headers = &.{contentType("application/x-ndjson")},
        });
    }

    fn respondError(
        self: *Server,
        req: *http.Server.Request,
        status: http.Status,
        msg: []const u8,
        err: anyerror,
    ) !void {
        _ = self;
        var buf: [256]u8 = undefined;
        const out = std.fmt.bufPrint(&buf, "{{\"error\":\"{s}: {t}\"}}\n", .{ msg, err }) catch
            return req.respond(
                "{\"error\":\"internal\"}\n",
                .{ .status = .internal_server_error, .extra_headers = &.{contentType("application/json")} },
            );
        try req.respond(out, .{
            .status = status,
            .extra_headers = &.{contentType("application/json")},
        });
    }

    fn logErr(self: *Server, comptime fmt: []const u8, args: anytype) void {
        var sb: [512]u8 = undefined;
        var sw: Io.File.Writer = .init(.stderr(), self.io, &sb);
        sw.interface.print("vantablack: " ++ fmt ++ "\n", args) catch {};
        sw.interface.flush() catch {};
    }
};

const Options = struct {
    temperature: ?f32 = null,
    top_k: ?usize = null,
    top_p: ?f32 = null,
    seed: ?u64 = null,
    num_predict: ?u32 = null,
};

/// Read the request body, capped at `max_body_bytes`.
fn readBody(req: *http.Server.Request, a: Allocator) ![]const u8 {
    var rb: [4096]u8 = undefined;
    const reader = req.readerExpectNone(&rb);
    return reader.allocRemaining(a, .limited(max_body_bytes));
}

/// Decode one token id into an arena-allocated piece slice.
fn decodeOne(tok: Tokenizer, a: Allocator, id: u32) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(a);
    var aw: Io.Writer.Allocating = .fromArrayList(a, &buf);
    try tok.decodeTo(&aw.writer, id);
    return aw.toOwnedSlice();
}

fn writeChunk(
    w: *Io.Writer,
    model_name: []const u8,
    mode: Server.StreamMode,
    done: bool,
    piece: []const u8,
) !void {
    switch (mode) {
        .raw => {
            const Chunk = struct {
                model: []const u8,
                created_at: []const u8,
                response: []const u8,
                done: bool,
            };
            const c: Chunk = .{
                .model = model_name,
                .created_at = created_at_placeholder,
                .response = piece,
                .done = done,
            };
            try json.Stringify.value(c, .{}, w);
        },
        .chat => {
            const Msg = struct { role: []const u8 = "assistant", content: []const u8 };
            const Chunk = struct {
                model: []const u8,
                created_at: []const u8,
                message: Msg,
                done: bool,
            };
            const c: Chunk = .{
                .model = model_name,
                .created_at = created_at_placeholder,
                .message = .{ .content = piece },
                .done = done,
            };
            try json.Stringify.value(c, .{}, w);
        },
    }
    try w.writeByte('\n');
}

fn contentType(value: []const u8) http.Header {
    return .{ .name = "content-type", .value = value };
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "writeChunk raw shape" {
    var buf: [256]u8 = undefined;
    var fw: Io.Writer = .fixed(&buf);
    try writeChunk(&fw, "model-x", .raw, false, "hi");
    const out = fw.buffered();
    // Must contain the response key and end with newline.
    try std.testing.expect(std.mem.indexOf(u8, out, "\"response\":\"hi\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"done\":false") != null);
    try std.testing.expect(out[out.len - 1] == '\n');
}

test "writeChunk chat shape" {
    var buf: [256]u8 = undefined;
    var fw: Io.Writer = .fixed(&buf);
    try writeChunk(&fw, "model-x", .chat, true, "");
    const out = fw.buffered();
    try std.testing.expect(std.mem.indexOf(u8, out, "\"role\":\"assistant\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"done\":true") != null);
}
