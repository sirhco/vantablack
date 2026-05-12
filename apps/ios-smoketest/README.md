# ios-smoketest

Minimal scaffold for validating vantablack on iOS. **Not yet runnable.**

## Status

- Zig static archive cross-compiles for `aarch64-ios` and
  `aarch64-ios-simulator` (see `build.zig`).
- The Zig public API (`generateStream`, `Backend`, `PressureHub`) is **not
  yet exposed as C symbols** — Zig dead-code-strips them in release
  builds because nothing references them externally.

## What's missing before this app runs

A C ABI shim layer, e.g. `src/c_api.zig`, that re-exports the streaming
+ pressure surface with `callconv(.c)` and `export`. The shim should
roughly mirror this header:

```c
typedef struct VtbModel VtbModel;
typedef struct VtbState VtbState;

typedef int (*VtbTokenCb)(void *ctx,
                          uint32_t token_id,
                          const uint8_t *piece_bytes,
                          size_t piece_len,
                          uint8_t is_final);

VtbModel *vtb_model_open(const char *path);
void      vtb_model_close(VtbModel *m);

VtbState *vtb_state_init(VtbModel *m);
void      vtb_state_deinit(VtbState *s);

int       vtb_generate_stream(VtbState *s,
                              const char *prompt,
                              size_t prompt_len,
                              size_t max_tokens,
                              VtbTokenCb cb,
                              void *cb_ctx);

void      vtb_signal_memory(uint8_t level);    // 0=normal, 1=warn, 2=critical
void      vtb_signal_thermal(uint8_t state);   // 0..3
```

## Build flow once the shim lands

```bash
# 1. Cross-compile the static library for arm64 device + arm64 simulator.
zig build -Dtarget=aarch64-ios          -Doptimize=ReleaseFast
mv zig-out/lib/libvantablack.a libvantablack-device.a
zig build -Dtarget=aarch64-ios-simulator -Doptimize=ReleaseFast
mv zig-out/lib/libvantablack.a libvantablack-sim.a

# 2. Lipo them together for an xcframework input.
lipo -create -output libvantablack.a libvantablack-device.a libvantablack-sim.a
lipo -info  libvantablack.a   # should show: arm64 (device + sim)

# 3. Open Xcode, create an iOS app target, drag in libvantablack.a +
#    vantablack.h (generated alongside the shim). Link the same
#    frameworks that build.zig already attaches: Foundation, Metal,
#    QuartzCore.
```

## Pressure / thermal wiring (in Swift)

```swift
NotificationCenter.default.addObserver(
    forName: UIApplication.didReceiveMemoryWarningNotification,
    object: nil, queue: .main
) { _ in
    vtb_signal_memory(1)  // .warning
}

NotificationCenter.default.addObserver(
    forName: ProcessInfo.thermalStateDidChangeNotification,
    object: nil, queue: .main
) { _ in
    let s = ProcessInfo.processInfo.thermalState
    let mapped: UInt8 = switch s {
        case .nominal: 0
        case .fair: 1
        case .serious: 2
        case .critical: 3
        @unknown default: 0
    }
    vtb_signal_thermal(mapped)
}
```

## Verification target (once the shim ships)

- Load TinyLlama Q4_K_M model from app bundle.
- Generate 64 tokens via `vtb_generate_stream`.
- Output bytes match the macOS reference for the same seed.
- Xcode Instruments: peak RSS < 1.5 GB on iPhone 15 (A17 Pro).
- Trigger `_performMemoryWarning` from test hook; generation
  continues without crash and KV cache shrinks visibly.
