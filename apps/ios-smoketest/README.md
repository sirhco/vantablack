# ios-smoketest

Skeleton for validating vantablack on iOS. **Library + header now ship;
Swift wrapper still TODO.**

## Status

- Zig static archive cross-compiles for `aarch64-ios` and
  `aarch64-ios-simulator` (see `build.zig`).
- C ABI shim lives in `src/c_api.zig` — exports 9 `_vtb_*` symbols
  (model_open/close, state_init/deinit, generate_stream,
  signal_memory/thermal, version_string, has_metal).
- C header lives in `include/vantablack.h`. The iOS build installs both
  archive + header into `zig-out/`.

## Build flow

```bash
# 1. Cross-compile the static library for arm64 device + arm64 simulator.
zig build -Dtarget=aarch64-ios          -Doptimize=ReleaseFast
mv zig-out/lib/libvantablack.a libvantablack-device.a
zig build -Dtarget=aarch64-ios-simulator -Doptimize=ReleaseFast
mv zig-out/lib/libvantablack.a libvantablack-sim.a

# 2. Lipo them together for an xcframework input.
lipo -create -output libvantablack.a libvantablack-device.a libvantablack-sim.a
lipo -info  libvantablack.a   # should show: arm64 (device + sim)

# 3. Header is at zig-out/include/vantablack.h after either build.

# 4. Open Xcode, create an iOS app target, drag in libvantablack.a +
#    vantablack.h. Link the same frameworks that build.zig attaches:
#    Foundation, Metal, QuartzCore.
```

## Minimal Swift call site

```swift
import Foundation

class VantablackSession {
    private var model: OpaquePointer?
    private var state: OpaquePointer?

    init?(modelPath: String) {
        var cfg = VtbSamplerConfig(temperature: 0.7, top_k: 40, top_p: 0.95, seed: 0)
        model = vtb_model_open(modelPath, 1)
        guard model != nil else { return nil }
        state = vtb_state_init(model, &cfg)
        guard state != nil else {
            vtb_model_close(model)
            return nil
        }
    }

    deinit {
        vtb_state_deinit(state)
        vtb_model_close(model)
    }

    func generate(prompt: String, maxTokens: Int, onToken: @escaping (String, Bool) -> Bool) {
        let bytes = Array(prompt.utf8)
        // Pass `self` (retained box) through cb_ctx; trampoline below.
        let ctx = Unmanaged.passUnretained(Box(onToken)).toOpaque()
        bytes.withUnsafeBufferPointer { bp in
            _ = vtb_generate_stream(state, bp.baseAddress, bytes.count, maxTokens,
                                    { ctx, _, piece, len, isFinal in
                let box = Unmanaged<Box>.fromOpaque(ctx!).takeUnretainedValue()
                let str = String(bytes: UnsafeBufferPointer(start: piece, count: len), encoding: .utf8) ?? ""
                return box.fn(str, isFinal != 0) ? 1 : 0
            }, ctx)
        }
    }

    private final class Box {
        let fn: (String, Bool) -> Bool
        init(_ fn: @escaping (String, Bool) -> Bool) { self.fn = fn }
    }
}
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
