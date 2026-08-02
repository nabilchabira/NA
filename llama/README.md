# llama module — llama.cpp Android inference runtime (vendored)

This Gradle module is the **official llama.cpp Android runtime** (`examples/llama.android/lib`
from the llama.cpp repository), vendored with the smallest possible footprint:

- Only the runtime classes and native JNI wrapper needed for inference are included.
- The upstream example app, tests, playground code, and GGUF metadata readers (used only by
  the example's file picker) are **not** vendored.
- The llama.cpp source tree lives at `llama/llama.cpp`, pinned as a git submodule.

## Version

- llama.cpp release: **b10223** (commit `11924d4c17abc27383376a1ac6a24fa3e36c1c0c`, 2026-08-01).
  This is the latest stable release at the time of integration (2026-08-02).
- Upstream Kotlin API: `com.arm.aichat` (`AiChat`, `InferenceEngine`).
- Native library: `libai-chat.so` (ABIs `arm64-v8a`, `x86_64`).

## Local modifications (all documented, isolated, and minimal)

1. `build.gradle.kts` — adapted to the PokeClaw build (AGP built-in Kotlin, no `kotlin-android`
   plugin), `minSdk` lowered **33 → 28** to match the app (the native code is JNI/C++ only and
   uses no API newer than 28), and the datastore/test dependencies removed (coroutines kept).
2. `src/main/cpp/CMakeLists.txt` — `LLAMA_SRC` repointed to the vendored submodule location
   (`../../../llama.cpp`). Pure path change; no build logic altered.
3. `src/main/cpp/ai_chat.cpp` — sampler parameters (`temperature`, `top-k`, `top-p`,
   `repeat penalty`, `seed`) are now passed from Kotlin via `processUserPrompt` instead of a
   fixed `0.3` temperature. Justification: PokeClaw's `LocalLlmClient` exposes these sampling
   values and must keep behaving exactly as before. The sampler is re-created per generation
   (isolated to one JNI function); everything else is untouched.
4. `src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt` —
   - the "system prompt must be set right after model load" gate is relaxed so a conversation
     can be re-created (system prompt re-applied) without unloading/reloading the model.
     Justification: PokeClaw recreates conversations per task and every 8 rounds; reloading a
     GGUF each time would add seconds of latency.
   - `sendUserPrompt(...)`/`processUserPrompt(...)` gained the sampling parameters above.
5. `consumer-rules.pro` — removed the keep rule for the un-vendored `com.arm.aichat.gguf`
   package.

## Updating llama.cpp later

```bash
git -C llama/llama.cpp fetch --depth 1 origin tag <new-tag>
git -C llama/llama.cpp checkout <new-tag>
git add llama/llama.cpp && git commit -m "llama: bump llama.cpp to <new-tag>"
```

Then re-check the three deltas above against the new upstream files; the `processUserPrompt`
JNI signature and the `setSystemPrompt` gate are the most likely spots to drift.

## Backend note

The official Android runtime builds the **CPU** backend only (no Vulkan/GPU in the upstream
Android example), so no GPU/Vulkan backend selection is performed; the app always uses CPU.
6. `src/main/cpp/logging.h` — `ai_should_log()` no longer calls `__android_log_is_loggable()`
   below API 30 (that function is unavailable on the app's `minSdk 28`); below API 30 it
   falls back to a plain priority comparison, which matches the default behavior when no
   per-tag property is set.
