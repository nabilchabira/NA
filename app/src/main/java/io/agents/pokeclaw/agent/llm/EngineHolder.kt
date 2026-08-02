// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw.agent.llm

import io.agents.pokeclaw.ClawApplication
import io.agents.pokeclaw.utils.XLog

/**
 * Process-wide singleton that keeps a single llama.cpp engine alive across
 * the chat UI and the task agent.
 *
 * Why: loading a GGUF model takes seconds. Without this, ComposeChatActivity would unload
 * the model before a task, TaskOrchestrator would load a new one, then after the task the
 * chat would reload again — seconds wasted per round trip. Same design as the old LiteRT
 * EngineHolder, now backed by [LlmEngine].
 *
 * Thread safety: all mutations are @Synchronized so chat executor and task
 * executor threads can both call getOrCreate() safely.
 */
object EngineHolder {

    private const val TAG = "EngineHolder"

    private var engine: LlmEngine? = null
    private var currentModelPath: String? = null

    /**
     * Return the existing engine if the model path matches, otherwise unload the old one
     * and load a fresh engine for the new model.
     *
     * @param modelPath absolute path to the .gguf model file
     * @param cacheDir  app's cacheDir.path — kept for signature compatibility; the llama.cpp
     *                  runtime does not need it
     */
    @Synchronized
    @JvmOverloads
    fun getOrCreate(modelPath: String, cacheDir: String): LlmEngine {
        val existing = engine
        if (existing != null && currentModelPath == modelPath) {
            XLog.d(TAG, "getOrCreate: reusing engine for $modelPath")
            return existing
        }

        // Different model or first call — unload old engine first
        if (existing != null) {
            XLog.i(TAG, "getOrCreate: model changed ($currentModelPath -> $modelPath), unloading old engine")
            try {
                existing.close()
            } catch (e: Exception) {
                XLog.w(TAG, "getOrCreate: error closing old engine", e)
            }
            engine = null
            currentModelPath = null
        }

        XLog.i(TAG, "getOrCreate: creating new engine for $modelPath")
        return try {
            val newEngine = LlmEngine(ClawApplication.instance)
            newEngine.ensureLoaded(modelPath)
            engine = newEngine
            currentModelPath = modelPath
            XLog.i(TAG, "getOrCreate: engine ready for $modelPath")
            newEngine
        } catch (e: Exception) {
            XLog.e(TAG, "getOrCreate: failed to create engine for $modelPath", e)
            throw e
        }
    }

    /**
     * Unload and release the engine. Call only when the model is being unloaded entirely
     * (e.g. user deletes the model file). Normal chat/task transitions should NOT call this.
     */
    @Synchronized
    fun close() {
        XLog.i(TAG, "close: releasing engine for $currentModelPath")
        try {
            engine?.close()
        } catch (e: Exception) {
            XLog.w(TAG, "close: error closing engine", e)
        }
        engine = null
        currentModelPath = null
        XLog.i(TAG, "close: done")
    }

    /** Returns true if an engine is live for the given model path. */
    @Synchronized
    fun isReady(modelPath: String): Boolean =
        engine != null && currentModelPath == modelPath && (engine?.isReady(modelPath) ?: false)

    /** llama.cpp runs on CPU only, so the backend label is always "CPU". */
    @Synchronized
    fun getBackendLabel(modelPath: String? = null): String? {
        return if (modelPath == null || currentModelPath == modelPath) "CPU" else null
    }
}
