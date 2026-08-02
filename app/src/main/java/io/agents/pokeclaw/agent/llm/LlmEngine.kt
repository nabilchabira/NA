// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw.agent.llm

import android.content.Context
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import io.agents.pokeclaw.utils.XLog
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * A handle to the process-wide llama.cpp engine (the official `com.arm.aichat` runtime).
 *
 * Replaces the LiteRT-LM Engine role: it loads/unloads the
 * GGUF model and creates [LlmConversation] sessions. The underlying InferenceEngine is a
 * singleton, so only one model is resident at a time — the same constraint the app already
 * relied on with LiteRT-LM.
 */
class LlmEngine(private val context: Context) {

    internal val inference: InferenceEngine = AiChat.getInferenceEngine(context)

    @Volatile
    private var loadedModelPath: String? = null

    /**
     * The engine initializes its native library asynchronously; wait until it is ready.
     */
    private fun awaitInitialized(timeoutMs: Long = 30_000) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (true) {
            when (val state = inference.state.value) {
                is InferenceEngine.State.Initialized -> {
                    XLog.d(TAG, "engine state: Initialized (native library ready)")
                    return
                }
                is InferenceEngine.State.Error -> {
                    XLog.e(TAG, "engine state: Error during initialization: ${state.exception.message}", state.exception)
                    throw state.exception
                }
                else -> {
                    if (System.currentTimeMillis() > deadline) {
                        throw IllegalStateException("llama.cpp engine failed to initialize (state=$state)")
                    }
                    if (state is InferenceEngine.State.Initializing) {
                        XLog.d(TAG, "engine state: Initializing (waiting for native library)")
                    }
                }
            }
            Thread.sleep(50)
        }
    }

    /**
     * Load the model at [modelPath], reloading if a different model is already loaded.
     */
    @Synchronized
    fun ensureLoaded(modelPath: String) {
        awaitInitialized()
        val stateBefore = inference.state.value
        if (loadedModelPath == modelPath && stateBefore is InferenceEngine.State.ModelReady) {
            XLog.d(TAG, "ensureLoaded: model already ready ($modelPath), reusing")
            return
        }
        XLog.i(TAG, "ensureLoaded: engine state before load = ${stateBefore::class.simpleName}")
        if (loadedModelPath != null) {
            XLog.i(TAG, "ensureLoaded: unloading previous model ${loadedModelPath?.substringAfterLast('/')}")
            unloadModel()
        }
        XLog.i(TAG, "ensureLoaded: loading $modelPath (${File(modelPath).length() / 1_000_000} MB)")
        runBlocking { inference.loadModel(modelPath) }
        val stateAfter = inference.state.value
        if (stateAfter !is InferenceEngine.State.ModelReady) {
            // Model load reported success but the engine is not READY — do not trust it.
            throw IllegalStateException(
                "llama.cpp engine did not reach ModelReady after loadModel (state=${stateAfter::class.simpleName})"
            )
        }
        loadedModelPath = modelPath
        XLog.i(TAG, "ensureLoaded: model READY (state=ModelReady, $modelPath)")
    }

    fun createConversation(config: LlmConversationConfig): LlmConversation = LlmConversation(this, config)

    /**
     * Apply (or re-apply) the system prompt, which clears the native chat history and KV cache.
     */
    @Synchronized
    fun resetConversation(config: LlmConversationConfig) {
        runBlocking { inference.setSystemPrompt(config.systemPrompt) }
    }

    @Synchronized
    private fun unloadModel() {
        runBlocking { inference.cleanUp() }
        loadedModelPath = null
    }

    /**
     * Unload the model and release its memory. The engine singleton stays alive and can
     * load again via [ensureLoaded].
     */
    @Synchronized
    fun close() {
        XLog.i(TAG, "close: unloading model")
        try {
            unloadModel()
        } catch (e: Exception) {
            XLog.w(TAG, "close: error unloading model", e)
        }
    }

    fun isReady(modelPath: String): Boolean =
        loadedModelPath == modelPath && inference.state.value is InferenceEngine.State.ModelReady

    companion object {
        private const val TAG = "LlmEngine"
    }
}
