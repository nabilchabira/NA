// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw.agent.llm

import android.content.Context
import io.agents.pokeclaw.utils.XLog

data class LocalEngineLease(
    val engine: LlmEngine,
    val backendLabel: String = "CPU",
)

data class LocalConversationLease(
    val engine: LlmEngine,
    val conversation: LlmConversation,
    val backendLabel: String = "CPU",
)

data class LocalSingleShotResult(
    val text: String?,
    val backendLabel: String = "CPU",
)

/**
 * Shared local (llama.cpp) runtime facade.
 *
 * Public surface kept identical to the previous LiteRT version so callers
 * (LocalLlmClient, LlmSessionManager, ChatSessionController) did not need redesigning.
 */
object LocalModelRuntime {

    private const val TAG = "LocalModelRuntime"
    private const val DEFAULT_RETRY_COUNT = 5
    private const val DEFAULT_RESET_ATTEMPT = 3
    private const val DEFAULT_RETRY_SLEEP_MS = 1500L

    fun acquireSharedEngine(
        context: Context,
        modelPath: String,
        preferCpu: Boolean = false,
    ): LocalEngineLease {
        val engine = EngineHolder.getOrCreate(modelPath, context.cacheDir.path)
        return LocalEngineLease(engine = engine, backendLabel = "CPU")
    }

    fun forceCpuEngine(context: Context, modelPath: String): LocalEngineLease {
        resetSharedEngine()
        val engine = EngineHolder.getOrCreate(modelPath, context.cacheDir.path)
        return LocalEngineLease(engine = engine, backendLabel = "CPU")
    }

    fun resetSharedEngine() {
        try {
            EngineHolder.close()
        } catch (e: Exception) {
            XLog.w(TAG, "resetSharedEngine: close failed", e)
        }
    }

    fun currentBackendLabel(modelPath: String?): String? {
        return EngineHolder.getBackendLabel(modelPath)
    }

    fun openConversation(
        context: Context,
        modelPath: String,
        conversationConfig: LlmConversationConfig,
        preferCpu: Boolean = false,
        maxRetries: Int = DEFAULT_RETRY_COUNT,
    ): LocalConversationLease {
        var lastError: Exception? = null

        for (attempt in 1..maxRetries) {
            try {
                val engineLease = acquireSharedEngine(context, modelPath, preferCpu = preferCpu)
                val conversation = engineLease.engine.createConversation(conversationConfig)
                conversation.reset() // applies the system prompt and clears native history
                return LocalConversationLease(
                    engine = engineLease.engine,
                    conversation = conversation,
                    backendLabel = engineLease.backendLabel,
                )
            } catch (e: Exception) {
                lastError = e
                XLog.w(TAG, "openConversation attempt $attempt failed for $modelPath: ${e.message}")

                if (attempt == DEFAULT_RESET_ATTEMPT) {
                    XLog.w(TAG, "openConversation: resetting shared runtime for $modelPath")
                    try {
                        resetSharedEngine()
                    } catch (resetError: Exception) {
                        XLog.e(TAG, "openConversation: shared runtime reset failed", resetError)
                    }
                }

                if (attempt < maxRetries) {
                    Thread.sleep(DEFAULT_RETRY_SLEEP_MS)
                }
            }
        }

        throw RuntimeException(
            "Failed to create conversation after $maxRetries retries: ${lastError?.message}",
            lastError
        )
    }

    fun runSingleShot(
        context: Context,
        modelPath: String,
        systemPrompt: String,
        prompt: String,
        temperature: Double = 0.3,
        preferCpu: Boolean = false,
    ): LocalSingleShotResult {
        val lease = openConversation(
            context = context,
            modelPath = modelPath,
            conversationConfig = LlmConversationConfig(
                systemPrompt = systemPrompt,
                sampler = LlmSamplerConfig(temperature = temperature),
            ),
            preferCpu = preferCpu,
        )

        return try {
            val response = lease.conversation.sendMessage(prompt)
            LocalSingleShotResult(
                text = response.trim().ifEmpty { null },
                backendLabel = lease.backendLabel,
            )
        } finally {
            try {
                lease.conversation.close()
            } catch (e: Exception) {
                XLog.w(TAG, "runSingleShot: conversation close failed", e)
            }
        }
    }

    /** llama.cpp runs on CPU only — kept for API compatibility with existing callers. */
    fun isGpuBackendFailure(error: Throwable?): Boolean = false

    /** No single-session conflict exists with the llama.cpp runtime — kept for API compatibility. */
    fun isSessionConflict(error: Throwable?): Boolean = false
}
