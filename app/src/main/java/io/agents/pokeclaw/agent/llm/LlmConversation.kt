// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw.agent.llm

import com.arm.aichat.InferenceEngine
import io.agents.pokeclaw.utils.XLog
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * A single chat session backed by the shared llama.cpp engine.
 *
 * Mirrors the LiteRT-LM Conversation: [reset] applies the system prompt (clearing the native
 * chat history), then each [sendMessage]/[streamSend] appends one user/tool message and
 * generates an assistant reply. The native engine keeps the chat-template history and KV
 * cache, so conversation history is preserved across turns.
 */
class LlmConversation internal constructor(
    private val engine: LlmEngine,
    private val config: LlmConversationConfig,
) {

    private val inference: InferenceEngine = engine.inference

    @Volatile
    private var generationJob: Job? = null

    /** Apply (or re-apply) the system prompt, clearing any previous chat history. */
    @Synchronized
    fun reset() {
        cancelGeneration()
        engine.resetConversation(config)
    }

    /** Blocking single-turn generation. Returns the full assistant text. */
    @Synchronized
    fun sendMessage(text: String): String = streamSend(text) {}

    /**
     * Blocking generation with a per-token streaming callback. Returns the full assistant text.
     * Callers may abort the generation from any thread via [cancel].
     */
    @Synchronized
    fun streamSend(text: String, onToken: (String) -> Unit): String {
        val sb = StringBuilder()
        runBlocking {
            val job = launch {
                inference.sendUserPrompt(
                    message = text,
                    predictLength = config.maxTokens,
                    temperature = config.sampler.temperature.toFloat(),
                    topK = config.sampler.topK,
                    topP = config.sampler.topP.toFloat(),
                    repeatPenalty = config.sampler.repeatPenalty.toFloat(),
                    seed = config.sampler.seed,
                ).collect { token ->
                    sb.append(token)
                    onToken(token)
                }
            }
            generationJob = job
            job.join()
            generationJob = null
        }
        return sb.toString()
    }

    /** Abort the in-flight generation at the next token boundary. */
    fun cancel() = cancelGeneration()

    private fun cancelGeneration() {
        generationJob?.let {
            XLog.d(TAG, "cancelling in-flight generation")
            it.cancel()
            generationJob = null
        }
    }

    /**
     * No-op for the shared engine: native chat state stays alive until the next
     * conversation calls [reset]. Kept for API parity with the LiteRT conversation close().
     */
    fun close() {
        cancelGeneration()
    }

    companion object {
        private const val TAG = "LlmConversation"
    }
}
