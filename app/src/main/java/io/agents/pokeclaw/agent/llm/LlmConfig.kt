// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw.agent.llm

/**
 * Sampling configuration for the llama.cpp engine.
 *
 * Mirrors the sampling values the app previously passed to LiteRT-LM's SamplerConfig
 * (topK = 64, topP = 0.95, temperature). Every value is passed through to llama.cpp
 * unchanged — nothing is hardcoded inside the engine.
 */
data class LlmSamplerConfig(
    val temperature: Double = 0.3,
    val topK: Int = 64,
    val topP: Double = 0.95,
    val repeatPenalty: Double = 1.0,
    val seed: Int = -1,
)

/**
 * Configuration for a single chat session (replaces LiteRT-LM ConversationConfig + Contents).
 *
 * @param systemPrompt system instruction applied when the conversation is (re)created
 * @param sampler      sampling parameters for every generation in this conversation
 * @param maxTokens    maximum number of tokens generated per turn (matches the previous
 *                     LiteRT maxNumTokens = 8192)
 */
data class LlmConversationConfig(
    val systemPrompt: String,
    val sampler: LlmSamplerConfig = LlmSamplerConfig(),
    val maxTokens: Int = 8192,
)
