// Copyright 2026 PokeClaw (agents.io). All rights reserved.
// Licensed under the Apache License, Version 2.0.

package io.agents.pokeclaw

import io.agents.pokeclaw.channel.Channel
import io.agents.pokeclaw.utils.XLog
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

enum class TaskSessionPhase {
    IDLE,
    RUNNING,
    STOPPING,
}

data class TaskSessionState(
    val phase: TaskSessionPhase = TaskSessionPhase.IDLE,
    val messageId: String = "",
    val channel: Channel? = null,
    val taskText: String = "",
    val startedAtMillis: Long = 0L,
    val stopRequested: Boolean = false,
    val autoReturnToChat: Boolean = false,
) {
    val isRunning: Boolean
        get() = phase != TaskSessionPhase.IDLE && messageId.isNotEmpty()
}

/**
 * Single authoritative state holder for the currently running task session.
 *
 * The orchestrator mutates this store; UI and service layers can observe it
 * without having to infer task truth from multiple ad-hoc fields.
 */
class TaskSessionStore {

    companion object {
        private const val TAG = "TaskSessionStore"
    }

    private val lock = Any()
    private val _state = MutableStateFlow(TaskSessionState())
    val state: StateFlow<TaskSessionState> = _state

    fun snapshot(): TaskSessionState = _state.value

    fun isTaskRunning(): Boolean = _state.value.isRunning

    fun tryAcquire(
        messageId: String,
        channel: Channel,
        taskText: String = "",
        autoReturnToChat: Boolean = channel == Channel.LOCAL,
    ): Boolean {
        synchronized(lock) {
            if (_state.value.isRunning) {
                XLog.w(TAG, "BusyState: tryAcquire rejected (session already BUSY: ${_state.value.messageId}/${_state.value.channel})")
                return false
            }
            _state.value = TaskSessionState(
                phase = TaskSessionPhase.RUNNING,
                messageId = messageId,
                channel = channel,
                taskText = taskText,
                startedAtMillis = System.currentTimeMillis(),
                stopRequested = false,
                autoReturnToChat = autoReturnToChat,
            )
            XLog.i(TAG, "BusyState: task session IDLE -> BUSY (messageId=$messageId, channel=$channel, task='${taskText.take(80)}')")
            return true
        }
    }

    fun updateTaskText(taskText: String) {
        synchronized(lock) {
            val current = _state.value
            if (!current.isRunning || current.taskText == taskText) return
            _state.value = current.copy(taskText = taskText)
        }
    }

    fun markStopping(): Boolean {
        synchronized(lock) {
            val current = _state.value
            if (!current.isRunning) return false
            if (current.phase == TaskSessionPhase.STOPPING && current.stopRequested) return false
            _state.value = current.copy(
                phase = TaskSessionPhase.STOPPING,
                stopRequested = true,
            )
            XLog.i(TAG, "BusyState: task session BUSY -> STOPPING (messageId=${current.messageId}, stopRequested=true)")
            return true
        }
    }

    fun release(): TaskSessionState {
        synchronized(lock) {
            val current = _state.value
            _state.value = TaskSessionState()
            if (current.isRunning) {
                XLog.i(TAG, "BusyState: task session ${current.phase} -> IDLE (release, messageId=${current.messageId}, channel=${current.channel})")
            } else {
                XLog.d(TAG, "BusyState: release() called while already IDLE (no-op)")
            }
            return current
        }
    }
}
