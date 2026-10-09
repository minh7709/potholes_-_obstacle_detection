package com.example.fcw.utils

/**
 * Rolling FPS over a fixed window.
 *
 * A window rather than a running average: the adaptive controller needs to react
 * to a sustained slowdown, not to one slow frame. Averaging since startup would
 * hide exactly the drop it is supposed to detect.
 */
class FpsCounter(private val windowSize: Int = 30) {

    private val timestamps = ArrayDeque<Long>(windowSize)

    @Synchronized
    fun onFrame(timeMs: Long = System.currentTimeMillis()) {
        timestamps.addLast(timeMs)
        while (timestamps.size > windowSize) timestamps.removeFirst()
    }

    fun fps(): Double {
        if (timestamps.size < 2) return 0.0
        val span = (timestamps.last() - timestamps.first()).toDouble()
        return if (span <= 0.0) 0.0 else (timestamps.size - 1) * 1000.0 / span
    }

    fun count(): Int = timestamps.size

    @Synchronized
    fun reset() = timestamps.clear()
}
