package com.example.fcw.performance

import com.example.fcw.config.FcwConfig

/**
 * Adaptive load control.
 *
 * Port of `app/fcw/monitor.py::AdaptiveController`.
 *
 * The rule this encodes is the important one: **only sample rates are adapted, never
 * the TTC thresholds.** Lowering the warning thresholds to buy frames would trade the
 * thing the system exists for — warning early enough to brake — for a number on a
 * counter. Load is reduced instead: fewer detections per second and longer gaps
 * between depth measurements, both of which the tracker interpolates across.
 *
 * Degradation needs `window` consecutive slow samples plus a cooldown, so a single
 * slow frame (first model call, a garbage-collection pause) cannot trigger it, and a
 * brief recovery resets the streak.
 */
class AdaptiveController(private val cfg: FcwConfig = FcwConfig()) {

    data class Step(val targetFps: Int, val depthInterval: Int)

    data class Change(
        val level: Int,
        val targetFps: Int,
        val depthInterval: Int,
        val reason: String
    )

    // config.yaml performance.adaptive_steps, in order.
    private val steps = listOf(
        Step(10, 5), Step(8, 6), Step(6, 8), Step(4, 10)
    )

    var level: Int = 0
        private set
    private var slowSamples = 0
    private var lastChange: Double? = null
    val changes = ArrayList<Change>()

    val targetFps: Int get() = steps[level].targetFps
    val depthInterval: Int get() = steps[level].depthInterval

    /**
     * @return the new step when it degraded, otherwise null.
     */
    fun maybeDegrade(measuredFps: Float, now: Double): Change? {
        if (!cfg.adaptiveEnabled) return null
        if (level >= steps.size - 1) return null

        if (measuredFps <= 0f || measuredFps >= cfg.adaptiveMinFps) {
            // Recovery clears the streak: a brief good patch should not leave the
            // controller one sample away from degrading.
            slowSamples = 0
            return null
        }

        slowSamples++
        if (slowSamples < cfg.adaptiveWindow) return null
        val last = lastChange
        if (last != null && now - last < cfg.adaptiveCooldownSec) return null

        level++
        slowSamples = 0
        lastChange = now
        val step = steps[level]
        val change = Change(
            level, step.targetFps, step.depthInterval,
            "xử lý ${"%.1f".format(measuredFps)} < ${"%.1f".format(cfg.adaptiveMinFps)} FPS"
        )
        changes.add(change)
        return change
    }

    /** Every measured sample, for the HUD. */
    fun noteFps(measuredFps: Float, now: Double): Boolean {
        val change = maybeDegrade(measuredFps, now) ?: return false
        return true
    }

    fun reset() {
        level = 0
        slowSamples = 0
        lastChange = null
        changes.clear()
    }
}