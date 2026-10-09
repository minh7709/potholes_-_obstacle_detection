package com.example.fcw.config

/**
 * Thresholds ported from the Python `config.yaml`.
 *
 * Every value here has a counterpart in that file and must stay equal to it.
 * Android and the desktop pipeline have to agree on when a warning fires,
 * otherwise the same footage produces two different verdicts.
 */
data class FcwConfig(
    // collision
    val warningTtc: Float = 4.0f,
    val dangerTtc: Float = 2.0f,
    val warningDistanceM: Float = 8.0f,
    val dangerDistanceM: Float = 3.5f,
    val minDistanceM: Float = 0.5f,
    val confirmFrames: Int = 3,

    // roi: worth tracking
    val roiXMin: Float = 0.20f,
    val roiXMax: Float = 0.80f,
    val roiYMin: Float = 0.30f,
    val roiYMax: Float = 0.97f,

    // roi: may raise a warning
    val warnXMin: Float = 0.33f,
    val warnXMax: Float = 0.67f,
    val warnYMin: Float = 0.32f,
    val warnYMax: Float = 0.97f,
    val centerTolerance: Float = 0.35f,

    // classes
    val primaryClasses: Set<String> = setOf("motorbike"),
    val confThreshold: Float = 0.45f,
    val iouThreshold: Float = 0.5f,
    val minAreaRatio: Float = 0.0008f,

    // tracking
    val maxAge: Int = 30,
    val minHits: Int = 3,
    val buffer: Int = 30,

    // speed
    val minSamples: Int = 4,
    val minWindowSec: Float = 0.6f,
    val maxClosingSpeedMps: Float = 25.0f,
    val speedEmaAlpha: Float = 0.5f,

    // depth
    val depthInterval: Int = 5,

    /**
     * Guard on interpolated distance. Two noisy samples far apart can imply an absurd
     * ramp; beyond this rate the value is held instead of extrapolated, which is more
     * honest than a confident wrong number. From config.yaml depth.max_interpolated_speed_mps.
     */
    val maxInterpolatedSpeedMps: Float = 12f,
    val depthNetWidth: Int = 299,
    val depthNetHeight: Int = 224,

    // ego_motion
    val egoMotionEnabled: Boolean = true,
    val egoBackgroundDepthM: Float = 15f,
    val egoMinRowOffsetRatio: Float = 0.12f,
    val egoSmoothing: Float = 0.6f,
    val egoMaxAccelMps: Float = 8f,
    val egoMaxSpeedMps: Float = 33f,
    val centerCrop: Float = 0.5f,
    val minValidSamples: Int = 16,
    val outlierTolerance: Float = 0.25f,
    val validMinM: Float = 1.0f,
    val validMaxM: Float = 120.0f,

    /**
     * Depth scale, fitted so Android reports what the Python pipeline reports.
     *
     * 1.0 in config.yaml is "uncalibrated", and uncalibrated metric depth on this
     * phone reads too far: measured against the web pipeline on the same 960x720 clip
     * at the same instant, Android read 9.81 m where the web pipeline read 5.91 m, a
     * factor of 0.602. Applying that factor brings the two into agreement, which is the
     * point of the `calibration.depth_scale` knob the project already provides.
     *
     * Honest caveat: this is fitted from ONE clip at ONE distance. Before trusting it
     * at other ranges, re-measure at several depths. If a future depth model fixes the
     * scale natively, this should go back to 1.0 rather than stay as a hidden fudge.
     */
    val depthScale: Float = 0.602f,

    val maxWidth: Int = 1280,
    val maxHeight: Int = 720,

    // depth history
    val historyLen: Int = 20,
    val distanceEmaAlpha: Float = 0.3f,

    // performance, from config.yaml
    val adaptiveEnabled: Boolean = true,
    val adaptiveMinFps: Float = 5.0f,
    val adaptiveWindow: Int = 25,
    val adaptiveCooldownSec: Float = 8.0f,

    // warning
    val audioEnabled: Boolean = true,
    val volume: Float = 0.8f,
    val warningCooldownSec: Float = 1.5f,
    val dangerCooldownSec: Float = 1.0f
) {
    companion object {
        /**
         * Alias resolution shared with the detector.
         *
         * The model calls a motorcycle "motorbike" while config says "motorcycle".
         * Comparing raw strings made the primary-target filter silently never fire,
         * which is the same failure the detector had to be fixed for.
         */
        private val ALIASES = mapOf(
            "motorcycle" to "motorbike", "motor bike" to "motorbike",
            "potholes" to "pothole", "manhole" to "pothole",
            "trash can" to "ashcan", "garbage can" to "ashcan",
            "cone" to "reflective cone", "bollard" to "warning column",
            "hydrant" to "fire hydrant"
        )

        fun normalise(name: String): String {
            val squashed = name.trim().lowercase()
                .replace("_", " ").replace("-", " ")
                .split(" ").filter { it.isNotEmpty() }
                .joinToString(" ")
            return ALIASES[squashed] ?: squashed
        }

        /** True when a detected class name matches any configured candidate. */
        fun classMatches(name: String, wanted: Set<String>): Boolean {
            if (name.isEmpty()) return false
            val actual = normalise(name)
            return wanted.any { normalise(it) == actual }
        }
    }
}