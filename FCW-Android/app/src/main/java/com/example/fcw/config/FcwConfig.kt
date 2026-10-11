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
    val roiXMin: Float = 0.10f,
    val roiXMax: Float = 0.90f,
    val roiYMin: Float = 0.22f,
    val roiYMax: Float = 1.00f,

    // roi: may raise a warning
    val warnXMin: Float = 0.22f,
    val warnXMax: Float = 0.78f,
    val warnYMin: Float = 0.26f,
    val warnYMax: Float = 1.00f,
    val centerTolerance: Float = 0.35f,

    // classes
    val primaryClasses: Set<String> = setOf("motorbike"),
    /**
     * YOLO network input edge, in pixels.
     *
     * This must match the exported ONNX exactly. Both `best.onnx` and `best_fp16.onnx`
     * are exported with static 640x640 inputs, so feeding a 448 tensor throws on every
     * inference and the HUD reports INFERENCE with a rising fail count. Lowering it
     * without re-exporting is not a speed knob, it is a crash.
     *
     * Measured cost of a smaller edge is real: at 448 the same clip yielded 53
     * detections against 73 at 640, and lost 4 of 5 on its first frame, which is
     * exactly the distant cut-in vehicle the warning exists for. Keep this at 640
     * unless the model asset is re-exported to match.
     */
    val yoloInputSize: Int = 640,

    /**
     * Emergency thresholds that bypass the multi-frame confirmation.
     *
     * The confirmation rules (min_hits 3, confirm_frames 3, speed min_samples 4) are
     * right for a vehicle already ahead in the same lane, which stays in view for many
     * seconds. They are wrong for a cut-in or a motorcycle appearing: those objects are
     * new and occupy the corridor for only a moment, and at 1.4 FPS the confirmation
     * alone takes about 3.6 s, by which time the hazard is behind the windscreen.
     *
     * So a corridor object that is already very close, or already closing very fast,
     * is allowed to sound the alarm on the first frame it is seen. The corridor test
     * and these two numbers are what suppress false positives here, not track history.
     *
     * Cost: more false alarms for near misses at speed. Bought deliberately.
     */
    val urgentDistanceM: Float = 6.0f,
    val urgentTtc: Float = 1.5f,

    /**
     * Classes allowed to use the emergency path. A pedestrian on the pavement is
     * tracked but never alarms at this range, so it is not on the list.
     */
    val urgentClasses: Set<String> = setOf("car", "motorcycle", "motorbike", "truck", "bus", "van", "auto", "threewheeler"),

    val confThreshold: Float = 0.45f,
    val iouThreshold: Float = 0.5f,
    val minAreaRatio: Float = 0.0008f,

    // tracking
    val maxAge: Int = 30,

    /**
     * Frames before a track counts as established. Dropped from 3 to 1.
     *
     * At 3 and roughly 1 FPS this alone cost 3.3 s before a vehicle that had just cut
     * in could be assessed at all, and `Track 0/0` on the test device was the symptom:
     * the confirmation lived in the tracker, one layer below the emergency warning, so
     * lowering the warning thresholds could never reach a cut-in.
     *
     * A one-frame track is still not enough to sound the alarm on its own. The
     * emergency path in RiskEngine.assess additionally requires the object to sit in
     * the warning corridor, be a road-vehicle class, and be measured at
     * [urgentDistanceM] or closing at [urgentTtc]. A first sighting cannot have a
     * valid TTC at all, because TTC needs speed.min_samples distance samples first,
     * so in practice only a genuine close-range reading can set the alarm off.
     * Non-urgent tracks still need [confirmFrames] consecutive agreeing frames.
     */
    val minHits: Int = 1,
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