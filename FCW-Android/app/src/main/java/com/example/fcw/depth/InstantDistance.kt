package com.example.fcw.depth

import kotlin.math.abs

/**
 * Range from a single frame, using the apparent width of a known-size object.
 *
 * The depth model answers this question properly but takes ~900 ms, and at roughly one
 * frame per second that is longer than the whole hazard window: a vehicle cutting in
 * can be gone again before a single depth sample lands. Geometry costs nothing and
 * answers on the frame it is asked.
 *
 *     d = f_px * realWidth / pixelWidth
 *
 * Accuracy is roughly plus or minus 30% on one frame. A vehicle is not a rigid body of
 * known width, it is seen in perspective, and its real width varies by model. That is
 * too coarse to replace the depth model, and it is not meant to: this is the estimate
 * used while depth has not produced a sample yet, and the depth model corrects it.
 */
object InstantDistance {

    /** Typical real-world width per class, in metres. Only warnable road classes. */
    private val WIDTH_M: Map<String, Float> = mapOf(
        "car" to 1.80f,
        "motorcycle" to 0.70f,
        "motorbike" to 0.70f,
        "tricycle" to 0.90f,
        "threewheeler" to 1.30f,
        "auto" to 1.40f,
        "van" to 1.90f,
        "truck" to 2.50f,
        "bus" to 2.55f
    )

    /** Below this the box is too small for a few pixels to say anything useful. */
    private const val MIN_BOX_PX: Float = 8f

    /** Beyond this the box is clipped by the frame and the estimate is meaningless. */
    private const val MAX_BOX_PX: Float = 900f

    /** Reject implausible readings outright rather than feeding them to the risk logic. */
    private const val MIN_M: Float = 0.4f
    private const val MAX_M: Float = 60f

    /** Classes this method can speak about. A pedestrian is tracked but not sized here. */
    fun knownWidth(className: String): Float? = WIDTH_M[className.lowercase()]


    /**
     * @param focalPx focal length in pixels for the analysis resolution, or 0 if unknown.
     * @return metres, or null when the inputs cannot support an estimate.
     */
    @Suppress("FloatingPointTypeParameter")
    fun estimate(focalPx: Float, boxWidthPx: Float, className: String): Float? {
        if (focalPx <= 1f || !focalPx.isFinite()) return null
        if (!boxWidthPx.isFinite() || boxWidthPx < MIN_BOX_PX || boxWidthPx > MAX_BOX_PX) return null
        val real: Float = knownWidth(className) ?: return null
        val metres: Float = focalPx * real / boxWidthPx
        if (!metres.isFinite() || metres < MIN_M || metres > MAX_M) return null
        return metres
    }

    /**
     * How far off this estimate can be trusted, as a fraction of its own value.
     *
     * Worth carrying rather than hiding: the emergency warning path uses a 6 m
     * threshold, and at 30% a true 8 m vehicle can land either side of it. Callers
     * that need a hard edge should widen the threshold by this factor.
     */
    const val RELATIVE_ERROR: Float = 0.30f

    /** Same estimate, as an interval. Anything outside is inconsistent with the frame. */
    fun range(focalPx: Float, boxWidthPx: Float, className: String): Pair<Float, Float>? {
        val m = estimate(focalPx, boxWidthPx, className) ?: return null
        val span = m * RELATIVE_ERROR
        return maxOf(MIN_M, m - span) to (m + span)
    }

    /** Kept for tests that want the raw quotient without the plausibility gates. */
    fun rawEstimate(focalPx: Float, boxWidthPx: Float, realWidthM: Float): Float =
        abs(focalPx * realWidthM / boxWidthPx)
}