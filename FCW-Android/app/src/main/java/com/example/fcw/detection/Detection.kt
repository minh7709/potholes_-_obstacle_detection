package com.example.fcw.detection

/**
 * One detected object, in SOURCE frame pixels.
 *
 * Boxes are kept in source coordinates (not model-input coordinates) so the
 * overlay never has to know about letterboxing, and so later stages (depth
 * sampling, ROI tests) work in the same space the user sees.
 */
data class Detection(
    /** Assigned by ByteTracker in Phase 3; null while untracked. */
    val trackId: Int?,
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val areaPx: Float
) {
    val width: Float get() = x2 - x1
    val height: Float get() = y2 - y1
    val centerX: Float get() = (x1 + x2) * 0.5f
    val centerY: Float get() = (y1 + y2) * 0.5f

    fun areaRatio(frameWidth: Int, frameHeight: Int): Float {
        val total = frameWidth.toFloat() * frameHeight.toFloat()
        return if (total <= 0f) 0f else areaPx / total
    }
}
