package com.example.fcw.detection

/**
 * Decodes the raw YOLO head output and applies NMS.
 *
 * Measured output layout of `best.onnx` (MIGRATION_PLAN.md §1.2):
 *   shape [1, 19, 8400], where 19 = 4 box values + 15 class scores
 *   box values are PIXELS of the 640x640 input, NOT normalised 0..1
 *   class scores are already sigmoid-activated
 *   the graph contains NO NMS, so it must be applied here
 *
 * This decoder was validated against `model.predict()` on real footage and
 * reproduced the same boxes and confidences exactly (see MIGRATION_PLAN.md §1.3),
 * so any divergence on device points at the preprocessor, not at this code.
 */
class YoloPostprocessor(
    private val inputSize: Int = 640,
    private val iouThreshold: Float = 0.5f
) {
    /** Number of channels: 4 box + one per class. Derived, never hard-coded. */
    private var numChannels: Int = 0
    private var numClasses: Int = 0
    private var numAnchors: Int = 0

    /**
     * Learns the tensor shape from the first real output.
     * @return true when the shape is usable.
     */
    fun configure(channels: Int, anchors: Int): Boolean {
        if (channels <= 4 || anchors <= 0) return false
        numChannels = channels
        numAnchors = anchors
        numClasses = channels - 4
        return true
    }

    fun classCount(): Int = numClasses

    /**
     * Extracts boxes and applies class-aware NMS.
     *
     * @param output flattened output, [numChannels, numAnchors]
     * @param names class names by index; size must be >= numClasses
     * @param confThreshold minimum score to keep
     * @param preprocessor used to map boxes back to source pixels
     * @return detections sorted by descending area
     */
    fun decode(
        output: FloatArray,
        names: Array<String>,
        confThreshold: Float,
        preprocessor: YoloPreprocessor
    ): List<Detection> {
        if (numChannels == 0) return emptyList()

        val candidates = ArrayList<Candidate>(64)
        for (a in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f
            for (c in 0 until numClasses) {
                // Class scores start at channel 4, after cx/cy/w/h. Reading from
                // `c` instead of `4 + c` shifts every score by four classes and
                // silently mislabels detections.
                val score = output[(4 + c) * numAnchors + a]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            if (bestClass < 0 || bestScore < confThreshold) continue

            val cx = output[0 * numAnchors + a]
            val cy = output[1 * numAnchors + a]
            val w = output[2 * numAnchors + a]
            val h = output[3 * numAnchors + a]

            val mapped = preprocessor.toSourceSpace(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
            candidates.add(
                Candidate(mapped[0], mapped[1], mapped[2], mapped[3], bestClass, bestScore)
            )
        }

        return nms(candidates)
            .map { c ->
                val bw = c.x2 - c.x1
                val bh = c.y2 - c.y1
                Detection(
                    trackId = null,          // assigned by ByteTracker in Phase 3
                    classId = c.classId,
                    className = names.getOrElse(c.classId) { "class_${c.classId}" },
                    confidence = c.score,
                    x1 = c.x1, y1 = c.y1, x2 = c.x2, y2 = c.y2,
                    areaPx = bw * bh
                )
            }
            .sortedByDescending { it.areaPx }
    }

    /**
     * Class-aware NMS.
     *
     * Class-aware matters: without it, a motorcycle overlapping a car would have
     * its box suppressed by the car, and the object we most need to warn about
     * would be the one deleted.
     */
    private fun nms(candidates: List<Candidate>): List<Candidate> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending { it.score }
        val kept = ArrayList<Candidate>(sorted.size)
        for (c in sorted) {
            var suppressed = false
            for (k in kept) {
                if (k.classId != c.classId) continue
                val iw = minOf(c.x2, k.x2) - maxOf(c.x1, k.x1)
                val ih = minOf(c.y2, k.y2) - maxOf(c.y1, k.y1)
                if (iw <= 0f || ih <= 0f) continue
                val inter = iw * ih
                val areaA = (c.x2 - c.x1) * (c.y2 - c.y1)
                val areaB = (k.x2 - k.x1) * (k.y2 - k.y1)
                val iou = inter / (areaA + areaB - inter)
                if (iou > iouThreshold) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) kept.add(c)
        }
        return kept
    }

    private data class Candidate(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val classId: Int, val score: Float
    )
}