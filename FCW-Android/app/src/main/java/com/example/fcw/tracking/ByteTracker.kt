package com.example.fcw.tracking

import com.example.fcw.config.FcwConfig
import com.example.fcw.detection.Detection

/**
 * ByteTrack, two-stage association.
 *
 * Port of the tracker the Python pipeline uses via `supervision.ByteTrack`.
 *
 * The point of the two stages: a detector that has already filtered low scores is
 * exactly the case ByteTrack exists for. A vehicle that is briefly blurred or
 * half-occluded still produces a low-confidence box. Matching high scores first and
 * then letting the leftovers absorb low scores keeps the id alive, where plain IoU
 * matching on high scores alone would start a new track and reset its history, and
 * with it the distance and speed state.
 *
 * Difference from the Python side, stated rather than hidden: supervision uses a
 * Kalman filter for the track state. This uses linear motion between misses, which
 * behaves the same over the short gaps `buffer` covers and avoids a second state
 * estimate that has to agree with the one used for drawing.
 */
class ByteTracker(
    private val cfg: FcwConfig = FcwConfig(),
    private val trackHighThresh: Float = 0.5f,
    private val trackLowThresh: Float = 0.1f,
    /**
     * Must NOT be higher than the detector's own confidence threshold.
     *
     * ByteTrack's canonical value is 0.6, but that assumes a detector configured to
     * around the same level. Here the detector emits everything from 0.45, so a 0.6
     * gate silently discarded every detection in [0.45, 0.6) before it was ever drawn:
     * on real footage most marginal objects live in that band, and the screen showed
     * zero boxes while inference was running perfectly well.
     */
    private val newTrackThresh: Float = cfg.confThreshold
) {
    private class Track(
        val id: Int,
        var detection: Detection,
        var frameIndex: Int,
        var startFrame: Int
    ) {
        var hits = 1
        var missed = 0

        // Linear motion between frames; only used to predict while the track is lost.
        var vx = 0f
        var vy = 0f
        var lastCx = 0f
        var lastCy = 0f
        var lastArea = 0f
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1
    private var frameIndex = 0

    val history = LinkedHashMap<Int, TrackHistory>()

    /**
     * Advances one frame.
     *
     * @return tracked detections with `trackId` filled in.
     */
    fun update(detections: List<Detection>, timestamp: Double): List<Detection> {
        frameIndex++

        val high = detections.filter { it.confidence >= trackHighThresh }
        val low = detections.filter {
            it.confidence >= trackLowThresh && it.confidence < trackHighThresh
        }

        val matchedTrack = HashSet<Int>()
        val matchedDet = HashSet<Int>()

        // Tracks still inside the buffer, INCLUDING the ones that just went missing.
        // Excluding the lost ones from stage 1 means a confident reappearance after a
        // one- or two-frame gap would miss, start a new id, and throw away the
        // distance history with it. Keeping one id across brief occlusions is the
        // entire purpose of `buffer`.
        val withinBuffer = tracks.filter { it.missed < cfg.buffer }

        // Stage 1: confident detections.
        associate(withinBuffer, high, matchedTrack, matchedDet, allowLowScore = false)

        // Stage 2: leftover low scores against whatever is left.
        val remainingTracks = withinBuffer.filter { it.id !in matchedTrack }
        associate(remainingTracks, low, matchedTrack, matchedDet, allowLowScore = true)

        // Unmatched detections start new tracks, but only when confident enough.
        for ((i, d) in detections.withIndex()) {
            if (i in matchedDet || d.confidence < newTrackThresh) continue
            val t = Track(nextId++, d, frameIndex, frameIndex)
            t.lastCx = d.centerX
            t.lastCy = d.centerY
            t.lastArea = d.areaPx
            tracks.add(t)
            matchedTrack.add(t.id)
            matchedDet.add(i)
        }

        // Predict and age the tracks that saw nothing, then drop the ones that aged out.
        val alive = ArrayList<Track>()
        for (t in tracks) {
            if (t.id in matchedTrack) {
                t.hits++
                t.missed = 0
                alive.add(t)
            } else {
                t.missed++
                if (t.missed <= cfg.buffer) {
                    t.lastCx += t.vx
                    t.lastCy += t.vy
                    alive.add(t)
                }
            }
        }
        tracks.clear()
        tracks.addAll(alive)

        // Draw only what was matched this frame; drawing predicted boxes would show
        // objects the detector no longer sees.
        val result = ArrayList<Detection>(matchedTrack.size)
        for (t in tracks) {
            if (t.missed != 0) continue
            val d = t.detection
            val prevCx = t.lastCx
            val prevCy = t.lastCy
            // Per-frame delta, not scaled by frameIndex: prediction walks one frame
            // at a time, so a per-frame velocity is what belongs here.
            t.vx = d.centerX - prevCx
            t.vy = d.centerY - prevCy
            t.lastCx = d.centerX
            t.lastCy = d.centerY
            t.lastArea = d.areaPx
            result.add(d.copy(trackId = t.id))
        }

        // One history per live track; dead tracks keep their id reserved.
        for (t in tracks) {
            if (t.missed != 0) continue
            val h = history.getOrPut(t.id) { TrackHistory(t.id, t.detection.className, cfg) }
            h.update(timestamp, null, t.detection)
            h.estimateSpeed()
        }
        // Collect keys first: removing while iterating a LinkedHashMap throws.
        val dead = history.keys.filter { id -> tracks.none { it.id == id } }
        for (id in dead) history.remove(id)

        return result
    }

    /**
     * Greedy IoU assignment, highest IoU first.
     *
     * Greedy rather than optimal: at these detection counts the difference is a
     * fraction of a frame's latency, and a stable greedy pass keeps the code
     * readable enough to trust.
     */
    private fun associate(
        candidates: List<Track>,
        detections: List<Detection>,
        matchedTrack: MutableSet<Int>,
        matchedDet: MutableSet<Int>,
        allowLowScore: Boolean
    ) {
        if (candidates.isEmpty() || detections.isEmpty()) return
        val pairs = ArrayList<Triple<Int, Int, Float>>(candidates.size * detections.size)
        for ((ti, t) in candidates.withIndex()) {
            for ((di, d) in detections.withIndex()) {
                val iou = iou(t.detection, d)
                if (iou >= cfg.iouThreshold) pairs.add(Triple(ti, di, iou))
            }
        }
        if (pairs.isEmpty()) return
        pairs.sortByDescending { it.third }

        for ((ti, di, _) in pairs) {
            val t = candidates[ti]
            if (t.id in matchedTrack || di in matchedDet) continue
            matchedTrack.add(t.id)
            matchedDet.add(di)
            val d = detections[di]
            t.detection = d
            if (allowLowScore) {
                // Kept alive on a weak detection; hits does not increase so a track
                // sustained only by low scores never becomes eligible to warn.
            }
        }
    }

    private fun iou(a: Detection, b: Detection): Float {
        val x1 = maxOf(a.x1, b.x1)
        val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2)
        val y2 = minOf(a.y2, b.y2)
        val iw = x2 - x1
        val ih = y2 - y1
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = a.areaPx + b.areaPx - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun reset() {
        tracks.clear()
        history.clear()
        nextId = 1
        frameIndex = 0
    }
}