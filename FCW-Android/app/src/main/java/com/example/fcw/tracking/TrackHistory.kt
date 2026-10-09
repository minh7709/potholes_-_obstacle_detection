package com.example.fcw.tracking

import com.example.fcw.detection.Detection
import com.example.fcw.config.FcwConfig

/**
 * Distance history and kinematics for one tracking id.
 *
 * Port of `app/fcw/tracking.py::TrackHistory`. Kept in step with the Python
 * version so both report the same level for the same footage.
 *
 * Depth does not arrive in Phase 3, so `update` is called with a null distance and
 * every field here stays unset. That is deliberate: reporting a guessed distance
 * would be worse than reporting none, because a wrong small distance sounds a
 * false alarm.
 */
class TrackHistory(
    val trackId: Int,
    var className: String = "unknown",
    private val cfg: FcwConfig = FcwConfig()
) {
    /** Raw (timestamp, distance) samples, oldest first. */
    private val samples = ArrayDeque<Sample>()

    var framesSeen: Int = 0
        private set
    var lastSeen: Double = 0.0
        private set
    var box: FloatArray = floatArrayOf(0f, 0f, 0f, 0f)
        private set
    var confidence: Float = 0f
        private set

    var rawDistance: Float? = null
        private set
    var distance: Float? = null
        private set
    var fitDistance: Float? = null
        private set
    var closingSpeed: Float = 0f
        private set
    var speedValid: Boolean = false
        private set
    var distanceSource: String = "none"
        private set

    private var speedEma: Float? = null

    private class Sample(val t: Double, val d: Float)

    /**
     * Feeds one frame's reading.
     *
     * @return false when there is no usable distance this frame, which is the normal
     *         case until depth is wired in.
     */
    fun update(
        timestamp: Double,
        distance: Float?,
        detection: Detection?,
        source: String = "measured"
    ): Boolean {
        framesSeen++
        lastSeen = timestamp
        if (detection != null) {
            box = floatArrayOf(detection.x1, detection.y1, detection.x2, detection.y2)
            confidence = detection.confidence
            className = detection.className
        }

        if (distance == null || !distance.isFinite() || distance <= 0f) return false
        rawDistance = distance

        if (distance != null) {
            val d = distance
            this.distance = if (this.distance == null) d
            else cfg.distanceEmaAlpha * d + (1f - cfg.distanceEmaAlpha) * this.distance!!
        }

        // Strictly increasing timestamps keep the regression well formed.
        val t = if (samples.isNotEmpty() && timestamp <= samples.last().t) {
            samples.last().t + 1e-3
        } else {
            timestamp
        }
        samples.addLast(Sample(t, distance))
        while (samples.size > cfg.historyLen) samples.removeFirst()
        distanceSource = source
        return true
    }

    /**
     * Relative closing speed in m/s by least-squares fit, positive when closing.
     *
     * A regression rather than a two-frame difference: 10% depth error on a 10 m
     * object is about 1 m, which at 30 FPS would give roughly +/-30 m/s from a naive
     * difference. Fitting the whole window suppresses that by an order of magnitude.
     */
    fun estimateSpeed(egoBiasMps: Float = 0f): Pair<Float, Boolean> {
        speedValid = false
        if (samples.size < cfg.minSamples) {
            closingSpeed = 0f
            return 0f to false
        }
        val span = samples.last().t - samples.first().t
        if (span < cfg.minWindowSec) {
            closingSpeed = 0f
            return 0f to false
        }

        // Recentre time: raw timestamps grow without bound and lose precision.
        val tRel = DoubleArray(samples.size) { samples[it].t - samples.last().t }
        val dAll = FloatArray(samples.size) { samples[it].d }
        val tMean = tRel.average()
        val dMean = dAll.average()
        var varT = 0.0
        var cov = 0.0
        for (i in tRel.indices) {
            varT += (tRel[i] - tMean) * (tRel[i] - tMean)
            cov += (tRel[i] - tMean) * (dAll[i] - dMean)
        }
        val n = tRel.size
        val variance = varT / n
        if (variance < 1e-9) {
            closingSpeed = 0f
            return 0f to false
        }

        val slope = (cov / n / variance).toFloat()   // d(distance)/dt, negative when closing
        val speed = (-slope) - egoBiasMps            // positive means closing
        val clamped = speed.coerceIn(-cfg.maxClosingSpeedMps, cfg.maxClosingSpeedMps)

        // Distance implied by the SAME fitted line at the newest sample. Using the
        // smoothed EMA distance instead would pair a lagged distance with an unlagged
        // slope and systematically over-report TTC.
        //
        // The intercept MUST use the raw slope, not the sign-flipped speed: they differ
        // in sign, so substituting one for the other inflated every distance and pushed
        // real TTCs from 2.8 s up to 4.0 s, which would have suppressed warnings exactly
        // when they matter.
        val intercept = (dMean - slope * tMean).toFloat()
        fitDistance = if (egoBiasMps != 0f) {
            intercept - egoBiasMps * (span * 0.5).toFloat()
        } else {
            intercept
        }

        val a = cfg.speedEmaAlpha
        speedEma = speedEma?.let { a * clamped + (1f - a) * it } ?: clamped

        closingSpeed = speedEma!!
        speedValid = true
        return closingSpeed to true
    }

    /**
     * TTC in seconds, or infinity.
     *
     * Defined only for a genuinely closing target: a non-positive closing speed
     * means the gap is not shrinking, which is a known safe state and must stay
     * distinct from "not enough data yet".
     */
    fun timeToCollision(): Pair<Float, Boolean> {
        if (!speedValid) return Float.MAX_VALUE to false
        val d = fitDistance ?: distance
        if (d == null || !d.isFinite() || d <= 0f) return Float.MAX_VALUE to false
        if (closingSpeed <= 1e-3f) return Float.MAX_VALUE to false
        return d / closingSpeed to true
    }

    /**
     * Best available distance estimate, extrapolated across frames with no reading.
     *
     * @return the distance and its source: "measured", "interpolated", "held" or
     *         "none". Never reported as measured when it is not.
     */
    fun interpolatedDistance(timestamp: Double, maxRateMps: Float): Pair<Float?, String> {
        if (samples.size < 2) {
            return if (samples.size == 1) samples.last().d to "held" else null to "none"
        }
        val a = samples[samples.size - 2]
        val b = samples[samples.size - 1]
        val dt = b.t - a.t
        if (dt <= 1e-6) return b.d to "held"

        val rate = ((b.d - a.d) / dt).toFloat()
        // Two noisy samples far apart can imply an absurd ramp; hold instead.
        if (kotlin.math.abs(rate) > maxRateMps) return b.d to "held"

        val predicted = b.d + rate * (timestamp - b.t).toFloat()
        if (!predicted.isFinite() || predicted <= 0f) return b.d to "held"
        return predicted to "interpolated"
    }
}