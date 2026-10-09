package com.example.fcw.risk

import com.example.fcw.config.FcwConfig
import com.example.fcw.tracking.TrackHistory

/**
 * TTC, warning levels, forward-road ROI and target selection.
 *
 * Port of `app/fcw/risk.py`. The logic is kept identical to the Python side,
 * including the parts that look redundant, because both must reach the same verdict
 * on the same footage.
 */
object RiskEngine {

    enum class Level(val rank: Int) {
        NORMAL(0), WARNING(1), DANGER(2)
    }

    data class Assessment(
        val trackId: Int,
        val className: String,
        val level: Level,
        /** Level without the corridor gate, so the UI can explain "TTC 1.8s but NORMAL". */
        val levelUnrestricted: Level,
        val inWarningCorridor: Boolean,
        val levelFromTtc: Level,
        val levelFromDistance: Level,
        /** Range alone triggered it: the target is close but not closing. */
        val proximityTrigger: Boolean,
        val ttc: Float,
        val ttcValid: Boolean,
        val score: Float,
        val centreScore: Float,
        val inRoi: Boolean,
        val imminent: Boolean,
        val distanceSource: String,
        /** Smoothed distance in metres, or null. Used only to break sort ties. */
        val distanceForSort: Float? = null,
        /** Frames the same level has been held, for the confirmation rule. */
        val consecutiveCount: Int = 0,
        val confirmed: Boolean = false
    )

    /** Maps a TTC in seconds (or MAX_VALUE for infinity) to a level. */
    fun classifyTtc(ttc: Float, cfg: FcwConfig): Level = when {
        !ttc.isFinite() || ttc >= Float.MAX_VALUE -> Level.NORMAL
        ttc <= cfg.dangerTtc -> Level.DANGER
        ttc <= cfg.warningTtc -> Level.WARNING
        else -> Level.NORMAL
    }

    /**
     * Maps a range in metres to a level, independently of closing speed.
     *
     * This exists because TTC fails in the case that matters most: when the gap stops
     * shrinking, closing speed goes to zero, TTC becomes infinite, and pure TTC logic
     * reports NORMAL for a car stopped 3 m ahead. Range warns on its own and the two
     * levels are combined by taking the worse.
     */
    fun classifyDistance(distance: Float?, cfg: FcwConfig): Level = when {
        distance == null || !distance.isFinite() || distance <= 0f -> Level.NORMAL
        distance <= cfg.dangerDistanceM -> Level.DANGER
        distance <= cfg.warningDistanceM -> Level.WARNING
        else -> Level.NORMAL
    }

    fun worseLevel(a: Level, b: Level): Level = if (a.rank >= b.rank) a else b

    /**
     * The part of the analysis buffer the driver actually sees, as
     * left, top, width, height, falling back to the whole buffer.
     *
     * CameraX hands over the full analysis buffer but only marks a band of it as
     * visible, because the shared ViewPort crops the sensor stream to the preview's
     * aspect. On a real device the buffer was 1280x960 while the visible band was
     * y 162..798, so a third of the image was never displayed.
     *
     * A crop rect that does not fit inside the buffer is rejected rather than trusted:
     * its meaning changed between CameraX versions, and one reading of it puts it in
     * the coordinate space of the pre-crop original, where the values can exceed the
     * buffer. Falling back to the full buffer is the safe reading, because it is the
     * one that cannot be off by an unknown amount.
     */
    fun visibleRegion(region: IntArray, frameW: Int, frameH: Int): IntArray {
        val full = intArrayOf(0, 0, frameW, frameH)
        if (region.size < 4) return full
        val (left, top, w, h) = listOf(region[0], region[1], region[2], region[3])
        if (w <= 0 || h <= 0) return full
        if (left < 0 || top < 0) return full
        if (left + w > frameW || top + h > frameH) return full
        return intArrayOf(left, top, w, h)
    }

    /**
     * Forward-road ROI as left, top, right, bottom in buffer pixels.
     *
     * The fractions in the config apply to [region], not to the whole buffer. The ROI
     * is drawn through the same crop the preview uses, so measuring it against the
     * full buffer put its bottom edge below the visible band: with a 1280x960 buffer
     * visible only from y 162 to 798, the ROI ended at y 931, which mapped to a
     * screen position far below the bottom edge and ran the box off the display.
     */
    fun roadRoi(frameW: Int, frameH: Int, cfg: FcwConfig): IntArray =
        roadRoi(visibleRegion(intArrayOf(0, 0, frameW, frameH), frameW, frameH), cfg)

    fun roadRoi(region: IntArray, cfg: FcwConfig): IntArray = intArrayOf(
        region[0] + (region[2] * cfg.roiXMin).toInt(),
        region[1] + (region[3] * cfg.roiYMin).toInt(),
        region[0] + (region[2] * cfg.roiXMax).toInt(),
        region[1] + (region[3] * cfg.roiYMax).toInt()
    )

    /**
     * The narrow central band where a warning may be raised.
     *
     * The wide ROI says "road object worth tracking"; this says "directly ahead, worth
     * interrupting the driver for". Objects outside are still detected, measured and
     * drawn, they just cannot escalate. Alarming on roadside cones or oncoming traffic
     * teaches a driver to ignore the alarm.
     */
    fun warningCorridor(frameW: Int, frameH: Int, cfg: FcwConfig): IntArray =
        warningCorridor(visibleRegion(intArrayOf(0, 0, frameW, frameH), frameW, frameH), cfg)

    fun warningCorridor(region: IntArray, cfg: FcwConfig): IntArray = intArrayOf(
        region[0] + (region[2] * cfg.warnXMin).toInt(),
        region[1] + (region[3] * cfg.warnYMin).toInt(),
        region[0] + (region[2] * cfg.warnXMax).toInt(),
        region[1] + (region[3] * cfg.warnYMax).toInt()
    )

    fun centreInside(cx: Float, cy: Float, rect: IntArray): Boolean =
        cx >= rect[0] && cx <= rect[2] && cy >= rect[1] && cy <= rect[3]

    /** 0..1 score for proximity to the horizontal centre of the frame. */
    fun centreScore(cx: Float, frameW: Int, cfg: FcwConfig): Float {
        if (frameW <= 0) return 0f
        val offset = kotlin.math.abs(cx - frameW * 0.5f) / (frameW * 0.5f)
        val tol = maxOf(1e-3f, cfg.centerTolerance)
        return maxOf(0f, 1f - offset / tol)
    }

    /**
     * Full risk assessment for one tracked object.
     *
     * @param consecutive how many frames this level has already been held.
     */
    fun assess(
        history: TrackHistory,
        cx: Float,
        cy: Float,
        frameW: Int,
        frameH: Int,
        roi: IntArray,
        corridor: IntArray,
        cfg: FcwConfig,
        consecutive: Int = 0
    ): Assessment {
        val (ttc, ttcValid) = history.timeToCollision()
        val distance = history.distance

        val levelTtc = classifyTtc(ttc, cfg)
        val levelDist = classifyDistance(distance, cfg)
        val rawLevel = worseLevel(levelTtc, levelDist)

        val centre = centreScore(cx, frameW, cfg)
        val inRoi = centreInside(cx, cy, roi)
        val centered = centreInside(cx, cy, corridor)

        // An object beside the ego vehicle must not sound the alarm however close it
        // is: a motorcycle stopped 2 m away in the next lane is not a risk here.
        val level = if (centered) rawLevel else Level.NORMAL

        val proximity = if (distance == null || !distance.isFinite()) 0f
        else ((30f - distance) / 30f).coerceIn(0f, 1f)

        val ttcRisk = if (ttcValid && ttc.isFinite() && ttc < Float.MAX_VALUE) {
            (1f - ttc / maxOf(cfg.warningTtc, 1e-3f)).coerceIn(0f, 1f)
        } else {
            0f
        }

        val confirmed = level != Level.NORMAL &&
            history.framesSeen >= maxOf(cfg.minHits, 1) &&
            consecutive >= cfg.confirmFrames

        return Assessment(
            trackId = history.trackId,
            className = history.className,
            level = level,
            levelUnrestricted = rawLevel,
            inWarningCorridor = centered,
            levelFromTtc = levelTtc,
            levelFromDistance = levelDist,
            proximityTrigger = level == Level.DANGER &&
                levelDist == Level.DANGER && levelTtc != Level.DANGER,
            ttc = ttc,
            ttcValid = ttcValid,
            score = 0.65f * ttcRisk + 0.25f * proximity + 0.10f * centre,
            centreScore = centre,
            inRoi = inRoi,
            imminent = distance != null && distance.isFinite() &&
                distance > 0f && distance <= cfg.minDistanceM,
            distanceSource = history.distanceSource,
            distanceForSort = distance,
            consecutiveCount = consecutive,
            confirmed = confirmed
        )
    }

    /**
     * Picks the ONE object that drives the banner and the sound.
     *
     * Only one target is ever announced: overlapping alarms make a driving aid
     * unusable. Priority is DANGER before WARNING, then lower TTC, then nearer the
     * frame centre, then nearer the camera.
     */
    fun selectPrimaryTarget(
        assessments: List<Assessment>,
        cfg: FcwConfig
    ): Assessment? {
        if (assessments.isEmpty()) return null

        val candidates = assessments.filter {
            it.level != Level.NORMAL &&
                FcwConfig.classMatches(it.className, cfg.primaryClasses) &&
                it.inRoi
        }

        val pool = candidates.ifEmpty {
            // A very close object counts even with no TTC yet: a real hazard
            // regardless of the speed estimate being unavailable.
            assessments.filter {
                it.imminent &&
                    FcwConfig.classMatches(it.className, cfg.primaryClasses) &&
                    it.inRoi
            }
        }
        if (pool.isEmpty()) return null

        // Highest rank first: DANGER(2) before WARNING(1). Sorting ascending, the way
        // the Python code does with its inverted rank map, would announce the milder
        // of two real hazards.
        fun ttcKey(a: Assessment): Float =
            if (a.ttcValid && a.ttc < Float.MAX_VALUE) a.ttc else 1e6f
        fun distanceKey(a: Assessment): Float =
            a.distanceForSort?.takeIf { it.isFinite() } ?: 1e6f

        val byLevel = Comparator<Assessment> { a, b -> b.level.rank - a.level.rank }
        val byTtc = Comparator<Assessment> { a, b -> ttcKey(a).compareTo(ttcKey(b)) }
        val byCentre = Comparator<Assessment> { a, b -> b.centreScore.compareTo(a.centreScore) }
        val byDistance = Comparator<Assessment> { a, b -> distanceKey(a).compareTo(distanceKey(b)) }

        val sorted = pool.sortedWith(
            byLevel.thenComparing(byTtc).thenComparing(byCentre).thenComparing(byDistance)
        )
        return sorted.first()
    }


}

/**
 * Sustained-evidence counter.
 *
 * Depth noise produces a spurious very low TTC for single frames. Requiring N
 * consecutive frames at the same level removes nearly all of that at 10-20 FPS for
 * about 0.1-0.2 s of extra latency. A track that disappears has its streak dropped,
 * so a warning cannot be accumulated across two separate sightings.
 */
class LevelStreak {
    private val counts = HashMap<Int, Pair<RiskEngine.Level, Int>>()

    fun advance(trackId: Int, level: RiskEngine.Level): Int {
        val (previous, count) = counts[trackId] ?: (RiskEngine.Level.NORMAL to 0)
        val next = if (previous == level) count + 1 else 1
        counts[trackId] = level to next
        return next
    }

    fun get(trackId: Int): Int = counts[trackId]?.second ?: 0

    fun retainOnly(liveIds: Set<Int>) {
        counts.keys.filter { it !in liveIds }.forEach { counts.remove(it) }
    }

    fun reset() = counts.clear()
}