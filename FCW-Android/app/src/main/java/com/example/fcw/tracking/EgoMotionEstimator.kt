package com.example.fcw.tracking

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Ego-vehicle speed from optical flow of the static background.
 *
 * Port of `app/fcw/tracking.py::EgoMotionEstimator`.
 *
 * Why it exists: the closing speed of a stopped car in front of you is only ever
 * apparent. The camera moves, so the car appears to recede. Without subtracting the
 * ego speed, a stationary obstacle reads as "closing at 10 m/s" and the app warns
 * about a parked car. This is the difference between a usable FCW and one that
 * screams at every junction.
 *
 * Model. For pure forward translation of a static scene, a pixel at row offset
 * `yr = y - cy` shifts by `dy = yr * v_ego * dt / Z`, so `v_ego = Z * dy / (yr * dt)`.
 *
 * There is deliberately NO focal length in that formula: the pinhole projection
 * already folded f into pixel coordinates. Multiplying by f again inflates the
 * estimate by roughly 1300x and still produces plausible-looking numbers.
 *
 * Differences from the Python version, stated rather than hidden:
 *  - `cv2.calcOpticalFlowPyrLK` is replaced by a pyramidal Lucas-Kanade written here,
 *    because OpenCV is not available on Android. Python tracks between consecutive
 *    video frames; here depth arrives every few seconds, so between consecutive depth
 *    frames the flow is tens to hundreds of pixels. A single-level LK cannot follow
 *    that, so the tracker runs coarse-to-fine over three levels.
 *  - Robustness is kept: far background only, points away from the horizon, one
 *    least-squares fit, MAD outlier rejection, refit, acceleration clamp, smoothing.
 */
class EgoMotionEstimator(
    private val enabled: Boolean = true,
    private val backgroundDepthM: Float = 15f,
    private val minRowOffsetRatio: Float = 0.12f,
    private val smoothing: Float = 0.6f,
    private val maxAccelMps: Float = 8f,
    private val maxSpeedMps: Float = 33f
) {
    /** Grid density, matching the Python `linspace` grid: 20 x 12. */
    private val gridCols = 20
    private val gridRows = 12

    /**
     * Downscale factors, coarse to fine.
     *
     * Depth frames arrive seconds apart, so between two of them the background can
     * move well over a hundred pixels. Starting at 1/8 scale makes that followable in
     * a 21px window; each finer level then only has to resolve its own residual.
     */
    private val scales = intArrayOf(8, 2, 1)

    private var prevGray: FloatArray? = null
    private var prevTime: Double = 0.0
    private var prevW = 0
    private var prevH = 0

    var speedMps: Float = 0f
        private set
    var quality: Float = 0f
        private set

    /** Points the tracker returned, before the depth and horizon gates. Exposed so a
     *  zero speed can be told apart from "nothing was trackable" and "gates rejected
     *  everything" without adding logging. */
    var trackedPoints: Int = 0
        private set

    /** Why points were dropped, so a zero speed can be diagnosed rather than guessed. */
    var rejectedBorder: Int = 0
        private set
    var rejectedTexture: Int = 0
        private set
    var acceptedPoints: Int = 0
        private set

    /** Speed from the fit, before the sign clamp. Diagnostic: distinguishes "the fit
     *  produced a negative number" from "the clamp removed it". */
    var rawSpeed: Float = 0f
        private set

    fun reset() {
        prevGray = null
        prevTime = 0.0
        prevW = 0
        prevH = 0
        speedMps = 0f
        quality = 0f
        trackedPoints = 0
    }

    /**
     * Estimates ego speed in m/s. Positive means moving forward.
     *
     * @param gray current frame as luminance, row-major, width*height
     * @param depth depth in metres at the same resolution as [gray]; may be smaller,
     *              in which case it is sampled with nearest-neighbour scaling
     */
    fun update(
        gray: FloatArray,
        width: Int,
        height: Int,
        timestamp: Double,
        depth: FloatArray,
        depthWidth: Int,
        depthHeight: Int,
        biasGain: Float = 1f
    ): Float {
        if (!enabled || gray.isEmpty() || depth.isEmpty()) {
            speedMps = 0f
            return 0f
        }

        val previous = prevGray
        val previousTime = prevTime
        prevGray = gray
        prevTime = timestamp

        if (previous == null || width != prevW || height != prevH) {
            prevW = width
            prevH = height
            speedMps = 0f
            return 0f
        }
        val dt = timestamp - previousTime
        if (dt <= 1e-3) {
            speedMps = 0f
            return 0f
        }

        val cy = height * 0.5f
        val pairs = trackFlow(previous, gray, width, height)
        trackedPoints = pairs.size
        if (pairs.size < 20) {
            speedMps = 0f
            quality = 0f
            return 0f
        }

        // Per-point depth, never one median for the frame: mixing depths of different
        // ranges into one number is what makes scene-flow ego estimates drift.
        val coeff = ArrayList<Double>(pairs.size)
        val dyList = ArrayList<Double>(pairs.size)
        var valid = 0

        for ((x0, y0, y1) in pairs) {
            val z = sampleDepth(depth, depthWidth, depthHeight, x0, y0, width, height)
            val yr = y0 - cy
            if (z <= 0f || z < backgroundDepthM) continue
            if (abs(yr) <= height * minRowOffsetRatio) continue
            coeff.add((yr.toDouble() * dt) / z.toDouble())
            dyList.add((y1 - y0).toDouble())
            valid++
        }
        quality = if (pairs.isEmpty()) 0f else valid.toFloat() / pairs.size
        if (valid < 12) {
            speedMps = 0f
            return 0f
        }

        val initial = leastSquares(coeff, dyList)
        if (initial == null) {
            speedMps = 0f
            return 0f
        }

        // One MAD pass, then refit. More stable than a median of per-point ratios.
        val residuals = DoubleArray(coeff.size) { dyList[it] - coeff[it] * initial }
        val centre = median(residuals)
        val mad = median(DoubleArray(residuals.size) { abs(residuals[it] - centre) })
        var speed = initial
        if (mad > 1e-9) {
            val keepCoeff = ArrayList<Double>()
            val keepDy = ArrayList<Double>()
            for (i in coeff.indices) {
                if (abs(residuals[i] - centre) <= 3.0 * mad) {
                    keepCoeff.add(coeff[i])
                    keepDy.add(dyList[i])
                }
            }
            if (keepCoeff.size >= 12) {
                speed = leastSquares(keepCoeff, keepDy) ?: initial
            }
        }

        rawSpeed = speed.toFloat()
        var result = speed.coerceIn(0.0, maxSpeedMps.toDouble()).toFloat()

        val previousSpeed = speedMps
        if (previousSpeed > 0f) {
            val maxDelta = maxAccelMps * dt.toFloat() + 0.5f
            result = result.coerceIn(previousSpeed - maxDelta, previousSpeed + maxDelta)
            speedMps = smoothing * result + (1f - smoothing) * previousSpeed
        } else {
            speedMps = result
        }
        return speedMps
    }

    /**
     * Bias to subtract from an apparent closing speed.
     *
     * A static object shows apparent closing equal to ego speed, first order.
     */
    fun biasMps(biasGain: Float = 1f): Float = speedMps * (if (biasGain == 0f) 1f else biasGain)

    private fun leastSquares(coeff: List<Double>, dy: List<Double>): Double? {
        var denom = 0.0
        var num = 0.0
        for (i in coeff.indices) {
            denom += coeff[i] * coeff[i]
            num += coeff[i] * dy[i]
        }
        if (denom < 1e-12) return null
        return num / denom
    }

    private fun median(values: DoubleArray): Double {
        val copy = values.copyOf()
        copy.sort()
        return copy[copy.size / 2]
    }

    // ── Optical flow ────────────────────────────────────────────────────────

    /**
     * Tracks a grid of points from [prev] to [cur] and returns x0, y0, y1 triples.
     *
     * Coarse-to-fine over scales 8, 2, 1. Depth frames are seconds apart, so the
     * total displacement can exceed a hundred pixels; starting at 1/8 scale turns
     * that into something a 21px window can follow, and each level only has to
     * resolve its own residual.
     */
    private fun trackFlow(
        prev: FloatArray, cur: FloatArray, w: Int, h: Int
    ): List<Triple<Int, Int, Int>> {
        val count = gridCols * gridRows
        // Full-resolution grid, computed once.
        val gx = FloatArray(count)
        val gy = FloatArray(count)
        var k = 0
        for (r in 0 until gridRows) {
            val fy = (0.15f + (0.95f - 0.15f) * r / (gridRows - 1f)) * h
            for (c in 0 until gridCols) {
                gx[k] = (0.12f + (0.88f - 0.12f) * c / (gridCols - 1f)) * w
                gy[k] = fy
                k++
            }
        }

        // Must be the PREVIOUS frame. Seeding this from `cur` leaves the tracker with
        // no temporal gradient at all and every point comes back unusable.
        var prevScaled = resize(prev, w, h, max(8, w / scales[0]), max(8, h / scales[0]))
        var prevSw = max(8, w / scales[0])
        var prevSh = max(8, h / scales[0])

        // Displacement carried between levels, always in FULL-resolution pixels.
        var carryX = FloatArray(count) { Float.NaN }
        var carryY = FloatArray(count) { Float.NaN }

        for (level in scales.indices) {
            val scale = scales[level]
            val sw = max(8, w / scale)
            val sh = max(8, h / scale)
            val curScaled = resize(cur, w, h, sw, sh)

            val px = FloatArray(count)
            val py = FloatArray(count)
            val seeded = BooleanArray(count)
            for (i in 0 until count) {
                // The seed is the position in the PREVIOUS frame, which is the
                // reference position plus whatever the coarser level already found.
                // Using the bare reference position here makes every level search for
                // the WHOLE displacement again, which a 13px window cannot do, and the
                // pyramid collapses to almost nothing.
                val carried = if (level > 0 && !carryX[i].isNaN()) carryX[i] else 0f
                val carriedY_ = if (level > 0 && !carryY[i].isNaN()) carryY[i] else 0f
                px[i] = (gx[i] + carried) / scale
                py[i] = (gy[i] + carriedY_) / scale
                seeded[i] = true
            }

            val window = windowFor(scale)
            val moved = lucasKanade(
                prevScaled, curScaled, prevSw, prevSh, sw, sh, px, py, seeded, window
            )

            // Full-res displacement from this level, used to seed the next finer one.
            val nextX = FloatArray(count)
            val nextY = FloatArray(count)
            for (i in 0 until count) {
                if (moved[i].isNaN() || moved[count + i].isNaN()) {
                    nextX[i] = Float.NaN
                    nextY[i] = Float.NaN
                } else {
                    val baseX = if (carryX[i].isNaN()) 0f else carryX[i]
                    val baseY = if (carryY[i].isNaN()) 0f else carryY[i]
                    nextX[i] = baseX + moved[i] * scale
                    nextY[i] = baseY + moved[count + i] * scale
                }
            }
            carryX = nextX
            carryY = nextY

            prevScaled = curScaled
            prevSw = sw
            prevSh = sh
        }

        val result = ArrayList<Triple<Int, Int, Int>>(count)
        for (i in 0 until count) {
            if (carryX[i].isNaN()) continue
            result.add(
                Triple(
                    gx[i].roundToInt(),
                    gy[i].roundToInt(),
                    (gy[i] + carryY[i]).roundToInt()
                )
            )
        }
        return result
    }

    /**
     * One refinement pass. Returns dx and dy for every point; NaN marks a failure.
     */
    private fun lucasKanade(
        prev: FloatArray, cur: FloatArray,
        pw: Int, ph: Int, cw: Int, ch: Int,
        px: FloatArray, py: FloatArray, seeded: BooleanArray,
        window: Int
    ): FloatArray {
        val n = px.size
        val out = FloatArray(n * 2)
        val half = window / 2
        var rejectedBorder = 0
        var rejectedTexture = 0
        var accepted = 0
        for (i in 0 until n) {
            if (!seeded[i]) {
                out[i] = Float.NaN
                out[n + i] = Float.NaN
                rejectedBorder++
                continue
            }
            val ix = px[i].roundToInt()
            val iy = py[i].roundToInt()
            if (ix - half < 1 || iy - half < 1 || ix + half >= pw || iy + half >= ph) {
                out[i] = Float.NaN
                out[n + i] = Float.NaN
                continue
            }
            var gxx = 0.0; var gxy = 0.0; var gyy = 0.0
            var bx = 0.0; var by = 0.0
            for (dy in -half until half) {
                val rowP = (iy + dy) * pw
                val rowC = (iy + dy) * cw
                for (dx in -half until half) {
                    val gx = ix + dx
                    val ixP = (prev[rowP + gx + 1] - prev[rowP + gx - 1]) * 0.5
                    val iyP = (prev[(iy + dy + 1) * pw + gx] - prev[(iy + dy - 1) * pw + gx]) * 0.5
                    // RHS of Ix*dx + Iy*dy = I_prev(x) - I(x). Written the other way
                    // round the whole estimate comes out negative and the [0, max_speed]
                    // clamp silently turns every estimate into zero.
                    val diff = prev[rowP + gx] - cur[rowC + gx]
                    gxx += ixP * ixP
                    gxy += ixP * iyP
                    gyy += iyP * iyP
                    bx += ixP * diff
                    by += iyP * diff
                }
            }
            val det = gxx * gyy - gxy * gxy
            val trace = gxx + gyy
            if (det <= 1e-9 || trace / det < MIN_TEXTURE) {
                out[i] = Float.NaN
                out[n + i] = Float.NaN
                rejectedTexture++
                continue
            }
            out[i] = ((gyy * bx - gxy * by) / det).toFloat()
            out[n + i] = ((gxx * by - gxy * bx) / det).toFloat()
            accepted++
        }
        this.rejectedBorder += rejectedBorder
        this.rejectedTexture += rejectedTexture
        this.acceptedPoints += accepted
        return out
    }

    /** Box-filter downscale. Averaging, not subsampling: subsampling throws away the
     *  high-frequency texture LK needs. */
    private fun resize(src: FloatArray, w: Int, h: Int, dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        val rx = w.toFloat() / dw
        val ry = h.toFloat() / dh
        for (y in 0 until dh) {
            val y0 = (y * ry).toInt()
            val y1 = min(h - 1, ((y + 1) * ry).roundToInt().coerceAtLeast(y0 + 1))
            for (x in 0 until dw) {
                val x0 = (x * rx).toInt()
                val x1 = min(w - 1, ((x + 1) * rx).roundToInt().coerceAtLeast(x0 + 1))
                var sum = 0f
                var count = 0
                for (yy in y0 until max(y0 + 1, y1)) {
                    val row = yy * w
                    for (xx in x0 until max(x0 + 1, x1)) {
                        sum += src[row + xx]
                        count++
                    }
                }
                out[y * dw + x] = if (count > 0) sum / count else 0f
            }
        }
        return out
    }

    /** Median depth over a small patch around a point, port of `sample_point`. */
    private fun sampleDepth(
        depth: FloatArray, dw: Int, dh: Int,
        x: Int, y: Int, w: Int, h: Int
    ): Float {
        val scaleX = dw.toFloat() / w
        val scaleY = dh.toFloat() / h
        val xi = (x * scaleX).roundToInt().coerceIn(0, dw - 1)
        val yi = (y * scaleY).roundToInt().coerceIn(0, dh - 1)
        val r = max(1, (5 * scaleX).roundToInt())
        val values = ArrayList<Float>(49)
        for (yy in max(0, yi - r)..min(dh - 1, yi + r)) {
            for (xx in max(0, xi - r)..min(dw - 1, xi + r)) {
                val v = depth[yy * dw + xx]
                if (v.isFinite() && v > 0f) values.add(v)
            }
        }
        if (values.isEmpty()) return -1f
        values.sort()
        return values[values.size / 2]
    }

    companion object {
        const val WINDOW = 21

        /**
         * Window per pyramid level.
         *
         * A 21px window at 1/8 scale covers almost the whole downscaled image, and the
         * box filter that produced that image has already averaged away the texture LK
         * needs. Levels get a smaller window, as pyramidal LK does.
         */
        fun windowFor(scale: Int): Int = when {
            scale >= 8 -> 9
            scale >= 2 -> 13
            else -> WINDOW
        }
        /** Reject points in textureless regions, where LK's normal matrix is singular.
         *  Expressed as trace/determinant so it is scale-free across image brightness. */
        const val MIN_TEXTURE = 1e-6
    }
}