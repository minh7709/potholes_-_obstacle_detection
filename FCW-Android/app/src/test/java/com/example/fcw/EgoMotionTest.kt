package com.example.fcw

import com.example.fcw.tracking.EgoMotionEstimator
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

/**
 * Ego-motion estimator checked against synthetic frames with a known answer.
 *
 * The scene is generated analytically from the same pinhole model the estimator
 * assumes, so the expected speed is derivable rather than hand-tuned: shifting the
 * image of a static scene by `v*dt*f/Z` pixels reproduces `v` exactly.
 */
class EgoMotionTest {

    // Real analysis resolution. The pyramid window sizes only make sense at this size:
    // at 1/8 of 320x240 a 21px window covers the whole image and LK has nothing left.
    private val width = 1280
    private val height = 960

    /**
     * 60 ms, the spacing between cheap analyzer frames when inference runs 1-in-5.
     *
     * This is not arbitrary: optical flow is only trackable when the displacement per
     * frame is small. At 280 ms spacing the background moves hundreds of pixels and no
     * sparse tracker can follow it, which is a real limitation of running ego-motion
     * off the same cadence as depth.
     */
    private val dt = 0.06

    /**
     * Renders a static scene from depth, then warps it by the ego motion.
     *
     * Points below the centre move up when the camera advances; `rowOffset * v * dt / Z`
     * is the pixel shift the estimator inverts.
     */
    private fun render(sceneDepth: FloatArray, speed: Float): FloatArray {
        val out = FloatArray(width * height)
        val cy = height * 0.5f
        for (y in 0 until height) {
            for (x in 0 until width) {
                val z = sceneDepth[y * width + x]
                if (z <= 0f) continue
                val yr = y - cy
                // Camera moved forward: the projected point moved up by this much.
                val shift = yr * speed * dt.toFloat() / z
                val sy = (y - shift).toInt()
                if (sy < 0 || sy >= height) continue
                // A textured surface so LK has something to track.
                val u = x.toFloat() / width * 24f
                val v = sy.toFloat() / height * 18f
                out[y * width + x] =
                    sin(u) * 40f + sin(v * 1.7f) * 40f + sin(u * 2.3f + v) * 25f + 120f
            }
        }
        return out
    }

    private fun constantDepth(metres: Float): FloatArray =
        FloatArray(width * height) { metres }

    @Test
    fun `recovers a known ego speed from synthetic flow`() {
        val depth = constantDepth(30f)
        val target = 10f

        val est = EgoMotionEstimator()
        var t = 0.0
        // First call has no previous frame; it must return 0 rather than garbage.
        val first = est.update(render(depth, 0f), width, height, t, depth, width, height)
        assertTrue("frame đầu chưa có ảnh trước nên phải bằng 0, thực tế $first", first == 0f)

        var speed = 0f
        repeat(30) {
            t += dt
            speed = est.update(render(depth, target), width, height, t, depth, width, height)
        }

        // Acceleration clamp and smoothing mean the estimate approaches the target
        // rather than hitting it immediately; 25% is well inside that.
        val err = abs(speed - target) / target
        assertTrue(
            "tốc độ ước lượng $speed, mục tiêu $target (sai $err)",
            err < 0.25f
        )
    }

    @Test
    fun `near objects are ignored because they violate the static-scene assumption`() {
        // Everything closer than background_depth_m must be discarded: those are road
        // and moving vehicles, and they swamp the estimate.
        val near = constantDepth(5f)
        val est = EgoMotionEstimator(backgroundDepthM = 15f)
        var t = 0.0
        est.update(render(near, 0f), width, height, t, near, width, height)
        var speed = 0f
        repeat(10) {
            t += dt
            speed = est.update(render(near, 10f), width, height, t, near, width, height)
        }
        assertTrue("toàn khung hình ở 5m thì không được ra tốc độ, thực tế $speed", speed == 0f)
    }

    @Test
    fun `a textureless frame yields zero rather than a wild number`() {
        val flat = FloatArray(width * height) { 100f }
        val est = EgoMotionEstimator()
        var t = 0.0
        est.update(flat, width, height, t, constantDepth(30f), width, height)
        var speed = 0f
        repeat(8) {
            t += dt
            speed = est.update(flat, width, height, t, constantDepth(30f), width, height)
        }
        assertTrue("ảnh phẳng không có kết cấu thì phải bằng 0, thực tế $speed", speed == 0f)
    }

    @Test
    fun `speed never goes negative because reverse flow is not ego speed`() {
        val depth = constantDepth(30f)
        val est = EgoMotionEstimator()
        var t = 0.0
        est.update(render(depth, 0f), width, height, t, depth, width, height)
        var speed = 0f
        repeat(20) {
            t += dt
            // Shift the other way: looks like the camera moves backwards.
            speed = est.update(render(depth, -10f), width, height, t, depth, width, height)
        }
        assertTrue("vận tốc phải >= 0, thực tế $speed", speed >= 0f)
    }

    @Test
    fun `dt too small is rejected rather than dividing by nearly zero`() {
        val depth = constantDepth(30f)
        val est = EgoMotionEstimator()
        val a = render(depth, 0f)
        val b = render(depth, 10f)
        est.update(a, width, height, 1.0, depth, width, height)
        val speed = est.update(b, width, height, 1.0005, depth, width, height)
        assertTrue("dt gần 0 phải trả 0, thực tế $speed", speed == 0f)
    }

    @Test
    fun `acceleration clamp stops an implausible jump`() {
        val depth = constantDepth(30f)
        val est = EgoMotionEstimator(maxAccelMps = 8f)
        var t = 0.0
        est.update(render(depth, 0f), width, height, t, depth, width, height)
        // Settle at a low speed first: with no previous estimate there is nothing for the
        // acceleration clamp to act on, which is why the jump has to be measured from an
        // established value rather than from zero.
        repeat(25) {
            t += dt
            est.update(render(depth, 2f), width, height, t, depth, width, height)
        }
        val before = est.speedMps
        t += dt
        val after = est.update(render(depth, 30f), width, height, t, depth, width, height)
        val jump = after - before
        assertTrue(
            "nhảy $jump m/s trong ${dt}s vượt giới hạn gia tốc 8 m/s²",
            jump <= 8f * dt + 0.5f
        )
    }

    @Test
    fun `quality reports how many points were usable`() {
        // Far in the top half and the bottom half, near in the middle rows, so points
        // both above and below the horizon survive the gates.
        val far = FloatArray(width * height) { i ->
            val row = i / width
            if (row < height * 0.30f || row > height * 0.75f) 30f else 3f
        }
        val est = EgoMotionEstimator(backgroundDepthM = 15f)
        var t = 0.0
        est.update(render(far, 0f), width, height, t, far, width, height)
        t += dt
        est.update(render(far, 10f), width, height, t, far, width, height)
        assertTrue("chất lượng phải nằm trong [0,1], thực tế ${est.quality}",
            est.quality in 0f..1f)
        assertTrue("một nửa khung xa thì quality phải > 0", est.quality > 0f)
    }

    @Test
    fun `reset clears the estimate`() {
        val depth = constantDepth(30f)
        val est = EgoMotionEstimator()
        var t = 0.0
        est.update(render(depth, 0f), width, height, t, depth, width, height)
        repeat(20) {
            t += dt
            est.update(render(depth, 10f), width, height, t, depth, width, height)
        }
        est.reset()
        assertTrue("sau reset phải về 0", est.speedMps == 0f)
        assertTrue("frame tiếp theo sau reset không có ảnh trước", est.update(
            render(depth, 10f), width, height, t + dt, depth, width, height
        ) == 0f)
    }
}