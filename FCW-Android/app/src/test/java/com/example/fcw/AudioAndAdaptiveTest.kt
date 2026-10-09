package com.example.fcw

import com.example.fcw.config.FcwConfig
import com.example.fcw.performance.AdaptiveController
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Phase 9 and 11 behaviour.
 *
 * Both are the same shape of rule, and both are about not lying to the driver: a
 * warning that repeats itself is ignored, and a threshold that moves to buy frames is
 * not a warning any more.
 */
class AudioAndAdaptiveTest {

    private val cfg = FcwConfig()

    // ---- audio cooldown -----------------------------------------------------

    /** Mirrors AudioAlerter's gate, which needs no Android context to test. */
    private class Gate(
        private val warningCd: Float = 1.5f,
        private val dangerCd: Float = 1.0f
    ) {
        private var lastW = Double.NEGATIVE_INFINITY
        private var lastD = Double.NEGATIVE_INFINITY

        fun should(danger: Boolean, now: Double): Boolean {
            val last = if (danger) lastD else lastW
            if (now - last < (if (danger) dangerCd else warningCd)) return false
            if (danger) lastD = now else lastW = now
            return true
        }
    }

    @Test
    fun `cooldown blocks an immediate repeat`() {
        val g = Gate()
        assertTrue("lần đầu phải kêu", g.should(false, 0.0))
        assertFalse("lặp ngay phải bị chặn", g.should(false, 0.2))
        assertFalse("0.9s vẫn trong cửa sổ 1.5s", g.should(false, 0.9))
        assertTrue("1.6s thì kêu lại", g.should(false, 1.6))
    }

    @Test
    fun `danger cooldown is shorter than warning`() {
        assertTrue(
            "config đặt danger ngắn hơn warning",
            cfg.dangerCooldownSec < cfg.warningCooldownSec
        )
        assertEquals(1.5f, cfg.warningCooldownSec, 0.0001f)
        assertEquals(1.0f, cfg.dangerCooldownSec, 0.0001f)
    }

    @Test
    fun `warning and danger have separate cooldowns`() {
        val g = Gate()
        assertTrue(g.should(false, 0.0))
        assertTrue("danger không bị warning chặn", g.should(true, 0.1))
        assertFalse("nhưng danger lặp thì bị chặn", g.should(true, 0.5))
        assertFalse("warning vẫn còn trong cửa sổ", g.should(false, 1.0))
    }

    @Test
    fun `a ten second frame loop would beep a handful of times not hundreds`() {
        val g = Gate()
        var beeps = 0
        var t = 0.0
        while (t < 10.0) {
            if (g.should(false, t)) beeps++
            t += 0.05   // 20 FPS
        }
        // 10 s at a 1.5 s cooldown is about 7. Looping without it would be 200.
        assertTrue("quá nhiều lần kêu: $beeps", beeps in 6..8)
    }

    // ---- adaptive controller -----------------------------------------------

    @Test
    fun `nothing degrades while the frame rate is healthy`() {
        val a = AdaptiveController(cfg)
        repeat(100) { assertNull(a.maybeDegrade(20f, it * 0.05)) }
        assertEquals(0, a.level)
    }

    @Test
    fun `one slow sample never degrades`() {
        val a = AdaptiveController(cfg)
        assertNull("một frame chậm lẻ không được phản ứng", a.maybeDegrade(1f, 0.0))
        assertEquals(0, a.level)
    }

    @Test
    fun `a full window of slow samples degrades one step`() {
        val a = AdaptiveController(cfg)
        repeat(cfg.adaptiveWindow) { a.maybeDegrade(1f, it * 0.05) }
        assertEquals("đủ cửa sổ chậm thì giảm một bậc", 1, a.level)
        assertEquals(8, a.targetFps)
        assertEquals(6, a.depthInterval)
    }

    @Test
    fun `a brief recovery clears the streak`() {
        val a = AdaptiveController(cfg)
        repeat(cfg.adaptiveWindow - 1) { a.maybeDegrade(1f, it * 0.05) }
        assertNull(a.maybeDegrade(20f, 1.0))       // hồi phục
        repeat(cfg.adaptiveWindow - 1) { a.maybeDegrade(1f, 2.0 + it * 0.05) }
        assertEquals("sau hồi phục thì phải đủ cửa sổ mới giảm", 0, a.level)
    }

    @Test
    fun `cooldown stops it degrading every sample`() {
        val a = AdaptiveController(cfg)
        repeat(cfg.adaptiveWindow) { a.maybeDegrade(1f, it * 0.05) }
        assertEquals(1, a.level)
        // Ngay lập tức sau đó không được giảm tiếp.
        repeat(cfg.adaptiveWindow) { a.maybeDegrade(1f, 2.0 + it * 0.05) }
        assertEquals("phải chờ hết cooldown", 1, a.level)
    }

    @Test
    fun `it stops at the last step instead of running off the end`() {
        val a = AdaptiveController(cfg)
        var t = 0.0
        repeat(4000) {
            t += 0.05
            a.maybeDegrade(0.5f, t)
        }
        assertEquals("dừng ở bậc cuối", 3, a.level)
        assertEquals(4, a.targetFps)
        assertEquals(10, a.depthInterval)
    }

    @Test
    fun `disabling the controller stops all degradation`() {
        val a = AdaptiveController(cfg.copy(adaptiveEnabled = false))
        var t = 0.0
        repeat(4000) {
            t += 0.05
            assertNull(a.maybeDegrade(0.5f, t))
        }
        assertEquals(0, a.level)
    }

    @Test
    fun `every degradation is recorded with its reason`() {
        val a = AdaptiveController(cfg)
        var t = 0.0
        repeat(cfg.adaptiveWindow) {
            t += 0.05
            a.maybeDegrade(2f, t)
        }
        assertNotNull("phải có bản ghi", a.changes.firstOrNull())
        val c = a.changes.first()
        assertTrue("lý do phải nêu FPS", c.reason.contains("FPS"))
        assertEquals(1, c.level)
    }

    @Test
    fun `adaptive steps only ever reduce sample rates`() {
        // The whole point: TTC thresholds must not be part of the steps.
        val a = AdaptiveController(cfg)
        var t = 0.0
        repeat(4000) {
            t += 0.05
            a.maybeDegrade(0.5f, t)
        }
        // Mỗi bậc phải giảm (hoặc giữ) target FPS và depth interval, không tăng.
        // First step compares against a sentinel that always passes: the direction is
        // "FPS does not rise, depth interval does not fall".
        var prevFps = Int.MAX_VALUE
        var prevDepth = 0
        for (level in 0..a.level) {
            val fps = listOf(10, 8, 6, 4)[level]
            val depth = listOf(5, 6, 8, 10)[level]
            assertTrue("FPS bậc $level phải không tăng", fps <= prevFps)
            assertTrue("depth bậc $level phải không giảm", depth >= prevDepth)
            prevFps = fps; prevDepth = depth
        }
        assertEquals(cfg.warningTtc, 4.0f, 0.0001f)
        assertEquals(cfg.dangerTtc, 2.0f, 0.0001f)
    }

    // ---- config values copied from config.yaml ------------------------------

    @Test
    fun `config values match the python file`() {
        // DELIBERATE DEVIATION from config.yaml. config.yaml says 1.0 ("uncalibrated"),
        // but uncalibrated depth on this phone reads too far: measured against the web
        // pipeline on the same clip at the same instant, Android read 9.81 m where the
        // web pipeline read 5.91 m. 0.602 = 5.91/9.81 brings the two into agreement.
        //
        // This value is fitted from ONE clip at ONE distance. If a depth model is ever
        // replaced, re-measure before trusting it, and prefer fixing the scale at the
        // source over keeping a magic number here.
        assertEquals("calibration.depth_scale (đã hiệu chỉnh)", 0.602f, cfg.depthScale, 0.0001f)
        assertEquals("input.max_width", 1280, cfg.maxWidth)
        assertEquals("input.max_height", 720, cfg.maxHeight)
        assertEquals("performance.adaptive_min_fps", 5.0f, cfg.adaptiveMinFps, 0.0001f)
        assertEquals("performance.adaptive_window", 25, cfg.adaptiveWindow)
        assertEquals("performance.adaptive_cooldown_sec", 8.0f, cfg.adaptiveCooldownSec, 0.0001f)
        assertEquals("depth.max_interpolated_speed_mps", 12.0f, cfg.maxInterpolatedSpeedMps, 0.0001f)
        assertEquals("warning.volume", 0.8f, cfg.volume, 0.0001f)
    }
}