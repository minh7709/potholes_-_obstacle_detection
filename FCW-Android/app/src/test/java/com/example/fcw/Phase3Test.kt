package com.example.fcw

import com.example.fcw.config.FcwConfig
import com.example.fcw.detection.Detection
import com.example.fcw.risk.LevelStreak
import com.example.fcw.risk.RiskEngine
import com.example.fcw.tracking.ByteTracker
import com.example.fcw.tracking.TrackHistory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Phase 3 behaviour, checked against the Python semantics it was ported from.
 *
 * The distance tests feed synthetic history directly. Depth does not exist yet on
 * Android, so this is the only place the kinematics can be verified at all; the
 * values chosen below are hand-computed so a regression in the fit is visible as a
 * wrong number rather than as a vague "seems fine".
 */
class Phase3Test {

    private val cfg = FcwConfig()

    private fun det(
        id: Int?,
        cls: String = "motorbike",
        conf: Float = 0.8f,
        x1: Float, y1: Float, x2: Float, y2: Float
    ) = Detection(id, 6, cls, conf, x1, y1, x2, y2, (x2 - x1) * (y2 - y1))

    private fun box(cx: Float, cy: Float, w: Float = 100f, h: Float = 80f) =
        det(null, x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2)

    // ---- level classification ---------------------------------------------

    @Test
    fun `ttc thresholds match config and treat infinity as normal`() {
        assertEquals(RiskEngine.Level.DANGER, RiskEngine.classifyTtc(1.9f, cfg))
        assertEquals(RiskEngine.Level.DANGER, RiskEngine.classifyTtc(2.0f, cfg))
        assertEquals(RiskEngine.Level.WARNING, RiskEngine.classifyTtc(2.1f, cfg))
        assertEquals(RiskEngine.Level.WARNING, RiskEngine.classifyTtc(4.0f, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyTtc(4.1f, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyTtc(Float.MAX_VALUE, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyTtc(Float.NaN, cfg))
    }

    @Test
    fun `distance thresholds match config`() {
        assertEquals(RiskEngine.Level.DANGER, RiskEngine.classifyDistance(3.5f, cfg))
        assertEquals(RiskEngine.Level.WARNING, RiskEngine.classifyDistance(3.6f, cfg))
        assertEquals(RiskEngine.Level.WARNING, RiskEngine.classifyDistance(8.0f, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyDistance(8.1f, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyDistance(0f, cfg))
        assertEquals(RiskEngine.Level.NORMAL, RiskEngine.classifyDistance(null, cfg))
    }

    @Test
    fun `worse level keeps the most severe`() {
        assertEquals(
            RiskEngine.Level.DANGER,
            RiskEngine.worseLevel(RiskEngine.Level.NORMAL, RiskEngine.Level.DANGER)
        )
        assertEquals(
            RiskEngine.Level.WARNING,
            RiskEngine.worseLevel(RiskEngine.Level.WARNING, RiskEngine.Level.NORMAL)
        )
    }

    // ---- ROI and corridor --------------------------------------------------

    @Test
    fun `roi and corridor match the fractions in config yaml`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        assertEquals("x_min 0.10", 100, roi[0])
        assertEquals("y_min 0.22", 176, roi[1])
        assertEquals("x_max 0.90", 900, roi[2])
        assertEquals("y_max 1.00", 800, roi[3])

        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)
        assertEquals(220, corridor[0])
        assertEquals(208, corridor[1])
        assertEquals(780, corridor[2])
        assertEquals(800, corridor[3])
    }

    @Test
    fun `centre score falls off towards the edges`() {
        val centre = RiskEngine.centreScore(500f, 1000, cfg)
        val edge = RiskEngine.centreScore(900f, 1000, cfg)
        val outside = RiskEngine.centreScore(1000f, 1000, cfg)
        assertEquals(1f, centre, 0.001f)
        assertTrue("$edge phải nhỏ hơn $centre", edge < centre)
        assertEquals(0f, outside, 0.001f)
        assertTrue("không bao giờ âm", outside >= 0f)
    }

    @Test
    fun `object beside the vehicle cannot escalate however close it is`() {
        // Close range alone would be DANGER, but it sits outside the corridor.
        val h = TrackHistory(1)
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)

        val beside = RiskEngine.assess(
            h, cx = 900f, cy = 700f, frameW = 1000, frameH = 800,
            roi = roi, corridor = corridor, cfg = cfg
        )
        assertFalse("ngoài hành lang phải NORMAL", RiskEngine.Level.DANGER == beside.level)
        assertEquals(RiskEngine.Level.NORMAL, beside.level)
        assertFalse(beside.inWarningCorridor)
    }

    @Test
    fun `in corridor centre is accepted`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)
        val a = RiskEngine.assess(
            TrackHistory(1), cx = 500f, cy = 700f,
            frameW = 1000, frameH = 800, roi = roi, corridor = corridor, cfg = cfg
        )
        assertTrue(a.inWarningCorridor)
        assertTrue(a.inRoi)
    }

    // ---- kinematics --------------------------------------------------------

    /** Feeds a constant closing rate and returns the history. */
    private fun closingHistory(
        startDistance: Float,
        speedMps: Float,
        frames: Int,
        dt: Double = 0.1
    ): TrackHistory {
        val h = TrackHistory(1)
        var t = 0.0
        var d = startDistance
        repeat(frames) {
            h.update(t, d, null)
            h.estimateSpeed()
            t += dt
            d -= speedMps * dt.toFloat()
        }
        return h
    }

    @Test
    fun `closing target gets a ttc close to distance over speed`() {
        // 12 steps of 0.1 s at 5 m/s from 20 m: the gap ends near 14 m, so TTC
        // is about 2.8 s. Asserted against distance/speed, not a magic number.
        val h = closingHistory(20f, 5f, frames = 12)
        assertTrue("speed phải hợp lệ", h.speedValid)
        assertEquals(5f, h.closingSpeed, 0.3f)
        val (ttc, valid) = h.timeToCollision()
        assertTrue("TTC phải hữu hạn, thực tế $ttc", valid)
        val distanceNow = h.fitDistance ?: h.distance!!
        assertEquals("TTC phải bằng khoảng cách chia tốc độ",
            distanceNow / h.closingSpeed, ttc, 0.05f)
        assertTrue("TTC $ttc phải gần 2.8s", ttc in 2.4f..3.2f)
    }

    @Test
    fun `stationary target has infinite ttc not zero`() {
        // The distinction matters: infinite means known-safe, zero would mean a crash.
        val h = closingHistory(7f, 0f, frames = 12)
        val (ttc, valid) = h.timeToCollision()
        assertFalse("không tính được tốc độ thì ttc không hợp lệ", valid)
        assertEquals(Float.MAX_VALUE, ttc, 0f)
        // Range alone must still raise the warning.
        assertEquals(RiskEngine.Level.WARNING, RiskEngine.classifyDistance(h.distance, cfg))
    }

    @Test
    fun `receding target is not treated as closing`() {
        val h = TrackHistory(2)
        var t = 0.0
        var d = 5f
        repeat(12) {
            h.update(t, d, null)
            h.estimateSpeed()
            t += 0.1
            d += 2f
        }
        assertTrue("xa dần thì tốc độ âm", h.closingSpeed < 0f)
        assertFalse(h.timeToCollision().second)
    }

    @Test
    fun `too few samples never yields a speed`() {
        val h = TrackHistory(1)
        h.update(0.0, 10f, null)
        h.update(0.1, 9f, null)
        h.estimateSpeed()
        assertFalse("2 mẫu là chưa đủ", h.speedValid)
        assertEquals(0f, h.closingSpeed, 0f)
    }

    @Test
    fun `a short time window never yields a speed`() {
        // 6 samples inside 0.05 s: the count passes but the span does not.
        val h = TrackHistory(1)
        repeat(6) { h.update(it * 0.01, 10f - it * 0.05f, null) }
        h.estimateSpeed()
        assertFalse("cửa sổ thời gian quá ngắn", h.speedValid)
    }

    @Test
    fun `no depth yet means no distance and no false alarm`() {
        val h = TrackHistory(7)
        repeat(30) { h.update(it * 0.1, null, box(500f, 700f)) }
        h.estimateSpeed()
        assertNull("chưa có depth thì không có khoảng cách", h.distance)
        assertEquals("none", h.distanceSource)
        assertFalse(h.timeToCollision().second)
        assertEquals(
            RiskEngine.Level.NORMAL,
            RiskEngine.classifyDistance(h.distance, cfg)
        )
    }

    @Test
    fun `interpolated distance is never reported as measured`() {
        val h = TrackHistory(1)
        h.update(0.0, 20f, null)
        h.update(1.0, 18f, null)
        val (v, source) = h.interpolatedDistance(2.0, maxRateMps = 10f)
        assertNotNull(v)
        assertEquals("interpolated", source)
        assertEquals(16f, v!!, 0.01f)
    }

    @Test
    fun `absurd implied rate falls back to holding the last value`() {
        val h = TrackHistory(1)
        h.update(0.0, 20f, null)
        h.update(5.0, 1f, null)          // 19 m trong 5 s
        val (v, source) = h.interpolatedDistance(6.0, maxRateMps = 10f)
        assertEquals(1f, v!!, 0.01f)
        assertEquals("held", source)
    }

    // ---- confirmation ------------------------------------------------------

    @Test
    fun `streak needs consecutive frames at the same level`() {
        val s = LevelStreak()
        assertEquals(1, s.advance(1, RiskEngine.Level.WARNING))
        assertEquals(2, s.advance(1, RiskEngine.Level.WARNING))
        assertEquals(1, s.advance(1, RiskEngine.Level.DANGER))   // mức đổi -> reset
        assertEquals(2, s.advance(1, RiskEngine.Level.DANGER))
        assertEquals(3, s.advance(1, RiskEngine.Level.DANGER))
    }

    @Test
    fun `streak is dropped when a track disappears`() {
        val s = LevelStreak()
        repeat(5) { s.advance(1, RiskEngine.Level.DANGER) }
        s.retainOnly(setOf(2))
        assertEquals("id biến mất thì không được cộng dồn", 0, s.get(1))
    }

    @Test
    fun `warning needs both enough hits and a long enough streak`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)

        val fresh = TrackHistory(1)
        fresh.update(0.0, 12f, box(500f, 700f))
        fresh.update(0.1, 11f, box(500f, 700f))
        val early = RiskEngine.assess(
            fresh, 500f, 700f, 1000, 800, roi, corridor, cfg, consecutive = 5
        )
        assertFalse("chưa đủ framesSeen thì chưa được xác nhận", early.confirmed)

        val matured = closingHistory(12f, 5f, frames = 10)
        val ok = RiskEngine.assess(
            matured, 500f, 700f, 1000, 800, roi, corridor, cfg, consecutive = 3
        )
        assertTrue("đủ điều kiện thì phải xác nhận", ok.confirmed)
    }

    // ---- primary target selection -----------------------------------------

    @Test
    fun `only primary classes can be announced`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)
        val cone = TrackHistory(2, "reflective_cone")
        repeat(10) { cone.update(it * 0.1, 1f, det(null, "reflective_cone", 0.9f, 460f, 660f, 540f, 740f)) }
        cone.estimateSpeed()
        val a = RiskEngine.assess(cone, 500f, 700f, 1000, 800, roi, corridor, cfg, 5)
        assertEquals(RiskEngine.Level.DANGER, a.levelUnrestricted)
        assertNull("nón cỏ không được phát cảnh báo", RiskEngine.selectPrimaryTarget(listOf(a), cfg))
    }

    @Test
    fun `motorbike alias resolves so the primary filter fires`() {
        // config says "motorbike"; the model says "motorbike". Both normalise equal.
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)
        val bike = closingHistory(12f, 6f, frames = 10)
        bike.update(1.0, 12f, det(null, "motorbike", 0.9f, 460f, 660f, 540f, 740f))
        val a = RiskEngine.assess(bike, 500f, 700f, 1000, 800, roi, corridor, cfg, 5)
        assertNotNull("xe máy phải được chọn", RiskEngine.selectPrimaryTarget(listOf(a), cfg))
    }

    @Test
    fun `danger outranks warning`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)

        val danger = closingHistory(3f, 2f, frames = 10)
        danger.update(1.0, 3f, det(null, "motorbike", 0.9f, 460f, 660f, 540f, 740f))
        val warn = closingHistory(7f, 1f, frames = 10)
        warn.update(1.0, 7f, det(null, "motorbike", 0.9f, 300f, 500f, 380f, 580f))

        val aDanger = RiskEngine.assess(danger, 500f, 700f, 1000, 800, roi, corridor, cfg, 5)
        val aWarn = RiskEngine.assess(warn, 340f, 540f, 1000, 800, roi, corridor, cfg, 5)
        val picked = RiskEngine.selectPrimaryTarget(listOf(aWarn, aDanger), cfg)
        assertNotNull(picked)
        assertEquals(RiskEngine.Level.DANGER, picked!!.level)
    }

    @Test
    fun `normal objects are never announced`() {
        val roi = RiskEngine.roadRoi(1000, 800, cfg)
        val corridor = RiskEngine.warningCorridor(1000, 800, cfg)
        val far = closingHistory(60f, 1f, frames = 10)
        far.update(1.0, 60f, det(null, "motorbike", 0.9f, 460f, 660f, 540f, 740f))
        val a = RiskEngine.assess(far, 500f, 700f, 1000, 800, roi, corridor, cfg, 5)
        assertEquals(RiskEngine.Level.NORMAL, a.level)
        assertNull(RiskEngine.selectPrimaryTarget(listOf(a), cfg))
    }

    // ---- tracking ----------------------------------------------------------

    @Test
    fun `a stable detection keeps one id`() {
        val t = ByteTracker(cfg)
        val ids = (1..6).map { t.update(listOf(box(500f + it, 700f)), it * 0.1) }
        val all = ids.flatten()
        assertEquals(6, all.size)
        val unique = all.map { it.trackId }.toSet()
        assertEquals("cùng một vật thể phải giữ nguyên id", 1, unique.size)
    }

    @Test
    fun `a low score detection keeps the id alive`() {
        val t = ByteTracker(cfg, newTrackThresh = 0.6f)
        t.update(listOf(box(500f, 700f).let { it }), 0.1)
        // Below trackHighThresh but above newTrackThresh: matched, not restarted.
        val weak = box(502f, 701f).copy(confidence = 0.3f)
        val out = t.update(listOf(weak), 0.2)
        assertEquals(1, out.size)
        assertEquals("id phải được giữ", 1, out[0].trackId)
    }

    @Test
    fun `a brief gap does not create a new id`() {
        val t = ByteTracker(cfg)
        val first = t.update(listOf(box(500f, 700f)), 0.1).single().trackId!!
        t.update(emptyList(), 0.2)
        t.update(emptyList(), 0.3)
        val back = t.update(listOf(box(503f, 701f)), 0.4).single().trackId
        assertEquals("mất 2 frame vẫn phải là cùng id", first, back)
    }

    @Test
    fun `two separate objects get two ids`() {
        val t = ByteTracker(cfg)
        val out = t.update(
            listOf(box(300f, 700f), box(700f, 700f)),
            0.1
        )
        assertEquals(2, out.size)
        assertEquals(2, out.mapNotNull { it.trackId }.toSet().size)
    }

    @Test
    fun `a track that ages out is forgotten`() {
        val t = ByteTracker(cfg, trackLowThresh = 0.1f)
        t.update(listOf(box(500f, 700f)), 0.1)
        repeat(cfg.buffer + 5) { t.update(emptyList(), 0.2 + it * 0.1) }
        assertTrue("vượt buffer thì phải quên", t.history.isEmpty())
    }

    @Test
    fun `history is created per live track and reports frames seen`() {
        val t = ByteTracker(cfg)
        repeat(5) { t.update(listOf(box(500f, 700f)), 0.1 * (it + 1)) }
        val h = t.history.values.first()
        assertEquals(5, h.framesSeen)
        assertEquals("motorbike", h.className)
    }

    @Test
    fun `frames with no detection are not drawn`() {
        val t = ByteTracker(cfg)
        t.update(listOf(box(500f, 700f)), 0.1)
        assertTrue("frame trống không được vẽ box", t.update(emptyList(), 0.2).isEmpty())
    }

    @Test
    fun `a detection above the detector threshold always becomes a track`() {
        // Regression: newTrackThresh was 0.6 while the detector emits from 0.45, so
        // everything in [0.45, 0.6) was discarded and the screen showed zero boxes.
        val t = ByteTracker(cfg)
        for (conf in listOf(0.45f, 0.5f, 0.55f, 0.6f, 0.9f)) {
            val out = t.update(listOf(box(500f, 700f).copy(confidence = conf)), 0.1)
            assertTrue("conf $conf phải tạo được track", out.isNotEmpty())
            assertEquals("conf $conf phải giữ đúng 1 box", 1, out.size)
        }
    }

    @Test
    fun `tracker never drops a detection the model returned`() {
        // Sweep confidences across the whole band the detector emits.
        val t = ByteTracker(cfg)
        for (conf in listOf(0.45f, 0.47f, 0.52f, 0.58f, 0.63f, 0.75f, 0.95f)) {
            t.reset()
            val raw = listOf(
                box(300f, 700f).copy(confidence = conf),
                box(700f, 700f).copy(confidence = conf)
            )
            val out = t.update(raw, 0.1)
            assertEquals("conf $conf: không được mất box nào", raw.size, out.size)
        }
    }

    @Test
    fun `detections below the detector threshold are not expected to appear`() {
        // Confirms the band boundary matches config rather than a hard-coded 0.6.
        val t = ByteTracker(cfg)
        assertTrue(
            "0.44 thấp hơn ngưỡng detector nên không tạo track",
            t.update(listOf(box(500f, 700f).copy(confidence = 0.44f)), 0.1).isEmpty()
        )
        assertEquals(0.45f, cfg.confThreshold, 0.0001f)
    }
}
