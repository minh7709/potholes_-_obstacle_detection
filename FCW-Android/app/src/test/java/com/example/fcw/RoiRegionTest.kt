package com.example.fcw

import com.example.fcw.config.FcwConfig
import com.example.fcw.risk.RiskEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ROI and warning corridor measured against the visible region, not the whole buffer.
 *
 * Found from a device screenshot showing the ROAD ROI and CANH BAO boxes running off
 * the bottom of the screen. The HUD in that same screenshot gave the numbers: the
 * analysis buffer was 1280x960 and the visible band was y 162..798, so a third of the
 * frame was never displayed. ROI and corridor were computed on the full height, put
 * their bottom edge at y 931, and the overlay mapped that through the crop to a
 * screen position 450 px below the display.
 *
 * A rectangle only describes what the driver can see if its fractions are taken from
 * the same region the overlay draws through. These tests pin that invariant with the
 * values the device actually reported.
 */
class RoiRegionTest {

    private val cfg = FcwConfig()

    // Exactly what the HUD printed on the device: fmt=1, p0 row=5120, and
    // "khung 1280 x960 | thấy 0,162,1280,636".
    private val frameW = 1280
    private val frameH = 960
    private val crop = intArrayOf(0, 162, 1280, 636)

    @Test
    fun `a crop that fits the buffer is accepted`() {
        val region = RiskEngine.visibleRegion(crop, frameW, frameH)
        assertEquals(4, region.size)
        assertEquals(0, region[0])
        assertEquals(162, region[1])
        assertEquals(1280, region[2])
        assertEquals(636, region[3])
    }

    @Test
    fun `roi stays inside the visible band on a real device frame`() {
        val region = RiskEngine.visibleRegion(crop, frameW, frameH)
        val roi = RiskEngine.roadRoi(region, cfg)

        assertTrue(
            "ROI đỉnh ${roi[1]} phải >= vùng nhìn thấy ${region[1]}",
            roi[1] >= region[1]
        )
        assertTrue(
            "ROI đáy ${roi[3]} phải <= đáy vùng nhìn thấy ${region[1] + region[3]}, " +
                "nếu không khung vẽ tràn xuống dưới màn hình",
            roi[3] <= region[1] + region[3]
        )
    }

    @Test
    fun `corridor stays inside the visible band too`() {
        val region = RiskEngine.visibleRegion(crop, frameW, frameH)
        val corridor = RiskEngine.warningCorridor(region, cfg)

        assertTrue("đỉnh ${corridor[1]} phải >= ${region[1]}", corridor[1] >= region[1])
        assertTrue(
            "đáy ${corridor[3]} phải <= ${region[1] + region[3]}",
            corridor[3] <= region[1] + region[3]
        )
    }

    @Test
    fun `roi fractions apply to the visible region, not the buffer`() {
        val region = RiskEngine.visibleRegion(crop, frameW, frameH)
        val roi = RiskEngine.roadRoi(region, cfg)

        // 0.30 and 0.97 of the 636-row band that is actually on screen.
        assertEquals(162 + (636 * cfg.roiYMin).toInt(), roi[1])
        assertEquals(162 + (636 * cfg.roiYMax).toInt(), roi[3])
    }

    @Test
    fun `roi horizontal placement ignores the crop offset only when there is none`() {
        val region = RiskEngine.visibleRegion(crop, frameW, frameH)
        val roi = RiskEngine.roadRoi(region, cfg)

        // The device crop starts at x 0, so the fractions must still land on 20%..80%.
        assertEquals((1280 * cfg.roiXMin).toInt(), roi[0])
        assertEquals((1280 * cfg.roiXMax).toInt(), roi[2])
    }

    @Test
    fun `a crop offset on the left shifts the roi with it`() {
        val shifted = intArrayOf(400, 100, 440, 800)
        val region = RiskEngine.visibleRegion(shifted, frameW, frameH)
        val roi = RiskEngine.roadRoi(region, cfg)

        assertEquals(400 + (440 * cfg.roiXMin).toInt(), roi[0])
        assertEquals(100 + (800 * cfg.roiYMin).toInt(), roi[1])
    }

    @Test
    fun `a zero crop falls back to the whole buffer`() {
        val region = RiskEngine.visibleRegion(intArrayOf(0, 0, 0, 0), frameW, frameH)
        assertEquals(0, region[0])
        assertEquals(0, region[1])
        assertEquals(frameW, region[2])
        assertEquals(frameH, region[3])
    }

    @Test
    fun `a crop rect outside the buffer is rejected rather than trusted`() {
        // Some CameraX versions report the crop in the pre-crop original's coordinate
        // space, where the values can exceed the delivered buffer. Scaling by that
        // would stretch every box by an unknown factor.
        val beyondBuffer = intArrayOf(120, 200, 1600, 1200)
        val region = RiskEngine.visibleRegion(beyondBuffer, frameW, frameH)

        assertEquals(frameW, region[2])
        assertEquals(frameH, region[3])
    }

    @Test
    fun `a negative crop origin is rejected`() {
        val region = RiskEngine.visibleRegion(intArrayOf(-4, 162, 1280, 636), frameW, frameH)
        assertEquals(0, region[0])
    }

    @Test
    fun `no crop rect at all leaves the roi identical to the old whole-frame maths`() {
        val roi = RiskEngine.roadRoi(frameW, frameH, cfg)
        val viaRegion = RiskEngine.roadRoi(
            RiskEngine.visibleRegion(intArrayOf(0, 0, frameW, frameH), frameW, frameH), cfg
        )
        assertTrue(roi.contentEquals(viaRegion))
    }
}