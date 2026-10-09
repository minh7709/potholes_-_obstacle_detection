package com.example.fcw

import com.example.fcw.detection.YoloPreprocessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Verifies the letterbox numerically on the JVM, without a camera.
 *
 * The report from the device was "YOLO 0 ms, Box 0", which means `detect()`
 * returned before running inference at all. The preprocessor is the first thing it
 * touches, so its arithmetic is checked here against values that can be worked out
 * by hand.
 */
class YoloPreprocessorTest {

    /** Deterministic source image: R=x, G=y, B=128, packed RGB. */
    private fun makeRgb(w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h * 3)
        var i = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                out[i++] = (x % 256).toByte()
                out[i++] = (y % 256).toByte()
                out[i++] = 128.toByte()
            }
        }
        return out
    }

    @Test
    fun `no padding when aspect matches`() {
        val pre = YoloPreprocessor(8)
        pre.setGeometryForTest(4, 4)
        // scale = min(8/4, 8/4) = 2 -> 8x8 fills the square, no pad.
        assertEquals(2f, pre.scale, 0.0001f)
        assertEquals(0, pre.padX)
        assertEquals(0, pre.padY)
    }

    @Test
    fun `vertical padding is applied for wide input`() {
        val pre = YoloPreprocessor(8)
        pre.setGeometryForTest(4, 2)
        // scale = min(8/4, 8/2) = 2 -> 8x4 content, padY = (8-4)/2 = 2
        assertEquals(2f, pre.scale, 0.0001f)
        assertEquals(0, pre.padX)
        assertEquals(2, pre.padY)
    }

    @Test
    fun `tensor carries the resized image in NCHW order`() {
        val size = 8
        val pre = YoloPreprocessor(size)
        val rgb = makeRgb(4, 4)
        pre.fill(rgb, 4, 4)
        val t = pre.inputTensor()
        val plane = size * size

        // scale 2: source (x,y) lands at (2x, 2y).
        for (y in 0 until 4) {
            for (x in 0 until 4) {
                val dx = x * 2
                val dy = y * 2
                val idx = dy * size + dx
                assertEquals("R tại ($dx,$dy)", x / 255f, t[0 * plane + idx], 0.002f)
                assertEquals("G tại ($dx,$dy)", y / 255f, t[1 * plane + idx], 0.002f)
                assertEquals("B tại ($dx,$dy)", 128 / 255f, t[2 * plane + idx], 0.002f)
            }
        }
    }

    @Test
    fun `letterbox border is 114 grey`() {
        val size = 8
        val pre = YoloPreprocessor(size)
        val rgb = makeRgb(4, 2)   // -> 8x4 content, padY = 2
        pre.fill(rgb, 4, 2)
        val t = pre.inputTensor()
        val plane = size * size
        val pad = 114f / 255f

        // Rows 0,1 and 6,7 are entirely padding.
        for (y in listOf(0, 1, 6, 7)) {
            for (x in 0 until size) {
                val idx = y * size + x
                assertEquals("R hàng $y", pad, t[0 * plane + idx], 0.002f)
                assertEquals("G hàng $y", pad, t[1 * plane + idx], 0.002f)
                assertEquals("B hàng $y", pad, t[2 * plane + idx], 0.002f)
            }
        }
        // Rows 2..5 hold the content.
        for (y in 2 until 6) {
            val idx = y * size + 3
            assertTrue(
                "hàng $y phải chứa ảnh, không phải padding",
                kotlin.math.abs(t[0 * plane + idx] - pad) > 0.01f
            )
        }
    }

    @Test
    fun `unletterbox inverts the transform exactly`() {
        val pre = YoloPreprocessor(640)
        pre.setGeometryForTest(1280, 960)
        // Device reported 1280x960: scale 0.5 -> 640x480, padY = 80.
        assertEquals(0.5f, pre.scale, 0.0001f)
        assertEquals(80, pre.padY)

        // A box at the very centre of the image must map to the frame centre.
        val cx = 320f
        val cy = 240f + 80f        // model space, inside the content
        val back = pre.toSourceSpace(cx - 40f, cy - 20f, cx + 40f, cy + 20f)
        assertEquals(560f, back[0], 0.01f)   // (320-40)/0.5
        assertEquals(440f, back[1], 0.01f)   // (240+80-20-80)/0.5 = 440
        assertEquals(720f, back[2], 0.01f)
        assertEquals(520f, back[3], 0.01f)   // (240+80+20-80)/0.5 = 520
    }

    @Test
    fun `bilinear averages a two by two block instead of picking one pixel`() {
        // Fine 1-pixel checkerboard scaled 4x4 -> 8x8. Every interior output pixel
        // averages a 2x2 source block holding two 0s and two 255s, so it must be the
        // midpoint. Nearest would return 0 or 255 and never 0.5.
        val src = 4
        val rgb = ByteArray(src * src * 3)
        for (y in 0 until src) {
            for (x in 0 until src) {
                val p = (y * src + x) * 3
                val v = if ((x + y) % 2 == 0) 0 else 255
                rgb[p] = v.toByte(); rgb[p + 1] = v.toByte(); rgb[p + 2] = v.toByte()
            }
        }
        val pre = YoloPreprocessor(8)
        pre.fill(rgb, src, src)
        val t = pre.inputTensor()

        // Bilinear weights are 0.75/0.25 here, not a plain 2x2 box, so the value is not
        // exactly the midpoint. The invariant that separates bilinear from nearest is
        // that interior pixels are strictly between the two extremes.
        // The first and last row/column clamp to a source edge, where the weights
        // collapse to one sample. Interior 1..6 is where interpolation must show.
        for (y in 1 until 7) {
            for (x in 1 until 7) {
                val v = t[y * 8 + x]
                assertTrue("($x,$y)=$v phải nằm giữa 0 và 1, không phải mẫu nguyên",
                    v > 0.05f && v < 0.95f)
            }
        }
        // Bilinear is area-preserving, so the mean over the interior stays at the
        // checkerboard mean. Nearest sampling would bias this.
        var sum = 0f
        var n = 0
        for (y in 1 until 7) for (x in 1 until 7) { sum += t[y * 8 + x]; n++ }
        assertEquals("trung bình phải giữ nguyên", 0.5f, sum / n, 0.05f)

        for (i in 0 until t.size) {
            assertTrue("tensor[$i]=${t[i]} ngoài [0,1]", t[i] in 0f..1f)
        }
    }

    @Test
    fun `bilinear keeps a constant image constant across the padding`() {
        // 8x4 into 8x8: s = 1, content 8x4, padY = 2, so this exercises both halves.
        val srcW = 8
        val srcH = 4
        val rgb = ByteArray(srcW * srcH * 3) { 200.toByte() }
        val pre = YoloPreprocessor(8)
        pre.fill(rgb, srcW, srcH)
        val t = pre.inputTensor()

        assertEquals(2, pre.padY)
        val pad = 114f / 255f
        val content = 200f / 255f
        for (y in 0 until 8) {
            for (x in 0 until 8) {
                val idx = y * 8 + x
                val expected = if (y < 2 || y >= 6) pad else content
                assertEquals("tensor[$idx]", expected, t[idx], 0.002f)
            }
        }
    }

    @Test
    fun `bilinear stays inside the byte range`() {
        // Extreme contrast across a block boundary is where naive interpolation
        // would produce values outside 0..1 and crash downstream normalisation.
        val src = 7
        val rgb = ByteArray(src * src * 3)
        for (y in 0 until src) for (x in 0 until src) {
            val p = (y * src + x) * 3
            val v = if (x == 0) 0 else 255
            rgb[p] = v.toByte(); rgb[p + 1] = v.toByte(); rgb[p + 2] = v.toByte()
        }
        val pre = YoloPreprocessor(16)
        pre.fill(rgb, src, src)
        val t = pre.inputTensor()
        for (i in 0 until t.size) assertTrue("tensor[$i]=${t[i]}", t[i] in 0f..1f)
    }

    @Test
    fun `crop changes geometry and detections stay in full frame coordinates`() {
        // Portrait visible region 720x1280 cropped to 4:3 gives 720x540.
        val srcW = 720
        val srcH = 1280
        val cropLeft = 0
        val cropTop = (1280 - 540) / 2
        val rgb = ByteArray(srcW * srcH * 3) { 180.toByte() }
        val pre = YoloPreprocessor(640)
        pre.fill(rgb, srcW, srcH, cropLeft, cropTop, 720, 540)

        assertEquals(cropTop, pre.cropTop)
        assertEquals(720, pre.cropWidth)
        assertEquals(540, pre.cropHeight)
        // scale = 640/720 = 0.888..., so the crop fills the width and pads vertically.
        assertEquals(640f / 720f, pre.scale, 0.0001f)
        assertEquals(0, pre.padX)
        assertEquals((640 - 480) / 2, pre.padY)

        // A box at the centre of the crop must map back into FULL frame coordinates,
        // which is where depth sampling and the ROI test expect it.
        val cx = 320f
        val cy = 240f + pre.padY.toFloat()
        val back = pre.toSourceSpace(cx - 40f, cy - 20f, cx + 40f, cy + 20f)
        assertEquals((cx - 40f) / pre.scale + cropLeft, back[0], 0.01f)
        assertEquals((cy - 20f - pre.padY) / pre.scale + cropTop, back[1], 0.01f)
        assertTrue("y phải nằm trong khung đầy đủ", back[1] >= 0f)
        assertTrue("y phải nằm trong khung đầy đủ", back[3] <= srcH.toFloat())
    }

    @Test
    fun `crop is clamped to the source bounds`() {
        val rgb = ByteArray(100 * 80 * 3) { 90.toByte() }
        val pre = YoloPreprocessor(64)
        pre.fill(rgb, 100, 80, 90, 70, 500, 500)
        assertEquals(90, pre.cropLeft)
        assertEquals(70, pre.cropTop)
        assertEquals(10, pre.cropWidth)
        assertEquals(10, pre.cropHeight)
    }

    @Test
    fun `full frame overload crops nothing`() {
        val rgb = ByteArray(64 * 32 * 3) { 120.toByte() }
        val pre = YoloPreprocessor(64)
        pre.fill(rgb, 64, 32)
        assertEquals(0, pre.cropLeft)
        assertEquals(0, pre.cropTop)
        assertEquals(64, pre.cropWidth)
        assertEquals(32, pre.cropHeight)
    }
}
