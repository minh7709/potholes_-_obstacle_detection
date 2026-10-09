package com.example.fcw

import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.example.fcw.utils.ImageUtils
import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the frame-format conversion, which is where the pipeline stalled on device.
 *
 * `rgbaToRgb` used to return false for any format other than FLEX_RGBA_8888, so a
 * device that delivered YUV_420_888 produced zero detections forever while the HUD
 * showed only "Box 0". A fake [ImageProxy] reproduces both formats on the JVM, so
 * this no longer needs a phone to verify.
 */
class ImageUtilsTest {

    private class FakePlane(
        data: ByteArray,
        rowStride: Int,
        pixelStride: Int,
        offset: Int = 0
    ) : ImageProxy.PlaneProxy {
        // position() is deliberately non-zero on some cases: reading from index 0
        // instead of the buffer position yields a shifted image on real devices.
        private val buf: ByteBuffer =
            ByteBuffer.wrap(data).also { it.position(offset.coerceAtMost(data.size)) }

        // Stored under different names on purpose: PlaneProxy is a val property in
        // Kotlin, so `= pixelStride` here would resolve back to this getter.
        private val rStride = rowStride
        private val pStride = pixelStride

        override fun getRowStride(): Int = rStride
        override fun getPixelStride(): Int = pStride
        override fun getBuffer() = buf
    }

    private class FakeImage(
        private val w: Int,
        private val h: Int,
        private val fmt: Int,
        private val planes: Array<ImageProxy.PlaneProxy>
    ) : ImageProxy {
        override fun close() {}
        // Deliberately unusable: toRgb() must not depend on crop rect, image or
        // metadata, and a loud failure here proves it if that ever changes.
        override fun getCropRect(): android.graphics.Rect = error("toRgb không được dùng cropRect")
        override fun setCropRect(rect: android.graphics.Rect?) {
            throw UnsupportedOperationException("không được dùng setCropRect")
        }
        override fun getFormat() = fmt
        override fun getHeight() = h
        override fun getWidth() = w
        override fun getPlanes() = planes
        override fun getImageInfo(): ImageInfo = error("không được dùng metadata")
        override fun getImage(): android.media.Image = error("không được dùng getImage")
    }

    private fun rgb(w: Int, h: Int) = ByteArray(ImageUtils.rgbBufferSize(w, h))

    // ---- RGBA ----------------------------------------------------------------

    @Test
    fun `tight rgba drops the alpha byte and keeps rgb order`() {
        val w = 3
        val h = 2
        val raw = ByteArray(w * h * 4)
        // R,G,B,A for each pixel, in order.
        for (i in 0 until w * h) {
            raw[i * 4] = (i * 10).toByte()
            raw[i * 4 + 1] = (i * 10 + 1).toByte()
            raw[i * 4 + 2] = (i * 10 + 2).toByte()
            raw[i * 4 + 3] = 0xFF.toByte()
        }
        val image = FakeImage(w, h, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = w * 4, pixelStride = 4)
        ))
        val out = rgb(w, h)

        val res = ImageUtils.toRgb(image, out)

        assertTrue("phải thành công, thực tế $res", res is ImageUtils.Result.Ok)
        val expected = ByteArray(w * h * 3)
        for (i in 0 until w * h) {
            expected[i * 3] = (i * 10).toByte()
            expected[i * 3 + 1] = (i * 10 + 1).toByte()
            expected[i * 3 + 2] = (i * 10 + 2).toByte()
        }
        assertArrayEquals(expected, out)
    }

    @Test
    fun `rgba with padded rows is not sheared`() {
        val w = 3
        val h = 2
        val rowStride = w * 4 + 8
        val raw = ByteArray(rowStride * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val p = y * rowStride + x * 4
                raw[p] = (y * 100 + x).toByte()
                raw[p + 1] = (y * 100 + x + 1).toByte()
                raw[p + 2] = (y * 100 + x + 2).toByte()
            }
        }
        val image = FakeImage(w, h, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = rowStride, pixelStride = 4)
        ))
        val out = rgb(w, h)

        assertTrue(ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok)

        // Every row must read its own pixels; a sheared read picks up padding.
        for (y in 0 until h) {
            for (x in 0 until w) {
                val di = (y * w + x) * 3
                assertEquals("pixel ($x,$y) R", (y * 100 + x).toByte(), out[di])
                assertEquals("pixel ($x,$y) G", (y * 100 + x + 1).toByte(), out[di + 1])
                assertEquals("pixel ($x,$y) B", (y * 100 + x + 2).toByte(), out[di + 2])
            }
        }
    }

    @Test
    fun `rgba reads from the buffer position not from index zero`() {
        val w = 2
        val h = 1
        val raw = ByteArray(4 + w * h * 4)
        raw[0] = 99 // junk before the real data
        for (i in 0 until w) {
            raw[4 + i * 4] = (10 + i).toByte()
            raw[4 + i * 4 + 1] = (20 + i).toByte()
            raw[4 + i * 4 + 2] = (30 + i).toByte()
        }
        val image = FakeImage(w, h, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = w * 4, pixelStride = 4, offset = 4)
        ))
        val out = rgb(w, h)

        assertTrue(ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok)
        assertArrayEquals(
            byteArrayOf(10, 20, 30, 11, 21, 31),
            out
        )
    }

    // ---- YUV -----------------------------------------------------------------

    /** Builds YUV_420_888 where every pixel has the requested colour. */
    private fun yuvImage(w: Int, h: Int, yVal: Int, uVal: Int, vVal: Int): FakeImage {
        val ySize = w * h
        val cSize = (w / 2) * (h / 2)
        return FakeImage(w, h, ImageUtils.YUV_420_888, arrayOf(
            FakePlane(ByteArray(ySize) { yVal.toByte() }, rowStride = w, pixelStride = 1),
            FakePlane(ByteArray(cSize) { uVal.toByte() }, rowStride = w / 2, pixelStride = 1),
            FakePlane(ByteArray(cSize) { vVal.toByte() }, rowStride = w / 2, pixelStride = 1)
        ))
    }

    @Test
    fun `yuv is converted instead of rejected`() {
        val w = 4
        val h = 4
        val image = yuvImage(w, h, yVal = 128, uVal = 128, vVal = 128)
        val out = rgb(w, h)

        val res = ImageUtils.toRgb(image, out)

        assertTrue("YUV phải được chuyển, không từ chối: $res", res is ImageUtils.Result.Ok)
        // Neutral chroma means R=G=B=Y.
        for (i in 0 until w * h) {
            assertEquals("pixel $i R", 128.toByte(), out[i * 3])
            assertEquals("pixel $i G", 128.toByte(), out[i * 3 + 1])
            assertEquals("pixel $i B", 128.toByte(), out[i * 3 + 2])
        }
    }

    @Test
    fun `yuv extremes are clamped to the byte range`() {
        val w = 2
        val h = 2
        // Y at max with extreme chroma would overflow without clamping.
        val image = yuvImage(w, h, yVal = 255, uVal = 0, vVal = 255)
        val out = rgb(w, h)

        assertTrue(ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok)
        // Y=255, U=0, V=255: R overflows and must saturate at 255 (stored as -1).
        // dv=127, du=-128
        //   r = 255 + 1.402*127 = 433     -> 255
        //   g = 255 - 0.344*-128 - 0.714*127 = 208 -> 208
        //   b = 255 + 1.772*-128 = 28     -> 28
        for (i in 0 until w * h) {
            assertEquals("R phải bão hoà ở 255", 255, out[i * 3].toInt() and 0xFF)
            assertEquals("G", 208, out[i * 3 + 1].toInt() and 0xFF)
            assertEquals("B", 28, out[i * 3 + 2].toInt() and 0xFF)
        }
    }

    @Test
    fun `yuv subsampled chroma is shared by a two by two block`() {
        val w = 4
        val h = 2
        val cW = w / 2
        val cH = h / 2
        // Neutral chroma is 128, not 0. With it the conversion is exactly Y.
        val yPlane = ByteArray(w * h) { 100.toByte() }
        val uPlane = ByteArray(cW * cH) { 128.toByte() }
        val vPlane = ByteArray(cW * cH) { 128.toByte() }
        val image = FakeImage(w, h, ImageUtils.YUV_420_888, arrayOf(
            FakePlane(yPlane, rowStride = w, pixelStride = 1),
            FakePlane(uPlane, rowStride = cW, pixelStride = 1),
            FakePlane(vPlane, rowStride = cW, pixelStride = 1)
        ))
        val out = rgb(w, h)

        assertTrue(ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok)
        for (i in 0 until w * h) {
            assertEquals("R tại pixel $i", 100.toByte(), out[i * 3])
            assertEquals("G tại pixel $i", 100.toByte(), out[i * 3 + 1])
            assertEquals("B tại pixel $i", 100.toByte(), out[i * 3 + 2])
        }
    }

    @Test
    fun `odd yuv height still reads chroma without overflowing`() {
        val w = 4
        val h = 3
        val cW = (w + 1) / 2
        val cH = (h + 1) / 2
        val image = FakeImage(w, h, ImageUtils.YUV_420_888, arrayOf(
            FakePlane(ByteArray(w * h) { 80.toByte() }, rowStride = w, pixelStride = 1),
            FakePlane(ByteArray(cW * cH) { 128.toByte() }, rowStride = cW, pixelStride = 1),
            FakePlane(ByteArray(cW * cH) { 128.toByte() }, rowStride = cW, pixelStride = 1)
        ))
        val out = rgb(w, h)

        val res = ImageUtils.toRgb(image, out)
        assertTrue("kích thước lẻ phải chạy được: $res", res is ImageUtils.Result.Ok)
    }

    // ---- failures ------------------------------------------------------------

    @Test
    fun `CameraX rgba sentinel converts instead of being rejected`() {
        // format 1 is what CameraX really hands over for OUTPUT_IMAGE_FORMAT_RGBA_8888.
        val w = 2
        val h = 2
        val raw = ByteArray(w * h * 4) { (it + 1).toByte() }
        val image = FakeImage(w, h, ImageUtils.CAMERAX_RGBA_SENTINEL, arrayOf(
            FakePlane(raw, rowStride = w * 4, pixelStride = 4)
        ))
        val out = rgb(w, h)

        val res = ImageUtils.toRgb(image, out)
        assertTrue("sentinel RGBA phải chuyển được: $res", res is ImageUtils.Result.Ok)
        assertEquals(1.toByte(), out[0])
        assertEquals("byte alpha bị bỏ nên pixel kế tiếp là 5", 5.toByte(), out[3])
    }

    @Test
    fun `unrecognised format with one plane still converts by plane count`() {
        val w = 2
        val h = 2
        val raw = ByteArray(w * h * 4) { (it * 2).toByte() }
        val image = FakeImage(w, h, 7777, arrayOf(
            FakePlane(raw, rowStride = w * 4, pixelStride = 4)
        ))
        val out = rgb(w, h)

        assertTrue(
            "số plane mới là nguồn sự thật, không phải format: ${ImageUtils.toRgb(image, out)}",
            ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok
        )
    }

    @Test
    fun `describe exposes the raw format and every plane stride`() {
        val w = 2
        val h = 2
        val image = FakeImage(w, h, ImageUtils.CAMERAX_RGBA_SENTINEL, arrayOf(
            FakePlane(ByteArray(64), rowStride = 12, pixelStride = 4)
        ))
        val text = ImageUtils.describe(image)

        assertTrue("phải nêu format: $text", text.contains("fmt=1"))
        assertTrue("phải nêu rowStride: $text", text.contains("row=12"))
        assertTrue("phải nêu pixelStride: $text", text.contains("pix=4"))
    }

    @Test
    fun `unknown format is reported not silently dropped`() {
        val image = FakeImage(2, 2, 999, emptyArray())
        val res = ImageUtils.toRgb(image, rgb(2, 2))
        assertTrue(res is ImageUtils.Result.Failed)
        assertEquals(ImageUtils.Reason.UNSUPPORTED_FORMAT, (res as ImageUtils.Result.Failed).reason)
        assertTrue("chi tiết phải nêu format: ${res.detail}", res.detail.contains("999"))
    }

    @Test
    fun `wrong buffer size is reported`() {
        val raw = ByteArray(2 * 2 * 4)
        val image = FakeImage(2, 2, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = 8, pixelStride = 4)
        ))
        val res = ImageUtils.toRgb(image, ByteArray(5))
        assertEquals(
            ImageUtils.Reason.SIZE_MISMATCH,
            (res as ImageUtils.Result.Failed).reason
        )
    }

    @Test
    fun `truncated buffer is reported instead of throwing`() {
        val raw = ByteArray(6)
        val image = FakeImage(4, 4, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = 16, pixelStride = 4)
        ))
        val res = ImageUtils.toRgb(image, rgb(4, 4))
        assertEquals(
            ImageUtils.Reason.TRUNCATED,
            (res as ImageUtils.Result.Failed).reason
        )
    }

    @Test
    fun `rgba with impossible pixel stride is rejected`() {
        val raw = ByteArray(64)
        val image = FakeImage(2, 2, ImageUtils.FLEX_RGBA_8888, arrayOf(
            FakePlane(raw, rowStride = 8, pixelStride = 2)
        ))
        val res = ImageUtils.toRgb(image, rgb(2, 2))
        assertEquals(
            ImageUtils.Reason.MALFORMED,
            (res as ImageUtils.Result.Failed).reason
        )
    }

    @Test
    fun `single plane that is neither rgba nor yuv is reported`() {
        // Dispatch is by plane count, so one plane means the RGBA path. A plane with
        // pixelStride 1 cannot be RGBA, so it must be reported, not misread.
        val image = FakeImage(2, 2, ImageUtils.YUV_420_888, arrayOf(
            FakePlane(ByteArray(4), rowStride = 2, pixelStride = 1)
        ))
        val res = ImageUtils.toRgb(image, rgb(2, 2))
        assertEquals(
            ImageUtils.Reason.MALFORMED,
            (res as ImageUtils.Result.Failed).reason
        )
    }

    @Test
    fun `three planes that are not yuv are reported not misread`() {
        val image = FakeImage(2, 2, ImageUtils.CAMERAX_RGBA_SENTINEL, arrayOf(
            FakePlane(ByteArray(4), rowStride = 2, pixelStride = 2),
            FakePlane(ByteArray(1), rowStride = 1, pixelStride = 1),
            FakePlane(ByteArray(1), rowStride = 1, pixelStride = 1)
        ))
        val res = ImageUtils.toRgb(image, rgb(2, 2))
        assertTrue("3 plane mà Y pixelStride khác 1 phải báo lỗi: $res",
            res is ImageUtils.Result.Failed)
    }
}