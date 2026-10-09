package com.example.fcw

import com.example.fcw.detection.FrameRgbCache
import com.example.fcw.utils.ImageUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The per-frame RGB conversion happens once, not once per stage.
 *
 * The detector, depth and ego motion all need the same pixels of the same frame. Each
 * used to walk 1.2M pixels with ByteBuffer.get() into one shared buffer, so a single
 * pass over a frame converted it three times. On the device that showed up as 0.7 FPS
 * while the HUD blamed the 317ms model: the model was never the bulk of the frame.
 *
 * The cache must therefore never serve one frame's pixels for another. That is the
 * failure mode worth testing here; the miss/hit counting is only the evidence.
 */
class FrameRgbCacheTest {

    /** Counts how many times the conversion body actually ran. */
    private class CountingReader(private val w: Int, private val h: Int) : (ByteArray) -> ImageUtils.Result {
        var calls = 0
            private set

        override fun invoke(buffer: ByteArray): ImageUtils.Result {
            calls++
            // Write the frame identity into the pixels so a stale buffer is visible.
            val stamp = (calls * 7 % 251).toByte()
            buffer.fill(stamp)
            return ImageUtils.Result.Ok(w, h)
        }
    }

    @Test
    fun `three stages reading one frame convert it once`() {
        val cache = FrameRgbCache()
        val reader = CountingReader(8, 8)

        // detect(), then depth, then ego motion — all on frame 1000.
        repeat(3) {
            assertTrue(
                cache.get(8, 8, 1000L, reader) is ImageUtils.Result.Ok
            )
        }

        assertEquals("3 lần đọc cùng 1 frame chỉ nên chuyển 1 lần", 1, reader.calls)
        assertEquals(1, cache.conversionCount)
    }

    @Test
    fun `the next frame always reconverts`() {
        val cache = FrameRgbCache()
        val reader = CountingReader(8, 8)

        cache.get(8, 8, 1000L, reader)
        cache.get(8, 8, 2000L, reader)

        assertEquals("frame mới phải chuyển lại", 2, reader.calls)
    }

    @Test
    fun `a frame with no usable identity never reuses a cached buffer`() {
        // Depth must never measure a distance from another frame's pixels. A frame
        // without a timestamp cannot be told apart, so it has to bypass the cache.
        val cache = FrameRgbCache()
        val reader = CountingReader(8, 8)

        cache.get(8, 8, 1000L, reader)
        cache.get(8, 8, FrameRgbCache.NO_FRAME, reader)
        cache.get(8, 8, FrameRgbCache.NO_FRAME, reader)

        assertEquals("frame không định danh phải chuyển mỗi lần", 3, reader.calls)
    }

    @Test
    fun `a failed conversion is not cached`() {
        val cache = FrameRgbCache()
        var calls = 0

        // Same stamp twice, both failing: the second attempt must actually retry.
        repeat(2) {
            cache.get(4, 4, 500L) {
                calls++
                ImageUtils.Result.Failed(
                    ImageUtils.Reason.TRUNCATED, "plane Y thiếu dữ liệu"
                )
            }
        }

        assertEquals("lỗi không được cache", 2, calls)
        assertEquals(0, cache.conversionCount)
    }

    @Test
    fun `changing frame size drops the cache`() {
        val cache = FrameRgbCache()
        val small = CountingReader(4, 4)
        cache.get(4, 4, 1000L, small)

        val large = CountingReader(16, 16)
        cache.get(16, 16, 1000L, large)

        assertEquals(
            "kích thước đổi thì cache cũ không còn ý nghĩa",
            2, cache.conversionCount
        )
    }

    @Test
    fun `cache hit reports the frame size it holds`() {
        val cache = FrameRgbCache()
        cache.get(8, 8, 1000L, CountingReader(8, 8))

        val hit = cache.get(8, 8, 1000L, CountingReader(8, 8))
        assertTrue(hit is ImageUtils.Result.Ok)
        assertEquals(8, (hit as ImageUtils.Result.Ok).width)
        assertEquals(8, hit.height)
    }

    @Test
    fun `peek returns null before anything was converted`() {
        assertEquals(null, FrameRgbCache().peek())
    }

    @Test
    fun `buffer length matches the packed rgb size`() {
        val cache = FrameRgbCache()
        cache.get(8, 8, 1L, CountingReader(8, 8))
        assertEquals(8 * 8 * 3, cache.peek()!!.size)
    }

    @Test
    fun `invalidate forces the next frame to convert again`() {
        val cache = FrameRgbCache()
        val reader = CountingReader(8, 8)
        cache.get(8, 8, 1000L, reader)
        cache.invalidate()
        cache.get(8, 8, 1000L, reader)

        assertEquals(2, reader.calls)
    }
}