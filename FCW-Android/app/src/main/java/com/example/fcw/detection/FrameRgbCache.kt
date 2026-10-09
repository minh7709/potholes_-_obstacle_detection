package com.example.fcw.detection

import com.example.fcw.utils.ImageUtils

/**
 * Converts a frame to tightly packed RGB at most once per frame.
 *
 * Three stages want the same pixels — the detector, depth and ego motion — and each
 * used to walk 1.2M pixels with ByteBuffer.get() on its own. The device HUD read
 * "YOLO 317ms" against a frame time of 1400ms, which pointed at the model; the rest was
 * here, converting the same frame three times over.
 *
 * The cache is keyed on the capture timestamp because that is the only thing
 * identifying a frame. A hit returns the same array: callers must treat it as read-only
 * for the rest of the frame, which is why they share one buffer rather than each owning
 * a copy — a copy per stage would put back the allocation cost this removes.
 *
 * Deliberately free of ONNX types so the cache logic is testable on the JVM.
 */
class FrameRgbCache {

    private var buffer: ByteArray = ByteArray(0)
    private var stamp: Long = NO_FRAME
    private var conversions: Int = 0
    private var lastConvertMs: Float = 0f
    private var measuredStamp: Long = NO_FRAME

    /** How many conversions actually ran. One per frame is the healthy value. */
    val conversionCount: Int get() = conversions

    /** Milliseconds of the most recent frame that did convert, not of a cache hit. */
    val milliseconds: Float get() = lastConvertMs

    /** The RGB of the frame identified by [frameStamp]. Null when it cannot be read. */
    fun get(
        frameW: Int,
        frameH: Int,
        frameStamp: Long,
        read: (ByteArray) -> ImageUtils.Result
    ): ImageUtils.Result {
        val need = ImageUtils.rgbBufferSize(frameW, frameH)
        if (buffer.size != need) {
            buffer = ByteArray(need)
            // A different geometry invalidates whatever was cached: the old array was
            // sized for the old frame and its pixels mean nothing now.
            stamp = NO_FRAME
        }

        if (frameStamp != NO_FRAME && frameStamp == stamp) {
            return ImageUtils.Result.Ok(frameW, frameH)
        }

        val started = System.nanoTime()
        val result = read(buffer)
        val elapsed = (System.nanoTime() - started) / 1_000_000f

        if (result is ImageUtils.Result.Ok) {
            stamp = frameStamp
            conversions++
            // Only frames that converted update the timing, otherwise a run of hits
            // would report the cost of reading a buffer that was never touched.
            if (frameStamp != measuredStamp) {
                measuredStamp = frameStamp
                lastConvertMs = elapsed
            }
        }
        return result
    }

    /** The cached RGB array, or null before the first conversion. */
    fun peek(): ByteArray? = if (buffer.isEmpty()) null else buffer

    fun invalidate() {
        stamp = NO_FRAME
    }

    companion object {
        /** Sentinel for "this frame has no usable identity". */
        const val NO_FRAME = Long.MIN_VALUE
    }
}