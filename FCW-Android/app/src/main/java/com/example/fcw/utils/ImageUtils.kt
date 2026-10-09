package com.example.fcw.utils

import androidx.camera.core.ImageProxy
import com.example.fcw.utils.ImageUtils.Result.Failed
import com.example.fcw.utils.ImageUtils.Result.Ok
import java.nio.ByteBuffer

/**
 * Converts a CameraX [ImageProxy] into tightly packed RGB bytes.
 *
 * Two formats are handled:
 *  - FLEX_RGBA_8888: one plane, 4 bytes per pixel. Fast path, no colour maths.
 *  - YUV_420_888: three planes, chroma subsampled 2x2. Needs real conversion.
 *
 * The RGBA path is requested from CameraX, but a device is free to hand back
 * YUV_420_888 instead, and the previous version of this file rejected anything it
 * did not recognise. That turned a format difference into a pipeline with no
 * detections at all, which is indistinguishable from "nothing on the road".
 */
object ImageUtils {

    /** FLEX_RGBA_8888. Not a constant in android.graphics.ImageFormat on all API levels. */
    const val FLEX_RGBA_8888 = 0x7F000888
    const val YUV_420_888 = 35

    /**
     * What CameraX actually reports for ImageAnalysis with
     * OUTPUT_IMAGE_FORMAT_RGBA_8888.
     *
     * CameraX stores its own OUTPUT_IMAGE_FORMAT_RGBA_8888 = 2 flag, then converts it
     * to the value 1 before it reaches ImageInfo. AndroidImageProxy.getFormat()
     * forwards android.media.Image.getFormat() verbatim, so 1 is what the app sees.
     * It is not android.graphics.ImageFormat at all, which is why comparing against
     * FLEX_RGBA_8888 and YUV_420_888 rejected every frame the device produced.
     */
    const val CAMERAX_RGBA_SENTINEL = 1

    enum class Reason(val label: String) {
        UNSUPPORTED_FORMAT("FORMAT KHÔNG HỖ TRỢ"),
        SIZE_MISMATCH("KÍCH THƯỚC KHÔNG KHỚP"),
        NO_PLANES("KHÔNG CÓ PLANE"),
        PLANE_COUNT("SỐ PLANE KHÔNG ĐÚNG"),
        MALFORMED("THAM SỐ PLANE KHÔNG HỢP LỆ"),
        TRUNCATED("BUFFER BỊ CẮT")
    }

    sealed interface Result {
        data class Ok(val width: Int, val height: Int) : Result
        data class Failed(val reason: Reason, val detail: String) : Result
    }

    /**
     * One-line dump of the frame layout, for the on-screen HUD.
     *
     * The HUD exists so a stalled pipeline can be diagnosed without a USB cable and
     * logcat. Showing the raw format integer matters because that is the value that
     * was wrong twice already.
     */
    fun describe(image: ImageProxy): String {
        val sb = StringBuilder()
        sb.append("fmt=${image.format}")
        for ((i, plane) in image.planes.withIndex()) {
            sb.append(" | p$i row=${plane.rowStride} pix=${plane.pixelStride}")
            sb.append(" lim=${plane.buffer.limit()} pos=${plane.buffer.position()}")
        }
        return sb.toString()
    }

    /** Bytes needed for a tightly packed RGB buffer of the given size. */
    fun rgbBufferSize(width: Int, height: Int): Int = width * height * 3

    /**
     * Copies [image] into [outRgb] as 3 bytes per pixel, top-down.
     *
     * Never throws. Returns the failure reason instead, so the caller can report
     * precisely what was wrong with the frame.
     */
    fun toRgb(image: ImageProxy, outRgb: ByteArray): Result {
        val planeCount = image.planes.size
        // Dispatch on plane count, not on the format integer. The format value comes
        // from CameraX and android.media.Image and has already surprised us once;
        // the number of planes is fixed by the image structure itself.
        return when {
            planeCount >= 3 -> fromYuv(image, outRgb)
            planeCount == 1 -> fromRgba(image, outRgb)
            else -> Failed(
                Reason.UNSUPPORTED_FORMAT,
                "format=${image.format} (${image.width}x${image.height}) planes=$planeCount"
            )
        }
    }

    private fun fromRgba(image: ImageProxy, outRgb: ByteArray): Result {
        val planes = image.planes
        if (planes.isEmpty()) return Failed(Reason.NO_PLANES, "planes=0")
        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val w = image.width
        val h = image.height

        if (pixelStride < 3) {
            return Failed(Reason.MALFORMED, "pixelStride=$pixelStride < 3")
        }
        if (outRgb.size != rgbBufferSize(w, h)) {
            return Failed(Reason.SIZE_MISMATCH, "cần ${rgbBufferSize(w, h)} byte, có ${outRgb.size}")
        }

        val base = buffer.position()
        val span = (h - 1) * rowStride + (w - 1) * pixelStride + 3
        if (base + span > buffer.limit()) {
            return Failed(Reason.TRUNCATED, "cần $span byte từ $base, buffer có ${buffer.limit()}")
        }

        var di = 0
        for (y in 0 until h) {
            var si = base + y * rowStride
            for (x in 0 until w) {
                outRgb[di] = buffer.get(si)
                outRgb[di + 1] = buffer.get(si + 1)
                outRgb[di + 2] = buffer.get(si + 2)
                si += pixelStride
                di += 3
            }
        }
        return Ok(w, h)
    }

    private fun fromYuv(image: ImageProxy, outRgb: ByteArray): Result {
        val planes = image.planes
        if (planes.size < 3) {
            return Failed(Reason.PLANE_COUNT, "YUV cần 3 plane, có ${planes.size}")
        }
        val (yPlane, uPlane, vPlane) = Triple(planes[0], planes[1], planes[2])
        val w = image.width
        val h = image.height

        if (outRgb.size != rgbBufferSize(w, h)) {
            return Failed(Reason.SIZE_MISMATCH, "cần ${rgbBufferSize(w, h)} byte, có ${outRgb.size}")
        }
        if (yPlane.pixelStride != 1) {
            return Failed(Reason.MALFORMED, "Y pixelStride=${yPlane.pixelStride}, cần 1")
        }

        val chromaRows = (h + 1) / 2
        val chromaCols = (w + 1) / 2
        val yNeed = (h - 1) * yPlane.rowStride + w
        val uNeed = (chromaRows - 1) * uPlane.rowStride + (chromaCols - 1) * uPlane.pixelStride + 1
        val vNeed = (chromaRows - 1) * vPlane.rowStride + (chromaCols - 1) * vPlane.pixelStride + 1
        if (yPlane.buffer.position() + yNeed > yPlane.buffer.limit()) {
            return Failed(Reason.TRUNCATED, "plane Y thiếu dữ liệu")
        }
        if (uPlane.buffer.position() + uNeed > uPlane.buffer.limit()) {
            return Failed(Reason.TRUNCATED, "plane U thiếu dữ liệu")
        }
        if (vPlane.buffer.position() + vNeed > vPlane.buffer.limit()) {
            return Failed(Reason.TRUNCATED, "plane V thiếu dữ liệu")
        }

        val yb = yPlane.buffer
        val ub = uPlane.buffer
        val vb = vPlane.buffer
        val yRow = yPlane.rowStride
        val uRow = uPlane.rowStride
        val vRow = vPlane.rowStride
        val uPix = uPlane.pixelStride
        val vPix = vPlane.pixelStride
        val yBase = yb.position()
        val uBase = ub.position()
        val vBase = vb.position()

        var di = 0
        for (y in 0 until h) {
            val yOff = yBase + y * yRow
            val cy = y / 2
            val uOff = uBase + cy * uRow
            val vOff = vBase + cy * vRow
            for (x in 0 until w) {
                val luma = yb.get(yOff + x).toInt() and 0xFF
                val cb = ub.get(uOff + (x / 2) * uPix).toInt() and 0xFF
                val cr = vb.get(vOff + (x / 2) * vPix).toInt() and 0xFF

                val dv = cr - 128
                val du = cb - 128
                val r = luma + 1.402f * dv
                val g = luma - 0.344136f * du - 0.714136f * dv
                val b = luma + 1.772f * du

                outRgb[di] = clampByte(r)
                outRgb[di + 1] = clampByte(g)
                outRgb[di + 2] = clampByte(b)
                di += 3
            }
        }
        return Ok(w, h)
    }

    private fun clampByte(v: Float): Byte {
        val i = (v + 0.5f).toInt()
        return when {
            i < 0 -> 0
            i > 255 -> 255.toByte()
            else -> i.toByte()
        }
    }
}