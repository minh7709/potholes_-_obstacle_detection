package com.example.fcw.camera

import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Area-averaged downscale of packed RGB, matching `cv2.INTER_AREA`.
 *
 * `config.yaml: input.max_width/max_height` caps the frame before inference. On a
 * 1280x960 analysis frame this produces 960x720, and the model then sees the same
 * 640x640 letterbox either way, so the saving is preprocessing CPU rather than model
 * time. It still matters: the preprocessor otherwise resamples 1.2 M pixels every
 * frame.
 *
 * A plain box filter, not bilinear. Nearest-neighbour and bilinear both alias thin
 * structures when shrinking, which is exactly what this stage exists to avoid.
 */
object AreaDownscale {

    /** @return [out] at the reduced size, or null when no reduction is needed. */
    fun downscale(
        src: ByteArray, w: Int, h: Int, maxW: Int, maxH: Int,
        out: ByteArray
    ): IntArray? {
        val scale = min(maxW.toDouble() / w, maxH.toDouble() / h)
        if (scale >= 1.0) return null

        val dw = (w * scale).roundToInt().coerceAtLeast(1)
        val dh = (h * scale).roundToInt().coerceAtLeast(1)
        if (dw * dh * 3 > out.size) return null

        val rx = w.toDouble() / dw
        val ry = h.toDouble() / dh

        for (dy in 0 until dh) {
            val y0 = floor(dy * ry).toInt().coerceIn(0, h - 1)
            val y1 = min(h, maxOf(y0 + 1, ((dy + 1) * ry).roundToInt()))
            val wy = y1 - y0
            if (wy <= 0) continue

            for (dx in 0 until dw) {
                val x0 = floor(dx * rx).toInt().coerceIn(0, w - 1)
                val x1 = min(w, maxOf(x0 + 1, ((dx + 1) * rx).roundToInt()))
                val wx = x1 - x0
                if (wx <= 0) continue

                val n = wx * wy
                var r = 0; var g = 0; var b = 0
                for (yy in y0 until y1) {
                    val row = yy * w
                    for (xx in x0 until x1) {
                        val p = (row + xx) * 3
                        r += src[p].toInt() and 0xFF
                        g += src[p + 1].toInt() and 0xFF
                        b += src[p + 2].toInt() and 0xFF
                    }
                }
                val o = (dy * dw + dx) * 3
                out[o] = (r / n).toByte()
                out[o + 1] = (g / n).toByte()
                out[o + 2] = (b / n).toByte()
            }
        }
        return intArrayOf(dw, dh)
    }
}
