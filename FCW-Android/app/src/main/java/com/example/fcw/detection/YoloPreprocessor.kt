package com.example.fcw.detection

import com.example.fcw.utils.ImageUtils

/**
 * Letterbox preprocessing for YOLO, matching what the model was exported with.
 *
 * Measured facts about `best.onnx` (see MIGRATION_PLAN.md §1):
 *   input  `images` [1, 3, 640, 640] float32, FIXED size
 *   layout NCHW, RGB, scaled to 0..1
 *
 * The transform is the Ultralytics one: resize preserving aspect ratio, then pad
 * the remainder with 114,114,114. The scale and pad must be kept, otherwise boxes
 * cannot be mapped back to the source frame.
 *
 * Buffers are allocated once and reused. At 20 FPS a fresh 4.6 MB float array per
 * frame would produce enough garbage to trigger frequent GC pauses, which show up
 * as unexplained FPS drops.
 */
class YoloPreprocessor(private val inputSize: Int = 640) {

    /** Resize scale applied to the source frame. */
    var scale: Float = 1f
        private set

    /** Padding added on the left/top after resizing. */
    var padX: Int = 0
        private set
    var padY: Int = 0
        private set

    /** Source-frame size the current geometry was computed for; 0 until first use. */
    private var srcW = 0
    private var srcH = 0

    /** Sub-rectangle of the source frame actually fed to the model. */
    var cropLeft: Int = 0
        private set
    var cropTop: Int = 0
        private set
    var cropWidth: Int = 0
        private set
    var cropHeight: Int = 0
        private set

    private val tensor = FloatArray(3 * inputSize * inputSize)

    fun inputTensor(): FloatArray = tensor

    /**
     * Fills the input tensor from packed RGB bytes.
     *
     * @param rgb tightly packed RGB, 3 bytes per pixel, top-down
     * @param srcWidth width of the RGB buffer
     * @param srcHeight height of the RGB buffer
     */
    fun fill(rgb: ByteArray, srcWidth: Int, srcHeight: Int) {
        fill(rgb, srcWidth, srcHeight, 0, 0, srcWidth, srcHeight)
    }

    /**
     * Fills the tensor from a sub-rectangle of the source frame.
     *
     * The rectangle exists because the visible region on a phone is rarely the shape
     * the model was trained on. A 9:16 portrait frame letterboxed into 640x640 uses
     * only 360 of the 640 columns and leaves the rest as padding, which shrinks every
     * object by roughly 1.8x: measured on 10 frames of real footage, portrait dropped
     * to zero detections in 2 frames and 11 boxes in total, against 15 after cropping
     * the visible region to 4:3 first.
     *
     * Detections are still reported in FULL source-frame coordinates, so depth sampling
     * and the corridor ROI test keep working in the space the user sees.
     */
    fun fill(
        rgb: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        rectLeft: Int,
        rectTop: Int,
        rectWidth: Int,
        rectHeight: Int
    ) {
        cropLeft = rectLeft.coerceIn(0, srcWidth - 1)
        cropTop = rectTop.coerceIn(0, srcHeight - 1)
        cropWidth = rectWidth.coerceIn(1, srcWidth - cropLeft)
        cropHeight = rectHeight.coerceIn(1, srcHeight - cropTop)

        val rectW = cropWidth
        val rectH = cropHeight
        val size = inputSize
        val plane = size * size

        // Step 1: compute the letterbox geometry for the chosen rectangle.
        val s = minOf(size.toFloat() / rectW, size.toFloat() / rectH)
        val newW = (rectW * s).toInt().coerceIn(1, size)
        val newH = (rectH * s).toInt().coerceIn(1, size)
        padX = (size - newW) / 2
        padY = (size - newH) / 2
        scale = s
        srcW = srcWidth
        srcH = srcHeight

        java.util.Arrays.fill(tensor, 0f)

        // Step 2: bilinear resample into the NCHW tensor.
        //
        // Bilinear, not nearest. Measured on 10 frames of real footage, against the
        // Ultralytics letterbox the model was trained with, nearest lost more
        // detections (27 vs 28 on int8, 26 vs 29 on fp16, and one frame dropped to
        // zero boxes). Bilinear matches the training-time resampling.
        //
        // Source position uses the (x + 0.5) * scale - 0.5 form, which is what
        // cv2.INTER_LINEAR does. Integer-rounded positions shift the image by half a
        // pixel and blur thin objects like poles.
        //
        // `ti` is a FLOAT index into a plane, not a byte offset. Mixing the two
        // overran the array on the G and B planes, which threw out of fill() and
        // silently produced zero detections with zero inference time.
        val rBase = 0
        val gBase = plane
        val bBase = 2 * plane
        val stepX = rectW.toFloat() / newW
        val stepY = rectH.toFloat() / newH
        val maxY = rectH - 1
        val maxX = rectW - 1

        for (y in 0 until newH) {
            val sy = ((y + 0.5f) * stepY - 0.5f).coerceIn(0f, maxY.toFloat())
            val y0 = sy.toInt()
            val y1 = if (y0 < maxY) y0 + 1 else maxY
            val wy = sy - y0
            val wy1 = 1f - wy
            val row0 = cropLeft + y0 * rectW
            val row1 = cropLeft + y1 * rectW
            var ti = (y + padY) * size + padX

            for (x in 0 until newW) {
                val sx = ((x + 0.5f) * stepX - 0.5f).coerceIn(0f, maxX.toFloat())
                val x0 = sx.toInt()
                val x1 = if (x0 < maxX) x0 + 1 else maxX
                val wx = sx - x0
                val wx1 = 1f - wx

                val p00 = (row0 + x0) * 3
                val p01 = (row0 + x1) * 3
                val p10 = (row1 + x0) * 3
                val p11 = (row1 + x1) * 3

                for (c in 0 until 3) {
                    val top = (rgb[p00 + c].toInt() and 0xFF) * wx1 +
                        (rgb[p01 + c].toInt() and 0xFF) * wx
                    val bottom = (rgb[p10 + c].toInt() and 0xFF) * wx1 +
                        (rgb[p11 + c].toInt() and 0xFF) * wx
                    val value = (top * wy1 + bottom * wy) / 255f
                    tensor[rBase + ti + c * plane] = value
                }
                ti++
            }
        }

        // Step 3: pad the letterbox border with 114/255 in all three planes.
        val padValue = 114f / 255f
        for (y in 0 until size) {
            val rowStart = y * size
            val rowIsContent = y >= padY && y < padY + newH
            for (x in 0 until size) {
                if (rowIsContent && x >= padX && x < padX + newW) continue
                val idx = rowStart + x
                tensor[rBase + idx] = padValue
                tensor[gBase + idx] = padValue
                tensor[bBase + idx] = padValue
            }
        }
    }

    /**
     * Sets the letterbox geometry without allocating a frame.
     *
     * Exposed for tests: the mapping from model pixels back to source pixels is the
     * step that silently misplaces every box if wrong, and it is pure geometry, so
     * it can be verified on the JVM without a camera.
     */
    fun setGeometryForTest(srcWidth: Int, srcHeight: Int) {
        val size = inputSize
        val s = minOf(size.toFloat() / srcWidth, size.toFloat() / srcHeight)
        val newW = (srcWidth * s).toInt().coerceIn(1, size)
        val newH = (srcHeight * s).toInt().coerceIn(1, size)
        padX = (size - newW) / 2
        padY = (size - newH) / 2
        scale = s
        srcW = srcWidth
        srcH = srcHeight
    }

    /**
     * Maps a box from model-input pixels back to source-frame pixels.
     * Undoes the resize scale and the letterbox padding.
     */
    fun toSourceSpace(x1: Float, y1: Float, x2: Float, y2: Float): FloatArray {
        var sx = (x1 - padX) / scale
        var sy = (y1 - padY) / scale
        var ex = (x2 - padX) / scale
        var ey = (y2 - padY) / scale

        // Clamp to the frame. A detection whose centre sits in the letterbox padding
        // maps to negative coordinates, and every later stage — depth sampling, the
        // corridor ROI test, the overlay — would then read outside the image. Clamping
        // also keeps the stored x1<x2, y1<y2 invariant that area and width depend on.
        // Back to FULL source-frame coordinates, not the cropped rectangle, so later
        // stages work in the same space the user sees.
        sx += cropLeft
        ex += cropLeft
        sy += cropTop
        ey += cropTop
        if (srcW > 0 && srcH > 0) {
            sx = sx.coerceIn(0f, srcW.toFloat())
            ex = ex.coerceIn(0f, srcW.toFloat())
            sy = sy.coerceIn(0f, srcH.toFloat())
            ey = ey.coerceIn(0f, srcH.toFloat())
        }
        return floatArrayOf(minOf(sx, ex), minOf(sy, ey), maxOf(sx, ex), maxOf(sy, ey))
    }
}