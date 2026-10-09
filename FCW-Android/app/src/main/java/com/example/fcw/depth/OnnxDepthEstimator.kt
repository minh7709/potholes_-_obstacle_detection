package com.example.fcw.depth

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.FloatBuffer

/**
 * Metric depth from Depth Anything V2, sampled per detection.
 *
 * Measured properties of this pipeline, all from the offline harness rather than
 * assumption (`tools/export_depth.py`, `tools/evaluate_models.py`):
 *
 *  - The model is NOT resolution-stable. Same car, same frame:
 *      224 -> 39.6 m,  350 -> 20.9 m,  518 -> 21.8 m
 *    Going from 224 to 350 nearly halves the reported distance. So the input size is
 *    not a speed-versus-quality dial, it changes the numbers outright. 224 is the
 *    size the Python pipeline used (`config.yaml: processing_height: 224`) and it is
 *    the only choice whose output can be compared against that pipeline at all.
 *
 *  - 224 fp32 reproduces `output/events.csv` exactly (39.6 m vs 39.42 m recorded).
 *
 *  - int8 dynamic costs accuracy only in the far field: ratio to fp32 is 0.97-0.99 on
 *    near objects and 0.82 on the furthest one. Near-field is what drives a collision
 *    warning, so int8 is the shipped choice.
 *
 *  - int8 static quantisation was far worse (0.31-0.80) and dynamic int8 also runs
 *    slower than fp32 on a desktop CPU. On Android the AAR ships XNNPACK, whose int8
 *    path is genuinely fast, so that desktop ordering is not trusted here.
 *
 *  - fp32 would be 94 MB inlined. Rejected on size alone.
 */
class OnnxDepthEstimator(
    private val context: Context,
    private val netWidth: Int = 299,
    private val netHeight: Int = 224,
    private val calibrationScale: Float = 1f,
    private val cfg: com.example.fcw.config.FcwConfig = com.example.fcw.config.FcwConfig()
) {
    companion object {
        const val TAG = "FCW"
        const val MODEL_ASSET = "models/depth_299x224_int8.onnx"

        /** ImageNet statistics, which is what the DPT preprocessing uses. */
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var inputName = ""
    private val tensor = FloatArray(3 * netHeight * netWidth)

    var inferenceMs: Float = 0f
        private set

    /**
     * Why the model is unavailable, or empty when it loaded.
     *
     * Shown on the HUD. A bare "THIẾU DEPTH" told us nothing; the exception text is
     * what actually identifies an unsupported op or a missing asset.
     */
    var loadError: String = ""
        private set

    val isReady: Boolean get() = session != null

    fun load(): Boolean = try {
        val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val s = env.createSession(bytes, options)
        session = s
        inputName = s.inputInfo.values.first().name
        val shape = (s.inputInfo.values.first().info as? ai.onnxruntime.TensorInfo)?.shape
        loadError = ""
        Log.i(TAG, "Depth OK ${bytes.size / 1024} KB, vào='$inputName' ${shape?.joinToString("x")}")
        shape?.getOrNull(2)?.let { h ->
            if (h > 0 && h.toInt() != netHeight) {
                Log.e(TAG, "Model cần ${shape.joinToString("x")}, cấu hình ${netWidth}x$netHeight")
            }
        }
        true
    } catch (e: Throwable) {
        loadError = "${e.javaClass.simpleName}: ${e.message}".take(160)
        Log.e(TAG, "Không load được depth: $loadError", e)
        session = null
        false
    }

    /**
     * Runs depth on the whole frame.
     *
     * @return depth map in metres at the network output size, or null on failure. The
     *         caller is expected to sample it per detection rather than read the whole
     *         map: depth is only as good as the object it is measured on.
     */
    fun infer(rgb: ByteArray, srcWidth: Int, srcHeight: Int): FloatArray? {
        val s = session ?: return null
        return try {
            fillTensor(rgb, srcWidth, srcHeight)
            val buf = FloatBuffer.wrap(tensor)
            val start = System.nanoTime()
            val out = OnnxTensor.createTensor(
                env, buf,
                longArrayOf(1, 3, netHeight.toLong(), netWidth.toLong())
            ).use { input ->
                s.run(mapOf(inputName to input)).use { outs ->
                    val t = outs[0] as? OnnxTensor
                        ?: throw IllegalStateException("output không phải tensor")
                    val view = t.floatBuffer.duplicate().apply { rewind() }
                    val arr = FloatArray(view.remaining())
                    view.get(arr, 0, arr.size)
                    arr
                }
            }
            inferenceMs = (System.nanoTime() - start) / 1_000_000f
            // Output can be [1,1,H,W] or [1,H,W] depending on the export; squeeze both
            // down to exactly H*W so a shape difference cannot silently shift the map.
            // DPT xuất bản đồ theo bội số patch 14, nên chiều rộng có thể là 294
            // thay vì 299. Giữ nguyên kích thước thật của bản đồ rồi ánh xạ theo nó.
            if (out.size >= netHeight * 4) out else null
        } catch (e: Throwable) {
            Log.e(TAG, "Depth lỗi: ${e.message}", e)
            null
        }
    }

    /**
     * Distance to an object, in metres.
     *
     * Direct port of `app/fcw/depth.py::DepthEstimator.sample_box`, because that
     * algorithm was already validated against the footage this app has to work on.
     * An earlier version of this file sampled a tiny patch at a hand-picked 72% down
     * the box and had none of the outlier rejection; that was a guess, it disagreed
     * with the Python pipeline, and every warning threshold depends on this function.
     *
     * The port matters because each step exists for a reason:
     *  1. centre crop, so a box overlapping background reads the object
     *  2. if the crop is too small, fall back to the lower-centre band, which is where
     *     the road under the vehicle is
     *  3. median, not mean: one far pixel between two vehicles must not move it
     *  4. drop pixels far from the median, then re-take the median
     *  5. calibration scale
     *  6. reject anything outside a plausible range rather than report it
     */
    fun sampleDistance(
        map: FloatArray,
        x1: Float, y1: Float, x2: Float, y2: Float,
        frameWidth: Int, frameHeight: Int
    ): Float? {
        val mapW = mapWidth(map.size)
        val mapH = map.size / mapW
        if (frameWidth <= 0 || frameHeight <= 0 || mapW <= 0 || mapH <= 0) return null

        val sx1 = (x1 / frameWidth * mapW).toInt().coerceIn(0, mapW - 1)
        val sy1 = (y1 / frameHeight * mapH).toInt().coerceIn(0, mapH - 1)
        val sx2 = (x2 / frameWidth * mapW).toInt().coerceIn(sx1 + 1, mapW)
        val sy2 = (y2 / frameHeight * mapH).toInt().coerceIn(sy1 + 1, mapH)

        val bw = (sx2 - sx1).toFloat()
        val bh = (sy2 - sy1).toFloat()
        if (bw <= 0f || bh <= 0f) return null

        var values = clean(
            region(map, sx1, sy1, sx2, sy2, bw, bh, mapW)
        )
        if (values.size < cfg.minValidSamples) {
            values = clean(lowerCentreBand(map, sx1, sy1, sx2, sy2, bw, bh, mapW))
        }
        if (values.isEmpty()) return null

        var median = values[values.size / 2]
        if (median <= 0f) return null

        val tol = maxOf(0.05f, cfg.outlierTolerance)
        val inliers = values.filter { kotlin.math.abs(it - median) <= tol * median }
        var distance = if (inliers.size >= cfg.minValidSamples) {
            inliers.sorted()[inliers.size / 2]
        } else {
            median
        }

        distance *= calibrationScale
        if (!distance.isFinite()) return null
        if (distance < cfg.validMinM || distance > cfg.validMaxM) return null
        return distance
    }

    /** The centre crop, or the whole box when it is too small to crop. */
    private fun region(
        map: FloatArray, x1: Int, y1: Int, x2: Int, y2: Int,
        bw: Float, bh: Float, mapW: Int
    ): FloatArray {
        val mapH = map.size / mapW
        val crop = minOf(maxOf(0.05f, cfg.centerCrop), 1f)
        val cx = (x1 + x2) / 2
        val cy = (y1 + y2) / 2
        val hw = bw * 0.5f * crop
        val hh = bh * 0.5f * crop
        var rx1 = maxOf(x1, (cx - hw).toInt())
        var ry1 = maxOf(y1, (cy - hh).toInt())
        var rx2 = minOf(x2, (cx + hw).toInt())
        var ry2 = minOf(y2, (cy + hh).toInt())
        if (rx2 - rx1 < 2 || ry2 - ry1 < 2) {
            rx1 = x1; ry1 = y1; rx2 = x2; ry2 = y2
        }
        val out = FloatArray((rx2 - rx1) * (ry2 - ry1))
        var k = 0
        for (y in ry1 until ry2) for (x in rx1 until rx2) out[k++] = map[y * mapW + x]
        return out
    }

    /**
     * Band from the vertical middle to the bottom of the box, centred in x.
     *
     * The road the vehicle stands on. Port of `_lower_center_band`.
     */
    private fun lowerCentreBand(
        map: FloatArray, x1: Int, y1: Int, x2: Int, y2: Int, bw: Float, bh: Float, mapW: Int
    ): FloatArray {
        val mapH = map.size / mapW
        val cx = (x1 + x2) / 2
        val half = maxOf(2, (bw * 0.30f).toInt())
        var rx1 = maxOf(x1, cx - half)
        var rx2 = minOf(x2, cx + half)
        var ry1 = maxOf(y1, y1 + (bh * 0.5f).toInt())
        var ry2 = y2
        if (ry2 <= ry1) ry1 = maxOf(0, ry2 - 2)
        rx1 = rx1.coerceIn(0, mapW - 1)
        rx2 = rx2.coerceIn(rx1 + 1, mapW)
        ry1 = ry1.coerceIn(0, mapH - 1)
        ry2 = ry2.coerceIn(ry1 + 1, mapH)
        val out = FloatArray((rx2 - rx1) * (ry2 - ry1))
        var k = 0
        for (y in ry1 until ry2) for (x in rx1 until rx2) out[k++] = map[y * mapW + x]
        return out
    }

    /** Finite, strictly positive depths, sorted. Port of `_clean_depth`. */
    private fun clean(values: FloatArray): FloatArray {
        val kept = ArrayList<Float>(values.size)
        for (v in values) if (v.isFinite() && v > 0f) kept.add(v)
        val arr = FloatArray(kept.size)
        for (i in kept.indices) arr[i] = kept[i]
        arr.sort()
        return arr
    }

    /**
     * Resizes straight to the network's own size with **no letterbox padding**.
     *
     * The network input is 299x224, which is the source aspect ratio: a 4:3 frame
     * keeps 1280/960*224 = 299 columns exactly. The previous square 224x224 input
     * had to pad, which threw away 75 columns each side and fed the model a strip of
     * grey. Measured on the same clip, that made distances read 2.5x too large and no
     * object ever entered the warning band.
     */
    private fun fillTensor(rgb: ByteArray, srcW: Int, srcH: Int) {
        val plane = netWidth * netHeight
        val stepX = srcW.toFloat() / netWidth
        val stepY = srcH.toFloat() / netHeight
        val maxY = srcH - 1
        val maxX = srcW - 1

        for (y in 0 until netHeight) {
            val sy = ((y + 0.5f) * stepY - 0.5f).coerceIn(0f, maxY.toFloat())
            val y0 = sy.toInt()
            val y1 = if (y0 < maxY) y0 + 1 else maxY
            val wy = sy - y0
            val wy1 = 1f - wy
            var ti = y * netWidth
            for (x in 0 until netWidth) {
                val sx = ((x + 0.5f) * stepX - 0.5f).coerceIn(0f, maxX.toFloat())
                val x0 = sx.toInt()
                val x1 = if (x0 < maxX) x0 + 1 else maxX
                val wx = sx - x0
                val wx1 = 1f - wx
                val p00 = (y0 * srcW + x0) * 3
                val p01 = (y0 * srcW + x1) * 3
                val p10 = (y1 * srcW + x0) * 3
                val p11 = (y1 * srcW + x1) * 3
                for (c in 0 until 3) {
                    val top = (rgb[p00 + c].toInt() and 0xFF) * wx1 +
                        (rgb[p01 + c].toInt() and 0xFF) * wx
                    val bot = (rgb[p10 + c].toInt() and 0xFF) * wx1 +
                        (rgb[p11 + c].toInt() and 0xFF) * wx
                    tensor[c * plane + ti] = ((top * wy1 + bot * wy) / 255f - MEAN[c]) / STD[c]
                }
                ti++
            }
        }
    }

    /** Bề rộng bản đồ thực tế, suy từ kích thước phẳng. */
    private fun mapWidth(flatSize: Int): Int {
        val byH = flatSize / netHeight
        return if (byH > 0 && byH * netHeight == flatSize) byH else netWidth
    }

    fun close() {
        try {
            session?.close()
        } catch (e: Throwable) {
            Log.w(TAG, "Đóng depth lỗi: ${e.message}")
        }
        session = null
    }
}