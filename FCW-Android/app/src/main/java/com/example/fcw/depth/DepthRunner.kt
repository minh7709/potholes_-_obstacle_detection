package com.example.fcw.depth

import android.content.Context
import android.util.Log
import com.example.fcw.config.FcwConfig
import com.example.fcw.detection.Detection
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Runs depth on its own thread so it overlaps with detection instead of queueing behind it.
 *
 * Before this, one thread did depth (565 ms) then YOLO (280 ms) in sequence: 855 ms of
 * model time per cycle, and the camera was fed roughly once per second. Splitting them
 * is the only lever that reduces latency without touching a threshold or the model.
 *
 * Division of labour, chosen so there is nothing to lock:
 *  - this thread owns the ORT session and the RGB buffers, and publishes a depth map
 *  - the pipeline thread owns TrackHistory and is the only writer to it
 *
 * The pipeline samples the published map, so the history never sees two writers. Only
 * immutable snapshots cross the thread boundary.
 *
 * Two dedicated RGB buffers are pooled. The pipeline's own pool cannot be shared: its
 * buffers are recycled as soon as detection finishes, and depth may still be reading
 * one. If depth is still busy when the next frame arrives, that frame is dropped for
 * depth only — the same keep-latest rule the camera uses, because a stale depth map is
 * far less harmful than a growing queue.
 */
class DepthRunner(
    context: Context,
    cfg: FcwConfig = FcwConfig()
) {
    companion object {
        const val TAG = "FCW"
    }

    private val cfg = cfg
    private val estimator = OnnxDepthEstimator(
        context, netWidth = cfg.depthNetWidth, netHeight = cfg.depthNetHeight,
        calibrationScale = cfg.depthScale, cfg = cfg
    )
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "fcw-depth").apply { priority = Thread.NORM_PRIORITY + 1 }
    }

    private val busy = AtomicBoolean(false)
    private val buffers = ArrayDeque<ByteArray>()

    @Volatile
    private var pendingRgb: ByteArray? = null

    @Volatile
    private var pendingW = 0

    @Volatile
    private var pendingH = 0

    /** Latest completed depth map, or null before the first one lands. */
    @Volatile
    var map: FloatArray? = null
        private set

    @Volatile
    var width = 0
        private set

    @Volatile
    var height = 0
        private set

    @Volatile
    var inferenceMs = 0f
        private set

    @Volatile
    var completed = 0
        private set

    @Volatile
    var dropped = 0
        private set

    val isReady: Boolean get() = estimator.isReady

    fun load(): Boolean = estimator.load()

    /**
     * Hands a frame to the depth thread. Never blocks the caller.
     *
     * Copies into a dedicated buffer first: the pipeline's own buffer is recycled as
     * soon as detection returns, so passing it by reference would let depth read
     * memory that is being overwritten.
     */
    fun request(rgb: ByteArray, w: Int, h: Int): Boolean {
        if (!estimator.isReady) return false
        val need = cfgBufferSize(w, h)
        val buffer = synchronized(buffers) {
            if (buffers.size >= 2) {
                dropped++
                null
            } else {
                buffers.addLast(ByteArray(need))
                buffers.removeLast()
            }
        } ?: return false
        if (buffer.size < need) return false
        System.arraycopy(rgb, 0, buffer, 0, need)
        pendingRgb = buffer
        pendingW = w
        pendingH = h
        if (!busy.compareAndSet(false, true)) {
            dropped++
            synchronized(buffers) { buffers.addLast(buffer) }
            return false
        }
        return try {
            executor.execute {
                val copy = pendingRgb
                val cw = pendingW
                val chh = pendingH
                try {
                    if (copy != null) {
                        val result = estimator.infer(copy, cw, chh)
                        if (result != null) {
                            map = result
                            width = mapWidthOf(result.size)
                            height = result.size / width
                            inferenceMs = estimator.inferenceMs
                            completed++
                        }
                    }
                } catch (e: Throwable) {
                    Log.e(TAG, "Luồng depth lỗi: ${e.message}", e)
                } finally {
                    if (copy != null) synchronized(buffers) { buffers.addLast(copy) }
                    busy.set(false)
                }
            }
            true
        } catch (e: Throwable) {
            busy.set(false)
            Log.e(TAG, "Không giao được việc cho luồng depth: ${e.message}")
            false
        }
    }

    /** Samples the published map. Call from the pipeline thread only. */
    fun sampleFor(detection: Detection, frameW: Int, frameH: Int): Float? {
        val m = map ?: return null
        return estimator.sampleDistance(
            m, detection.x1, detection.y1, detection.x2, detection.y2, frameW, frameH
        )
    }

    fun shutdown() {
        executor.shutdownNow()
        estimator.close()
    }

    private fun cfgBufferSize(w: Int, h: Int) = w * h * 3

    /**
     * Bề rộng bản đồ thực tế. DPT xuất theo bội số patch 14 nên có thể là 294
     * thay vì 299; suy từ kích thước phẳng thay vì giả định.
     */
    private fun mapWidthOf(flat: Int): Int {
        val h = cfg.depthNetHeight
        val byH = flat / h
        return if (byH > 0 && byH * h == flat) byH else cfg.depthNetWidth
    }
}