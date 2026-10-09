package com.example.fcw.camera

import android.util.Log
import androidx.camera.core.ImageProxy
import com.example.fcw.utils.ImageUtils
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Takes frames off the camera and hands them to a worker, closing each one first.
 *
 * The previous version ran the whole pipeline while still holding the frame. Because
 * CameraX cannot deliver a new frame until the current one is closed, a 280 ms
 * inference blocked the camera for 280 ms and pinned the display rate at about 3.5 FPS
 * no matter how much else was reduced. Nothing about the model could fix that.
 *
 * So the split is:
 *   - this class: convert to packed RGB and CLOSE the frame. That is a few milliseconds
 *     and must stay cheap, because it runs at the camera's rate.
 *   - a single worker thread: YOLO, depth, tracking. Slow work happens there, and if it
 *     is still busy the incoming frame is dropped rather than queued.
 *
 * Dropping is deliberate. Queuing would let a backlog grow without bound, and the
 * warning would describe a scene from seconds ago — worse than no warning.
 *
 * Two RGB buffers are pooled and swapped, so a steady stream does not allocate 3.7 MB
 * per frame.
 */
class FrameAnalyzer(
    private val captureAndPrepare: (ImageProxy, ByteArray) -> Boolean,
    private val process: (rgb: ByteArray, width: Int, height: Int, crop: IntArray, captureSeconds: Float) -> Unit,
    private val onFps: ((Float) -> Unit)? = null
) {
    companion object {
        const val TAG = "FCW"
    }

    private val busy = AtomicBoolean(false)
    private val worker = Executors.newSingleThreadExecutor { r ->
        Thread(r, "fcw-pipeline").apply { priority = Thread.NORM_PRIORITY }
    }

    @Volatile private var bufferA: ByteArray = ByteArray(0)
    @Volatile private var bufferB: ByteArray = ByteArray(0)
    @Volatile private var width = 0
    @Volatile private var height = 0
    private val crop = IntArray(4)

    private val frameIndex = AtomicLong(0)
    private val dropped = AtomicLong(0)
    private var windowStart = 0L
    private var windowFrames = 0

    var processingFps: Float = 0f
        private set

    fun submit(image: ImageProxy) {
        // A newer frame makes any frame still waiting irrelevant, so there is no
        // reason to queue one behind a busy worker.
        if (busy.get()) {
            dropped.incrementAndGet()
            closeQuietly(image)
            return
        }

        val w = image.width
        val h = image.height
        if (w <= 0 || h <= 0) {
            closeQuietly(image)
            return
        }
        val need = ImageUtils.rgbBufferSize(w, h)
        if (bufferA.size != need) {
            bufferA = ByteArray(need)
            bufferB = ByteArray(need)
        }

        val target = if (busy.compareAndSet(false, true)) bufferA else {
            dropped.incrementAndGet()
            closeQuietly(image)
            return
        }

        val captureSeconds = image.imageInfo.timestamp / 1_000_000_000f
        val rect = image.cropRect
        crop[0] = rect.left; crop[1] = rect.top; crop[2] = rect.width(); crop[3] = rect.height()
        val snapshot = crop.copyOf()

        val ok = try {
            captureAndPrepare(image, target)
        } catch (e: Throwable) {
            Log.e(TAG, "Chuyển khung hình lỗi: ${e.message}", e)
            false
        }
        // Closed BEFORE any inference. This is the whole point of the class.
        closeQuietly(image)

        if (!ok) {
            busy.set(false)
            return
        }

        width = w
        height = h
        val swap = { val t = bufferA; bufferA = bufferB; bufferB = t }
        // Hand this buffer to the worker and keep the other one for the next capture.
        val forWorker = bufferA
        swap()

        try {
            worker.execute {
                try {
                    process(forWorker, w, h, snapshot, captureSeconds)
                } catch (e: Throwable) {
                    Log.e(TAG, "Pipeline lỗi: ${e.message}", e)
                } finally {
                    busy.set(false)
                    frameIndex.incrementAndGet()
                    windowFrames++
                    val now = System.currentTimeMillis()
                    if (windowStart == 0L) windowStart = now
                    val span = now - windowStart
                    if (span >= 1000) {
                        processingFps = windowFrames * 1000f / span
                        windowStart = now
                        windowFrames = 0
                        onFps?.invoke(processingFps)
                    }
                }
            }
        } catch (e: Throwable) {
            busy.set(false)
            Log.e(TAG, "Không giao được việc cho worker: ${e.message}")
        }
    }

    private fun closeQuietly(image: ImageProxy) {
        try {
            image.close()
        } catch (e: Throwable) {
            Log.w(TAG, "Đóng frame lỗi: ${e.message}")
        }
    }

    fun droppedFrames(): Long = dropped.get()
    fun processedFrames(): Long = frameIndex.get()

    fun reset() {
        frameIndex.set(0)
        dropped.set(0)
        windowStart = 0
        windowFrames = 0
    }

    fun shutdown() {
        worker.shutdownNow()
    }
}