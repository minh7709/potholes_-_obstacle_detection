package com.example.fcw.camera

import android.util.Log
import android.util.Rational
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import android.hardware.camera2.CameraCharacteristics
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import kotlin.coroutines.resumeWithException

/**
 * Owns the CameraX graph.
 *
 * Phase 1 scope: bind Preview + ImageAnalysis so the pipeline has frames and FPS
 * can be measured. Detection is attached in Phase 2.
 *
 * The analysis executor is single threaded on purpose. A multi-threaded executor
 * would let several frames be in flight at once, which builds a backlog whose
 * latency grows without bound when inference is slower than the camera. CameraX
 * also enforces KEEP_ONLY_LATEST, so a slow consumer simply receives the newest
 * frame instead of a queue.
 */
class CameraManager(
    private val context: android.content.Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onFrame: (ImageProxy) -> Unit = {}
) {
    companion object {
        const val TAG = "FCW"

        /**
         * Requested analysis size. 1280x720 matches the Python pipeline's
         * `input.max_width/max_height`, so Android and desktop agree on what the
         * model sees. Anything larger only costs time for pixels that get
         * discarded during resize.
         */
        val ANALYSIS_SIZE = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    android.util.Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()
    }

    /**
     * Shared crop between Preview and ImageAnalysis. Exposed so PreviewView can be
     * given the same one; without it Preview and analysis disagree about the field of
     * view and the overlay has nothing to map against.
     */
    var viewPort: ViewPort? = null
        private set

    private var provider: ProcessCameraProvider? = null

    private var boundCamera: Camera? = null

    /** focalMm / sensorWidthMm, so that focalPx = focalRatio * imageWidthPx. */
    private var focalRatio: Float = 0f

    /** True once the hardware report gave a usable focal length. */
    val hasFocal: Boolean get() = focalRatio > 0f

    /** Focal length in pixels for a frame [imageWidthPx] wide, or 0 if unavailable. */
    fun focalPxFor(imageWidthPx: Int): Float =
        if (focalRatio > 0f && imageWidthPx > 0) focalRatio * imageWidthPx else 0f

    /**
     * Reads the hardware focal length and converts it to pixels.
     *
     * Only the ratio is stored. The pixel value depends on the analysis resolution,
     * which the request does not pin down: 1280x720 is asked for and this device
     * delivered 1280x960, so the width is taken from the frame in hand instead.
     */
    private fun readFocalLength() {
        focalRatio = 0f
        val cam = boundCamera ?: return
        try {
            val info = Camera2CameraInfo.from(cam.cameraInfo)
            val focals = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            val sensor = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val mm = focals?.firstOrNull() ?: return
            val sensorW = sensor?.width ?: return
            if (mm <= 0f || sensorW <= 0f) return
            focalRatio = mm / sensorW
            Log.i(TAG, "Tiêu cự ${"%.2f".format(mm)}mm / cảm biến ${"%.2f".format(sensorW)}mm")
        } catch (e: Throwable) {
            Log.w(TAG, "Không đọc được tiêu cự: ${e.message}")
        }
    }
    private var analysisUseCase: ImageAnalysis? = null
    private var selectorChosen: CameraSelector? = null
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    var isRunning = false
        private set

    /**
     * @param viewWidth width of the view in pixels
     * @param viewHeight height of the view in pixels
     * @param rotation display rotation, so CameraX orients both use cases the same way
     */
    suspend fun start(viewWidth: Int, viewHeight: Int, rotation: Int) {
        val cameraProvider = try {
            provider ?: suspendCoroutine { cont ->
                val future = ProcessCameraProvider.getInstance(context)
                future.addListener({
                    try {
                        cont.resume(future.get())
                    } catch (e: Exception) {
                        cont.resumeWithException(e)
                    }
                }, ContextCompat.getMainExecutor(context))
            }.also { provider = it }
        } catch (e: Exception) {
            Log.e(TAG, "Không lấy được CameraProvider: ${e.message}", e)
            return
        }

        val preview = Preview.Builder()
            .setResolutionSelector(ANALYSIS_SIZE)
            .build()
            .also {
                it.targetRotation = rotation
                it.surfaceProvider = null // bound below by Compose
            }

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(ANALYSIS_SIZE)
            // Critical for latency: drop frames instead of queuing them.
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            // Inference must never touch the main thread.
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .apply { setTargetRotation(rotation) }
            .build()

        // Image ownership: the consumer owns the frame once it is handed over.
        // CameraX will not deliver another frame until this one is closed, so
        // leaking it stalls the camera entirely. It must therefore be closed by
        // whoever finishes using it, not here.
        analysis.setAnalyzer(analysisExecutor) { image ->
            try {
                onFrame(image)
            } catch (e: Exception) {
                Log.e(TAG, "Frame callback lỗi: ${e.message}", e)
                runCatching { image.close() }
            }
        }

        val selector = when {
            cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ->
                CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ->
                CameraSelector.DEFAULT_FRONT_CAMERA
            else -> {
                Log.e(TAG, "Không tìm thấy camera nào")
                return
            }
        }

        // One ViewPort for both use cases: PreviewView crops to the view aspect
        // (FILL_CENTER) while analysis returns the full frame, so without a shared
        // ViewPort the overlay stretches a frame the user never saw.
        val vp = ViewPort.Builder(
            Rational(if (viewWidth > 0) viewWidth else 16, if (viewHeight > 0) viewHeight else 9),
            rotation
        ).setScaleType(ViewPort.FILL_CENTER).build()
        viewPort = vp

        try {
            cameraProvider.unbindAll()
            boundCamera = cameraProvider.bindToLifecycle(
                lifecycleOwner, selector,
                UseCaseGroup.Builder()
                    .setViewPort(vp)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()
            )
            readFocalLength()
            isRunning = true
            Log.i(TAG, "CameraX sẵn sàng ($selector) viewPort=${viewWidth}x$viewHeight rot=$rotation")
        } catch (e: Exception) {
            Log.e(TAG, "bindToLifecycle lỗi: ${e.message}", e)
        }

        this.preview = preview
        analysisUseCase = analysis
        selectorChosen = selector
        lastViewWidth = viewWidth
        lastViewHeight = viewHeight
        lastRotation = rotation
    }

    private var lastViewWidth = 0
    private var lastViewHeight = 0
    private var lastRotation = -1

    /**
     * Re-binds with a new ViewPort after the view size or rotation changes.
     *
     * A ViewPort is immutable, so rotating the phone means building a new one and
     * binding again. Both use cases keep the same analyzer and executor, so no frame
     * is lost beyond the single rebind.
     */
    suspend fun rebind(viewWidth: Int, viewHeight: Int, rotation: Int) {
        val cameraProvider = provider ?: return
        val preview = this.preview ?: return
        val analysis = analysisUseCase ?: return
        val selector = selectorChosen ?: return
        if (viewWidth == lastViewWidth && viewHeight == lastViewHeight &&
            rotation == lastRotation
        ) return

        lastViewWidth = viewWidth
        lastViewHeight = viewHeight
        lastRotation = rotation

        try {
            preview.targetRotation = rotation
            analysis.targetRotation = rotation
            val vp = ViewPort.Builder(
                Rational(if (viewWidth > 0) viewWidth else 16, if (viewHeight > 0) viewHeight else 9),
                rotation
            ).setScaleType(ViewPort.FILL_CENTER).build()
            viewPort = vp

            cameraProvider.unbindAll()
            boundCamera = cameraProvider.bindToLifecycle(
                lifecycleOwner, selector,
                UseCaseGroup.Builder()
                    .setViewPort(vp)
                    .addUseCase(preview)
                    .addUseCase(analysis)
                    .build()
            )
            Log.i(TAG, "rebind ${viewWidth}x$viewHeight rot=$rotation")
        } catch (e: Exception) {
            Log.e(TAG, "rebind lỗi: ${e.message}", e)
        }
    }

    /** Set by the caller so the composable can attach its own surface provider. */
    var preview: Preview? = null
        private set

    fun stop() {
        try {
            provider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "unbind lỗi: ${e.message}")
        }
        analysisExecutor.shutdown()
        isRunning = false
        Log.i(TAG, "CameraX đã dừng")
    }
}