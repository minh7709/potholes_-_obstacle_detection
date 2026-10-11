package com.example.fcw.detection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.example.fcw.utils.ImageUtils
import java.nio.FloatBuffer

/**
 * YOLO inference over ONNX Runtime, loading `assets/models/best.onnx`.
 *
 * Verified properties of that model (MIGRATION_PLAN.md §1):
 *   input  `images`  [1, 3, 640, 640] float32, fixed
 *   output `output0` [1, 19, 8400]
 *   19 = 4 box values + 15 classes; no NMS inside the graph
 *
 * The output shape is read from the model itself, never assumed. Hardcoding 19 and
 * 8400 would silently mis-decode any future model.
 *
 * Class names come from ONNX metadata, not a hardcoded list. The model says
 * `motorbike` where COCO says `motorcycle`; a hardcoded list that disagrees with
 * the model produces a system that runs happily and never detects the class that
 * matters most.
 */
/** Output tensor was not in the expected layout. Distinct from an inference failure
 *  because the model did run and the problem is in how its result is read. */
internal class BadOutput(message: String) : RuntimeException(message)

class YoloDetector(
    private val context: Context,
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.45f,
    private val iouThreshold: Float = 0.5f,
    private val wantedClasses: Set<String> = DEFAULT_WANTED_CLASSES,
    private val preferNnapi: Boolean = true,
    private val cfg: com.example.fcw.config.FcwConfig = com.example.fcw.config.FcwConfig(),
    private val trainingAspect: Float = 4f / 3f,
    /**
     * Split from the core count rather than taken from it. Depth runs at the same
     * time and asked for a share of its own; when both asked for everything, the two
     * sessions together oversubscribed the device and each ran slower than it would
     * have alone.
     */
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
) {
    /** Where [detect] stopped. A returned-empty list with no stage change used to be
     *  indistinguishable between "no vehicles in view" and "never ran at all". */
    enum class DetectStage(val label: String) {
        OK("OK"),
        NO_SESSION("CHƯA NẠP MODEL"),
        BAD_SIZE("KÍCH THƯỚC 0"),
        PIXEL_FORMAT("FORMAT ẢNH KHÔNG HỖ TRỢ"),
        PIXEL_SIZE("KÍCH THƯỚC KHÔNG KHỚP"),
        PIXEL_PLANES("PLANE ẢNH KHÔNG HỢP LỆ"),
        PIXEL_BOUNDS("BUFFER ẢNH BỊ CẮT"),
        PREPROCESS("LỖI TIỀN XỬ LÝ"),
        INFERENCE("LỖI INFERENCE"),
        BAD_OUTPUT("SAI KÍCH THƯỚC OUTPUT"),
        CLASS_MISSING("MODEL THIẾU LỚP CẦN THEO DÕI")
    }

    companion object {
        const val TAG = "FCW"
        /** int8, for the CPU execution provider. */
        const val MODEL_ASSET = "models/best.onnx"

        /** fp16, for NNAPI. int8 graphs are routinely rejected or silently slow on
         *  NNAPI drivers, while fp16 is the format those drivers handle well. */
        const val MODEL_ASSET_NNAPI = "models/best_fp16.onnx"

        /** Every class `best.pt` was trained on. */
        val DEFAULT_WANTED_CLASSES: Set<String> = setOf(
            "motorbike", "motorcycle", "car", "bus", "truck", "bicycle", "person",
            "pothole", "reflective_cone", "warning_column", "spherical_roadblock",
            "ashcan", "stop_sign", "tricycle", "fire_hydrant", "dog"
        )
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    /**
     * Reused RGB buffer. The analyzer closes each CameraX frame before inference
     * starts, so pixels arrive by copy into this array rather than from the frame.
     */
    private var rgbBuffer: ByteArray = ByteArray(0)


    private val preprocessor = YoloPreprocessor(inputSize)
    private val postprocessor = YoloPostprocessor(inputSize, iouThreshold)

    private var inputName: String = ""
    private var outputChannels: Int = 0
    private var outputAnchors: Int = 0
    private var modelW: Int = 0
    private var modelH: Int = 0

    /** Class names indexed by model class id, read from ONNX metadata. */
    private var classNames: Array<String> = emptyArray()
    private var classFilter: IntArray? = null

    // Reused buffers. Allocating these per frame is the easiest way to make GC
    // pauses frequent enough to dominate frame time.
    private var outputBuffer: FloatArray = FloatArray(0)

    /**
     * Converts each frame to RGB once, no matter how many stages ask.
     *
     * The detector, depth and ego motion all need the same pixels. Each used to walk
     * 1.2M pixels with ByteBuffer.get() on its own, which is pure CPU and shows up as
     * frame time the HUD cannot otherwise explain: 0.7 FPS against a 317ms model.
     */
    private val rgbCache = FrameRgbCache()

    /** How many RGB conversions have run. Exposed so the HUD can prove the cache works. */
    val rgbConversionCount: Int get() = if (rgbCount > 0) rgbCount else rgbCache.conversionCount

    /** Milliseconds spent turning a frame into RGB, counting only frames that converted. */
    /** Rolling average of the live capture conversion, see [readRgbInto]. */
    private var rgbMs = 0f
    private var rgbCount = 0

    /** Conversion time for the path the pipeline actually runs. */
    val rgbMilliseconds: Float get() = if (rgbCount > 0) rgbMs else rgbCache.milliseconds

    /** Last [DetectStage] reached. Read by the HUD so a stall is self-diagnosing. */
    var lastStage: DetectStage = DetectStage.NO_SESSION
        private set

    /** Consecutive frames that ended early. Reset on any successful decode. */
    var failStreak: Int = 0
        private set

    /**
     * Writes the exact tensor handed to the model to a JPEG whenever a frame yields
     * nothing.
     *
     * Every previous round of debugging this pipeline burned a device round trip on
     * guesses about image format, rotation and crop rect. Writing the input down ends
     * that: the file shows whether the model is receiving an upright, correctly
     * letterboxed RGB image or something structurally wrong, with no interpretation
     * needed.
     */
    var dumpEmptyInput: Boolean = true

    @Volatile
    var lastDumpPath: String = "(chưa xuất)"
        private set

    /** Raw frame layout from the last call, shown on the HUD. */
    var frameLayout: String = "(chưa có frame)"
        private set

    /**
     * Visible sub-rectangle of the analysed frame: left, top, width, height.
     *
     * CameraX crops the analysis buffer to match the ViewPort shared with Preview, so
     * the analysed frame is wider than what the user sees. The overlay must map
     * through this or every box lands in the wrong place.
     */
    var visibleRegion: IntArray = intArrayOf(0, 0, 0, 0)
        private set

    var inferenceMs: Float = 0f
        private set

    /** Which execution provider is actually running, shown on the HUD. */
    var providerLabel: String = "(chưa nạp)"
        private set

    /** Rectangle actually fed to the model, for the HUD. */
    var fedRect: IntArray = intArrayOf(0, 0, 0, 0)
        private set

    /** Both measured times from the head-to-head run, for the HUD. */
    var benchNnapiMs: Float = -1f
        private set
    var benchCpuMs: Float = -1f
        private set

    /**
     * Both providers are built and timed on a real frame, then the slower one is
     * closed. Measured on this device: NNAPI+fp16 328ms versus CPU+int8 224ms, so
     * assuming NNAPI is the fast path would have made the app slower.
     */
    private var nnapiSession: OrtSession? = null
    private var cpuSession: OrtSession? = null
    private var activeSession: OrtSession? = null

    /**
     * Which session inference is actually using, as a short HUD string.
     *
     * NNAPI is requested first and falls back to CPU silently, so a device can end up
     * running the int8 CPU graph without ever showing it. That matters because the
     * int8 CPU graph is built from ConvInteger, the same op ORT Android has no kernel
     * for in the depth model. On-screen is the only place this is visible without a
     * cable, so it belongs here rather than in logcat.
     */
    val sessionLabel: String
        get() = when {
            // Both candidate sessions are closed once the benchmark picks a winner
            // (see runInference), so their fields are null even while inference runs
            // fine on the survivor. Reading them reported "not loaded" on a healthy
            // app. providerLabel is what actually records the outcome.
            providerLabel.isNotBlank() -> providerLabel
            nnapiSession != null || cpuSession != null -> "đang benchmark"
            else -> "CHƯA NẠP"
        }
    private var benchDone = false

    /** Inference times for the first few runs, to separate warmup from steady state. */
    private val warmupMs = ArrayList<Float>(8)
    private var warmupLogged = false
    var inferenceCount: Long = 0L
        private set
    val modelClassCount: Int get() = classNames.size
    val isReady: Boolean get() = session != null

    /**
     * Loads the model and its metadata.
     *
     * Never throws: a failed load disables detection but leaves the camera
     * running, which is far easier to debug than a crash.
     */
    fun load(): Boolean {
        return try {
            // Try NNAPI with fp16 first, then CPU with int8. A device that refuses to
            // build the NNAPI session must still run, so the fallback is not optional.
            if (preferNnapi) nnapiSession = tryCreate(context, MODEL_ASSET_NNAPI, threads, true)
            cpuSession = tryCreate(context, MODEL_ASSET, threads, false)
            val session = nnapiSession ?: cpuSession
            if (session == null) {
                Log.e(TAG, "Không tạo được session nào")
                return false
            }
            // Both built when available: the winner is decided by measurement.
            this.session = session
            nnapiSession?.let { if (it !== session) activeSession = it }
            cpuSession?.let { if (it !== session) activeSession = it }
            providerLabel = "đang đo…"

            // ORT 1.20 exposes shape through TensorInfo, reached via NodeInfo.getInfo().
            val inNode = session!!.inputInfo.values.first()
            val outNode = session!!.outputInfo.values.first()
            inputName = inNode.name

            val inShape = (inNode.info as? TensorInfo)?.shape
            modelH = inShape?.getOrNull(2)?.takeIf { it > 0 }?.toInt() ?: inputSize
            modelW = inShape?.getOrNull(3)?.takeIf { it > 0 }?.toInt() ?: inputSize
            if (modelH != inputSize || modelW != inputSize) {
                // The preprocessor owns the tensor at inputSize x inputSize. If that
                // does not match the graph, feeding it would silently produce garbage,
                // so refuse instead of running a pipeline that cannot be right.
                Log.e(TAG, "Model input ${modelW}x$modelH != $inputSize — không chạy được")
                return false
            }

            val outShape = (outNode.info as? TensorInfo)?.shape
            if (outShape == null) {
                Log.e(TAG, "Output không phải tensor — không decode được")
                return false
            }
            outputChannels = outShape.getOrNull(1)?.toInt() ?: 0
            outputAnchors = outShape.getOrNull(2)?.toInt() ?: 0
            if (outputChannels <= 4 || outputAnchors <= 0) {
                Log.e(TAG, "Output shape không hợp lệ: ${outShape.joinToString("x")}")
                return false
            }
            if (!postprocessor.configure(outputChannels, outputAnchors)) {
                Log.e(TAG, "Postprocessor không cấu hình được")
                return false
            }
            Log.i(TAG, "ONNX vào='$inputName' ${inShape?.joinToString("x")} " +
                    "ra=[1,$outputChannels,$outputAnchors] " +
                    "=> ${postprocessor.classCount()} lớp")

            classNames = readClassNames()
            classFilter = resolveClassFilter()
            Log.i(TAG, "Model có ${classNames.size} lớp: ${classNames.joinToString(", ")}")
            Log.i(TAG, "Lọc ${classFilter?.size ?: 0} lớp: ${describeFilter()}")
            outputBuffer = FloatArray(outputChannels * outputAnchors)
            true
        } catch (e: Throwable) {
            Log.e(TAG, "Không load được ONNX: ${e.message}", e)
            session = null
            classNames = emptyArray()
            false
        }
    }

    /**
     * Builds one session, returning null instead of throwing.
     *
     * NNAPI availability is device and driver specific, and a session that cannot be
     * created must not take the whole app down.
     */
    private fun tryCreate(
        context: Context,
        asset: String,
        threads: Int,
        nnapi: Boolean
    ): OrtSession? = try {
        val bytes = context.assets.open(asset).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (nnapi) {
                addNnapi()
                // Partitioning failures must not abort the run.
                addConfigEntry("session.disable_cpu_ep_fallback", "0")
            }
        }
        val s = env.createSession(bytes, options)
        Log.i(TAG, "Session OK: $asset ${bytes.size / 1024} KB, $threads luồng, nnapi=$nnapi")
        s
    } catch (e: Throwable) {
        Log.w(TAG, "Không tạo được session từ $asset: ${e.message}")
        null
    }

    /**
     * Reads class names from ONNX metadata.
     *
     * Ultralytics writes a Python-repr dict such as "{0: 'car', 1: 'person'}".
     * It is parsed with a regex rather than eval: evaluating content that arrived
     * inside a model file would execute whatever it contained.
     */
    private fun readClassNames(): Array<String> {
        val meta = session?.metadata?.customMetadata
        val raw = meta?.get("names")
        if (raw.isNullOrBlank()) {
            Log.e(TAG, "ONNX thiếu metadata 'names'. KHÔNG đoán danh sách lớp.")
            return emptyArray()
        }
        val entries = Regex("(\\d+)\\s*:\\s*'([^']*)'").findAll(raw)
            .mapNotNull { m ->
                val id = m.groups[1]?.value?.toIntOrNull()
                val name = m.groups[2]?.value
                if (id == null || name == null) null else id to name
            }
            .toList()
        if (entries.isEmpty()) {
            Log.e(TAG, "Không parse được metadata names: $raw")
            return emptyArray()
        }
        val size = entries.maxOf { it.first } + 1
        val arr = Array(size) { "class_$it" }
        for ((index, name) in entries) if (index in 0 until size) arr[index] = name
        return arr
    }

    /**
     * Resolves which class ids to keep.
     *
     * Matching is alias-aware, mirroring the Python side, so config written as
     * "motorcycle" still resolves against a model that says "motorbike".
     */
    private fun resolveClassFilter(): IntArray? {
        if (classNames.isEmpty()) return null
        val wanted = wantedClasses.map(::normalise).toSet()
        val ids = classNames.indices.filter { normalise(classNames[it]) in wanted }
        if (ids.isEmpty()) {
            Log.e(TAG, "Không lớp nào khớp yêu cầu. Model có: ${classNames.toList()}")
        }
        return if (ids.isEmpty()) null else ids.toIntArray()
    }

    private fun describeFilter(): String =
        classFilter?.joinToString(", ") { "$it=${classNames[it]}" } ?: "(không lọc)"

    private fun normalise(name: String): String {
        val base = name.trim().lowercase().replace("_", " ").replace("-", " ")
        val squashed = base.split(" ").filter { it.isNotEmpty() }.joinToString(" ")
        return when (squashed) {
            "motorcycle", "motor bike" -> "motorbike"
            "potholes", "manhole" -> "pothole"
            "trash can", "garbage can" -> "ashcan"
            "cone" -> "reflective cone"
            "bollard" -> "warning column"
            "hydrant" -> "fire hydrant"
            else -> squashed
        }
    }

    /**
     * Runs detection on one CameraX frame.
     *
     * The caller must keep the frame open until this returns.
     *
     * @return detections in source pixels, largest first. Empty on any failure: a
     *         bad frame must not take the pipeline down.
     */
    fun detect(image: androidx.camera.core.ImageProxy): List<Detection> {
        val s = session ?: return fail(DetectStage.NO_SESSION)
        if (image.width <= 0 || image.height <= 0) return fail(DetectStage.BAD_SIZE)

        frameLayout = ImageUtils.describe(image)
        val crop = image.cropRect
        visibleRegion = intArrayOf(crop.left, crop.top, crop.width(), crop.height())

        // Shares the cache with rgbOf(): depth and ego motion ask for this same frame,
        // and converting it once instead of three times is most of the frame budget.
        // The Result is kept rather than just a ByteArray so a failure still reports
        // WHICH part of the frame was wrong, which is the whole point of the HUD.
        when (val read = readRgb(image)) {
            is ImageUtils.Result.Ok -> Unit
            is ImageUtils.Result.Failed -> return fail(
                when (read.reason) {
                    ImageUtils.Reason.SIZE_MISMATCH -> DetectStage.PIXEL_SIZE
                    ImageUtils.Reason.NO_PLANES,
                    ImageUtils.Reason.PLANE_COUNT,
                    ImageUtils.Reason.MALFORMED -> DetectStage.PIXEL_PLANES
                    ImageUtils.Reason.TRUNCATED -> DetectStage.PIXEL_BOUNDS
                    ImageUtils.Reason.UNSUPPORTED_FORMAT -> DetectStage.PIXEL_FORMAT
                },
                "${read.reason.label}: ${read.detail}"
            )
        }

        return runInference(image.width, image.height)
    }

    private fun runInference(frameW: Int, frameH: Int): List<Detection> {
        val s = session ?: return fail(DetectStage.NO_SESSION)
        if (frameW <= 0 || frameH <= 0) return fail(DetectStage.BAD_SIZE)

        try {
            val rect = modelFittedRect(frameW, frameH)
            fedRect = rect
            preprocessor.fill(rgbBuffer, frameW, frameH, rect[0], rect[1], rect[2], rect[3])
        } catch (e: Throwable) {
            return fail(DetectStage.PREPROCESS, "${e.javaClass.simpleName}: ${e.message}")
        }

        val expected = outputChannels * outputAnchors
        val raw: FloatArray = try {
            val shape = longArrayOf(1, 3, modelH.toLong(), modelW.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(preprocessor.inputTensor()), shape).use { input ->
                val result = runOne(activeSession ?: s, input)

                if (!benchDone && nnapiSession != null && cpuSession != null) {
                    benchDone = true
                    val nn = nnapiSession
                    val cpu = cpuSession
                    benchNnapiMs = median(runTimed(nn!!, input), "NNAPI")
                    benchCpuMs = median(runTimed(cpu!!, input), "CPU")
                    val keepNnapi = benchNnapiMs <= benchCpuMs
                    activeSession = if (keepNnapi) nn else cpu
                    session = activeSession
                    providerLabel = (if (keepNnapi) "NNAPI + fp16" else "CPU + int8") +
                        " — thắng (NNAPI ${"%.0f".format(benchNnapiMs)}ms vs " +
                        "CPU ${"%.0f".format(benchCpuMs)}ms)"
                    Log.i(TAG, providerLabel)
                    if (keepNnapi) closeQuietly(cpu) else closeQuietly(nn)
                    nnapiSession = null
                    cpuSession = null
                }

                inferenceMs = result.ms
                inferenceCount++
                result.flat
            }
        } catch (e: BadOutput) {
            return fail(DetectStage.BAD_OUTPUT, e.message ?: "")
        } catch (e: Throwable) {
            return fail(DetectStage.INFERENCE, "${e.javaClass.simpleName}: ${e.message}")
        }

        if (raw.size != expected) {
            return fail(DetectStage.BAD_OUTPUT, "nhận ${raw.size} float, cần $expected")
        }
        if (outputBuffer.size != raw.size) outputBuffer = FloatArray(raw.size)
        System.arraycopy(raw, 0, outputBuffer, 0, raw.size)

        val all = postprocessor.decode(outputBuffer, classNames, confThreshold, preprocessor)

        // classes.min_area_ratio from config.yaml. Missing here, so noise boxes were
        // reaching the tracker and the corridor test. Python applies this in
        // detector.py::_parse before anything downstream sees the box.
        val minArea = cfg.minAreaRatio * frameW.toFloat() * frameH.toFloat()
        val kept = if (minArea > 0f) all.filter { it.areaPx >= minArea } else all

        if (kept.isEmpty() && dumpEmptyInput) {
            dumpInputTensor()
        }
        if (failStreak != 0 || lastStage != DetectStage.OK) {
            Log.i(TAG, "detect() chạy lại bình thường sau $failStreak frame lỗi, thấy ${all.size} box")
        }
        failStreak = 0
        lastStage = DetectStage.OK

        // A model that loaded but cannot name the classes we must watch is a
        // silent failure: it runs, it draws boxes, it never warns about a vehicle.
        val filter = classFilter ?: run {
            Log.e(TAG, "DỪNG phát hiện: không lớp nào khớp. Cần: $wantedClasses")
            return emptyList()
        }
        return all.filter { it.classId in filter }
    }

    /**
     * Records a failed frame and logs it.
     *
     * Logging is throttled to a stage change plus once per 60 frames: a per-frame
     * failure logs hundreds of lines a second and buries everything useful.
     */
    private fun fail(stage: DetectStage, detail: String = ""): List<Detection> {
        val previous = lastStage
        lastStage = stage
        failStreak++
        if (previous != stage || failStreak % 60 == 1) {
            val tag = if (stage == DetectStage.PIXEL_FORMAT) Log.WARN else Log.ERROR
            Log.println(
                tag, TAG, "detect() dừng ở ${stage.label} x${failStreak}" +
                        if (detail.isEmpty()) "" else " — $detail"
            )
        }
        return emptyList()
    }

    /**
     * Crops the analysis buffer to the shape the model was trained on.
     *
     * Deliberately derived from the BUFFER's own dimensions, never from
     * ImageProxy.cropRect. Crop rect semantics changed across CameraX versions: it may
     * describe a region of the delivered buffer, or the region inside the pre-crop
     * original. Feeding a crop rect measured in the wrong frame made the preprocessor
     * read the wrong strip of memory, and detection collapsed to nothing in BOTH
     * orientations the moment the shared ViewPort was introduced. Crop rect is still
     * used for drawing, where a wrong value costs an offset rather than every box.
     *
     * The crop is what makes portrait usable: letterboxing a tall buffer into 640x640
     * fills only 360 of the 640 columns and shrinks every object by about 1.8x.
     * Measured on 10 frames of real footage, portrait found 11 objects with plain
     * letterboxing and 15 after cropping to 4:3.
     *
     * @return left, top, width, height in buffer pixels.
     */
    private fun modelFittedRect(frameW: Int, frameH: Int): IntArray {
        if (frameW <= 0 || frameH <= 0) return intArrayOf(0, 0, maxOf(frameW, 1), maxOf(frameH, 1))
        val aspect = frameW.toFloat() / frameH.toFloat()
        return if (aspect > trainingAspect) {
            val cw = (frameH * trainingAspect).toInt().coerceIn(1, frameW)
            intArrayOf((frameW - cw) / 2, 0, cw, frameH)
        } else {
            val ch = (frameW / trainingAspect).toInt().coerceIn(1, frameH)
            intArrayOf(0, (frameH - ch) / 2, frameW, ch)
        }
    }

    /** Writes the current input tensor to filesDir so it can be pulled and inspected. */
    private fun dumpInputTensor() {
        try {
            val size = inputSize
            val plane = size * size
            val pixels = IntArray(size * size)
            val t = preprocessor.inputTensor()
            for (i in 0 until size * size) {
                val r = (t[i] * 255f).toInt().coerceIn(0, 255)
                val g = (t[plane + i] * 255f).toInt().coerceIn(0, 255)
                val b = (t[2 * plane + i] * 255f).toInt().coerceIn(0, 255)
                pixels[i] = (r shl 16) or (g shl 8) or b
            }
            val bitmap = android.graphics.Bitmap.createBitmap(
                pixels, size, size, android.graphics.Bitmap.Config.ARGB_8888
            )
            val file = java.io.File(context.filesDir, "fcw_input.jpg")
            java.io.FileOutputStream(file).use { out ->
                bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, out)
            }
            bitmap.recycle()
            lastDumpPath = file.absolutePath
        } catch (e: Throwable) {
            Log.w(TAG, "Không xuất được tensor: ${e.message}")
        }
    }

    private data class Run(val flat: FloatArray, val ms: Float)

    private fun runOne(s: OrtSession, input: OnnxTensor): Run {
        val start = System.nanoTime()
        val flat = s.run(mapOf(inputName to input)).use { outs ->
            val out = outs[0] as? OnnxTensor
                ?: throw BadOutput("output[0] là ${outs[0]?.javaClass?.simpleName}, không phải tensor")
            val view = out.floatBuffer.duplicate().apply { rewind() }
            val arr = FloatArray(view.remaining())
            view.get(arr, 0, arr.size)
            arr
        }
        return Run(flat, (System.nanoTime() - start) / 1_000_000f)
    }

    private fun runTimed(s: OrtSession, input: OnnxTensor): List<Float> {
        val times = ArrayList<Float>(3)
        repeat(4) { pass ->
            val start = System.nanoTime()
            s.run(mapOf(inputName to input)).use { it[0] }
            val ms = (System.nanoTime() - start) / 1_000_000f
            if (pass > 0) times.add(ms)   // pass 0 is warmup
        }
        return times
    }

    private fun median(values: List<Float>, tag: String): Float {
        val sorted = values.sorted()
        return if (sorted.size % 2 == 1) sorted[sorted.size / 2]
        else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
    }

    private fun closeQuietly(s: OrtSession?) {
        try {
            s?.close()
        } catch (e: Throwable) {
            Log.w(TAG, "Đóng session phụ lỗi: ${e.message}")
        }
    }

    /**
     * The current frame as tightly packed RGB, filling the buffer if needed.
     *
     * Depth runs on a different cadence from detection, so it cannot rely on the
     * buffer already holding this frame. Reusing the same buffer avoids a second
     * per-frame conversion of 3.7 MB.
     *
     * @return null when the frame could not be read.
     */
    /**
     * Runs detection on already-captured RGB.
     *
     * The analyzer closes each CameraX frame before inference starts, so this entry
     * point works on a copy rather than on the frame. `visibleRegion` is passed in
     * because crop rect is a property of the frame, not of the buffer.
     */
    fun detectRgb(
        rgb: ByteArray,
        width: Int,
        height: Int,
        cropRect: IntArray
    ): List<Detection> {
        val session = session ?: return fail(DetectStage.NO_SESSION)
        if (width <= 0 || height <= 0) return fail(DetectStage.BAD_SIZE)
        visibleRegion = cropRect.copyOf()

        val need = ImageUtils.rgbBufferSize(width, height)
        if (rgbBuffer.size != need) rgbBuffer = ByteArray(need)
        System.arraycopy(rgb, 0, rgbBuffer, 0, need)
        return runInference(width, height)
    }

    fun rgbOf(image: androidx.camera.core.ImageProxy): ByteArray? {
        if (image.width <= 0 || image.height <= 0) return null
        return when (val read = readRgb(image)) {
            is ImageUtils.Result.Ok -> rgbCache.peek()
            is ImageUtils.Result.Failed -> null
        }
    }

    /**
     * This frame as tightly packed RGB, converted at most once per frame.
     *
     * Keyed on the capture timestamp so a repeat call inside one frame is free and the
     * NEXT frame always reconverts. A stale buffer fed to depth would put a measured
     * distance on the wrong scene, so a frame that cannot be identified falls back to
     * converting rather than trusting the cache.
     */
    /**
     * Converts a frame into a caller-supplied buffer. Used by the analyzer, which must
     * close the frame immediately and cannot hand an ImageProxy to the worker.
     */
    /**
     * Converts straight into [out] and times it.
     *
     * The timers used to live only on the readRgb path, and the live pipeline calls
     * this one, so the HUD showed "RGB 0ms x0" and the conversion ran unmeasured. That
     * is the stage most likely to dominate: it walks all 1.23M pixels of 1280x960 one
     * at a time in Kotlin, then a second full pass shrinks it for inference.
     */
    fun readRgbInto(image: androidx.camera.core.ImageProxy, out: ByteArray): Boolean {
        if (image.width <= 0 || image.height <= 0) return false
        if (out.size < ImageUtils.rgbBufferSize(image.width, image.height)) return false
        val t0 = SystemClock.elapsedRealtimeNanos()
        val ok = ImageUtils.toRgb(image, out) is ImageUtils.Result.Ok
        rgbMs = rgbMs * 0.8f + (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f * 0.2f
        rgbCount++
        return ok
    }

    private fun readRgb(image: androidx.camera.core.ImageProxy): ImageUtils.Result =
        rgbCache.get(image.width, image.height, frameStamp(image)) { buf ->
            ImageUtils.toRgb(image, buf)
        }

    /**
     * Capture timestamp used to identify a frame, or [FrameRgbCache.NO_FRAME] when the
     * camera does not provide one, which disables the cache rather than risking another
     * frame's pixels.
     */
    private fun frameStamp(image: androidx.camera.core.ImageProxy): Long {
        val ts = try {
            image.imageInfo.timestamp
        } catch (e: Throwable) {
            0L
        }
        return if (ts > 0L) ts else FrameRgbCache.NO_FRAME
    }

    fun close() {
        try {
            session?.close()
        } catch (e: Throwable) {
            Log.w(TAG, "Đóng session lỗi: ${e.message}")
        }
        session = null
        activeSession = null
        closeQuietly(nnapiSession)
        closeQuietly(cpuSession)
        nnapiSession = null
        cpuSession = null
    }
}