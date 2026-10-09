package com.example.fcw.ui

import android.util.Log
import androidx.camera.core.Preview
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.fcw.camera.AreaDownscale
import com.example.fcw.camera.CameraManager
import com.example.fcw.camera.FrameAnalyzer
import com.example.fcw.detection.Detection
import com.example.fcw.audio.AudioAlerter
import com.example.fcw.config.FcwConfig
import com.example.fcw.depth.DepthRunner
import com.example.fcw.detection.YoloDetector
import com.example.fcw.risk.LevelStreak
import com.example.fcw.risk.RiskEngine
import com.example.fcw.performance.AdaptiveController
import com.example.fcw.tracking.ByteTracker
import com.example.fcw.tracking.EgoMotionEstimator

/**
 * Phase 2: camera -> best.onnx -> bounding boxes.
 *
 * Purpose of this phase is measurement, not polish. Every later decision depends
 * on knowing what the device sustains: the depth interval in Phase 4, and whether
 * adaptive quality in Phase 11 is even needed. The HUD therefore shows processing
 * FPS, YOLO latency, and dropped frames.
 */
@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    var cameraManager by remember { mutableStateOf<CameraManager?>(null) }
    var preview by remember { mutableStateOf<Preview?>(null) }
    var errorText by remember { mutableStateOf<String?>(null) }

    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var frameW by remember { mutableStateOf(0) }
    var frameH by remember { mutableStateOf(0) }
    var procFps by remember { mutableStateOf(0f) }
    var yoloMs by remember { mutableStateOf(0f) }
    var dropped by remember { mutableStateOf(0L) }
    var processed by remember { mutableStateOf(0L) }
    var modelStatus by remember { mutableStateOf("Đang nạp model...") }
    var detectStage by remember { mutableStateOf(YoloDetector.DetectStage.NO_SESSION) }
    var failStreak by remember { mutableStateOf(0) }
    var frameLayout by remember { mutableStateOf("(chưa có frame)") }
    var viewSize by remember { mutableStateOf(androidx.compose.ui.unit.IntSize.Zero) }

    // LocalConfiguration changes when the activity handles the rotation itself, which
    // is why configChanges lists orientation and screenSize. Reading display.rotation
    // once at startup is not enough: the phone can be turned after that, and rebinding
    // with a stale rotation misaligns the ViewPort and every box with it.
    val configuration = LocalConfiguration.current
    val viewRotation = remember(configuration) { currentDisplayRotation(context) }
    var visibleRegion by remember { mutableStateOf(intArrayOf(0, 0, 0, 0)) }
    var assessments by remember { mutableStateOf(emptyList<RiskEngine.Assessment>()) }
    var primary by remember { mutableStateOf<RiskEngine.Assessment?>(null) }
    var roiBox by remember { mutableStateOf(intArrayOf(0, 0, 0, 0)) }
    var corridorBox by remember { mutableStateOf(intArrayOf(0, 0, 0, 0)) }
    var detectEveryNth by remember { mutableIntStateOf(1) }
    var rawDetections by remember { mutableIntStateOf(0) }
    var frameCounter by remember { mutableIntStateOf(0) }
    var depthMs by remember { mutableFloatStateOf(0f) }
    var depthSamples by remember { mutableIntStateOf(0) }
    var inferCount by remember { mutableIntStateOf(0) }
    var adaptiveTarget by remember { mutableIntStateOf(0) }
    var adaptiveLevel by remember { mutableIntStateOf(0) }
    var adaptiveDepth by remember { mutableIntStateOf(0) }
    var depthEnabled by remember { mutableStateOf(true) }
    var overlayObjects by remember { mutableStateOf(emptyList<OverlayObject>()) }
    // Local copy: `primary` is a delegated property, so its smart cast is refused and
    // the banner must not recompute the confirmation rule from stale state.
    val confirmedPrimary = primary?.takeIf { it.confirmed }
    // 0 until Phase 10 wires the ego-motion estimator in; shown in the HUD so the
    // gap is visible rather than silently reporting a hardcoded zero.
    var egoSpeed by remember { mutableFloatStateOf(0f) }
    var egoQuality by remember { mutableFloatStateOf(0f) }
    var inputFps by remember { mutableStateOf(0f) }
    var showDiagnostics by remember { mutableStateOf(true) }
    var rgbMs by remember { mutableFloatStateOf(0f) }
    var rgbRuns by remember { mutableIntStateOf(0) }
    var droppedFrames by remember { mutableLongStateOf(0L) }

    val cfg = remember { FcwConfig() }
    val detector = remember {
        YoloDetector(
            context = context, inputSize = 640,
            confThreshold = 0.45f, iouThreshold = 0.5f,
            cfg = FcwConfig()
        )
    }
    val tracker = remember { ByteTracker(cfg) }
    val depth = remember { DepthRunner(context, cfg) }
    val ego = remember { EgoMotionEstimator(
        enabled = cfg.egoMotionEnabled,
        backgroundDepthM = cfg.egoBackgroundDepthM,
        minRowOffsetRatio = cfg.egoMinRowOffsetRatio,
        smoothing = cfg.egoSmoothing,
        maxAccelMps = cfg.egoMaxAccelMps,
        maxSpeedMps = cfg.egoMaxSpeedMps
    ) }
    val streak = remember { LevelStreak() }
    val audio = remember { AudioAlerter(context, cfg) }
    val adaptive = remember { AdaptiveController(cfg) }
    // Adaptive target FPS is expressed as a processing interval in inference frames.
    val effectiveDepthInterval = adaptive.depthInterval.coerceAtLeast(1)

    LaunchedEffect(Unit) {
        val ok = detector.load()
        modelStatus = if (ok) {
            "Model OK — ${detector.modelClassCount} lớp — ${detector.providerLabel}"
        } else {
            "LỖI: không nạp được model (xem logcat tag FCW)"
        }
        val depthOk = depth.load()
        audio.init()
        Log.i("FCW", "load() = $ok, depth = $depthOk")
        modelStatus = if (ok) {
            "Model OK — ${detector.modelClassCount} lớp — ${detector.providerLabel}"
        } else {
            "LỖI: không nạp được model (xem logcat tag FCW)"
        } + if (depthOk) " + Depth OK" else " — THIẾU DEPTH"
    }

    val analyzer = remember {
        FrameAnalyzer(
            // Cheap path, runs at the camera rate: convert then close immediately.
            captureAndPrepare = { image, out ->
                detector.readRgbInto(image, out)
            },
            process = { rgb, w, h, cropRect, captureSeconds ->
                frameW = w
                frameH = h
                visibleRegion = cropRect

                // config.yaml input.max_width/max_height: shrink before inference, then
                // scale boxes back so depth sampling, the corridor test and the overlay
                // all keep working in full-frame coordinates.
                val smallBuffer = ByteArray(cfg.maxWidth * cfg.maxHeight * 3)
                var workRgb = rgb
                var workW = w
                var workH = h
                val reduced = AreaDownscale.downscale(
                    rgb, w, h, cfg.maxWidth, cfg.maxHeight, smallBuffer
                )
                if (reduced != null) {
                    workRgb = smallBuffer
                    workW = reduced[0]
                    workH = reduced[1]
                }
                val boxScaleX = w.toFloat() / workW
                val boxScaleY = h.toFloat() / workH

                val fresh = detector.detectRgb(workRgb, workW, workH, cropRect)
                    .map {
                        if (boxScaleX == 1f && boxScaleY == 1f) it
                        else it.copy(
                            x1 = it.x1 * boxScaleX, y1 = it.y1 * boxScaleY,
                            x2 = it.x2 * boxScaleX, y2 = it.y2 * boxScaleY
                        )
                    }
                rawDetections = fresh.size
                yoloMs = detector.inferenceMs
                detectStage = detector.lastStage
                failStreak = detector.failStreak
                frameLayout = detector.frameLayout
                inferCount++

                val roi = RiskEngine.roadRoi(w, h, cfg)
                val corridor = RiskEngine.warningCorridor(w, h, cfg)
                roiBox = roi
                corridorBox = corridor

                // Depth runs on its own cadence. At ~565 ms it cannot share a frame
                // budget with a ~280 ms YOLO, and TrackHistory interpolates across the
                // gaps, so the gaps are cheaper than the staleness.
                if (depthEnabled && inferCount % effectiveDepthInterval == 0) {
                    depth.request(rgb, w, h)
                }
                depthMs = depth.inferenceMs
                depthSamples = depth.completed
                val depthMap = depth.map
                if (depth.dropped > 0 && inferCount % 30 == 0) {
                    Log.w("FCW", "Depth bỏ ${depth.dropped} frame vì luồng depth bận")
                }

                // Ego motion every frame: optical flow is only trackable when the
                // per-frame displacement is small, so it cannot wait for depth.
                if (depthMap != null) {
                    egoSpeed = ego.update(
                        toGray(rgb, w, h), w, h, captureSeconds.toDouble(),
                        depthMap, depth.width, depth.height
                    )
                    egoQuality = ego.quality
                }

                val tracked = tracker.update(fresh, captureSeconds.toDouble())
                detections = tracked
                if (tracked.size != rawDetections) {
                    Log.w("FCW", "Tracker bỏ rơi: model ra $rawDetections, giữ ${tracked.size}")
                }

                // Port of pipeline.py::_distance_for. Measured when depth ran, otherwise
                // interpolated from the last two samples.
                //
                // This gap was the missing piece: depth runs every 5 inference frames,
                // roughly every 2 s, but at 10 m/s a car crosses the 8 m warning
                // distance to the 3.5 m danger distance in 0.45 s. Refreshing distance
                // only when depth ran meant the range half of the risk never saw that
                // transition, so range-based warnings fired late or not at all.
                for (d in tracked) {
                    val id = d.trackId ?: continue
                    val hist = tracker.history[id] ?: continue
                    var handled = false
                    if (depthMap != null) {
                        val dist = depth.sampleFor(d, w, h)
                        if (dist != null) {
                            hist.update(captureSeconds.toDouble(), dist, d, "measured")
                            handled = true
                        }
                    }
                    if (!handled) {
                        val (estimated, source) =
                            hist.interpolatedDistance(captureSeconds.toDouble(), cfg.maxInterpolatedSpeedMps)
                        if (estimated != null) {
                            hist.update(captureSeconds.toDouble(), estimated, d, source)
                        }
                    }
                }
                // Remove the ego's own motion before judging closing speed: a parked car
                // in front of a moving camera has apparent closing equal to ego speed.
                if (depthMap != null) {
                    for (d in tracked) {
                        val id = d.trackId ?: continue
                        tracker.history[id]?.estimateSpeed(egoSpeed)
                    }
                }

                val list = tracked.mapNotNull { d ->
                    val id = d.trackId ?: return@mapNotNull null
                    val hist = tracker.history[id] ?: return@mapNotNull null
                    val base = RiskEngine.assess(
                        hist, d.centerX, d.centerY, w, h, roi, corridor, cfg,
                        consecutive = streak.get(id)
                    )
                    val count = streak.advance(id, base.level)
                    base.copy(
                        consecutiveCount = count,
                        confirmed = base.level != RiskEngine.Level.NORMAL &&
                            hist.framesSeen >= maxOf(cfg.minHits, 1) &&
                            count >= cfg.confirmFrames
                    )
                }
                streak.retainOnly(tracked.mapNotNull { it.trackId }.toSet())
                assessments = list
                primary = RiskEngine.selectPrimaryTarget(list, cfg)
                val confirmed = primary?.takeIf { it.confirmed }
                if (confirmed != null) {
                    audio.alert(confirmed.level == RiskEngine.Level.DANGER, captureSeconds.toDouble())
                }
                overlayObjects = tracked.mapNotNull { d ->
                    val id = d.trackId ?: return@mapNotNull null
                    val a = list.firstOrNull { it.trackId == id } ?: return@mapNotNull null
                    OverlayObject(
                        detection = d, assessment = a, history = tracker.history[id],
                        isPrimary = primary?.trackId == id
                    )
                }
            },
            onFps = { fps ->
                procFps = fps
                inputFps = fps * maxOf(1, detectEveryNth)
                rgbMs = detector.rgbMilliseconds
                rgbRuns = detector.rgbConversionCount
            }
        )
    }

    LaunchedEffect(Unit) {
        try {
            val manager = CameraManager(context, lifecycleOwner) { image ->
                analyzer.submit(image)
            }
            // View size drives the ViewPort crop, so it must be the real laid-out size
            // of the preview, not the analysis resolution.
            val metrics = context.resources.displayMetrics
            val w = if (viewSize.width > 0) viewSize.width else metrics.widthPixels
            val h = if (viewSize.height > 0) viewSize.height else metrics.heightPixels
            val rot = viewRotation
            Log.i("FCW", "start view ${w}x$h rot=$rot")
            manager.start(w, h, rot)
            cameraManager = manager
            preview = manager.preview
        } catch (e: Throwable) {
            errorText = "Không mở được camera: ${e.message}"
            Log.e("FCW", "Camera lỗi: ${e.message}", e)
        }
    }

    // Rebind whenever the view is re-laid out or the phone rotates. A ViewPort is
    // immutable, so the crop cannot be adjusted in place.
    LaunchedEffect(viewSize, viewRotation) {
        val m = cameraManager ?: return@LaunchedEffect
        if (viewSize.width <= 0 || viewSize.height <= 0) return@LaunchedEffect
        val w = viewSize.width
        val h = viewSize.height
        val rot = viewRotation
        Log.i("FCW", "layout view ${w}x$h rot=$rot")
        m.rebind(w, h, rot)
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraManager?.stop()
            analyzer.shutdown()
            audio.release()
            detector.close()
            depth.shutdown()
        }
    }

    // Poll counters once a second; reading them per frame would recompose 30x/s
    // for text that only changes visibly once per second.
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(1000)
            dropped = analyzer.droppedFrames()
            droppedFrames = dropped
            processed = analyzer.processedFrames()
            detectStage = detector.lastStage
            failStreak = detector.failStreak
            frameLayout = detector.frameLayout
            visibleRegion = detector.visibleRegion

            // Adaptive load control. Only sample rates move; TTC thresholds never do.
            adaptive.noteFps(analyzer.processingFps, System.currentTimeMillis() / 1000.0)
            adaptiveTarget = adaptive.targetFps
            adaptiveLevel = adaptive.level
            adaptiveDepth = adaptive.depthInterval
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreview(
            previewProvider = { preview },
            // fillMaxSize() first, then observe the size. Replacing it with
            // onSizeChanged alone left the AndroidView with no size constraint, so it
            // collapsed to 0x0: the preview showed nothing while analysis kept running
            // and the overlay drew normally over a black background.
            modifier = Modifier
                .fillMaxSize()
                .onSizeChanged { viewSize = it }
        )

        // Overlay is a separate layer from the preview surface.
        DetectionOverlay(
            objects = overlayObjects,
            frameWidth = frameW,
            frameHeight = frameH,
            roi = roiBox,
            corridor = corridorBox,
            visibleRegion = visibleRegion,
            modifier = Modifier.fillMaxSize(),
            stats = OverlayStats(
                procFps = procFps,
                inputFps = inputFps,
                frameStep = maxOf(1, detectEveryNth),
                trackCount = assessments.size,
                trackedCount = overlayObjects.size,
                egoSpeedMps = egoSpeed,
                egoQuality = egoQuality,
                yoloMs = yoloMs,
                depthMs = depthMs,
                depthComputed = depthEnabled && depthSamples > 0,
                depthInterval = cfg.depthInterval,
                warnCount = assessments.count { it.level == RiskEngine.Level.WARNING },
                dangerCount = assessments.count { it.level == RiskEngine.Level.DANGER },
                stageLabel = "${detectStage.label} | fail x$failStreak | depth $depthSamples | " +
                    "khung $frameW x$frameH | thấy ${visibleRegion.joinToString(",")}",
                frameLayout = "$frameLayout\n" +
                    "RGB ${"%.0f".format(rgbMs)}ms x$rgbRuns | bỏ $droppedFrames | " +
                    "mục tiêu ${"%.0f".format(1000f / maxOf(procFps, 0.1f))}ms/frame",
                showDiagnostics = showDiagnostics,
                overallLevel = confirmedPrimary?.level ?: RiskEngine.Level.NORMAL
            )
        )

        Row(
            modifier = Modifier.align(Alignment.BottomCenter).padding(10.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            FcwChip("Suy luận 1/$detectEveryNth") {
                detectEveryNth = if (detectEveryNth >= 6) 1 else detectEveryNth + 1
            }
            FcwChip(if (depthEnabled) "Depth 1/${maxOf(1, cfg.depthInterval)}" else "Depth TẮT") {
                depthEnabled = !depthEnabled
            }
            FcwChip(if (showDiagnostics) "Chẩn đoán BẬT" else "Chẩn đoán TẮT") {
                showDiagnostics = !showDiagnostics
            }
        }

        errorText?.let {
            Column(
                modifier = Modifier.align(Alignment.Center).padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(it, color = Color(0xFFFF5555), textAlign = TextAlign.Center)
                Button(onClick = {}) { Text("Camera") }
            }
        }
    }
}

/** Shown when CAMERA permission is missing. */
@Composable
fun PermissionRequest(
    onRequest: () -> Unit,
    onOpenSettings: () -> Unit,
    settingsNeeded: Boolean
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101014))
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Cần quyền CAMERA để chạy cảnh báo va chạm",
            color = Color.White, textAlign = TextAlign.Center, fontSize = 16.sp
        )
        Box(modifier = Modifier.padding(16.dp)) {
            Button(onClick = if (settingsNeeded) onOpenSettings else onRequest) {
                Text(if (settingsNeeded) "Mở Cài đặt" else "Cấp quyền")
            }
        }
    }
}

private fun currentDisplayRotation(context: android.content.Context): Int {
    @Suppress("DEPRECATION")
    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
        context.display?.rotation ?: 0
    } else {
        (context.getSystemService(android.content.Context.WINDOW_SERVICE)
            as android.view.WindowManager).defaultDisplay.rotation
    }
}

/** Small tappable label. Kept local rather than pulled from a UI library. */
@Composable
private fun FcwChip(label: String, onClick: () -> Unit) {
    Text(
        text = label,
        color = Color(0xFF80DEEA),
        fontSize = 11.sp,
        modifier = Modifier
            .background(Color(0xB3000000))
            .clickable(onClick = onClick)
            .padding(horizontal = 8.dp, vertical = 5.dp)
    )
}

/**
 * Luminance from packed RGB, BT.601 weights.
 *
 * Only needed for optical flow; the detector and the depth model work on RGB directly.
 */
private fun toGray(rgb: ByteArray, width: Int, height: Int): FloatArray {
    val out = FloatArray(width * height)
    var i = 0
    var o = 0
    while (i + 2 < rgb.size && o < out.size) {
        val r = rgb[i].toInt() and 0xFF
        val g = rgb[i + 1].toInt() and 0xFF
        val b = rgb[i + 2].toInt() and 0xFF
        out[o++] = 0.299f * r + 0.587f * g + 0.114f * b
        i += 3
    }
    return out
}
