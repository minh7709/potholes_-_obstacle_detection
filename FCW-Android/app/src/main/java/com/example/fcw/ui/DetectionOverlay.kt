package com.example.fcw.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import com.example.fcw.detection.Detection
import com.example.fcw.risk.RiskEngine
import com.example.fcw.tracking.TrackHistory
import kotlin.math.max
import kotlin.math.min

/**
 * One tracked object with everything the overlay has to say about it.
 *
 * Grouped so the drawing code never reaches back into the tracker: the overlay
 * renders a snapshot, and the pipeline keeps running.
 */
data class OverlayObject(
    val detection: Detection,
    val assessment: RiskEngine.Assessment,
    val history: TrackHistory?,
    val isPrimary: Boolean
)

/** Frame-level counters for the HUD panel. */
data class OverlayStats(
    val procFps: Float = 0f,
    val inputFps: Float = 0f,
    val frameStep: Int = 1,
    val trackCount: Int = 0,
    val trackedCount: Int = 0,
    val egoSpeedMps: Float = 0f,
    val egoQuality: Float = 0f,
    val yoloMs: Float = 0f,
    val depthMs: Float = 0f,
    val depthComputed: Boolean = false,
    val depthInterval: Int = 5,
    val warnCount: Int = 0,
    val dangerCount: Int = 0,
    val stageLabel: String = "",
    val frameLayout: String = "",
    val overallLevel: RiskEngine.Level = RiskEngine.Level.NORMAL,
    val showDiagnostics: Boolean = false
)

/**
 * Port of `app/fcw/visualizer.py`.
 *
 * Every component the Python renderer draws is here: road ROI, warning corridor,
 * per-object boxes, the per-object telemetry panels, the DANGER bottom bar, the HUD,
 * the primary-target panel and the big status banner.
 *
 * Colours are converted from the Python BGR tuples to RGB:
 *   NORMAL (0,200,0)BGR, WARNING (0,165,255)BGR, DANGER (0,0,255)BGR,
 *   PRIMARY (255,220,0)BGR
 */
@Composable
fun DetectionOverlay(
    objects: List<OverlayObject>,
    frameWidth: Int,
    frameHeight: Int,
    roi: IntArray = intArrayOf(0, 0, 0, 0),
    corridor: IntArray = intArrayOf(0, 0, 0, 0),
    stats: OverlayStats = OverlayStats(),
    modifier: Modifier = Modifier,
    visibleRegion: IntArray = intArrayOf(0, 0, 0, 0)
) {
    val paints = rememberOverlayPaints()
    if (frameWidth <= 0 || frameHeight <= 0) return

    // visibleRegion is the crop CameraX actually shows; boxes are stored in FULL
    // analysis-frame coordinates, so mapping has to go through it. Stretching the
    // whole frame instead put every box in the wrong place and the wrong size.
    //
    // Validated the same way as in RiskEngine, because a crop rect that does not fit
    // the buffer is measured in some other space and would scale every box by an
    // unknown factor.
    val region = com.example.fcw.risk.RiskEngine.visibleRegion(
        visibleRegion, frameWidth, frameHeight
    )
    val regionLeft = region[0]
    val regionTop = region[1]
    val regionW = region[2]
    val regionH = region[3]

    Canvas(modifier = modifier) {
        val sx = size.width / regionW
        val sy = size.height / regionH
        val mx: (Float) -> Float = { v -> (v - regionLeft) * sx }
        val my: (Float) -> Float = { v -> (v - regionTop) * sy }

        drawRegions(roi, corridor, mx, my, paints)
        for (o in objects) drawObject(o, mx, my, paints)
        drawDangerBar(objects, paints)
        drawHud(stats, paints)
        drawPrimaryPanel(objects, stats, paints)
        drawBanner(stats.overallLevel, paints)
    }
}

// ── Regions ────────────────────────────────────────────────────────────────

private fun DrawScope.drawRegions(
    roi: IntArray, corridor: IntArray,
    mx: (Float) -> Float, my: (Float) -> Float,
    paints: OverlayPaints
) {
    // ROI first so boxes draw on top of it.
    drawGuidedRect(roi, mx, my, paints.roiColour, 1f, "ROAD ROI", paints)

    // The corridor is the only band that may raise an alarm. Drawing it makes it
    // obvious why an object was or was not announced.
    drawGuidedRect(corridor, mx, my, paints.corridorColour, 2f, "CANH BAO", paints)
}

private fun DrawScope.drawGuidedRect(
    rect: IntArray,
    mx: (Float) -> Float, my: (Float) -> Float,
    colour: Color, stroke: Float, title: String, paints: OverlayPaints
) {
    if (rect.size < 4 || rect[2] <= rect[0] || rect[3] <= rect[1]) return
    val l = mx(rect[0].toFloat())
    val t = my(rect[1].toFloat())
    val r = mx(rect[2].toFloat())
    val b = my(rect[3].toFloat())
    if (r - l < 2f || b - t < 2f) return
    drawRect(
        color = colour,
        topLeft = Offset(l, t),
        size = Size(r - l, b - t),
        style = Stroke(width = stroke)
    )
    drawIntoCanvas { canvas ->
        val n = canvas.nativeCanvas
        n.drawText(title, l + 6f * paints.scale, t + 14f * paints.scale, paints.text.apply {
            color = android.graphics.Color.argb(
                (colour.alpha * 255).toInt(), (colour.red * 255).toInt(),
                (colour.green * 255).toInt(), (colour.blue * 255).toInt()
            )
        })
    }
}

// ── Objects ────────────────────────────────────────────────────────────────

private fun DrawScope.drawObject(
    o: OverlayObject,
    mx: (Float) -> Float, my: (Float) -> Float,
    paints: OverlayPaints
) {
    val d = o.detection
    val l = mx(d.x1)
    val t = my(d.y1)
    val r = mx(d.x2)
    val b = my(d.y2)
    if (r <= l || b <= t) return

    val level = o.assessment.level
    val colour = if (o.isPrimary) paints.primaryColour else levelColour(level, paints)
    val thickness = if (o.isPrimary) paints.thick + 2f else paints.thick

    drawRect(
        color = colour,
        topLeft = Offset(l, t),
        size = Size(r - l, b - t),
        style = Stroke(width = thickness)
    )
    drawLabel(labelLines(o, paints), l, t, colour, paints)
}

/**
 * The telemetry lines, ported from `_label_lines`.
 *
 * The two explanatory lines are not decoration. "TTC inf DANGER" looks like a
 * contradiction unless the screen says range triggered it, and a red box reading
 * NORMAL looks like a bug unless the screen says it sits outside the corridor.
 */
private fun labelLines(o: OverlayObject, paints: OverlayPaints): List<Pair<String, Color>> {
    val a = o.assessment
    val h = o.history
    val d = o.detection

    val ttcText = if (!a.ttcValid || a.ttc >= Float.MAX_VALUE) "inf" else "%.2fs".format(a.ttc)
    val distText = h?.distance?.let { "%.1fm".format(it) } ?: "--"
    val speedText = if (h != null && h.speedValid) "%.1fm/s".format(h.closingSpeed) else "--m/s"
    val tid = d.trackId?.toString() ?: "?"
    val marker = when (h?.distanceSource) {
        "measured" -> "M"
        "interpolated" -> "I"
        "held" -> "H"
        else -> ""
    }

    val lines = ArrayList<Pair<String, Color>>(6)
    lines.add("ID:$tid ${d.className} ${"%.2f".format(d.confidence)}" to Color(0xFFE6E6E6))
    lines.add(
        "Dist:$distText${if (marker.isEmpty()) "" else " $marker"} Closing:$speedText" to Color(0xFFE6E6E6)
    )
    lines.add("TTC:$ttcText ${a.level}" to levelColour(a.level, paints))

    if (a.proximityTrigger) {
        lines.add("!! QUÁ GẦN" to levelColour(a.level, paints))
    }
    // Would have warned but sits outside the corridor.
    if (!a.inWarningCorridor && a.levelUnrestricted != RiskEngine.Level.NORMAL) {
        lines.add("ngoài CẢNH BÁO" to Color(0xFFE6E6E6))
    }
    if (o.isPrimary) {
        lines.add("<< MỤC TIÊU >>" to paints.primaryColour)
    }
    return lines
}

/**
 * Dark panel with level-coloured text, above the box when there is room.
 *
 * The panel is not filled with the level colour: an earlier version did that and
 * it hid the very object the box points at.
 */
private fun DrawScope.drawLabel(
    lines: List<Pair<String, Color>>,
    boxX: Float, boxY: Float,
    accent: Color,
    paints: OverlayPaints
) {
    if (lines.isEmpty()) return
    val lineH = 13f * paints.scale
    val padX = 5f * paints.scale
    val height = lineH * lines.size + 4f * paints.scale

    val measure: (String) -> Float = { paints.text.measureText(it) }
    val fitted = OverlayLayout.fittedPanel(lines.map { it.first }, padX, size.width, measure)
    val width = fitted.width

    // Shift left rather than clipping: a label anchored to a box near the right edge
    // used to be truncated by the screen, hiding the telemetry it exists to show.
    val left = OverlayLayout.clampLeft(boxX, width, size.width)
    var top = boxY - height
    if (top < 0f) top = min(size.height - 2f, boxY + 4f * paints.scale)
    top = max(0f, top)
    val bottom = min(size.height - 1f, top + height)
    val right = left + width

    drawRect(color = paints.panelFill, topLeft = Offset(left, top),
        size = Size(right - left, max(bottom - top, 2f)))
    drawRect(color = accent, topLeft = Offset(left, top),
        size = Size(right - left, max(bottom - top, 2f)), style = Stroke(width = 1f))

    drawIntoCanvas { canvas ->
        val n = canvas.nativeCanvas
        fitted.lines.forEachIndexed { i, text ->
            val colour = lines[i].second
            val paint = paints.text
            paint.color = android.graphics.Color.argb(
                (colour.alpha * 255).toInt(),
                (colour.red * 255).toInt(), (colour.green * 255).toInt(),
                (colour.blue * 255).toInt()
            )
            n.drawText(text, left + padX, top + lineH * (i + 1), paint)
        }
    }
}

/** Solid red bar along the bottom when the primary target is in DANGER. */
private fun DrawScope.drawDangerBar(objects: List<OverlayObject>, paints: OverlayPaints) {
    val primary = objects.firstOrNull { it.isPrimary } ?: return
    if (primary.assessment.level != RiskEngine.Level.DANGER) return
    val bar = max(6f, size.height * 0.014f)
    drawRect(
        color = levelColour(RiskEngine.Level.DANGER, paints),
        topLeft = Offset(0f, size.height - bar),
        size = Size(size.width, bar)
    )
}

// ── Panels ─────────────────────────────────────────────────────────────────

private fun DrawScope.drawHud(stats: OverlayStats, paints: OverlayPaints) {
    val lines = ArrayList<String>(4)
    lines.add(
        "Xử lý ${"%.1f".format(stats.procFps)} FPS   Track ${stats.trackCount}/${stats.trackedCount}"
    )
    lines.add(
        "Ego ${"%.1f".format(stats.egoSpeedMps)} m/s (${"%.2f".format(stats.egoQuality)})  " +
            "YOLO ${"%.0f".format(stats.yoloMs)}ms  Depth ${"%.0f".format(stats.depthMs)}ms"
    )
    lines.add(
        "Depth chu kỳ: ${stats.depthInterval}   CẢNH: ${stats.warnCount}  NGUY: ${stats.dangerCount}"
    )
    if (stats.showDiagnostics) {
        // Build stamp, so a screenshot says which APK produced it. Confusing a fresh
        // report with a stale one wastes a whole round trip: the panel layout was
        // already fixed while the screenshot still showed the old overlap.
        lines.add("build ${com.example.fcw.BuildConfig.BUILD_STAMP}")
        // Newlines split into rows: a diagnostic that outgrew one line used to be
        // truncated by the panel instead of being shown.
        lines.addAll(stats.stageLabel.split("\n"))
        lines.addAll(stats.frameLayout.split("\n"))
    }

    val measure: (String) -> Float = { paints.text.measureText(it) }
    val panel = OverlayLayout.fittedPanel(lines, 8f * paints.scale, size.width, measure)

    val panelW = panel.width
    val panelH = (12f + 15f * lines.size) * paints.scale
    drawRect(color = paints.panelFill, topLeft = Offset.Zero, size = Size(panelW, panelH))
    drawRect(color = paints.panelBorder, topLeft = Offset.Zero,
        size = Size(panelW, panelH), style = Stroke(width = 1f))

    drawIntoCanvas { canvas ->
        val n = canvas.nativeCanvas
        val paint = paints.text
        paint.color = android.graphics.Color.rgb(230, 230, 230)
        panel.lines.forEachIndexed { i, text ->
            n.drawText(text, 8f * paints.scale, (13f + 15f * i) * paints.scale, paint)
        }
    }
}

/** Telemetry for the primary target, bottom right. Port of `_draw_primary_panel`. */
private fun DrawScope.drawPrimaryPanel(
    objects: List<OverlayObject>, stats: OverlayStats, paints: OverlayPaints
) {
    val o = objects.firstOrNull { it.isPrimary } ?: return
    val h = o.history
    val a = o.assessment
    val d = o.detection

    val ttcText = if (!a.ttcValid || a.ttc >= Float.MAX_VALUE) "inf" else "%.2fs".format(a.ttc)
    val rows = listOf(
        Triple("Khoảng cách: ${h?.distance?.let { "%.1f".format(it) + " m" } ?: "--"}",
            Color(0xFFFFFFFF), false),
        Triple("Tiến lại: ${if (h != null && h.speedValid) "%.1f m/s".format(h.closingSpeed) else "--"}",
            Color(0xFFFFFFFF), false),
        Triple("TTC: $ttcText", levelColour(a.level, paints), true),
        Triple("TRẠNG THÁI: ${a.level}", levelColour(a.level, paints), true),
        Triple("Nguồn: ${h?.distanceSource ?: "none"}   ID:${d.trackId ?: "?"}",
            Color(0xFFC8C8C8), false)
    )

    val lineH = 18f * paints.scale
    val measure: (String) -> Float = { paints.text.measureText(it) }
    val panel = OverlayLayout.fittedPanel(rows.map { it.first }, 8f * paints.scale, size.width, measure)

    val panelW = panel.width
    val panelH = (10f + lineH * rows.size) * paints.scale
    val left = max(0f, size.width - panelW - 8f * paints.scale)
    val top = max(0f, size.height - panelH - 8f * paints.scale)

    drawRect(color = paints.panelFill, topLeft = Offset(left, top), size = Size(panelW, panelH))
    drawRect(color = paints.primaryColour, topLeft = Offset(left, top),
        size = Size(panelW, panelH), style = Stroke(width = 1f))

    drawIntoCanvas { canvas ->
        val n = canvas.nativeCanvas
        val paint = paints.text
        panel.lines.forEachIndexed { i, text ->
            val colour = rows[i].second
            paint.color = android.graphics.Color.argb(
                (colour.alpha * 255).toInt(), (colour.red * 255).toInt(),
                (colour.green * 255).toInt(), (colour.blue * 255).toInt()
            )
            n.drawText(text, left + 8f * paints.scale, top + (16f + lineH * i) * paints.scale, paint)
        }
    }
}

/** Big centred banner. Port of `_draw_banner`. */
private fun DrawScope.drawBanner(level: RiskEngine.Level, paints: OverlayPaints) {
    if (level == RiskEngine.Level.NORMAL) return
    val raw = if (level == RiskEngine.Level.DANGER) "  DANGER  " else "  WARNING "
    val colour = levelColour(level, paints)
    val paint = paints.banner

    // The banner font is 34sp, so on a small screen the box used to start off the left
    // edge and run off the right one. Centre it, then clamp it back inside.
    val padX = 12f * paints.scale
    val boxW = paint.measureText(raw) + padX * 2f
    val text = OverlayLayout.fitLine(raw, max(0f, size.width - padX * 2f)) { paint.measureText(it) }
    val x = OverlayLayout.clampLeft((size.width - boxW) / 2f, boxW, size.width) + padX
    val y = max(paint.textSize * 1.2f, size.height * 0.12f)

    drawRect(
        color = colour,
        topLeft = Offset(x - padX, y - paint.textSize - 10f * paints.scale),
        size = Size(min(size.width, boxW), paint.textSize + 20f * paints.scale)
    )
    drawIntoCanvas { canvas ->
        canvas.nativeCanvas.drawText(text, x, y, paint)
    }
}

private fun levelColour(level: RiskEngine.Level, paints: OverlayPaints) = when (level) {
    RiskEngine.Level.NORMAL -> paints.normalColour
    RiskEngine.Level.WARNING -> paints.warningColour
    RiskEngine.Level.DANGER -> paints.dangerColour
}

private class OverlayPaints(density: Float) {
    val scale = density
    val thick = 2f * density
    val normalColour = Color(0xFF00C800)      // BGR (0,200,0)
    val warningColour = Color(0xFFFFA500)     // BGR (0,165,255)
    val dangerColour = Color(0xFFFF0000)      // BGR (0,0,255)
    val primaryColour = Color(0xFF00DCFF)     // BGR (255,220,0)
    val roiColour = Color(0xFF00C8FF)         // BGR (255,200,0)
    val corridorColour = Color(0xFFFFD700)    // BGR (0,215,255)
    val panelFill = Color(0xFF191919)
    val panelBorder = Color(0xFF5F5F5F)
    val text = android.graphics.Paint().apply {
        isAntiAlias = true
        textSize = 11f * density
        color = android.graphics.Color.WHITE
    }
    val banner = android.graphics.Paint().apply {
        isAntiAlias = true
        isFakeBoldText = true
        textSize = 34f * density
        color = android.graphics.Color.WHITE
    }
}

@Composable
private fun rememberOverlayPaints(): OverlayPaints {
    val density = androidx.compose.ui.platform.LocalDensity.current.density
    return remember(density) { OverlayPaints(density) }
}