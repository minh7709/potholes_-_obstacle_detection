package com.example.fcw.ui

import kotlin.math.max

/**
 * Geometry shared by every panel the overlay draws.
 *
 * The panels used to take a fixed width in density-independent pixels. That holds
 * only while every line of telemetry happens to be shorter than the guess, and two
 * lines are not: the stage line grows with the frame size and the frame-layout line
 * is a raw byte count ("fmt=1 | p0 row=5120 pix=4 lim=4915200 pos=0"). On a 1080-wide
 * phone those ran past the panel edge and landed on the camera image, where they were
 * unreadable and covered the road.
 *
 * So the width is measured from the text, and a line too long even for the whole
 * canvas is shortened instead of allowed to spill. Panel positions are clamped the
 * same way: a label anchored to a box at the right edge used to be cut off by the
 * screen rather than shifted left to fit.
 *
 * Deliberately free of Canvas and Paint so the arithmetic can be tested on the JVM,
 * where android.graphics is a stub that returns zero for every measurement.
 */
object OverlayLayout {

    /**
     * Width that fits the widest of [lineWidths], including [padX] on both sides.
     *
     * Clamped to [canvasWidth]: a panel wider than the screen is not readable either,
     * it is just off-screen.
     */
    fun panelWidth(lineWidths: List<Float>, padX: Float, canvasWidth: Float): Float {
        val widest = lineWidths.maxOrNull() ?: 0f
        return (widest + padX * 2f).coerceIn(0f, max(canvasWidth, 0f))
    }

    /**
     * Left edge that keeps a [panelWidth] wide panel inside a [canvasWidth] canvas.
     *
     * A panel wider than the canvas starts at zero and overflows to the right, which
     * is the least surprising of the available positions.
     */
    fun clampLeft(desiredLeft: Float, panelWidth: Float, canvasWidth: Float): Float {
        if (canvasWidth <= panelWidth) return 0f
        return desiredLeft.coerceIn(0f, canvasWidth - panelWidth)
    }

    /**
     * [text] shortened, with an ellipsis, until [measure] says it fits [available].
     *
     * Returns the text unchanged when it already fits, and an empty string when not
     * even the ellipsis fits, so a caller never draws a clipped half-character.
     */
    fun fitLine(text: String, available: Float, measure: (String) -> Float): String {
        if (available <= 0f) return ""
        if (measure(text) <= available) return text

        val ellipsis = "…"
        if (measure(ellipsis) > available) return ""

        var low = 0
        var high = text.length - 1
        var best = ""
        while (low <= high) {
            val mid = (low + high) / 2
            val candidate = text.substring(0, mid) + ellipsis
            if (measure(candidate) <= available) {
                best = candidate
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return best
    }

    /**
     * Shortens [lines] to the canvas and returns them with the panel width that fits.
     *
     * The one call a panel needs: lines that fit the whole canvas minus padding are
     * kept, the rest are ellipsized, and the width follows what actually remains.
     */
    fun fittedPanel(
        lines: List<String>,
        padX: Float,
        canvasWidth: Float,
        measure: (String) -> Float
    ): FittedPanel {
        val usable = max(canvasWidth - padX * 2f, 0f)
        val fitted = lines.map { fitLine(it, usable, measure) }
        return FittedPanel(fitted, panelWidth(fitted.map(measure), padX, canvasWidth))
    }

    data class FittedPanel(val lines: List<String>, val width: Float)
}