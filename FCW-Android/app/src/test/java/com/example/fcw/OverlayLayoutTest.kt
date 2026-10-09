package com.example.fcw

import com.example.fcw.ui.OverlayLayout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the overlay panel geometry.
 *
 * The panels used to take a fixed width in dp. That is only correct while every line
 * of telemetry is shorter than the guess, and the frame-layout line is not: "fmt=1 |
 * p0 row=5120 pix=4 lim=4915200 pos=0" on a 1080-wide phone ran past the panel edge
 * and onto the camera image. A screenshot of the running app is what made it obvious,
 * so the arithmetic is pinned here instead.
 *
 * Text measurement is injected because android.graphics is a stub under plain JUnit
 * and returns zero for every measureText call.
 */
class OverlayLayoutTest {

    /** Stands in for Paint.measureText: 7 px per character. */
    private val measure: (String) -> Float = { it.length * 7f }

    @Test
    fun `panel is as wide as its widest line plus padding`() {
        val widths = OverlayLayout.panelWidth(listOf(100f, 260f, 180f), padX = 10f, canvasWidth = 1080f)
        assertEquals(280f, widths, 0.01f)
    }

    @Test
    fun `empty line list still yields a usable panel`() {
        val widths = OverlayLayout.panelWidth(emptyList(), padX = 10f, canvasWidth = 1080f)
        assertEquals(20f, widths, 0.01f)
    }

    @Test
    fun `panel wider than the screen is clamped instead of overflowing`() {
        val widths = OverlayLayout.panelWidth(listOf(2000f), padX = 10f, canvasWidth = 1080f)
        assertEquals(1080f, widths, 0.01f)
    }

    @Test
    fun `the frame layout line from the device fits inside its panel`() {
        val line = "fmt=1 | p0 row=5120 pix=4 lim=4915200 pos=0"
        val panel = OverlayLayout.fittedPanel(listOf(line), padX = 8f, canvasWidth = 1080f, measure)

        assertEquals(line, panel.lines.single())
        assertTrue(
            "panel ${panel.width} phải <= bề rộng canvas",
            panel.width <= 1080f
        )
        assertTrue(
            "text ${measure(line)} + padding phải <= panel ${panel.width}",
            measure(line) + 16f <= panel.width
        )
    }

    @Test
    fun `a line longer than the canvas is ellipsized, not spilled`() {
        val line = "x".repeat(400)
        val panel = OverlayLayout.fittedPanel(listOf(line), padX = 8f, canvasWidth = 1080f, measure)

        assertTrue("phải cắt ngắn", panel.lines.single().length < line.length)
        assertTrue("phải có dấu …", panel.lines.single().endsWith("…"))
        assertTrue(
            "chiều rộng sau khi cắt ${measure(panel.lines.single())} phải <= ${1080f - 16f}",
            measure(panel.lines.single()) <= 1080f - 16f
        )
    }

    @Test
    fun `a line that already fits is returned untouched`() {
        assertEquals("ngắn", OverlayLayout.fitLine("ngắn", 400f, measure))
    }

    @Test
    fun `nothing is drawn when not even an ellipsis fits`() {
        assertEquals("", OverlayLayout.fitLine("bất kỳ", available = 3f, measure = measure))
    }

    @Test
    fun `zero available width yields no text`() {
        assertEquals("", OverlayLayout.fitLine("bất kỳ", available = 0f, measure = measure))
    }

    @Test
    fun `panel near the right edge is shifted left to stay on screen`() {
        val left = OverlayLayout.clampLeft(
            desiredLeft = 1000f, panelWidth = 300f, canvasWidth = 1080f
        )
        assertEquals(780f, left, 0.01f)
        assertTrue("phải nằm trong canvas", left + 300f <= 1080f)
    }

    @Test
    fun `panel near the left edge is not pushed off screen`() {
        val left = OverlayLayout.clampLeft(
            desiredLeft = -40f, panelWidth = 300f, canvasWidth = 1080f
        )
        assertEquals(0f, left, 0.01f)
    }

    @Test
    fun `panel wider than the canvas starts at the left edge`() {
        val left = OverlayLayout.clampLeft(
            desiredLeft = 500f, panelWidth = 1400f, canvasWidth = 1080f
        )
        assertEquals(0f, left, 0.01f)
    }

    @Test
    fun `fitted panel keeps every line, one per input line`() {
        val panel = OverlayLayout.fittedPanel(
            listOf("a", "bb", "ccc"), padX = 4f, canvasWidth = 1080f, measure
        )
        assertEquals(3, panel.lines.size)
        assertEquals(listOf("a", "bb", "ccc"), panel.lines)
    }
}