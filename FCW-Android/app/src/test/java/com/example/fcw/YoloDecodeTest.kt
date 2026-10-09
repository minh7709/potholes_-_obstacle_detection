package com.example.fcw

import com.example.fcw.detection.YoloPostprocessor
import com.example.fcw.detection.YoloPreprocessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Validates the decode path against values captured from the real ONNX model.
 *
 * The preprocessor's letterbox geometry and the postprocessor's NMS are the two
 * places where a mistake produces plausible-looking but wrong output, and neither
 * is visible without a device. These fixtures come from an actual
 * `best_int8.onnx` forward pass on real footage, so a regression here means the
 * boxes on screen would be wrong.
 */
class YoloDecodeTest {

    private val names = arrayOf(
        "stop_sign", "person", "bicycle", "bus", "truck", "car", "motorbike",
        "reflective_cone", "ashcan", "warning_column", "spherical_roadblock",
        "pothole", "dog", "tricycle", "fire_hydrant"
    )

    // From a real forward pass: source 1280x720, model 640, padY 140.
    // anchors (class, score, cx, cy, w, h)
    private val anchors = listOf(
        intArrayOf(5, 3412), intArrayOf(0, 2472), intArrayOf(0, 2471),
        intArrayOf(5, 7266), intArrayOf(0, 2552)
    )
    private val cx = mapOf(3412 to 422.556f, 2472 to 576.253f, 2471 to 576.312f,
        7266 to 422.620f, 2552 to 576.049f)
    private val cy = mapOf(3412 to 337.758f, 2472 to 247.104f, 2471 to 246.963f,
        7266 to 338.105f, 2552 to 247.500f)
    private val bw = mapOf(3412 to 46.629f, 2472 to 38.163f, 2471 to 38.225f,
        7266 to 45.657f, 2552 to 38.382f)
    private val bh = mapOf(3412 to 25.431f, 2472 to 50.424f, 2471 to 50.392f,
        7266 to 26.341f, 2552 to 50.330f)
    private val score = mapOf(3412 to 0.8648f, 2472 to 0.8571f, 2471 to 0.8360f,
        7266 to 0.8330f, 2552 to 0.8304f)

    private val channels = 19
    private val anchorCount = 8400

    /** Builds a synthetic head output holding only the captured anchors. */
    private fun buildTensor(): FloatArray {
        val t = FloatArray(channels * anchorCount)
        for ((clsIdx, a) in anchors) {
            t[0 * anchorCount + a] = cx.getValue(a)
            t[1 * anchorCount + a] = cy.getValue(a)
            t[2 * anchorCount + a] = bw.getValue(a)
            t[3 * anchorCount + a] = bh.getValue(a)
            t[(4 + clsIdx) * anchorCount + a] = score.getValue(a)
        }
        return t
    }

    @Test
    fun `shape is read from the model not assumed`() {
        val post = YoloPostprocessor(640, 0.5f)
        assertTrue(post.configure(19, 8400))
        assertEquals(15, post.classCount())
        assertTrue("must reject a shape with no class channels", !post.configure(4, 8400))
    }

    @Test
    fun `letterbox geometry matches the export`() {
        val pre = YoloPreprocessor(640)
        // 1280x720 -> scale 0.5, resized 640x360, padded 140 top and bottom.
        pre.setGeometryForTest(srcWidth = 1280, srcHeight = 720)
        assertEquals(0.5f, pre.scale, 0.001f)
        assertEquals(0, pre.padX)
        assertEquals(140, pre.padY)
    }

    @Test
    fun `decoded boxes match the python reference`() {
        val pre = YoloPreprocessor(640)
        pre.setGeometryForTest(1280, 720)
        val post = YoloPostprocessor(640, 0.5f)
        post.configure(19, 8400)

        val dets = post.decode(buildTensor(), names, 0.45f, pre)
        // Anchor 3412 (car) and 2472 (stop_sign) survive; the duplicates
        // 2471/7266/2552 must be suppressed by NMS.
        assertEquals("NMS phải gộp 2471/7266/2552 vào box chính", 2, dets.size)

        val car = dets.first { it.className == "car" }
        val stop = dets.first { it.className == "stop_sign" }

        // Python: x1 = cx - w/2 = 422.556 - 23.3145 = 399.2415 (model px)
        //          x2 = cx + w/2 = 445.8705
        // un-letterbox divides by the 0.5 scale: 798.5 and 891.7 (source px)
        assertEquals("car", car.className)
        assertEquals(5, car.classId)
        assertEquals(798.5f, car.x1, 1.0f)
        assertEquals(891.7f, car.x2, 1.0f)
        assertEquals(0.8648f, car.confidence, 0.001f)

        // stop_sign: y1 = 247.104 - 25.212 = 221.892; un-letterbox (221.892-140)/0.5
        assertEquals("stop_sign", stop.className)
        assertEquals(0, stop.classId)
        assertEquals(163.7f, stop.y1, 1.0f)
    }

    @Test
    fun `low confidence anchors are dropped`() {
        val pre = YoloPreprocessor(640)
        pre.setGeometryForTest(1280, 720)
        val post = YoloPostprocessor(640, 0.5f)
        post.configure(19, 8400)
        assertEquals(0, post.decode(buildTensor(), names, 0.95f, pre).size)
    }

    @Test
    fun `nms does not suppress across different classes`() {
        // Two boxes overlapping heavily but of different classes must both survive:
        // a motorcycle behind a car still has to be warned about.
        val t = FloatArray(19 * 8400)
        t[0 * 8400 + 0] = 320f; t[1 * 8400 + 0] = 320f
        t[2 * 8400 + 0] = 200f; t[3 * 8400 + 0] = 200f
        t[(4 + 5) * 8400 + 0] = 0.9f   // car
        t[0 * 8400 + 1] = 322f; t[1 * 8400 + 1] = 322f
        t[2 * 8400 + 1] = 200f; t[3 * 8400 + 1] = 200f
        t[(4 + 6) * 8400 + 1] = 0.8f   // motorbike, same place

        val pre = YoloPreprocessor(640)
        pre.setGeometryForTest(640, 640)
        val post = YoloPostprocessor(640, 0.5f)
        post.configure(19, 8400)
        val dets = post.decode(t, names, 0.45f, pre)
        assertEquals("hai lớp khác nhau phải cùng giữ", 2, dets.size)
    }
}
