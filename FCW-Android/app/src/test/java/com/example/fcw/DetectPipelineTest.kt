package com.example.fcw

import com.example.fcw.detection.YoloPostprocessor
import com.example.fcw.detection.YoloPreprocessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the seam between preprocessing and decoding.
 *
 * The two halves each had their own tests and each passed, yet the pipeline still
 * produced zero boxes on device: `fill()` mixed a byte offset with a plane index and
 * wrote out of bounds on the G and B planes. A unit test per half cannot see that,
 * because neither half is wrong on its own. These tests drive one synthetic
 * detection all the way through both halves and check the numbers land in the right
 * place in source pixels, which is the only place the app ever consumes them.
 */
class DetectPipelineTest {

    private val names = arrayOf(
        "car", "person", "motorbike", "bus", "truck", "bicycle", "dog",
        "pothole", "reflective_cone", "warning_column", "spherical_roadblock",
        "ashcan", "stop_sign", "tricycle", "fire_hydrant"
    )
    private val classes = names.size
    private val anchors = 8400

    /**
     * Builds a head output with one confident detection and nothing else.
     * `classId` picks which channel carries the score, so a test cannot accidentally
     * pass by scoring above threshold in every channel.
     */
    private fun headWith(
        cx: Float, cy: Float, w: Float, h: Float,
        classId: Int, score: Float = 0.9f
    ): FloatArray {
        val out = FloatArray((4 + classes) * anchors)
        out[0 * anchors] = cx
        out[1 * anchors] = cy
        out[2 * anchors] = w
        out[3 * anchors] = h
        out[(4 + classId) * anchors] = score
        return out
    }

    private fun pipeline(srcW: Int, srcH: Int): Pair<YoloPreprocessor, YoloPostprocessor> {
        val pre = YoloPreprocessor(640)
        pre.setGeometryForTest(srcW, srcH)
        val post = YoloPostprocessor(640, 0.5f)
        assertTrue("postprocessor phải cấu hình được", post.configure(4 + classes, anchors))
        return pre to post
    }

    @Test
    fun `box survives letterbox round trip at the right source pixels`() {
        val (pre, post) = pipeline(1280, 960)   // scale 0.5, padY 80
        // Model-space box that must land on a known patch of the source frame.
        val head = headWith(cx = 350f, cy = 350f, w = 100f, h = 100f, classId = 7)

        val found = post.decode(head, names, 0.45f, pre)

        assertEquals("phải ra đúng 1 box", 1, found.size)
        val d = found.single()
        assertEquals("pothole", d.className)
        assertEquals(7, d.classId)
        assertEquals(0.9f, d.confidence, 0.001f)
        // (300-0)/0.5, (300-80)/0.5, (400-0)/0.5, (400-80)/0.5
        assertEquals(600f, d.x1, 0.01f)
        assertEquals(440f, d.y1, 0.01f)
        assertEquals(800f, d.x2, 0.01f)
        assertEquals(640f, d.y2, 0.01f)
        assertEquals(200f, d.width, 0.01f)
        assertEquals(200f, d.height, 0.01f)
        assertEquals(40_000f, d.areaPx, 1f)
    }

    @Test
    fun `a box in the letterbox padding maps above the image and gets clamped`() {
        val (pre, post) = pipeline(1280, 960)
        // y range 20..60 sits entirely in the 80px top pad: nothing real there.
        val head = headWith(cx = 320f, cy = 40f, w = 100f, h = 40f, classId = 2)

        val found = post.decode(head, names, 0.45f, pre)

        assertEquals(1, found.size)
        val d = found.single()
        assertTrue("y1 âm phải bị chặn về 0, thực tế ${d.y1}", d.y1 >= 0f)
        assertTrue("box phải nằm trong khung, thực tế ${d.y2}", d.y2 <= 960f)
    }

    @Test
    fun `only the highest scoring class survives per anchor`() {
        val (pre, post) = pipeline(1280, 960)
        val head = headWith(cx = 320f, cy = 320f, w = 80f, h = 80f, classId = 2)
        // A competing score that must NOT win, on a different anchor.
        head[(4 + 0) * anchors + 1] = 0.95f
        head[0 * anchors + 1] = 200f
        head[1 * anchors + 1] = 200f
        head[2 * anchors + 1] = 60f
        head[3 * anchors + 1] = 60f

        val found = post.decode(head, names, 0.45f, pre)

        assertEquals(2, found.size)
        assertEquals("moto (0.9) phải đứng trước car (0.95) vì diện tích lớn hơn", 2, found[0].classId)
        assertEquals(0, found[1].classId)
    }

    @Test
    fun `preprocess writes every element of the tensor without overrunning`() {
        val pre = YoloPreprocessor(8)
        val plane = 8 * 8
        val rgb = ByteArray(4 * 2 * 3) { (it * 7 % 256).toByte() }
        pre.fill(rgb, 4, 2)
        val t = pre.inputTensor()
        assertEquals(3 * plane, t.size)
        // Every value must be a legal normalised colour in the G and B planes too;
        // the old byte-offset bug left them partly unwritten or out of bounds.
        for (i in 0 until t.size) {
            assertTrue("tensor[$i]=${t[i]} ngoài [0,1]", t[i] in 0f..1f)
        }
        // 4x2 into 8x8 letterbox: scale 2, padX 0, padY 2, content occupies rows 2..5.
        val pad = 114f / 255f
        assertEquals("viền trên phải là 114", pad, t[0], 0.0001f)
        assertEquals("viền dưới phải là 114", pad, t[7 * 8], 0.0001f)
        assertTrue("hàng nội dung phải khác 114", t[2 * 8] != pad)
    }
}