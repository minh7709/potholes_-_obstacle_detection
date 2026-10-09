package com.example.fcw.camera

import androidx.camera.core.ImageProxy

/**
 * One analysis frame, decoupled from CameraX.
 *
 * The timestamp here is the CAPTURE time, not the moment inference finishes.
 * That distinction matters: closing speed divides by delta time, so using the
 * completion time would silently corrupt every speed the system reports once
 * processing latency varies between frames.
 */
data class CameraFrame(
    val image: ImageProxy,
    val captureTimeNanos: Long,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int
) {
    val timestampSeconds: Double get() = captureTimeNanos / 1_000_000_000.0
}
