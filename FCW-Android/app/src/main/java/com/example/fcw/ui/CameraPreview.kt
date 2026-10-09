package com.example.fcw.ui

import android.util.Log
import androidx.camera.core.Preview
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView

/**
 * Camera preview as its own layer, separate from the overlay.
 *
 * PreviewView has no setViewPort: the shared ViewPort is applied through the
 * UseCaseGroup in CameraManager, and the visible region is read back from
 * ImageProxy.cropRect. FILL_CENTER here then matches the ViewPort aspect.
 *
 * Keeping the layers separate matters for correctness, not just looks: the preview is
 * a hardware-accelerated surface owned by CameraX, while the overlay is drawn by
 * Compose on top. If the preview were drawn inside the same canvas as the boxes,
 * every bounding-box redraw would invalidate the preview surface.
 */
@Composable
fun CameraPreview(
    previewProvider: () -> Preview?,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            }
        },
        update = { view ->
            previewProvider()?.let { preview ->
                // Re-attach whenever the use case has lost its provider. After a rebind
                // CameraX can drop it, and a Preview with no provider leaves the view
                // black while analysis keeps working, which looks like a dead camera.
                if (preview.surfaceProvider == null ||
                    preview.surfaceProvider != view.surfaceProvider
                ) {
                    Log.i("FCW", "gắn lại surface cho Preview")
                    preview.setSurfaceProvider(view.surfaceProvider)
                }
            }
        }
    )
}
