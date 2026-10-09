package com.example.fcw

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.example.fcw.ui.CameraScreen
import com.example.fcw.ui.PermissionRequest

private const val TAG = "FCW"

/**
 * Entry point.
 *
 * Permission handling lives here so [CameraScreen] can assume the camera is
 * available when it composes. Asking again after a permanent denial does nothing,
 * so the UI offers a shortcut to system settings instead of silently failing.
 *
 * `rememberLauncherForActivityResult` is a composable function, which is why the
 * whole permission flow sits inside [AppRoot] rather than in onCreate.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AppRoot() }
    }
}

@Composable
private fun AppRoot() {
    val context = androidx.compose.ui.platform.LocalContext.current

    // requestCount lets us tell "not asked yet" from "permanently denied":
    // Android stops showing the dialog after two denials, so the only signal is
    // that we asked and got nothing.
    var askedCount by remember { mutableIntStateOf(0) }
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                android.content.pm.PackageManager.PERMISSION_GRANTED
        )
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        if (!granted) {
            Log.w(TAG, "CAMERA permission bị từ chối (lần ${askedCount})")
        }
    }

    if (hasPermission) {
        CameraScreen()
    } else {
        PermissionRequest(
            onRequest = {
                askedCount++
                launcher.launch(Manifest.permission.CAMERA)
            },
            onOpenSettings = {
                runCatching {
                    context.startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                            data = Uri.fromParts("package", context.packageName, null)
                        }
                    )
                }.onFailure { Log.e(TAG, "Mở Settings lỗi: ${it.message}") }
            },
            settingsNeeded = askedCount >= 2
        )
    }
}