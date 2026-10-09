package com.example.fcw.audio

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.util.Log
import com.example.fcw.config.FcwConfig

/**
 * Cooldown-gated alert audio.
 *
 * Port of `app/fcw/audio.py::AudioAlerter`.
 *
 * Cooldown is the whole point. Without it a sustained DANGER fires on every frame,
 * which at 10-20 FPS means a continuous beep — the driver stops hearing it as a
 * warning and starts hearing it as noise. `warning_cooldown_sec: 1.5` and
 * `danger_cooldown_sec: 1.0` come from config.yaml, DANGER being shorter because it
 * is the state that matters.
 *
 * Two differences from Python, both forced by the platform:
 *  - Python synthesises WAV files with numpy; here ToneGenerator produces the tones so
 *    no assets are needed and nothing can go missing.
 *  - Python falls back to a console beep with no audio device; here there is always a
 *    ToneGenerator, but it can fail on some devices, so the failure is logged and
 *    playback silently skipped rather than taking the frame loop down.
 */
class AudioAlerter(
    private val context: Context,
    private val cfg: FcwConfig = FcwConfig()
) {
    companion object {
        const val TAG = "FCW"

        /** WARNING 660 Hz 0.18 s, DANGER 950 Hz 0.11 s, from audio.py. */
        private const val WARNING_HZ = 660.0
        private const val WARNING_MS = 180
        private const val DANGER_HZ = 950.0
        private const val DANGER_MS = 110
    }

    private var tone: ToneGenerator? = null
    private var available = false
    private var counts = 0

    private var lastWarning = Double.NEGATIVE_INFINITY
    private var lastDanger = Double.NEGATIVE_INFINITY

    val playCount: Int get() = counts

    fun init() {
        if (!cfg.audioEnabled) {
            Log.i(TAG, "Âm thanh cảnh báo đã tắt (config warning.audio_enabled)")
            return
        }
        available = try {
            // STREAM_NOTIFICATION is the constructor this API version offers; the
            // AudioAttributes overload needs API 21+ and a different builder call.
            tone = ToneGenerator(AudioManager.STREAM_NOTIFICATION, (cfg.volume * 100).toInt())
            tone != null
        } catch (e: Throwable) {
            Log.w(TAG, "Không tạo được audio: ${e.message}")
            false
        }
        Log.i(TAG, "Âm thanh sẵn sàng=$available")
    }

    fun cooldownFor(danger: Boolean): Float =
        if (danger) cfg.dangerCooldownSec else cfg.warningCooldownSec

    /**
     * The cooldown gate. Updates state, so call it once per candidate alert.
     *
     * @return true when the alert may be played now.
     */
    @Synchronized
    fun shouldPlay(danger: Boolean, now: Double): Boolean {
        val last = if (danger) lastDanger else lastWarning
        if (now - last < cooldownFor(danger)) return false
        if (danger) lastDanger = now else lastWarning = now
        return true
    }

    /** Plays at most one alert. Never throws and never blocks the frame loop. */
    fun play(danger: Boolean) {
        if (!available) return
        try {
            val hz = if (danger) DANGER_HZ else WARNING_HZ
            val ms = if (danger) DANGER_MS else WARNING_MS
            tone?.startTone(hz.toInt(), ms)
            counts++
        } catch (e: Throwable) {
            Log.w(TAG, "Phát âm lỗi: ${e.message}")
        }
    }

    /**
     * Convenience: cooldown gate then playback.
     *
     * @return true when something was actually played.
     */
    fun alert(danger: Boolean, now: Double): Boolean {
        if (!shouldPlay(danger, now)) return false
        play(danger)
        return true
    }

    fun release() {
        try {
            tone?.release()
        } catch (e: Throwable) {
            Log.w(TAG, "Đóng audio lỗi: ${e.message}")
        }
        tone = null
        available = false
    }
}