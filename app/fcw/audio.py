"""Audio alerts: WAV generation and non-blocking playback.

Two tones are synthesised rather than spoken:

    warning.wav  2 medium beeps (660 Hz)
    danger.wav   3 fast high beeps (950 Hz)

The original spec suggested Vietnamese speech ("Canh bao, xe may phia truoc").
No TTS engine is installed and adding one is a heavy dependency for a
prototype, but tone choice is arguably better for a driving aid anyway: a
driver has well under a second of attention available, tone meaning is learned
from a single exposure, whereas speech must be parsed. Replace the .wav files
with real recordings if you disagree.

Why a background thread
-----------------------
pygame.mixer.Sound.play() is normally non-blocking, but mixer init, file I/O and
the mixer lock can block. On a machine already saturated by YOLO and depth, a
blocking call in the frame loop adds latency to every subsequent frame. Playback
is therefore queued to a worker thread, and the frame loop only ever enqueues.
"""

import os
import queue
import threading
import wave

import numpy as np

SAMPLE_RATE = 44100


# ── WAV synthesis ─────────────────────────────────────────────────────
def _write_wav(path, samples, sample_rate=SAMPLE_RATE):
    """Write float samples in [-1, 1] to a 16-bit mono PCM WAV."""
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return path


def _tone(freq, duration, sample_rate=SAMPLE_RATE, volume=0.6,
          harmonics=(1.0, 0.25, 0.10)):
    """One tone with a few harmonics, so it sounds less harsh than a sine."""
    n = max(1, int(sample_rate * duration))
    t = np.linspace(0.0, duration, n, endpoint=False)
    out = np.zeros_like(t)
    for i, amp in enumerate(harmonics, start=1):
        out += amp * np.sin(2.0 * np.pi * freq * i * t)
    fade = max(1, int(sample_rate * 0.01))
    ramp = np.linspace(0.0, 1.0, fade)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out * volume


def generate_warning_sound(path, sample_rate=SAMPLE_RATE):
    """Two medium beeps: a 'heads up' tone, clearly distinct from danger."""
    gap = np.zeros(int(sample_rate * 0.10))
    beep = _tone(660.0, 0.18, sample_rate, volume=0.55)
    return _write_wav(path, np.concatenate([beep, gap, beep]), sample_rate)


def generate_danger_sound(path, sample_rate=SAMPLE_RATE):
    """Three fast high beeps: urgent, unmistakable from the warning tone."""
    gap = np.zeros(int(sample_rate * 0.06))
    beep = _tone(950.0, 0.11, sample_rate, volume=0.75)
    return _write_wav(path, np.concatenate([beep, gap, beep, gap, beep]), sample_rate)


def ensure_audio_assets(warning_path, danger_path):
    """Create the two alert WAVs if they do not exist. Returns created paths."""
    created = []
    if not os.path.exists(warning_path):
        created.append(generate_warning_sound(warning_path))
    if not os.path.exists(danger_path):
        created.append(generate_danger_sound(danger_path))
    return created


# ── Playback ──────────────────────────────────────────────────────────
class AudioAlerter:
    """Cooldown-gated alert playback on a background thread.

    Cooldown is what stops a sustained DANGER from firing 10-20 times a second:
    it replays at most once per cooldown window instead.

    Falls back to a console beep when no audio device exists (common on
    headless machines) so the rest of the pipeline keeps working.
    """

    def __init__(self, enabled=True, volume=0.8, warning_path=None, danger_path=None,
                 warning_cooldown=1.5, danger_cooldown=1.0):
        self.enabled = bool(enabled)
        self.volume = float(volume)
        self.warning_path = warning_path
        self.danger_path = danger_path
        self.warning_cooldown = float(warning_cooldown)
        self.danger_cooldown = float(danger_cooldown)

        self.available = False
        self._pygame = None
        self._sounds = {}
        self._last_played = {"WARNING": -1e9, "DANGER": -1e9}
        self._counts = {"WARNING": 0, "DANGER": 0}
        self._lock = threading.Lock()

        # Queue is bounded: if the worker falls behind we drop the oldest pending
        # alert rather than growing memory without limit.
        self._queue = queue.Queue(maxsize=4)
        self._running = False
        self._thread = None

        if not self.enabled:
            print("[FCW] Audio alerts disabled.")
            return

        self._init_backend()
        ensure_audio_assets(self.warning_path, self.danger_path)
        if self.available:
            self._load_sounds()
        self._start_worker()

    # ── Setup ────────────────────────────────────────────────────────
    def _init_backend(self):
        try:
            import pygame
            pygame.mixer.init(frequency=SAMPLE_RATE)
            pygame.mixer.set_num_channels(8)
            self._pygame = pygame
            self.available = True
            print("[FCW] Audio alerts enabled.")
        except Exception as exc:
            self._pygame = None
            self.available = False
            print(f"[FCW] No audio device ({exc}); using console beeps.")

    def _load_sounds(self):
        try:
            for level, path in (("WARNING", self.warning_path), ("DANGER", self.danger_path)):
                if path and os.path.exists(path):
                    sound = self._pygame.mixer.Sound(path)
                    sound.set_volume(self.volume)
                    self._sounds[level] = sound
        except Exception as exc:
            print(f"[FCW] Could not load alert sounds: {exc}")
            self._sounds = {}

    def _start_worker(self):
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="FCWAudio", daemon=True)
        self._thread.start()

    def _worker(self):
        while self._running:
            try:
                level = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if level is None:
                break
            self._emit(level)

    def _emit(self, level):
        if self.available:
            try:
                sound = self._sounds.get(level)
                if sound is not None:
                    sound.play()
                    return
            except Exception:
                pass
        self._console_beep(level)

    def _console_beep(self, level):
        try:
            print(f"\n[FCW] *** {level} ***\a" if level == "DANGER"
                  else f"\n[FCW] --- {level} ---", flush=True)
        except Exception:
            pass

    # ── Public API ───────────────────────────────────────────────────
    def cooldown_for(self, level):
        return self.danger_cooldown if level == "DANGER" else self.warning_cooldown

    def should_play(self, level, now):
        """Cooldown gate. Updates state, so call once per candidate alert."""
        with self._lock:
            if (now - self._last_played[level]) < self.cooldown_for(level):
                return False
            self._last_played[level] = now
            return True

    def play(self, level, now, force=False):
        """Queue an alert if the cooldown allows. Returns True if queued.

        Never blocks: this is the method the frame loop calls.
        """
        if not self.enabled:
            return False
        if not force and not self.should_play(level, now):
            return False

        with self._lock:
            self._counts[level] = self._counts.get(level, 0) + 1

        try:
            self._queue.put_nowait(level)
        except queue.Full:
            # Worker is behind. Dropping is correct here: a stale alert is worse
            # than no alert, since it would describe an out-of-date gap.
            return False
        return True

    def reset(self):
        """Clear cooldown state, e.g. between two videos."""
        with self._lock:
            self._last_played = {"WARNING": -1e9, "DANGER": -1e9}

    def stats(self):
        with self._lock:
            return dict(self._counts)

    def close(self):
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.available:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass


def default_audio_paths(base_dir):
    return (os.path.join(base_dir, "warning.wav"),
            os.path.join(base_dir, "danger.wav"))