"""Voice Activity Detection wrapper.

Uses webrtcvad (tiny C extension, <100 KB) to detect whether a window of
audio actually contains speech. If webrtcvad isn't installed, falls back to
a simple RMS-energy threshold so the app keeps working.
"""
import numpy as np

import config


class VAD:
    def __init__(self, aggressiveness=2, enabled=True):
        self.available = False
        self.enabled = enabled
        self._vad = None
        self.frame_ms = 30
        self.sample_rate = 16000
        self.samples_per_frame = int(self.sample_rate * self.frame_ms / 1000)  # 480

        if not enabled:
            print("[*] VAD: disabled (--no-vad)")
            return
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            self.available = True
            print(f"[*] VAD: webrtcvad (aggressiveness={aggressiveness})")
        except ImportError:
            print("[!] VAD: webrtcvad not installed; using RMS-only fallback.")
            print("     pip install webrtcvad-wheels  for proper VAD.")
        except Exception as e:
            print(f"[!] VAD init failed: {type(e).__name__}: {e}")

    def has_speech(self, audio_16k_mono_float32, min_speech_ratio=0.04):
        """Return True if at least min_speech_ratio of frames contain speech.
        Falls back to RMS energy if webrtcvad is unavailable.
        """
        if not self.enabled:
            return True
        if audio_16k_mono_float32 is None or audio_16k_mono_float32.size == 0:
            return False

        if not self.available or self._vad is None:
            # RMS fallback
            rms = float(np.sqrt(np.mean(np.square(
                audio_16k_mono_float32, dtype=np.float64))))
            return rms >= config.SILENCE_RMS_THRESHOLD

        clipped = np.clip(audio_16k_mono_float32, -1.0, 1.0)
        audio_int16 = (clipped * 32767.0).astype(np.int16)
        n_frames = len(audio_int16) // self.samples_per_frame
        if n_frames == 0:
            return False

        speech_frames = 0
        try:
            for i in range(n_frames):
                start = i * self.samples_per_frame
                end = start + self.samples_per_frame
                frame_bytes = audio_int16[start:end].tobytes()
                if self._vad.is_speech(frame_bytes, self.sample_rate):
                    speech_frames += 1
        except Exception as e:
            print(f"[vad warn] {type(e).__name__}: {e}")
            return True  # fail-open: assume speech rather than drop audio

        return (speech_frames / n_frames) >= min_speech_ratio

    def last_speech_frame(self, audio_16k_mono_float32, buffer_ms=200):
        """Returns sample index just past the last speech frame + buffer_ms of audio.
        Falls back to len(audio) if webrtcvad is unavailable or no speech is found.
        """
        if (not self.enabled or not self.available or self._vad is None
                or audio_16k_mono_float32 is None
                or audio_16k_mono_float32.size == 0):
            return len(audio_16k_mono_float32)

        clipped = np.clip(audio_16k_mono_float32, -1.0, 1.0)
        audio_int16 = (clipped * 32767.0).astype(np.int16)
        n_frames = len(audio_int16) // self.samples_per_frame
        if n_frames == 0:
            return len(audio_16k_mono_float32)

        last_speech = 0
        try:
            for i in range(n_frames):
                start = i * self.samples_per_frame
                frame_bytes = audio_int16[start : start + self.samples_per_frame].tobytes()
                if self._vad.is_speech(frame_bytes, self.sample_rate):
                    last_speech = i
        except Exception:
            return len(audio_16k_mono_float32)

        buffer_samples = int(self.sample_rate * buffer_ms / 1000)
        return min(
            (last_speech + 1) * self.samples_per_frame + buffer_samples,
            len(audio_16k_mono_float32),
        )
