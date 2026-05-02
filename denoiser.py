"""Optional spectral noise reduction via noisereduce."""
import numpy as np

try:
    import noisereduce as nr
    _HAS_NR = True
except ImportError:
    _HAS_NR = False
    nr = None


class Denoiser:
    available = _HAS_NR

    def __init__(self):
        if not _HAS_NR:
            raise RuntimeError("noisereduce not installed — pip install noisereduce")

    def reduce(self, audio_16k):
        """Apply spectral Wiener noise reduction. Returns cleaned float32 array."""
        if audio_16k is None or audio_16k.size == 0:
            return audio_16k
        try:
            return nr.reduce_noise(y=audio_16k, sr=16000).astype(np.float32)
        except Exception as e:
            print(f"[denoiser] {type(e).__name__}: {e}")
            return audio_16k
