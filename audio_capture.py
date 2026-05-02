"""Default microphone capture using soundcard.

Captures the user's default input device for dictation. Output samples are
at `self.sample_rate` (native device rate, typically 48 kHz on Windows);
resampling to 16 kHz happens downstream.
"""
import threading
import queue
import numpy as np

import soundcard as sc

import config


class MicrophoneCapturer:
    def __init__(self, sample_rate=config.NATIVE_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._thread = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def start(self):
        self._stop_event.clear()
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MicCapture"
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_chunk(self, timeout=0.2):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        try:
            mic = sc.default_microphone()
            block = self.sample_rate // 10  # 100 ms blocks
            with mic.recorder(samplerate=self.sample_rate, channels=1, blocksize=block) as rec:
                while not self._stop_event.is_set():
                    data = rec.record(numframes=block)
                    if data is None or data.size == 0:
                        continue
                    if data.ndim > 1:
                        mono = data.mean(axis=1).astype(np.float32, copy=False)
                    else:
                        mono = data.astype(np.float32, copy=False).reshape(-1)
                    self._queue.put(mono)
        except Exception as e:
            print(f"\n[capture error] {type(e).__name__}: {e}")
            self._stop_event.set()
