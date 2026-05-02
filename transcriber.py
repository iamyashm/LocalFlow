"""OpenVINO GenAI Whisper wrapper. Loaded once and reused for the whole session."""
import time

import numpy as np

import openvino_genai as ov_genai


class WhisperTranscriber:
    def __init__(self, model_path, device="GPU", language="<|en|>"):
        print(f"[*] Loading Whisper model from: {model_path}")
        print(f"[*] Device: {device}")
        self.pipeline = ov_genai.WhisperPipeline(str(model_path), device=device)
        self.config = self.pipeline.get_generation_config()
        for attr, value in (
            ("language", language),
            ("task", "transcribe"),
            ("return_timestamps", False),
        ):
            if hasattr(self.config, attr):
                try:
                    setattr(self.config, attr, value)
                except Exception:
                    pass
        print("[*] Model loaded.")

    def warmup(self, seconds=1.0, sample_rate=16000):
        """Run a dummy transcription to compile GPU/NPU kernels. The first
        real transcription is otherwise 2-5x slower because OpenVINO compiles
        the model on first use. Silent input is fine — we discard the output.
        """
        try:
            print("[*] Warming up model (compiling kernels)...", flush=True)
            t0 = time.monotonic()
            dummy = np.zeros(int(sample_rate * seconds), dtype=np.float32)
            self.pipeline.generate(dummy, self.config)
            print(f"[*] Warmup done ({time.monotonic() - t0:.2f}s).")
        except Exception as e:
            print(f"[!] Warmup failed (non-fatal): {type(e).__name__}: {e}")

    def transcribe(self, samples_16k_mono_float32, initial_prompt=""):
        """Transcribe audio. `initial_prompt` is fed to the Whisper decoder as
        prior context so it continues from the previous chunk without repeating."""
        if samples_16k_mono_float32 is None or samples_16k_mono_float32.size == 0:
            return ""
        audio = np.ascontiguousarray(samples_16k_mono_float32, dtype=np.float32)
        if initial_prompt and hasattr(self.config, "initial_prompt"):
            try:
                self.config.initial_prompt = initial_prompt
            except Exception:
                pass
        elif hasattr(self.config, "initial_prompt"):
            try:
                self.config.initial_prompt = ""
            except Exception:
                pass
        try:
            result = self.pipeline.generate(audio, self.config)
        except Exception as e:
            print(f"\n[transcribe error] {type(e).__name__}: {e}")
            return ""
        return _extract_text(result).strip()


def _extract_text(result):
    if hasattr(result, "texts"):
        texts = getattr(result, "texts")
        if texts:
            return texts[0]
    return str(result)
