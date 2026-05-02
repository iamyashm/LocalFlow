"""Local on-demand dictation for Windows 11. Wispr Flow-style push-to-talk.

HOLD the global hotkey to record from the default microphone. Text streams
into the focused app incrementally every few seconds while you hold. On
RELEASE the final tail is flushed and pasted. Pass --no-stream for classic
single-shot-on-release behavior.
"""
import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import pyperclip
from scipy.signal import resample_poly
import keyboard

import config
import dedup as dedup_mod
from audio_capture import MicrophoneCapturer
from transcriber import WhisperTranscriber
from text_injector import inject_text, paste_delta_fast
from focus import get_foreground_hwnd, get_window_title
from text_postprocess import is_hallucination, apply_punctuation
from vad import VAD

try:
    from overlay import AudioOverlay
    _HAS_OVERLAY = AudioOverlay.available
except Exception:
    _HAS_OVERLAY = False
    AudioOverlay = None

try:
    from denoiser import Denoiser
    _HAS_DENOISER = Denoiser.available
except Exception:
    _HAS_DENOISER = False
    Denoiser = None


class DictationSession:
    def __init__(self, transcriber, capturer, hotkey_label,
                 restore_clipboard=True, save_to_file=False, transcript_dir=None,
                 streaming=False, vad=None, apply_punct=True,
                 on_state_change=None, on_level=None, record_label="recording",
                 denoiser=None):
        self.transcriber = transcriber
        self.capturer = capturer
        self.hotkey_label = hotkey_label
        self.restore_clipboard = restore_clipboard
        self.save_to_file = save_to_file
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self.streaming = streaming
        self.vad = vad
        self.apply_punct = apply_punct
        self._on_state_change = on_state_change
        self._on_level = on_level
        self._record_label = record_label
        self._denoiser = denoiser

        self.recording = False
        self._lock = threading.Lock()
        self._collect_stop = threading.Event()
        self._collector = None
        self._chunks = []
        self._target_hwnd = None
        self._target_title = ""
        self._started_at = None

        self._handsfree_upgraded = False

        self._native_rate = capturer.sample_rate
        g = gcd(self._native_rate, config.SAMPLE_RATE)
        self._up = config.SAMPLE_RATE // g
        self._down = self._native_rate // g

        # --- Streaming state (reset on each key-down) ---
        self._buf = np.empty(0, dtype=np.float32)  # shared growing audio buffer
        self._buf_lock = threading.Lock()
        self._stream_stop = threading.Event()
        self._stream_worker = None
        # Published by worker; read by on_key_up only after worker is joined.
        self._stream_consumed = 0      # native-rate sample pointer
        self._stream_prev_merged = ""  # dedup-merged transcript so far
        self._pre_dictation_clip = None  # clipboard contents before session started

    def _set_state(self, state):
        if self._on_state_change is not None:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def _clean_text(self, text):
        """Apply hallucination filter + voice-punctuation. Returns cleaned text
        (possibly empty if it was a known canned hallucination)."""
        if not text:
            return ""
        if is_hallucination(text):
            return ""
        if self.apply_punct:
            text = apply_punctuation(text)
        return text

    def _has_speech(self, audio_16k):
        """Speech-presence check: cheap RMS first, then VAD if available."""
        if self._is_silent(audio_16k):
            return False
        if self.vad is not None:
            return self.vad.has_speech(audio_16k)
        return True

    def _prepare_audio(self, audio_16k):
        """Trim trailing silence → denoise → RMS-normalize before transcription."""
        # Trim trailing silence: keep 200ms buffer after the last speech frame
        if self.vad is not None:
            end = self.vad.last_speech_frame(audio_16k)
            if end < len(audio_16k):
                audio_16k = audio_16k[:end]
        if audio_16k.size == 0:
            return audio_16k
        # Spectral noise reduction (optional)
        if self._denoiser is not None:
            audio_16k = self._denoiser.reduce(audio_16k)
        # RMS normalize to 0.1 — Whisper expects near-±1 audio
        rms = float(np.sqrt(np.mean(np.square(audio_16k, dtype=np.float64))))
        if rms > 1e-6:
            audio_16k = np.clip(audio_16k * (0.1 / rms), -1.0, 1.0).astype(np.float32)
        return audio_16k

    def on_key_down(self):
        with self._lock:
            if self.recording:
                return
            self.recording = True
            self._chunks = []
            self._target_hwnd = get_foreground_hwnd()
            self._target_title = get_window_title(self._target_hwnd)
            print(f"[hotkey] down — target: {self._target_title!r}", flush=True)
            self._collect_stop.clear()
            self._started_at = datetime.now()

            if self.streaming:
                with self._buf_lock:
                    self._buf = np.empty(0, dtype=np.float32)
                    self._stream_consumed = 0
                    self._stream_prev_merged = ""
                self._stream_stop.clear()
                if self.restore_clipboard:
                    try:
                        self._pre_dictation_clip = pyperclip.paste()
                    except Exception:
                        self._pre_dictation_clip = None

            self.capturer.start()
            self._collector = threading.Thread(
                target=self._collect, daemon=True, name="STT-Collector"
            )
            self._collector.start()

            if self.streaming:
                self._stream_worker = threading.Thread(
                    target=self._stream_loop, daemon=True, name="STT-StreamWorker"
                )
                self._stream_worker.start()

            self._set_state(self._record_label)

            sys.stdout.write(
                f"\n[REC] target: {self._target_title!r}  "
                f"— speak. release {self.hotkey_label!r} to paste.\n"
            )
            sys.stdout.flush()

    def on_key_up(self):
        with self._lock:
            if not self.recording:
                return
            if self._handsfree_upgraded:
                return  # session is now HF-controlled; overlay buttons end it
            self.recording = False
            self._set_state("processing")
            try:
                self._handle_key_up()
            finally:
                self._set_state("hidden")
                self._handsfree_upgraded = False

    def on_upgrade_to_handsfree(self):
        """Called when the upgrade key is pressed while PTT is active."""
        with self._lock:
            if not self.recording:
                return
            self._handsfree_upgraded = True
        self._set_state("handsfree")
        sys.stdout.write("[HF] upgraded to hands-free — click ■ to stop or ✕ to cancel.\n")
        sys.stdout.flush()

    def handsfree_stop(self):
        """Overlay ■ button: end recording and paste."""
        threading.Thread(
            target=self._do_handsfree_end, args=(True,),
            daemon=True, name="HF-Stop",
        ).start()

    def handsfree_cancel(self):
        """Overlay ✕ button: end recording, discard audio."""
        threading.Thread(
            target=self._do_handsfree_end, args=(False,),
            daemon=True, name="HF-Cancel",
        ).start()

    def _do_handsfree_end(self, do_paste: bool):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            self._handsfree_upgraded = False
            if do_paste:
                self._set_state("processing")

        if do_paste:
            try:
                self._handle_key_up()
            finally:
                self._set_state("hidden")
        else:
            # Cancel: stop collection without transcribing.
            self._collect_stop.set()
            self.capturer.stop()
            collector = self._collector
            self._collector = None
            if collector is not None:
                collector.join(timeout=2.0)
            if self.streaming:
                self._stream_stop.set()
                worker = self._stream_worker
                self._stream_worker = None
                if worker is not None:
                    worker.join(timeout=5.0)
            self._chunks = []
            self._set_state("hidden")
            print("[HF] cancelled.", flush=True)

    def _handle_key_up(self):
        # --- Stop audio collection ---
        self._collect_stop.set()
        self.capturer.stop()
        if self._collector is not None:
            self._collector.join(timeout=2.0)
            self._collector = None

        # Drain any audio still queued in the capturer.
        while True:
            chunk = self.capturer.get_chunk(timeout=0.05)
            if chunk is None:
                break
            self._chunks.append(chunk)
            if self.streaming:
                with self._buf_lock:
                    self._buf = np.concatenate((self._buf, chunk))

        # ── SINGLE-SHOT MODE ──────────────────────────────────────────────
        if not self.streaming:
            if not self._chunks:
                print("[!] no audio captured.")
                return
            audio_native = np.concatenate(self._chunks)
            self._chunks = []
            duration = audio_native.size / float(self._native_rate)
            if duration < config.MIN_RECORD_SECONDS:
                print(f"[!] too short ({duration:.2f}s) — ignored.")
                return
            audio_16k = self._resample_to_16k(audio_native)
            if not self._has_speech(audio_16k):
                print(f"[!] {duration:.1f}s captured but no speech — nothing to paste.")
                return
            audio_16k = self._prepare_audio(audio_16k)
            print(f"[...] transcribing {duration:.1f}s...", flush=True)
            t0 = time.monotonic()
            text = self.transcriber.transcribe(audio_16k).strip()
            dt = time.monotonic() - t0
            text = self._clean_text(text)
            if not text:
                print(f"[!] no speech recognized ({dt:.2f}s).")
                return
            preview = text if len(text) <= 100 else text[:97] + "..."
            print(f"[OK] {dt:.2f}s -> {preview!r}")
            ok = inject_text(
                text,
                target_hwnd=self._target_hwnd,
                restore_clipboard=self.restore_clipboard,
            )
            if ok:
                print(f"[+] pasted {len(text)} chars into {self._target_title!r}")
            else:
                print("[!] paste failed — text on clipboard, press Ctrl+V manually.")
            if self.save_to_file and self.transcript_dir is not None:
                self._save(text)
            return

        # ── STREAMING MODE ────────────────────────────────────────────────
        # Signal worker to exit after its current transcription, then wait.
        self._stream_stop.set()
        if self._stream_worker is not None:
            self._stream_worker.join(timeout=30.0)
            self._stream_worker = None

        # Read worker's final published state (safe: worker is joined).
        with self._buf_lock:
            buf_snapshot     = self._buf.copy()
            consumed         = self._stream_consumed
            prev_merged_base = self._stream_prev_merged
        self._chunks = []

        overlap_native = int(
            (config.STREAM_WINDOW_SECONDS - config.STREAM_HOP_SECONDS)
            * self._native_rate
        )
        min_native = int(config.MIN_RECORD_SECONDS * self._native_rate)

        # Tail window: overlap lookback for dedup context + any unseen audio.
        tail_start = max(0, consumed - overlap_native)
        remaining  = buf_snapshot[tail_start:]

        if len(remaining) < min_native:
            if not prev_merged_base:
                print("[!] no audio captured.")
            # Nothing new to paste; clipboard already holds last delta.
            self._restore_pre_dictation_clip()
            return

        remaining_16k = self._resample_to_16k(remaining)
        if not self._has_speech(remaining_16k):
            print("[stream] tail silent — finalizing.")
            self._restore_pre_dictation_clip()
            return
        remaining_16k = self._prepare_audio(remaining_16k)

        tail_dur = len(remaining) / float(self._native_rate)
        print(f"[...] final flush: {tail_dur:.1f}s...", flush=True)

        # Use prev_merged_base as initial_prompt so Whisper continues
        # naturally from where the stream worker left off.
        prompt_words   = prev_merged_base.split()
        initial_prompt = " ".join(prompt_words[-50:]) if prompt_words else ""

        t0   = time.monotonic()
        text = self.transcriber.transcribe(remaining_16k, initial_prompt=initial_prompt)
        dt   = time.monotonic() - t0
        text = self._clean_text(text)

        if not text:
            print(f"[stream] no speech in tail ({dt:.2f}s).")
            self._restore_pre_dictation_clip()
            return

        final_delta = self._safe_delta(prev_merged_base, text)

        if final_delta.strip():
            print(f"[stream] final {len(final_delta)} chars — pasting...", flush=True)
            ok = inject_text(
                final_delta,
                target_hwnd=self._target_hwnd,
                restore_clipboard=False,   # restored manually below
            )
            total = len(prev_merged_base) + len(final_delta)
            if ok:
                print(f"[+] done — {total} total chars into {self._target_title!r}")
            else:
                print("[!] final paste failed — text on clipboard, press Ctrl+V.")
        else:
            print("[stream] tail already covered — nothing new to paste.")

        self._restore_pre_dictation_clip()

        if self.save_to_file and self.transcript_dir is not None:
            self._save(prev_merged_base + final_delta)

    def _restore_pre_dictation_clip(self):
        if self.restore_clipboard and self._pre_dictation_clip is not None:
            time.sleep(0.15)   # let target app finish consuming the last paste
            try:
                pyperclip.copy(self._pre_dictation_clip)
            except Exception:
                pass
            self._pre_dictation_clip = None

    def _collect(self):
        while not self._collect_stop.is_set():
            chunk = self.capturer.get_chunk(timeout=0.2)
            if chunk is not None:
                self._chunks.append(chunk)
                if self.streaming:
                    with self._buf_lock:
                        self._buf = np.concatenate((self._buf, chunk))
                if self._on_level is not None and chunk.size:
                    try:
                        rms = float(np.sqrt(np.mean(np.square(
                            chunk, dtype=np.float64))))
                        self._on_level(rms)
                    except Exception:
                        pass

    def _stream_loop(self):
        """Runs on STT-StreamWorker.

        Every hop_native new samples, grabs that exact non-overlapping chunk,
        feeds Whisper the previous transcript as initial_prompt so it continues
        without repeating, then pastes only the new text.

        No word-overlap dedup is needed: initial_prompt handles continuity at
        the text level. Each window transcribes only the new audio.
        """
        hop_native = int(config.STREAM_HOP_SECONDS * self._native_rate)
        consumed   = 0
        prev_pasted = ""   # running concatenation of everything pasted so far

        while not self._stream_stop.is_set():
            with self._buf_lock:
                buf_len = len(self._buf)

            if buf_len < consumed + hop_native:
                time.sleep(0.05)
                continue

            # Non-overlapping: only the new hop of audio.
            with self._buf_lock:
                window = self._buf[consumed : consumed + hop_native].copy()
            consumed += hop_native

            window_16k = self._resample_to_16k(window)
            if not self._has_speech(window_16k):
                with self._buf_lock:
                    self._stream_consumed = consumed
                continue
            window_16k = self._prepare_audio(window_16k)

            # Build a short trailing prompt (≤50 words) so Whisper continues
            # naturally without re-transcribing what was already pasted.
            prompt_words = prev_pasted.split()
            initial_prompt = " ".join(prompt_words[-50:]) if prompt_words else ""

            text = self.transcriber.transcribe(window_16k, initial_prompt=initial_prompt)
            text = self._clean_text(text)
            if not text:
                with self._buf_lock:
                    self._stream_consumed = consumed
                continue

            # Space-separated append; trim any accidental leading repetition
            # (Whisper respects initial_prompt but occasionally echoes 1-2 words).
            delta = self._safe_delta(prev_pasted, text)
            if delta.strip():
                prev_pasted += delta
                ok = paste_delta_fast(delta, target_hwnd=self._target_hwnd)
                print(
                    f"[stream] {len(delta)} chars pasted" + ("" if ok else " (FAILED)"),
                    flush=True,
                )

            with self._buf_lock:
                self._stream_consumed    = consumed
                self._stream_prev_merged = prev_pasted

        # Publish final state after loop exits.
        with self._buf_lock:
            self._stream_consumed    = consumed
            self._stream_prev_merged = prev_pasted

    @staticmethod
    def _safe_delta(prev_pasted, new_text, max_trim=3):
        """Return new_text as a space-prefixed delta, trimming up to max_trim
        leading words that duplicate the end of prev_pasted (safety net for
        the rare case where Whisper echoes the last word despite initial_prompt).
        """
        new_text = new_text.strip()
        if not new_text:
            return ""
        if not prev_pasted:
            return new_text  # first chunk: no leading space, no dedup needed

        prev_words = prev_pasted.split()
        new_words  = new_text.split()

        # Remove up to max_trim leading words from new_words if they repeat
        # the tail of prev_words.
        trim = 0
        for n in range(min(max_trim, len(prev_words), len(new_words)), 0, -1):
            prev_tail = [w.lower().rstrip(".,!?;:") for w in prev_words[-n:]]
            new_head  = [w.lower().rstrip(".,!?;:") for w in new_words[:n]]
            if prev_tail == new_head:
                trim = n
                break

        remaining = new_words[trim:]
        if not remaining:
            return ""
        return " " + " ".join(remaining)

    def _resample_to_16k(self, samples_native):
        if self._up == 1 and self._down == 1:
            return samples_native.astype(np.float32, copy=False)
        return resample_poly(samples_native, up=self._up, down=self._down).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _is_silent(samples):
        if samples.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        return rms < config.SILENCE_RMS_THRESHOLD

    def _save(self, text):
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            ts = (self._started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
            path = self.transcript_dir / f"transcript_{ts}.txt"
            path.write_text(text + "\n", encoding="utf-8")
            print(f"[debug] saved to {path}")
        except Exception as e:
            print(f"[debug] save failed: {type(e).__name__}: {e}")


class PushToTalkHotkey:
    """Push-to-talk hotkey supporting a single key ("right ctrl", "f9") or a
    combo ("ctrl+shift"). Uses a single global `keyboard.hook` so we never
    suppress key events — the OS always sees both press and release, which
    avoids stuck-modifier bugs.

    Combo semantics: on_down fires once when every key in the combo is held;
    on_up fires when any of them is released.
    """

    _MOD_ALIASES = {
        "left ctrl": "ctrl", "right ctrl": "right ctrl",
        "left control": "ctrl", "right control": "right ctrl",
        "control": "ctrl",
        # "right alt" / "alt gr" / "altgr" are the same physical key on most
        # keyboards; the keyboard library reports it differently per layout.
        "left alt": "alt",
        "right alt": "right alt", "alt gr": "right alt", "altgr": "right alt",
        "left shift": "shift", "right shift": "shift",
        "left windows": "windows", "right windows": "windows",
        "win": "windows", "left win": "windows", "right win": "windows",
    }

    def __init__(self, hotkey_string, on_down, on_up,
                 upgrade_keys=None, on_upgrade=None):
        self._keys = [self._canonical(k.strip())
                      for k in hotkey_string.split("+") if k.strip()]
        self._key_set = set(self._keys)
        self._on_down = on_down
        self._on_up = on_up
        # Keys that upgrade an active PTT session to hands-free mode.
        self._upgrade_keys = set(self._canonical(k) for k in (upgrade_keys or []))
        self._on_upgrade = on_upgrade
        self._held = set()
        self._fired = False     # True while PTT combo is being held
        self._upgraded = False  # True after upgrade key was pressed this session
        self._lock = threading.Lock()
        self._hook_handle = None

    @classmethod
    def _canonical(cls, name):
        if not name:
            return ""
        n = name.lower().strip()
        return cls._MOD_ALIASES.get(n, n)

    def install(self):
        # Single global hook. We never pass suppress=True — suppressing key
        # events (especially modifier releases) leaves the OS in an
        # inconsistent state ("stuck Win key", "stuck Ctrl"). Letting the
        # event flow through is always safe for sane PTT keys.
        self._hook_handle = keyboard.hook(self._on_event)

    def uninstall(self):
        try:
            if self._hook_handle is not None:
                keyboard.unhook(self._hook_handle)
        except Exception:
            pass
        self._hook_handle = None

    def _on_event(self, event):
        canonical = self._canonical(getattr(event, "name", "") or "")
        ev_type = getattr(event, "event_type", "")

        # Upgrade-key tracking: pressing an upgrade key while PTT is active
        # transitions the session to hands-free mode.
        if canonical in self._upgrade_keys:
            if ev_type == "down":
                fire_upgrade = False
                with self._lock:
                    if self._fired and not self._upgraded:
                        self._upgraded = True
                        fire_upgrade = True
                if fire_upgrade and self._on_upgrade is not None:
                    threading.Thread(
                        target=self._safe_call, args=(self._on_upgrade,),
                        daemon=True, name="PTT-Upgrade",
                    ).start()
            return  # upgrade keys are never processed as PTT keys

        if not canonical or canonical not in self._key_set:
            return
        if ev_type == "down":
            self._handle_down(canonical)
        elif ev_type == "up":
            self._handle_up(canonical)

    def _handle_down(self, canonical):
        fire_now = False
        with self._lock:
            self._held.add(canonical)
            if not self._fired and self._key_set.issubset(self._held):
                self._fired = True
                fire_now = True
        if fire_now:
            threading.Thread(
                target=self._safe_call, args=(self._on_down,),
                daemon=True, name="PTT-Down",
            ).start()

    def _handle_up(self, canonical):
        fire_up = False
        with self._lock:
            self._held.discard(canonical)
            if self._fired:
                # Don't fire on_up if session was upgraded to HF; the overlay
                # buttons control the end of that session.
                fire_up = not self._upgraded
                self._fired = False
                self._upgraded = False
        if fire_up:
            threading.Thread(
                target=self._safe_call, args=(self._on_up,),
                daemon=True, name="PTT-Up",
            ).start()

    @staticmethod
    def _safe_call(fn):
        try:
            fn()
        except Exception as e:
            print(f"[hotkey error] {type(e).__name__}: {e}")



def parse_args():
    p = argparse.ArgumentParser(
        description="Local push-to-talk dictation via OpenVINO Whisper. "
                    "Hold the hotkey to record; release to paste at the cursor."
    )
    p.add_argument("--model", default=str(config.DEFAULT_MODEL_PATH),
                   help="Path to OpenVINO Whisper model directory.")
    p.add_argument("--device", default=config.DEFAULT_DEVICE,
                   help="OpenVINO device: GPU (default), CPU, NPU, AUTO.")
    p.add_argument("--hotkey", default=config.DEFAULT_HOTKEY,
                   help="Push-to-talk key. Single key recommended "
                        "('right ctrl', 'right alt', 'f9', 'caps lock', ...).")
    p.add_argument("--save", action="store_true",
                   help="Also save each transcript to disk (debug).")
    p.add_argument("--save-dir", default=str(config.DEFAULT_TRANSCRIPT_DIR),
                   help="Directory for debug transcript files (used with --save).")
    p.add_argument("--no-restore-clipboard", action="store_true",
                   help="Don't restore the previous clipboard contents after pasting.")
    p.add_argument("--stream", action="store_true",
                   help="Enable incremental streaming paste every "
                        f"~{config.STREAM_HOP_SECONDS:.0f}s while holding the hotkey "
                        "(default: off, transcribe once on release).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Disable the floating audio-level overlay (headless mode).")
    p.add_argument("--no-denoise", action="store_true",
                   help="(Deprecated — noise reduction is now off by default; this flag is a no-op.)")
    p.add_argument("--denoise", action="store_true",
                   help="Enable spectral noise reduction (~30ms/window; useful in noisy environments).")
    p.add_argument("--handsfree-hotkey", default=config.DEFAULT_HANDSFREE_HOTKEY,
                   help="Hands-free upgrade combo. Keys beyond --hotkey that, when added "
                        "while holding PTT, upgrade the session to hands-free mode. "
                        f"(default: {config.DEFAULT_HANDSFREE_HOTKEY!r})")
    p.add_argument("--no-handsfree", action="store_true",
                   help="Disable the hands-free toggle hotkey.")
    p.add_argument("--no-vad", action="store_true",
                   help="Disable Voice Activity Detection (every window is transcribed).")
    p.add_argument("--no-punctuation", action="store_true",
                   help="Disable spoken-punctuation commands (\"period\", \"new line\", ...).")
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3],
                   help="WebRTC VAD aggressiveness: 0 (least, lets through more) ... 3 (most strict). Default: 2.")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip model warmup (first transcription will be slower).")
    return p.parse_args()


def main():
    args = parse_args()

    if "," in args.hotkey:
        print(f"[!] --hotkey may not contain ',', got {args.hotkey!r}.",
              file=sys.stderr)
        sys.exit(2)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] Model directory not found: {model_path}", file=sys.stderr)
        sys.exit(2)

    transcriber = WhisperTranscriber(
        model_path, device=args.device, language=config.LANGUAGE
    )
    if not args.no_warmup:
        transcriber.warmup()
    capturer = MicrophoneCapturer()
    vad = VAD(aggressiveness=args.vad_aggressiveness, enabled=not args.no_vad)
    streaming = args.stream

    use_overlay = _HAS_OVERLAY and not args.no_overlay
    overlay = None
    if use_overlay:
        try:
            overlay = AudioOverlay()
        except Exception as e:
            print(f"[!] Overlay init failed (non-fatal): {type(e).__name__}: {e}")
            overlay = None
            use_overlay = False

    _ov_state = overlay.set_state if overlay is not None else None
    _ov_level = overlay.set_level if overlay is not None else None

    denoiser = None
    if _HAS_DENOISER and args.denoise:
        try:
            denoiser = Denoiser()
        except Exception as e:
            print(f"[!] Denoiser init failed (non-fatal): {type(e).__name__}: {e}")

    session = DictationSession(
        transcriber=transcriber,
        capturer=capturer,
        hotkey_label=args.hotkey,
        restore_clipboard=not args.no_restore_clipboard,
        save_to_file=args.save,
        transcript_dir=args.save_dir,
        streaming=streaming,
        vad=vad,
        apply_punct=not args.no_punctuation,
        on_state_change=_ov_state,
        on_level=_ov_level,
        record_label="recording",
        denoiser=denoiser,
    )

    # Derive the upgrade keys: keys in the HF combo not in the PTT combo.
    # Pressing any of these while holding PTT upgrades the session to HF mode.
    upgrade_keys = set()
    if not args.no_handsfree:
        _hf_keys  = {PushToTalkHotkey._canonical(k.strip())
                     for k in args.handsfree_hotkey.split("+") if k.strip()}
        _ptt_keys = {PushToTalkHotkey._canonical(k.strip())
                     for k in args.hotkey.split("+") if k.strip()}
        upgrade_keys = _hf_keys - _ptt_keys

    upgrade_str = ("+".join(sorted(upgrade_keys)) or "(none)") if not args.no_handsfree else "disabled"

    mode_str = (
        f"streaming (paste every ~{config.STREAM_HOP_SECONDS:.0f}s)"
        if streaming else "single-shot (transcribe on release)"
    )
    print()
    print(f"[*] PTT hotkey:        {args.hotkey}  (hold to record)")
    if not args.no_handsfree:
        print(f"[*] Hands-free:        hold PTT then add {upgrade_str!r} — click ■ to stop/paste, ✕ to cancel")
    print(f"[*] Device:            {args.device}")
    print(f"[*] Mode:              {mode_str}")
    print(f"[*] VAD:               {'on (aggr=' + str(args.vad_aggressiveness) + ')' if not args.no_vad else 'off'}")
    print(f"[*] Voice punct:       {'on' if not args.no_punctuation else 'off'}")
    print(f"[*] Noise reduction:   {'on' if denoiser is not None else 'off (use --denoise to enable)'}")
    print(f"[*] Overlay:           {'on' if use_overlay else 'off'}")
    if args.save:
        print(f"[*] Debug save dir:    {args.save_dir}")
    print()
    print(f"  PUSH-TO-TALK  : click into any text field, HOLD {args.hotkey!r}, speak, release.")
    if not args.no_handsfree:
        print(f"  HANDS-FREE    : hold PTT, then press {upgrade_str!r} to upgrade; click ■ to stop.")
    print(f"  VOICE PUNCTUATION: say 'period', 'comma', 'new line', 'question mark', etc.")
    print(f"  QUIT          : Ctrl+C in this terminal.")
    print()

    ptt = PushToTalkHotkey(
        args.hotkey,
        on_down=session.on_key_down,
        on_up=session.on_key_up,
        upgrade_keys=upgrade_keys or None,
        on_upgrade=session.on_upgrade_to_handsfree if upgrade_keys else None,
    )
    try:
        ptt.install()
    except Exception as e:
        print(f"[!] Failed to register hotkey {args.hotkey!r}: {e}", file=sys.stderr)
        sys.exit(3)

    if overlay is not None and not args.no_handsfree:
        overlay.connect_stop(session.handsfree_stop)
        overlay.connect_cancel(session.handsfree_cancel)

    # PySide6's C++ event loop catches exceptions raised in Python slots and
    # prints them as tracebacks without propagating them — KeyboardInterrupt
    # raised in a slot is swallowed and app.exec() keeps running.
    # Restoring SIG_DFL makes Ctrl+C terminate the process at the OS level,
    # below PySide6's exception machinery.  The --no-overlay path uses
    # time.sleep() which raises KeyboardInterrupt normally, so this is safe.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    try:
        if overlay is not None:
            overlay.run()
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[*] Quitting...")
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        if overlay is not None:
            try:
                overlay.shutdown()
            except Exception:
                pass
        os._exit(0)


if __name__ == "__main__":
    main()
