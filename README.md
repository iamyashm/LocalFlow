# LocalFlow — Local Push-to-Talk Dictation for Windows

Privacy-first, fully local speech-to-text that pastes directly into any app.
Hold a hotkey, speak, release — transcribed text appears at the cursor.
No cloud calls, no background services, no admin rights required.

Powered by [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3) running locally via [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai).

---

## Features

- **Push-to-talk** — hold a combo, speak, release to paste
- **Hands-free upgrade** — press an extra key mid-session to go hands-free; stop via overlay buttons
- **Streaming mode** — text pastes incrementally every few seconds while you speak
- **Voice punctuation** — say *"period"*, *"comma"*, *"new line"*, etc.
- **Floating pill overlay** — unobtrusive macOS-style indicator at the bottom of the screen
- **VAD** — WebRTC voice-activity detection skips silent recordings automatically
- **Noise reduction** — optional spectral filtering via `noisereduce`
- **Works everywhere** — browsers, Electron apps, IDEs, native controls, terminals

---

## Requirements

| | |
|---|---|
| **OS** | Windows 10 / 11 x64 |
| **Python** | 3.10 or newer |
| **GPU** | Intel Arc, Intel iGPU (Xe), or any OpenVINO-supported device — CPU also works |
| **RAM** | ~1 GB free (model stays resident) |

---

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/your-username/localflow.git
cd localflow

py -m venv .venv
.\.venv\Scripts\Activate.ps1

py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### 2. Get a Whisper model in OpenVINO format

**Option A — download a pre-converted INT8 model from Hugging Face (recommended)**

```powershell
pip install huggingface_hub
huggingface-cli download OpenVINO/distil-whisper-large-v3-int8 --local-dir distil-whisper-large-v3-int8-ov
```

> Other compatible models: any Whisper variant exported to OpenVINO format works.
> Smaller models (`whisper-base`, `whisper-small`) run faster; larger ones are more accurate.

**Option B — convert yourself with optimum-intel**

```powershell
pip install optimum[openvino]
optimum-cli export openvino --model distil-whisper/distil-large-v3 --weight-format int8 distil-whisper-large-v3-int8-ov
```

### 3. Configure (optional)

Edit `config.py` to change defaults — hotkeys, device, model path, etc.
Everything can also be overridden at runtime via CLI flags (see below).

### 4. Run

```powershell
py main.py
```

Wait for `[*] Warmup done` then start dictating.

---

## Usage

### Push-to-talk (PTT)

1. Click into any text field.
2. Hold **Right Ctrl + Right Alt** and speak.
3. Release — text is pasted at the cursor.

### Hands-free upgrade

While holding the PTT combo, press **Space** to upgrade the session to hands-free mode.
You can now release the PTT keys. The overlay pill expands showing two buttons:

| Button | Action |
|--------|--------|
| **■** (green, right) | Stop recording and paste the transcript |
| **✕** (red, left) | Cancel — discard recording, nothing pasted |

### Streaming mode

```powershell
py main.py --stream
```

Pastes text every ~3 seconds while you hold the key, then flushes the tail on release.
Useful for long dictations where you want to see output sooner.

---

## CLI Reference

```
py main.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | `distil-whisper-large-v3-int8-ov` | Path to OpenVINO model directory |
| `--device DEVICE` | `GPU` | OpenVINO device: `GPU`, `CPU`, `NPU`, `AUTO` |
| `--hotkey COMBO` | `right ctrl+right alt` | Push-to-talk key combo |
| `--handsfree-hotkey COMBO` | `right ctrl+right alt+space` | Full hands-free combo (extra keys beyond PTT trigger the upgrade) |
| `--no-handsfree` | — | Disable hands-free upgrade |
| `--stream` | — | Enable streaming paste mode |
| `--no-overlay` | — | Disable the floating pill overlay |
| `--no-denoise` | — | Disable spectral noise reduction (faster on weak CPUs) |
| `--no-vad` | — | Disable voice-activity detection |
| `--vad-aggressiveness N` | `2` | VAD strictness: 0 (loose) … 3 (strict) |
| `--no-punctuation` | — | Disable spoken punctuation commands |
| `--save` | — | Also save each transcript to a `.txt` file |
| `--save-dir PATH` | `~/stt_tool_transcripts` | Directory for saved transcripts |
| `--no-restore-clipboard` | — | Leave transcript on clipboard after pasting |
| `--no-warmup` | — | Skip model warmup (first transcription will be slower) |

---

## Voice Punctuation

Say these words to insert punctuation:

| Say | Inserts |
|-----|---------|
| *period* / *full stop* | `. ` |
| *comma* | `, ` |
| *question mark* | `? ` |
| *exclamation mark* / *exclamation point* | `! ` |
| *colon* | `: ` |
| *semicolon* | `; ` |
| *new line* / *next line* / *newline* | `\n` |
| *new paragraph* | `\n\n` |
| *em dash* | ` — ` |
| *en dash* | ` – ` |
| *hyphen* | `-` |
| *open quote* / *close quote* | `"` / `"` |
| *open paren* / *close paren* | `(` / `)` |
| *ellipsis* / *dot dot dot* | `... ` |

---

## Architecture

| File | Responsibility |
|------|---------------|
| `main.py` | Entry point. `PushToTalkHotkey` listens for key events. `DictationSession` owns the audio → transcribe → paste pipeline. |
| `config.py` | All tunable defaults (hotkeys, device, thresholds, stream window sizes). |
| `audio_capture.py` | `MicrophoneCapturer` — records from the default mic via `soundcard`, pushes 100 ms mono float32 chunks onto a queue. |
| `transcriber.py` | `WhisperTranscriber` — wraps `openvino_genai.WhisperPipeline`; loaded once, kept resident. |
| `vad.py` | `VAD` — WebRTC voice-activity detection with RMS fallback. Used to skip silent audio and trim trailing silence. |
| `denoiser.py` | `Denoiser` — optional spectral noise reduction via `noisereduce`. |
| `text_postprocess.py` | Hallucination filter and spoken-punctuation regex replacements. |
| `text_injector.py` | Clipboard write + `keyboard.send("ctrl+v")` paste. |
| `focus.py` | Win32 foreground HWND capture and restore (`AttachThreadInput` trick). |
| `dedup.py` | Overlap-dedup merge for streaming mode (stitches adjacent Whisper windows). |
| `overlay.py` | PySide6 frameless pill overlay. States: `hidden` / `recording` / `handsfree` / `processing`. |

---

## Notes

- **First launch is slow** — OpenVINO compiles GPU kernels on the first transcription. Subsequent sessions are fast.
- **CPU fallback** — pass `--device CPU` if you don't have a supported GPU. Latency will be higher but everything still works.
- **Hotkey collisions** — the PTT combo is not suppressed (key events reach other apps as normal). Choose a combo you don't use for anything else.
- **Paste compatibility** — `Ctrl+V` paste works in every standard text surface. A handful of apps (raw `cmd.exe` without QuickEdit, some games) don't honor it; in those cases the transcript is left on the clipboard for manual paste.
- **Privacy** — all processing is local. No network calls are made at any point.
