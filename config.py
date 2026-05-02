from pathlib import Path

# Path to the OpenVINO Whisper model directory.
# Place the model folder next to this file, or pass --model at runtime.
DEFAULT_MODEL_PATH = Path("distil-whisper-large-v3-int8-ov")
DEFAULT_DEVICE = "GPU"

# Push-to-talk hotkey. Single key recommended — combos with modifier keys
# (especially the Windows key) are fragile because suppressing the release
# event leaves the OS thinking the modifier is still held. 
DEFAULT_HOTKEY            = "right ctrl+right alt"
DEFAULT_HANDSFREE_HOTKEY  = "right ctrl+right alt+space"

SAMPLE_RATE = 16000
NATIVE_SAMPLE_RATE = 48000

LANGUAGE = "<|en|>"

DEFAULT_TRANSCRIPT_DIR = Path.home() / "stt_tool_transcripts"

SILENCE_RMS_THRESHOLD = 1e-3

# Minimum hold duration; anything shorter is treated as a stray tap.
MIN_RECORD_SECONDS = 0.3

# Streaming paste: how often the worker fires and how much context it feeds Whisper.
STREAM_HOP_SECONDS    = 3.0   # seconds of new audio before each intermediate paste
STREAM_WINDOW_SECONDS = 6.0   # total context window size (hop + lookback overlap)

# Map lowercase substrings of window titles to short Whisper prompt prefixes.
# Whisper uses these to bias vocabulary and style toward the target context.
APP_PROMPTS: dict[str, str] = {
    "outlook":  "Professional email.",
    "teams":    "Chat message.",
    "slack":    "Chat message.",
    "word":     "Document.",
    "notion":   "Document.",
    "code":     "Code comment.",
    "notepad":  "Note.",
    "obsidian": "Note.",
}
