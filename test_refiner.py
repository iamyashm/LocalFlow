"""Standalone test for TextRefiner - no microphone, no Whisper, no hotkeys."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path

import config
from refiner import TextRefiner

MODEL_PATH = Path(config.DEFAULT_ENHANCE_MODEL_PATH)
DEVICE = config.DEFAULT_DEVICE
TIMEOUT = 30.0  # generous - includes GPU kernel compilation

TESTS = [
    # (description, transcript, context_hint)
    ("numbered list - number one/two/three",
     "number one get milk number two get eggs number three get bread",
     ""),
    ("numbered list - first/second/third",
     "first call John second send the report third update the calendar",
     ""),
    ("numbered list - with intro text (should become header)",
     "to-do list for today number one buy milk number two call John number three send email",
     ""),
    ("numbered list - with intro text 2",
     "things to buy number one eggs number two bread number three milk",
     ""),
    ("self-correction - scratch that",
     "I need to go to the store scratch that I need to go to the market",
     ""),
    ("self-correction - no actually (with value)",
     "increase weight by 10% no actually 15%",
     ""),
    ("self-correction - no wait",
     "the budget is 20 dollars no wait 25 dollars",
     ""),
    ("self-correction - actually",
     "the meeting is at 3pm actually the meeting is at 4pm",
     ""),
    ("punctuation only - no structural change",
     "the quick brown fox jumps over the lazy dog",
     ""),
    ("context-aware formality - email",
     "please find the attached document for your review",
     "Professional email."),
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def run():
    if not MODEL_PATH.exists():
        print(f"{RED}[!] Model not found: {MODEL_PATH}{RESET}")
        print(f"    Download: huggingface-cli download OpenVINO/Qwen3-1.7B-int8-ov --local-dir {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading refiner from {MODEL_PATH} on {DEVICE}...")
    refiner = TextRefiner(MODEL_PATH, device=DEVICE)

    if not refiner._ready.wait(timeout=TIMEOUT):
        print(f"{RED}[!] Refiner failed to load within {TIMEOUT:.0f}s{RESET}")
        sys.exit(1)

    if refiner._pipeline is None:
        print(f"{RED}[!] Refiner pipeline is None - load failed, check logs above{RESET}")
        sys.exit(1)

    print(f"\nChat method: {refiner._chat_method}")
    print("=" * 70)

    passed = 0
    for desc, transcript, context in TESTS:
        print(f"\n{YELLOW}TEST:{RESET} {desc}")
        print(f"  IN:  {transcript!r}")
        t0 = time.monotonic()
        result = refiner.refine(transcript, context_hint=context, timeout_s=TIMEOUT)
        dt = time.monotonic() - t0
        changed = result != transcript
        colour = GREEN if changed else RED
        print(f"  OUT: {colour}{result!r}{RESET}  ({dt:.2f}s)")
        if changed:
            passed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(TESTS)} produced a different output from input")
    print("(Punctuation-only tests may correctly return near-identical text)")


if __name__ == "__main__":
    run()
