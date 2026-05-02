"""Insert recognized text at the cursor position of the currently focused app.

Sends a synthetic Ctrl+V via the `keyboard` library — works in every text
surface (browsers, Electron apps, IDEs, native edit controls), unlike
WM_PASTE which only native Win32 controls honor.

Now that the PTT hotkey is a single non-modifier-conflicting key
(default: Right Ctrl), the synthetic Left-Ctrl events injected here don't
collide with any held modifier — the original conflict only existed when
the hotkey itself contained Ctrl/Win.
"""
import time

import keyboard
import pyperclip

from focus import restore_foreground


PRE_PASTE_DELAY_S = 0.030
POST_PASTE_DELAY_S = 0.20


def _send_paste():
    """Send Ctrl+V. `keyboard.send` defaults to left-ctrl scan code, which
    has a different scan code from right-ctrl — so even if the user is
    holding right-ctrl as the PTT key, our hook (bound to right-ctrl only)
    won't see this synthetic event as the hotkey.
    """
    try:
        keyboard.send("ctrl+v")
        return True
    except Exception as e:
        print(f"[inject error] keyboard.send failed: {type(e).__name__}: {e}")
        return False


def inject_text(text, target_hwnd=None, restore_clipboard=True):
    """Paste `text` into the focused app. If `target_hwnd` is given, focus is
    forced back to it first.

    Returns True if the paste shortcut was dispatched, False otherwise.
    """
    text = (text or "").strip()
    if not text:
        return False

    prev_clip = None
    if restore_clipboard:
        try:
            prev_clip = pyperclip.paste()
        except Exception:
            prev_clip = None

    try:
        pyperclip.copy(text)
    except Exception as e:
        print(f"[inject error] clipboard write failed: {type(e).__name__}: {e}")
        return False

    if target_hwnd:
        restore_foreground(target_hwnd)

    time.sleep(PRE_PASTE_DELAY_S)

    ok = _send_paste()
    if not ok:
        return False

    if restore_clipboard and prev_clip is not None:
        time.sleep(POST_PASTE_DELAY_S)
        try:
            pyperclip.copy(prev_clip)
        except Exception:
            pass

    return True


def paste_delta_fast(text, target_hwnd=None):
    """Paste a streaming delta. No clipboard backup/restore (the next delta
    arrives seconds later) and no focus shift (the target app is already
    focused while the user holds PTT).

    NOTE: leading whitespace is intentionally preserved — the caller adds a
    space separator between chunks, and stripping it here would lose word gaps.
    """
    if not text or not text.strip():
        return False
    try:
        pyperclip.copy(text)
    except Exception as e:
        print(f"[inject error] delta clipboard write failed: {type(e).__name__}: {e}")
        return False
    time.sleep(PRE_PASTE_DELAY_S)
    return _send_paste()
