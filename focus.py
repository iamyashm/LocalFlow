"""Win32 foreground-window capture + restore.

Lets us remember which app the user was in when they started dictating, and
force focus back to it before we paste — so paste lands in the right app
even if focus drifted while we were transcribing.
"""
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetFocus.restype = wintypes.HWND
user32.SendMessageW.argtypes = [wintypes.HWND, ctypes.c_uint,
                                wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = ctypes.c_long

kernel32 = ctypes.windll.kernel32
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

SW_RESTORE = 9
WM_PASTE   = 0x0302


def get_foreground_hwnd():
    try:
        return user32.GetForegroundWindow()
    except Exception:
        return None


def get_window_title(hwnd):
    if not hwnd:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value
    except Exception:
        return ""


def get_focused_control(target_hwnd):
    """Return the HWND of the focused control inside target_hwnd, falling back
    to target_hwnd itself if focus can't be queried. Uses AttachThreadInput so
    we can read another process's input state.
    """
    if not target_hwnd or not user32.IsWindow(target_hwnd):
        return target_hwnd
    target_tid = user32.GetWindowThreadProcessId(target_hwnd, None)
    our_tid = kernel32.GetCurrentThreadId()
    if not target_tid or target_tid == our_tid:
        try:
            return user32.GetFocus() or target_hwnd
        except Exception:
            return target_hwnd
    try:
        user32.AttachThreadInput(our_tid, target_tid, True)
        try:
            focused = user32.GetFocus()
        finally:
            user32.AttachThreadInput(our_tid, target_tid, False)
        return focused or target_hwnd
    except Exception:
        return target_hwnd


def send_wm_paste(target_hwnd):
    """Send WM_PASTE to the focused control inside target_hwnd. Returns True
    if the message was dispatched (we can't verify the app honored it).

    This bypasses keyboard simulation entirely, so it works correctly even
    when the user is holding a modifier key (Ctrl, Win, etc.) as part of a
    push-to-talk combo.
    """
    if not target_hwnd:
        target_hwnd = user32.GetForegroundWindow()
    if not target_hwnd:
        return False
    try:
        focused = get_focused_control(target_hwnd) or target_hwnd
        user32.SendMessageW(focused, WM_PASTE, 0, 0)
        return True
    except Exception:
        return False


def restore_foreground(hwnd):
    """Force focus to hwnd.

    The AttachThreadInput trick: attach OUR thread's input queue to the
    foreground thread's queue, which lets us call SetForegroundWindow
    successfully.  The previous code incorrectly attached (target_tid,
    fg_tid) — that gives the TARGET app permission, not ours.
    """
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)

        current = user32.GetForegroundWindow()
        if current == hwnd:
            return True

        fg_tid  = user32.GetWindowThreadProcessId(current, None) if current else 0
        our_tid = kernel32.GetCurrentThreadId()

        if fg_tid and fg_tid != our_tid:
            user32.AttachThreadInput(our_tid, fg_tid, True)
            try:
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
            finally:
                user32.AttachThreadInput(our_tid, fg_tid, False)
        else:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)

        return user32.GetForegroundWindow() == hwnd
    except Exception:
        return False
