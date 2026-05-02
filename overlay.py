"""Floating audio overlay — Apple-style pill at bottom-center.

Borderless, always-on-top, click-through capsule.

Public API kept compatible with the old tkinter version:
  - AudioOverlay.available
  - AudioOverlay()
  - set_level(rms)
  - set_state(state)
  - connect_stop(slot)
  - connect_cancel(slot)
  - run()
  - shutdown()

States:
  - "hidden"
  - "recording"
  - "handsfree"   ← expands pill; shows ✕/■ buttons
  - "processing"
"""

import math
import sys
import threading
import ctypes

try:
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, Slot
    from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen, QRegion
    from PySide6.QtWidgets import QApplication, QWidget
    _HAS_QT = True
except ImportError:
    _HAS_QT = False
    QApplication = None
    QWidget = object


# -- Geometry ----------------------------------------------------------------
WIDTH      = 120   # visible pill width (bar section)
HEIGHT     = 30    # visible pill height
EDGE_PAD   = 0    # transparent outer margin so AA isn't clipped at window bounds

WINDOW_W   = WIDTH + EDGE_PAD * 2
WINDOW_H   = HEIGHT + EDGE_PAD * 2

# Extra width added to the pill in hands-free mode for stop/cancel buttons
HF_BUTTON_W = 60
HF_WINDOW_W = WIDTH + HF_BUTTON_W + EDGE_PAD * 2

N_BARS     = 7
BAR_W      = 3
BAR_GAP    = 4
BAR_MAX_H  = HEIGHT - 12
BOTTOM_PAD = 60
TICK_MS    = 33

# -- Colors ------------------------------------------------------------------
KEY_COLOR  = "#010102"   # kept for compatibility, not used by Qt version
PILL_COLOR = "#2c2c2e"
BAR_RECORD = "#f5f5f7"
BAR_HF     = "#27d2a7"
BAR_PROC   = "#ff9500"
BTN_CANCEL = "#ff453a"   # red  ✕
BTN_STOP   = "#30d158"   # green ■


if _HAS_QT:
    class _OverlayWidget(QWidget):
        state_changed    = Signal(str)
        level_changed    = Signal(float)
        stop_requested   = Signal()
        cancel_requested = Signal()

        def __init__(self):
            flags = (
                Qt.FramelessWindowHint
                | Qt.WindowStaysOnTopHint
                | Qt.Tool
                | Qt.WindowDoesNotAcceptFocus
                | Qt.NoDropShadowWindowHint
                # No Qt.WindowTransparentForInput here: that flag makes Qt's
                # WM_NCHITTEST handler return HTTRANSPARENT unconditionally, so
                # our ctypes WS_EX_TRANSPARENT toggle would have no effect.
                # Click-through is handled purely via WS_EX_TRANSPARENT below.
            )
            super().__init__(None, flags)

            # True per-pixel transparency.
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAutoFillBackground(False)
            self.setFocusPolicy(Qt.NoFocus)

            self.resize(WINDOW_W, WINDOW_H)

            self._state = "hidden"
            self._phase = 0.0
            self._level = 0.0
            self._raw_level = 0.0
            self._level_lock = threading.Lock()
            self._native_tweaks_applied = False

            self.state_changed.connect(self._apply_state)
            self.level_changed.connect(self._apply_level)

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(TICK_MS)

            self._reposition_bottom_center()

            # Force native handle creation early, then apply Windows compositor hints.
            self.winId()
            self._apply_windows_compositor_hints()
            self._set_transparent_for_input(True)

            self.hide()

        # -------------------------------------------------------------------
        # Native window shaping / compositor hints
        # -------------------------------------------------------------------


        def _apply_windows_compositor_hints(self):
            """Suppress native Windows border/shadow around frameless translucent window."""
            if self._native_tweaks_applied:
                return
            if sys.platform != "win32":
                return

            try:
                hwnd = int(self.winId())
                dwmapi = ctypes.windll.dwmapi

                DWMWA_NCRENDERING_POLICY = 2
                DWMNCRP_DISABLED = 1

                # Windows 11 compositor hints
                DWMWA_WINDOW_CORNER_PREFERENCE = 33
                DWMWCP_DONOTROUND = 1
                DWMWA_BORDER_COLOR = 34
                DWMWA_COLOR_NONE = 0xFFFFFFFE

                policy = ctypes.c_int(DWMNCRP_DISABLED)
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_NCRENDERING_POLICY,
                    ctypes.byref(policy),
                    ctypes.sizeof(policy),
                )

                corners = ctypes.c_int(DWMWCP_DONOTROUND)
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_WINDOW_CORNER_PREFERENCE,
                    ctypes.byref(corners),
                    ctypes.sizeof(corners),
                )

                border = ctypes.c_uint(DWMWA_COLOR_NONE)
                dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_BORDER_COLOR,
                    ctypes.byref(border),
                    ctypes.sizeof(border),
                )

                self._native_tweaks_applied = True
            except Exception:
                pass

        def _set_transparent_for_input(self, transparent: bool):
            """Toggle OS-level click-through via WS_EX_TRANSPARENT.

            transparent=True  → clicks pass through to windows below (recording/processing)
            transparent=False → overlay receives clicks (HF mode stop/cancel buttons)

            WS_EX_NOACTIVATE is preserved by Qt.WindowDoesNotAcceptFocus so the
            overlay never steals keyboard focus from the dictation target.
            """
            if sys.platform != "win32":
                return
            try:
                hwnd = int(self.winId())
                user32 = ctypes.windll.user32
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                current = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                new_style = (current | WS_EX_TRANSPARENT) if transparent else (current & ~WS_EX_TRANSPARENT)
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            except Exception:
                pass

        def _show_noactivate(self):
            """Show the window and assert topmost Z-order without stealing focus.

            Qt's plain show() + raise_() call BringWindowToTop() which can
            activate the window and steal keyboard focus from the dictation
            target app — causing ctrl+v to paste into the overlay instead.
            SetWindowPos with SWP_NOACTIVATE prevents that.
            """
            self.show()
            if sys.platform != "win32":
                return
            try:
                hwnd = int(self.winId())
                HWND_TOPMOST  = -1
                SWP_NOMOVE    = 0x0002
                SWP_NOSIZE    = 0x0001
                SWP_NOACTIVATE = 0x0010
                ctypes.windll.user32.SetWindowPos(
                    hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE,
                )
            except Exception:
                pass

        # -------------------------------------------------------------------
        # State / level handling
        # -------------------------------------------------------------------

        @Slot(float)
        def _apply_level(self, rms: float):
            with self._level_lock:
                self._raw_level = max(0.0, float(rms))

        @Slot(str)
        def _apply_state(self, state: str):
            self._state = state
            if state == "hidden":
                self._set_transparent_for_input(True)
                self.resize(WINDOW_W, WINDOW_H)
                #self._update_window_mask()
                self.hide()
                return

            if state == "handsfree":
                self.resize(HF_WINDOW_W, WINDOW_H)
                self._set_transparent_for_input(False)
            else:
                self.resize(WINDOW_W, WINDOW_H)
                self._set_transparent_for_input(True)

            self._reposition_bottom_center()
            self._show_noactivate()
            self._apply_windows_compositor_hints()
            self.update()

        def _reposition_bottom_center(self):
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                return
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + geo.height() - self.height() - BOTTOM_PAD
            self.move(x, y)

        def _bar_color(self) -> QColor:
            if self._state == "recording":
                return QColor(BAR_RECORD)
            if self._state == "handsfree":
                return QColor(BAR_HF)
            return QColor(BAR_PROC)

        def _tick(self):
            if self._state == "hidden":
                return

            with self._level_lock:
                raw = self._raw_level

            target = min(1.0, (raw / 0.15) ** 0.7) if raw > 0 else 0.0
            self._level += (target - self._level) * 0.30
            self._phase = (self._phase + 0.16) % (2 * math.pi)
            self.update()

        # -------------------------------------------------------------------
        # Painting / input
        # -------------------------------------------------------------------

        def paintEvent(self, event):
            if self._state == "hidden":
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            # Clear the whole widget to fully transparent first.
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            hf = self._state == "handsfree"
            pill_w = WIDTH + (HF_BUTTON_W if hf else 0)

            # Draw pill directly; do NOT use the masked/window-shape path.
            rect = QRectF(0.0, 0.0, float(pill_w), float(HEIGHT))
            radius = rect.height() / 2.0

            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(PILL_COLOR))
            painter.drawPath(path)

            # Animated bars centered inside the left (bar) section.
            total_w = N_BARS * BAR_W + (N_BARS - 1) * BAR_GAP
            x0 = (WIDTH - total_w) / 2.0
            mid_y = HEIGHT / 2.0
            painter.setBrush(self._bar_color())

            for i in range(N_BARS):
                if self._state in ("recording", "handsfree"):
                    offset = math.sin(self._phase + i * 0.55) * 0.5 + 0.5
                    h = max(2.0, BAR_MAX_H * self._level * (0.4 + 0.6 * offset))
                else:
                    pulse = math.sin(self._phase + i * 0.45) * 0.5 + 0.5
                    h = max(2.0, BAR_MAX_H * 0.45 * pulse)

                bx = x0 + i * (BAR_W + BAR_GAP)
                by = mid_y - h / 2.0
                bar_rect = QRectF(bx, by, BAR_W, h)
                painter.drawRoundedRect(bar_rect, BAR_W / 2.0, BAR_W / 2.0)

            # HF mode: divider + ✕ cancel + ■ stop buttons.
            if hf:
                div_x = float(WIDTH)
                cancel_cx = div_x + HF_BUTTON_W / 4.0
                stop_cx   = div_x + 3.0 * HF_BUTTON_W / 4.0

                painter.setPen(QPen(QColor("#444446"), 1.0))
                painter.drawLine(
                    QPointF(div_x, 5.0),
                    QPointF(div_x, HEIGHT - 5.0),
                )

                s = 4.5
                painter.setPen(QPen(QColor(BTN_CANCEL), 2.0, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(
                    QPointF(cancel_cx - s, mid_y - s),
                    QPointF(cancel_cx + s, mid_y + s),
                )
                painter.drawLine(
                    QPointF(cancel_cx + s, mid_y - s),
                    QPointF(cancel_cx - s, mid_y + s),
                )

                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(BTN_STOP))
                sq = 7.0
                painter.drawRect(QRectF(stop_cx - sq / 2.0, mid_y - sq / 2.0, sq, sq))

        def mousePressEvent(self, event):
            if self._state != "handsfree":
                return
            x = event.position().x()
            btn_start = float(WIDTH + EDGE_PAD)
            if x < btn_start:
                return
            btn_x = x - btn_start
            if btn_x < HF_BUTTON_W / 2.0:
                self.cancel_requested.emit()
            else:
                self.stop_requested.emit()


class AudioOverlay:
    available = _HAS_QT

    def __init__(self):
        if not _HAS_QT:
            raise RuntimeError("PySide6 not available")

        self._app = QApplication.instance()
        self._owns_app = self._app is None

        if self._app is None:
            self._app = QApplication(sys.argv)
            self._app.setQuitOnLastWindowClosed(False)

        self._widget = _OverlayWidget()

    # -----------------------------------------------------------------------
    # Public API (thread-safe)
    # -----------------------------------------------------------------------

    def set_level(self, rms):
        self._widget.level_changed.emit(float(rms))

    def set_state(self, state):
        self._widget.state_changed.emit(state)

    def connect_stop(self, slot):
        """Connect a callable to the stop (■) button — ends session and pastes."""
        self._widget.stop_requested.connect(slot)

    def connect_cancel(self, slot):
        """Connect a callable to the cancel (✕) button — ends session, no paste."""
        self._widget.cancel_requested.connect(slot)

    # -----------------------------------------------------------------------
    # Compatibility methods
    # -----------------------------------------------------------------------

    def run(self):
        if self._owns_app:
            self._app.exec()

    def shutdown(self):
        self._widget.close()
        if self._owns_app:
            self._app.quit()