"""
╔══════════════════════════════════════════════════════════════════╗
║        IMU 3D ORIENTATION VISUALIZER  —  PROFESSIONAL EDITION   ║
║        MPU6050 + HMC5883L  |  Madgwick AHRS  |  ~60 FPS         ║
╠══════════════════════════════════════════════════════════════════╣
║  FIXES v3:                                                       ║
║  • ViewBox overflow crash fixed (explicit ranges, no auto-range) ║
║  • Stats throttled to every 30 frames (no main-thread freeze)    ║
║  • Empty-data guards on all plot refresh paths                   ║
╚══════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install pyserial pyqtgraph PyQt5 PyOpenGL numpy

Usage:
    python imu_visualizer.py
    → Close Arduino Serial Monitor first!
"""

import sys, re, time, threading, math, csv, warnings
from collections import deque
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports

# Suppress the pyqtgraph ViewBox overflow RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="pyqtgraph")

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
HEX_BG      = "#0b0d12"
HEX_PANEL   = "#111520"
HEX_BORDER  = "#1e2538"
HEX_TEXT    = "#c8d0ea"
HEX_DIM     = "#4a5270"
HEX_CYAN    = "#00f0ff"
HEX_ORANGE  = "#ff6b2b"
HEX_GREEN   = "#2dff6e"
HEX_PURPLE  = "#c060ff"
HEX_YELLOW  = "#ffe066"
HEX_RED     = "#ff3c3c"

C_BG        = (0.043, 0.051, 0.071, 1.0)
C_X         = (1.0,  0.42, 0.17, 1.0)
C_Y         = (0.18, 1.0,  0.43, 1.0)
C_Z         = (0.0,  0.94, 1.0,  1.0)
C_BOX_EDGE  = (0.0,  0.94, 1.0,  0.3)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BAUD_RATE    = 115200
TRAIL_LEN    = 400
HISTORY_LEN  = 500
TIMER_MS     = 16        # ~60 fps
STATS_EVERY  = 30        # recompute heavy stats every N frames
BOX_SCALE    = (1.8, 0.9, 0.32)

# ══════════════════════════════════════════════════════════════
#  PORT PICKER
# ══════════════════════════════════════════════════════════════
def pick_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise RuntimeError("No serial ports found.")
    print("\n┌─ Available Ports ─────────────────────────┐")
    for i, p in enumerate(ports):
        print(f"│  [{i}]  {p.device:<12} {p.description[:30]}")
    print("└───────────────────────────────────────────┘")
    if len(ports) == 1:
        print(f"  Auto-selecting: {ports[0].device}")
        return ports[0].device
    return ports[int(input("  Select port index: "))].device


# ══════════════════════════════════════════════════════════════
#  SHARED STATE
# ══════════════════════════════════════════════════════════════
class IMUState:
    def __init__(self):
        self.roll = self.pitch = self.yaw = 0.0
        self.lock    = threading.Lock()
        self.trail   = deque(maxlen=TRAIL_LEN)
        self.history = deque(maxlen=HISTORY_LEN)   # (t, roll, pitch, yaw)
        self.fps_serial = 0.0
        self.connected  = False
        self.t0 = time.time()

state = IMUState()

PATTERN = re.compile(
    r"Roll:\s*([-\d.]+)\s+Pitch:\s*([-\d.]+)\s+Yaw:\s*([-\d.]+)"
)

def serial_reader(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        state.connected = True
        print(f"\n  ✔  Connected to {port} @ {baud} baud\n")
    except Exception as e:
        print(f"\n  ✘  {e}\n  → Close Arduino Serial Monitor first.\n")
        return
    t0 = time.time()
    count = 0
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            m = PATTERN.search(line)
            if m:
                r = float(m.group(1))
                p = float(m.group(2))
                y = float(m.group(3))
                now = time.time()
                with state.lock:
                    state.roll  = r
                    state.pitch = p
                    state.yaw   = y
                    state.trail.append((
                        math.radians(r), math.radians(p), math.radians(y)
                    ))
                    state.history.append((now - state.t0, r, p, y))
                count += 1
                if now - t0 >= 1.0:
                    state.fps_serial = count / (now - t0)
                    count = 0
                    t0 = now
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  MATH
# ══════════════════════════════════════════════════════════════
def euler_to_rot(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,     cp*sr,             cp*cr            ]
    ], dtype=np.float64)

def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    return (
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    )

def make_box_geometry():
    sx, sy, sz = [s * 0.5 for s in BOX_SCALE]
    v = np.array([
        [-sx,-sy,-sz],[ sx,-sy,-sz],[ sx, sy,-sz],[-sx, sy,-sz],
        [-sx,-sy, sz],[ sx,-sy, sz],[ sx, sy, sz],[-sx, sy, sz],
    ], dtype=np.float32)
    f = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [1,2,6],[1,6,5],[0,3,7],[0,7,4],
    ], dtype=np.uint32)
    fc = np.array([
        [0.06,0.14,0.32,0.92],[0.06,0.14,0.32,0.92],
        [0.12,0.26,0.52,0.92],[0.12,0.26,0.52,0.92],
        [0.07,0.12,0.26,0.92],[0.07,0.12,0.26,0.92],
        [0.07,0.12,0.26,0.92],[0.07,0.12,0.26,0.92],
        [0.09,0.16,0.33,0.92],[0.09,0.16,0.33,0.92],
        [0.09,0.16,0.33,0.92],[0.09,0.16,0.33,0.92],
    ], dtype=np.float32)
    return v, f, fc


# ══════════════════════════════════════════════════════════════
#  ARTIFICIAL HORIZON WIDGET
# ══════════════════════════════════════════════════════════════
class ArtificialHorizon(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._roll  = 0.0
        self._pitch = 0.0
        self.setMinimumSize(160, 160)

    def update_attitude(self, roll_deg, pitch_deg):
        self._roll  = roll_deg
        self._pitch = pitch_deg
        self.update()

    def paintEvent(self, _):
        p  = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h  = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r  = min(cx, cy) - 4

        clip = QtGui.QPainterPath()
        clip.addEllipse(cx - r, cy - r, 2*r, 2*r)
        p.setClipPath(clip)

        p.translate(cx, cy)
        p.rotate(-self._roll)

        pitch_px = self._pitch * 2.0

        # Sky
        p.fillRect(QtCore.QRectF(-r, -r*2 + pitch_px, 2*r, 2*r),
                   QtGui.QColor("#1a3a5c"))
        # Ground
        p.fillRect(QtCore.QRectF(-r, pitch_px, 2*r, 2*r),
                   QtGui.QColor("#4a2d10"))
        # Horizon
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_TEXT), 2))
        p.drawLine(int(-r), int(pitch_px), int(r), int(pitch_px))

        # Pitch ladder
        p.setFont(QtGui.QFont("monospace", 6))
        for deg in range(-40, 50, 10):
            if deg == 0:
                continue
            y_off  = pitch_px - deg * 2.0
            if abs(y_off) > r:
                continue
            lw = r * (0.35 if deg % 20 == 0 else 0.22)
            p.setPen(QtGui.QPen(QtGui.QColor(HEX_TEXT), 1))
            p.drawLine(int(-lw), int(y_off), int(lw), int(y_off))
            p.drawText(int(lw + 3), int(y_off + 4), f"{abs(deg)}")

        p.resetTransform()
        p.translate(cx, cy)

        # Roll arc + ticks
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_DIM), 1))
        p.drawArc(int(-r*0.92), int(-r*0.92), int(r*1.84), int(r*1.84), 0, 360*16)
        for angle in [-60,-45,-30,-20,-10,0,10,20,30,45,60]:
            a_rad = math.radians(angle - 90)
            r1, r2 = r*0.88, r*0.95
            p.drawLine(
                int(r1*math.cos(a_rad)), int(r1*math.sin(a_rad)),
                int(r2*math.cos(a_rad)), int(r2*math.sin(a_rad))
            )

        # Roll pointer
        roll_rad = math.radians(-self._roll - 90)
        tip  = (r*0.83*math.cos(roll_rad), r*0.83*math.sin(roll_rad))
        l    = (r*0.94*math.cos(roll_rad-0.07), r*0.94*math.sin(roll_rad-0.07))
        rr   = (r*0.94*math.cos(roll_rad+0.07), r*0.94*math.sin(roll_rad+0.07))
        poly = QtGui.QPolygon([
            QtCore.QPoint(int(tip[0]), int(tip[1])),
            QtCore.QPoint(int(l[0]),   int(l[1])),
            QtCore.QPoint(int(rr[0]),  int(rr[1])),
        ])
        p.setBrush(QtGui.QColor(HEX_YELLOW))
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(poly)

        # Aircraft symbol
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_YELLOW), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawLine(-int(r*0.35), 0, -int(r*0.12), 0)
        p.drawLine( int(r*0.12), 0,  int(r*0.35), 0)
        p.drawLine(0, -int(r*0.12), 0, int(r*0.12))
        p.drawEllipse(-5, -5, 10, 10)

        # Outer ring
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_BORDER), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(-r, -r, 2*r, 2*r)

        p.resetTransform()
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_DIM), 1))
        p.setFont(QtGui.QFont("monospace", 7))
        p.drawText(4, h - 4, "ATTITUDE")
        p.end()


# ══════════════════════════════════════════════════════════════
#  COMPASS ROSE WIDGET
# ══════════════════════════════════════════════════════════════
class CompassRose(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._yaw = 0.0
        self.setMinimumSize(160, 160)

    def update_heading(self, yaw_deg):
        self._yaw = yaw_deg
        self.update()

    def paintEvent(self, _):
        p   = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h  = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r  = min(cx, cy) - 4

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(HEX_PANEL))
        p.drawEllipse(cx - r, cy - r, 2*r, 2*r)

        p.translate(cx, cy)
        for deg in range(0, 360, 5):
            a     = math.radians(deg)
            long  = (deg % 30 == 0)
            r1    = r * (0.78 if long else 0.83)
            r2    = r * 0.92
            p.setPen(QtGui.QPen(
                QtGui.QColor(HEX_TEXT if long else HEX_DIM), 1
            ))
            p.drawLine(
                int(r1*math.sin(a)), int(-r1*math.cos(a)),
                int(r2*math.sin(a)), int(-r2*math.cos(a))
            )

        font = QtGui.QFont("monospace", 8, QtGui.QFont.Bold)
        p.setFont(font)
        for label, angle, color in [
            ("N",   0,   HEX_RED),
            ("E",  90,   HEX_TEXT),
            ("S", 180,   HEX_TEXT),
            ("W", 270,   HEX_TEXT),
        ]:
            a   = math.radians(angle)
            lx  = int(r * 0.65 * math.sin(a))
            ly  = int(-r * 0.65 * math.cos(a))
            p.setPen(QtGui.QPen(QtGui.QColor(color), 1))
            fm  = QtGui.QFontMetrics(font)
            bw  = fm.horizontalAdvance(label)
            p.drawText(lx - bw//2, ly + 4, label)

        # Needle
        a         = math.radians(self._yaw)
        nlen      = r * 0.55
        tip_x     = int(nlen * math.sin(a))
        tip_y     = int(-nlen * math.cos(a))
        tail_x    = int(-nlen * 0.4 * math.sin(a))
        tail_y    = int( nlen * 0.4 * math.cos(a))
        perp_x    = int(r * 0.07 * math.cos(a))
        perp_y    = int(r * 0.07 * math.sin(a))

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(HEX_CYAN))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(tip_x,   tip_y),
            QtCore.QPoint(perp_x,  perp_y),
            QtCore.QPoint(-perp_x, -perp_y),
        ]))
        p.setBrush(QtGui.QColor(HEX_RED))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(tail_x,  tail_y),
            QtCore.QPoint(perp_x,  perp_y),
            QtCore.QPoint(-perp_x, -perp_y),
        ]))
        p.setBrush(QtGui.QColor(HEX_TEXT))
        p.drawEllipse(-4, -4, 8, 8)

        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_BORDER), 2))
        p.drawEllipse(-r, -r, 2*r, 2*r)

        p.resetTransform()
        p.setPen(QtGui.QPen(QtGui.QColor(HEX_CYAN), 1))
        font2 = QtGui.QFont("monospace", 9, QtGui.QFont.Bold)
        p.setFont(font2)
        hdg = f"{self._yaw:05.1f}°"
        fm2 = QtGui.QFontMetrics(font2)
        p.drawText(cx - fm2.horizontalAdvance(hdg)//2, h - 6, hdg)

        p.setPen(QtGui.QPen(QtGui.QColor(HEX_DIM), 1))
        p.setFont(QtGui.QFont("monospace", 7))
        p.drawText(4, h - 4, "HEADING")
        p.end()


# ══════════════════════════════════════════════════════════════
#  SCROLLING TIME-SERIES PLOT  — overflow-safe
# ══════════════════════════════════════════════════════════════
class TimeSeriesPlot(pg.PlotWidget):
    def __init__(self, label, color, y_min, y_max):
        super().__init__()
        self._y_min = float(y_min)
        self._y_max = float(y_max)

        self.setBackground(HEX_PANEL)
        self.setTitle(label, color=HEX_TEXT, size="8pt")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.getPlotItem().hideButtons()

        # ── KEY FIX: disable auto-range completely ──────────────
        self.disableAutoRange()
        self.setXRange(0.0, 10.0, padding=0)          # safe initial range
        self.setYRange(self._y_min, self._y_max, padding=0)
        self.setMouseEnabled(False, False)
        self.setLimits(
            xMin=0, xMax=1e6,
            yMin=self._y_min - 5, yMax=self._y_max + 5
        )
        # ────────────────────────────────────────────────────────

        for ax in ["bottom", "left", "top", "right"]:
            self.getAxis(ax).setPen(HEX_BORDER)
        self.getAxis("bottom").setTextPen(HEX_DIM)
        self.getAxis("left").setTextPen(HEX_TEXT)
        self.getAxis("bottom").setLabel("time (s)", color=HEX_DIM,
                                        **{"font-size": "7pt"})
        self.getAxis("left").setLabel("deg", color=color,
                                      **{"font-size": "7pt"})

        self.addLine(y=0, pen=pg.mkPen(HEX_DIM, width=1,
                                        style=QtCore.Qt.DotLine))
        self.addLine(y= 45, pen=pg.mkPen(HEX_YELLOW, width=1,
                                          style=QtCore.Qt.DashLine))
        self.addLine(y=-45, pen=pg.mkPen(HEX_YELLOW, width=1,
                                          style=QtCore.Qt.DashLine))

        self._curve = self.plot([], [], pen=pg.mkPen(color, width=1.8))

    def refresh(self, ts, vals):
        # ── Guard: need at least 2 finite points ───────────────
        if len(ts) < 2:
            return
        t = np.asarray(ts,   dtype=np.float64)
        v = np.asarray(vals, dtype=np.float64)

        # Drop any NaN / inf that could cause overflow
        mask = np.isfinite(t) & np.isfinite(v)
        if mask.sum() < 2:
            return
        t, v = t[mask], v[mask]

        t_end   = float(t[-1])
        t_start = max(float(t[0]), t_end - 10.0)

        # Only update range if values are valid floats
        if math.isfinite(t_start) and math.isfinite(t_end) and t_end > t_start:
            self.setXRange(t_start, t_end, padding=0)

        self._curve.setData(t, v)


# ══════════════════════════════════════════════════════════════
#  STAT LABEL HELPER
# ══════════════════════════════════════════════════════════════
def _stat_label(name, color=HEX_TEXT):
    lbl = QtWidgets.QLabel(f"{name}:  ---")
    lbl.setStyleSheet(
        f"color:{color}; font-family:monospace; font-size:9px; padding:1px 4px;"
    )
    return lbl


# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "IMU 3D Orientation Visualizer — Professional Edition"
        )
        self.resize(1600, 900)
        self.setStyleSheet(
            f"background-color:{HEX_BG}; color:{HEX_TEXT}; font-family:monospace;"
        )

        self._paused     = False
        self._recording  = False
        self._csv_file   = None
        self._csv_writer = None
        self._frame      = 0          # frame counter for throttling

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── title ──────────────────────────────────────────────
        title_lbl = QtWidgets.QLabel(
            "IMU · 3D ORIENTATION VISUALIZER  —  Madgwick AHRS  "
            "(MPU6050 + HMC5883L)"
        )
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)
        title_lbl.setStyleSheet(
            f"color:{HEX_CYAN}; font-size:13px; font-weight:bold; padding:2px;"
        )
        root.addWidget(title_lbl)

        # ── main content row ───────────────────────────────────
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(6)
        root.addLayout(content, stretch=1)

        # LEFT — 3D view
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(4)
        content.addLayout(left_col, stretch=5)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=5.5, elevation=22, azimuth=45)
        self.gl_view.setBackgroundColor(
            QtGui.QColor(
                int(C_BG[0]*255), int(C_BG[1]*255), int(C_BG[2]*255)
            )
        )
        self.gl_view.setStyleSheet(
            f"border:1px solid {HEX_BORDER}; border-radius:4px;"
        )
        left_col.addWidget(self.gl_view, stretch=1)
        self._build_3d_scene()

        # MIDDLE — time-series
        mid_col = QtWidgets.QVBoxLayout()
        mid_col.setSpacing(4)
        content.addLayout(mid_col, stretch=3)

        self.ts_roll  = TimeSeriesPlot("Roll  (°)", HEX_ORANGE, -185,  185)
        self.ts_pitch = TimeSeriesPlot("Pitch (°)", HEX_GREEN,   -95,   95)
        self.ts_yaw   = TimeSeriesPlot("Yaw   (°)", HEX_CYAN,     -5,  365)
        for ts in [self.ts_roll, self.ts_pitch, self.ts_yaw]:
            ts.setStyleSheet(
                f"border:1px solid {HEX_BORDER}; border-radius:4px;"
            )
            mid_col.addWidget(ts, stretch=1)

        # RIGHT — instruments + data
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(6)
        content.addLayout(right_col, stretch=2)

        inst_row = QtWidgets.QHBoxLayout()
        inst_row.setSpacing(6)
        right_col.addLayout(inst_row)

        self.ah      = ArtificialHorizon()
        self.compass = CompassRose()
        for w in [self.ah, self.compass]:
            w.setStyleSheet(
                f"background:{HEX_PANEL}; border:1px solid {HEX_BORDER};"
                "border-radius:4px;"
            )
        inst_row.addWidget(self.ah)
        inst_row.addWidget(self.compass)

        # Data panel
        data_box = QtWidgets.QGroupBox("  Live Data")
        data_box.setStyleSheet(
            f"QGroupBox {{ color:{HEX_DIM}; border:1px solid {HEX_BORDER};"
            "border-radius:4px; margin-top:8px; font-size:9px; }}"
            "QGroupBox::title { subcontrol-origin:margin;"
            "subcontrol-position:top left; padding:0 4px; }"
        )
        dl = QtWidgets.QVBoxLayout(data_box)
        dl.setSpacing(2)
        dl.setContentsMargins(6, 10, 6, 6)

        self.lbl_q0   = _stat_label("Q0 (w)",       HEX_PURPLE)
        self.lbl_q1   = _stat_label("Q1 (x)",       HEX_ORANGE)
        self.lbl_q2   = _stat_label("Q2 (y)",       HEX_GREEN)
        self.lbl_q3   = _stat_label("Q3 (z)",       HEX_CYAN)
        self.lbl_tilt = _stat_label("Tilt  |°|",    HEX_YELLOW)
        self.lbl_warn = QtWidgets.QLabel("")
        self.lbl_warn.setStyleSheet(
            f"color:{HEX_RED}; font-family:monospace; font-size:9px;"
            "font-weight:bold; padding:1px 4px;"
        )

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color:{HEX_BORDER};")

        self.lbl_r_minmax = _stat_label("Roll   min/max", HEX_ORANGE)
        self.lbl_p_minmax = _stat_label("Pitch  min/max", HEX_GREEN)
        self.lbl_y_minmax = _stat_label("Yaw    min/max", HEX_CYAN)
        self.lbl_r_std    = _stat_label("Roll   σ",       HEX_ORANGE)
        self.lbl_p_std    = _stat_label("Pitch  σ",       HEX_GREEN)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setStyleSheet(f"color:{HEX_BORDER};")

        self.lbl_samples = _stat_label("Samples", HEX_DIM)
        self.lbl_uptime  = _stat_label("Uptime",  HEX_DIM)

        for w in [
            self.lbl_q0, self.lbl_q1, self.lbl_q2, self.lbl_q3,
            self.lbl_tilt, self.lbl_warn, sep,
            self.lbl_r_minmax, self.lbl_p_minmax, self.lbl_y_minmax,
            self.lbl_r_std, self.lbl_p_std, sep2,
            self.lbl_samples, self.lbl_uptime,
        ]:
            dl.addWidget(w)
        dl.addStretch()
        right_col.addWidget(data_box, stretch=1)

        # ── HUD bar ────────────────────────────────────────────
        hud = QtWidgets.QHBoxLayout()
        hud.setContentsMargins(4, 0, 4, 0)
        root.addLayout(hud)

        self.lbl_status = QtWidgets.QLabel("○  NO SIGNAL")
        self.lbl_rpy    = QtWidgets.QLabel(
            "ROLL  +0.00°     PITCH  +0.00°     YAW  0.00°"
        )
        self.lbl_fps    = QtWidgets.QLabel(
            "Plot  0.0 fps  |  Serial  0.0 fps"
        )

        def _hs(c):
            return (f"color:{c}; font-family:monospace;"
                    "font-size:10px; padding:2px 8px;")
        self.lbl_status.setStyleSheet(_hs(HEX_DIM))
        self.lbl_rpy.setStyleSheet(_hs(HEX_TEXT))
        self.lbl_fps.setStyleSheet(_hs(HEX_PURPLE))

        self.btn_pause  = self._btn("⏸  Pause",  HEX_YELLOW, self._toggle_pause)
        self.btn_record = self._btn("⏺  Record", HEX_RED,    self._toggle_record)
        self.btn_reset  = self._btn("↺  Reset",  HEX_CYAN,   self._reset_trail)

        hud.addWidget(self.lbl_status)
        hud.addWidget(self.lbl_rpy)
        hud.addStretch()
        hud.addWidget(self.btn_pause)
        hud.addWidget(self.btn_record)
        hud.addWidget(self.btn_reset)
        hud.addWidget(self.lbl_fps)

        # ── fps tracking ───────────────────────────────────────
        self._fps_t    = time.time()
        self._fps_n    = 0
        self._fps_plot = 0.0
        self._start_t  = time.time()

        # ── timer ──────────────────────────────────────────────
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(TIMER_MS)

    # ─────────────────────────────────────────────────────────
    def _btn(self, text, color, cb):
        b = QtWidgets.QPushButton(text)
        b.setCursor(QtCore.Qt.PointingHandCursor)
        b.clicked.connect(cb)
        b.setStyleSheet(
            f"QPushButton {{ color:{color}; background:{HEX_PANEL};"
            f"border:1px solid {color}44; border-radius:4px;"
            "padding:3px 10px; font-size:9px; font-family:monospace; }}"
            f"QPushButton:hover {{ background:{color}22; }}"
            f"QPushButton:pressed {{ background:{color}44; }}"
        )
        return b

    # ─────────────────────────────────────────────────────────
    def _build_3d_scene(self):
        grid = gl.GLGridItem()
        grid.setSize(6, 6, 6)
        grid.setSpacing(0.5, 0.5, 0.5)
        grid.setColor(QtGui.QColor(40, 48, 72, 90))
        grid.translate(0, 0, -1.0)
        self.gl_view.addItem(grid)

        # Static world-frame reference axes
        for vec, col in [([1.8,0,0], HEX_ORANGE),
                         ([0,1.8,0], HEX_GREEN),
                         ([0,0,1.8], HEX_CYAN)]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0,0,0], vec], dtype=np.float32),
                color=QtGui.QColor(col).getRgbF(),
                width=1.0, antialias=True, mode='lines'
            )
            self.gl_view.addItem(ln)

        # Board mesh
        self._bv, self._bf, self._bfc = make_box_geometry()
        self._box = gl.GLMeshItem(
            vertexes=self._bv, faces=self._bf, faceColors=self._bfc,
            smooth=False, drawEdges=True, edgeColor=C_BOX_EDGE
        )
        self.gl_view.addItem(self._box)

        # Body axes
        L = 1.45
        self._ax_x = gl.GLLinePlotItem(
            pos=np.array([[0,0,0],[L,0,0]], dtype=np.float32),
            color=C_X, width=3.0, antialias=True, mode='lines'
        )
        self._ax_y = gl.GLLinePlotItem(
            pos=np.array([[0,0,0],[0,L,0]], dtype=np.float32),
            color=C_Y, width=3.0, antialias=True, mode='lines'
        )
        self._ax_z = gl.GLLinePlotItem(
            pos=np.array([[0,0,0],[0,0,L]], dtype=np.float32),
            color=C_Z, width=3.0, antialias=True, mode='lines'
        )
        for item in [self._ax_x, self._ax_y, self._ax_z]:
            self.gl_view.addItem(item)

        # Trail
        self._trail_item = gl.GLLinePlotItem(
            pos=np.zeros((2,3), dtype=np.float32),
            color=(1.0, 0.42, 0.17, 0.0),
            width=1.8, antialias=True, mode='line_strip'
        )
        self.gl_view.addItem(self._trail_item)

        # Overlay label
        self.lbl_3d_info = QtWidgets.QLabel(
            "Sample  0 / 400\nDrag to rotate view", self.gl_view
        )
        self.lbl_3d_info.setStyleSheet(
            f"color:{HEX_DIM}; font-family:monospace;"
            "font-size:8px; background:transparent;"
        )
        self.lbl_3d_info.move(8, 8)
        self.lbl_3d_info.resize(200, 30)

    # ─────────────────────────────────────────────────────────
    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.btn_pause.setText("▶  Resume")
        else:
            self.btn_pause.setText("⏸  Pause")

    def _toggle_record(self):
        if not self._recording:
            fname = (
                "imu_log_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )
            self._csv_file   = open(fname, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["time_s", "roll_deg", "pitch_deg", "yaw_deg"]
            )
            self._recording = True
            self.btn_record.setText("⏹  Stop REC")
            self.statusBar().showMessage(f"  Recording → {fname}")
        else:
            self._recording = False
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
            self.btn_record.setText("⏺  Record")
            self.statusBar().showMessage("  Recording stopped.")

    def _reset_trail(self):
        with state.lock:
            state.trail.clear()
            state.history.clear()
        self._trail_item.setData(
            pos=np.zeros((2,3), dtype=np.float32),
            color=(1.0, 0.42, 0.17, 0.0)
        )

    # ─────────────────────────────────────────────────────────
    def _update(self):
        if self._paused:
            return

        self._frame += 1

        # ── read shared state ──────────────────────────────────
        with state.lock:
            roll_r  = math.radians(state.roll)
            pitch_r = math.radians(state.pitch)
            yaw_r   = math.radians(state.yaw)
            trail   = list(state.trail)
            history = list(state.history)
            rd, pd, yd = state.roll, state.pitch, state.yaw

        R = euler_to_rot(roll_r, pitch_r, yaw_r)
        n = len(trail)

        # ── CSV logging ────────────────────────────────────────
        if self._recording and self._csv_writer and history:
            row = history[-1]
            self._csv_writer.writerow([
                f"{row[0]:.4f}", f"{row[1]:.4f}",
                f"{row[2]:.4f}", f"{row[3]:.4f}"
            ])

        # ── 3D board ───────────────────────────────────────────
        rotated = (R @ self._bv.T).T.astype(np.float32)
        self._box.setMeshData(
            vertexes=rotated, faces=self._bf, faceColors=self._bfc
        )
        L = 1.45
        O = np.zeros(3, dtype=np.float32)
        self._ax_x.setData(pos=np.array([O, R[:,0]*L], dtype=np.float32))
        self._ax_y.setData(pos=np.array([O, R[:,1]*L], dtype=np.float32))
        self._ax_z.setData(pos=np.array([O, R[:,2]*L], dtype=np.float32))

        # ── trail ──────────────────────────────────────────────
        if n > 1:
            tips = np.array([
                (euler_to_rot(r, p, y) @ np.array([1., 0., 0.])) * 1.4
                for r, p, y in trail
            ], dtype=np.float32)
            alphas = np.linspace(0.04, 0.88, n, dtype=np.float32)
            colors = np.zeros((n, 4), dtype=np.float32)
            colors[:, 0] = 1.0
            colors[:, 1] = 0.42
            colors[:, 2] = 0.17
            colors[:, 3] = alphas
            self._trail_item.setData(pos=tips, color=colors)

        self.lbl_3d_info.setText(
            f"Sample  {n:>4} / {TRAIL_LEN}\nDrag to rotate view"
        )

        # ── time series (every frame, but guarded) ─────────────
        if len(history) >= 2:
            ts = [h[0] for h in history]
            rs = [h[1] for h in history]
            ps = [h[2] for h in history]
            ys = [h[3] for h in history]
            self.ts_roll.refresh(ts, rs)
            self.ts_pitch.refresh(ts, ps)
            self.ts_yaw.refresh(ts, ys)

        # ── instruments ────────────────────────────────────────
        self.ah.update_attitude(rd, pd)
        self.compass.update_heading(yd)

        # ── quaternion (every frame, cheap) ────────────────────
        q0, q1, q2, q3 = euler_to_quat(roll_r, pitch_r, yaw_r)
        self.lbl_q0.setText(f"Q0 (w):  {q0:+.5f}")
        self.lbl_q1.setText(f"Q1 (x):  {q1:+.5f}")
        self.lbl_q2.setText(f"Q2 (y):  {q2:+.5f}")
        self.lbl_q3.setText(f"Q3 (z):  {q3:+.5f}")

        # ── tilt magnitude (cheap) ─────────────────────────────
        val = max(-1.0, min(1.0, float(R[2, 2])))
        tilt = math.degrees(math.acos(val))
        col  = HEX_RED if tilt > 60 else (HEX_YELLOW if tilt > 30 else HEX_GREEN)
        self.lbl_tilt.setText(f"Tilt  |°|:  {tilt:.2f}°")
        self.lbl_tilt.setStyleSheet(
            f"color:{col}; font-family:monospace; font-size:9px; padding:1px 4px;"
        )

        # Gimbal lock warning
        self.lbl_warn.setText(
            "⚠  GIMBAL LOCK PROXIMITY" if abs(pd) > 80 else ""
        )

        # ── heavy stats — throttled to every STATS_EVERY frames ─
        if self._frame % STATS_EVERY == 0 and len(history) > 5:
            ra = np.array([h[1] for h in history], dtype=np.float32)
            pa = np.array([h[2] for h in history], dtype=np.float32)
            ya = np.array([h[3] for h in history], dtype=np.float32)
            self.lbl_r_minmax.setText(
                f"Roll   min/max:  {ra.min():.1f} / {ra.max():.1f}"
            )
            self.lbl_p_minmax.setText(
                f"Pitch  min/max:  {pa.min():.1f} / {pa.max():.1f}"
            )
            self.lbl_y_minmax.setText(
                f"Yaw    min/max:  {ya.min():.1f} / {ya.max():.1f}"
            )
            self.lbl_r_std.setText(f"Roll   σ:  {ra.std():.2f}°")
            self.lbl_p_std.setText(f"Pitch  σ:  {pa.std():.2f}°")
            self.lbl_samples.setText(f"Samples:  {len(history)}")
            uptime = time.time() - self._start_t
            self.lbl_uptime.setText(
                f"Uptime:   {int(uptime//60):02d}:{int(uptime%60):02d}"
            )

        # ── HUD ────────────────────────────────────────────────
        if state.connected:
            self.lbl_status.setText("●  LIVE")
            self.lbl_status.setStyleSheet(
                f"color:{HEX_GREEN}; font-family:monospace; "
                "font-size:10px; font-weight:bold; padding:2px 8px;"
            )
        rec = "  ⏺ REC" if self._recording else ""
        self.lbl_rpy.setText(
            f"ROLL {rd:+8.2f}°    PITCH {pd:+8.2f}°    YAW {yd:6.2f}°{rec}"
        )

        # ── fps ────────────────────────────────────────────────
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps_plot = self._fps_n / (now - self._fps_t)
            self._fps_n = 0
            self._fps_t = now
        self.lbl_fps.setText(
            f"Plot  {self._fps_plot:.1f} fps  |  "
            f"Serial  {state.fps_serial:.1f} fps"
        )


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*56)
    print("   IMU 3D Orientation Visualizer  —  Professional Edition")
    print("   MPU6050 + HMC5883L  |  Madgwick AHRS")
    print("="*56)

    port = pick_port()

    t = threading.Thread(
        target=serial_reader, args=(port, BAUD_RATE), daemon=True
    )
    t.start()
    time.sleep(0.5)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,     QtGui.QColor(11, 13, 18))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(200, 208, 234))
    pal.setColor(QtGui.QPalette.Base,       QtGui.QColor(17, 21, 32))
    pal.setColor(QtGui.QPalette.Text,       QtGui.QColor(200, 208, 234))
    pal.setColor(QtGui.QPalette.Button,     QtGui.QColor(17, 21, 32))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(200, 208, 234))
    app.setPalette(pal)

    win = MainWindow()
    win.statusBar().setStyleSheet(
        f"color:{HEX_DIM}; font-family:monospace; font-size:9px;"
    )
    win.show()
    sys.exit(app.exec_())
