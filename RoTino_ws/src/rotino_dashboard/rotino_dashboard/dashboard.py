"""RoTino real-time dashboard (PyQt5 + pyqtgraph).

    ros2 run rotino_dashboard dashboard

Reads the topics of wbr_controller (and the /rotino/debug + /rotino/jump_state subset of the PID
controller) and redraws at a fixed frame rate: phase LEDs, diagnostic LEDs, a sagittal sketch of the
robot with force arrows, live values and a grid of rolling plots.
"""

import math
import signal
import sys
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from rotino_dashboard.ros_bridge import RosBridge

G = 9.81
DEG = 180.0 / math.pi
MAX_POINTS = 800             # per curve after min/max decimation (peaks preserved)
FRESH = 0.25                # s of wall time after which a stream is considered stopped
LATENCY_LIMIT_MS = 100.0
WHEEL_TORQUE_LIMIT = 10.0   # wbr_controller WHEEL_TORQUE_MAX
LEG_TORQUE_LIMIT = 50.0     # wbr_controller LEG_TORQUE_MAX
FORCE_SCALE = 0.003         # m per N in the robot sketch
ZMP_TRAIL_S = 1.5           # s of ZMP history drawn as a fading trail in the top view
ZMP_TRAIL_POINTS = 90
ZMP_MARGIN_WARN = 0.4       # fraction of d/2 left before the ZMP reaches a wheel
ZMP_MARGIN_ALARM = 0.2
# live command limits: same as wbr_controller CMD_* (the controller clamps anyway)
CMD_V_MAX = 2.0
CMD_W_MAX = 1.5
CMD_H_MIN, CMD_H_MAX = -0.05, 0.04
CMD_PUSH_MAX = 6.0

BG = '#15181d'
PANEL = '#1d2128'
FG = '#d6dae0'
MUTED = '#6b7280'
C = {
    'blue': '#4ea1ff', 'orange': '#ff9f43', 'green': '#2ecc71', 'red': '#ff5c5c', 'violet': '#b48cff',
    'cyan': '#3dd6d0', 'yellow': '#f5d547', 'pink': '#ff7ab6', 'grey': '#9aa3ad',
}

PHASES = [
    ('HOLD', 'Aggancio', '#7f8c8d'),
    ('BALANCE', 'Bilanciamento', '#2ecc71'),
    ('PRELOAD', 'Accovacciamento', '#f5d547'),
    ('THRUST', 'Spinta', '#ff9f43'),
    ('FLIGHT', 'Volo', '#b48cff'),
    ('LANDING', 'Atterraggio', '#4ea1ff'),
    ('RECOVERY', 'Recupero (PID)', '#3dd6d0'),
    ('SETTLE', 'Assestamento (PID)', '#9be7a8'),
]


def col(i):
    """Column of a stream field (column 0 is time)."""
    return lambda a: a[:, 1 + i]


def deg(i):
    return lambda a: a[:, 1 + i] * DEG


# (title, unit, minimum y span, [(stream, y(array), label, colour, dashed)])
PLOTS = [
    ('Inclinazione θ', '°', 2.0, [
        ('debug', deg(1), 'θ misurato', C['blue'], False),
        ('wbr', deg(4), 'θ riferimento', C['orange'], True),
        ('odom', deg(3), 'beccheggio torso', C['grey'], False)]),
    ('Posizione s', 'm', 0.05, [
        ('debug', col(3), 's', C['blue'], False),
        ('wbr', col(2), 's riferimento', C['orange'], True)]),
    ('Velocità ṡ', 'm/s', 0.1, [
        ('debug', col(4), 'ṡ', C['blue'], False),
        ('wbr', col(8), 'ṡ riferimento', C['orange'], True)]),
    ('Altezza baricentro', 'm', 0.02, [
        ('wbr', col(9), 'z (CoM − asse)', C['blue'], False),
        ('wbr', col(10), 'z riferimento', C['orange'], True),
        ('debug', col(9), 'luce ruote', C['violet'], False)]),
    ('Forze verticali sul corpo', 'N', 10.0, [
        ('wbr', col(12), 'F_z MPC (richiesta)', C['orange'], False),
        ('leg_force', lambda a: a[:, 2] + a[:, 4], 'gambe totale (J⁻ᵀτ)', C['green'], False),
        ('leg_force', col(1), 'gamba SX', C['blue'], False),
        ('leg_force', col(3), 'gamba DX', C['cyan'], True)]),
    ('Coppie ruote', 'Nm', 1.0, [
        ('wheel_cmd', col(0), 'τ sinistra', C['blue'], False),
        ('wheel_cmd', col(1), 'τ destra', C['cyan'], False)]),
    ('Coppie gambe', 'Nm', 2.0, [
        ('leg_cmd', col(0), 'anca SX', C['blue'], False),
        ('leg_cmd', col(1), 'anca DX', C['cyan'], True),
        ('leg_cmd', col(2), 'ginocchio SX', C['orange'], False),
        ('leg_cmd', col(3), 'ginocchio DX', C['yellow'], True)]),
    ('Spostamento baricentro Δs', 'mm', 2.0, [
        ('wbr', lambda a: a[:, 12] * 1e3, 'Δs (MPC)', C['orange'], False)]),
    ('Imbardata φ', '°', 2.0, [
        ('wbr', deg(5), 'φ', C['blue'], False),
        ('wbr', deg(6), 'φ riferimento', C['orange'], True)]),
    ('Angoli gambe', 'rad', 0.2, [
        ('joints', col(0), 'anca SX', C['blue'], False),
        ('joints', col(2), 'ginocchio SX', C['orange'], False),
        ('joints', col(1), 'anca DX', C['cyan'], True),
        ('joints', col(3), 'ginocchio DX', C['yellow'], True)]),
    ('Velocità ruote', 'rad/s', 1.0, [
        ('joints', col(10), 'ruota SX', C['blue'], False),
        ('joints', col(11), 'ruota DX', C['cyan'], False)]),
    ('ZMP ricostruito (frame robot)', 'mm', 10.0, [
        ('zmp', lambda a: a[:, 13] * 1e3, 'laterale (+ SX)', C['pink'], False),
        ('zmp', lambda a: a[:, 14] * 1e3, 'longitudinale', C['cyan'], False)]),
    ('Carico ruote da ZMP', 'N', 5.0, [
        ('zmp', col(15), 'ruota SX', C['blue'], False),
        ('zmp', col(16), 'ruota DX', C['cyan'], True)]),
    ('Errore di stima (Kalman − verità)', 'mm, mm/s', 5.0, [
        ('est', lambda a: np.linalg.norm(a[:, 1:4], axis=1) * 1e3, '|posizione| mm', C['red'], False),
        ('est', lambda a: np.linalg.norm(a[:, 4:7], axis=1) * 1e3, '|velocità| mm/s', C['pink'], True)]),
]
PLOT_COLUMNS = 3


def minmax_decimate(x, y, n_points):
    """Keeps min and max of each bucket so that spikes survive the decimation."""
    n = len(x)
    buckets = n_points // 2
    if n <= n_points or buckets < 2:
        return x, y
    size = n // buckets
    m = size * buckets
    yb = y[:m].reshape(buckets, size)
    xb = x[:m].reshape(buckets, size)
    rows = np.arange(buckets)
    i_min = np.nanargmin(np.where(np.isnan(yb), np.inf, yb), axis=1)
    i_max = np.nanargmax(np.where(np.isnan(yb), -np.inf, yb), axis=1)
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    xo = np.column_stack((xb[rows, first], xb[rows, second])).ravel()
    yo = np.column_stack((yb[rows, first], yb[rows, second])).ravel()
    return np.concatenate((xo, x[m:])), np.concatenate((yo, y[m:]))


class Led(QtWidgets.QWidget):
    COLORS = {'off': '#2b3038', 'ok': '#2ecc71', 'warn': '#f5d547', 'alarm': '#ff5c5c', 'info': '#4ea1ff'}

    def __init__(self, text, size=16):
        super().__init__()
        self._size = size
        self._color = QtGui.QColor(self.COLORS['off'])
        self._lit = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        self._dot = QtWidgets.QWidget()
        self._dot.setFixedSize(size + 6, size + 6)
        self._dot.paintEvent = self._paint_dot
        self.label = QtWidgets.QLabel(text)
        lay.addWidget(self._dot)
        lay.addWidget(self.label, 1)

    def set(self, color, lit=True, text=None):
        color = self.COLORS.get(color, color)
        if text is not None and text != self.label.text():
            self.label.setText(text)
        if QtGui.QColor(color) != self._color or lit != self._lit:
            self._color, self._lit = QtGui.QColor(color), lit
            self.label.setStyleSheet(f'color: {FG if lit else MUTED};')
            self._dot.update()

    def _paint_dot(self, _event):
        p = QtGui.QPainter(self._dot)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(3, 3, self._size, self._size)
        if self._lit:
            glow = QtGui.QRadialGradient(r.center(), self._size * 0.9)
            c = QtGui.QColor(self._color)
            c.setAlpha(110)
            glow.setColorAt(0.0, c)
            glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
            p.setBrush(glow)
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(r.adjusted(-3, -3, 3, 3))
        p.setBrush(self._color if self._lit else QtGui.QColor(self.COLORS['off']))
        p.setPen(QtGui.QPen(QtGui.QColor('#0b0d10'), 1))
        p.drawEllipse(r)


def group(title):
    box = QtWidgets.QGroupBox(title)
    box.setStyleSheet(
        f'QGroupBox {{ background: {PANEL}; border: 1px solid #2a2f38; border-radius: 6px; margin-top: 14px;'
        f' padding: 8px 8px 6px 8px; color: {FG}; font-weight: 600; }}'
        f'QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}')
    return box


class RobotView(pg.PlotWidget):
    """Sagittal sketch: torso, both legs, wheels, CoM and force arrows, following the robot."""

    def __init__(self):
        super().__init__(background=PANEL)
        self.setAspectLocked(True)
        self.setXRange(-0.26, 0.26, padding=0)
        self.setYRange(-0.05, 0.40, padding=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.model = None
        self.box = None
        pen = lambda c, w=2, s=QtCore.Qt.SolidLine: pg.mkPen(c, width=w, style=s)
        self.ground = self.plot([-1, 1], [0, 0], pen=pen('#4b5563', 2))
        self.ticks = self.plot([], [], pen=pen('#374151', 1), connect='pairs')
        self.leg_far = self.plot([], [], pen=pen('#7a3a33', 5))
        self.leg_near = self.plot([], [], pen=pen('#d0473a', 6))
        self.joints = self.plot([], [], pen=None, symbol='o', symbolSize=8, symbolBrush='#f0f0f0')
        self.torso = QtWidgets.QGraphicsPolygonItem()
        self.torso.setPen(pen('#c9ced6', 2))
        self.torso.setBrush(pg.mkBrush(138, 144, 153, 150))
        self.addItem(self.torso)
        self.wheel = self.plot([], [], pen=pen('#e5e7eb', 2))
        self.spokes = self.plot([], [], pen=pen('#3dd6d0', 2), connect='pairs')
        self.com = self.plot([], [], pen=None, symbol='+', symbolSize=16, symbolPen=pg.mkPen('#f5d547', width=3))
        self.f_contact = self.plot([], [], pen=pen(C['blue'], 3), connect='pairs')
        self.f_mpc = self.plot([], [], pen=pen(C['orange'], 3), connect='pairs')
        self.f_trac = self.plot([], [], pen=pen(C['green'], 3), connect='pairs')
        self.zmp = self.plot([], [], pen=None, symbol='t1', symbolSize=13, symbolBrush=C['pink'],
                             symbolPen=pg.mkPen('#0b0d10', width=1))
        self.text = pg.TextItem('In attesa di /robot_description ...', color=FG, anchor=(0, 0))
        self.text.setPos(-0.25, 0.395)
        self.addItem(self.text)
        legend = pg.TextItem(html=f'<span style="color:{C["blue"]}">▲ reazione suolo</span>  '
                                  f'<span style="color:{C["orange"]}">▲ F_z MPC</span>  '
                                  f'<span style="color:{C["green"]}">▶ trazione</span>  '
                                  f'<span style="color:#f5d547">+ CoM</span>  '
                                  f'<span style="color:{C["pink"]}">▲ ZMP</span>', anchor=(0.5, 1))
        legend.setPos(0, -0.045)
        self.addItem(legend)

    def set_model(self, xml):
        from rotino_description.model import WBRModel
        self.model = WBRModel(xml)
        for v in self.model.robot.link_map['base_link'].visuals:
            if hasattr(v.geometry, 'size'):
                ox = v.origin.xyz[0] if v.origin is not None else 0.0
                oz = v.origin.xyz[2] if v.origin is not None else 0.0
                sx, _, sz = v.geometry.size
                self.box = np.array([[ox - sx / 2, oz - sz / 2], [ox + sx / 2, oz - sz / 2],
                                     [ox + sx / 2, oz + sz / 2], [ox - sx / 2, oz + sz / 2],
                                     [ox - sx / 2, oz - sz / 2]])
        self.text.setText('')

    @staticmethod
    def _arrows(segments):
        xs, ys = [], []
        for (x0, y0), (x1, y1) in segments:
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-3:
                continue
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            h = min(0.02, 0.4 * L)
            for sgn in (1, -1):
                hx = x1 - h * (ux * 0.87 - sgn * uy * 0.5)
                hy = y1 - h * (uy * 0.87 + sgn * ux * 0.5)
                xs += [x1, hx]
                ys += [y1, hy]
            xs += [x0, x1]
            ys += [y0, y1]
        return xs, ys

    def update_robot(self, joints, odom, touching, leg_force, f_z, wheel_tau, phase_text, zmp_x=None):
        if self.model is None or joints is None:
            return
        m = self.model
        q = dict(zip(['left_hip', 'right_hip', 'left_knee', 'right_knee'], joints[1:5]))
        wheel_angle = joints[5:7]
        x0, z0, pitch = (odom[1], odom[3], odom[4]) if odom is not None else (0.0, m.p.r - m.leg_fk(0, 0)[0][1], 0.0)
        c, s = math.cos(pitch), math.sin(pitch)

        def world(p):
            return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1] + z0])

        def knee(hip):
            kx, kz = m.knee_xz
            ch, sh = math.cos(hip), math.sin(hip)
            return m.hip_xz + np.array([kx * ch + kz * sh, -kx * sh + kz * ch])

        legs = []
        for side, hip_n, knee_n in (('L', 'left_hip', 'left_knee'), ('R', 'right_hip', 'right_knee')):
            hip = world(m.hip_xz)
            kn = world(knee(q[hip_n]))
            wh = world(m.leg_fk(q[hip_n], q[knee_n])[0])
            legs.append((hip, kn, wh))
        self.leg_far.setData(*zip(*legs[1]))
        self.leg_near.setData(*zip(*legs[0]))
        self.joints.setData([p[0] for leg in legs for p in leg[:2]], [p[1] for leg in legs for p in leg[:2]])

        if self.box is not None:
            self.torso.setPolygon(QtGui.QPolygonF([QtCore.QPointF(*world(p)) for p in self.box]))

        r = m.p.r
        th = np.linspace(0, 2 * math.pi, 40)
        wx, wz = [], []
        sx, sz = [], []
        for i, (_, _, wh) in enumerate(legs):
            wx += list(wh[0] + r * np.cos(th)) + [np.nan]
            wz += list(wh[1] + r * np.sin(th)) + [np.nan]
            a = wheel_angle[i] + q['left_hip' if i == 0 else 'right_hip'] + \
                q['left_knee' if i == 0 else 'right_knee'] + pitch
            for k in range(3):
                ak = a + k * 2 * math.pi / 3
                sx += [wh[0], wh[0] + r * math.cos(ak)]
                sz += [wh[1], wh[1] - r * math.sin(ak)]
        self.wheel.setData(wx, wz, connect='finite')
        self.spokes.setData(sx, sz)

        com = world(m.upper_body_com_base(dict(q))[[0, 2]])
        self.com.setData([com[0]], [com[1]])

        contact_segments = []
        for i, (_, _, wh) in enumerate(legs):
            if touching[i] and leg_force is not None and not math.isnan(leg_force[1 + 2 * i]):
                contact_segments.append(((wh[0], 0.0), (wh[0], max(leg_force[2 + 2 * i], 0.0) * FORCE_SCALE)))
        self.f_contact.setData(*self._arrows(contact_segments))
        self.f_mpc.setData(*self._arrows([((com[0], com[1]), (com[0], com[1] + f_z * FORCE_SCALE))])
                           if f_z is not None else ([], []))
        axle_x = 0.5 * (legs[0][2][0] + legs[1][2][0])
        trac = sum(wheel_tau) / r if wheel_tau is not None else 0.0
        self.f_trac.setData(*self._arrows([((axle_x, 0.006), (axle_x + trac * FORCE_SCALE, 0.006))]))

        # ground ticks scroll with the robot position
        xs = []
        start = math.floor((x0 - 0.4) / 0.1) * 0.1
        for k in range(10):
            gx = start + k * 0.1 - x0
            xs += [gx, gx - 0.02]
        self.ticks.setData(xs, [0.0, -0.02] * 10)
        self.zmp.setData(*(([zmp_x], [0.0]) if zmp_x is not None and abs(zmp_x) < 0.3 else ([], [])))
        self.text.setText(phase_text)

class ZmpView(pg.PlotWidget):
    """Top view in the robot frame (forward up, left wheel on the left): wheel footprints, support
    segment, projected CoM, multibody ZMP with a fading trail, LIPM ZMP and load bars per wheel."""

    HALF_W = 0.27
    FWD = (-0.12, 0.13)

    def __init__(self):
        super().__init__(background=PANEL)
        self.setAspectLocked(True)
        self.setXRange(-self.HALF_W, self.HALF_W, padding=0)
        self.setYRange(*self.FWD, padding=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.band = QtWidgets.QGraphicsRectItem()
        self.band.setPen(pg.mkPen(None))
        self.band.setBrush(pg.mkBrush(46, 204, 113, 28))
        self.addItem(self.band)
        self.wheels = []
        for _ in range(2):
            w = QtWidgets.QGraphicsRectItem()
            w.setPen(pg.mkPen('#e5e7eb', width=2))
            w.setBrush(pg.mkBrush(60, 66, 76, 200))
            self.addItem(w)
            self.wheels.append(w)
        self.support = self.plot([], [], pen=pg.mkPen('#e5e7eb', width=3))
        self.loads = [self.plot([], [], pen=pg.mkPen(C['blue'], width=9)),
                      self.plot([], [], pen=pg.mkPen(C['cyan'], width=9))]
        self.load_text = [pg.TextItem('', color=FG, anchor=(0.5, 0)) for _ in range(2)]
        for t in self.load_text:
            self.addItem(t)
        self.lean = self.plot([], [], pen=pg.mkPen('#f5d547', width=1.5, style=QtCore.Qt.DashLine))
        self.trail = pg.ScatterPlotItem(pen=None, size=6)
        self.addItem(self.trail)
        self.com = self.plot([], [], pen=None, symbol='+', symbolSize=16, symbolPen=pg.mkPen('#f5d547', width=3))
        self.lipm = self.plot([], [], pen=None, symbol='o', symbolSize=13, symbolBrush=None,
                              symbolPen=pg.mkPen(C['cyan'], width=2))
        self.zmp = self.plot([], [], pen=None, symbol='o', symbolSize=14, symbolBrush=C['pink'],
                             symbolPen=pg.mkPen('#0b0d10', width=1))
        self.text = pg.TextItem('ZMP: in attesa di dati ...', color=FG, anchor=(0, 0))
        self.text.setPos(-self.HALF_W + 0.005, self.FWD[1] - 0.003)
        self.addItem(self.text)
        legend = pg.TextItem(html=f'<span style="color:{C["pink"]}">● ZMP</span>  '
                                  f'<span style="color:{C["cyan"]}">○ LIPM</span>  '
                                  f'<span style="color:#f5d547">+ CoM</span>  '
                                  f'<span style="color:#2ecc71">▮ appoggio</span>  '
                                  f'<span style="color:{C["blue"]}">▮ carico</span>', anchor=(0.5, 1))
        legend.setPos(0, self.FWD[0] + 0.002)
        self.addItem(legend)
        self.r = 0.06
        self.wheel_width = 0.056
        self.weight = None

    @staticmethod
    def to_robot(points, mid, lat, fwd):
        """World xy (..., 2) -> screen (x = -lateral, y = forward)."""
        d = np.asarray(points) - mid
        return np.stack([-(d @ lat), d @ fwd], -1)

    def reset_view(self):  # not clear(): PlotWidget binds PlotItem.clear on the instance
        for item in (self.support, self.lean, self.com, self.lipm, self.zmp, *self.loads):
            item.setData([], [])
        self.trail.setData([], [])
        self.text.setText('ZMP: in attesa di dati ...')

    def update_zmp(self, last, trail):
        if last is None:
            self.reset_view()
            return
        zmp, lipm, com = last[1:3], last[3:5], last[5:7]
        cl, cr = last[8:10], last[10:12]
        y_rel, lat_m, e_long, margin, fn_l, fn_r = last[12:18]
        seg = cl - cr
        d = float(np.linalg.norm(seg))
        if d < 1e-6:
            return
        lat = seg / d
        fwd = np.array([lat[1], -lat[0]])
        mid = 0.5 * (cl + cr)
        h = 0.5 * d
        clip = lambda p: np.array([np.clip(p[0], -self.HALF_W, self.HALF_W), np.clip(p[1], *self.FWD)])

        self.band.setRect(QtCore.QRectF(-h, -self.r, d, 2 * self.r))
        for w, x in zip(self.wheels, (-h, h)):
            w.setRect(QtCore.QRectF(x - self.wheel_width / 2, -self.r, self.wheel_width, 2 * self.r))
        self.support.setData([-h, h], [0.0, 0.0])

        # load bars behind each wheel, full length = half the weight
        fz = fn_l + fn_r
        if self.weight is None and fz > 0:
            self.weight = fz
        scale = self.r / self.weight if self.weight else 0.0
        for bar, text, x, f in ((self.loads[0], self.load_text[0], -h - 0.055, fn_l),
                                (self.loads[1], self.load_text[1], h + 0.055, fn_r)):
            bar.setData([x, x], [-self.r, -self.r + max(f, 0.0) * scale * 2.0])   # full wheel height = weight
            text.setText(f'{f:4.1f} N')
            text.setPos(x, -self.r - 0.004)

        z_s = self.to_robot(zmp, mid, lat, fwd)
        c_s = self.to_robot(com, mid, lat, fwd)
        l_s = self.to_robot(lipm, mid, lat, fwd)
        self.zmp.setData([clip(z_s)[0]], [clip(z_s)[1]])
        self.com.setData([clip(c_s)[0]], [clip(c_s)[1]])
        self.lipm.setData([clip(l_s)[0]], [clip(l_s)[1]])
        self.lean.setData([clip(c_s)[0], clip(z_s)[0]], [clip(c_s)[1], clip(z_s)[1]])

        if trail is not None and len(trail) > 1:
            step = max(1, len(trail) // ZMP_TRAIL_POINTS)
            tr = trail[::step]
            pts = np.stack([-tr[:, 13], tr[:, 14]], -1)        # each sample relative to its own support
            pts = np.stack([np.clip(pts[:, 0], -self.HALF_W, self.HALF_W), np.clip(pts[:, 1], *self.FWD)], -1)
            n = len(pts)
            brushes = [pg.mkBrush(255, 122, 182, int(20 + 180 * k / max(n - 1, 1))) for k in range(n)]
            self.trail.setData(pos=pts, brush=brushes)
        else:
            self.trail.setData([], [])

        frac = max(0.0, 1.0 - abs(y_rel))
        colour = (C['green'] if frac > ZMP_MARGIN_WARN else C['yellow'] if frac > ZMP_MARGIN_ALARM else C['red'])
        self.text.setHtml(
            f'<span style="color:{FG}">laterale {lat_m * 1e3:+6.1f} mm   longitudinale {e_long * 1e3:+5.1f} mm</span><br>'
            f'<span style="color:{colour}">margine {margin * 1e3:5.1f} mm ({100 * frac:3.0f} % di d/2)</span>')


class Dashboard(QtWidgets.QMainWindow):

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.window_s = 10.0
        self.paused = False
        self.last_counts = dict(bridge.counts)
        self.last_rate_time = time.monotonic()
        self.rates = {k: 0.0 for k in bridge.counts}
        self.frame_times = []
        self.render_ms = 0.0
        self.frame_ms = 0.0
        self.last_frame = None
        self.weight_line = None

        self.setWindowTitle('RoTino – Dashboard real-time')
        self.resize(1680, 980)
        self.setStyleSheet(f'QMainWindow, QWidget {{ background: {BG}; color: {FG}; font-size: 12px; }}'
                           f'QComboBox, QPushButton {{ background: #262b33; border: 1px solid #353b45;'
                           f' border-radius: 4px; padding: 3px 10px; }}'
                           f'QPushButton:checked {{ background: #b45309; }}'
                           f'QDoubleSpinBox {{ background: #262b33; border: 1px solid #353b45; border-radius: 4px; padding: 2px; }}'
                           f'QSlider::groove:horizontal {{ height: 6px; background: #2b3038; border-radius: 3px; }}'
                           f'QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; border-radius: 7px; background: #4ea1ff; }}'
                           f'QListWidget {{ background: {PANEL}; border: none; font-family: monospace; font-size: 11px; }}')

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)
        splitter.addWidget(self._left_panel())
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(4, 8, 8, 0)
        right_lay.addWidget(self._command_bar())
        right_lay.addWidget(self._plots_panel(), 1)
        splitter.addWidget(right)
        splitter.setSizes([470, 1210])

        self.status = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status, 1)
        self.statusBar().setStyleSheet(f'background: {PANEL}; color: {MUTED};')

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.refresh)
        self.set_fps(30)

    # ------------------------------------------------------------------ layout
    def _left_panel(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 4, 8)

        controls = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton('Pausa')
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(lambda on: setattr(self, 'paused', on))
        win = QtWidgets.QComboBox()
        win.addItems(['5 s', '10 s', '20 s', '30 s', '60 s'])
        win.setCurrentIndex(1)
        win.currentTextChanged.connect(lambda t: setattr(self, 'window_s', float(t.split()[0])))
        fps = QtWidgets.QComboBox()
        fps.addItems(['15 FPS', '30 FPS', '60 FPS'])
        fps.setCurrentIndex(1)
        fps.currentTextChanged.connect(lambda t: self.set_fps(int(t.split()[0])))
        reset = QtWidgets.QPushButton('Azzera')
        reset.clicked.connect(self.bridge.clear)
        shot = QtWidgets.QPushButton('Salva immagine')
        shot.clicked.connect(self.save_image)
        for w in (self.pause_btn, QtWidgets.QLabel('Finestra'), win, fps, reset, shot):
            controls.addWidget(w)
        lay.addLayout(controls)

        top = QtWidgets.QHBoxLayout()
        phase_box = group('Stato del robot')
        pl = QtWidgets.QVBoxLayout(phase_box)
        self.phase_leds = {}
        for key, label, color in PHASES:
            led = Led(label, 18)
            self.phase_leds[key] = (led, color)
            pl.addWidget(led)
        self.phase_time = QtWidgets.QLabel('—')
        self.phase_time.setStyleSheet(f'color: {MUTED}; font-family: monospace;')
        pl.addWidget(self.phase_time)
        top.addWidget(phase_box)

        diag_box = group('Diagnostica')
        dl = QtWidgets.QVBoxLayout(diag_box)
        self.diag = {}
        for key, label in [('sim', 'Simulazione attiva'), ('ctrl', 'Controllore attivo'),
                           ('contact_l', 'Ruota SX'), ('contact_r', 'Ruota DX'),
                           ('loaded', 'Robot a terra'), ('wheel_sat', 'Saturazione ruote'),
                           ('leg_sat', 'Saturazione gambe'), ('estimate', 'Stima di stato'),
                           ('fall', 'Caduta'), ('zmp', 'Margine ZMP'), ('latency', 'Ritardo < 100 ms'),
                           ('teleop', 'Comandi manuali')]:
            self.diag[key] = Led(label, 14)
            dl.addWidget(self.diag[key])
        top.addWidget(diag_box)
        lay.addLayout(top)

        robot_box = group('Vista laterale')
        rl = QtWidgets.QVBoxLayout(robot_box)
        self.robot = RobotView()
        self.robot.setMinimumHeight(230)
        rl.addWidget(self.robot)
        lay.addWidget(robot_box, 1)

        zmp_box = group('Vista dall\'alto – ZMP')
        zl = QtWidgets.QVBoxLayout(zmp_box)
        self.zmp_view = ZmpView()
        self.zmp_view.setMinimumHeight(210)
        zl.addWidget(self.zmp_view)
        lay.addWidget(zmp_box, 1)

        bottom = QtWidgets.QHBoxLayout()
        values_box = group('Valori istantanei')
        grid = QtWidgets.QGridLayout(values_box)
        grid.setVerticalSpacing(2)
        self.values = {}
        rows = [('t', 'Tempo simulato'), ('theta', 'θ / θ_ref'), ('s', 's / s_ref'), ('v', 'ṡ / ṡ_ref'),
                ('z', 'z / z_ref'), ('ds', 'Δs'), ('fz', 'F_z MPC'), ('fc', 'Spinta gambe SX / DX'),
                ('tau', 'τ ruote SX / DX'), ('yaw', 'Imbardata'), ('pos', 'Posizione x, y'),
                ('rate', 'Frequenza controllore'), ('delay', 'Ritardo dati → schermo')]
        for i, (key, label) in enumerate(rows):
            name = QtWidgets.QLabel(label)
            name.setStyleSheet(f'color: {MUTED};')
            val = QtWidgets.QLabel('—')
            val.setStyleSheet('font-family: monospace;')
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(name, i, 0)
            grid.addWidget(val, i, 1)
            self.values[key] = val
        bottom.addWidget(values_box, 3)

        log_box = group('Cambi di stato')
        ll = QtWidgets.QVBoxLayout(log_box)
        self.log = QtWidgets.QListWidget()
        ll.addWidget(self.log)
        bottom.addWidget(log_box, 3)
        lay.addLayout(bottom)
        return panel

    def _command_bar(self):
        box = group('Comandi   ·   W/S velocità   A/D rotazione   R/F altezza   Spazio stop   J salto')
        lay = QtWidgets.QHBoxLayout(box)
        lay.setSpacing(10)
        self.cmd_active = QtWidgets.QCheckBox('Attiva')
        self.cmd_active.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cmd_active.setStyleSheet('QCheckBox { font-weight: 600; } QCheckBox::indicator { width: 18px; height: 18px; }')
        self.cmd_active.toggled.connect(lambda _: self._send_teleop())
        lay.addWidget(self.cmd_active)

        def slider(title, lo, hi):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(0)
            head = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(title)
            name.setStyleSheet(f'color: {MUTED};')
            value = QtWidgets.QLabel()
            value.setStyleSheet('font-family: monospace;')
            value.setAlignment(QtCore.Qt.AlignRight)
            head.addWidget(name)
            head.addWidget(value, 1)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setMinimumWidth(150)
            sl.setFocusPolicy(QtCore.Qt.NoFocus)
            sl.valueChanged.connect(lambda _: self._send_teleop())
            col.addLayout(head)
            col.addWidget(sl)
            lay.addLayout(col, 1)
            return sl, value

        # slider units: cm/s, crad/s (slider to the right = turn right), mm
        self.sl_v, self.lbl_v = slider('Velocità', -int(CMD_V_MAX * 100), int(CMD_V_MAX * 100))
        self.sl_w, self.lbl_w = slider('Rotazione', -int(CMD_W_MAX * 100), int(CMD_W_MAX * 100))
        self.sl_h, self.lbl_h = slider('Altezza', int(CMD_H_MIN * 1000), int(CMD_H_MAX * 1000))

        def button(text, slot, tip=''):
            b = QtWidgets.QPushButton(text)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        button('Stop', self._cmd_stop, 'Velocità e rotazione a zero (Spazio)')
        button('Altezza 0', lambda: self.sl_h.setValue(0), 'Altezza nominale')
        button('Salta', self._cmd_jump, 'Il robot si ferma e poi salta (J)')
        self.push_spin = QtWidgets.QDoubleSpinBox()
        self.push_spin.setRange(0.5, CMD_PUSH_MAX)
        self.push_spin.setSingleStep(0.5)
        self.push_spin.setValue(2.0)
        self.push_spin.setSuffix(' N·s')
        self.push_spin.setFocusPolicy(QtCore.Qt.ClickFocus)
        button('◀ Spinta', lambda: self.bridge.send_push(-self.push_spin.value()), 'Impulso sul torso all\'indietro')
        lay.addWidget(self.push_spin)
        button('Spinta ▶', lambda: self.bridge.send_push(self.push_spin.value()), 'Impulso sul torso in avanti')

        for key, slot in (('W', lambda: self._nudge(self.sl_v, 5)), ('S', lambda: self._nudge(self.sl_v, -5)),
                          ('A', lambda: self._nudge(self.sl_w, -10)), ('D', lambda: self._nudge(self.sl_w, 10)),
                          ('R', lambda: self._nudge(self.sl_h, 5)), ('F', lambda: self._nudge(self.sl_h, -5)),
                          ('Space', self._cmd_stop), ('J', self._cmd_jump)):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(slot)
        self._update_command_labels()
        return box

    def _nudge(self, sl, step):
        self.cmd_active.setChecked(True)
        sl.setValue(sl.value() + step)

    def _cmd_stop(self):
        self.sl_v.setValue(0)
        self.sl_w.setValue(0)

    def _cmd_jump(self):
        self._cmd_stop()
        self.bridge.send_jump()
        self.statusBar().showMessage('Salto richiesto: il robot si ferma e poi salta', 3000)

    def _command_values(self):
        return self.sl_v.value() / 100.0, -self.sl_w.value() / 100.0, self.sl_h.value() / 1000.0

    def _update_command_labels(self):
        v, w, h = self._command_values()
        self.lbl_v.setText(f'{v:+.2f} m/s')
        turn = 'sinistra' if w > 0 else 'destra' if w < 0 else ''
        self.lbl_w.setText(f'{abs(w):.2f} rad/s {turn}')
        self.lbl_h.setText(f'{h * 1000:+.0f} mm')

    def _send_teleop(self):
        self._update_command_labels()
        self.bridge.send_teleop(self.cmd_active.isChecked(), *self._command_values())

    def _plots_panel(self):
        pg.setConfigOptions(antialias=False, background=BG, foreground=FG)
        self.glw = pg.GraphicsLayoutWidget()
        self.curves = []
        first = None
        for i, (title, unit, min_span, specs) in enumerate(PLOTS):
            if i and i % PLOT_COLUMNS == 0:
                self.glw.nextRow()
            p = self.glw.addPlot(title=f'<span style="font-size:10pt">{title}</span>')
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel('left', unit)
            p.getAxis('left').enableAutoSIPrefix(False)
            p.getAxis('bottom').enableAutoSIPrefix(False)
            p.getAxis('left').setWidth(62)
            p.setXRange(-self.window_s, 0, padding=0)
            p.setMouseEnabled(x=False, y=True)
            p.getViewBox().setLimits(minYRange=min_span)
            p.addLegend(offset=(6, 4), labelTextSize='8pt', brush=pg.mkBrush(21, 24, 29, 200))
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if title.startswith('Forze'):
                self.weight_plot = p
                p.getViewBox().setLimits(yMin=-150, yMax=400)   # near-singular legs give unbounded J^-T tau
            for stream, fn, label, color, dashed in specs:
                pen = pg.mkPen(color, width=1.6, style=QtCore.Qt.DashLine if dashed else QtCore.Qt.SolidLine)
                name = f'<span style="color:{color}">{label}</span>'
                self.curves.append((stream, fn, p.plot([], [], pen=pen, name=name)))
        first.setLabel('bottom', 't − adesso [s]')
        self.first_plot = first
        return self.glw

    # ------------------------------------------------------------------ helpers
    def set_fps(self, fps):
        self.timer.start(int(1000 / fps))

    def save_image(self):
        path = f'rotino_dashboard_{datetime.now():%Y%m%d_%H%M%S}.png'
        self.grab().save(path)
        self.statusBar().showMessage(f'Immagine salvata: {path}', 5000)

    @staticmethod
    def _fmt(a, b=None, unit='', prec=3):
        if a is None or (isinstance(a, float) and math.isnan(a)):
            return '—'
        if b is None or (isinstance(b, float) and math.isnan(b)):
            return f'{a:+.{prec}f} {unit}'
        return f'{a:+.{prec}f} / {b:+.{prec}f} {unit}'

    # ------------------------------------------------------------------ frame
    def refresh(self):
        t0 = time.perf_counter()
        b = self.bridge
        b.poll_events()
        wall = time.monotonic()
        now = b.now()
        wall_last = b.wall_last
        fresh = {k: wall - v < FRESH for k, v in wall_last.items()}

        if wall - self.last_rate_time >= 1.0:
            counts = b.counts
            dt = wall - self.last_rate_time
            self.rates = {k: (counts[k] - self.last_counts[k]) / dt for k in counts}
            self.last_counts, self.last_rate_time = counts, wall

        last = {k: (buf.last() if fresh[k] else None) for k, buf in b.buffers.items()}

        # ---- phase LEDs
        phase = b.phase if fresh['debug'] or fresh['wbr'] else None
        if phase is None and fresh['joints'] and not fresh['debug']:
            phase = 'HOLD'
        for key, (led, color) in self.phase_leds.items():
            led.set(color, lit=(key == phase))
        if phase and b.phase_since is not None and phase == b.phase:
            self.phase_time.setText(f'{phase} da {max(now - b.phase_since, 0.0):5.2f} s')
        else:
            self.phase_time.setText(phase or 'nessun dato')
        if self.log.count() != len(b.phase_log):
            self.log.clear()
            for t, name in reversed(b.phase_log):
                self.log.addItem(f'{t:6.2f}s {name}')

        # ---- diagnostics
        dbg, wbr, cnt, wc, lc, est, od, lf, zm = (last[k] for k in ('debug', 'wbr', 'contact', 'wheel_cmd',
                                                                    'leg_cmd', 'est', 'odom', 'leg_force', 'zmp'))
        touching = (cnt is not None and cnt[1] > 0.5, cnt is not None and cnt[2] > 0.5)
        d = self.diag
        d['sim'].set('ok', lit=fresh['joints'])
        d['ctrl'].set('ok', lit=fresh['debug'])
        for key, side, i in (('contact_l', 'SX', 0), ('contact_r', 'DX', 1)):
            force = f'  spinta {lf[2 + 2 * i]:5.1f} N' if lf is not None and not math.isnan(lf[2 + 2 * i]) else ''
            d[key].set('info', lit=touching[i],
                       text=f'Ruota {side} ' + ('a terra' if touching[i] else 'sollevata') + force)
        d['loaded'].set('ok', lit=dbg is not None and dbg[11] > 0.5)
        d['wheel_sat'].set('alarm', lit=wc is not None and np.abs(wc[1:3]).max() >= 0.95 * WHEEL_TORQUE_LIMIT)
        d['leg_sat'].set('alarm', lit=lc is not None and np.abs(lc[1:5]).max() >= 0.95 * LEG_TORQUE_LIMIT)
        if est is not None:
            e = np.linalg.norm(est[1:4]) * 1e3
            d['estimate'].set('ok' if e < 20 else 'warn' if e < 100 else 'alarm',
                              text=f'Stima di stato  err {e:5.1f} mm')
        else:
            d['estimate'].set('off', lit=False, text='Stima di stato')
        fallen = ((dbg is not None and abs(dbg[2]) > math.radians(30))
                  or (od is not None and abs(od[4]) > math.radians(45)))
        d['fall'].set('alarm', lit=bool(fallen))
        if zm is not None:
            frac = max(0.0, 1.0 - abs(zm[12]))
            d['zmp'].set('ok' if frac > ZMP_MARGIN_WARN else 'warn' if frac > ZMP_MARGIN_ALARM else 'alarm',
                         text=f'Margine ZMP  {100 * frac:3.0f} %  ({zm[16]:4.1f} / {zm[17]:4.1f} N)')
        else:
            d['zmp'].set('off', lit=False, text='Margine ZMP')
        age_ms = (wall - wall_last['joints']) * 1e3 if fresh['joints'] else float('nan')
        delay_ms = age_ms + max(b.transport_delay, 0.0) * 1e3 + self.frame_ms
        v_cmd, w_cmd, h_cmd = self._command_values()
        d['teleop'].set('warn', lit=self.cmd_active.isChecked(),
                        text=f'Comandi manuali  {v_cmd:+.2f} m/s  {w_cmd:+.2f} rad/s  {h_cmd * 1000:+.0f} mm'
                        if self.cmd_active.isChecked() else 'Comandi manuali')
        d['latency'].set('ok' if delay_ms < LATENCY_LIMIT_MS else 'alarm', lit=fresh['joints'],
                         text=f'Ritardo {delay_ms:5.1f} ms' if fresh['joints'] else 'Ritardo < 100 ms')

        # ---- values
        v = self.values
        v['t'].setText(f'{now:10.3f} s')
        v['theta'].setText(self._fmt(dbg[2] * DEG if dbg is not None else None,
                                     wbr[5] * DEG if wbr is not None else None, '°', 2))
        v['s'].setText(self._fmt(dbg[4] if dbg is not None else None, wbr[3] if wbr is not None else None, 'm'))
        v['v'].setText(self._fmt(dbg[5] if dbg is not None else None, wbr[9] if wbr is not None else None, 'm/s'))
        v['z'].setText(self._fmt(wbr[10] if wbr is not None else None, wbr[11] if wbr is not None else None, 'm'))
        v['ds'].setText(self._fmt(wbr[12] * 1e3 if wbr is not None else None, unit='mm', prec=1))
        v['fz'].setText(self._fmt(wbr[13] if wbr is not None else None, unit='N', prec=1))
        v['fc'].setText(self._fmt(lf[2] if lf is not None else None, lf[4] if lf is not None else None, 'N', 1))
        v['tau'].setText(self._fmt(wc[1] if wc is not None else None, wc[2] if wc is not None else None, 'Nm', 2))
        v['yaw'].setText(self._fmt(od[5] * DEG if od is not None else None, unit='°', prec=1))
        v['pos'].setText(f'{od[1]:+.3f}, {od[2]:+.3f} m' if od is not None else '—')
        v['rate'].setText(f'{self.rates.get("debug", 0):5.0f} Hz')
        v['delay'].setText(f'{delay_ms:5.1f} ms' if not math.isnan(delay_ms) else '—')

        # ---- robot sketch
        if self.robot.model is None and b.robot_description:
            try:
                self.robot.set_model(b.robot_description)
                self.weight_line = pg.InfiniteLine(pos=self.robot.model.p.m_b * G, angle=0,
                                                   pen=pg.mkPen('#6b7280', width=1, style=QtCore.Qt.DotLine),
                                                   label='peso m_b·g', labelOpts={'color': MUTED, 'position': 0.92})
                self.weight_plot.addItem(self.weight_line)
            except Exception as exc:  # malformed description: keep the rest of the dashboard running
                self.robot.text.setText(f'robot_description non valido: {exc}')
                b.robot_description = None
        if not self.paused:
            self.robot.update_robot(
                last['joints'], od, touching, lf, wbr[13] if wbr is not None else None,
                wc[1:3] if wc is not None else None,
                self.phase_leds[phase][0].label.text() if phase in self.phase_leds else '',
                zmp_x=((zm[1] - od[1]) * math.cos(od[5]) + (zm[2] - od[2]) * math.sin(od[5]))
                if zm is not None and od is not None else None)
            self.zmp_view.update_zmp(zm, b.buffers['zmp'].window(now - ZMP_TRAIL_S) if zm is not None else None)

            # ---- plots
            t_from = now - self.window_s
            windows = {}
            for stream, fn, curve in self.curves:
                if stream not in windows:
                    windows[stream] = b.buffers[stream].window(t_from)
                a = windows[stream]
                if a is None or len(a) == 0:
                    curve.setData([], [])
                    continue
                x, y = minmax_decimate(a[:, 0] - now, np.asarray(fn(a), dtype=float), MAX_POINTS)
                curve.setData(x, y, skipFiniteCheck=True)
            if abs(self.first_plot.viewRange()[0][0] + self.window_s) > 1e-6:
                self.first_plot.setXRange(-self.window_s, 0, padding=0)

        self.render_ms = (time.perf_counter() - t0) * 1e3
        if self.last_frame is not None:
            self.frame_ms = 0.9 * self.frame_ms + 0.1 * (wall - self.last_frame) * 1e3
        self.last_frame = wall
        self.frame_times.append(wall)
        self.frame_times = [f for f in self.frame_times if wall - f < 1.0]
        self.status.setText(
            f'FPS {len(self.frame_times):3d}   periodo frame {self.frame_ms:5.1f} ms   '
            f'aggiornamento {self.render_ms:5.1f} ms   '
            f'topic Hz: debug {self.rates.get("debug", 0):.0f} · wbr_state {self.rates.get("wbr", 0):.0f} · '
            f'joint_states {self.rates.get("joints", 0):.0f} · odom {self.rates.get("odom", 0):.0f} · '
            f'zmp {self.rates.get("zmp", 0):.0f}   '
            f'ritardo trasporto {max(b.transport_delay, 0) * 1e3:.1f} ms'
            + (f'   {b.error}' if b.error else '') + ('   [PAUSA]' if self.paused else ''))


def main():
    bridge = RosBridge()
    app = QtWidgets.QApplication([a for a in sys.argv if not a.startswith('--ros-args')][:1])
    app.setStyle('Fusion')
    win = Dashboard(bridge)
    win.show()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)   # lets Python handle Ctrl+C while Qt runs
    keepalive.start(200)
    code = app.exec_()
    bridge.close()
    sys.exit(code)


if __name__ == '__main__':
    main()
