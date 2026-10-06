"""
Zero Moment Point of RoTino, reconstructed from the multibody dynamics.

Gazebo Fortress publishes wheel contacts without wrenches, so the centre of pressure cannot be
measured: the ZMP is rebuilt from the motion of every link (flat ground, z = 0):

    x_zmp = [sum m_i ((zdd_i + g) x_i - xdd_i z_i) - sum dL_y,i] / sum m_i (zdd_i + g)
    y_zmp = [sum m_i ((zdd_i + g) y_i - ydd_i z_i) + sum dL_x,i] / sum m_i (zdd_i + g)

with L_i the angular momentum of link i about its own CoM. Accelerations and dL come from a
Savitzky-Golay fit (numpy only: the system scipy does not load with numpy 2.x).

With two point contacts the support polygon degenerates into the segment between the wheels, so:
  - laterally the ZMP must stay inside [-d/2, d/2] (tip-over / wheel lift-off margin);
  - longitudinally the true ZMP lies on the contact line: the residual e_long checks the
    reconstruction, and the LIPM point x_com - z_com xdd_com / (zdd_com + g) shows what a lumped
    model misses.

Used online by rotino_dashboard (ZmpEstimator, causal) and offline by rotino_benchmark (zmp_series).
"""

import math
from collections import deque

import numpy as np

from rotino_description.kinematics import RobotKinematics, quat_to_matrix, rpy_to_matrix

G = 9.81
WHEEL_JOINTS = ('left_wheel_joint', 'right_wheel_joint')
JOINTS = ('left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_wheel_joint', 'right_wheel_joint')


# ---------------------------------------------------------------------------- filtering
def savgol_weights(n_points, order, deriv, dt, eval_at):
    """Weights w such that w @ y[k] is the deriv-th derivative, at sample index eval_at, of the
    least-squares polynomial of the given order through n_points uniformly spaced samples."""
    k = np.arange(n_points, dtype=float) - eval_at
    V = np.vander(k * dt, order + 1, increasing=True)          # columns 1, s, s^2, ...
    return math.factorial(deriv) * np.linalg.pinv(V)[deriv]


def savgol_filter(y, n_points, order, deriv, dt):
    """Centred Savitzky-Golay along axis 0; the ends use the one-sided fit of the first/last window."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < n_points:
        raise ValueError(f'need at least {n_points} samples, got {n}')
    half = n_points // 2
    out = np.empty_like(y)
    w = savgol_weights(n_points, order, deriv, dt, half)
    windows = np.lib.stride_tricks.sliding_window_view(y, n_points, axis=0)   # (n-n_points+1, ..., n_points)
    out[half:n - half] = windows @ w
    for i in range(half):
        out[i] = np.tensordot(savgol_weights(n_points, order, deriv, dt, i), y[:n_points], axes=(0, 0))
        out[n - 1 - i] = np.tensordot(savgol_weights(n_points, order, deriv, dt, n_points - 1 - i),
                                      y[-n_points:], axes=(0, 0))
    return out


# ---------------------------------------------------------------------------- ZMP formulas
def multibody_zmp(m, p, a, dL, g=G):
    """m (n,), p/a/dL (..., n, 3) -> zmp (..., 2), vertical ground reaction Fz (...)."""
    fz_i = m * (a[..., 2] + g)
    fz = fz_i.sum(-1)
    x = ((fz_i * p[..., 0]).sum(-1) - (m * a[..., 0] * p[..., 2]).sum(-1) - dL[..., 1].sum(-1)) / fz
    y = ((fz_i * p[..., 1]).sum(-1) - (m * a[..., 1] * p[..., 2]).sum(-1) + dL[..., 0].sum(-1)) / fz
    return np.stack([x, y], -1), fz


def lipm_zmp(com, com_acc, g=G):
    """Lumped-mass ZMP: the whole robot as a point at its CoM (no angular momentum)."""
    s = com[..., 2] / (com_acc[..., 2] + g)
    return np.stack([com[..., 0] - s * com_acc[..., 0], com[..., 1] - s * com_acc[..., 1]], -1)


def support_metrics(zmp, c_left, c_right):
    """ZMP against the wheel contact segment (ground plane).

    Returns lam (share of the load on the left wheel, 0..1 inside the support), y_rel = 2 lam - 1
    (+1 = on the left wheel), e_long (signed distance ahead of the contact line [m]), track d [m].
    """
    seg = c_left[..., :2] - c_right[..., :2]
    d = np.linalg.norm(seg, axis=-1)
    lat = seg / d[..., None]                                    # points to the left wheel
    fwd = np.stack([lat[..., 1], -lat[..., 0]], -1)            # lateral axis rotated by -90 deg
    rel = zmp - c_right[..., :2]
    lam = (rel * lat).sum(-1) / d
    e_long = ((zmp - 0.5 * (c_left[..., :2] + c_right[..., :2])) * fwd).sum(-1)
    return lam, 2.0 * lam - 1.0, e_long, d


def wheel_loads(lam, fz):
    """Static split of Fz between the wheels from the lateral ZMP (clipped: a wheel cannot pull)."""
    lam = np.clip(lam, 0.0, 1.0)
    return lam * fz, (1.0 - lam) * fz


# ---------------------------------------------------------------------------- model
class ZmpModel:
    """Link masses/inertias from the URDF and the world state of every link for a robot pose."""

    def __init__(self, urdf_xml):
        from urdf_parser_py.urdf import URDF
        self.kin = RobotKinematics(urdf_xml)
        robot = URDF.from_xml_string(urdf_xml)
        links = {link.name: link for link in robot.links}
        self.names, masses, coms, inertias = [], [], [], []
        for name, mass, com_local in self.kin.masses:
            inertial = links[name].inertial
            i = inertial.inertia
            I = np.array([[i.ixx, i.ixy, i.ixz], [i.ixy, i.iyy, i.iyz], [i.ixz, i.iyz, i.izz]])
            rpy = inertial.origin.rpy if inertial.origin is not None and inertial.origin.rpy else [0, 0, 0]
            R = rpy_to_matrix(*rpy)
            self.names.append(name)
            masses.append(mass)
            coms.append(com_local)
            inertias.append(R @ I @ R.T)                        # inertia in the link frame
        self.m = np.array(masses)
        self.com_local = np.array(coms)
        self.I_local = np.array(inertias)
        joints = {j.name: j for j in robot.joints}
        self.wheel_links = [joints[j].child for j in WHEEL_JOINTS]
        self.wheel_axes = [np.array(joints[j].axis, dtype=float) for j in WHEEL_JOINTS]
        self.wheel_radius = next(float(c.geometry.radius) for c in links[self.wheel_links[0]].collisions
                                 if hasattr(c.geometry, 'radius'))
        mu = float('nan')
        import xml.etree.ElementTree as ET
        for gz in ET.fromstring(urdf_xml).findall('gazebo'):
            if gz.get('reference') == self.wheel_links[0] and gz.find('mu1') is not None:
                mu = float(gz.find('mu1').text)
        self.mu = mu

    def state(self, base_pos, base_rot, joints):
        """-> link CoMs (n,3), link rotations (n,3,3), wheel contact points (2,3), all in world."""
        frames = self.kin.link_frames(base_pos, base_rot, joints)
        R = np.array([frames[n][0] for n in self.names])
        p = np.array([frames[n][1] for n in self.names]) + np.einsum('nij,nj->ni', R, self.com_local)
        contacts = []
        for link, axis in zip(self.wheel_links, self.wheel_axes):
            Rw, pw = frames[link]
            a = Rw @ axis
            down = np.array([0.0, 0.0, 1.0]) - a[2] * a        # z projected on the wheel plane
            contacts.append(pw - self.wheel_radius * down / np.linalg.norm(down))
        return p, R, np.array(contacts)

    def angular_momentum(self, R_prev, R, dt):
        """L_i = I_i,world omega_i with omega_i from the rotation increment over dt (small angle)."""
        dR = R @ np.transpose(R_prev, (0, 2, 1))
        W = 0.5 * (dR - np.transpose(dR, (0, 2, 1))) / dt
        omega = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], -1)
        I_world = R @ self.I_local @ np.transpose(R, (0, 2, 1))
        return np.einsum('nij,nj->ni', I_world, omega)


def _result(model, p, a, L_dot, contacts):
    zmp, fz = multibody_zmp(model.m, p, a, L_dot)
    M = model.m.sum()
    com = (model.m[:, None] * p).sum(-2) / M
    com_acc = (model.m[:, None] * a).sum(-2) / M
    lip = lipm_zmp(com, com_acc)
    lam, y_rel, e_long, d = support_metrics(zmp, contacts[..., 0, :], contacts[..., 1, :])
    f_left, f_right = wheel_loads(lam, fz)
    f_h = M * np.linalg.norm(com_acc[..., :2], axis=-1)
    return {
        'zmp': zmp, 'lipm': lip, 'com': com, 'com_acc': com_acc, 'fz': fz,
        'contact_l': contacts[..., 0, :], 'contact_r': contacts[..., 1, :],
        'lam': lam, 'y_rel': y_rel, 'e_long': e_long, 'track': d,
        'margin': (1.0 - np.abs(y_rel)) * 0.5 * d,
        'fn_left': f_left, 'fn_right': f_right,
        'friction_use': f_h / (model.mu * fz) if not math.isnan(model.mu) else np.full_like(fz, np.nan),
    }


# ---------------------------------------------------------------------------- online
class ZmpEstimator:
    """ZMP for live display: quadratic fit over the last `window` seconds, evaluated at the centre of
    the window, so the result lags by window / 2. Evaluating the second derivative at the newest
    sample instead amplifies the noise by an order of magnitude (seen live: +-200 mm spikes)."""

    def __init__(self, urdf_xml, window=0.05, rate=500.0, order=2):
        self.model = ZmpModel(urdf_xml)
        self.n = max(int(round(window * rate)) | 1, order + 3)
        self.order = order
        self.hist = deque(maxlen=self.n)
        self.reset()

    def reset(self):
        self.hist.clear()
        self.prev = None
        self.pending = deque(maxlen=50)
        self.jhist = deque(maxlen=50)

    # Live streams: odom and joint_states arrive separately and ~2.5 % of the pairs are one sample
    # apart; pairing "latest with latest" turns that into tens of mm of ZMP noise after two
    # derivatives. Each base pose waits for the joint sample at (or after) its stamp instead.
    def push_base(self, t, base_pos, base_quat):
        self.pending.append((t, base_pos, base_quat))
        return self._drain()

    def push_joints(self, t, joints):
        if self.jhist and t <= self.jhist[-1][0]:
            if t < self.jhist[-1][0] - 0.5:
                self.reset()                                    # simulation restarted
            else:
                return None
        self.jhist.append((t, dict(joints)))
        return self._drain()

    def _drain(self):
        out = None
        while self.pending and self.jhist and self.jhist[-1][0] >= self.pending[0][0] - 1e-6:
            t, pos, quat = self.pending.popleft()
            ts = np.array([j[0] for j in self.jhist])
            i = int(np.searchsorted(ts, t - 1e-6))
            if i < len(ts) and abs(ts[i] - t) < 1e-6 or i == 0:
                joints = self.jhist[min(i, len(ts) - 1)][1]
            else:                                               # missing sample: linear in between
                (t0, j0), (t1, j1) = self.jhist[i - 1], self.jhist[i]
                w = (t - t0) / (t1 - t0)
                joints = {n: (1 - w) * j0[n] + w * j1[n] for n in j0}
            out = self.update(t, pos, quat, joints) or out
        return out

    def update(self, t, base_pos, base_quat, joints):
        """base_quat = (x, y, z, w); joints: {name: angle}. Returns the result for the sample at the
        centre of the window (its time in result['t']), or None while filling."""
        p, R, contacts = self.model.state(np.asarray(base_pos, float), quat_to_matrix(*base_quat), joints)
        if self.prev is not None and t <= self.prev[0]:
            if t < self.prev[0] - 0.5:
                self.reset()                                    # simulation restarted
            else:
                return None
        L = np.zeros_like(p) if self.prev is None else self.model.angular_momentum(self.prev[1], R, t - self.prev[0])
        self.prev = (t, R)
        self.hist.append((t, p, L, contacts))
        if len(self.hist) < self.n:
            return None
        centre = self.hist[self.n // 2]
        # fit on the actual stamps: live topics drop samples (~484 of 500 Hz) and uniform
        # Savitzky-Golay weights would put the points after a gap at the wrong time
        s_ = np.array([h[0] for h in self.hist]) - centre[0]
        pinv = np.linalg.pinv(np.vander(s_, self.order + 1, increasing=True))
        P = np.array([h[1] for h in self.hist])
        Ls = np.array([h[2] for h in self.hist])
        p_f = np.tensordot(pinv[0], P, axes=(0, 0))
        dL = np.tensordot(pinv[1], Ls, axes=(0, 0))
        a = np.tensordot(2.0 * pinv[2], P, axes=(0, 0))
        out = _result(self.model, p_f, a, dL, centre[3])
        out['t'] = centre[0]
        return out


# ---------------------------------------------------------------------------- offline
def _unique_increasing(t):
    """Mask keeping the first sample of every new, increasing stamp (logs repeat the last message)."""
    t = np.asarray(t, float)
    keep = np.isfinite(t)
    last = -np.inf
    for i in np.flatnonzero(keep):
        if t[i] <= last:
            keep[i] = False
        else:
            last = t[i]
    return keep


def zmp_series(model, t, base_pos, base_quat, joints, window=0.04, order=2, joint_t=None):
    """Non-causal ZMP over a whole log.

    t (N,) stamps of the base samples, base_pos (N,3), base_quat (N,4) as x,y,z,w, joints {name: (N,)}
    stamped by joint_t (default t). Each signal keeps only its new stamps and is interpolated on a
    uniform grid at the median base period; the result dict has arrays on that grid plus 't'.
    """
    t = np.asarray(t, float)
    keep = _unique_increasing(t)
    tb = t[keep]
    dt = float(np.median(np.diff(tb)))
    tj = t if joint_t is None else np.asarray(joint_t, float)
    keep_j = _unique_increasing(tj)
    t_start, t_end = max(tb[0], tj[keep_j][0]), min(tb[-1], tj[keep_j][-1])
    tu = np.arange(t_start, t_end + 0.5 * dt, dt)

    def interp(y, stamps=tb, mask=keep):
        return np.interp(tu, stamps, np.asarray(y, float)[mask])

    pos = np.stack([interp(base_pos[:, k]) for k in range(3)], -1)
    q = np.asarray(base_quat, float)[keep]
    flips = np.concatenate([[1.0], np.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0)])
    q = q * np.cumprod(flips)[:, None]                          # continuous sign before interpolating
    q = np.stack([np.interp(tu, tb, q[:, k]) for k in range(4)], -1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    jq = {name: interp(v, tj[keep_j], keep_j) for name, v in joints.items()}

    N = len(tu)
    n_links = len(model.m)
    P = np.empty((N, n_links, 3))
    Rs = np.empty((N, n_links, 3, 3))
    C = np.empty((N, 2, 3))
    for k in range(N):
        P[k], Rs[k], C[k] = model.state(pos[k], quat_to_matrix(*q[k]), {n: v[k] for n, v in jq.items()})
    L = np.zeros_like(P)
    for k in range(1, N):
        L[k] = model.angular_momentum(Rs[k - 1], Rs[k], dt)
    L[0] = L[1]

    n_pts = max(int(round(window / dt)) | 1, order + 3)
    p_f = savgol_filter(P, n_pts, order, 0, dt)
    a = savgol_filter(P, n_pts, order, 2, dt)
    dL = savgol_filter(L, n_pts, order, 1, dt)
    out = _result(model, p_f, a, dL, C)
    out['t'] = tu
    out['yaw'] = np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    return out
