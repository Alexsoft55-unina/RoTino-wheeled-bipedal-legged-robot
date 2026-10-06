"""ZMP-based cascaded balance for the RoTino PID controller (pure numpy, no ROS: unit-tested offline)."""

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


@dataclass
class SagittalGains:
    k_s: float = 60.0
    k_sd: float = 5.0
    authority: float = 0.4
    k_dcm: float = 1.5
    k_dcm_i: float = 0.5
    dcm_i_max: float = 0.08
    acc_max: float = 3.0
    torque_max: float = 10.0


class ZmpSagittalBalance:
    """Capture-point PI -> desired ZMP -> ZMP-offset PD on the wheel torque."""

    def __init__(self, a1, a2, b1, b2, k_sc, gains=None):
        """a1, a2, b1, b2: VL-WIP coefficients; k_sc: CoM offset over the axle per rad of pendulum angle."""
        self.gains = gains or SagittalGains()
        det = a1 * 2.0 * b2 - 2.0 * b1 * a2
        th_per_acc = 2.0 * b2 / det
        self.lean_per_acc = k_sc * th_per_acc
        self.torque_per_acc = -a2 * th_per_acc / (2.0 * b2)
        self.reset()

    @classmethod
    def from_model(cls, model, gains=None):
        """model: rotino_description.model.WBRModel, linearised in the nominal standing pose."""
        e = model.equivalent_centroid(0.0, 0.0)
        c = model.vlwip_coefficients(e['l'], I_y=e['I_y'])
        k_sc = model.p.m_b * e['l'] / model.kin.total_mass
        return cls(c['a1'], c['a2'], c['b1'], c['b2'], k_sc, gains)

    def reset(self):
        self.dcm_i = 0.0

    def feedforward(self, acc):
        """ZMP offset (CoM ahead of the contact) and wheel torque that sustain the acceleration acc."""
        return self.lean_per_acc * acc, self.torque_per_acc * acc

    def step(self, dt, h, pos_err, vel_err, s, s_dot, acc_ref=0.0, jerk_ref=0.0, outer_scale=1.0,
             integrate=True):
        """One control step: returns (per-wheel torque, info dict)."""
        g = self.gains
        omega = math.sqrt(G / max(h, 0.03))
        if abs(acc_ref) > g.acc_max:
            acc_ref, jerk_ref = clamp(acc_ref, -g.acc_max, g.acc_max), 0.0
        s_ff, tau_ff = self.feedforward(acc_ref)
        s_dot_ff = self.lean_per_acc * jerk_ref

        c_err = pos_err + (s - s_ff)
        cd_err = vel_err + (s_dot - s_dot_ff)
        xi_err = c_err + cd_err / omega
        if integrate:
            self.dcm_i = clamp(self.dcm_i + xi_err * dt, -g.dcm_i_max, g.dcm_i_max)

        s_fb = -(cd_err + g.k_dcm * xi_err + g.k_dcm_i * self.dcm_i) / omega
        s_max = g.acc_max / omega ** 2
        s_des = clamp(s_ff + g.authority * outer_scale * s_fb, -s_max, s_max)

        tau = tau_ff + g.k_s * (s - s_des) + g.k_sd * (s_dot - s_dot_ff)
        tau_sat = clamp(tau, -g.torque_max, g.torque_max)
        if tau_sat != tau and integrate:
            self.dcm_i -= xi_err * dt
        return tau_sat, {
            'omega': omega, 'xi_err': xi_err, 's_ff': s_ff, 's_des': s_des, 'tau_ff': tau_ff,
            'zmp_des': -s_des, 'zmp': -s,
            'saturated': tau_sat != tau,
        }


class LipmPreview:
    """CoM reference for a planned ZMP (= contact) path: the bounded solution of c_dd = omega^2 (c - p_ref)."""

    def __init__(self, horizon_tau=8.0, step=0.005):
        self.horizon_tau = horizon_tau
        self.step = step

    def smooth(self, signal, t, omega):
        """(omega / 2) int e^(-omega |u|) signal(t + u) du for a vectorised signal(times)."""
        half = self.horizon_tau / omega
        u = np.arange(-half, half + 0.5 * self.step, self.step)
        w = np.exp(-omega * np.abs(u))
        w /= w.sum()
        out = signal(t + u)
        return tuple(float(w @ o) for o in out) if isinstance(out, tuple) else float(w @ out)

    def __call__(self, reference, t, omega):
        """reference(times: ndarray) -> (p, p_dot) arrays of the contact path; returns (acc, jerk)."""
        c, cd = self.smooth(reference, t, omega)
        p0, pd0 = reference(np.array([t]))
        return omega ** 2 * (c - float(p0[0])), omega ** 2 * (cd - float(pd0[0]))


@dataclass
class LateralGains:
    k_zmp: float = 0.0
    lean_max: float = 0.15
    rate_max: float = 2.0
    filter_tau: float = 0.03


class ZmpLateralCompensation:
    """Leans the torso into the turn so that the ZMP stays at the centre of the wheel segment."""

    def __init__(self, track, gains=None):
        self.track = track
        self.gains = gains or LateralGains()
        self.reset()

    def reset(self):
        self.lean = 0.0
        self.corr = 0.0

    def step(self, dt, h, shift, y_zmp=None):
        """shift: wanted leftward CoM shift [m]; y_zmp: measured lateral ZMP. Returns (lean, leg difference, info)."""
        g = self.gains
        if y_zmp is not None:
            a = dt / (g.filter_tau + dt)
            self.corr += a * (-g.k_zmp * y_zmp - self.corr)
        shift += self.corr
        target = clamp(math.asin(clamp(shift / max(h, 0.03), -0.99, 0.99)), -g.lean_max, g.lean_max)
        self.lean += clamp(target - self.lean, -g.rate_max * dt, g.rate_max * dt)
        dz = self.track * math.tan(self.lean)
        return self.lean, dz, {'target': target, 'shift': shift, 'corr': self.corr}


def trapezoid_reference(t, v_max, a_max, distance):
    """Trapezoidal velocity profile over `distance`: (s, v, a) at time t >= 0 (same law as the MPC)."""
    v = min(v_max, math.sqrt(a_max * distance))
    ta = v / a_max
    tc = max(0.0, (distance - v * ta) / v)
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t < ta:
        return 0.5 * a_max * t * t, a_max * t, a_max
    if t < ta + tc:
        return 0.5 * v * ta + v * (t - ta), v, 0.0
    td = min(t - ta - tc, ta)
    s = 0.5 * v * ta + v * tc + v * td - 0.5 * a_max * td * td
    return s, v - a_max * td, (-a_max if td < ta else 0.0)


def trapezoid_path(times, v_max, a_max, distance):
    """Vectorised trapezoid_reference: (s, v) arrays (for LipmPreview)."""
    v = min(v_max, math.sqrt(a_max * distance))
    ta = v / a_max
    tc = max(0.0, (distance - v * ta) / v)
    t = np.clip(np.asarray(times, float), 0.0, 2.0 * ta + tc)
    td = np.clip(t - ta - tc, 0.0, ta)
    s = np.where(t < ta, 0.5 * a_max * t * t,
                 np.where(t < ta + tc, 0.5 * v * ta + v * (t - ta),
                          0.5 * v * ta + v * tc + v * td - 0.5 * a_max * td * td))
    vel = np.where(t < ta, a_max * t, np.where(t < ta + tc, v, v - a_max * td))
    return s, vel
