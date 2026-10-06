"""
ZMP-based cascaded balance for the RoTino PID controller (pure numpy, no ROS: unit-tested offline).

On a two-wheeled robot the support polygon degenerates into the segment between the wheel contacts,
so the ZMP is not free to move inside a foot: longitudinally it sits on the contact line, and the
wheels choose where that line is. That makes the ZMP the natural *control input* of the balance
(cart-table / LIPM view, Kajita 2003):

    c_dd = omega^2 (c - p),   omega^2 = g / h

with c the CoM, p the ZMP (= contact) and h the CoM height. Sagittally the controller is a cascade:

  outer  PI on the divergent component of motion (capture point, Englsberger 2011)
             xi = c + c_dot / omega,   xi_dot = omega (xi - p)
         -> desired ZMP offset behind the CoM, s_des = c - p_des (the "lean" of the CoM over the axle)
  inner  PD on the ZMP offset, wheel torque:  tau = tau_ff + K_s (s - s_des) + K_sd (s_dot - s_dot_ff)
  ff     s_ff, tau_ff: the lean and torque that keep the reference acceleration in steady state
         (from the linear VL-WIP model of rotino_description.model, eqs. 13-14 of Cui et al.)

A pure LIPM cascade assumes the inner loop is infinitely fast. Here it cannot be: the wheel/pendulum
mode is ~10 rad/s and the loop runs at 500 Hz with up to 2 samples of delay, and the full-authority
cascade feeds back +K_s/omega on the wheel speed (unstable). The outer loop therefore commands the ZMP
with authority rho < 1; the gains below were chosen by robust pole placement on the discretised
model (+-20 % mass, CoM height from squat to stretch, 1-2 samples of delay: minimum damping 0.69).

Laterally the ZMP is free to move along the wheel segment and must stay inside it. In a turn the
centripetal acceleration a_y pushes it outwards by h a_y / g; the controller leans the torso into
the turn with the legs (one shorter, one longer) so that the CoM moves inwards by the same amount,
with the same LIPM preview as the sagittal plane (the lean starts before the turn).
"""

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------- sagittal
@dataclass
class SagittalGains:
    k_s: float = 60.0          # Nm per m of ZMP-offset error, per wheel
    k_sd: float = 5.0          # Nm per m/s of ZMP-offset rate, per wheel
    authority: float = 0.4     # rho: share of the capture-point correction passed to the ZMP
    k_dcm: float = 1.5         # 1/s, capture-point error decay rate
    k_dcm_i: float = 0.5       # 1/s^2, integral on the capture-point error
    dcm_i_max: float = 0.08    # m s, anti-windup bound of the integral
    acc_max: float = 3.0       # m/s^2: bound of the commanded CoM acceleration (ZMP offset h a / g)
    torque_max: float = 10.0   # Nm per wheel


class ZmpSagittalBalance:
    """Capture-point PI -> desired ZMP -> ZMP-offset PD on the wheel torque."""

    def __init__(self, a1, a2, b1, b2, k_sc, gains=None):
        """a1, a2, b1, b2: VL-WIP coefficients (s_dd = a1 th + b1 (tau_l + tau_r), th_dd = a2 th + b2 (..));
        k_sc: whole-robot CoM offset over the axle per rad of upper-body pendulum angle."""
        self.gains = gains or SagittalGains()
        # steady acceleration a with th_dd = 0:  [a1 2b1; a2 2b2] [th; tau] = [a; 0]
        det = a1 * 2.0 * b2 - 2.0 * b1 * a2
        th_per_acc = 2.0 * b2 / det
        self.lean_per_acc = k_sc * th_per_acc              # m of ZMP offset per m/s^2
        self.torque_per_acc = -a2 * th_per_acc / (2.0 * b2)   # Nm per wheel per m/s^2
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
        """One control step.

        h: CoM height above the ground [m]; pos_err / vel_err: axle position / speed minus reference
        along the heading (the reference of the contact, i.e. of the ZMP); s / s_dot: horizontal CoM
        offset ahead of the axle and its rate (the longitudinal ZMP offset: the true ZMP sits on the
        contact line, s behind the CoM); acc_ref / jerk_ref: CoM reference acceleration and its
        derivative (LipmPreview). outer_scale scales the capture-point correction (jump phases),
        integrate freezes the integral. Returns (per-wheel torque, info dict).
        """
        g = self.gains
        omega = math.sqrt(G / max(h, 0.03))
        if abs(acc_ref) > g.acc_max:
            acc_ref, jerk_ref = clamp(acc_ref, -g.acc_max, g.acc_max), 0.0
        s_ff, tau_ff = self.feedforward(acc_ref)
        s_dot_ff = self.lean_per_acc * jerk_ref

        c_err = pos_err + (s - s_ff)                       # CoM error = axle error + lean error
        cd_err = vel_err + (s_dot - s_dot_ff)
        xi_err = c_err + cd_err / omega                    # capture-point error
        if integrate:
            self.dcm_i = clamp(self.dcm_i + xi_err * dt, -g.dcm_i_max, g.dcm_i_max)

        # desired ZMP relative to the CoM from xi_dot = omega (xi - p): p_des - c = cd/omega + k xi/omega + ...
        s_fb = -(cd_err + g.k_dcm * xi_err + g.k_dcm_i * self.dcm_i) / omega
        s_max = g.acc_max / omega ** 2
        s_des = clamp(s_ff + g.authority * outer_scale * s_fb, -s_max, s_max)

        tau = tau_ff + g.k_s * (s - s_des) + g.k_sd * (s_dot - s_dot_ff)
        tau_sat = clamp(tau, -g.torque_max, g.torque_max)
        if tau_sat != tau and integrate:                   # anti-windup: undo the step that saturated
            self.dcm_i -= xi_err * dt
        return tau_sat, {
            'omega': omega, 'xi_err': xi_err, 's_ff': s_ff, 's_des': s_des, 'tau_ff': tau_ff,
            'zmp_des': -s_des, 'zmp': -s,                  # ZMP ahead of the CoM [m]
            'saturated': tau_sat != tau,
        }


class LipmPreview:
    """CoM reference for a planned ZMP (= contact) path p_ref(t): the bounded solution of
    c_dd = omega^2 (c - p_ref), i.e. p_ref smoothed by the two-sided kernel (omega / 2) e^(-omega |u|).

    The CoM starts leaning *before* the wheels accelerate, which is what an inverted pendulum needs
    (non-minimum phase: to go forward the contact must first slip back under a forward-leaning CoM).
    Returns (c_ref - p_ref, its rate) -> acc_ref = omega^2 (c_ref - p_ref), jerk_ref likewise.
    """

    def __init__(self, horizon_tau=8.0, step=0.005):
        self.horizon_tau = horizon_tau     # kernel truncated at horizon_tau / omega on each side
        self.step = step

    def smooth(self, signal, t, omega):
        """(omega / 2) int e^(-omega |u|) signal(t + u) du for a vectorised signal(times) -> ndarray (or a
        tuple of them, each smoothed)."""
        half = self.horizon_tau / omega
        u = np.arange(-half, half + 0.5 * self.step, self.step)
        w = np.exp(-omega * np.abs(u))
        w /= w.sum()                       # normalised: a constant signal is returned exactly
        out = signal(t + u)
        return tuple(float(w @ o) for o in out) if isinstance(out, tuple) else float(w @ out)

    def __call__(self, reference, t, omega):
        """reference(times: ndarray) -> (p, p_dot) arrays of the contact path; returns (acc, jerk)."""
        c, cd = self.smooth(reference, t, omega)
        p0, pd0 = reference(np.array([t]))
        return omega ** 2 * (c - float(p0[0])), omega ** 2 * (cd - float(pd0[0]))


# ---------------------------------------------------------------------------- lateral
@dataclass
class LateralGains:
    # Proportional correction on the lateral ZMP measured by the multibody estimator. Off: rolling the
    # torso first moves the ZMP the wrong way (the CoM must accelerate sideways), and with the 25 ms lag
    # of the estimator a gain of 0.3 already sustained a +-175 mm limit cycle in Gazebo.
    k_zmp: float = 0.0          # m of CoM shift per m of measured lateral ZMP error
    lean_max: float = 0.15      # rad of torso roll into the turn
    rate_max: float = 2.0       # rad/s: safety slew limit (the previewed command stays well below)
    filter_tau: float = 0.03    # s, first-order filter of the measured-ZMP correction


class ZmpLateralCompensation:
    """Leans the torso into the turn so that the ZMP stays at the centre of the wheel segment.

    LIPM: y_zmp = y_c - (h / g) y_c_dd - h a_y / g. With one leg shorter by Delta the torso rolls by
    phi = atan(Delta / d) and, the cambered wheels rolling their contacts too, the CoM moves sideways
    by h sin(phi) with respect to the contact midpoint (kinematics of the URDF: 0.656 mm per mm of
    Delta, i.e. an arm of 0.193 m = h). The wanted CoM shift y_c comes from LipmPreview.smooth of
    h a_y / g along the planned turn, the bounded solution of y_zmp = 0 including the y_c_dd term:
    the torso starts leaning before the turn and without the jerks that a stepwise lean feeds to
    the ZMP. Static limit: sin(phi) = a_y / g, the lean of a bicycle.
    """

    def __init__(self, track, gains=None):
        self.track = track
        self.gains = gains or LateralGains()
        self.reset()

    def reset(self):
        self.lean = 0.0
        self.corr = 0.0

    def step(self, dt, h, shift, y_zmp=None):
        """shift: wanted CoM shift to the left of the contact midpoint [m]; y_zmp: measured lateral ZMP
        (+ left) or None. Returns (lean [rad, + = left side down], leg length difference right - left
        [m], info)."""
        g = self.gains
        if y_zmp is not None:
            a = dt / (g.filter_tau + dt)
            self.corr += a * (-g.k_zmp * y_zmp - self.corr)
        shift += self.corr
        target = clamp(math.asin(clamp(shift / max(h, 0.03), -0.99, 0.99)), -g.lean_max, g.lean_max)
        self.lean += clamp(target - self.lean, -g.rate_max * dt, g.rate_max * dt)
        dz = self.track * math.tan(self.lean)
        return self.lean, dz, {'target': target, 'shift': shift, 'corr': self.corr}


# ---------------------------------------------------------------------------- helpers
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
