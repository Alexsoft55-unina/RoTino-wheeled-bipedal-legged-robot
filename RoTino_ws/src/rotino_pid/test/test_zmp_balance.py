"""Offline checks of the ZMP balance law of the PID (rotino_pid.zmp_balance) on the linear VL-WIP model."""

import math
import shutil
import subprocess

import numpy as np
import pytest

from rotino_pid.zmp_balance import (G, LateralGains, LipmPreview, ZmpLateralCompensation, ZmpSagittalBalance,
                                    trapezoid_path, trapezoid_reference)

DT = 0.002


@pytest.fixture(scope='module')
def wbr():
    if shutil.which('xacro') is None:
        pytest.skip('xacro not available')
    try:
        share = subprocess.check_output(['ros2', 'pkg', 'prefix', 'rotino_description']).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('rotino_description not built/sourced')
    from rotino_description.model import WBRModel
    xml = subprocess.check_output(['xacro', f'{share}/share/rotino_description/urdf/rotino.urdf.xacro']).decode()
    return WBRModel(xml)


def _expm(X):
    E, term = np.eye(len(X)), np.eye(len(X))
    for n in range(1, 40):
        term = term @ X / n
        E = E + term
    return E


class LinearPlant:
    """Sagittal VL-WIP x = [s, theta, s_dot, theta_dot], same torque on both wheels, ZOH at DT."""

    def __init__(self, wbr, hip=0.0, mass_scale=1.0):
        p = wbr.p
        e = wbr.equivalent_centroid(hip, -2.0 * hip)
        m_b = p.m_b
        p.m_b = m_b * mass_scale
        try:
            A6, B6 = wbr.vlwip_matrices(e['l'], I_y=e['I_y'] * mass_scale)
        finally:
            p.m_b = m_b
        idx = [0, 1, 3, 4]
        A = A6[np.ix_(idx, idx)]
        B = B6[idx] @ np.array([1.0, 1.0])
        X = np.zeros((5, 5))
        X[:4, :4] = A * DT
        X[:4, 4] = B * DT
        E = _expm(X)
        self.Ad, self.Bd = E[:4, :4], E[:4, 4]
        M = wbr.kin.total_mass - m_b + m_b * mass_scale
        self.k_sc = m_b * mass_scale * e['l'] / M
        self.h = (m_b * mass_scale * (e['l'] + p.r) + 2 * p.m_w * p.r) / M
        self.x = np.zeros(4)

    def step(self, tau):
        self.x = self.Ad @ self.x + self.Bd * tau


def run(wbr, plant, ctrl, T, path=None, delay=1, impulse_at=None, dv=0.0):
    """Closed loop with `delay` samples of actuation delay; path(times) -> (s, v) is the planned contact
    path, previewed as in the controller. Returns time, states, torques, desired ZMP offsets."""
    preview = LipmPreview()
    queue = [0.0] * delay
    log = []
    for k in range(int(T / DT)):
        t = k * DT
        if impulse_at is not None and abs(t - impulse_at) < DT / 2:
            plant.x[2] += dv
        s_ref = v_ref = acc = jerk = 0.0
        if path is not None:
            (s_ref,), (v_ref,) = path(np.array([t]))
            acc, jerk = preview(path, t, math.sqrt(G / plant.h))
        s, th, sd, thd = plant.x
        tau, info = ctrl.step(DT, plant.h, s - s_ref, sd - v_ref, plant.k_sc * th, plant.k_sc * thd, acc, jerk)
        queue.append(tau)
        plant.step(queue.pop(0))
        log.append((t, *plant.x, tau, info['s_des']))
    return np.array(log)


def test_feedforward_holds_the_reference_acceleration(wbr):
    ctrl = ZmpSagittalBalance.from_model(wbr)
    e = wbr.equivalent_centroid(0.0, 0.0)
    c = wbr.vlwip_coefficients(e['l'], I_y=e['I_y'])
    k_sc = wbr.p.m_b * e['l'] / wbr.kin.total_mass
    s_ff, tau_ff = ctrl.feedforward(1.0)
    th = s_ff / k_sc
    assert math.isclose(c['a2'] * th + c['b2'] * 2 * tau_ff, 0.0, abs_tol=1e-9)   # no pitch acceleration
    assert math.isclose(c['a1'] * th + c['b1'] * 2 * tau_ff, 1.0, rel_tol=1e-9)   # wheels accelerate at 1 m/s^2
    # the ZMP falls behind the CoM by about h a / g, the LIPM value
    h = (wbr.p.m_b * (e['l'] + wbr.p.r) + 2 * wbr.p.m_w * wbr.p.r) / wbr.kin.total_mass
    assert 0.8 < s_ff / (h / G) < 1.25


@pytest.mark.parametrize('hip, mass_scale, delay', [
    (0.0, 1.0, 1), (0.0, 0.8, 2), (0.0, 1.2, 2), (-0.3, 1.0, 2), (0.3, 1.0, 2)])
def test_recovers_from_a_lean_robustly(wbr, hip, mass_scale, delay):
    plant = LinearPlant(wbr, hip, mass_scale)
    plant.x[1] = 0.03 / plant.k_sc                  # CoM 3 cm ahead of the axle
    log = run(wbr, plant, ZmpSagittalBalance.from_model(wbr), 12.0, delay=delay)
    tail = log[log[:, 0] > 9.0]
    assert np.all(np.abs(tail[:, 2] * plant.k_sc) < 1e-3)      # lean back under 1 mm
    assert np.all(np.abs(tail[:, 1]) < 0.02)                   # back to the start within 2 cm
    assert np.max(np.abs(log[:, 5])) <= 10.0


def test_tracks_a_trapezoid_with_small_error(wbr):
    plant = LinearPlant(wbr)
    path = lambda t: trapezoid_path(t - 1.5, 1.0, 0.6, 2.0)   # the zmp_velocity campaign
    log = run(wbr, plant, ZmpSagittalBalance.from_model(wbr), 10.0, path=path)
    err = log[:, 1] - path(log[:, 0])[0]
    assert np.max(np.abs(err)) < 0.01
    assert abs(err[-1]) < 0.005
    lean = log[:, 2] * plant.k_sc
    assert np.max(np.abs(lean)) < 1.5 * 0.6 * plant.h / G       # ZMP offset close to the LIPM h a / g


def test_rejects_the_push_of_the_benchmark(wbr):
    plant = LinearPlant(wbr)
    dv = 2.7 / wbr.kin.total_mass                   # 2.7 N s on the torso, as in the zmp_push campaign
    log = run(wbr, plant, ZmpSagittalBalance.from_model(wbr), 12.0, impulse_at=1.0, dv=-dv)
    assert np.max(np.abs(log[:, 5])) <= 10.0
    assert abs(log[-1, 1]) < 0.01 and abs(log[-1, 3]) < 0.01     # back at the start, at rest


def test_integral_removes_a_constant_disturbance(wbr):
    """A CoM offset the model does not know (like the torso CoM ahead of the hip) must not leave a drift."""
    plant = LinearPlant(wbr)
    ctrl = ZmpSagittalBalance.from_model(wbr)
    queue = [0.0]
    for k in range(int(15.0 / DT)):
        s, th, sd, thd = plant.x
        tau, _ = ctrl.step(DT, plant.h, s, sd, plant.k_sc * th + 0.01, plant.k_sc * thd)   # 1 cm bias
        queue.append(tau)
        plant.step(queue.pop(0))
    assert abs(plant.x[3]) < 1e-3                   # no residual speed
    assert abs(plant.x[1]) < 0.05


def test_lateral_lean_cancels_the_centripetal_zmp_shift():
    lat = ZmpLateralCompensation(track=0.294, gains=LateralGains(rate_max=100.0))
    h, a_y = 0.19, 1.0
    for _ in range(100):
        lean, dz, info = lat.step(DT, h, h * a_y / G)
    assert math.isclose(math.sin(lean), a_y / G, rel_tol=1e-9)     # the lean of a bicycle
    assert lean > 0 and dz > 0                      # left turn: lean left, right leg longer
    assert math.isclose(dz, 0.294 * math.tan(lean))


def test_lateral_lean_is_rate_limited_and_saturated():
    lat = ZmpLateralCompensation(track=0.294)
    lean, _, _ = lat.step(DT, 0.19, 1.0)
    assert math.isclose(lean, lat.gains.rate_max * DT)
    for _ in range(2000):
        lean, _, _ = lat.step(DT, 0.19, 1.0)
    assert math.isclose(lean, lat.gains.lean_max)


def test_measured_zmp_feedback_moves_the_com_away_from_it():
    lat = ZmpLateralCompensation(track=0.294, gains=LateralGains(k_zmp=0.3, rate_max=100.0))
    for _ in range(500):
        lean, _, _ = lat.step(DT, 0.19, 0.0, y_zmp=-0.02)          # ZMP 2 cm to the right
    assert lean > 0                                              # lean left


def test_lateral_preview_leans_before_the_turn_and_zeroes_the_lipm_zmp():
    """y_c = smooth(h a_y / g) solves y_c - y_c_dd / omega^2 = h a_y / g: the LIPM lateral ZMP stays at 0."""
    h = 0.19
    omega = math.sqrt(G / h)
    a_y = lambda t: 1.5 * np.clip(t - 1.0, 0.0, 1.0) * np.clip(3.0 - t, 0.0, 1.0)   # turn from 1 s to 3 s
    prev = LipmPreview()
    ts = np.arange(0.0, 4.0, DT)
    y = np.array([prev.smooth(lambda tt: h * a_y(tt) / G, t, omega) for t in ts])
    assert y[int(0.9 / DT)] > 0.0                                # leaning 0.1 s before the turn starts
    ydd = np.gradient(np.gradient(y, DT), DT)
    zmp = y - ydd / omega ** 2 - h * a_y(ts) / G
    assert np.max(np.abs(zmp[100:-100])) < 1e-3                  # within 1 mm (kernel truncation, sampling)


def test_lipm_preview_leans_before_the_acceleration_step():
    path = lambda t: trapezoid_path(t - 1.0, 1.0, 0.6, 2.0)
    omega = math.sqrt(G / 0.19)
    acc, _ = LipmPreview()(path, 0.9, omega)
    assert 0.0 < acc < 0.6                          # already leaning forward 0.1 s before the start
    acc, _ = LipmPreview()(path, 2.0, omega)
    assert math.isclose(acc, 0.6, rel_tol=0.02)     # steady acceleration: ZMP h a / g behind the CoM
    acc, jerk = LipmPreview()(lambda t: (0 * t + 0.3, 0 * t), 5.0, omega)
    assert acc == 0.0 and jerk == 0.0               # standing still: no lean


def test_trapezoid_path_matches_the_scalar_profile():
    t = np.linspace(-1, 6, 701)
    s, v = trapezoid_path(t, 1.0, 0.6, 2.0)
    ref = np.array([trapezoid_reference(x, 1.0, 0.6, 2.0)[:2] for x in t])
    assert np.allclose(s, ref[:, 0]) and np.allclose(v, ref[:, 1])


def test_trapezoid_reference():
    s, v, a = zip(*(trapezoid_reference(k * DT, 1.0, 0.6, 2.0) for k in range(int(8 / DT))))
    assert math.isclose(s[-1], 2.0, abs_tol=1e-9) and v[-1] == 0.0
    assert max(v) <= 1.0 + 1e-12 and max(np.abs(np.diff(s))) < 1.0 * DT + 1e-9
    assert np.allclose(np.diff(v) / DT, np.array(a[1:]), atol=0.61)
