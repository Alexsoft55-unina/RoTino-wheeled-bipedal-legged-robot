"""Offline checks of the ZMP reconstruction (rotino_description.zmp) used by the benchmark and the dashboard."""

import math
import shutil
import subprocess

import numpy as np
import pytest

from rotino_description import zmp
from rotino_description.zmp import G

DT = 0.002


def test_savgol_recovers_polynomial_derivatives_exactly():
    t = np.arange(200) * DT
    y = 0.3 + 2.0 * t - 1.5 * t ** 2
    assert np.allclose(zmp.savgol_filter(y, 21, 2, 0, DT), y)
    assert np.allclose(zmp.savgol_filter(y, 21, 2, 1, DT), 2.0 - 3.0 * t)
    assert np.allclose(zmp.savgol_filter(y, 21, 2, 2, DT), -3.0)


def test_savgol_second_derivative_of_noisy_sine():
    rng = np.random.default_rng(0)
    t = np.arange(2000) * DT
    w = 2 * math.pi * 1.5
    y = 0.01 * np.sin(w * t) + 1e-6 * rng.standard_normal(t.size)
    a = zmp.savgol_filter(y, 21, 2, 2, DT)
    ref = -0.01 * w * w * np.sin(w * t)
    assert np.sqrt(np.mean((a - ref)[50:-50] ** 2)) < 0.05 * 0.01 * w * w


def test_static_zmp_is_com_projection():
    m = np.array([1.0, 2.0, 0.5])
    p = np.array([[0.1, 0.0, 0.3], [-0.05, 0.02, 0.2], [0.0, -0.1, 0.05]])
    z, fz = zmp.multibody_zmp(m, p, np.zeros_like(p), np.zeros_like(p))
    com = (m[:, None] * p).sum(0) / m.sum()
    assert np.allclose(z, com[:2])
    assert math.isclose(fz, m.sum() * G)


def test_constant_lateral_acceleration_shifts_zmp_by_lipm_law():
    m = np.array([3.0])
    p = np.array([[0.0, 0.0, 0.2]])
    a = np.array([[0.0, 2.0, 0.0]])
    z, _ = zmp.multibody_zmp(m, p, a, np.zeros_like(p))
    assert np.allclose(z, [0.0, -0.2 * 2.0 / G])
    assert np.allclose(zmp.lipm_zmp(p[0], a[0]), z)


def test_angular_momentum_rate_moves_zmp():
    m = np.array([2.0])
    p = np.array([[0.0, 0.0, 0.2]])
    dL = np.array([[0.0, 0.1, 0.0]])            # spinning up nose-down about +y
    z, _ = zmp.multibody_zmp(m, p, np.zeros_like(p), dL)
    assert np.allclose(z, [-0.1 / (2.0 * G), 0.0])


def test_support_metrics_frame_follows_heading():
    for yaw in (0.0, 0.7, -2.5):
        c, s = math.cos(yaw), math.sin(yaw)
        fwd, left = np.array([c, s]), np.array([-s, c])
        cl, cr = np.r_[0.15 * left, 0.0], np.r_[-0.15 * left, 0.0]
        point = 0.075 * left + 0.02 * fwd
        lam, y_rel, e_long, d = zmp.support_metrics(point, cl, cr)
        assert math.isclose(d, 0.3)
        assert math.isclose(y_rel, 0.5, abs_tol=1e-12)
        assert math.isclose(lam, 0.75, abs_tol=1e-12)
        assert math.isclose(e_long, 0.02, abs_tol=1e-12)
    fl, fr = zmp.wheel_loads(np.array([0.75, 1.3]), np.array([40.0, 40.0]))
    assert np.allclose(fl, [30.0, 40.0]) and np.allclose(fr, [10.0, 0.0])


@pytest.fixture(scope='module')
def model():
    if shutil.which('xacro') is None:
        pytest.skip('xacro not available')
    try:
        share = subprocess.check_output(['ros2', 'pkg', 'prefix', 'rotino_description']).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('rotino_description not built/sourced')
    xml = subprocess.check_output(['xacro', f'{share}/share/rotino_description/urdf/rotino.urdf.xacro']).decode()
    return zmp.ZmpModel(xml), xml


def test_urdf_robot_at_rest(model):
    m, _ = model
    N = 300
    t = np.arange(N) * DT
    out = zmp.zmp_series(m, t, np.tile([0.0, 0.0, 0.2439], (N, 1)), np.tile([0, 0, 0, 1.0], (N, 1)), {})
    assert np.allclose(out['fz'], m.m.sum() * G)
    assert np.allclose(out['zmp'], out['com'][:, :2])
    assert np.allclose(out['y_rel'], 0.0, atol=1e-12)
    assert np.allclose(out['fn_left'], out['fn_right'])
    assert np.allclose(out['contact_l'][:, 2], 0.0, atol=1e-3)     # wheels touch z = 0 at spawn height


def test_urdf_constant_turn_matches_centripetal_shift(model):
    """Rigid robot driving on a circle: the ZMP moves outward by z_com a_c / g (no pitch/roll change)."""
    m, xml = model
    N, R, v = 600, 1.0, 0.8
    t = np.arange(N) * DT
    ang = v / R * t
    pos = np.stack([R * np.sin(ang), R * (1 - np.cos(ang)), np.full(N, 0.2439)], -1)
    quat = np.stack([np.zeros(N), np.zeros(N), np.sin(ang / 2), np.cos(ang / 2)], -1)
    out = zmp.zmp_series(m, t, pos, quat, {})
    k = N // 2
    a_c = v * v / R
    shift = -out['com'][k, 2] * a_c / G            # left turn: centre on +y side, ZMP moves right (outward)
    # rotation about z at constant rate adds no dL, so multibody and LIPM agree up to link offsets
    assert math.isclose(out['y_rel'][k] * 0.5 * out['track'][k], shift, rel_tol=0.05)
    assert out['fn_right'][k] > out['fn_left'][k]
    assert math.isclose(out['friction_use'][k], a_c / (m.mu * G), rel_tol=0.02)

    est = zmp.ZmpEstimator(xml)
    res = None
    for i in range(k + 1):
        res = est.update(t[i], pos[i], quat[i], {}) or res
    assert math.isclose(res['y_rel'], out['y_rel'][k], rel_tol=0.05)


def test_repeated_log_rows_are_resampled_on_their_stamps(model):
    """Logger rows repeat the latest odom/joint message: with the stamps the ZMP must not see the steps."""
    m, _ = model
    N, v = 400, 0.8
    t = np.arange(N) * DT
    pos = np.stack([v * t + 0.05 * np.sin(4 * t), np.zeros(N), np.full(N, 0.2439)], -1)
    quat = np.tile([0, 0, 0, 1.0], (N, 1))
    clean = zmp.zmp_series(m, t, pos, quat, {})
    rng = np.random.default_rng(1)
    idx = np.maximum.accumulate(np.clip(np.arange(N) - rng.integers(0, 2, N), 0, None))  # stale rows
    stepped = zmp.zmp_series(m, t[idx], pos[idx], quat[idx], {}, joint_t=t[idx])
    k = slice(50, -50)
    common = np.intersect1d(np.round(clean['t'][k], 6), np.round(stepped['t'], 6))
    a = clean['zmp'][np.isin(np.round(clean['t'], 6), common)]
    b = stepped['zmp'][np.isin(np.round(stepped['t'], 6), common)]
    assert np.max(np.abs(a - b)) < 2e-3


def test_online_estimator_handles_dropped_samples_and_split_streams(model):
    """Live odom/joint_states arrive separately and drop ~3 % of the samples: the causal estimator
    must still match the analytic turn (uniform-weight fits put samples after a gap at the wrong time)."""
    m, xml = model
    N, R, v = 800, 1.0, 0.8
    t = np.arange(N) * DT
    ang = v / R * t
    pos = np.stack([R * np.sin(ang), R * (1 - np.cos(ang)), np.full(N, 0.2439)], -1)
    quat = np.stack([np.zeros(N), np.zeros(N), np.sin(ang / 2), np.cos(ang / 2)], -1)
    rng = np.random.default_rng(3)
    drop_base, drop_joint = rng.random(N) < 0.03, rng.random(N) < 0.03
    est = zmp.ZmpEstimator(xml)
    got = []
    for i in range(N):
        if not drop_joint[i]:
            got.append(est.push_joints(t[i], {'left_hip': 0.0}))
        if not drop_base[i]:
            got.append(est.push_base(t[i], pos[i], quat[i]))
    got = [r for r in got if r is not None]
    lat = np.array([r['y_rel'] * 0.5 * r['track'] for r in got[len(got) // 3:]])
    e_long = np.array([r['e_long'] for r in got[len(got) // 3:]])
    ref = -got[-1]['com'][2] * v * v / R / G
    assert np.max(np.abs(lat - ref)) < 0.05 * abs(ref)
    assert np.max(np.abs(e_long - e_long.mean())) < 1e-3
