"""Offline checks of the PID vs MPC comparison (compare, scenarios): no simulation, synthetic logs."""

import csv
import json
import math
import os

import pytest

from rotino_benchmark import compare
from rotino_benchmark.common import LAWS, find_csv, is_suite, scenario_dirs
from rotino_benchmark.scenarios import SCENARIOS

HEADER = ['time_s', 'theta_deg', 'x_m', 'xdot_ms', 'wheel_u', 'com_z_m', 'fn_total_N', 'loaded',
          'wheel_L_torque_cmd', 'wheel_R_torque_cmd', 'theta_err_deg', 's_err_m', 'xdot_err_ms', 'push_force_N']


def write_log(path, pitch_amp, pos_err, n=4000):
    """A damped pitch oscillation and a constant position error: easy to predict metrics."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for k in range(n):
            t = 0.002 * k
            th = pitch_amp * math.exp(-(t - 2.0)) * math.cos(6.0 * (t - 2.0)) if t >= 2.0 else 0.0
            w.writerow([t, th, 0.1 * t, 0.1, 0.0, 0.19, 42.0, 1, 0.5, 0.5, th, pos_err, 0.0, 0.0])


def test_catalogue_uses_known_metrics_and_arguments():
    for name, sc in SCENARIOS.items():
        assert sc.key, name
        assert all(k in compare.METRIC for k in sc.key), name
        assert all(':=' in a for a in sc.args), name
        assert sc.duration >= 10.0


def test_better_follows_the_direction_of_the_metric():
    assert compare.better('pitch_peak_deg', {'pid': 5.0, 'mpc': 8.0}) == 'pid'        # lower is better
    assert compare.better('wheel_load_min_N', {'pid': 5.0, 'mpc': 8.0}) == 'mpc'      # higher is better
    assert compare.better('pitch_peak_deg', {'pid': 5.0, 'mpc': 5.1}) == '='          # within 5 %
    assert compare.better('travel_m', {'pid': 1.0, 'mpc': 3.0}) == ''                 # descriptive
    assert compare.better('pitch_peak_deg', {'pid': 5.0, 'mpc': float('nan')}) == ''
    assert compare.better('pitch_rms_deg', {'pid': 0.001, 'mpc': 0.0}) == '='         # under the resolution


def test_metrics_of_a_synthetic_log(tmp_path):
    p = tmp_path / 'rotino_pid_1.csv'
    write_log(p, 4.0, 0.02)
    m = compare.metrics(compare.read_csv(str(p)))
    assert math.isclose(m['pitch_peak_deg'], 4.0, rel_tol=1e-6)
    assert math.isclose(m['pos_err_rms_m'], 0.02, rel_tol=1e-6)
    assert math.isclose(m['pos_err_max_m'], 0.02, rel_tol=1e-6)
    assert math.isclose(m['wheel_tau_rms_Nm'], 0.5, rel_tol=1e-6)
    assert m['chatter_Nm'] == 0.0 and m['airborne_pct'] == 0.0
    assert 0.0 < m['recovery_s'] < 6.0


def test_release_transient_is_not_the_peak(tmp_path):
    p = tmp_path / 'rotino_pid_1.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for k in range(4000):
            t = 0.002 * k
            th = 4.7 * math.exp(-5 * t) + (1.0 if 3.0 < t < 3.5 else 0.0)   # release, then a 1 deg event
            w.writerow([t, th, 0, 0, 0, 0.19, 42, 1, 0, 0, th, 0, 0, 0])
    assert math.isclose(compare.metrics(compare.read_csv(str(p)))['pitch_peak_deg'], 1.0, rel_tol=1e-3)


def make_suite(root):
    for name, (pid, mpc) in (('spinta', (6.0, 9.0)), ('trapezio', (3.0, 2.0))):
        d = root / name
        d.mkdir()
        write_log(d / 'rotino_pid_1.csv', pid, 0.01)
        write_log(d / 'rotino_mpc_1.csv', mpc, 0.03)
        (d / 'scenario.json').write_text(json.dumps({'name': name}))
    return root


def test_suite_report_counts_the_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'zmp_row_metrics', lambda path, result=None: {})   # no URDF needed
    suite = make_suite(tmp_path)
    assert is_suite(str(suite)) and len(scenario_dirs(str(suite))) == 2
    results = compare.analyse_suite(str(suite), plots=False)
    names = [compare.scenario_name(d) for d, _ in results]
    assert names == ['spinta', 'trapezio']                      # catalogue order, not alphabetical
    text = (suite / 'riepilogo.md').read_text()
    assert 'Spinta sul torso' in text and 'Trapezio' in text
    rows = list(csv.DictReader(open(suite / 'riepilogo.csv')))
    peak = {r['scenario']: r['migliore'] for r in rows if r['metrica'] == 'pitch_peak_deg'}
    assert peak == {'spinta': 'pid', 'trapezio': 'mpc'}
    for d in scenario_dirs(str(suite)):
        assert os.path.exists(os.path.join(d, 'confronto.md'))
        assert os.path.exists(os.path.join(d, 'metriche.json'))


def test_cached_metrics_are_reused_until_the_logs_change(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(compare, 'zmp_row_metrics', lambda path, result=None: calls.append(path) or {})
    d = tmp_path / 'spinta'
    d.mkdir()
    write_log(d / 'rotino_pid_1.csv', 5.0, 0.0)
    compare.scenario_metrics(str(d))
    compare.scenario_metrics(str(d))
    assert len(calls) == 1                                       # second call from metriche.json
    write_log(d / 'rotino_mpc_1.csv', 5.0, 0.0)                  # a new log invalidates the cache
    found = compare.scenario_metrics(str(d))
    assert set(found) == set(LAWS) and len(calls) == 3
    assert find_csv(str(d), 'mpc').endswith('rotino_mpc_1.csv')


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_scenario_report_marks_the_key_metrics(tmp_path, monkeypatch, name):
    monkeypatch.setattr(compare, 'zmp_row_metrics', lambda path, result=None: {})
    d = tmp_path / name
    d.mkdir()
    write_log(d / 'rotino_pid_1.csv', 5.0, 0.0)
    write_log(d / 'rotino_mpc_1.csv', 4.0, 0.0)
    found = compare.scenario_metrics(str(d))
    compare.write_scenario_report(str(d), name, found)
    md = (d / 'confronto.md').read_text()
    assert SCENARIOS[name].title in md
    for key in SCENARIOS[name].key:
        assert f'**{compare.METRIC[key][1]}**' in md


# ---------------------------------------------------------------------------- disturbances and platforms
STEP_HEADER = HEADER + ['step_force_N', 'roll_deg', 'yaw_deg']


def write_step_log(path, lean_deg, drift_m, n=8000):
    """A 3 N step at t = 4 s: the pitch moves to `lean_deg` with a damped transient, the position error to
    `drift_m`; roll and yaw wiggle once around t = 10 s."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(STEP_HEADER)
        for k in range(n):
            t = 0.002 * k
            on = t >= 4.0
            th = lean_deg * (1.0 - math.exp(-2.0 * (t - 4.0)) * math.cos(5.0 * (t - 4.0))) if on else 0.0
            err = drift_m * (1.0 - math.exp(-(t - 4.0))) if on else 0.0
            roll = 2.0 if 10.0 <= t < 10.2 else 0.0
            yaw = 179.0 if 10.0 <= t < 10.2 else -178.0                  # wraps across +-180 deg
            tau = 1.5 if 4.0 <= t < 4.1 else 0.2
            w.writerow([t, th, 0, 0, 0, 0.19, 42, 1, tau, -tau, th, err, 0, 0, 3.0 if on else 0.0, roll, yaw])


def test_step_metrics_measure_the_new_equilibrium(tmp_path):
    p = tmp_path / 'rotino_mpc_1.csv'
    write_step_log(p, 4.0, 0.3)
    m = compare.metrics(compare.read_csv(str(p)))
    assert math.isclose(m['pitch_ss_deg'], 4.0, abs_tol=0.01)          # leaning into the force
    assert math.isclose(m['pos_err_ss_m'], 0.3, abs_tol=0.01)          # no integral: a residual error
    assert 0.5 < m['settle_ss_s'] < 3.0                                 # measured from the onset, around 4 deg
    assert math.isclose(m['roll_peak_deg'], 2.0, abs_tol=1e-9)
    assert math.isclose(m['yaw_dev_max_deg'], 3.0, abs_tol=1e-6)       # 179 vs -178: 3 deg, not 357
    assert math.isclose(m['tau_peak_Nm'], 1.5, abs_tol=1e-9)


def test_without_disturbance_there_is_no_settling_time(tmp_path):
    p = tmp_path / 'rotino_pid_1.csv'
    write_log(p, 4.0, 0.02)                                             # old layout: no step/roll/yaw columns
    m = compare.metrics(compare.read_csv(str(p)))
    assert math.isnan(m['settle_ss_s']) and math.isnan(m['roll_peak_deg'])
    assert math.isclose(m['pos_err_ss_m'], 0.02, rel_tol=1e-6)


def _launch_value(args, key, default=0.0):
    for a in args:
        k, v = a.split(':=')
        if k == key:
            return float(v)
    return default


@pytest.mark.parametrize('name', [n for n, sc in SCENARIOS.items() if sc.zones])
def test_platform_obstacles_lie_on_the_path(name):
    """The obstacles must be crossed: within the travelled distance, and the robot must spawn before them."""
    sc = SCENARIOS[name]
    assert 'velocity_enable:=true' in sc.args
    distance = _launch_value(sc.args, 'velocity_distance')
    for z0, z1, _ in sc.zones:
        assert 0.5 < z0 < z1 < distance, (name, z0, z1, distance)
    # time to drive the trapezoid (motion starts 2 s after release) fits in the log
    v, a = _launch_value(sc.args, 'velocity_max'), _launch_value(sc.args, 'accel_max')
    assert 2.0 + distance / v + v / a + 3.0 < sc.duration


def test_platform_spawn_points_are_clear_of_the_obstacles():
    """Spawn pose + zones give the obstacles in world coordinates: they must match rotino_world.sdf."""
    expected = {'dossi': (6.0, 7.6), 'rampa': (-5.0, -8.0), 'piastrelle': (-4.13, -6.12)}
    for name, (first, last) in expected.items():
        sc = SCENARIOS[name]
        x0, y0, yaw = (_launch_value(sc.args, k) for k in ('spawn_x', 'spawn_y', 'spawn_yaw'))
        start = x0 if abs(math.sin(yaw)) < 0.5 else y0
        sign = math.cos(yaw) if abs(math.sin(yaw)) < 0.5 else math.sin(yaw)
        z_first, z_last = sc.zones[0][0], sc.zones[-1][1]
        assert abs(start + sign * z_first - first) < 0.1, name
        assert abs(start + sign * z_last - last) < 0.15, name


def test_step_scenario_starts_the_disturbance_node():
    from rotino_benchmark.campaign import disturbance_cmd
    sc = SCENARIOS['gradino']
    cmd = disturbance_cmd(sc.disturbance)
    assert cmd[:4] == ['ros2', 'run', 'rotino_benchmark', 'disturbance']
    assert 'force:=3.0' in cmd and 'start_time:=4.0' in cmd
    assert disturbance_cmd(None) is None


def test_zone_metrics_ignore_the_start_and_see_the_obstacle(tmp_path):
    """A 6 deg lean when accelerating at the start must not count; a 3 deg bump at 1.5 m must."""
    p = tmp_path / 'rotino_pid_1.csv'
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for k in range(8000):
            t = 0.002 * k
            x = max(0.0, 0.5 * (t - 2.0))                                # motion starts at 2 s
            th = 6.0 if 0.0 < x < 0.2 else (3.0 if 1.45 < x < 1.6 else 0.0)
            z = 0.19 + (0.02 if 1.45 < x < 1.55 else 0.0)
            tau = 2.0 if 0.0 < x < 0.2 else (0.8 if 1.45 < x < 1.6 else 0.1)
            w.writerow([t, th, x, 0.5, 0, z, 42, 1, tau, tau, th, 0, 0, 0])
    m = compare.metrics(compare.read_csv(str(p)), zones=((1.44, 1.56, 'dosso'),))
    assert m['pitch_peak_deg'] == 6.0                                   # the whole run sees the start
    assert m['zone_pitch_peak_deg'] == 3.0 and m['zone_tau_peak_Nm'] == 0.8
    assert math.isclose(m['zone_com_dev_mm'], 20.0, abs_tol=1e-6)
    assert math.isnan(compare.metrics(compare.read_csv(str(p)))['zone_pitch_peak_deg'])


# ---------------------------------------------------------------------------- velocity error and torque norm
LEG_HEADER = HEADER + ['hip_L_torque_cmd', 'hip_R_torque_cmd', 'knee_L_torque_cmd', 'knee_R_torque_cmd',
                       'xdot_ref_ms']


def write_torque_log(path, n=4000):
    """Wheels 0.3 / 0.4 Nm and legs 1 Nm each (norm 2.06 Nm), with a release spike and a 2 Nm wheel event."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(LEG_HEADER)
        for k in range(n):
            t = 0.002 * k
            leg = 9.0 if t < 0.1 else 1.0
            wl = 2.0 if 3.0 <= t < 3.5 else 0.3
            w.writerow([t, 0, 0.1 * t, 0.12, 0, 0.19, 42, 1, wl, 0.4, 0, 0, 0.02, 0, leg, leg, leg, leg, 0.1])


def test_torque_norm_uses_every_joint_and_skips_the_release(tmp_path):
    p = tmp_path / 'rotino_pid_1.csv'
    write_torque_log(p)
    m = compare.metrics(compare.read_csv(str(p)))
    assert math.isclose(m['tau_norm_peak_Nm'], math.sqrt(2.0 ** 2 + 0.4 ** 2 + 4.0), rel_tol=1e-9)
    assert 0.5 < m['wheel_tau_norm_rms_Nm'] < 2.0 and m['tau_norm_rms_Nm'] > m['wheel_tau_norm_rms_Nm']
    assert math.isclose(m['vel_err_rms_ms'], 0.02, rel_tol=1e-9)
    write_log(p, 4.0, 0.0)                                              # a log without the leg torques
    m = compare.metrics(compare.read_csv(str(p)))
    assert math.isnan(m['tau_norm_rms_Nm']) and math.isclose(m['wheel_tau_norm_rms_Nm'], math.hypot(0.5, 0.5))


def test_velocity_torque_csv_has_one_row_per_sample(tmp_path):
    from rotino_benchmark import plot
    write_torque_log(tmp_path / 'rotino_pid_1.csv', n=2000)
    data = plot.load_run_data(str(tmp_path))
    path, = plot.write_velocity_torque_csv(data, str(tmp_path))
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert os.path.basename(path) == 'velocita_coppie_pid.csv' and len(rows) == 2000
    last = rows[-1]
    assert math.isclose(float(last['xdot_err_ms']), 0.02) and math.isclose(float(last['xdot_ref_ms']), 0.1)
    assert math.isclose(float(last['tau_norm_Nm']), math.sqrt(0.3 ** 2 + 0.4 ** 2 + 4.0), rel_tol=1e-9)
    assert math.isclose(float(last['wheel_tau_norm_Nm']), 0.5, rel_tol=1e-9)
    assert os.path.exists(plot.plot_velocity_torque(data, str(tmp_path)))
