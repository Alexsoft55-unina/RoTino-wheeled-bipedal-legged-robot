"""Offline checks of the CSV export (export) and of the columns the logger appends: synthetic logs, no simulation."""

import csv
import math
import os

import pytest

from rotino_benchmark import compare, export

N = 400
# torso yawed 30 deg and pitched 10 deg (yaw about z, then pitch about the new y)
QUAT = (-math.sin(math.radians(5.0)) * math.sin(math.radians(15.0)),
        math.sin(math.radians(5.0)) * math.cos(math.radians(15.0)),
        math.cos(math.radians(5.0)) * math.sin(math.radians(15.0)),
        math.cos(math.radians(5.0)) * math.cos(math.radians(15.0)))

OLD = ['time_s', 'jump_state', 'theta_deg', 'theta_ref_deg', 'theta_err_deg', 'x_m', 's_ref_m', 's_err_m', 'yaw_deg',
       'com_z_m', 'com_z_ref_m', 'wheel_L_torque_cmd', 'wheel_R_torque_cmd', 'hip_L_torque_cmd', 'hip_R_torque_cmd',
       'knee_L_torque_cmd', 'knee_R_torque_cmd', 'wheel_L_vel', 'wheel_R_vel', 'base_x_m', 'base_qx', 'base_qy',
       'base_qz', 'base_qw', 'loaded', 'min_wheel_gap_m']
NEW = ['yaw_ref_deg', 'yaw_err_deg', 'com_z_axle_m', 'com_z_axle_err_mm']
EST = ['est_err_x_m', 'est_err_vz_ms']


def write_log(path, new=True, height_err=0.0, extra=(), n=N):
    """Wheels 0.3 / 0.5 Nm, legs 1 Nm each, constant errors: 1.5 deg of pitch, 1 deg of heading, 1 cm of travel."""
    header = OLD + (NEW if new else []) + list(extra)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(n):
            t = 0.002 * (k + 1)
            row = {'time_s': t, 'jump_state': 'BALANCE', 'theta_deg': 2.0, 'theta_ref_deg': 0.5, 'theta_err_deg': 1.5,
                   'x_m': 0.1 * t, 's_ref_m': 0.1 * t - 0.01, 's_err_m': 0.01, 'yaw_deg': 31.0, 'com_z_m': 0.193,
                   'com_z_ref_m': 0.133 + 1e-3 * height_err, 'wheel_L_torque_cmd': 0.3, 'wheel_R_torque_cmd': 0.5,
                   'hip_L_torque_cmd': 1.0, 'hip_R_torque_cmd': 1.0, 'knee_L_torque_cmd': 1.0,
                   'knee_R_torque_cmd': 1.0, 'wheel_L_vel': 2.0, 'wheel_R_vel': 2.0, 'base_x_m': 0.1 * t,
                   'base_qx': QUAT[0], 'base_qy': QUAT[1], 'base_qz': QUAT[2], 'base_qw': QUAT[3], 'loaded': 1,
                   'min_wheel_gap_m': 0.0, 'yaw_ref_deg': 30.0, 'yaw_err_deg': 1.0, 'com_z_axle_m': 0.133,
                   'com_z_axle_err_mm': -height_err, 'est_err_x_m': 0.003, 'est_err_vz_ms': 0.01}
            w.writerow([row[c] for c in header])


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def make_scenario(tmp_path, **mpc):
    write_log(tmp_path / 'rotino_pid_1.csv')
    write_log(tmp_path / 'rotino_mpc_1.csv', height_err=2.0, **mpc)
    return export.export_scenario(str(tmp_path))


def test_every_file_has_one_row_per_control_step(tmp_path):
    out = make_scenario(tmp_path)
    assert out == os.path.join(str(tmp_path), export.DATA_DIR)
    files = [f'{theme}_{law}.csv' for theme in export.THEMES for law in ('pid', 'mpc')]
    assert sorted(os.listdir(out)) == sorted(['colonne.csv', *files])
    for theme in export.THEMES:
        for law in ('pid', 'mpc'):
            rows = read(os.path.join(out, f'{theme}_{law}.csv'))
            assert len(rows) == N and rows[0]['time_s'] == '0.0020' and rows[-1]['time_s'] == f'{0.002 * N:.4f}'
    track = read(os.path.join(out, 'inseguimento_pid.csv'))
    assert float(track[7]['theta_err_deg']) == 1.5 and float(track[7]['yaw_err_deg']) == 1.0
    assert math.isclose(float(track[-1]['s_ref_m']), 0.1 * 0.002 * N - 0.01, rel_tol=1e-5)
    # contact and phase belong to the raw log only
    assert not {'jump_state', 'loaded', 'min_wheel_gap_m'} & set(track[0])
    assert not {'jump_state', 'loaded', 'min_wheel_gap_m'} & set(read(os.path.join(out, 'stato_pid.csv'))[0])


def test_actuator_signals_and_their_norms(tmp_path):
    out = make_scenario(tmp_path)
    act = read(os.path.join(out, 'attuazione_pid.csv'))[10]
    assert list(act)[:7] == ['time_s', *compare.TORQUE_COLUMNS]
    assert math.isclose(float(act['wheel_torque_common_Nm']), 0.4)
    assert math.isclose(float(act['wheel_torque_diff_Nm']), 0.1)                 # right - left: turning left
    assert math.isclose(float(act['wheel_tau_norm_Nm']), math.hypot(0.3, 0.5), rel_tol=1e-5)
    assert math.isclose(float(act['leg_tau_norm_Nm']), 2.0)
    assert math.isclose(float(act['tau_norm_Nm']), math.sqrt(0.3 ** 2 + 0.5 ** 2 + 4.0), rel_tol=1e-5)


def test_error_norms_group_the_errors_by_unit(tmp_path):
    out = make_scenario(tmp_path)
    pid, mpc = (read(os.path.join(out, f'inseguimento_{law}.csv'))[5] for law in ('pid', 'mpc'))
    assert math.isclose(float(pid['err_angle_norm_deg']), math.hypot(1.5, 1.0), rel_tol=1e-5)
    assert math.isclose(float(mpc['err_angle_norm_deg']), math.hypot(1.5, 1.0), rel_tol=1e-5)
    assert math.isclose(float(pid['err_pos_norm_m']), 0.01)                      # no height reference: travel only
    assert math.isclose(float(mpc['err_pos_norm_m']), math.hypot(0.01, 0.002), rel_tol=1e-5)


def test_torso_attitude_comes_from_the_quaternion(tmp_path):
    state = read(os.path.join(make_scenario(tmp_path), 'stato_pid.csv'))[0]
    assert math.isclose(float(state['base_yaw_deg']), 30.0, rel_tol=1e-5)
    assert math.isclose(float(state['base_pitch_deg']), 10.0, rel_tol=1e-5)
    assert abs(float(state['base_roll_deg'])) < 1e-4


def test_columns_follow_what_each_log_has(tmp_path):
    out = make_scenario(tmp_path, extra=EST)
    pid, mpc = (read(os.path.join(out, f'inseguimento_{law}.csv'))[0] for law in ('pid', 'mpc'))
    # the PID publishes its measured height in the place of the reference: no reference, no error
    assert 'com_z_axle_m' in pid and 'com_z_ref_m' not in pid and 'com_z_axle_err_mm' not in pid
    assert float(mpc['com_z_axle_err_mm']) == -2.0 and math.isclose(float(mpc['com_z_ref_m']), 0.135)
    assert float(read(os.path.join(out, 'stato_mpc.csv'))[0]['est_err_vz_ms']) == 0.01
    assert 'est_err_x_m' not in read(os.path.join(out, 'stato_pid.csv'))[0]
    listed = {(r['file'], r['colonna']): r for r in read(os.path.join(out, 'colonne.csv'))}
    assert listed[('stato_<legge>.csv', 'est_err_x_m')]['leggi'] == 'mpc'
    assert listed[('inseguimento_<legge>.csv', 'theta_err_deg')]['leggi'] == 'pid mpc'
    assert listed[('inseguimento_<legge>.csv', 'theta_err_deg')]['unita'] == 'deg'
    written = {(f'{theme}_<legge>.csv', c) for theme in export.THEMES for law in ('pid', 'mpc')
               for c in read(os.path.join(out, f'{theme}_{law}.csv'))[0]}
    assert written == set(listed)


def test_logs_older_than_the_new_columns_still_export(tmp_path):
    write_log(tmp_path / 'rotino_pid_1.csv', new=False)
    out = export.export_scenario(str(tmp_path))
    row = read(os.path.join(out, 'inseguimento_pid.csv'))[0]
    assert float(row['theta_err_deg']) == 1.5 and float(row['com_z_ref_m']) == 0.133
    assert not {'yaw_err_deg', 'com_z_axle_m', 'err_angle_norm_deg', 'err_pos_norm_m'} & set(row)
    assert 'leg_tau_norm_Nm' in read(os.path.join(out, 'attuazione_pid.csv'))[0]
    assert sorted(os.listdir(out)) == ['attuazione_pid.csv', 'colonne.csv', 'inseguimento_pid.csv', 'stato_pid.csv']


def test_empty_or_truncated_logs_do_not_break_the_export(tmp_path):
    (tmp_path / 'rotino_pid_1.csv').write_text(','.join(OLD) + '\n')
    assert export.export_scenario(str(tmp_path)) is None
    write_log(tmp_path / 'rotino_mpc_1.csv', n=50)
    with open(tmp_path / 'rotino_mpc_1.csv', 'a') as f:
        f.write('0.102,BALANCE,2.0')                         # the logger was killed while writing a row
    out = export.export_scenario(str(tmp_path))
    rows = read(os.path.join(out, 'attuazione_mpc.csv'))
    assert len(rows) == 51 and rows[-1]['wheel_L_torque_cmd'] == 'nan'
    assert export.main([str(tmp_path)]) == 0


def test_analyse_scenario_writes_the_data_once(tmp_path, capsys):
    header = ['time_s', 'theta_deg', 'x_m', 'xdot_ms', 'wheel_u', 'com_z_m', 'fn_total_N', 'loaded',
              'wheel_L_torque_cmd', 'wheel_R_torque_cmd', 'theta_err_deg', 's_err_m', 'xdot_err_ms', 'push_force_N']
    with open(tmp_path / 'rotino_pid_1.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([0.002 * k, 1.0, 0.0, 0.0, 0.0, 0.19, 42.0, 1, 0.5, 0.5, 1.0, 0.01, 0.0, 0.0] for k in range(2000))
    assert compare.analyse_scenario(str(tmp_path), plots=False)
    track = os.path.join(str(tmp_path), export.DATA_DIR, 'inseguimento_pid.csv')
    assert len(read(track)) == 2000
    stamp = os.path.getmtime(track)
    os.utime(track, (stamp - 100.0, stamp - 100.0))
    assert compare.analyse_scenario(str(tmp_path), plots=False)          # cached: the CSVs are not rebuilt
    assert os.path.getmtime(track) == stamp - 100.0
    assert compare.analyse_scenario(str(tmp_path), recompute=True, plots=False)
    assert os.path.getmtime(track) > stamp - 100.0
    capsys.readouterr()


# ---------------------------------------------------------------------------- logger
def test_logger_writes_one_value_per_column(tmp_path):
    rclpy = pytest.importorskip('rclpy')
    from std_msgs.msg import Float64MultiArray
    from rotino_benchmark import logger

    def array(values):
        return Float64MultiArray(data=[float(v) for v in values])

    wbr = [0.0] * 18
    wbr[5], wbr[6], wbr[9], wbr[10] = math.radians(31.0), math.radians(30.0), 0.160, 0.162
    rclpy.init(args=['--ros-args', '-p', f'output_dir:={tmp_path}', '-p', 'controller:=mpc',
                     '-p', 'extra_topics:=true'])
    try:
        node = logger.TestBenchLogger()
        node._debug_cb(array([0.05, 0, 0, 0, 0, 0, 0.19, 0, 42, 0, 1]))       # nothing else received yet
        node._wbr_cb(array(wbr))
        node._est_err_cb(array([1e-3, 2e-3, 3e-3, 0.01, 0.02, 0.03]))
        node._debug_cb(array([0.1, 0, 0, 0, 0, 0, 0.19, 0, 42, 0, 1]))
        path = node.csv_path
        node.close()
        node.destroy_node()
    finally:
        rclpy.shutdown()
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0] == logger.HEADER and len(rows) == 3
    assert all(len(r) == len(logger.HEADER) for r in rows)
    empty, full = (dict(zip(rows[0], r)) for r in rows[1:])
    assert empty['yaw_err_deg'] == empty['com_z_axle_m'] == empty['est_err_x_m'] == 'nan'
    assert math.isclose(float(full['yaw_ref_deg']), 30.0) and math.isclose(float(full['yaw_err_deg']), 1.0)
    assert math.isclose(float(full['com_z_axle_m']), 0.160) and math.isclose(float(full['com_z_axle_err_mm']), -2.0)
    assert float(full['est_err_x_m']) == 1e-3 and float(full['est_err_vz_ms']) == 0.03
