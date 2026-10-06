"""CSVs of the error, reference, state and actuator signals of a scenario, split by theme.

    ros2 run rotino_benchmark export -- <scenario_dir>     # -> <scenario_dir>/dati/
    ros2 run rotino_benchmark export -- <suite_dir>        # every scenario of the suite

compare, campaign and suite call it after the comparison. It only reads the logs, when the simulation is over.
One row per control step, the same rows in every file of a law, six significant digits:

    inseguimento_<law>.csv   states, references and tracking errors (pitch, position, speed, heading, height),
                             norms of the errors, disturbance forces
    attuazione_<law>.csv     commanded torques (each joint, common and differential on the wheels), their norms,
                             joint positions and velocities
    stato_<law>.csv          ground-truth torso pose and twist, estimator error of the MPC
    colonne.csv              unit and meaning of every column above

The errors are grouped by unit before taking a norm: angles (pitch and heading) and positions (travel and
height). A column that a log does not have, or that is NaN from start to end, is left out: the estimator error
needs a run with --topic-extra, and logs older than this module have no heading or height error, hence no
error norms either.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np

from rotino_benchmark.common import LAWS, TORQUE_COLUMNS, WHEEL_TORQUE_COLUMNS, find_csv, is_suite, scenario_dirs

DATA_DIR = 'dati'
LEG_TORQUE_COLUMNS = tuple(c for c in TORQUE_COLUMNS if c not in WHEEL_TORQUE_COLUMNS)
NAN = float('nan')


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


class Log:
    """A raw log as float arrays by column name; a column the log does not have reads as NaN."""

    def __init__(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = [r for r in reader if r]
        width = len(header)
        cells = list(zip(*(r[:width] + [''] * (width - len(r)) for r in rows))) if rows else [()] * width
        cols = dict(zip(header, cells))
        self.n, self.cols = 0, {}
        if 'time_s' not in cols:
            return
        ok = np.isfinite(np.array([_float(v) for v in cols['time_s']]))
        self.n = int(ok.sum())
        self.cols = {name: np.array([_float(v) for v in values])[ok] for name, values in cols.items()
                     if name != 'jump_state'}

    def __getitem__(self, name):
        return self.cols[name] if name in self.cols else np.full(self.n, NAN)


# ---------------------------------------------------------------------------- derived signals
def _norm(*signals):
    """Euclidean norm of the signals at every sample (NaN where one of them is missing)."""
    return np.sqrt(sum(s ** 2 for s in signals))


def _rpy(d, axis):
    """Roll, pitch or yaw [deg] of the torso from the ground-truth quaternion."""
    x, y, z, w = (d[c] for c in ('base_qx', 'base_qy', 'base_qz', 'base_qw'))
    angle = (np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
             np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
             np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))[axis]
    return np.degrees(angle)


def _has_height_reference(d):
    """False for a law that publishes its measured CoM height in the place of the reference (the PID): the
    error would read as a perfect zero."""
    err = d['com_z_axle_err_mm']
    return not (np.isfinite(err).any() and not np.nanmax(np.abs(err)) > 0.0)


def _height(column):
    return lambda d: d[column] if _has_height_reference(d) else np.full(d.n, NAN)


# ---------------------------------------------------------------------------- what goes where
JOINTS = OrderedDict([('hip_L', 'anca sinistra'), ('hip_R', 'anca destra'), ('knee_L', 'ginocchio sinistro'),
                      ('knee_R', 'ginocchio destro'), ('wheel_L', 'ruota sinistra'), ('wheel_R', 'ruota destra')])

# (column, unit, meaning, source): the source is a column of the log (None = the same name) or a function of it
THEMES = OrderedDict([
    ('inseguimento', [
        ('theta_deg', 'deg', 'inclinazione del CoM dalla verticale, positiva in avanti', None),
        ('theta_ref_deg', 'deg', 'riferimento di inclinazione', None),
        ('theta_err_deg', 'deg', 'errore di inclinazione, stato - riferimento', None),
        ('theta_dot_degs', 'deg/s', 'velocita di inclinazione', None),
        ('x_m', 'm', "spazio percorso dall'asse delle ruote lungo la direzione di marcia", None),
        ('s_ref_m', 'm', 'riferimento di posizione', None),
        ('s_err_m', 'm', 'errore di posizione, stato - riferimento', None),
        ('xdot_ms', 'm/s', 'velocita di avanzamento', None),
        ('xdot_ref_ms', 'm/s', 'riferimento di velocita', None),
        ('xdot_err_ms', 'm/s', 'errore di velocita, stato - riferimento', None),
        ('yaw_deg', 'deg', "imbardata misurata dall'IMU", None),
        ('yaw_ref_deg', 'deg', 'riferimento di imbardata', None),
        ('yaw_err_deg', 'deg', 'errore di imbardata, stato - riferimento', None),
        ('roll_deg', 'deg', "rollio misurato dall'IMU", None),
        ('com_z_m', 'm', 'altezza del CoM dal suolo (PID: robot intero, MPC: corpo superiore)', None),
        ('com_z_axle_m', 'm', "altezza del CoM sopra l'asse delle ruote", None),
        ('com_z_ref_m', 'm', "riferimento di altezza del CoM sopra l'asse", _height('com_z_ref_m')),
        ('com_z_axle_err_mm', 'mm', 'errore di altezza del CoM, stato - riferimento', _height('com_z_axle_err_mm')),
        ('com_z_vel_ms', 'm/s', 'velocita verticale del CoM', None),
        ('delta_s_m', 'm', "offset desiderato del CoM davanti all'asse (MPC: delta_s, PID: s_des)", None),
        ('err_angle_norm_deg', 'deg', 'norma degli errori angolari: inclinazione e imbardata',
         lambda d: _norm(d['theta_err_deg'], d['yaw_err_deg'])),
        ('err_pos_norm_m', 'm', 'norma degli errori di posizione: avanzamento e altezza del CoM (il PID non ha un '
         'riferimento di altezza: solo avanzamento)', lambda d: _norm(d['s_err_m'], 1e-3 * d['com_z_axle_err_mm'])),
        ('push_force_N', 'N', 'spinta impulsiva sul torso, modulo', None),
        ('step_force_N', 'N', 'forza a gradino sul torso, modulo', None),
    ]),
    ('attuazione', [
        *[(c, 'Nm', f"coppia comandata, {JOINTS[c[:-len('_torque_cmd')]]}", None) for c in TORQUE_COLUMNS],
        ('wheel_torque_common_Nm', 'Nm', 'coppia comune delle ruote, (destra + sinistra) / 2',
         lambda d: 0.5 * (d['wheel_R_torque_cmd'] + d['wheel_L_torque_cmd'])),
        ('wheel_torque_diff_Nm', 'Nm', 'coppia differenziale delle ruote, (destra - sinistra) / 2: positiva = '
         'imbardata verso sinistra', lambda d: 0.5 * (d['wheel_R_torque_cmd'] - d['wheel_L_torque_cmd'])),
        ('tau_norm_Nm', 'Nm', 'norma delle coppie dei sei giunti', lambda d: _norm(*(d[c] for c in TORQUE_COLUMNS))),
        ('wheel_tau_norm_Nm', 'Nm', 'norma delle coppie delle due ruote',
         lambda d: _norm(*(d[c] for c in WHEEL_TORQUE_COLUMNS))),
        ('leg_tau_norm_Nm', 'Nm', 'norma delle coppie delle gambe, anche e ginocchia',
         lambda d: _norm(*(d[c] for c in LEG_TORQUE_COLUMNS))),
        *[(f'{j}_pos', 'rad', f'posizione del giunto, {label}', None) for j, label in JOINTS.items()],
        *[(f'{j}_vel', 'rad/s', f'velocita del giunto, {label}', None) for j, label in JOINTS.items()],
    ]),
    ('stato', [
        *[(f'base_{a}_m', 'm', f'posizione del torso in Gazebo, {a}', None) for a in 'xyz'],
        ('base_roll_deg', 'deg', 'rollio del torso in Gazebo', lambda d: _rpy(d, 0)),
        ('base_pitch_deg', 'deg', "beccheggio del torso in Gazebo (theta_deg e invece l'inclinazione del CoM)",
         lambda d: _rpy(d, 1)),
        ('base_yaw_deg', 'deg', 'imbardata del torso in Gazebo', lambda d: _rpy(d, 2)),
        *[(f'base_v{a}', 'm/s', f'velocita lineare del torso in Gazebo, {a} nel riferimento del torso', None)
          for a in 'xyz'],
        *[(f'base_w{a}', 'rad/s', f'velocita angolare del torso in Gazebo, {a} nel riferimento del torso', None)
          for a in 'xyz'],
        *[(f'est_err_{a}_m', 'm', f'MPC: errore di stima della posizione del torso lungo {a}, stima - Gazebo', None)
          for a in 'xyz'],
        *[(f'est_err_v{a}_ms', 'm/s', f'MPC: errore di stima della velocita del torso lungo {a}, stima - Gazebo',
           None) for a in 'xyz'],
    ]),
])
TIME_COLUMN = ('time_s', 's', "tempo dal rilascio dall'ancora")


def theme_table(d, theme):
    """[(column, unit, meaning, values)] of a theme for one log, time first; [] when the log has none of it."""
    cols = []
    for name, unit, meaning, source in THEMES[theme]:
        values = source(d) if callable(source) else d[source or name]
        if np.isfinite(values).any():
            cols.append((name, unit, meaning, values))
    return [(*TIME_COLUMN, d['time_s']), *cols] if cols else []


def write_theme(d, theme, path):
    """One theme of one log; returns its table, [] if nothing was written."""
    table = theme_table(d, theme)
    if not table:
        return table
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in table])
        w.writerows(zip(*(np.char.mod('%.4f' if name == 'time_s' else '%.6g', values)
                          for name, _, _, values in table)))
    return table


def export_scenario(run_dir):
    """Writes <run_dir>/dati/ from the newest log of each law; returns the directory, None without logs."""
    logs = {law: Log(path) for law in LAWS for path in [find_csv(run_dir, law)] if path}
    logs = {law: d for law, d in logs.items() if d.n}
    if not logs:
        return None
    out_dir = os.path.join(run_dir, DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)

    columns, written = OrderedDict(), 0
    for theme in THEMES:
        for law, d in logs.items():
            table = write_theme(d, theme, os.path.join(out_dir, f'{theme}_{law}.csv'))
            written += bool(table)
            for column, unit, meaning, _ in table:
                columns.setdefault((theme, column), [unit, meaning, []])[2].append(law)
    with open(os.path.join(out_dir, 'colonne.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'colonna', 'unita', 'significato', 'leggi'])
        w.writerows([f'{theme}_<legge>.csv', column, unit, meaning, ' '.join(laws)]
                    for (theme, column), (unit, meaning, laws) in columns.items())
    print(f'{written + 1} CSV in {out_dir}')
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', help='scenario directory (CSVs) or suite directory (one sub-directory per scenario)')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])
    if not os.path.isdir(args.path):
        p.error(f'directory non trovata: {args.path}')
    dirs = scenario_dirs(args.path) if is_suite(args.path) else [args.path]
    return 0 if [d for d in dirs if export_scenario(d)] else 1


if __name__ == '__main__':
    sys.exit(main())
