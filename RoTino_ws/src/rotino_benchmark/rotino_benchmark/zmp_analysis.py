"""Zero Moment Point study of one campaign: metrics table, plots and a per-law ZMP CSV.

The ZMP is rebuilt from the logged ground-truth base pose and joint angles with the multibody
formula of rotino_description.zmp (Fortress contacts carry no forces, so no CoP can be measured).
Needs CSVs written by the logger with the base_* columns; older logs are skipped.

    ros2 run rotino_benchmark zmp -- <scenario_dir>
"""

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

from rotino_benchmark.common import COLORS, LAWS as CONTROLLERS
from rotino_description import zmp

BASE_COLS = ('base_x_m', 'base_y_m', 'base_z_m')
QUAT_COLS = ('base_qx', 'base_qy', 'base_qz', 'base_qw')
JOINT_COLS = {'left_hip': 'hip_L_pos', 'right_hip': 'hip_R_pos', 'left_knee': 'knee_L_pos',
              'right_knee': 'knee_R_pos', 'left_wheel_joint': 'wheel_L_pos', 'right_wheel_joint': 'wheel_R_pos'}
CONTACT_COLS = ('contact_L_x', 'contact_L_y', 'contact_R_x', 'contact_R_y')
STAMP_COLS = ('odom_stamp_s', 'joints_stamp_s')
MIN_LOAD = 0.2      # below this fraction of m g the ZMP is undefined (wheels almost unloaded)

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        from rotino_description.model import load_urdf_from_package
        _MODEL = zmp.ZmpModel(load_urdf_from_package())
    return _MODEL


def read_log(path):
    """Columns needed by the ZMP study as float arrays; None if the log predates the base_* columns."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(BASE_COLS + QUAT_COLS) <= set(reader.fieldnames):
            return None
        cols = ('time_s',) + BASE_COLS + QUAT_COLS + tuple(JOINT_COLS.values()) + CONTACT_COLS + STAMP_COLS
        rows, phase = {c: [] for c in cols}, []
        for r in reader:
            for c in cols:
                try:
                    rows[c].append(float(r.get(c, 'nan')))
                except (TypeError, ValueError):
                    rows[c].append(float('nan'))
            phase.append(r.get('jump_state', ''))
    d = {c: np.array(v) for c, v in rows.items()}
    ok = np.isfinite(d['time_s']) & np.isfinite(d['base_x_m']) & np.isfinite(d['base_qw'])
    ok &= np.isfinite(np.stack([d[c] for c in JOINT_COLS.values()])).all(0)
    ok &= np.array([p != 'HOLD' for p in phase])            # anchored: the ZMP means nothing yet
    d = {c: v[ok] for c, v in d.items()}
    # jump_state is published on change only and the logger may miss it: while anchored the base
    # does not move at all, so start at the first sample where the base pose changes
    pose = np.stack([d[c] for c in BASE_COLS + QUAT_COLS], -1)
    moved = np.flatnonzero(np.abs(np.diff(pose, axis=0)).max(-1) > 1e-7)
    if moved.size:
        d = {c: v[moved[0]:] for c, v in d.items()}
    return d if len(d['time_s']) > 50 else None


def analyse(d, model=None, window=0.04):
    model = model or load_model()
    stamped = np.isfinite(d['odom_stamp_s']).any() and np.isfinite(d['joints_stamp_s']).any()
    t_base = d['odom_stamp_s'] if stamped else d['time_s']      # older logs: row time for everything
    out = zmp.zmp_series(model, t_base, np.stack([d[c] for c in BASE_COLS], -1),
                         np.stack([d[c] for c in QUAT_COLS], -1),
                         {name: d[c] for name, c in JOINT_COLS.items()}, window=window,
                         joint_t=d['joints_stamp_s'] if stamped else None)
    out['stamped'] = stamped
    # measured contact points (Gazebo) on the same grid, to cross-check the kinematic ones
    fin = np.isfinite(t_base)
    for side in ('L', 'R'):
        xy = [np.interp(out['t'], t_base[fin], d[f'contact_{side}_{a}'][fin], left=np.nan, right=np.nan)
              for a in ('x', 'y')]
        out[f'meas_contact_{side}'] = np.stack(xy, -1)
    # nearly unloaded wheels: the ZMP (a ratio over Fz) is undefined there
    out['valid'] = out['fz'] > MIN_LOAD * model.m.sum() * zmp.G
    for key in ('zmp', 'lipm', 'y_rel', 'e_long', 'margin', 'fn_left', 'fn_right', 'friction_use', 'lam'):
        v = np.array(out[key], dtype=float)
        v[~out['valid']] = np.nan
        out[key] = v
    lam_l, _, e_lipm, _ = zmp.support_metrics(out['lipm'], out['contact_l'], out['contact_r'])
    out['y_rel_lipm'] = 2.0 * lam_l - 1.0
    out['e_long_lipm'] = e_lipm
    return out


def rms(x):
    x = np.asarray(x)[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float('nan')


def peak(x):
    x = np.asarray(x)[np.isfinite(x)]
    return float(np.max(np.abs(x))) if x.size else float('nan')


def zmp_metrics(out):
    half = 0.5 * out['track']
    meas = []
    for side, key in (('L', 'contact_l'), ('R', 'contact_r')):
        e = np.linalg.norm(out[f'meas_contact_{side}'] - out[key][:, :2], axis=-1)
        meas.append(e)
    return {
        'zmp_lat_rms_mm': 1e3 * rms(out['y_rel'] * half),
        'zmp_lat_ratio_pct': 100.0 * peak(out['y_rel']),
        'zmp_margin_min_mm': 1e3 * float(np.nanmin(out['margin'])),
        'wheel_load_min_N': float(np.nanmin(np.minimum(out['fn_left'], out['fn_right']))),
        'zmp_long_rms_mm': 1e3 * rms(out['e_long']),
        'zmp_long_peak_mm': 1e3 * peak(out['e_long']),
        'lipm_err_rms_mm': 1e3 * rms(np.linalg.norm(out['zmp'] - out['lipm'], axis=-1)),
        'friction_use_pct': 100.0 * float(np.nanmax(out['friction_use'])),
        'contact_check_mm': 1e3 * rms(np.concatenate(meas)),
        'unloaded_pct': 100.0 * float(np.mean(~out['valid'])),
    }


ROWS = [
    ('zmp_lat_rms_mm', 'ZMP laterale rms [mm]', '{:.1f}'),
    ('zmp_lat_ratio_pct', 'ZMP laterale max / (d/2) [%]', '{:.1f}'),
    ('zmp_margin_min_mm', 'margine laterale minimo [mm]', '{:.1f}'),
    ('wheel_load_min_N', 'carico minimo ruota scarica [N]', '{:.1f}'),
    ('zmp_long_rms_mm', 'residuo longitudinale rms [mm]', '{:.2f}'),
    ('zmp_long_peak_mm', 'residuo longitudinale picco [mm]', '{:.1f}'),
    ('lipm_err_rms_mm', '|ZMP multicorpo - LIPM| rms [mm]', '{:.2f}'),
    ('friction_use_pct', 'utilizzo attrito max [%]', '{:.1f}'),
    ('contact_check_mm', 'contatti cinematica vs Gazebo rms [mm]', '{:.1f}'),
    ('unloaded_pct', f'tempo con F_z < {MIN_LOAD:.0%} m g [%]', '{:.1f}'),
]


def latest_logs(run_dir):
    found = {}
    for law in CONTROLLERS:
        files = sorted(glob.glob(os.path.join(run_dir, f'rotino_{law}_*.csv')))
        if files:
            found[law] = files[-1]
    return found


def analyse_run(run_dir):
    results = {}
    for law, path in latest_logs(run_dir).items():
        d = read_log(path)
        if d is None:
            print(f'  {law}: {os.path.basename(path)} senza colonne base_* (logger vecchio) - saltato')
            continue
        results[law] = analyse(d)
    return results


def write_zmp_csv(out, path):
    cols = [('t_s', out['t']),
            ('zmp_x_m', out['zmp'][:, 0]), ('zmp_y_m', out['zmp'][:, 1]),
            ('lipm_x_m', out['lipm'][:, 0]), ('lipm_y_m', out['lipm'][:, 1]),
            ('com_x_m', out['com'][:, 0]), ('com_y_m', out['com'][:, 1]), ('com_z_m', out['com'][:, 2]),
            ('zmp_lat_mm', 1e3 * out['y_rel'] * 0.5 * out['track']), ('zmp_y_rel', out['y_rel']),
            ('zmp_long_mm', 1e3 * out['e_long']), ('lipm_long_mm', 1e3 * out['e_long_lipm']),
            ('margin_mm', 1e3 * out['margin']), ('fz_N', out['fz']),
            ('fn_left_N', out['fn_left']), ('fn_right_N', out['fn_right']),
            ('friction_use', out['friction_use'])]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols])
        w.writerows(np.column_stack([v for _, v in cols]).round(6).tolist())


# ---------------------------------------------------------------------------- plots
def generate_zmp_plots(run_dir, out_dir=None, results=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = results if results is not None else analyse_run(run_dir)
    if not results:
        return []
    out_dir = out_dir or os.path.join(run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def t0(o):
        return o['t'] - o['t'][0]

    # 1. lateral ZMP and wheel loads: one row per law, own y scale (a chattering law would flatten the others)
    n = len(results)
    half_mm = 1e3 * 0.5 * float(np.median(next(iter(results.values()))['track']))
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * n + 0.6), sharex=True, squeeze=False)
    for (ax1, ax2), (law, o) in zip(axes, results.items()):
        lat = 1e3 * o['y_rel'] * 0.5 * o['track']
        ax1.axhspan(-half_mm, half_mm, color='0.93', zorder=0)
        for s in (-1, 1):
            ax1.axhline(s * half_mm, color='k', lw=1.0)
        ax1.plot(t0(o), lat, color=COLORS[law], lw=1.1)
        span = max(np.nanmax(np.abs(lat)) if np.isfinite(lat).any() else 0.0, 2.0) * 1.3
        if span < half_mm:
            ax1.set_ylim(-span, span)                       # zoom: the support edges are off-scale
            ax1.text(0.99, 0.95, f'bordi ±{half_mm:.0f} mm fuori scala', transform=ax1.transAxes,
                     ha='right', va='top', fontsize=8, color='0.4')
        ax1.set_ylabel(f'{law.upper()}\nZMP laterale [mm]')
        ax1.grid(alpha=0.3)
        ax2.plot(t0(o), o['fn_left'], color=COLORS[law], lw=1.1, label='sinistra')
        ax2.plot(t0(o), o['fn_right'], color=COLORS[law], lw=1.1, ls='--', label='destra')
        ax2.axhline(0.0, color='r', lw=1, ls=':')
        ax2.set_ylabel('carico [N]')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)
    axes[0, 0].set_title('ZMP laterale (+ = verso ruota SX; fascia grigia = appoggio ±d/2)')
    axes[0, 1].set_title('Carico normale stimato per ruota (da ZMP laterale)')
    axes[-1, 0].set_xlabel('t [s]')
    axes[-1, 1].set_xlabel('t [s]')
    fig.tight_layout()
    saved.append(os.path.join(out_dir, 'zmp_lateral.png'))
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)

    # 2. longitudinal: residual of the multibody ZMP vs the LIPM point, and friction use
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * n + 0.6), sharex=True, squeeze=False)
    for (ax1, ax2), (law, o) in zip(axes, results.items()):
        ax1.plot(t0(o), 1e3 * o['e_long_lipm'], color='0.55', lw=0.9, label='LIPM')
        ax1.plot(t0(o), 1e3 * o['e_long'], color=COLORS[law], lw=1.1, label='multicorpo')
        ax1.axhline(0.0, color='k', lw=0.8)
        ax1.set_ylabel(f'{law.upper()}\ndalla linea contatti [mm]')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(alpha=0.3)
        ax2.plot(t0(o), 100.0 * o['friction_use'], color=COLORS[law], lw=1.1)
        ax2.set_ylabel('|F_t|/(μF_z) [%]')
        ax2.grid(alpha=0.3)
    axes[0, 0].set_title('ZMP longitudinale: multicorpo (≈ 0 se consistente) vs LIPM')
    axes[0, 1].set_title("Utilizzo del cono d'attrito")
    axes[-1, 0].set_xlabel('t [s]')
    axes[-1, 1].set_xlabel('t [s]')
    fig.tight_layout()
    saved.append(os.path.join(out_dir, 'zmp_longitudinal.png'))
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)

    # 3. top view, world frame
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    for ax, (law, o) in zip(axes[0], results.items()):
        step = max(1, int(round(0.5 / float(np.median(np.diff(o['t']))))))
        for k in range(0, len(o['t']), step):
            ax.plot([o['contact_l'][k, 0], o['contact_r'][k, 0]], [o['contact_l'][k, 1], o['contact_r'][k, 1]],
                    color='0.6', lw=1.5)
        ax.plot(o['com'][:, 0], o['com'][:, 1], color='#f0a500', lw=1.2, label='CoM proiettato')
        ax.plot(o['zmp'][:, 0], o['zmp'][:, 1], color=COLORS[law], lw=0.9, label='ZMP multicorpo')
        ax.plot([], [], color='0.6', lw=1.5, label='appoggio ogni 0,5 s')
        ax.set_aspect('equal', 'datalim')
        ax.set_title(f'{law.upper()} – vista dall\'alto')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    saved.append(os.path.join(out_dir, 'zmp_topview.png'))
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)

    # 4. footprint in the robot frame: where the ZMP lives relative to the wheels
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([-half_mm, half_mm], [0, 0], color='k', lw=3, label='segmento di appoggio')
    for law, o in results.items():
        ax.plot(1e3 * o['y_rel'] * 0.5 * o['track'], 1e3 * o['e_long'], '.', ms=1.5, alpha=0.4,
                color=COLORS[law], label=f'{law.upper()} multicorpo')
        ax.plot(1e3 * o['y_rel_lipm'] * 0.5 * o['track'], 1e3 * o['e_long_lipm'], '.', ms=1.0, alpha=0.15,
                color=COLORS[law], mec='none')
    ax.invert_xaxis()                                        # robot seen from above: left wheel on the left
    ax.set_xlabel('laterale [mm] (sinistra ←)')
    ax.set_ylabel('longitudinale [mm] (avanti ↑)')
    ax.set_title('Impronta dello ZMP nel frame del robot (punti tenui: LIPM)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, markerscale=6)
    fig.tight_layout()
    saved.append(os.path.join(out_dir, 'zmp_footprint.png'))
    fig.savefig(saved[-1], dpi=150)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------- CLI
def print_table(run_dir, metrics):
    laws = list(metrics)
    width = max(len(lbl) for _, lbl, _ in ROWS) + 2
    print(f'\nstudio ZMP: {os.path.basename(os.path.normpath(run_dir))}\n')
    print('  ' + 'metrica'.ljust(width) + ''.join(law.upper().rjust(12) for law in laws))
    print('  ' + '-' * (width + 12 * len(laws)))
    for key, label, fmt in ROWS:
        cells = ''
        for law in laws:
            v = metrics[law].get(key, float('nan'))
            cells += ('n/d' if math.isnan(v) else fmt.format(v)).rjust(12)
        print('  ' + label.ljust(width) + cells)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help='campaign directory produced by rotino_benchmark campaign')
    p.add_argument('--out', default=None, help='output directory for plots and ZMP CSVs (default <run>/plots)')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])

    results = analyse_run(args.run_dir)
    if not results:
        print(f'nessun log utilizzabile in {args.run_dir}')
        return 1
    out_dir = args.out or os.path.join(args.run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    for law, o in results.items():
        write_zmp_csv(o, os.path.join(out_dir, f'zmp_{law}.csv'))
    print_table(args.run_dir, {law: zmp_metrics(o) for law, o in results.items()})
    if not args.no_plots:
        for path in generate_zmp_plots(args.run_dir, out_dir, results):
            print(f'  - {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
