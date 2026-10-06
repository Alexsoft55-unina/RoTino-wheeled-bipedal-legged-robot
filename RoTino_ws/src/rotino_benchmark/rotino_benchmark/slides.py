"""Slide versions of the comparison figures: one message per figure, 16:9, large type, the release left out.

    ros2 run rotino_benchmark slides -- <scenario_dir>     # -> <scenario_dir>/slide/*.png
    ros2 run rotino_benchmark slides -- <suite_dir>        # every scenario of the suite

The figures of plot.py are made for a report: up to eight panels and small type. These are made for a projector,
at most three panels each, and start RELEASE_SKIP_S after the release, whose transient would set the scale of
every axis. Which ones are drawn depends on what the scenario does:

    errori.png        pitch and position error                                    always
    coppie.png        common and differential wheel torque                        always
    avvio.png         speeds and pitch around the start of the scripted motion    scenarios with a motion
    disturbo.png      pitch after the push or the step force, phase portrait      scenarios with a disturbance
    terreno.png       pitch, roll and CoM height along the path, obstacles shaded platform scenarios
    percorso.png      path of the torso seen from above                           planar scenarios
    zmp_laterale.png  lateral ZMP and load of the lighter wheel                   planar scenarios with plots/zmp_*.csv
    quota.png         CoM height and pitch                                        height and jump scenarios
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rotino_benchmark.common import COLORS, LABELS, LAWS, find_csv, is_suite, scenario_dirs  # noqa: E402
from rotino_benchmark.compare import DIST_THRESHOLD_N, RELEASE_SKIP_S, scenario_name  # noqa: E402
from rotino_benchmark.export import Log  # noqa: E402
from rotino_benchmark.scenarios import SCENARIOS  # noqa: E402

SLIDE_DIR = 'slide'
INK, MUTED, ZONE = '#222222', '#777777', '#b9770e'
STYLE = {
    'figure.figsize': (13.333, 7.5), 'savefig.dpi': 150, 'font.size': 19, 'axes.labelsize': 19,
    'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 18, 'lines.linewidth': 2.6,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.9, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': MUTED, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
}
TIME_LABEL = 'tempo dal rilascio [s]'
TERRAIN_BEFORE_M, TERRAIN_AFTER_M = 0.4, 0.8     # terreno.png: metres shown before the first and after the last obstacle


def load(run_dir):
    """{law: Log} of the laws with a log in the scenario directory."""
    logs = {law: Log(path) for law in LAWS for path in [find_csv(run_dir, law)] if path}
    return {law: d for law, d in logs.items() if d.n}


def scenario_args(run_dir):
    """Launch arguments of the run: scenario.json, else the catalogue."""
    try:
        with open(os.path.join(run_dir, 'scenario.json')) as f:
            return list(json.load(f).get('args') or [])
    except (OSError, ValueError):
        sc = SCENARIOS.get(scenario_name(run_dir))
        return list(sc.args) if sc else []


def late(d):
    """Samples after the release transient."""
    t = d['time_s']
    return t >= t[0] + RELEASE_SKIP_S


def onset(logs):
    """(time, 'spinta' | 'gradino') of the first disturbance of the run, or (None, None)."""
    found = []
    for d in logs.values():
        for column, kind in (('push_force_N', 'spinta'), ('step_force_N', 'gradino')):
            hit = np.flatnonzero(np.nan_to_num(d[column]) > DIST_THRESHOLD_N)
            if hit.size:
                found.append((float(d['time_s'][hit[0]]), kind))
    return min(found) if found else (None, None)


def motion_start(logs):
    """Time at which the reference speed leaves zero, or None."""
    starts = [float(d['time_s'][hit[0]]) for d in logs.values()
              for hit in [np.flatnonzero(np.abs(np.nan_to_num(d['xdot_ref_ms'])) > 1e-3)] if hit.size]
    return min(starts) if starts else None


def zone_spans(d, zones):
    """(from s, to s, label) of the obstacles in time: when the position reference is on each of them."""
    spans = []
    for z0, z1, label in zones:
        inside = d['time_s'][(d['s_ref_m'] >= z0) & (d['s_ref_m'] <= z1)]
        if inside.size:
            spans.append((float(inside.min()), float(inside.max()), label))
    return spans


def _label(ax, text):
    """What a panel shows, written above it: stacked y labels run into each other at this type size."""
    ax.set_title(text, loc='left', fontsize=18, pad=6)


def _finish(fig, axes, title, out_dir, name, xlabel=TIME_LABEL):
    for ax in axes:
        ax.margins(x=0.01)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle(title, x=0.012, y=0.985, ha='left', fontsize=25, fontweight='bold')
    # the legend has its own row under the title: inside the axes it would cover the curves
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(0.005, 0.93), ncol=5,
               frameon=False, handlelength=1.6, columnspacing=1.6)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def _mark_time(axes, logs, zones):
    """Disturbance onset and obstacles on time axes."""
    t0, kind = onset(logs)
    spans = zone_spans(next(iter(logs.values())), zones)
    for i, ax in enumerate(axes):
        if t0 is not None:
            ax.axvline(t0, color=INK, lw=1.6, ls='--', label=kind if i == 0 else None)
        for k, (a, b, label) in enumerate(spans):
            ax.axvspan(a, b, color=ZONE, alpha=0.16, lw=0, label=label if i == 0 and k == 0 else None)


# ---------------------------------------------------------------------------- figures
def fig_errors(logs, title, out_dir, zones=()):
    fig, ax = plt.subplots(2, 1, sharex=True)
    for law, d in logs.items():
        m = late(d)
        ax[0].plot(d['time_s'][m], d['theta_err_deg'][m], color=COLORS[law], label=LABELS[law])
        ax[1].plot(d['time_s'][m], 100.0 * d['s_err_m'][m], color=COLORS[law])
    for a in ax:
        a.axhline(0.0, color=MUTED, lw=1.0)
    _mark_time(ax, logs, zones)
    _label(ax[0], 'errore di inclinazione [deg]')
    _label(ax[1], 'errore di posizione [cm]')
    return _finish(fig, ax, f'{title}: errori di inseguimento', out_dir, 'errori.png')


def fig_torques(logs, title, out_dir, zones=()):
    fig, ax = plt.subplots(2, 1, sharex=True)
    for law, d in logs.items():
        m = late(d)
        left, right = d['wheel_L_torque_cmd'][m], d['wheel_R_torque_cmd'][m]
        ax[0].plot(d['time_s'][m], 0.5 * (right + left), color=COLORS[law], label=LABELS[law])
        ax[1].plot(d['time_s'][m], 0.5 * (right - left), color=COLORS[law])
    for a in ax:
        a.axhline(0.0, color=MUTED, lw=1.0)
    _mark_time(ax, logs, zones)
    _label(ax[0], 'coppia comune [Nm]')
    _label(ax[1], 'coppia differenziale [Nm]')
    return _finish(fig, ax, f'{title}: coppie alle ruote', out_dir, 'coppie.png')


def fig_start(logs, title, out_dir):
    """Speeds and pitch from one second before the reference speed leaves zero."""
    t0 = motion_start(logs)
    if t0 is None:
        return None
    fig, ax = plt.subplots(2, 1, sharex=True)
    for k, (law, d) in enumerate(logs.items()):
        m = (d['time_s'] >= t0 - 1.0) & (d['time_s'] <= t0 + 2.5)
        if k == 0:
            ax[0].plot(d['time_s'][m], d['xdot_ref_ms'][m], color=INK, ls='--', lw=2.0, label='riferimento')
        ax[0].plot(d['time_s'][m], d['xdot_ms'][m], color=COLORS[law], label=LABELS[law])
        ax[1].plot(d['time_s'][m], d['theta_deg'][m], color=COLORS[law])
    for a in ax:
        a.axvline(t0, color=MUTED, lw=1.4, ls=':')
    ax[1].axhline(0.0, color=MUTED, lw=1.0)
    _label(ax[0], 'velocità [m/s]')
    _label(ax[1], 'inclinazione [deg]')
    return _finish(fig, ax, f'{title}: partenza del moto', out_dir, 'avvio.png')


def fig_disturbance(logs, title, out_dir):
    """Pitch from just before the disturbance, and the (pitch, pitch rate) portrait of the same window."""
    t0, kind = onset(logs)
    if t0 is None:
        return None
    span = 10.0 if kind == 'gradino' else 5.0          # a held force: show the new equilibrium too
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1.35, 1.0]})
    for law, d in logs.items():
        m = (d['time_s'] >= t0 - 0.5) & (d['time_s'] <= t0 + span)
        ax[0].plot(d['time_s'][m], d['theta_deg'][m], color=COLORS[law], label=LABELS[law])
        ax[1].plot(d['theta_deg'][m], d['theta_dot_degs'][m], color=COLORS[law], lw=2.0)
    ax[0].axvline(t0, color=INK, lw=1.6, ls='--', label=kind)
    ax[0].axhline(0.0, color=MUTED, lw=1.0)
    ax[1].axhline(0.0, color=MUTED, lw=1.0)
    ax[1].axvline(0.0, color=MUTED, lw=1.0)
    _label(ax[0], 'inclinazione [deg]')
    ax[0].set_xlabel(TIME_LABEL)
    _label(ax[1], 'velocità di inclinazione [deg/s]')
    return _finish(fig, ax, f'{title}: risposta al disturbo', out_dir, 'disturbo.png', 'inclinazione [deg]')


def fig_terrain(logs, title, out_dir, zones):
    """Pitch, roll and change of CoM height against the distance travelled, obstacles shaded."""
    if not zones:
        return None
    # the obstacles and what follows them: the start and the braking of the trapezoid stay out of the picture
    lo, hi = zones[0][0] - TERRAIN_BEFORE_M, zones[-1][1] + TERRAIN_AFTER_M
    fig, ax = plt.subplots(3, 1, sharex=True)
    for law, d in logs.items():
        m = late(d) & (d['x_m'] >= lo) & (d['x_m'] <= hi)
        x, z = d['x_m'][m], d['com_z_m'][m]
        before = z[x < zones[0][0]]
        z0 = np.nanmedian(before) if np.isfinite(before).any() else np.nanmedian(z)
        ax[0].plot(x, d['theta_deg'][m], color=COLORS[law], label=LABELS[law])
        ax[1].plot(x, d['roll_deg'][m], color=COLORS[law])
        ax[2].plot(x, 1e3 * (z - z0), color=COLORS[law])
    for i, a in enumerate(ax):
        a.axhline(0.0, color=MUTED, lw=1.0)
        for k, (z0, z1, label) in enumerate(zones):
            a.axvspan(z0, z1, color=ZONE, alpha=0.16, lw=0, label=label if i == 0 and k == 0 else None)
    _label(ax[0], 'inclinazione [deg]')
    _label(ax[1], 'rollio [deg]')
    _label(ax[2], 'altezza del baricentro [mm]')
    return _finish(fig, ax, f'{title}: lungo il percorso', out_dir, 'terreno.png', 'distanza percorsa [m]')


def fig_path(logs, title, out_dir):
    """Ground-truth path of the torso seen from above, from the point of release."""
    fig, ax = plt.subplots()
    for law, d in logs.items():
        x, y = d['base_x_m'], d['base_y_m']
        if not np.isfinite(x).any():
            plt.close(fig)
            return None
        ax.plot(x - x[0], y - y[0], color=COLORS[law], label=LABELS[law])
    ax.plot(0.0, 0.0, 'o', color=INK, ms=10, label='partenza')
    ax.set_aspect('equal', 'datalim')
    _label(ax, 'spostamento laterale [m]')
    return _finish(fig, [ax], f'{title}: percorso visto dall\'alto', out_dir, 'percorso.png', 'avanzamento [m]')


def fig_lateral_zmp(run_dir, logs, title, out_dir):
    """Lateral ZMP as a share of the half track and load of the lighter wheel, from plots/zmp_<law>.csv."""
    series = {}
    for law, d in logs.items():
        path = os.path.join(run_dir, 'plots', f'zmp_{law}.csv')
        if not os.path.exists(path):
            return None
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        col = {k: np.array([float(r[k]) for r in rows]) for k in ('t_s', 'zmp_y_rel', 'fn_left_N', 'fn_right_N')}
        t = col['t_s'] - np.nanmedian(d['odom_stamp_s'] - d['time_s'])      # simulation stamps -> time since release
        series[law] = (t, col, t >= RELEASE_SKIP_S)
    fig, ax = plt.subplots(2, 1, sharex=True)
    for law, (t, col, m) in series.items():
        ax[0].plot(t[m], 100.0 * col['zmp_y_rel'][m], color=COLORS[law], label=LABELS[law])
        ax[1].plot(t[m], np.minimum(col['fn_left_N'], col['fn_right_N'])[m], color=COLORS[law])
    ax[0].axhline(0.0, color=MUTED, lw=1.0)
    _label(ax[0], 'ZMP laterale [% di mezza carreggiata]')
    _label(ax[1], 'carico della ruota più scarica [N]')
    ax[1].set_ylim(bottom=0.0)
    return _finish(fig, ax, f'{title}: stabilità laterale', out_dir, 'zmp_laterale.png')


def fig_height(logs, title, out_dir):
    """Change of CoM height from the standing value and pitch."""
    fig, ax = plt.subplots(2, 1, sharex=True)
    for law, d in logs.items():
        m = late(d)
        t, z = d['time_s'][m], d['com_z_m'][m]
        z0 = np.nanmedian(z[t <= t[0] + 0.3]) if t.size else 0.0
        ax[0].plot(t, 1e3 * (z - z0), color=COLORS[law], label=LABELS[law])
        ax[1].plot(t, d['theta_deg'][m], color=COLORS[law])
    for a in ax:
        a.axhline(0.0, color=MUTED, lw=1.0)
    _label(ax[0], 'altezza del baricentro [mm]')
    _label(ax[1], 'inclinazione [deg]')
    return _finish(fig, ax, f'{title}: altezza e equilibrio', out_dir, 'quota.png')


def generate_slides(run_dir):
    """The slide figures of a scenario in <run_dir>/slide/; returns their paths ([] without logs)."""
    logs = load(run_dir)
    if not logs:
        print(f'nessun CSV valido in {run_dir}')
        return []
    name = scenario_name(run_dir)
    sc = SCENARIOS.get(name)
    title, zones = (sc.title, sc.zones) if sc else (name, ())
    args = ' '.join(scenario_args(run_dir))
    out_dir = os.path.join(run_dir, SLIDE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(STYLE):
        saved = [fig_errors(logs, title, out_dir, zones), fig_torques(logs, title, out_dir, zones),
                 fig_start(logs, title, out_dir), fig_disturbance(logs, title, out_dir),
                 fig_terrain(logs, title, out_dir, zones)]
        if 'planar_enable:=true' in args:
            saved += [fig_path(logs, title, out_dir), fig_lateral_zmp(run_dir, logs, title, out_dir)]
        if 'height_enable:=true' in args or 'jump_enable:=true' in args:
            saved.append(fig_height(logs, title, out_dir))
    saved = [p for p in saved if p]
    print(f'{len(saved)} figure per le slide in {out_dir}')
    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', help='scenario directory (CSVs) or suite directory (one sub-directory per scenario)')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])
    if not os.path.isdir(args.path):
        p.error(f'directory non trovata: {args.path}')
    dirs = scenario_dirs(args.path) if is_suite(args.path) else [args.path]
    return 0 if [d for d in dirs if generate_slides(d)] else 1


if __name__ == '__main__':
    sys.exit(main())
