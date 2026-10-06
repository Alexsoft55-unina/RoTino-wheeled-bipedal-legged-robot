"""Comparison plots PID vs MPC of one scenario: every figure overlays the two laws (PID red, MPC blue)."""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from rotino_benchmark.common import COLORS, LABELS, LAWS, find_csv, norm_series  # noqa: E402

DPI = 130
NAN = float('nan')

COLUMNS = {
    't': 'time_s', 'th': 'theta_deg', 'th_d': 'theta_dot_degs', 'x': 'x_m', 'v': 'xdot_ms', 'z': 'com_z_m',
    'th_ref': 'theta_ref_deg', 'th_err': 'theta_err_deg', 's_ref': 's_ref_m', 's_err': 's_err_m',
    'v_ref': 'xdot_ref_ms', 'v_err': 'xdot_err_ms', 'push_f': 'push_force_N', 'step_f': 'step_force_N',
    'tau_l': 'wheel_L_torque_cmd', 'tau_r': 'wheel_R_torque_cmd',
    'hip': 'hip_L_pos', 'knee': 'knee_L_pos', 'hip_u': 'hip_L_torque_cmd', 'knee_u': 'knee_L_torque_cmd',
    'hip_u_r': 'hip_R_torque_cmd', 'knee_u_r': 'knee_R_torque_cmd',
    'hip_v': 'hip_L_vel', 'knee_v': 'knee_L_vel', 'wheel_v': 'wheel_L_vel',
    'roll': 'roll_deg', 'yaw': 'yaw_deg', 'ax': 'imu_accel_x', 'ay': 'imu_accel_y', 'az': 'imu_accel_z',
    'ds': 'delta_s_m',
}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def load_run_data(run_dir):
    """{law: {signal: list}} for the laws with a CSV in run_dir."""
    data = {}
    for law in LAWS:
        path = find_csv(run_dir, law)
        if not path:
            continue
        with open(path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if not math.isnan(_float(r.get('time_s')))]
        if not rows:
            continue
        if 'delta_s_m' not in rows[0]:
            for r in rows:
                r['delta_s_m'] = r.get('smc_s1')
        d = {k: [_float(r.get(col)) for r in rows] for k, col in COLUMNS.items()}
        d['u'] = [0.5 * (a + b) for a, b in zip(d['tau_l'], d['tau_r'])]
        d['tau_w_norm'] = norm_series(d['tau_l'], d['tau_r'])
        d['tau_norm'] = norm_series(d['tau_l'], d['tau_r'], d['hip_u'], d['hip_u_r'], d['knee_u'], d['knee_u_r'])
        data[law] = d
    return data


def _laws(data):
    return [law for law in LAWS if law in data]


def _save(fig, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _style(ax, ylabel, title=None, legend=True):
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc='best')


def push_time(data):
    """Onset of the disturbance (impulsive push or step force), or None."""
    return disturbance_onset(data)[0]


def disturbance_onset(data):
    """(time, 'spinta' | 'gradino') of the first push or step force above 1 N, or (None, None)."""
    for d in data.values():
        for ti, fp, fs in zip(d['t'], d['push_f'], d['step_f']):
            if fp > 1.0:
                return ti, 'spinta'
            if fs > 1.0:
                return ti, 'gradino'
    return None, None


def plot_tracking(data, out_dir):
    """References (dashed) and actual signals, left; tracking errors, right."""
    fig, ax = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    rows = [('th', 'th_ref', 'th_err', 'beccheggio [deg]', r'$e_\theta$ [deg]'),
            ('x', 's_ref', 's_err', 'posizione [m]', r'$e_x$ [m]'),
            ('v', 'v_ref', 'v_err', 'velocita [m/s]', r'$e_v$ [m/s]')]
    for i, (sig, sig_ref, err, lab, lab_err) in enumerate(rows):
        for law in _laws(data):
            d = data[law]
            ax[i, 0].plot(d['t'], d[sig], color=COLORS[law], lw=1.1, label=LABELS[law])
            ax[i, 0].plot(d['t'], d[sig_ref], color=COLORS[law], lw=0.9, ls='--', alpha=0.7,
                          label=f'rif. {LABELS[law]}')
            ax[i, 1].plot(d['t'], d[err], color=COLORS[law], lw=1.1, label=LABELS[law])
        ax[i, 1].axhline(0, color='gray', lw=0.8, ls=':')
        _style(ax[i, 0], lab, 'Riferimento (tratteggio) e stato' if i == 0 else None)
        _style(ax[i, 1], lab_err, 'Errori di inseguimento' if i == 0 else None)
    for law in _laws(data):
        d = data[law]
        ax[3, 0].plot(d['t'], d['z'], color=COLORS[law], lw=1.1, label=LABELS[law])
        ax[3, 1].plot(d['t'], [1e3 * v for v in d['ds']], color=COLORS[law], lw=1.1, label=LABELS[law])
    _style(ax[3, 0], 'altezza CoM [m]')
    _style(ax[3, 1], 'offset CoM desiderato [mm]')
    ax[3, 0].set_xlabel('tempo dal rilascio [s]')
    ax[3, 1].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'inseguimento.png')


def _rms(xs):
    xs = [v for v in xs if not math.isnan(v)]
    return math.sqrt(sum(v * v for v in xs) / len(xs)) if xs else NAN


def zone_times(d, zones):
    """(from s, to s, label) of the obstacles: when the position reference is on each of them."""
    spans = []
    for z0, z1, lab in zones:
        inside = [t for t, s in zip(d['t'], d['s_ref']) if z0 <= s <= z1]
        if inside:
            spans.append((min(inside), max(inside), lab))
    return spans


def plot_velocity_torque(data, out_dir, zones=()):
    """Velocity error and norm of the actuation torques at every instant: six joints, and the wheels alone."""
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    rows = [('v_err', 'm/s'), ('tau_norm', 'Nm'), ('tau_w_norm', 'Nm')]
    for law in _laws(data):
        d = data[law]
        for a, (sig, unit) in zip(ax, rows):
            a.plot(d['t'], d[sig], color=COLORS[law], lw=1.0, alpha=0.9,
                   label=f'{LABELS[law]}: rms {_rms(d[sig]):.3f} {unit}')
    ax[0].axhline(0, color='gray', lw=0.8, ls=':')
    if zones and data:
        spans = zone_times(data[_laws(data)[0]], zones)
        for a in ax:
            for k, (t0, t1, lab) in enumerate(spans):
                a.axvspan(t0, t1, color='#b9770e', alpha=0.15, lw=0, label=lab if a is ax[0] and k == 0 else None)
    _style(ax[0], r'$e_v = \dot s - \dot s_{rif}$ [m/s]', 'Errore di velocita e norma delle coppie di attuazione')
    _style(ax[1], r'$\|\tau\|$ ruote, anche, ginocchia [Nm]')
    _style(ax[2], r'$\|\tau\|$ ruote [Nm]')
    ax[2].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'velocita_coppie.png')


def write_velocity_torque_csv(data, out_dir):
    """velocita_coppie_<law>.csv: the samples of velocita_coppie.png, one row per control step."""
    cols = [('time_s', 't'), ('x_m', 'x'), ('xdot_ref_ms', 'v_ref'), ('xdot_ms', 'v'), ('xdot_err_ms', 'v_err'),
            ('tau_norm_Nm', 'tau_norm'), ('wheel_tau_norm_Nm', 'tau_w_norm')]
    paths = []
    for law in _laws(data):
        path = os.path.join(out_dir, f'velocita_coppie_{law}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([name for name, _ in cols])
            w.writerows(zip(*(data[law][k] for _, k in cols)))
        paths.append(path)
    return paths


def plot_disturbance(data, out_dir):
    """Pitch and wheel torque around the largest disturbance, and the (theta, theta_dot) portrait."""
    t0, kind = disturbance_onset(data)
    if t0 is None:
        t0 = max(((abs(th), t) for d in data.values() for t, th in zip(d['t'], d['th']) if not math.isnan(th)),
                 default=(0.0, 4.0))[1] - 1.0
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax_th, ax_tau, ax_ph = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for law in _laws(data):
        d = data[law]
        span = 12.0 if kind == 'gradino' else 6.0
        idx = [i for i, t in enumerate(d['t']) if t0 - 1.0 <= t <= t0 + span] or list(range(len(d['t'])))
        sub = lambda k: [d[k][i] for i in idx]  # noqa: E731
        ax_th.plot(sub('t'), sub('th'), color=COLORS[law], lw=1.4, label=LABELS[law])
        ax_tau.plot(sub('t'), sub('u'), color=COLORS[law], lw=1.2, label=LABELS[law])
        ax_ph.plot(sub('th'), sub('th_d'), color=COLORS[law], lw=1.2, alpha=0.85, label=LABELS[law])
        ax_ph.plot(d['th'][idx[0]], d['th_d'][idx[0]], 'o', color=COLORS[law], mec='k')
    if kind is not None:
        for a in (ax_th, ax_tau):
            a.axvline(t0, color='#e74c3c', ls='--', lw=1.1, label=kind)
    _style(ax_th, 'beccheggio [deg]', 'Risposta al disturbo')
    _style(ax_tau, 'coppia ruote media [Nm]')
    ax_tau.set_xlabel('tempo dal rilascio [s]')
    ax_ph.axhline(0, color='gray', lw=0.8, ls=':')
    ax_ph.axvline(0, color='gray', lw=0.8, ls=':')
    ax_ph.set_xlabel(r'$\theta$ [deg]')
    _style(ax_ph, r'$\dot\theta$ [deg/s]', r'Ritratto di fase $(\theta, \dot\theta)$ (pallino = inizio finestra)')
    return _save(fig, out_dir, 'disturbo_e_fase.png')


def plot_wheels(data, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for law in _laws(data):
        d = data[law]
        ax[0].plot(d['t'], d['u'], color=COLORS[law], lw=1.0, label=f'{LABELS[law]} media')
        ax[1].plot(d['t'], [0.5 * (r - l) for l, r in zip(d['tau_l'], d['tau_r'])], color=COLORS[law], lw=1.0,
                   label=f'{LABELS[law]} differenziale')
    _style(ax[0], 'coppia comune [Nm]', 'Coppie alle ruote')
    _style(ax[1], 'coppia differenziale [Nm]')
    ax[1].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'coppie_ruote.png')


def plot_legs(data, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for law in _laws(data):
        d = data[law]
        ax[0].plot(d['t'], d['hip'], color=COLORS[law], lw=1.1, label=f'{LABELS[law]} anca')
        ax[0].plot(d['t'], d['knee'], color=COLORS[law], lw=1.1, ls='--', label=f'{LABELS[law]} ginocchio')
        ax[1].plot(d['t'], d['hip_u'], color=COLORS[law], lw=1.0, label=f'{LABELS[law]} anca')
        ax[1].plot(d['t'], d['knee_u'], color=COLORS[law], lw=1.0, ls='--', label=f'{LABELS[law]} ginocchio')
    _style(ax[0], 'angolo gamba sinistra [rad]', 'Gambe')
    _style(ax[1], 'coppia gamba sinistra [Nm]')
    ax[1].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'gambe.png')


def plot_attitude(data, out_dir):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for law in _laws(data):
        d = data[law]
        ax[0].plot(d['t'], d['roll'], color=COLORS[law], lw=1.1, label=LABELS[law])
        ax[1].plot(d['t'], d['yaw'], color=COLORS[law], lw=1.1, label=LABELS[law])
    _style(ax[0], 'rollio [deg]', 'Assetto del torso')
    _style(ax[1], 'imbardata [deg]')
    ax[1].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'assetto.png')


def plot_power(data, out_dir):
    """Mechanical power sum |tau omega| of wheels and legs (both sides) and its integral."""
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for law in _laws(data):
        d = data[law]
        power, energy = [], [0.0]
        for i in range(len(d['t'])):
            p = 2.0 * (abs(d['u'][i] * d['wheel_v'][i]) + abs(d['hip_u'][i] * d['hip_v'][i])
                       + abs(d['knee_u'][i] * d['knee_v'][i]))
            power.append(0.0 if math.isnan(p) else p)
            if i:
                energy.append(energy[-1] + power[-1] * (d['t'][i] - d['t'][i - 1]))
        ax[0].plot(d['t'], power, color=COLORS[law], lw=0.9, alpha=0.85, label=LABELS[law])
        ax[1].plot(d['t'], energy, color=COLORS[law], lw=1.6, label=f'{LABELS[law]}: {energy[-1]:.1f} J')
    _style(ax[0], 'potenza [W]', 'Potenza meccanica ed energia')
    _style(ax[1], 'energia [J]')
    ax[1].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'potenza_energia.png')


def plot_horizontal_phase(data, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for law in _laws(data):
        d = data[law]
        ax.plot(d['s_err'], d['v_err'], color=COLORS[law], lw=1.2, alpha=0.85, label=LABELS[law])
        ax.plot(d['s_err'][0], d['v_err'][0], 's', color=COLORS[law], mec='k')
        ax.plot(d['s_err'][-1], d['v_err'][-1], '*', ms=11, color=COLORS[law], mec='k')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$e_x$ [m]')
    _style(ax, r'$e_v$ [m/s]', 'Errore di traslazione (quadrato = inizio, stella = fine)')
    return _save(fig, out_dir, 'fase_orizzontale.png')


def plot_imu(data, out_dir):
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for law in _laws(data):
        d = data[law]
        for a, k in zip(ax, ('ax', 'ay', 'az')):
            a.plot(d['t'], d[k], color=COLORS[law], lw=0.8, alpha=0.85, label=LABELS[law])
    for a, lab in zip(ax, ('avanti', 'laterale', 'verticale')):
        _style(a, f'acc. {lab} [m/s$^2$]', 'Accelerazioni IMU' if lab == 'avanti' else None)
    ax[2].set_xlabel('tempo dal rilascio [s]')
    return _save(fig, out_dir, 'imu.png')


def plot_terrain(data, out_dir, zones, title):
    """Pitch, CoM height, roll and wheel torque against the distance travelled, obstacles shaded."""
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for law in _laws(data):
        d = data[law]
        s = d['x']
        ax[0].plot(s, d['th'], color=COLORS[law], lw=1.1, label=LABELS[law])
        z0 = sorted(v for v in d['z'][:250] if not math.isnan(v))
        z0 = z0[len(z0) // 2] if z0 else 0.0
        ax[1].plot(s, [1e3 * (v - z0) for v in d['z']], color=COLORS[law], lw=1.1, label=LABELS[law])
        ax[2].plot(s, d['roll'], color=COLORS[law], lw=1.1, label=LABELS[law])
        ax[3].plot(s, d['tau_l'], color=COLORS[law], lw=0.9, label=f'{LABELS[law]} sinistra')
        ax[3].plot(s, d['tau_r'], color=COLORS[law], lw=0.9, ls='--', label=f'{LABELS[law]} destra')
    for a in ax:
        for k, (z0, z1, lab) in enumerate(zones):
            a.axvspan(z0, z1, color='#b9770e', alpha=0.15, lw=0, label=lab if a is ax[0] and k == 0 else None)
    _style(ax[0], 'beccheggio [deg]', f'{title}: segnali lungo il percorso (zone = ostacoli)')
    _style(ax[1], 'variazione altezza CoM [mm]')
    _style(ax[2], 'rollio [deg]')
    _style(ax[3], 'coppia ruote [Nm]')
    ax[3].set_xlabel('distanza percorsa dalla partenza [m]')
    return _save(fig, out_dir, 'terreno.png')


def generate_plots(run_dir, out_dir=None):
    """All figures of a scenario; returns the zmp_analysis results ({} if unavailable), None without data."""
    data = load_run_data(run_dir)
    if not data:
        print(f'nessun CSV valido in {run_dir}')
        return None
    out_dir = out_dir or os.path.join(run_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    saved = [f(data, out_dir) for f in (plot_tracking, plot_disturbance, plot_wheels, plot_legs, plot_attitude,
                                        plot_power, plot_horizontal_phase, plot_imu)]
    from rotino_benchmark.compare import scenario_name
    from rotino_benchmark.scenarios import SCENARIOS
    sc = SCENARIOS.get(scenario_name(run_dir))
    saved.append(plot_velocity_torque(data, out_dir, sc.zones if sc else ()))
    write_velocity_torque_csv(data, out_dir)
    if sc and sc.zones:
        saved.append(plot_terrain(data, out_dir, sc.zones, sc.title))
    results = {}
    try:
        from rotino_benchmark.zmp_analysis import analyse_run, generate_zmp_plots, write_zmp_csv
        results = analyse_run(run_dir)
        for law, o in results.items():
            write_zmp_csv(o, os.path.join(out_dir, f'zmp_{law}.csv'))
        saved += generate_zmp_plots(run_dir, out_dir, results)
    except Exception as exc:
        print(f'grafici ZMP non generati: {exc}')
    print(f'{len(saved)} grafici in {out_dir}')
    return results


def summary_figure(results, path):
    """One panel per scenario: key metrics of PID and MPC, bars normalised to the larger of the two."""
    from rotino_benchmark.compare import METRIC, better, scenario_name
    from rotino_benchmark.scenarios import SCENARIOS
    n = len(results)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.1 * rows), squeeze=False)
    for ax, (run_dir, found) in zip(axes.flat, results):
        name = scenario_name(run_dir)
        sc = SCENARIOS.get(name)
        keys = [k for k in (sc.key if sc else ()) if k in METRIC]
        labels = []
        for j, key in enumerate(keys):
            vals = {law: found.get(law, {}).get(key, NAN) for law in LAWS}
            scale = max((abs(v) for v in vals.values() if not math.isnan(v)), default=0.0) or 1.0
            for k, law in enumerate(LAWS):
                v = vals[law]
                if math.isnan(v):
                    continue
                y = j + (0.2 if law == 'pid' else -0.2)
                ax.barh(y, abs(v) / scale, height=0.38, color=COLORS[law], alpha=0.85,
                        label=LABELS[law] if j == 0 else None)
                ax.text(abs(v) / scale + 0.02, y, METRIC[key][2].format(v), va='center', fontsize=7)
            b = better(key, vals)
            tag = {'pid': '  (meglio PID)', 'mpc': '  (meglio MPC)', '=': '  (pari)'}.get(b, '')
            arrow = {-1: ' [min]', 1: ' [max]', 0: ''}[METRIC[key][3]]
            labels.append(METRIC[key][1] + arrow + tag)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.35)
        ax.set_xticks([])
        ax.set_title(sc.title if sc else name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('PID (rosso) e MPC (blu): metriche chiave, barre normalizzate al valore maggiore', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help='scenario directory (CSVs of PID and/or MPC)')
    p.add_argument('--out', default=None, help='output directory for the images (default <run_dir>/plots)')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])
    if not os.path.isdir(args.run_dir):
        p.error(f'directory non trovata: {args.run_dir}')
    return 0 if generate_plots(args.run_dir, args.out) is not None else 1


if __name__ == '__main__':
    sys.exit(main())
