"""PID versus MPC: metrics of one scenario, or the summary of a whole suite.

    ros2 run rotino_benchmark compare -- <scenario_dir>     # table PID | MPC | diff | better, confronto.md
    ros2 run rotino_benchmark compare -- <suite_dir>        # every scenario + riepilogo.md / .csv / .png

Classic metrics use only the columns both controllers publish (/rotino/debug, /rotino/wbr_state and the
commanded torques), so the two laws are scored on exactly the same signals; the ZMP metrics come from the
multibody reconstruction (zmp_analysis). Results are cached in <scenario_dir>/metriche.json.
"""

import argparse
import csv
import json
import math
import os
import sys

from rotino_benchmark.common import (LABELS, LAWS, TORQUE_COLUMNS, WHEEL_TORQUE_COLUMNS, find_csv, is_suite,
                                     norm_series, scenario_dirs)
from rotino_benchmark.scenarios import SCENARIOS

SETTLE_BAND_DEG = 0.5
# Peak, recovery and recovery energy ignore the release from the anchor: the CoM hangs 1.1 cm ahead of the axle
# (~4.7 deg) and every law has the same transient, which otherwise was "the peak" of every quiet scenario.
# Scenario motions start at 2 s, pushes and jumps at 4 s.
RELEASE_SKIP_S = 1.5
STEADY_TAIL_S = 3.0      # steady state after a disturbance: mean of the last 3 s of the run
ZONE_BEFORE_M = 0.05     # platform metrics: from just before each obstacle ...
ZONE_AFTER_M = 0.6       # ... to 0.6 m after it, so that the response is included but not the start/stop
DIST_THRESHOLD_N = 1.0   # an impulsive push or a step force above this marks the disturbance onset
TIE_REL = 0.05            # relative difference under which the two laws are considered equal
NAN = float('nan')

# (key, label, format, direction, tie): direction -1 lower is better, +1 higher is better, 0 descriptive
# only; below `tie` (absolute, in the unit of the metric) the two laws are equal whatever the ratio
METRICS = [
    ('duration_s', 'durata [s]', '{:.1f}', 0, 0.0),
    ('pitch_rms_deg', 'beccheggio rms a regime [deg]', '{:.3f}', -1, 0.05),
    ('pitch_peak_deg', 'beccheggio di picco [deg]', '{:.2f}', -1, 0.2),
    ('recovery_s', 'recupero dopo il picco [s]', '{:.2f}', -1, 0.1),
    ('pitch_err_rms_deg', 'errore di beccheggio rms [deg]', '{:.3f}', -1, 0.05),
    ('pos_err_rms_m', 'errore di posizione rms [m]', '{:.3f}', -1, 0.003),
    ('pos_err_max_m', 'errore di posizione max [m]', '{:.3f}', -1, 0.005),
    ('vel_err_rms_ms', 'errore di velocita rms [m/s]', '{:.3f}', -1, 0.005),
    ('speed_peak_ms', 'velocita di picco [m/s]', '{:.2f}', 0, 0.0),
    ('travel_m', 'spazio percorso [m]', '{:.2f}', 0, 0.0),
    ('height_rms_mm', 'oscillazione di quota rms [mm]', '{:.1f}', 0, 0.0),
    ('com_rise_mm', 'salita massima del CoM [mm]', '{:.0f}', 0, 0.0),
    ('dist_energy_J', 'energia di recupero (int tau^2 dt) [J]', '{:.2f}', -1, 0.01),
    ('settle_ss_s', 'assestamento dopo il disturbo [s]', '{:.2f}', -1, 0.1),
    ('pitch_ss_deg', 'beccheggio medio a regime [deg]', '{:+.2f}', 0, 0.0),
    ('pos_err_ss_m', 'errore di posizione a regime [m]', '{:.3f}', -1, 0.005),
    ('roll_peak_deg', 'rollio di picco [deg]', '{:.2f}', -1, 0.1),
    ('yaw_dev_max_deg', 'deviazione di rotta max [deg]', '{:.2f}', -1, 0.2),
    ('tau_peak_Nm', 'coppia ruota di picco [Nm]', '{:.2f}', -1, 0.05),
    ('zone_pitch_peak_deg', 'ostacoli: beccheggio di picco [deg]', '{:.2f}', -1, 0.2),
    ('zone_tau_peak_Nm', 'ostacoli: coppia ruota di picco [Nm]', '{:.2f}', -1, 0.05),
    ('zone_com_dev_mm', 'ostacoli: escursione verticale del CoM [mm]', '{:.1f}', -1, 1.0),
    ('wheel_tau_rms_Nm', 'coppia ruote rms [Nm]', '{:.3f}', -1, 0.01),
    ('tau_norm_rms_Nm', 'norma delle coppie rms, 6 giunti [Nm]', '{:.3f}', -1, 0.01),
    ('tau_norm_peak_Nm', 'norma delle coppie di picco, 6 giunti [Nm]', '{:.2f}', -1, 0.05),
    ('wheel_tau_norm_rms_Nm', 'norma delle coppie ruote rms [Nm]', '{:.3f}', -1, 0.01),
    ('chatter_Nm', 'chattering, |d tau| per campione [Nm]', '{:.4f}', -1, 0.002),
    ('airborne_pct', 'tempo senza contatto [%]', '{:.1f}', -1, 0.5),
    ('zmp_lat_ratio_pct', 'ZMP laterale max / (d/2) [%]', '{:.1f}', -1, 1.0),
    ('zmp_margin_min_mm', 'ZMP: margine laterale minimo [mm]', '{:.1f}', +1, 1.0),
    ('wheel_load_min_N', 'ZMP: carico minimo ruota [N]', '{:.1f}', +1, 0.5),
    ('zmp_long_rms_mm', 'ZMP: residuo longitudinale rms [mm]', '{:.2f}', 0, 0.0),
]
METRIC = {m[0]: m for m in METRICS}


# ---------------------------------------------------------------------------- metrics
def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in ('time_s', 'theta_deg', 'x_m', 'xdot_ms', 'wheel_u', 'com_z_m',
                'fn_total_N', 'loaded', *TORQUE_COLUMNS,
                'theta_err_deg', 's_err_m', 'xdot_err_ms', 'push_force_N', 'step_force_N', 'roll_deg',
                'yaw_deg'):
        col = []
        for r in rows:
            try:
                col.append(float(r.get(key, 'nan')))
            except (TypeError, ValueError):
                col.append(NAN)
        out[key] = col
    return out


def finite(xs):
    return [x for x in xs if not math.isnan(x)]


def _rms(xs):
    xs = finite(xs)
    return math.sqrt(sum(v * v for v in xs) / len(xs)) if xs else NAN


def metrics(d, zones=()):
    t = d['time_s']
    th = d['theta_deg']
    if not finite(t):
        return None
    dur = max(finite(t)) - min(finite(t))

    # steady state: last third of the run
    cut = min(finite(t)) + 2.0 * dur / 3.0
    tail = [v for ti, v in zip(t, th) if not math.isnan(ti) and ti >= cut and not math.isnan(v)]
    rms = math.sqrt(sum(v * v for v in tail) / len(tail)) if tail else NAN

    t0 = min(finite(t)) + RELEASE_SKIP_S if min(finite(t)) < RELEASE_SKIP_S else min(finite(t))
    fin = [(ti, v) for ti, v in zip(t, th) if not math.isnan(ti) and not math.isnan(v) and ti >= t0]
    peak = max((abs(v) for _, v in fin), default=NAN)

    # recovery: last instant the pitch was outside the band, measured from the peak onwards
    settle = NAN
    if fin:
        i_peak = max(range(len(fin)), key=lambda i: abs(fin[i][1]))
        after = fin[i_peak:]
        out_of_band = [ti for ti, v in after if abs(v) > SETTLE_BAND_DEG]
        if out_of_band:
            settle = max(out_of_band) - fin[i_peak][0]
        elif after:
            settle = 0.0

    taus = [0.5 * (a + b) for a, b in zip(d['wheel_L_torque_cmd'], d['wheel_R_torque_cmd'])
            if not (math.isnan(a) or math.isnan(b))]
    tau_rms = math.sqrt(sum(v * v for v in taus) / len(taus)) if taus else NAN
    # chattering proxy: mean |d(tau)| per sample over the steady tail
    tail_tau = taus[int(len(taus) * 0.66):] if taus else []
    chat = (sum(abs(b - a) for a, b in zip(tail_tau, tail_tau[1:])) / max(len(tail_tau) - 1, 1)
            if len(tail_tau) > 2 else NAN)

    # disturbance recovery energy: integral of tau^2 dt over the response to the pitch peak
    dist_energy = NAN
    if fin and taus:
        i_peak = max(range(len(fin)), key=lambda i: abs(fin[i][1]))
        t_peak = fin[i_peak][0]
        t_end = t_peak + (settle if not math.isnan(settle) and settle > 0 else 3.0)
        e_samples = [tau ** 2 for ti, tau in zip(t, taus) if not math.isnan(ti) and t_peak <= ti <= t_end]
        if e_samples:
            dist_energy = sum(e_samples) * 0.002

    # norm of the actuation torques at every sample: the six joints, and the wheels alone (the legs hold
    # the weight, about 2.6 Nm whatever the law); the peak ignores the release like the other peaks
    tau_norm = norm_series(*(d.get(c, [NAN] * len(t)) for c in TORQUE_COLUMNS))
    wheel_norm = norm_series(*(d[c] for c in WHEEL_TORQUE_COLUMNS))
    tau_norm_peak = max((v for ti, v in zip(t, tau_norm) if not math.isnan(v) and ti >= t0), default=NAN)

    ss = disturbance_metrics(d, t0)

    th_err = finite(d.get('theta_err_deg', []))
    pitch_err_rms = math.sqrt(sum(v * v for v in th_err) / len(th_err)) if th_err else rms

    loaded = finite(d['loaded'])
    airborne = 100.0 * sum(1 for v in loaded if v < 0.5) / len(loaded) if loaded else NAN

    x = finite(d['x_m'])
    v = finite(d['xdot_ms'])
    z = finite(d['com_z_m'])
    s_err = finite(d.get('s_err_m', []))
    z0 = sorted(z[:250])[len(z[:250]) // 2] if z else NAN          # standing height, first 0.5 s
    return {
        'duration_s': dur,
        'pitch_rms_deg': rms,
        'pitch_peak_deg': peak,
        'recovery_s': settle,
        'pitch_err_rms_deg': pitch_err_rms,
        'pos_err_rms_m': _rms(s_err),
        'pos_err_max_m': max((abs(e) for e in s_err), default=NAN),
        'vel_err_rms_ms': _rms(d.get('xdot_err_ms', [])),
        'dist_energy_J': dist_energy,
        'speed_peak_ms': max((abs(q) for q in v), default=NAN),
        'travel_m': (max(x) - min(x)) if x else NAN,
        'height_rms_mm': (1000.0 * math.sqrt(sum((q - (sum(z) / len(z))) ** 2 for q in z) / len(z))
                          if z else NAN),
        'com_rise_mm': 1000.0 * (max(z) - z0) if z else NAN,
        'wheel_tau_rms_Nm': tau_rms,
        'tau_norm_rms_Nm': _rms(tau_norm),
        'tau_norm_peak_Nm': tau_norm_peak,
        'wheel_tau_norm_rms_Nm': _rms(wheel_norm),
        'chatter_Nm': chat,
        'airborne_pct': airborne,
        **ss,
        **zone_metrics(d, zones),
    }


def zone_metrics(d, zones):
    """Pitch, wheel torque and vertical CoM excursion while crossing the obstacles of a platform scenario.

    The windows are in distance travelled (x_m), so the acceleration at the start and the braking at the end
    of the trapezoid stay out of them; the CoM excursion is measured from the height just before the first
    obstacle (the two laws publish different CoM heights: whole robot for the PID, upper body for the MPC).
    """
    out = {'zone_pitch_peak_deg': NAN, 'zone_tau_peak_Nm': NAN, 'zone_com_dev_mm': NAN}
    if not zones:
        return out
    x, n = d['x_m'], len(d['x_m'])
    windows = [(z0 - ZONE_BEFORE_M, z1 + ZONE_AFTER_M) for z0, z1, *_ in zones]
    inside = [i for i in range(n) if not math.isnan(x[i]) and any(a <= x[i] <= b for a, b in windows)]
    if not inside:
        return out
    first = windows[0][0]
    before = [d['com_z_m'][i] for i in range(n) if not math.isnan(x[i]) and first - 0.3 <= x[i] < first]
    out['zone_pitch_peak_deg'] = max((abs(d['theta_deg'][i]) for i in inside
                                      if not math.isnan(d['theta_deg'][i])), default=NAN)
    out['zone_tau_peak_Nm'] = max((abs(v) for i in inside
                                   for v in (d['wheel_L_torque_cmd'][i], d['wheel_R_torque_cmd'][i])
                                   if not math.isnan(v)), default=NAN)
    z_ref = _median(finite(before))
    if not math.isnan(z_ref):
        out['zone_com_dev_mm'] = 1000.0 * max((abs(d['com_z_m'][i] - z_ref) for i in inside
                                               if not math.isnan(d['com_z_m'][i])), default=NAN)
    return out


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else NAN


def disturbance_metrics(d, t_skip):
    """Steady state after a disturbance and the attitude/torque peaks of the uneven-ground runs.

    settle_ss_s: from the onset of the push or step force, time until the pitch stays within the band
    around its final value (a held force leaves the robot leaning: the band is not around 0);
    pitch_ss_deg / pos_err_ss_m: mean pitch and mean |position error| over the last STEADY_TAIL_S;
    roll_peak_deg / yaw_dev_max_deg: largest deviation from the attitude at the start of the log.
    """
    t = d['time_s']
    rows = [i for i, ti in enumerate(t) if not math.isnan(ti)]
    out = {k: NAN for k in ('settle_ss_s', 'pitch_ss_deg', 'pos_err_ss_m', 'roll_peak_deg',
                            'yaw_dev_max_deg', 'tau_peak_Nm')}
    if not rows:
        return out
    t_end = t[rows[-1]]
    tail = [i for i in rows if t[i] >= t_end - STEADY_TAIL_S]
    th_tail = finite([d['theta_deg'][i] for i in tail])
    if th_tail:
        out['pitch_ss_deg'] = sum(th_tail) / len(th_tail)
    err_tail = finite([d.get('s_err_m', [NAN] * len(t))[i] for i in tail])
    if err_tail:
        out['pos_err_ss_m'] = sum(abs(e) for e in err_tail) / len(err_tail)
    force = [max((v for v in (d.get('push_force_N', [NAN] * len(t))[i], d.get('step_force_N', [NAN] * len(t))[i])
                  if not math.isnan(v)), default=0.0) for i in range(len(t))]
    onset = next((t[i] for i in rows if force[i] > DIST_THRESHOLD_N), None)
    if onset is not None and th_tail:
        final = out['pitch_ss_deg']
        out_band = [t[i] for i in rows if t[i] >= onset and not math.isnan(d['theta_deg'][i])
                    and abs(d['theta_deg'][i] - final) > SETTLE_BAND_DEG]
        out['settle_ss_s'] = (max(out_band) - onset) if out_band else 0.0
    start = [i for i in rows if t[i] <= t[rows[0]] + 0.5]
    late = [i for i in rows if t[i] >= t_skip]
    for key, col, wrap in (('roll_peak_deg', 'roll_deg', False), ('yaw_dev_max_deg', 'yaw_deg', True)):
        sig = d.get(col, [])
        ref = _median(finite([sig[i] for i in start])) if sig else NAN
        if not math.isnan(ref):
            devs = [abs(_wrap_deg(sig[i] - ref)) if wrap else abs(sig[i] - ref)
                    for i in late if not math.isnan(sig[i])]
            out[key] = max(devs, default=NAN)
    taus = [abs(v) for i in late for v in (d['wheel_L_torque_cmd'][i], d['wheel_R_torque_cmd'][i])
            if not math.isnan(v)]
    out['tau_peak_Nm'] = max(taus, default=NAN)
    return out


def zmp_row_metrics(path, result=None):
    """ZMP metrics of one log (or of an already computed zmp_analysis result); {} if not available."""
    try:
        from rotino_benchmark import zmp_analysis
        if result is None:
            d = zmp_analysis.read_log(path)
            result = zmp_analysis.analyse(d) if d is not None else None
        return zmp_analysis.zmp_metrics(result) if result is not None else {}
    except Exception as exc:   # missing URDF/xacro must not break the classic table
        print(f'  ZMP non calcolato per {os.path.basename(path)}: {exc}')
        return {}


# ---------------------------------------------------------------------------- scenario
def _cache_path(run_dir):
    return os.path.join(run_dir, 'metriche.json')


def scenario_metrics(run_dir, zmp_results=None, recompute=False):
    """{law: metrics} of a scenario directory, cached in metriche.json (keyed by the CSV names)."""
    csvs = {law: find_csv(run_dir, law) for law in LAWS}
    csvs = {law: p for law, p in csvs.items() if p}
    cache = _cache_path(run_dir)
    if not recompute and zmp_results is None and os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        if data.get('csv') == {law: os.path.basename(p) for law, p in csvs.items()}:
            return {law: {k: (NAN if v is None else v) for k, v in m.items()} for law, m in data['metrics'].items()}
    sc = SCENARIOS.get(scenario_name(run_dir))
    found = {}
    for law, path in csvs.items():
        m = metrics(read_csv(path), sc.zones if sc else ())
        if m:
            m.update(zmp_row_metrics(path, (zmp_results or {}).get(law)))
            found[law] = m
    with open(cache, 'w') as f:
        json.dump({'csv': {law: os.path.basename(p) for law, p in csvs.items()},
                   'metrics': {law: {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                     for k, v in m.items()} for law, m in found.items()}}, f, indent=1)
    return found


def better(key, values):
    """'pid', 'mpc', '=' or '' (descriptive metric / missing value)."""
    direction, tie = METRIC[key][3], METRIC[key][4]
    a, b = values.get('pid', NAN), values.get('mpc', NAN)
    if direction == 0 or math.isnan(a) or math.isnan(b):
        return ''
    scale = max(abs(a), abs(b))
    if abs(a - b) <= tie or scale < 1e-9 or abs(a - b) / scale < TIE_REL:
        return '='
    return 'pid' if (a - b) * direction > 0 else 'mpc'


def _fmt(key, v):
    return 'n/d' if v is None or math.isnan(v) else METRIC[key][2].format(v)


def _diff(key, found):
    a, b = found.get('pid', {}).get(key, NAN), found.get('mpc', {}).get(key, NAN)
    if math.isnan(a) or math.isnan(b) or max(abs(a), abs(b)) < 1e-9 or abs(a - b) <= METRIC[key][4]:
        return ''                  # below the resolution of the metric a percentage means nothing
    return f'{100.0 * (a - b) / max(abs(b), 1e-9):+.0f} %' if abs(b) > 1e-9 else ''


def print_table(title, found, keys=None, name=None):
    laws = [law for law in LAWS if law in found]
    width = max(len(m[1]) for m in METRICS) + 2
    print(f'\n{title}\n')
    print('  ' + 'metrica'.ljust(width) + ''.join(LABELS[law].rjust(11) for law in laws)
          + 'PID-MPC'.rjust(10) + '  migliore')
    print('  ' + '-' * (width + 11 * len(laws) + 20))
    for key, label, *_ in METRICS:
        if keys and key not in keys:
            continue
        values = {law: found[law].get(key, NAN) for law in laws}
        mark = '*' if keys is None and key in _key_of(name or title) else ' '
        b = better(key, values)
        print(f' {mark}' + label.ljust(width) + ''.join(_fmt(key, values[law]).rjust(11) for law in laws)
              + _diff(key, found).rjust(10) + '  ' + {'pid': 'PID', 'mpc': 'MPC', '=': 'pari'}.get(b, ''))


def _key_of(name):
    sc = SCENARIOS.get(name)
    return sc.key if sc else ()


def write_scenario_report(run_dir, name, found):
    """confronto.md and confronto.csv next to the CSVs of the scenario."""
    sc = SCENARIOS.get(name)
    keys = sc.key if sc else ()
    with open(os.path.join(run_dir, 'confronto.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metrica', 'chiave', *[LABELS[law] for law in LAWS], 'migliore'])
        for key, label, *_ in METRICS:
            values = {law: found.get(law, {}).get(key, NAN) for law in LAWS}
            w.writerow([label, key, *[values[law] for law in LAWS], better(key, values)])
    lines = [f'# {sc.title if sc else name}', '']
    if sc:
        step = (f", gradino di {sc.disturbance['force']:+.1f} N a {sc.disturbance.get('start_time', 4.0):.0f} s"
                if sc.disturbance else '')
        lines += [sc.description, '', f'Argomenti: `{" ".join(sc.args) or "(nessuno)"}`, {sc.duration:.0f} s{step}.', '']
    lines += ['| metrica | PID | MPC | PID vs MPC | migliore |', '|---|---:|---:|---:|:---:|']
    for key, label, *_ in METRICS:
        values = {law: found.get(law, {}).get(key, NAN) for law in LAWS}
        b = better(key, values)
        lab = f'**{label}**' if key in keys else label
        lines.append(f'| {lab} | {_fmt(key, values["pid"])} | {_fmt(key, values["mpc"])} | {_diff(key, found)} | '
                     + {'pid': 'PID', 'mpc': 'MPC', '=': 'pari'}.get(b, '') + ' |')
    lines += ['', 'In grassetto le metriche chiave dello scenario. "PID vs MPC" e la differenza relativa '
              'rispetto all\'MPC; "pari" sotto il 5 % o sotto la risoluzione della metrica.', '', '## Grafici', '']
    plots = os.path.join(run_dir, 'plots')
    if os.path.isdir(plots):
        lines += [f'![{p}](plots/{p})' for p in sorted(os.listdir(plots)) if p.endswith('.png')]
    with open(os.path.join(run_dir, 'confronto.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------- suite
def scenario_name(run_dir):
    """Name recorded in scenario.json by campaign/suite, else the directory name."""
    try:
        with open(os.path.join(run_dir, 'scenario.json')) as f:
            name = json.load(f).get('name')
        if name:
            return name
    except (OSError, ValueError):
        pass
    return os.path.basename(os.path.normpath(run_dir))


def ordered(dirs):
    names = list(SCENARIOS)
    return sorted(dirs, key=lambda d: (names.index(scenario_name(d)) if scenario_name(d) in names else 99,
                                       scenario_name(d)))


def write_suite_report(suite_dir, results):
    """riepilogo.md / .csv: key metrics of every scenario and how often each law is better."""
    score = {'pid': 0, 'mpc': 0, '=': 0}
    lines = [f'# Confronto PID - MPC: {os.path.basename(os.path.normpath(suite_dir))}', '',
             'Stesso mondo, stesso URDF, stessa attuazione in coppia e stessi argomenti di scenario: cambia solo '
             'il controllore. Per ogni scenario le metriche chiave; il dettaglio e i grafici sono in '
             '`<scenario>/confronto.md`.', '']
    rows = []
    for run_dir, found in results:
        name = scenario_name(run_dir)
        sc = SCENARIOS.get(name)
        lines += [f'## [{sc.title if sc else name}]({name}/confronto.md)', '',
                  (sc.description if sc else ''), '',
                  '| metrica | PID | MPC | PID vs MPC | migliore |', '|---|---:|---:|---:|:---:|']
        for key in (sc.key if sc else [m[0] for m in METRICS if m[3]]):
            values = {law: found.get(law, {}).get(key, NAN) for law in LAWS}
            b = better(key, values)
            if b in score:
                score[b] += 1
            rows.append([name, key, *[values[law] for law in LAWS], b])
            lines.append(f'| {METRIC[key][1]} | {_fmt(key, values["pid"])} | {_fmt(key, values["mpc"])} | '
                         f'{_diff(key, found)} | ' + {'pid': 'PID', 'mpc': 'MPC', '=': 'pari'}.get(b, '') + ' |')
        lines.append('')
    total = sum(score.values())
    head = ['## Bilancio sulle metriche chiave', '',
            f'Su {total} metriche chiave: **PID migliore in {score["pid"]}**, **MPC migliore in {score["mpc"]}**, '
            f'pari in {score["="]} (differenza sotto il {TIE_REL * 100:.0f} % o sotto la risoluzione della metrica).', '',
            '![Riepilogo](riepilogo.png)', '']
    lines[4:4] = head                                  # after the title and the introduction
    with open(os.path.join(suite_dir, 'riepilogo.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(suite_dir, 'riepilogo.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'metrica', *[LABELS[law] for law in LAWS], 'migliore'])
        w.writerows(rows)
    return score


def analyse_suite(suite_dir, recompute=False, plots=True):
    results = []
    for run_dir in ordered(scenario_dirs(suite_dir)):
        found = analyse_scenario(run_dir, recompute=recompute, plots=plots)
        if found:
            results.append((run_dir, found))
    if not results:
        return None
    score = write_suite_report(suite_dir, results)
    if plots:
        from rotino_benchmark.plot import summary_figure
        summary_figure(results, os.path.join(suite_dir, 'riepilogo.png'))
    print(f'\nBilancio metriche chiave: PID {score["pid"]}, MPC {score["mpc"]}, pari {score["="]}')
    print(f'Riepilogo: {os.path.join(suite_dir, "riepilogo.md")}')
    return results


def analyse_scenario(run_dir, recompute=False, plots=True):
    """Metrics (+ ZMP), plots, confronto.md and the CSVs of dati/ (export) of one scenario directory; returns
    {law: metrics}."""
    name = scenario_name(run_dir)
    zmp_results = None
    need = recompute or not os.path.exists(_cache_path(run_dir))
    if plots and (need or not os.path.isdir(os.path.join(run_dir, 'plots'))):
        from rotino_benchmark.plot import generate_plots
        zmp_results = generate_plots(run_dir)
    found = scenario_metrics(run_dir, zmp_results or None, recompute=recompute and not zmp_results)
    if not found:
        print(f'nessun CSV in {run_dir}')
        return None
    write_scenario_report(run_dir, name, found)
    from rotino_benchmark.export import DATA_DIR, export_scenario
    if need or not os.path.isdir(os.path.join(run_dir, DATA_DIR)):
        export_scenario(run_dir)
    print_table(f'scenario: {name}  (* = metrica chiave)', found, name=name)
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', help='scenario directory (CSVs) or suite directory (one sub-directory per scenario)')
    p.add_argument('--ricalcola', action='store_true', help='ignore metriche.json and redo plots and ZMP analysis')
    p.add_argument('--no-plots', action='store_true', help='tables only')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])
    if not os.path.isdir(args.path):
        p.error(f'directory non trovata: {args.path}')
    if is_suite(args.path):
        return 0 if analyse_suite(args.path, args.ricalcola, not args.no_plots) else 1
    return 0 if analyse_scenario(args.path, args.ricalcola, not args.no_plots) else 1


if __name__ == '__main__':
    sys.exit(main())
