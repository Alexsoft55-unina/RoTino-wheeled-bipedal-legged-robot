"""Runs one scenario with the PID and the MPC, headless, and collects a CSV per law, then compares them.

Every run uses the same world, the same URDF and the same scenario arguments; only the controller
package changes. Runs are sequential and each one starts from a clean process table, because a
leftover node from a previous run keeps publishing on /rotino/* and silently corrupts the next
result (a stale dashboard publishing /rotino/cmd_vel put a controller into teleop once).

    ros2 run rotino_benchmark campaign -- spinta                        # a scenario of scenarios.py
    ros2 run rotino_benchmark campaign -- --scenario push_enable:=true push_impulse:=4.0 --duration 20
    ros2 run rotino_benchmark campaign -- --gradino 5.0 --gradino-t 4.0     # step force of 5 N at 4 s
    ros2 run rotino_benchmark campaign -- trapezio --topic-extra        # also the estimator error of the MPC
    ros2 run rotino_benchmark suite                                     # every scenario (suite.py)

--topic-extra makes the logger subscribe to one more topic, /rotino/estimation_error. The MPC then has one
more reader to serve and the logger one more message per control step, on threads that are already nearly
full: when in doubt, collect it in a separate run and keep the default run for the metrics.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from rotino_benchmark.common import LAWS, default_out
from rotino_benchmark.scenarios import SCENARIOS

# On Fortress the server runs as "ruby /usr/bin/ign gazebo ...": 'gz sim' alone missed a Gazebo left over
# from a manual launch, and the next controller attached to that old world (robot already fallen).
STALE_PATTERNS = ('rotino_pid/controller', 'rotino_mpc/controller',
                  'rotino_benchmark/logger', 'rotino_benchmark/disturbance', 'rotino_dashboard', 'gz sim', 'ign gazebo', 'parameter_bridge',
                  'robot_state_publisher', 'ros_gz_sim')
# lines of the simulation log that make a run invalid
INVALID_MARKS = ('Failed to configure controller', 'Live commands received')


def kill_stale():
    """Nothing from a previous run may still be publishing on /rotino/*."""
    for pat in STALE_PATTERNS:
        subprocess.run(['pkill', '-f', pat], capture_output=True)
    time.sleep(2.0)


def disturbance_cmd(disturbance):
    """Command line of the step-disturbance node, or None."""
    if not disturbance:
        return None
    return ['ros2', 'run', 'rotino_benchmark', 'disturbance', '--ros-args', '-p', 'use_sim_time:=true',
            '-p', f"force:={float(disturbance['force'])}",
            '-p', f"start_time:={float(disturbance.get('start_time', 4.0))}",
            '-p', f"duration:={float(disturbance.get('duration', 0.0))}"]


def run_one(law, scenario, duration, out_dir, log_dir, disturbance=None, extra_topics=False):
    print(f'\n=== {law.upper()} ===', flush=True)
    kill_stale()
    os.makedirs(out_dir, exist_ok=True)

    launch = ['ros2', 'launch', f'rotino_{law}', f'rotino_{law}.launch.py', 'gui:=false', *scenario]
    logger = ['ros2', 'run', 'rotino_benchmark', 'logger', '--ros-args',
              '-p', 'use_sim_time:=true', '-p', f'controller:={law}', '-p', f'output_dir:={out_dir}',
              '-p', f'extra_topics:={"true" if extra_topics else "false"}']

    with open(os.path.join(log_dir, f'{law}.log'), 'w') as sim_log:
        sim = subprocess.Popen(launch, stdout=sim_log, stderr=subprocess.STDOUT,
                               preexec_fn=os.setsid)
        time.sleep(2.0)                       # let Gazebo come up
        # Start the logger well before the release: its /joint_states subscription connects ~1.6 s after
        # /rotino/odom, and when the release came first the ZMP analysis (which needs the joints) lost up
        # to the first 4 s of the run. Rows are written only from /rotino/debug, i.e. after the release.
        log = subprocess.Popen(logger, stdout=sim_log, stderr=subprocess.STDOUT,
                               preexec_fn=os.setsid)
        procs = [log, sim]
        cmd = disturbance_cmd(disturbance)
        if cmd:                               # waits for /rotino/debug, i.e. for the release
            procs.insert(0, subprocess.Popen(cmd, stdout=sim_log, stderr=subprocess.STDOUT,
                                             preexec_fn=os.setsid))
        try:
            time.sleep(4.0 + duration)
        finally:
            for proc in procs:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
            time.sleep(3.0)
            for proc in procs:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    kill_stale()

    produced = sorted(f for f in os.listdir(out_dir) if f.startswith(f'rotino_{law}_'))
    if not produced:
        print(f'  NESSUN CSV per {law}: vedi {log_dir}/{law}.log', flush=True)
        return None
    with open(os.path.join(log_dir, f'{law}.log'), errors='replace') as f:
        bad = sorted({m for line in f for m in INVALID_MARKS if m in line})
    if bad:
        print(f'  ATTENZIONE, prova probabilmente non valida ({"; ".join(bad)}): vedi {log_dir}/{law}.log')
    print(f'  {produced[-1]}', flush=True)
    return os.path.join(out_dir, produced[-1])


def run_scenario(run_dir, laws, args, duration, name=None, disturbance=None, extra_topics=False):
    """Runs the laws one after the other in run_dir (CSVs) and run_dir/logs (simulation output);
    scenario.json records what was run, so that compare knows the scenario whatever the directory name."""
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'scenario.json'), 'w') as f:
        json.dump({'name': name, 'args': list(args), 'duration': duration, 'laws': list(laws),
                   'disturbance': disturbance, 'extra_topics': bool(extra_topics),
                   'date': f'{datetime.now():%Y-%m-%d %H:%M}'}, f, indent=1)
    return {law: run_one(law, args, duration, run_dir, log_dir, disturbance, extra_topics) for law in laws}


def parse_laws(p, text):
    laws = [c.strip() for c in text.split(',') if c.strip()]
    unknown = [c for c in laws if c not in LAWS]
    if unknown:
        p.error(f'controllori sconosciuti: {", ".join(unknown)} (disponibili: {", ".join(LAWS)})')
    return laws


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('name', nargs='?', choices=list(SCENARIOS), help='scenario of the catalogue (scenarios.py)')
    p.add_argument('--scenario', nargs='*', default=None,
                   help='free launch arguments instead of a named scenario, e.g. push_enable:=true')
    p.add_argument('--controllers', default=','.join(LAWS), help=f'subset of {",".join(LAWS)}')
    p.add_argument('--gradino', type=float, default=None, metavar='N',
                   help='step force on the torso [N, + = backwards], with --scenario or alone')
    p.add_argument('--gradino-t', type=float, default=4.0, metavar='S', help='step start after release [s]')
    p.add_argument('--duration', type=float, default=None, help='seconds of logging per run')
    p.add_argument('--run-name', default=None, help='name of the run directory')
    p.add_argument('--out', default=default_out(), help='parent directory (default <ws>/benchmark_runs)')
    p.add_argument('--no-analysis', action='store_true', help='only the CSVs, no plots nor comparison')
    p.add_argument('--topic-extra', action='store_true',
                   help='also log /rotino/estimation_error, the estimator error of the MPC (more load on logger '
                        'and controller)')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])

    if args.name and args.scenario is not None:
        p.error('usa uno scenario del catalogo oppure --scenario, non entrambi')
    laws = parse_laws(p, args.controllers)
    sc = SCENARIOS.get(args.name) if args.name else None
    launch_args = sc.args if sc else (args.scenario or [])
    duration = args.duration or (sc.duration if sc else 20.0)
    if args.name and args.gradino is not None:
        p.error('--gradino vale con --scenario o da solo, non con uno scenario del catalogo')
    disturbance = (sc.disturbance if sc else
                   ({'force': args.gradino, 'start_time': args.gradino_t} if args.gradino is not None else None))
    base = args.name or ('_'.join(a.split(':=')[0] for a in launch_args)
                         or ('gradino' if disturbance else 'equilibrio'))
    run_dir = os.path.join(args.out, args.run_name or f'{base}_{datetime.now():%Y%m%d_%H%M%S}')

    print(f'scenario : {sc.title if sc else "argomenti liberi"} ({" ".join(launch_args) or "nessun argomento"})')
    print(f'leggi    : {", ".join(laws)}')
    print(f'durata   : {duration:.0f} s ciascuna')
    if disturbance:
        print(f"gradino  : {disturbance['force']:+.2f} N a t={disturbance.get('start_time', 4.0):.1f} s")
    if args.topic_extra:
        print('logger   : anche /rotino/estimation_error')
    print(f'output   : {run_dir}')
    run_scenario(run_dir, laws, launch_args, duration, args.name, disturbance, args.topic_extra)

    if not args.no_analysis:
        from rotino_benchmark.compare import analyse_scenario
        analyse_scenario(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
