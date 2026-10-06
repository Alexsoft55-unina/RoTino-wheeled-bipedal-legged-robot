"""Runs the whole benchmark: every scenario of scenarios.py with the PID and the MPC, then the comparison."""

import argparse
import os
import sys
from datetime import datetime

from rotino_benchmark.campaign import parse_laws, run_scenario
from rotino_benchmark.common import LAWS, default_out
from rotino_benchmark.scenarios import SCENARIOS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scenarios', nargs='*', help=f'subset of: {", ".join(SCENARIOS)} (default: all)')
    p.add_argument('--controllers', default=','.join(LAWS), help=f'subset of {",".join(LAWS)}')
    p.add_argument('--out', default=default_out(), help='parent directory (default <ws>/benchmark_runs)')
    p.add_argument('--name', default=None, help='name of the suite directory (default suite_<date>)')
    p.add_argument('--list', action='store_true', help='print the scenarios and exit')
    p.add_argument('--topic-extra', action='store_true',
                   help='also log /rotino/estimation_error, the estimator error of the MPC (see campaign)')
    p.add_argument('--analisi', metavar='SUITE_DIR', default=None,
                   help='do not simulate: only (re)build the comparison of an existing suite')
    args = p.parse_args([a for a in (argv if argv is not None else sys.argv[1:]) if a != '--'])

    if args.list:
        for name, sc in SCENARIOS.items():
            step = f"  gradino {sc.disturbance['force']:+.1f} N" if sc.disturbance else ''
            print(f'{name:16s} {sc.duration:4.0f} s  {sc.title}: {sc.description}  '
                  f'[{" ".join(sc.args) or "-"}]{step}')
        return 0
    from rotino_benchmark.compare import analyse_suite
    if args.analisi:
        return 0 if analyse_suite(args.analisi, recompute=True) else 1

    unknown = [s for s in args.scenarios if s not in SCENARIOS]
    if unknown:
        p.error(f'scenari sconosciuti: {", ".join(unknown)} (vedi --list)')
    names = args.scenarios or list(SCENARIOS)
    laws = parse_laws(p, args.controllers)
    suite_dir = os.path.join(args.out, args.name or f'suite_{datetime.now():%Y%m%d_%H%M}')
    total = sum(SCENARIOS[n].duration + 15.0 for n in names) * len(laws)
    print(f'suite   : {suite_dir}\nscenari : {", ".join(names)}\nleggi   : {", ".join(laws)}\n'
          f'durata  : circa {total / 60:.0f} min di simulazione, poi l\'analisi')
    for i, name in enumerate(names, 1):
        sc = SCENARIOS[name]
        print(f'\n######## [{i}/{len(names)}] {name}: {sc.title}', flush=True)
        run_scenario(os.path.join(suite_dir, name), laws, sc.args, sc.duration, name, sc.disturbance,
                     args.topic_extra)
    return 0 if analyse_suite(suite_dir) else 1


if __name__ == '__main__':
    sys.exit(main())
