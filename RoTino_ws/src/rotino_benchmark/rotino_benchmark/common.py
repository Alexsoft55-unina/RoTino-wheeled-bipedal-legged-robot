"""What every benchmark tool shares: the two control laws, their colours and where results go."""

import glob
import math
import os

LAWS = ('pid', 'mpc')
LABELS = {'pid': 'PID', 'mpc': 'MPC'}
COLORS = {'pid': '#d62728', 'mpc': '#1f77b4'}
# commanded actuation torques in the log: wheels, hips, knees
WHEEL_TORQUE_COLUMNS = ('wheel_L_torque_cmd', 'wheel_R_torque_cmd')
TORQUE_COLUMNS = WHEEL_TORQUE_COLUMNS + ('hip_L_torque_cmd', 'hip_R_torque_cmd',
                                         'knee_L_torque_cmd', 'knee_R_torque_cmd')


def norm_series(*signals):
    """Euclidean norm of the signals at every sample (NaN where one of them is missing)."""
    return [math.sqrt(sum(v * v for v in row)) for row in zip(*signals)]


def workspace_dir():
    """<ws> from <ws>/install/rotino_benchmark/share/rotino_benchmark, else the current directory."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('rotino_benchmark')
        ws = os.path.abspath(os.path.join(share, '..', '..', '..', '..'))
        if os.path.isdir(os.path.join(ws, 'src')):
            return ws
    except Exception:
        pass
    return os.getcwd()


def default_out():
    return os.path.join(workspace_dir(), 'benchmark_runs')


def find_csv(run_dir, law):
    """Newest CSV of `law` in a scenario directory, or None."""
    files = sorted(glob.glob(os.path.join(run_dir, f'rotino_{law}_*.csv')))
    return files[-1] if files else None


def is_suite(path):
    """A suite directory holds one sub-directory per scenario (each with the CSVs)."""
    return any(os.path.isdir(os.path.join(path, d)) and any(find_csv(os.path.join(path, d), law) for law in LAWS)
               for d in os.listdir(path))


def scenario_dirs(suite_dir):
    return sorted(os.path.join(suite_dir, d) for d in os.listdir(suite_dir)
                  if os.path.isdir(os.path.join(suite_dir, d))
                  and any(find_csv(os.path.join(suite_dir, d), law) for law in LAWS))
