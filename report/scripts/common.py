"""Shared helpers of the report scripts: the RoTino model library loaded without a ROS graph, the
linear closed loops of the two control laws, the benchmark logs and the plotting style.

The scripts import the *same* modules the controllers run (rotino_description, rotino_pid, rotino_mpc),
so every number and curve of the report comes from the code of this workspace. Requirements: numpy,
matplotlib, xacro and urdf_parser_py (all present in a ROS 2 Humble environment; on another machine
`pip install numpy matplotlib xacro urdf-parser-py`).
"""

import csv
import glob
import math
import os
import sys

import numpy as np

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _pkg in ('rotino_description', 'rotino_pid', 'rotino_mpc', 'rotino_benchmark'):
    sys.path.insert(0, os.path.join(WS, 'src', _pkg))
if not hasattr(np, 'trapz'):                # numpy >= 2.4 dropped the alias used by planar_trajectory
    np.trapz = np.trapezoid

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
G = 9.81
DT = 0.002                                   # control period, 500 Hz

# Constants of rotino_mpc/controller.py (the node itself imports rclpy, so they are repeated here;
# print_numbers.py checks them against the source file).
LQR_Q = np.diag([30.0, 400.0, 80.0, 15.0, 6.0, 2.0])
_T_CD = np.array([[0.5, 0.5], [-0.5, 0.5]])
LQR_R = _T_CD.T @ np.diag([2.0, 100.0]) @ _T_CD
MPC_HORIZON, MPC_DT = 25, 0.02
MPC_S_H, MPC_W_H = (50.0, 20.0), 200.0
MPC_S_V, MPC_W_V = (5000.0, 150.0), 2e-3
DS_MAX_CAP = 0.03
MOTION_START_TIME = 2.0


# ---------------------------------------------------------------------------- model
def load_urdf():
    import xacro
    pkg = os.path.join(WS, 'src', 'rotino_description')
    return xacro.process_file(os.path.join(pkg, 'urdf', 'rotino.urdf.xacro'), mappings={
        'controllers_file': os.path.join(pkg, 'config', 'rotino_controllers.yaml')}).toxml()


def load_model():
    from rotino_description.model import WBRModel
    return WBRModel(load_urdf())


def expm(X):
    E, term = np.eye(len(X)), np.eye(len(X))
    for n in range(1, 60):
        term = term @ X / n
        E = E + term
    return E


def scheduling_grid(model):
    """(l, I_y) of the 25 leg postures used by the controller for the TV-LQR, sorted by l."""
    out = []
    for hip in np.linspace(-0.45, 0.45, 25):
        e = model.equivalent_centroid(hip, -2.0 * hip)
        out.append((e['l'], e['I_y']))
    return sorted(out)


def lqr_schedule(model):
    from rotino_mpc.solvers import LQRSchedule
    samples = scheduling_grid(model)
    return LQRSchedule(model, samples, LQR_Q, LQR_R), samples


def lumped(model, hip=0.0, mass_scale=1.0):
    """Sagittal VL-WIP coefficients, CoM lever k_sc (whole-robot CoM offset per rad of pendulum angle)
    and whole-robot CoM height for a leg posture and an upper-body mass scale."""
    p = model.p
    e = model.equivalent_centroid(hip, -2.0 * hip)
    mb0 = p.m_b
    p.m_b = mb0 * mass_scale
    try:
        c = model.vlwip_coefficients(e['l'], I_y=e['I_y'] * mass_scale)
    finally:
        p.m_b = mb0
    M = model.kin.total_mass - mb0 + mb0 * mass_scale
    return c, mb0 * mass_scale * e['l'] / M, (mb0 * mass_scale * (e['l'] + p.r) + 2 * p.m_w * p.r) / M


def pid_cascade(model, hip=0.0, mass_scale=1.0, rho=None, gains=None):
    """Linear closed loop of the ZMP cascade of rotino_pid on the sagittal VL-WIP.

    State x = [s, theta, s_dot, theta_dot, int(xi)]; returns (A, B, k) with x_dot = A x + B tau and
    tau = k x the per-wheel torque of ZmpSagittalBalance.step without clamps and with zero reference."""
    from rotino_pid.zmp_balance import SagittalGains
    g = gains or SagittalGains()
    rho = g.authority if rho is None else rho
    c, k_sc, h = lumped(model, hip, mass_scale)
    w = math.sqrt(G / h)
    A, B = np.zeros((5, 5)), np.zeros(5)
    A[0, 2] = A[1, 3] = 1.0
    A[2, 1], A[3, 1] = c['a1'], c['a2']
    B[2], B[3] = 2 * c['b1'], 2 * c['b2']
    c_err = np.array([1.0, k_sc, 0, 0, 0])              # CoM error = axle error + lean
    cd_err = np.array([0, 0, 1.0, k_sc, 0])
    xi = c_err + cd_err / w                             # capture-point error
    A[4] = xi
    s_des = rho * (-(cd_err + g.k_dcm * xi + g.k_dcm_i * np.array([0, 0, 0, 0, 1.0])) / w)
    k = g.k_s * (np.array([0, k_sc, 0, 0, 0.0]) - s_des) + g.k_sd * np.array([0, 0, 0, k_sc, 0.0])
    return A, B, k


def pid_cascade_poles(model, hip=0.0, mass_scale=1.0, delay=1, rho=None, gains=None, dt=DT):
    """Poles (mapped to the s-plane) of the cascade sampled at dt with `delay` samples of actuation delay."""
    A, B, k = pid_cascade(model, hip, mass_scale, rho, gains)
    n = 5
    X = np.zeros((n + 1, n + 1))
    X[:n, :n], X[:n, n] = A * dt, B * dt
    E = expm(X)
    N = n + delay
    F = np.zeros((N, N))
    F[:n, :n], F[:n, n] = E[:n, :n], E[:n, n]
    for i in range(delay - 1):
        F[n + i, n + i + 1] = 1.0
    F[N - 1, :n] = k
    z = np.linalg.eigvals(F)
    z = z[np.abs(z) > 1e-9]
    return np.log(z.astype(complex)) / dt


def min_damping(poles):
    osc = poles[np.abs(poles.imag) > 1e-6]
    return 1.0 if len(osc) == 0 else float(np.min(-osc.real / np.abs(osc)))


# Robustness set of the gain design: nominal, mass -20 % / +20 %, squat and stretched legs, each with
# one and two samples of actuation delay.
ROBUST_CASES = [(hip, ms, d) for hip, ms in ((0.0, 1.0), (0.0, 0.8), (0.0, 1.2), (-0.3, 1.0), (0.3, 1.0))
                for d in (1, 2)]


# ---------------------------------------------------------------------------- benchmark logs
LAWS = ('pid', 'mpc')
LABEL = {'pid': 'PID with ZMP', 'mpc': 'MPC + TV-LQR'}
COLOR = {'pid': '#d62728', 'mpc': '#1f77b4'}     # same colours as rotino_benchmark and the slides
INK, MUTED, GRID = '#1a1a1a', '#6b6b6b', '#e4e4e4'


def find_log(suite, scenario, law):
    files = sorted(glob.glob(os.path.join(suite, scenario, f'rotino_{law}_*.csv')))
    if not files:
        raise FileNotFoundError(f'no log of {law} in {os.path.join(suite, scenario)}')
    return files[-1]


def read_csv(path, text=('jump_state',)):
    """Columns of a CSV as float arrays (NaN where empty); the `text` columns as string arrays."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        head = next(reader)
        cols = [[] for _ in head]
        for row in reader:
            for i, v in enumerate(row[:len(head)]):
                cols[i].append(v)
    out = {}
    for name, col in zip(head, cols):
        if name in text:
            out[name] = np.array(col)
        else:
            arr = np.empty(len(col))
            for i, v in enumerate(col):
                try:
                    arr[i] = float(v)
                except ValueError:
                    arr[i] = np.nan
            out[name] = arr
    return out


def load_run(suite, scenario, law):
    """Main log of a run plus the multibody ZMP series of rotino_benchmark (plots/zmp_<law>.csv),
    resampled on the time since release of the main log."""
    d = read_csv(find_log(suite, scenario, law))
    zpath = os.path.join(suite, scenario, 'plots', f'zmp_{law}.csv')
    if os.path.exists(zpath):
        z = read_csv(zpath)
        ok = np.isfinite(d['odom_stamp_s'])
        offset = float(np.median(d['odom_stamp_s'][ok] - d['time_s'][ok]))   # odom stamp -> time since release
        tz = z['t_s'] - offset
        for name, col in z.items():
            if name != 't_s':
                d['zmp_' + name] = np.interp(d['time_s'], tz, col, left=np.nan, right=np.nan)
    return d


def ground_truth(model_zmp, d):
    """Ground-truth quantities both laws can be compared on, from the logged base pose and joints:
    axle midpoint (N,3), lowest wheel point height above z = 0 (N,), whole-robot CoM (N,3)."""
    from rotino_description.kinematics import quat_to_matrix
    names = {'left_hip': 'hip_L_pos', 'right_hip': 'hip_R_pos', 'left_knee': 'knee_L_pos',
             'right_knee': 'knee_R_pos', 'left_wheel_joint': 'wheel_L_pos', 'right_wheel_joint': 'wheel_R_pos'}
    n = len(d['time_s'])
    axle, low, com = np.full((n, 3), np.nan), np.full(n, np.nan), np.full((n, 3), np.nan)
    kin = model_zmp.kin
    for k in range(n):
        q = (d['base_qx'][k], d['base_qy'][k], d['base_qz'][k], d['base_qw'][k])
        if not np.all(np.isfinite(q)):
            continue
        pos = np.array([d['base_x_m'][k], d['base_y_m'][k], d['base_z_m'][k]])
        joints = {j: d[c][k] for j, c in names.items()}
        frames = kin.link_frames(pos, quat_to_matrix(*q), joints)
        wl, wr = frames['left_wheel_link'][1], frames['right_wheel_link'][1]
        axle[k] = 0.5 * (wl + wr)
        low[k] = min(wl[2], wr[2]) - model_zmp.wheel_radius
        com[k] = kin.center_of_mass(frames)
    return axle, low, com


def true_travel(model_zmp, d):
    """Ground-truth travel of the axle midpoint along the heading the robot has at release (N,)."""
    axle, _, _ = ground_truth(model_zmp, d)
    q = (d['base_qx'][0], d['base_qy'][0], d['base_qz'][0], d['base_qw'][0])
    yaw0 = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    rel = axle[:, :2] - axle[0, :2]
    return rel[:, 0] * math.cos(yaw0) + rel[:, 1] * math.sin(yaw0)


# ---------------------------------------------------------------------------- plotting
def setup_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # Computer Modern as in the LaTeX text; cmr10 ships with matplotlib (mathtext draws the minus signs)
        'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'cm', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.6, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.15, 'lines.solid_capstyle': 'round',
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.2,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
    })
    return plt


FULL_W = 6.3        # inches: text width of the report


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    print('wrote', os.path.relpath(path, WS))


def panel_label(ax, text, loc='left'):
    ax.set_title(text, loc=loc, pad=3, fontsize=8.0, color=INK)
