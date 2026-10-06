"""Figures of the results section, from the logs of a benchmark suite
(`ros2 run rotino_benchmark suite`, one sub-directory per scenario with a CSV per control law).

    python3 report/scripts/make_result_figures.py --suite <ws>/benchmark_runs/suite_<date>

Quantities compared between the two laws are taken from signals with the same meaning for both: the
ground-truth base pose and the joint angles (axle position, wheel clearance, whole-robot CoM), or the
multibody ZMP reconstruction of rotino_benchmark, and the commanded torques.
"""

import argparse
import math

import numpy as np

from common import (COLOR, FULL_W, INK, LABEL, LAWS, MOTION_START_TIME, MUTED, ground_truth, load_run, load_urdf,
                    panel_label, save, setup_style, true_travel)

plt = setup_style()
ZONE = '#f0ead6'            # obstacle zones: a neutral tint, darker than the grid


def legend_top(fig, ax, extra=(), y=1.0):
    handles = [plt.Line2D([], [], color=COLOR[law], label=LABEL[law]) for law in LAWS] + list(extra)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=len(handles), columnspacing=2.0)


def window(d, t0, t1):
    return (d['time_s'] >= t0) & (d['time_s'] <= t1)


# ---------------------------------------------------------------------------- push and step force
def fig_disturbances(suite, model_zmp):
    runs = {(sc, law): load_run(suite, sc, law) for sc in ('spinta', 'gradino') for law in LAWS}
    travel = {key: true_travel(model_zmp, d) for key, d in runs.items()}       # the reference is to stand still
    fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 3.9), sharex='col', gridspec_kw={'hspace': 0.16, 'wspace': 0.2})
    spans = {'spinta': (3.6, 9.0), 'gradino': (3.6, 12.0)}
    titles = {'spinta': '(a) impulsive push: 2.7 N s backwards on the torso', 'gradino': '(b) step force: 3 N backwards, held'}
    for col, sc in enumerate(('spinta', 'gradino')):
        t0, t1 = spans[sc]
        for law in LAWS:
            d = runs[(sc, law)]
            w = window(d, t0, t1)
            axes[0, col].plot(d['time_s'][w], d['theta_deg'][w], color=COLOR[law])
            axes[1, col].plot(d['time_s'][w], 1e3 * d['delta_s_m'][w], color=COLOR[law])
            axes[2, col].plot(d['time_s'][w], 1e3 * travel[(sc, law)][w], color=COLOR[law])
        for ax in axes[:, col]:
            ax.axvline(4.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
            ax.set_xlim(t0, t1)
        axes[1, col].axhline(30.0, color=COLOR['mpc'], lw=0.6, ls=(0, (1, 1.5)))
        axes[1, col].axhline(59.3, color=COLOR['pid'], lw=0.6, ls=(0, (1, 1.5)))
        axes[1, col].set_ylim(-36, 72)
        axes[2, col].set_xlabel('time since release [s]')
        panel_label(axes[0, col], titles[sc])
    axes[1, 0].text(8.95, 31.5, 'MPC bound: 30 mm', ha='right', va='bottom', fontsize=7.5)
    axes[1, 0].text(8.95, 60.5, 'PID bound: 59 mm', ha='right', va='bottom', fontsize=7.5)
    axes[0, 0].set_ylabel('pitch $\\theta$ [deg]')
    axes[1, 0].set_ylabel('commanded CoM\noffset [mm]')
    axes[2, 0].set_ylabel('axle position [mm]')
    fig.align_ylabels(axes[:, 0])
    legend_top(fig, axes[0, 0], y=0.93)
    save(fig, 'fig_disturbances.pdf')
    for sc in ('spinta', 'gradino'):
        for law in LAWS:
            d = runs[(sc, law)]
            w = window(d, 4.0, 20.0)
            tail = window(d, d['time_s'][-1] - 3.0, d['time_s'][-1])
            bound = 0.03 if law == 'mpc' else 0.0593
            print(f"   {sc:8s} {law}: pitch peak {np.nanmax(np.abs(d['theta_deg'][w])):6.2f} deg, offset peak "
                  f"{1e3 * np.nanmax(d['delta_s_m'][w]):5.1f} mm, on the bound {np.sum(w & (d['delta_s_m'] > bound - 5e-4)) * 0.002:5.2f} s, "
                  f"true drift max {1e3 * np.nanmax(np.abs(travel[(sc, law)][w])):6.1f} mm, final pitch {np.nanmean(d['theta_deg'][tail]):+5.2f} deg, "
                  f"final true position {1e3 * np.nanmean(travel[(sc, law)][tail]):+7.1f} mm, final offset {1e3 * np.nanmean(d['delta_s_m'][tail]):5.2f} mm, "
                  f"final torque {np.nanmean(d['wheel_L_torque_cmd'][tail]):.4f} Nm")


# ---------------------------------------------------------------------------- trapezoid
def fig_tracking(suite, model_zmp):
    from rotino_pid.zmp_balance import trapezoid_reference
    runs = {law: load_run(suite, 'trapezio', law) for law in LAWS}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.0), sharex=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    t0, t1 = 1.0, 9.0
    ref = runs['pid']
    w = window(ref, t0, t1)
    axes[0, 0].plot(ref['time_s'][w], ref['xdot_ref_ms'][w], color=INK, lw=0.9, ls='--')
    for law in LAWS:
        d = runs[law]
        w = window(d, t0, t1)
        axes[0, 0].plot(d['time_s'][w], d['xdot_ms'][w], color=COLOR[law])
        s_ref = np.array([trapezoid_reference(t - MOTION_START_TIME, 1.0, 0.6, 2.0)[0] for t in d['time_s']])
        err = true_travel(model_zmp, d) - s_ref                  # same ground-truth signal for both laws
        axes[0, 1].plot(d['time_s'][w], 1e3 * err[w], color=COLOR[law])
        print(f"   trapezio {law}: ground-truth position error rms {1e3 * np.sqrt(np.nanmean(err[window(d, 2.0, 12.0)] ** 2)):.1f} mm, "
              f"logged by the controller {1e3 * np.sqrt(np.nanmean(d['s_err_m'][window(d, 2.0, 12.0)] ** 2)):.1f} mm")
        axes[1, 0].plot(d['time_s'][w], d['theta_deg'][w], color=COLOR[law])
        axes[1, 1].plot(d['time_s'][w], 1e3 * d['delta_s_m'][w], color=COLOR[law])
    axes[0, 0].set_ylabel('speed [m/s]')
    axes[0, 1].set_ylabel('position error [mm]')
    axes[1, 0].set_ylabel('pitch $\\theta$ [deg]')
    axes[1, 1].set_ylabel('commanded CoM offset [mm]')
    for ax, name in zip(axes.ravel(), ('(a) forward speed', '(b) position error', '(c) pitch', '(d) commanded CoM offset')):
        ax.axvline(MOTION_START_TIME, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(t0, t1)
        panel_label(ax, name)
    for ax in axes[1]:
        ax.set_xlabel('time since release [s]')
    legend_top(fig, axes[0, 0], extra=[plt.Line2D([], [], color=INK, lw=0.9, ls='--', label='reference')], y=0.94)
    save(fig, 'fig_tracking.pdf')
    for law in LAWS:
        d = runs[law]
        w = window(d, 1.5, 2.0)
        print(f"   trapezio {law}: pitch at t = 2.0 s (start of the motion) {d['theta_deg'][np.argmin(np.abs(d['time_s'] - 2.0))]:+.2f} deg, "
              f"mean offset command in 1.5-2.0 s {1e3 * np.nanmean(d['delta_s_m'][w]):+.2f} mm, speed peak {np.nanmax(d['xdot_ms']):.3f} m/s")


# ---------------------------------------------------------------------------- S-curve
def planar_errors(model_zmp, d, lateral, duration, length=3.0):
    """Ground-truth axle path in the release frame and its error from the planned S-curve."""
    from rotino_description.planar_trajectory import CubicSTrajectory
    axle, _, _ = ground_truth(model_zmp, d)
    q = np.array([d['base_qx'][0], d['base_qy'][0], d['base_qz'][0], d['base_qw'][0]])
    yaw0 = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    rel = axle[:, :2] - axle[0, :2]
    xy = np.stack([c0 * rel[:, 0] + s0 * rel[:, 1], -s0 * rel[:, 0] + c0 * rel[:, 1]], -1)
    traj = CubicSTrajectory(length, lateral, duration)
    ref = np.array([traj.sample(max(t - MOTION_START_TIME, 0.0))[:3] for t in d['time_s']])
    err = xy - ref[:, :2]
    along = err[:, 0] * np.cos(ref[:, 2]) + err[:, 1] * np.sin(ref[:, 2])
    cross = -err[:, 0] * np.sin(ref[:, 2]) + err[:, 1] * np.cos(ref[:, 2])
    xs = np.linspace(0.0, traj.X, 200)
    return xy, along, cross, np.stack([xs, traj._y(xs)], -1)


def fig_scurve(suite, model_zmp):
    fig = plt.figure(figsize=(FULL_W, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.42, wspace=0.26)
    top, cross_ax, zmp_ax = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    half = None
    for law in LAWS:
        d = load_run(suite, 'curva_S_veloce', law)
        xy, along, cross, path = planar_errors(model_zmp, d, 1.0, 5.0)
        w = window(d, 1.0, 10.0)
        if law == 'pid':
            top.plot(path[:, 0], path[:, 1], color=INK, lw=0.9, ls='--')
        top.plot(xy[w, 0], xy[w, 1], color=COLOR[law])
        cross_ax.plot(d['time_s'][w], 1e3 * cross[w], color=COLOR[law])
        zmp_ax.plot(d['time_s'][w], 100.0 * d['zmp_zmp_y_rel'][w], color=COLOR[law])
        m = window(d, 2.0, 9.0)
        print(f"   curva_S_veloce {law}: ground-truth cross-track error rms {1e3 * np.sqrt(np.nanmean(cross[m] ** 2)):5.1f} mm, "
              f"max {1e3 * np.nanmax(np.abs(cross[m])):5.1f} mm; along-track rms {1e3 * np.sqrt(np.nanmean(along[m] ** 2)):5.1f} mm, "
              f"max {1e3 * np.nanmax(np.abs(along[m])):5.1f} mm; final lateral offset {1e3 * np.nanmean(cross[window(d, 11.0, 13.0)]):+6.1f} mm; "
              f"|lateral ZMP| max {100 * np.nanmax(np.abs(d['zmp_zmp_y_rel'][m])):4.1f} % of d/2")
    top.set_aspect('equal')
    top.set_xlabel('$x$ [m]')
    top.set_ylabel('$y$ [m]')
    top.set_ylim(-0.25, 1.3)
    panel_label(top, '(a) path of the axle midpoint (ground truth)')
    cross_ax.set_ylabel('cross-track error [mm]')
    panel_label(cross_ax, '(b) lateral deviation from the path')
    zmp_ax.set_ylabel('lateral ZMP [% of $d/2$]')
    zmp_ax.set_xlabel('time since release [s]')
    panel_label(zmp_ax, '(c) lateral ZMP, 100 % = on a wheel')
    cross_ax.set_xticklabels([])
    for ax in (cross_ax, zmp_ax):
        ax.set_xlim(1.0, 10.0)
        ax.axhline(0, color=MUTED, lw=0.6)
    legend_top(fig, top, extra=[plt.Line2D([], [], color=INK, lw=0.9, ls='--', label='planned path')], y=0.94)
    save(fig, 'fig_scurve.pdf')
    for law in LAWS:                                   # slow S-curve: numbers only
        d = load_run(suite, 'curva_S', law)
        xy, along, cross, _ = planar_errors(model_zmp, d, 0.6, 12.0)
        m = window(d, 2.0, 16.0)
        print(f"   curva_S        {law}: ground-truth cross-track error rms {1e3 * np.sqrt(np.nanmean(cross[m] ** 2)):5.1f} mm, "
              f"max {1e3 * np.nanmax(np.abs(cross[m])):5.1f} mm; along-track rms {1e3 * np.sqrt(np.nanmean(along[m] ** 2)):5.1f} mm")


# ---------------------------------------------------------------------------- uneven ground
def fig_terrain(suite, model_zmp):
    zones = {'dossi': ((1.44, 1.56), (2.24, 2.36), (3.04, 3.16)), 'piastrelle': ((1.53, 3.52),)}
    spans = {'dossi': (1.0, 4.0), 'piastrelle': (1.0, 4.2)}
    fig, axes = plt.subplots(3, 2, figsize=(FULL_W, 3.9), sharex='col', gridspec_kw={'hspace': 0.16, 'wspace': 0.33})
    for col, sc in enumerate(('dossi', 'piastrelle')):
        x0, x1 = spans[sc]
        for law in LAWS:
            d = load_run(suite, sc, law)
            axle, low, com = ground_truth(model_zmp, d)
            x = d['x_m']
            w = (x >= x0) & (x <= x1) & (d['time_s'] > 1.0)
            before = (x > x0 - 0.3) & (x < x0 + 0.3)
            body = 1e3 * (d['base_z_m'] - np.nanmedian(d['base_z_m'][before]))
            common = 0.5 * (d['wheel_L_torque_cmd'] + d['wheel_R_torque_cmd'])
            if sc == 'dossi':
                rows = (d['theta_deg'], body, np.degrees(d['hip_L_pos']))
            else:
                rows = (d['roll_deg'], 100.0 * d['zmp_zmp_y_rel'], d['theta_deg'])
            for ax, y in zip(axes[:, col], rows):
                ax.plot(x[w], y[w], color=COLOR[law])
            zw = np.zeros(len(x), bool)
            for a, b in zones[sc]:
                zw |= (x >= a - 0.05) & (x <= b + 0.6)
            zw &= d['time_s'] > 1.0
            print(f"   {sc:10s} {law}: over the obstacles pitch peak {np.nanmax(np.abs(d['theta_deg'][zw])):5.2f} deg, roll peak "
                  f"{np.nanmax(np.abs(d['roll_deg'][zw])):5.2f} deg, torso height excursion {np.nanmax(body[zw]) - np.nanmin(body[zw]):5.1f} mm "
                  f"(+{np.nanmax(body[zw]):.1f}/{np.nanmin(body[zw]):.1f}), hip range {np.degrees(np.nanmax(d['hip_L_pos'][zw]) - np.nanmin(d['hip_L_pos'][zw])):5.1f} deg, "
                  f"common torque peak {np.nanmax(np.abs(common[zw])):.2f} Nm, |lat ZMP| max {100 * np.nanmax(np.abs(d['zmp_zmp_y_rel'][zw])):5.1f} %, "
                  f"min wheel load {np.nanmin(np.minimum(d['zmp_fn_left_N'][zw], d['zmp_fn_right_N'][zw])):5.1f} N, "
                  f"CoM excursion (ground truth) {1e3 * (np.nanmax(com[zw, 2]) - np.nanmin(com[zw, 2])):5.1f} mm")
        for ax in axes[:, col]:
            for a, b in zones[sc]:
                ax.axvspan(a, b, color=ZONE, lw=0)
            ax.set_xlim(x0, x1)
        axes[2, col].set_xlabel('distance travelled [m]')
    axes[0, 0].set_ylabel('pitch $\\theta$ [deg]')
    axes[1, 0].set_ylabel('torso height\nchange [mm]')
    axes[2, 0].set_ylabel('hip joint angle [deg]')
    axes[0, 1].set_ylabel('roll [deg]')
    axes[1, 1].set_ylabel('lateral ZMP\n[% of $d/2$]')
    axes[2, 1].set_ylabel('pitch $\\theta$ [deg]')
    panel_label(axes[0, 0], '(a) three speed bumps, 20 mm high, at 0.5 m/s')
    panel_label(axes[0, 1], '(b) field of tiles, 5 to 20 mm high, at 0.5 m/s')
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    from matplotlib.patches import Patch
    legend_top(fig, axes[0, 0], extra=[Patch(color=ZONE, label='obstacle')], y=0.93)
    save(fig, 'fig_terrain.pdf')
    for law in LAWS:                                   # ramp: numbers only
        d = load_run(suite, 'rampa', law)
        x = d['x_m']
        zw = (x >= 1.45) & (x <= 5.1) & (d['time_s'] > 1.0)
        body = 1e3 * (d['base_z_m'] - np.nanmedian(d['base_z_m'][(x > 0.7) & (x < 1.3)]))
        print(f"   rampa      {law}: pitch peak {np.nanmax(np.abs(d['theta_deg'][zw])):5.2f} deg, torso height on the plateau "
              f"{np.nanmedian(body[(x > 2.6) & (x < 3.4)]):+5.1f} mm, friction use max {100 * np.nanmax(d['zmp_friction_use'][zw]):4.1f} %")


# ---------------------------------------------------------------------------- jump
def fig_jump(suite, model_zmp):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.95), gridspec_kw={'wspace': 0.42})
    t0, t1 = 3.8, 8.5
    for law in LAWS:
        d = load_run(suite, 'salto', law)
        axle, low, com = ground_truth(model_zmp, d)
        w = window(d, t0, t1)
        stand = np.nanmedian(com[window(d, 3.0, 3.9), 2])
        axes[0].plot(d['time_s'][w], 1e3 * low[w], color=COLOR[law])
        axes[1].plot(d['time_s'][w], 1e3 * (com[w, 2] - stand), color=COLOR[law])
        axes[2].plot(d['time_s'][w], d['theta_deg'][w], color=COLOR[law])
        air = w & (low > 0.002)
        ta = d['time_s'][air]
        common = 0.5 * (d['wheel_L_torque_cmd'] + d['wheel_R_torque_cmd'])
        legs = np.nanmax(np.abs(np.stack([d['hip_L_torque_cmd'], d['knee_L_torque_cmd']])), axis=0)
        print(f"   salto {law}: wheels more than 2 mm off the ground for {air.sum() * 0.002:.3f} s (t = {ta.min():.3f}..{ta.max():.3f}), "
              f"peak clearance {1e3 * np.nanmax(low[w]):5.1f} mm, CoM rise {1e3 * (np.nanmax(com[w, 2]) - stand):5.1f} mm, "
              f"lowest CoM in the squat {1e3 * (np.nanmin(com[window(d, 4.0, ta.min()), 2]) - stand):+5.1f} mm, pitch peak after landing "
              f"{np.nanmax(np.abs(d['theta_deg'][window(d, ta.max(), t1)])):5.2f} deg, |wheel torque| peak {np.nanmax(np.abs(common[w])):5.2f} Nm, "
              f"|leg torque| peak {np.nanmax(legs[w]):5.1f} Nm")
    axes[0].set_ylabel('wheel clearance [mm]')
    axes[1].set_ylabel('CoM height change [mm]')
    axes[2].set_ylabel('pitch $\\theta$ [deg]')
    for ax, name in zip(axes, ('(a) wheels above the ground', '(b) whole-robot CoM', '(c) pitch')):
        ax.set_xlim(t0, t1)
        ax.set_xlabel('time since release [s]')
        panel_label(ax, name)
    legend_top(fig, axes[0], y=0.95)
    save(fig, 'fig_jump.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', required=True, help='suite directory written by `ros2 run rotino_benchmark suite`')
    args = parser.parse_args()
    from rotino_description.zmp import ZmpModel
    zmp_model = ZmpModel(load_urdf())
    fig_disturbances(args.suite, zmp_model)
    fig_tracking(args.suite, zmp_model)
    fig_scurve(args.suite, zmp_model)
    fig_terrain(args.suite, zmp_model)
    fig_jump(args.suite, zmp_model)
