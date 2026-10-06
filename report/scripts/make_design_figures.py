"""Figures of the modelling and design sections, computed from the code of this workspace (no simulation data).

    python3 report/scripts/make_design_figures.py        # -> report/figures/fig_*.pdf
"""

import math
import os
import re

import numpy as np

from common import (COLOR, DS_MAX_CAP, FULL_W, G, INK, MPC_DT, MPC_HORIZON, MPC_S_H, MPC_W_H, MUTED,
                    ROBUST_CASES, WS, load_model, lqr_schedule, lumped, min_damping, panel_label,
                    pid_cascade_poles, save, setup_style)

plt = setup_style()
ACCENT = '#7a3e9d'          # equivalent pendulum / design point: not one of the two control-law colours


# ---------------------------------------------------------------------------- robot and equivalent pendulum
def robot_points(m, hip, knee):
    """Sagittal points (x forward, z up, wheel contact at the origin) for a leg posture, torso level."""
    kx, kz = m.knee_xz
    wheel, _ = m.leg_fk(hip, knee)
    c, s = math.cos(hip), math.sin(hip)
    knee_p = np.array([kx * c + kz * s, -kx * s + kz * c])
    base = np.array([-wheel[0], m.p.r - wheel[1]])               # hip position
    com_b = m.upper_body_com_base(m.leg_joints(hip, knee))
    return {'hip': base, 'knee': base + knee_p, 'wheel': base + wheel,
            'com': base + np.array([com_b[0], com_b[2]])}


def draw_robot(ax, m, hip, knee, color=INK, lw=1.4, alpha=1.0, torso_fill='#eeeeee'):
    from matplotlib.patches import Circle, Rectangle
    pt = robot_points(m, hip, knee)
    hx, hz = pt['hip']
    ax.add_patch(Rectangle((hx - 0.05, hz - 0.055), 0.15, 0.11, fc=torso_fill, ec=color, lw=0.8, alpha=alpha, zorder=2))
    ax.add_patch(Rectangle((hx + 0.045 - 0.035, hz - 0.045 - 0.018), 0.07, 0.036, fc='#cfcfcf', ec=color, lw=0.6,
                           alpha=alpha, zorder=3))
    ax.add_patch(Circle(pt['wheel'], m.p.r, fc='white', ec=color, lw=lw, alpha=alpha, zorder=2))
    ax.plot(*zip(pt['hip'], pt['knee'], pt['wheel']), color=color, lw=lw + 0.8, alpha=alpha, zorder=4,
            solid_capstyle='round')
    for key in ('hip', 'knee', 'wheel'):
        ax.plot(*pt[key], 'o', ms=3.4, mfc='white', mec=color, mew=0.9, alpha=alpha, zorder=5)
    return pt


def fig_robot(m):
    from matplotlib.transforms import Affine2D
    p = m.p
    xa, xb = (-0.20, 0.17), (-0.43, 0.43)            # same metric scale in the two panels
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.2),
                               gridspec_kw={'width_ratios': [xa[1] - xa[0], xb[1] - xb[0]], 'wspace': 0.05})
    for ax, xl in ((a, xa), (b, xb)):
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axhline(0.0, color=MUTED, lw=0.8)
        ax.set_xlabel('$x$ [m]')
        ax.set_xlim(*xl)
        ax.set_ylim(-0.012, 0.44)
    a.set_ylabel('$z$ [m]')

    pt = draw_robot(a, m, 0.0, 0.0)
    e = m.equivalent_centroid(0.0, 0.0)
    a.plot(*zip(pt['wheel'], pt['com']), color=ACCENT, lw=1.3, zorder=6)
    a.plot(*pt['com'], 'o', ms=5.5, color=ACCENT, zorder=7)
    a.annotate(r'$m_b$', pt['com'], xytext=(8, 3), textcoords='offset points', color=INK, zorder=8)
    a.annotate(r'$l,\ \theta$', 0.5 * (pt['wheel'] + pt['com']), xytext=(7, -2), textcoords='offset points', color=INK)
    a.annotate('', (-0.125, p.r), (-0.125, 0.0), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.6))
    a.text(-0.133, 0.5 * p.r, r'$r$', ha='right', va='center')
    a.annotate('hip', pt['hip'], xytext=(-34, 3), textcoords='offset points', color=INK, va='center',
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    a.annotate('knee', pt['knee'], xytext=(-9, 0), textcoords='offset points', color=INK, ha='right', va='center')
    a.annotate('axle', pt['wheel'], xytext=(24, -11), textcoords='offset points', color=INK,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    a.text(pt['hip'][0] + 0.025, pt['hip'][1] + 0.062, 'torso', ha='center', va='bottom')
    panel_label(a, '(a) nominal posture')
    print('   nominal posture: l = %.3f m, theta = %.1f deg' % (e['l'], math.degrees(e['theta'])))

    for hip, x0 in ((0.45, -0.27), (0.0, 0.0), (-0.45, 0.27)):
        pts = robot_points(m, hip, -2 * hip)
        shift = np.array([x0, 0.0])
        tr = Affine2D().translate(*shift) + b.transData
        n_before = len(b.patches), len(b.lines)
        draw_robot(b, m, hip, -2 * hip, color=INK if hip == 0.0 else '#8e8e8e')
        for artist in b.patches[n_before[0]:] + b.lines[n_before[1]:]:
            artist.set_transform(tr)
        b.plot(*zip(pts['wheel'] + shift, pts['com'] + shift), color=ACCENT, lw=1.3, zorder=6)
        b.plot(*(pts['com'] + shift), 'o', ms=5.0, color=ACCENT, zorder=7)
        ee = m.equivalent_centroid(hip, -2 * hip)
        b.text(x0 + 0.02, 0.435, f'hip ${hip:+.2f}$ rad\n$l$ = {ee["l"]:.3f} m', ha='center', va='top', fontsize=7.5)
    b.set_yticklabels([])
    panel_label(b, '(b) the two ends and the centre of the gain-scheduling grid')
    save(fig, 'fig_robot.pdf')


# ---------------------------------------------------------------------------- test arena
def fig_arena():
    from matplotlib.patches import Rectangle
    sdf = open(os.path.join(WS, 'src', 'rotino_description', 'worlds', 'rotino_world.sdf')).read()
    tiles = [(float(x), float(y), float(h)) for x, y, h in re.findall(
        r'name="tile\d+_\d+_collision"><pose>([-\d.]+) ([-\d.]+) [-\d.]+ 0 0 0</pose>'
        r'<geometry><box><size>0\.24 0\.24 ([\d.]+)</size>', sdf)]
    assert len(tiles) == 64, len(tiles)
    heights = np.array([t[2] for t in tiles]) * 1e3

    fig = plt.figure(figsize=(FULL_W, 2.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1.0], hspace=1.0, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_aspect('equal')
    ax.grid(False)
    for x in (6.0, 6.8, 7.6):                                     # speed bumps: 12 cm wide at the ground
        ax.add_patch(Rectangle((x - 0.06, -1.0), 0.12, 2.0, fc='#8c8c8c', ec='none'))
    ax.add_patch(Rectangle((-5.6, -1.0), 0.6, 2.0, fc='#c9c9c9', ec='none'))      # up-ramp
    ax.add_patch(Rectangle((-7.4, -1.0), 1.8, 2.0, fc='#8c8c8c', ec='none'))      # plateau
    ax.add_patch(Rectangle((-8.0, -1.0), 0.6, 2.0, fc='#c9c9c9', ec='none'))      # down-ramp
    for x, y, h in tiles:
        g = 0.86 - 0.5 * (h * 1e3 - heights.min()) / (heights.max() - heights.min())
        ax.add_patch(Rectangle((x - 0.12, y - 0.12), 0.24, 0.24, fc=(g, g, g), ec='none'))
    paths = [((4.5, 0.0), (9.0, 0.0), 'bumps', (6.8, 1.45)),
             ((-3.5, 0.0), (-9.0, 0.0), 'ramp and plateau', (-6.5, 1.45)),
             ((4.875, -2.6), (4.875, -7.6), 'tiles', (6.9, -5.1))]
    for start, end, name, at in paths:
        ax.annotate('', end, start, arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9, mutation_scale=7))
        ax.plot(*start, 'o', ms=3.2, color=INK)
        ax.text(*at, name, ha='center', va='center')
    # motions of the flat-ground scenarios, from the origin
    from rotino_description.planar_trajectory import CubicSTrajectory
    for lateral, dur in ((0.6, 12.0), (1.0, 5.0)):
        tr = CubicSTrajectory(3.0, lateral, dur)
        xs = np.linspace(0.0, tr.X, 100)
        ax.plot(xs, tr._y(xs), color=INK, lw=0.9)
    ax.plot(0, 0, 'o', ms=3.2, color=INK)
    ax.text(1.4, -1.25, 'flat-ground scenarios\n(S-curves shown)', ha='center', va='center')
    ax.set_xlim(-9.6, 9.6)
    ax.set_ylim(-7.9, 2.3)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    panel_label(ax, '(a) top view of the test arena and benchmark paths')

    prof = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-0.15, 0.15, 301)
    zb = np.sqrt(np.clip(0.10 ** 2 - xs ** 2, 0, None)) - 0.08
    prof.fill_between(xs * 1e3, 0, np.clip(zb, 0, None) * 1e3, color='#8c8c8c', lw=0)
    prof.axhline(0, color=MUTED, lw=0.8)
    prof.set_ylim(0, 34)
    prof.set_xlabel('across a bump [mm]')
    prof.set_ylabel('height [mm]')
    panel_label(prof, '(b) bump: 20 mm high')
    hist = fig.add_subplot(gs[1, 1])
    hist.hist(heights, bins=np.arange(5, 21, 2.5), color='#8c8c8c', ec='white', lw=0.8)
    hist.set_xlabel('tile height [mm]')
    hist.set_ylabel('tiles')
    panel_label(hist, '(c) the 64 tiles')
    save(fig, 'fig_arena.pdf')


# ---------------------------------------------------------------------------- open loop and TV-LQR
def fig_vlwip_lqr(m):
    e0 = m.equivalent_centroid(0.0, 0.0)
    c = m.vlwip_coefficients(e0['l'], I_y=e0['I_y'])
    pole = math.sqrt(c['a2'])
    zero = math.sqrt((c['a2'] * c['b1'] - c['a1'] * c['b2']) / c['b1'])
    sched, samples = lqr_schedule(m)
    K0 = sched.gain(e0['l'])
    kc0 = 0.5 * (K0[0] + K0[1])
    sag = [0, 1, 3, 4]
    ls = np.array([s[0] for s in samples])
    gains, sched_poles, fixed_poles = [], [], []
    for l, iy in samples:
        A, B = m.vlwip_matrices(l, I_y=iy)
        Kl = sched.gain(l)
        kc = 0.5 * (Kl[0] + Kl[1])
        gains.append(kc[sag])
        As, Bs = A[np.ix_(sag, sag)], B[sag] @ np.ones(2)
        for store, k in ((sched_poles, kc), (fixed_poles, kc0)):
            ev = np.linalg.eigvals(As - np.outer(Bs, k[sag]))
            ev = ev[np.argsort(-ev.real)]
            store.append((ev[0].real, ev[2].real))               # slow pair, pendulum pole
    gains, sched_poles, fixed_poles = map(np.array, (gains, sched_poles, fixed_poles))

    fig, (a, b, d) = plt.subplots(1, 3, figsize=(FULL_W, 1.95), gridspec_kw={'wspace': 0.5})
    a.axhline(0, color=MUTED, lw=0.6)
    a.axvline(0, color=MUTED, lw=0.6)
    a.plot([-pole, pole, 0], [0, 0, 0], 'x', ms=6, mew=1.3, color=INK, label='poles')
    a.plot([-zero, zero], [0, 0], 'o', ms=5.5, mfc='white', mec=INK, mew=1.1, label='zeros')
    a.text(14.5, 0.62, f'RHP zero\n$+${zero:.1f} rad/s', ha='right', va='center', fontsize=7.5)
    a.plot([zero, zero], [0.09, 0.42], color=MUTED, lw=0.5)
    a.text(14.5, -0.55, f'unstable pole\n$+${pole:.1f} rad/s', ha='right', va='center', fontsize=7.5)
    a.plot([pole, pole], [-0.09, -0.36], color=MUTED, lw=0.5)
    a.text(-14.5, 0.62, 'double pole\nat the origin', ha='left', va='center', fontsize=7.5)
    a.plot([0, -5.5], [0.09, 0.42], color=MUTED, lw=0.5)
    a.set_xlim(-15, 15)
    a.set_ylim(-1, 1)
    a.set_yticks([])
    a.grid(False)
    a.set_xlabel('real axis [rad/s]')
    a.legend(loc='lower left', ncol=1, handletextpad=0.3)
    panel_label(a, '(a) open-loop poles and zeros')

    i0 = int(np.argmin(np.abs(ls - e0['l'])))
    ratio = gains / gains[i0]                          # columns: position (constant), pitch, speed, pitch rate
    for j, style, name, dy, va in ((1, '-', 'pitch', 0.014, 'bottom'), (2, '--', 'speed', 0.014, 'bottom'),
                                   (3, '-.', 'pitch rate', -0.016, 'top')):
        b.plot(ls, ratio[:, j], color=INK, ls=style)
        b.text(ls[0] + 0.002, ratio[0, j] + dy, name, ha='left', va=va, fontsize=7.5)
    b.axvline(e0['l'], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    b.set_xlabel('pendulum length $l$ [m]')
    b.set_ylabel('gain / gain at $l_{nom}$')
    b.set_ylim(0.78, 1.2)
    panel_label(b, '(b) scheduled LQR gains')

    d.plot(ls, sched_poles[:, 0], color=INK, label='scheduled $K(l)$')
    d.plot(ls, sched_poles[:, 1], color=INK)
    d.plot(ls, fixed_poles[:, 0], color=MUTED, ls='--', label='fixed $K(l_{nom})$')
    d.plot(ls, fixed_poles[:, 1], color=MUTED, ls='--')
    d.text(ls[-1], sched_poles[-1, 0] - 0.45, 'slow pair', ha='right', va='top', fontsize=7.5)
    d.text(ls[0] + 0.004, sched_poles[0, 1] - 0.5, 'pendulum pole', ha='left', va='top', fontsize=7.5)
    d.set_xlabel('pendulum length $l$ [m]')
    d.set_ylabel('real part [1/s]')
    d.set_ylim(-11.5, 0.5)
    d.legend(loc='center', bbox_to_anchor=(0.5, 0.62))
    panel_label(d, '(c) closed-loop poles')
    save(fig, 'fig_vlwip_lqr.pdf')


# ---------------------------------------------------------------------------- LIPM preview
def fig_lipm_preview(m):
    from rotino_pid.zmp_balance import LipmPreview, trapezoid_path, trapezoid_reference
    _, _, h = lumped(m)
    w = math.sqrt(G / h)
    fig, b = plt.subplots(figsize=(0.72 * FULL_W, 2.0))      # included at 0.72 of the text width

    t0 = 2.0
    path = lambda t: trapezoid_path(np.asarray(t) - t0, 1.0, 0.6, 2.0)
    ts = np.arange(1.0, 6.6, 0.01)
    acc_path = np.array([trapezoid_reference(t - t0, 1.0, 0.6, 2.0)[2] for t in ts])
    prev = LipmPreview()
    acc_ref = np.array([prev(path, t, w)[0] for t in ts])
    b.plot(ts, acc_path, color=MUTED, ls='--', label='acceleration of the planned wheel path')
    b.plot(ts, acc_ref, color=COLOR['pid'], label=r'CoM acceleration reference $\omega^2(c_{ref}-p_{ref})$')
    lead = ts[np.argmax(acc_ref > 0.05 * 0.6)]
    b.annotate(f'the lean starts {t0 - lead:.2f} s before\nthe wheels accelerate', (lead, 0.03), xytext=(1.0, -0.5),
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5), fontsize=7.5, va='center')
    b.set_xlabel('time since release [s]')
    b.set_ylabel(r'acceleration [m/s$^2$]')
    b.set_ylim(-0.95, 1.3)
    b.legend(loc='upper right')
    save(fig, 'fig_lipm_preview.pdf')
    print('   preview: the reference acceleration exceeds 5 %% of its plateau %.2f s before the start' % (t0 - lead))


# ---------------------------------------------------------------------------- authority of the outer loop
def fig_pid_authority(m):
    from rotino_pid.zmp_balance import SagittalGains
    g = SagittalGains()
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.05), gridspec_kw={'wspace': 0.3})
    rhos = np.linspace(0.1, 1.0, 46)
    cmap = plt.get_cmap('Greys')
    for rho in rhos:
        sv = pid_cascade_poles(m, 0.0, 1.0, 2, rho)
        sv = sv[np.abs(sv) < 100]
        a.plot(sv.real, sv.imag, '.', ms=3.2, color=cmap(0.25 + 0.7 * (rho - 0.1) / 0.9))
    for rho, marker, lab in ((g.authority, 'o', f'$\\rho$ = {g.authority} (design)'), (1.0, 's', r'$\rho$ = 1')):
        sv = pid_cascade_poles(m, 0.0, 1.0, 2, rho)
        sv = sv[np.abs(sv) < 100]
        a.plot(sv.real, sv.imag, marker, ms=5.5, mfc='none', mec=COLOR['pid'] if marker == 'o' else INK, mew=1.2,
               ls='none', label=lab)
        if rho == 1.0:
            top = sv[np.argmax(sv.imag)]
            a.annotate(f'damping {min_damping(sv):.2f}', (top.real, top.imag), xytext=(-8, 2), textcoords='offset points',
                       ha='right', va='center', fontsize=7.5)
    a.set_xlim(-70, 3)
    a.set_ylim(-19, 19)
    a.set_xlabel('real part [1/s]')
    a.set_ylabel('imaginary part [rad/s]')
    a.legend(loc='lower left')
    panel_label(a, r'(a) dominant poles, $\rho$ from 0.1 (light) to 1 (dark)')

    rr = np.linspace(0.15, 1.0, 18)

    def limit(rho, cases):
        lo, hi = 10.0, 3000.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            ok = all(pid_cascade_poles(m, hip, ms, d, rho, SagittalGains(k_s=mid)).real.max() < 0 for hip, ms, d in cases)
            lo, hi = (mid, hi) if ok else (lo, mid)
        return lo

    two = np.array([limit(r, ROBUST_CASES) for r in rr])
    one = np.array([limit(r, [c for c in ROBUST_CASES if c[2] == 1]) for r in rr])
    b.fill_between(rr, two, 1200, color='#eeeeee', lw=0)
    b.plot(rr, one, color=MUTED, ls='--', label='limit with 1 sample of delay')
    b.plot(rr, two, color=INK, label='limit with 2 samples of delay')
    b.plot(g.authority, g.k_s, 'o', ms=5.5, color=COLOR['pid'])
    b.annotate(f'design point\n$\\rho$ = {g.authority}, $K_s$ = {g.k_s:.0f} Nm/m', (g.authority, g.k_s), xytext=(9, 0),
               textcoords='offset points', va='center', fontsize=7.5)
    b.text(0.98, 420, 'unstable', ha='right', color=INK)
    b.set_ylim(0, 650)
    b.set_xlim(0.15, 1.0)
    b.set_xlabel(r'authority $\rho$ of the capture-point loop')
    b.set_ylabel('inner stiffness $K_s$ [Nm/m]')
    b.legend(loc='upper right')
    panel_label(b, '(b) stability limit over the robustness cases')
    save(fig, 'fig_pid_authority.pdf')
    print('   stability limit at rho = 0.4 / 1.0 (two delays): %.0f / %.0f Nm/m' % (limit(0.4, ROBUST_CASES), limit(1.0, ROBUST_CASES)))


# ---------------------------------------------------------------------------- MPC plan and solver
def fig_mpc_plan(m):
    from rotino_mpc.solvers import _condense, box_qp
    e0 = m.equivalent_centroid(0.0, 0.0)
    N, dt, z = MPC_HORIZON, MPC_DT, e0['Z_C']
    Ah = np.array([[1.0, dt], [0.0, 1.0]])
    Bh = np.array([[0.5 * dt * dt], [dt]]) * G / z
    Phi, Gam, Cst = _condense(Ah, Bh, np.zeros(2), N)
    Sb = np.kron(np.eye(N), np.diag(MPC_S_H))
    H = Gam.T @ Sb @ Gam + MPC_W_H * np.eye(N)
    ref = np.zeros((N, 2))

    def grad(x0):
        return Gam.T @ Sb @ (Phi @ x0 + Cst - ref.reshape(-1))

    x0 = np.array([-0.12, -0.45])                   # CoM pushed back: 12 cm behind the reference, 0.45 m/s
    g0 = grad(x0)
    u_opt = box_qp(H, g0, -DS_MAX_CAP, DS_MAX_CAP, None, iters=50000)
    u_free = np.linalg.solve(H, -g0)
    pred = (Phi @ x0 + Gam @ u_opt).reshape(N, 2)
    pred_free = (Phi @ x0 + Gam @ u_free).reshape(N, 2)
    tt = dt * np.arange(1, N + 1)

    fig, (a, b, d) = plt.subplots(1, 3, figsize=(FULL_W, 1.95), gridspec_kw={'wspace': 0.42})
    a.axhline(0, color=MUTED, lw=0.6, ls='--')
    a.plot(np.r_[0, tt], 1e3 * np.r_[x0[0], pred[:, 0]], color=COLOR['mpc'], label='constrained plan')
    a.plot(np.r_[0, tt], 1e3 * np.r_[x0[0], pred_free[:, 0]], color=MUTED, ls=(0, (1, 1)), label='without bounds')
    a.set_xlabel('time along the horizon [s]')
    a.set_ylabel('CoM position error [mm]')
    a.set_ylim(-190, 12)
    a.legend(loc='center right', bbox_to_anchor=(1.0, 0.68))
    panel_label(a, '(a) predicted CoM position')

    b.axhline(1e3 * DS_MAX_CAP, color=MUTED, lw=0.6, ls='--')
    b.step(np.r_[0, tt], 1e3 * np.r_[u_opt, u_opt[-1]], where='post', color=COLOR['mpc'])
    b.step(np.r_[0, tt], 1e3 * np.r_[u_free, u_free[-1]], where='post', color=MUTED, ls=(0, (1, 1)))
    b.text(0.5, 1e3 * DS_MAX_CAP + 3, 'bound: 30 mm', ha='right', va='bottom', fontsize=7.5, color=INK)
    b.set_ylim(-12, 1e3 * u_free.max() * 1.06)
    b.set_xlabel('time along the horizon [s]')
    b.set_ylabel(r'planned offset $\Delta s$ [mm]')
    panel_label(b, r'(b) planned input sequence')

    cost = lambda u, g_: 0.5 * u @ H @ u + g_ @ u
    its = np.arange(1, 61)
    cold = [(cost(box_qp(H, g0, -DS_MAX_CAP, DS_MAX_CAP, None, iters=int(i)), g0) - cost(u_opt, g0)) / abs(cost(u_opt, g0))
            for i in its]
    # warm start as in UpperBodyMPC._solve: previous solution shifted by one step, state advanced by one step
    x1 = Ah @ x0 + Bh[:, 0] * u_opt[0]
    g1 = grad(x1)
    u1 = box_qp(H, g1, -DS_MAX_CAP, DS_MAX_CAP, None, iters=50000)
    warm0 = np.clip(np.concatenate((u_opt[1:], u_opt[-1:])), -DS_MAX_CAP, DS_MAX_CAP)
    warm = [(cost(box_qp(H, g1, -DS_MAX_CAP, DS_MAX_CAP, warm0, iters=int(i)), g1) - cost(u1, g1)) / abs(cost(u1, g1))
            for i in its]
    d.semilogy(its, np.maximum(cold, 1e-16), color=MUTED, ls='--', label='cold start')
    d.semilogy(its, np.maximum(warm, 1e-16), color=COLOR['mpc'], label='warm start')
    d.axvline(60, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    d.set_xlabel('FISTA iterations')
    d.set_ylabel('relative cost gap')
    d.legend(loc='upper right')
    panel_label(d, '(c) solver convergence')
    save(fig, 'fig_mpc_plan.pdf')
    print('   MPC plan: %d of %d inputs on the bound; cost gap after 60 it. cold %.1e, warm %.1e'
          % (int(np.sum(np.abs(u_opt - DS_MAX_CAP) < 1e-9)), N, cold[-1], warm[-1]))


if __name__ == '__main__':
    model = load_model()
    fig_robot(model)
    fig_arena()
    fig_vlwip_lqr(model)
    fig_lipm_preview(model)
    fig_pid_authority(model)
    fig_mpc_plan(model)
