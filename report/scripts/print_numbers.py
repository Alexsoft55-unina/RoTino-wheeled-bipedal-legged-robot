"""Prints every model and design number quoted in the report, computed from the code of this workspace.

    python3 report/scripts/print_numbers.py
"""

import math
import os
import re
import time

import numpy as np

from common import (DS_MAX_CAP, G, LQR_Q, LQR_R, MPC_DT, MPC_HORIZON, MPC_S_H, MPC_S_V, MPC_W_H, MPC_W_V,
                    ROBUST_CASES, WS, load_model, lqr_schedule, lumped, min_damping, pid_cascade,
                    pid_cascade_poles)

np.set_printoptions(precision=4, suppress=True, linewidth=170)


def check_constants():
    """The constants repeated in common.py must match rotino_mpc/controller.py."""
    src = open(os.path.join(WS, 'src', 'rotino_mpc', 'rotino_mpc', 'controller.py')).read()
    expect = {'LQR_Q': 'np.diag([30.0, 400.0, 80.0, 15.0, 6.0, 2.0])', 'LQR_R_COMMON': '2.0', 'LQR_R_DIFF': '100.0',
              'MPC_HORIZON': '25', 'MPC_DT': '0.02', 'MPC_S_H': '(50.0, 20.0)', 'MPC_W_H': '200.0',
              'MPC_S_V': '(5000.0, 150.0)', 'MPC_W_V': '2e-3', 'DS_MAX_CAP': '0.03'}
    for name, value in expect.items():
        found = re.search(rf'^{name} = (.+?)(\s+#.*)?$', src, re.M).group(1).strip()
        assert found == value, (name, found, value)
    print('constants of common.py match rotino_mpc/controller.py')


def model_numbers(m):
    p = m.p
    print('\n=== robot parameters (URDF) ===')
    for k, v in vars(p).items():
        print(f'  {k:>18} = {np.round(v, 6)}')
    print('  total mass =', round(m.kin.total_mass, 4))
    e0 = m.equivalent_centroid(0.0, 0.0)
    print('nominal equivalent centroid:', {k: round(v, 5) for k, v in e0.items()})
    print('   hip    knee    z_b      S_C      Z_C      l     theta[deg]   I_y     m_b l^2/3')
    for hip in (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45):
        e = m.equivalent_centroid(hip, -2 * hip)
        print(f' {hip:+.2f}  {-2 * hip:+.2f}  {e["z_b"]:.4f}  {e["S_C"]:+.4f}  {e["Z_C"]:.4f}  {e["l"]:.4f}  '
              f'{math.degrees(e["theta"]):+6.2f}   {e["I_y"]:.5f}  {p.m_b * e["l"] ** 2 / 3:.5f}')
    frames = m.kin.link_frames(np.array([0, 0, e0['z_b'] + p.r]), np.eye(3), {})
    com = m.kin.center_of_mass(frames)
    print('whole-robot CoM in the nominal pose: height', round(com[2], 4), 'm,', round(1e3 * com[0], 2),
          'mm ahead of the axle; omega =', round(math.sqrt(G / com[2]), 3), 'rad/s')
    for label, iy in (('I_y = m_b l^2/3 (Table 1 of the paper)', None), ('I_y of the URDF (used by the code)', e0['I_y'])):
        c = m.vlwip_coefficients(e0['l'], I_y=iy)
        print(f'  eq. (14), {label}: ' + ', '.join(f'{k}={v:+.3f}' for k, v in c.items()))
    c = m.vlwip_coefficients(e0['l'], I_y=e0['I_y'])
    print('unstable pole sqrt(a2) =', round(math.sqrt(c['a2']), 3), 'rad/s; right-half-plane zero of torque -> s =',
          round(math.sqrt((c['a2'] * c['b1'] - c['a1'] * c['b2']) / c['b1']), 3), 'rad/s')
    A, B = m.vlwip_matrices(e0['l'], I_y=e0['I_y'])
    print('controllability rank:', np.linalg.matrix_rank(np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(6)])))
    _, J = m.leg_fk(0.0, 0.0)
    for name, Kp, Kd in (('PID stance', (1500.0, 5000.0), (40.0, 80.0)), ('VMC stance', (1500.0, 0.0), (40.0, 15.0))):
        print(f'{name}: joint stiffness diag(J^T Kp J) = {np.diag(J.T @ np.diag(Kp) @ J)} Nm/rad, '
              f'damping {np.diag(J.T @ np.diag(Kd) @ J)} Nm s/rad')
    return e0


def lqr_numbers(m, e0):
    print('\n=== TV-LQR ===')
    sched, samples = lqr_schedule(m)
    K = sched.gain(e0['l'])
    print('R =\n', LQR_R, '\nK(l_nom) =\n', K)
    print('pendulum length grid:', round(sched.l_grid[0], 4), '..', round(sched.l_grid[-1], 4), 'm')
    sag = [0, 1, 3, 4]
    print('   l      K_s     K_th    K_sd   K_thd | K_phi K_phid | sagittal poles | yaw poles')
    for i in (0, 6, 12, 18, 24):
        l, iy = samples[i]
        A, B = m.vlwip_matrices(l, I_y=iy)
        Kl = sched.gain(l)
        kc, kd = 0.5 * (Kl[0] + Kl[1]), 0.5 * (Kl[1] - Kl[0])
        es = np.sort_complex(np.linalg.eigvals(A[np.ix_(sag, sag)] - np.outer(B[sag] @ np.ones(2), kc[sag])))
        ey = np.sort_complex(np.linalg.eigvals(A[np.ix_([2, 5], [2, 5])]
                                               - np.outer(B[[2, 5]] @ np.array([-1.0, 1.0]), kd[[2, 5]])))
        print(f' {l:.4f} {kc[0]:7.3f} {kc[1]:7.3f} {kc[3]:7.3f} {kc[4]:7.3f} | {kd[2]:5.3f} {kd[5]:5.3f} | '
              f'{np.round(es, 2)} | {np.round(ey, 2)}')
    Kc = 0.5 * (K[0] + K[1])
    for i in (0, 24):
        l, iy = samples[i]
        A, B = m.vlwip_matrices(l, I_y=iy)
        es = np.sort_complex(np.linalg.eigvals(A[np.ix_(sag, sag)] - np.outer(B[sag] @ np.ones(2), Kc[sag])))
        print(f'   fixed nominal gain at l = {l:.3f} m: sagittal poles {np.round(es, 2)}')


def pid_numbers(m):
    from rotino_pid.zmp_balance import SagittalGains, ZmpSagittalBalance
    print('\n=== PID with ZMP ===')
    bal = ZmpSagittalBalance.from_model(m)
    g = SagittalGains()
    c, k_sc, h = lumped(m)
    w = math.sqrt(G / h)
    print('feedforward: ZMP offset', round(1e3 * bal.lean_per_acc, 2), 'mm per m/s^2 (LIPM h/g =', round(1e3 * h / G, 2),
          '), torque', round(bal.torque_per_acc, 4), 'Nm per wheel per m/s^2')
    print('h =', round(h, 4), 'm, omega =', round(w, 3), 'rad/s, 1/omega =', round(1 / w, 3), 's, largest offset a_max/omega^2 =',
          round(1e3 * g.acc_max / w ** 2, 1), 'mm')
    A, B, k = pid_cascade(m)
    print('closed-loop poles, continuous:', np.round(np.sort_complex(np.linalg.eigvals(A + np.outer(B, k))), 2))
    for hip, ms, d in ROBUST_CASES:
        sv = pid_cascade_poles(m, hip, ms, d)
        dom = np.sort_complex(sv[np.abs(sv) < 100])
        print(f'   hip {hip:+.1f}, mass x{ms:.1f}, delay {d}: dominant poles {np.round(dom, 2)}, damping {min_damping(dom):.2f};'
              f' all poles: damping {min_damping(sv):.2f}')
    print('largest inner stiffness K_s keeping the ten cases stable, versus the authority rho:')
    for rho in (0.2, 0.4, 0.6, 0.8, 1.0):
        lo, hi = 10.0, 2000.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            stable = all(pid_cascade_poles(m, hip, ms, d, rho, SagittalGains(k_s=mid)).real.max() < 0
                         for hip, ms, d in ROBUST_CASES)
            lo, hi = (mid, hi) if stable else (lo, mid)
        dom = pid_cascade_poles(m, 0.0, 1.0, 2, rho)
        dom = dom[np.abs(dom) < 100]
        print(f'   rho = {rho:.1f}: K_s < {lo:6.1f} Nm/m (margin {lo / g.k_s:.1f} on the design value);'
              f' dominant damping at K_s = {g.k_s:.0f}: {min_damping(dom):.2f}')
    # steady state under a constant force F on the torso (step scenario): the integral sits on its bound
    s_star = 0.01734                       # measured CoM offset holding 3 N (m), from the log
    s_i = g.authority * g.k_dcm_i * g.dcm_i_max / w
    print('step force: integral bound contributes', round(1e3 * s_i, 2), 'mm of ZMP offset; capture-point error for',
          round(1e3 * s_star, 2), 'mm:', round((0.0159 - s_i) * w / (g.authority * g.k_dcm), 4), 'm')


def mpc_numbers(m, e0):
    from rotino_mpc.solvers import AxisKalman, UpperBodyMPC, _condense, box_qp
    print('\n=== upper-body MPC ===')
    p = m.p
    N, dt, z = MPC_HORIZON, MPC_DT, e0['Z_C']
    W = p.m_b * G
    print('g/h =', round(G / z, 1), '1/s^2; bounds on delta_s: mu h =', round(p.mu * z, 3), 'm, leg workspace =',
          round(math.sqrt(p.L_max ** 2 - e0['z_b'] ** 2), 3), 'm, cap =', DS_MAX_CAP, 'm ->', round(math.degrees(math.atan2(DS_MAX_CAP, z)), 2),
          'deg,', round(G / z * DS_MAX_CAP, 2), 'm/s^2')
    print('F_z bounds:', round(0.3 * W, 1), '..', round(3 * W, 1), 'N (', round(6 * W, 1), 'N in the thrust); m_b g =', round(W, 2))
    Ah = np.array([[1.0, dt], [0.0, 1.0]])
    Bh = np.array([[0.5 * dt * dt], [dt]]) * G / z
    Phi, Gam, Cst = _condense(Ah, Bh, np.zeros(2), N)
    Sb = np.kron(np.eye(N), np.diag(MPC_S_H))
    H = Gam.T @ Sb @ Gam + MPC_W_H * np.eye(N)
    ev = np.linalg.eigvalsh(H)
    Bv = np.array([[0.5 * dt * dt], [dt]]) / p.m_b
    cv = np.array([-0.5 * dt * dt * G, -dt * G])
    Phiv, Gamv, Cstv = _condense(Ah, Bv, cv, N)
    Hv = Gamv.T @ np.kron(np.eye(N), np.diag(MPC_S_V)) @ Gamv + MPC_W_V * np.eye(N)
    evv = np.linalg.eigvalsh(Hv)
    print('condition number of the Hessian: horizontal', round(ev[-1] / ev[0], 1), ', vertical', round(evv[-1] / evv[0], 0))
    ref = np.tile([0.3, 0.0], (N, 1))
    g_ = Gam.T @ Sb @ (Cst - ref.reshape(-1))
    exact = box_qp(H, g_, -DS_MAX_CAP, DS_MAX_CAP, None, iters=100000)
    cost = lambda u: 0.5 * u @ H @ u + g_ @ u
    for it in (10, 20, 60):
        u = box_qp(H, g_, -DS_MAX_CAP, DS_MAX_CAP, None, iters=it)
        print(f'   FISTA, cold start, {it:3d} iterations: max error {1e3 * np.abs(u - exact).max():.3f} mm, '
              f'cost gap {(cost(u) - cost(exact)) / abs(cost(exact)):.1e}')
    mpc = UpperBodyMPC(p.m_b, N, dt, MPC_S_H, MPC_W_H, MPC_S_V, MPC_W_V)
    refv = np.tile([z, 0.0], (N, 1))
    t0 = time.perf_counter()
    for _ in range(200):
        mpc.solve([0.0, 0.0], ref, z, 0.0, DS_MAX_CAP, [z, 0.0], refv, 0.3 * W, 3 * W)
    print('   one MPC update (two QPs): %.2f ms on this computer' % ((time.perf_counter() - t0) / 200 * 1e3))

    print('\n=== Kalman filter, steady state at 500 Hz ===')
    kf = AxisKalman(0.5, 1e-4, 1e-3)
    for _ in range(20000):
        kf.predict(0.0, 0.002)
        Kg = kf.P @ np.linalg.inv(kf.P + kf.R_nom)
        kf.correct(0.0, 0.0)
    zz = np.linalg.eigvals((np.eye(2) - Kg) @ np.array([[1.0, 0.002], [0.0, 1.0]]))
    print('gain =\n', Kg, '\nestimator poles [rad/s]:', np.round(np.log(zz.astype(complex)) / 0.002, 2))


if __name__ == '__main__':
    check_constants()
    model = load_model()
    e0 = model_numbers(model)
    lqr_numbers(model, e0)
    pid_numbers(model)
    mpc_numbers(model, e0)
