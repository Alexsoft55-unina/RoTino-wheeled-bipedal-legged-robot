"""
Numerical blocks of the WBR controller (Cui et al., Micromachines 2022, 13, 747), NumPy only:
  - care / lqr_gain:   continuous LQR for the TV-LQR of the VL-WIP (eqs. 15-16)
  - LQRSchedule:       K(l) precomputed on a grid of pendulum lengths and interpolated
  - box_qp:            accelerated projected gradient for box-constrained QPs
  - UpperBodyMPC:      condensed MPC of the lumped-mass upper body (eqs. 9, 17-18)
  - AxisKalman:        linear Kalman filter of eqs. 20-21, one instance per world axis
"""

import numpy as np

G = 9.81


def care(A, B, Q, R):
    """Stabilizing solution of A'P + PA - PBR^-1B'P + Q = 0 from the stable invariant subspace of the Hamiltonian."""
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T], [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    stable = V[:, w.real < 0.0]
    if stable.shape[1] != n:
        raise RuntimeError('Hamiltonian has eigenvalues on the imaginary axis: (A, B) not stabilizable or Q too small')
    P = np.real(stable[n:, :] @ np.linalg.inv(stable[:n, :]))
    return 0.5 * (P + P.T)


def lqr_gain(A, B, Q, R):
    P = care(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class LQRSchedule:
    """TV-LQR: K(l) on a grid of (l, I_y) samples, linearly interpolated in l."""

    def __init__(self, model, samples, Q, R):
        samples = sorted(samples)
        self.l_grid = np.array([s[0] for s in samples])
        self.K_grid = np.array([lqr_gain(*model.vlwip_matrices(l, I_y=iy), Q, R) for l, iy in samples])

    def gain(self, l):
        l = float(np.clip(l, self.l_grid[0], self.l_grid[-1]))
        i = int(np.clip(np.searchsorted(self.l_grid, l) - 1, 0, len(self.l_grid) - 2))
        a = (l - self.l_grid[i]) / (self.l_grid[i + 1] - self.l_grid[i])
        return (1.0 - a) * self.K_grid[i] + a * self.K_grid[i + 1]


def box_qp(H, g, lo, hi, x0=None, iters=60):
    """min 0.5 x'Hx + g'x  s.t. lo <= x <= hi  (FISTA; H is small and dense)."""
    L = float(np.linalg.eigvalsh(H)[-1])
    x = np.clip(np.zeros_like(g) if x0 is None else x0, lo, hi)
    y, t = x.copy(), 1.0
    for _ in range(iters):
        x_new = np.clip(y - (H @ y + g) / L, lo, hi)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def _condense(A, B, c, N):
    """x_i = Phi_i x0 + Gamma_i U + C_i for i = 1..N (stacked)."""
    n, m = B.shape
    Phi = np.zeros((N * n, n))
    Gam = np.zeros((N * n, N * m))
    Cst = np.zeros(N * n)
    Ak = np.eye(n)
    for i in range(N):
        Ak = A @ Ak
        Phi[i * n:(i + 1) * n] = Ak
        row = i * n
        prev = Gam[row - n:row] if i > 0 else np.zeros((n, N * m))
        Gam[row:row + n] = A @ prev
        Gam[row:row + n, i * m:(i + 1) * m] = B
        Cst[row:row + n] = (A @ Cst[row - n:row] if i > 0 else 0.0) + c
    return Phi, Gam, Cst


class UpperBodyMPC:
    """Eq. (9) split into its two decoupled blocks, each a box-constrained condensed QP (eqs. 17-18).

    horizontal: [s, s_dot],  s_ddot = (g + z_ddot) / h * delta_s,   |delta_s| <= min(mu h, sqrt(L_max^2 - z_b^2))
    vertical:   [z, z_dot],  z_ddot = F_z / m_b - g,                F_min <= F_z <= F_max
    """

    def __init__(self, m_b, horizon, dt, S_h, W_h, S_v, W_v):
        self.m_b, self.N, self.dt = m_b, horizon, dt
        self.S_h, self.W_h, self.S_v, self.W_v = np.diag(S_h), W_h, np.diag(S_v), W_v
        self.Ah = np.array([[1.0, dt], [0.0, 1.0]])
        self.u_h = np.zeros(horizon)
        self.u_v = np.full(horizon, m_b * G)
        self.x_h = None   # predicted [s, s_dot] at t + dt, t + 2 dt, ...
        self.x_v = None

    def _solve(self, A, B, c, x0, x_ref, S, W, lo, hi, warm):
        Phi, Gam, Cst = _condense(A, B, c, self.N)
        Sb = np.kron(np.eye(self.N), S)
        H = Gam.T @ Sb @ Gam + W * np.eye(self.N)
        g = Gam.T @ Sb @ (Phi @ x0 + Cst - x_ref.reshape(-1))
        warm = np.clip(np.concatenate((warm[1:], warm[-1:])), lo, hi)
        u = box_qp(H, g, lo, hi, warm)
        return u, (Phi @ x0 + Gam @ u + Cst).reshape(self.N, -1)

    def solve(self, x_h, ref_h, h, zdd_ref, ds_max, x_v, ref_v, f_min, f_max):
        """x_h = [s, s_dot], ref_h (N x 2); x_v = [z, z_dot], ref_v (N x 2). Returns (delta_s, F_z) to apply now."""
        dt = self.dt
        Bh = np.array([[0.5 * dt * dt], [dt]]) * (G + zdd_ref) / max(h, 1e-3)
        self.u_h, self.x_h = self._solve(self.Ah, Bh, np.zeros(2), np.asarray(x_h), ref_h, self.S_h, self.W_h,
                               -ds_max, ds_max, self.u_h)
        Bv = np.array([[0.5 * dt * dt], [dt]]) / self.m_b
        cv = np.array([-0.5 * dt * dt * G, -dt * G])
        self.u_v, self.x_v = self._solve(self.Ah, Bv, cv, np.asarray(x_v), ref_v, self.S_v, self.W_v,
                               f_min, f_max, self.u_v)
        return float(self.u_h[0]), float(self.u_v[0])

    def reset(self):
        self.x_h = self.x_v = None
        self.u_h[:] = 0.0
        self.u_v[:] = self.m_b * G


class AxisKalman:
    """Eqs. (20)-(21) for one world axis: state [p, v] of the torso, input IMU acceleration,
    observation [p, v] from wheel odometry + leg kinematics."""

    def __init__(self, q_acc, r_pos, r_vel):
        self.q_acc = q_acc
        self.R_nom = np.diag([r_pos, r_vel])
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1e-4

    def reset(self, p, v):
        self.x = np.array([p, v], dtype=float)
        self.P = np.eye(2) * 1e-4

    def predict(self, acc, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Bu = np.array([0.5 * dt * dt, dt])
        self.x = F @ self.x + Bu * acc
        self.P = F @ self.P @ F.T + self.q_acc * np.outer(Bu, Bu) + 1e-9 * np.eye(2)

    def correct(self, p_obs, v_obs, noise_scale=1.0):
        R = self.R_nom * noise_scale
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([p_obs, v_obs]) - self.x)
        self.P = (np.eye(2) - K) @ self.P
