import math

import numpy as np


class CubicSTrajectory:
    """Planar S-curve y(x) = Y (3 (x/X)^2 - 2 (x/X)^3) with a quintic arc-length time law.

    The path starts and ends parallel to the local x axis; X is solved so the arc length equals `length`.
    Coordinates are in the robot frame at release (x forward, y left).
    """

    def __init__(self, length, lateral, duration, samples=4000):
        self.length = length
        self.lateral = lateral
        self.duration = duration
        self.X = self._solve_X(samples)
        self.x_table = np.linspace(0.0, self.X, samples)
        dy = self._dy(self.x_table)
        ds = np.sqrt(1.0 + dy * dy)
        self.s_table = np.concatenate(([0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(self.x_table))))

    def _y(self, x):
        u = x / self.X
        return self.lateral * (3.0 * u ** 2 - 2.0 * u ** 3)

    def _dy(self, x):
        u = x / self.X
        return self.lateral * (6.0 * u - 6.0 * u ** 2) / self.X

    def _ddy(self, x):
        u = x / self.X
        return self.lateral * (6.0 - 12.0 * u) / self.X ** 2

    def _arc_length(self, X, samples):
        self.X = X
        x = np.linspace(0.0, X, samples)
        return float(np.trapz(np.sqrt(1.0 + self._dy(x) ** 2), x))

    def _solve_X(self, samples):
        lo, hi = 1e-3, self.length
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._arc_length(mid, samples) < self.length:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _time_law(self, t):
        tau = min(max(t / self.duration, 0.0), 1.0)
        s = self.length * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        v = self.length * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.duration
        return s, v

    def sample(self, t):
        """Return (x, y, heading, speed, yaw_rate) at time t (clamped to [0, duration])."""
        s, v = self._time_law(t)
        x = float(np.interp(s, self.s_table, self.x_table))
        dy = self._dy(x)
        heading = math.atan(dy)
        curvature = self._ddy(x) / (1.0 + dy * dy) ** 1.5
        return x, float(self._y(x)), heading, v, curvature * v
