"""
Decoupled WBR model of Cui et al., "Modeling and Control of a Wheeled Biped Robot",
Micromachines 2022, 13, 747, with every parameter of its Table 1 computed from the RoTino URDF.

  - VL-WIP (Sec. 3.2, eqs. 4-5, 12-14): wheels + variable-length pendulum carrying the upper body m_b
  - equivalent centroid (Sec. 3.1, eqs. 2-3): l, theta of the upper-body CoM in the axle frame
  - lumped-mass upper body (Sec. 3.3, eq. 9) for the MPC

Run `ros2 run rotino_description wbr_model` to print the parameter table.
"""

import math
import subprocess
from dataclasses import dataclass

import numpy as np
from urdf_parser_py.urdf import URDF

from rotino_description.kinematics import RobotKinematics, rpy_to_matrix

G = 9.81
LEFT_WHEEL_LINK = 'left_wheel_link'
RIGHT_WHEEL_LINK = 'right_wheel_link'
WHEEL_LINKS = (LEFT_WHEEL_LINK, RIGHT_WHEEL_LINK)
TORSO_LINKS = ('base_link', 'ballast_link')


@dataclass
class WBRParams:
    m_w: float      # mass of one wheel [kg]
    I_w: float      # wheel inertia about its spin axis [kg m^2]
    r: float        # wheel radius [m]
    d: float        # distance between the wheels [m]
    m_b: float      # upper body mass = total - 2 m_w [kg]
    m_1: float      # shank mass (one leg) [kg]
    m_2: float      # thigh mass (one leg) [kg]
    m_3: float      # torso mass (base + ballast) [kg]
    l_1: float      # shank length, knee -> wheel axle [m]
    l_2: float      # thigh length, hip -> knee [m]
    l_3: float      # torso height (box) [m]
    torso_com: np.ndarray  # torso CoM in the hip frame (base_link) [m]
    L_max: float    # geometric leg length l_1 + l_2 [m]
    mu: float       # wheel/ground friction coefficient
    wheel_torque_max: float  # [Nm]
    leg_torque_max: float    # [Nm]
    hip_limit: float         # [rad]
    knee_limit: float        # [rad]
    leg_damping: float       # hip/knee viscous damping [Nm s/rad]


def _joint(robot, name):
    return next(j for j in robot.joints if j.name == name)


def _link(robot, name):
    return next(link for link in robot.links if link.name == name)


def _wheel_radius(robot):
    for c in _link(robot, LEFT_WHEEL_LINK).collisions:
        if hasattr(c.geometry, 'radius'):
            return float(c.geometry.radius)
    raise RuntimeError('wheel collision cylinder not found')


def _mu(urdf_xml):
    """<mu1> of the wheel gazebo reference (urdf_parser_py drops <gazebo> tags)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(urdf_xml)
    for gz in root.findall('gazebo'):
        if gz.get('reference') == LEFT_WHEEL_LINK and gz.find('mu1') is not None:
            return float(gz.find('mu1').text)
    return float('nan')


class WBRModel:

    def __init__(self, urdf_xml):
        self.robot = URDF.from_xml_string(urdf_xml)
        self.kin = RobotKinematics(urdf_xml)
        rb = self.robot

        m = {link.name: link.inertial.mass for link in rb.links if link.inertial is not None}
        knee = _joint(rb, 'left_knee')
        wheel = _joint(rb, 'left_wheel_joint')
        hip = _joint(rb, 'left_hip')
        torso_box = next(v.geometry.size for v in _link(rb, 'base_link').visuals if hasattr(v.geometry, 'size'))

        self.hip_xz = np.array([hip.origin.xyz[0], hip.origin.xyz[2]])
        self.knee_xz = (knee.origin.xyz[0], knee.origin.xyz[2])
        self.wheel_xz = (wheel.origin.xyz[0], wheel.origin.xyz[2])

        torso_mass = sum(m[n] for n in TORSO_LINKS)
        frames0 = self.kin.link_frames(np.zeros(3), np.eye(3), {})
        torso_com = sum(m[n] * self._link_com(frames0, n) for n in TORSO_LINKS) / torso_mass

        self.p = WBRParams(
            m_w=m[LEFT_WHEEL_LINK],
            I_w=_link(rb, LEFT_WHEEL_LINK).inertial.inertia.iyy,
            r=_wheel_radius(rb),
            d=float(np.linalg.norm(frames0[LEFT_WHEEL_LINK][1] - frames0[RIGHT_WHEEL_LINK][1])),
            m_b=self.kin.total_mass - sum(m[n] for n in WHEEL_LINKS),
            m_1=m['left_shank_link'],
            m_2=m['left_thigh_link'],
            m_3=torso_mass,
            l_1=float(np.linalg.norm([wheel.origin.xyz[0], wheel.origin.xyz[2]])),
            l_2=float(np.linalg.norm([knee.origin.xyz[0], knee.origin.xyz[2]])),
            l_3=float(torso_box[2]),
            torso_com=torso_com,
            L_max=float(np.linalg.norm([knee.origin.xyz[0], knee.origin.xyz[2]])
                        + np.linalg.norm([wheel.origin.xyz[0], wheel.origin.xyz[2]])),
            mu=_mu(urdf_xml),
            wheel_torque_max=float(wheel.limit.effort),
            leg_torque_max=float(hip.limit.effort),
            hip_limit=float(hip.limit.upper),
            knee_limit=float(knee.limit.upper),
            leg_damping=float(hip.dynamics.damping) if hip.dynamics is not None else 0.0,
        )

    # ------------------------------------------------------------------
    def _link_com(self, frames, name):
        link = _link(self.robot, name)
        origin = link.inertial.origin
        c = np.array(origin.xyz) if origin is not None and origin.xyz else np.zeros(3)
        rot, pos = frames[name]
        return pos + rot @ c

    def _link_inertia_world(self, frames, name):
        link = _link(self.robot, name)
        i = link.inertial.inertia
        I_local = np.array([[i.ixx, i.ixy, i.ixz], [i.ixy, i.iyy, i.iyz], [i.ixz, i.iyz, i.izz]])
        origin = link.inertial.origin
        rpy = origin.rpy if origin is not None and origin.rpy else [0.0, 0.0, 0.0]
        R = frames[name][0] @ rpy_to_matrix(*rpy)
        return R @ I_local @ R.T

    def upper_body_com_base(self, joint_positions):
        """Upper-body CoM (everything but the wheels) in base_link for the given joint angles."""
        frames = self.kin.link_frames(np.zeros(3), np.eye(3), joint_positions)
        weighted = np.zeros(3)
        for name, mass, com_local in self.kin.masses:
            if name not in WHEEL_LINKS:
                rot, pos = frames[name]
                weighted += mass * (pos + rot @ com_local)
        return weighted / self.p.m_b

    @staticmethod
    def leg_joints(hip, knee):
        return {'left_hip': hip, 'right_hip': hip, 'left_knee': knee, 'right_knee': knee}

    def equivalent_centroid(self, hip, knee, pitch=0.0):
        """Eqs. (2)-(3): upper-body CoM in the axle frame -> (S_C, Z_C, l, theta, I_y, I_z, hip height).

        I_y / I_z are the upper-body inertias about its own CoM (pitch / yaw axes); hip height is
        the vertical hip-to-axle distance z_b.
        """
        R = rpy_to_matrix(0.0, pitch, 0.0)
        frames = self.kin.link_frames(np.zeros(3), R, self.leg_joints(hip, knee))
        axle = 0.5 * (frames[LEFT_WHEEL_LINK][1] + frames[RIGHT_WHEEL_LINK][1])

        names = [link.name for link in self.robot.links
                 if link.inertial is not None and link.name not in WHEEL_LINKS]
        masses = np.array([_link(self.robot, n).inertial.mass for n in names])
        coms = np.array([self._link_com(frames, n) for n in names])
        com = masses @ coms / masses.sum()

        I = np.zeros((3, 3))
        for mass, c, n in zip(masses, coms, names):
            dv = c - com
            I += self._link_inertia_world(frames, n) + mass * (dv @ dv * np.eye(3) - np.outer(dv, dv))

        S_C, Z_C = float(com[0] - axle[0]), float(com[2] - axle[2])
        return {
            'S_C': S_C, 'Z_C': Z_C,
            'l': math.hypot(S_C, Z_C), 'theta': math.atan2(S_C, Z_C),
            'I_y': float(I[1, 1]), 'I_z': float(I[2, 2]),
            'z_b': float(-axle[2]), 'axle_x': float(axle[0]),
        }

    def vlwip_coefficients(self, l, I_y=None, I_z=None):
        """Eq. (14). Defaults follow Table 1: I_y = m_b l^2 / 3, I_z from the nominal pose."""
        p = self.p
        if I_y is None:
            I_y = p.m_b * l * l / 3.0
        if I_z is None:
            I_z = self.equivalent_centroid(0.0, 0.0)['I_z']
        mb, mw, Iw, r, d = p.m_b, p.m_w, p.I_w, p.r, p.d
        den = 2 * Iw * (I_y + mb * l * l) + (2 * l * l * mb * mw + I_y * (mb + 2 * mw)) * r * r
        return {
            'a1': -G * l * l * mb * mb * r * r / den,
            'a2': G * l * mb * (2 * Iw + (mb + 2 * mw) * r * r) / den,
            'b1': r * (I_y + l * mb * (l + r)) / den,
            'b2': -(2 * Iw + r * (l * mb + (mb + 2 * mw) * r)) / den,
            'b3': d * r / (2 * I_z * r * r + d * d * (Iw + mw * r * r)),
        }

    def vlwip_matrices(self, l, I_y=None, I_z=None):
        """Eq. (13): X = [s, theta, phi, s_dot, theta_dot, phi_dot], U = [tau_l, tau_r]."""
        c = self.vlwip_coefficients(l, I_y, I_z)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 1] = c['a1']
        A[4, 1] = c['a2']
        B = np.zeros((6, 2))
        B[3] = [c['b1'], c['b1']]
        B[4] = [c['b2'], c['b2']]
        # Left wheel on +y (REP 103): a forward left torque turns the robot clockwise, the opposite of Fig. 2a.
        B[5] = [-c['b3'], c['b3']]
        return A, B

    def leg_fk(self, hip, knee):
        """Sagittal wheel-centre position (x, z) in base_link and its 2x2 Jacobian w.r.t. (hip, knee)."""
        kx, kz = self.knee_xz
        wx, wz = self.wheel_xz

        def rot(q, x, z):
            c, s = math.cos(q), math.sin(q)
            return np.array([x * c + z * s, -x * s + z * c]), np.array([-x * s + z * c, -x * c - z * s])

        pk, dpk = rot(hip, kx, kz)
        pw, dpw = rot(hip + knee, wx, wz)
        J = np.column_stack((dpk + dpw, dpw))
        return self.hip_xz + pk + pw, J

    def upper_body_matrices(self, h, zdd=0.0):
        """Eq. (9): x = [s, s_dot, z, z_dot, -g], u = [delta_s, F_z]."""
        A = np.zeros((5, 5))
        A[0, 1] = A[2, 3] = A[3, 4] = 1.0
        B = np.zeros((5, 2))
        B[1, 0] = (G + zdd) / h
        B[3, 1] = 1.0 / self.p.m_b
        return A, B


def load_urdf_from_package():
    share = subprocess.check_output(['ros2', 'pkg', 'prefix', 'rotino_description']).decode().strip()
    return subprocess.check_output(['xacro', f'{share}/share/rotino_description/urdf/rotino.urdf.xacro']).decode()


def main():
    model = WBRModel(load_urdf_from_package())
    p = model.p
    print('Table 1 - RoTino')
    for k, v in vars(p).items():
        print(f'  {k:>18} = {np.round(v, 5)}')

    print('\nEquivalent centroid (hip = -knee/2 keeps the axle under the hip, torso level)')
    print('   hip    knee    z_b      S_C      Z_C      l      theta    I_y(real)  I_y(m_b l^2/3)')
    for hip in (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45):
        e = model.equivalent_centroid(hip, -2 * hip)
        print(f' {hip:+.2f}  {-2 * hip:+.2f}  {e["z_b"]:.4f}  {e["S_C"]:+.4f}  {e["Z_C"]:.4f}  {e["l"]:.4f}'
              f'  {math.degrees(e["theta"]):+.2f}deg  {e["I_y"]:.5f}    {p.m_b * e["l"] ** 2 / 3:.5f}')

    e0 = model.equivalent_centroid(0.0, 0.0)
    print(f'\nI_z (upper body, nominal pose) = {e0["I_z"]:.5f} kg m^2')
    for label, iy in (('Table 1 I_y = m_b l^2/3', None), ('URDF I_y', e0['I_y'])):
        c = model.vlwip_coefficients(e0['l'], I_y=iy)
        print(f'Eq. 14 at l = {e0["l"]:.4f} m ({label}): ' + ', '.join(f'{k}={v:+.3f}' for k, v in c.items()))


if __name__ == '__main__':
    main()
