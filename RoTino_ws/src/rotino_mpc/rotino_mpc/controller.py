"""
RoTino decoupled WBR controller (Cui et al., "Modeling and Control of a Wheeled Biped Robot",
Micromachines 2022, 13, 747), replacing the PID balance/jump controller of the PID workspace.

  state estimator (Sec. 4.3)   IMU + wheel odometry + leg kinematics -> linear Kalman filter, 500 Hz
  VL-WIP TV-LQR (Sec. 4.1)     X = [s, theta, phi, s_dot, theta_dot, phi_dot] -> wheel torques, 500 Hz
  upper-body MPC (Sec. 4.2)    [s, s_dot, z, z_dot] -> CoM offset delta_s and vertical force F_z, 100 Hz
  task-space VMC (eq. 19)      delta_s, z_ref, F_z -> hip/knee torques, 500 Hz

Only proprioceptive/IMU data drive the controller; /rotino/odom (ground truth) is used only to publish
the estimation error.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts, Entity, EntityWrench
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Empty, Float64, Float64MultiArray, String

from rotino_description.kinematics import quat_to_matrix
from rotino_description.planar_trajectory import CubicSTrajectory
from rotino_description.model import G, WBRModel
from rotino_mpc.solvers import AxisKalman, LQRSchedule, UpperBodyMPC

LEG_JOINTS = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
WHEEL_JOINTS = ['left_wheel_joint', 'right_wheel_joint']

HOLD_TIME = 1.5
RELEASE_REPEAT_TIME = 0.05
MOTION_START_TIME = 2.0      # reference motions start this long after release
# wait_start: the motions start this long after /rotino/cmd_start instead, so that the preview of the PID
# (about 1.1 s ahead) and the horizon of the MPC (0.5 s) see them coming exactly as after a release
START_LEAD = 1.0
PHYSICS_DT = 0.0005          # rotino_world.sdf max_step_size (push impulse)
PUSH_TICKS = 25

# ---------------- TV-LQR (eq. 15) ----------------
LQR_Q = np.diag([30.0, 400.0, 80.0, 15.0, 6.0, 2.0])   # s, theta, phi, s_dot, theta_dot, phi_dot
# R weighs common-mode (balance) and differential (yaw) wheel torque separately. The differential weight keeps
# the yaw loop (crossover ~16 rad/s) well below the torsional resonance of the stance legs (~76 rad/s, 12 Hz:
# torso yaw inertia on the VMC springs), which a high yaw gain excites into a permanent limit cycle.
LQR_R_COMMON = 2.0           # equivalent to R = I for the sagittal gains
LQR_R_DIFF = 100.0
_T_CD = np.array([[0.5, 0.5], [-0.5, 0.5]])   # [common, diff] = T @ [tau_l, tau_r]
LQR_R = _T_CD.T @ np.diag([LQR_R_COMMON, LQR_R_DIFF]) @ _T_CD
WHEEL_TORQUE_MAX = 10.0
DIFF_TORQUE_MAX = 2.5        # Nm per wheel for yaw: balance (common mode) keeps priority
# wheel scrub while turning: Coulomb + viscous feedforward from the commanded yaw rate (identified in Gazebo)
# and a slow integral on the heading error; both act far below the leg torsional resonance
YAW_FRICTION = 0.25          # Nm per wheel
YAW_VISCOUS = 0.10           # Nm per wheel per rad/s
YAW_FRICTION_SMOOTH = 0.01   # rad/s (acts on the reference only: no chattering)
YAW_KI = 0.6                 # Nm per wheel per rad s
YAW_I_MAX = 0.6              # Nm per wheel
YAW_I_DEADBAND = 0.035       # rad: no integration on small static errors (avoids stick-slip hunting)
YAW_ERROR_MAX = 0.35         # rad; larger yaw errors are clamped (turn-rate commands the wheels cannot follow)
THETA_DOT_FILTER = 0.8        # first-order filters, weight of the previous value at 500 Hz
S_DOT_FILTER = 0.8
DELTA_S_FILTER = 0.9

# ---------------- MPC (eqs. 17-18) ----------------
MPC_EVERY = 5                # 500 Hz / 5 = 100 Hz, as in Fig. 4
MPC_HORIZON = 25
MPC_DT = 0.02
MPC_S_H = (50.0, 20.0)       # s, s_dot
MPC_W_H = 200.0
MPC_S_V = (5000.0, 150.0)    # z, z_dot
MPC_W_V = 2e-3
DS_MAX_CAP = 0.03
F_MIN_RATIO = 0.3
F_MAX_RATIO = 3.0
F_MAX_THRUST_RATIO = 6.0

# ---------------- VMC (eq. 19), per leg, base frame (x, z) ----------------
VMC_KP_HOLD = np.diag([1500.0, 2000.0])
VMC_KD_HOLD = np.diag([40.0, 50.0])
# stance: horizontal spring realizes delta_s; vertically only damping, the support force is F_z from the MPC
VMC_KP = np.diag([1500.0, 0.0])
VMC_KD = np.diag([40.0, 15.0])
VMC_KP_FLIGHT = np.diag([800.0, 800.0])
VMC_KD_FLIGHT = np.diag([20.0, 20.0])
LEG_TORQUE_MAX = 50.0
FLIGHT_WHEEL_DAMPING = 0.05

# ---------------- Kalman filter (eqs. 20-21) ----------------
KF_Q_ACC = 0.5
KF_R_POS = 1e-4
KF_R_VEL = 1e-3

# ---------------- contact / jump ----------------
F_CONTACT_OFF = 1.5
F_CONTACT_ON = 6.0
CONTACT_LOSS_DEBOUNCE = 0.004
CONTACT_GAIN_DEBOUNCE = 0.010
CONTACT_STALE_TIME = 0.006

HIP_AXLE_LOW = 0.12          # squat hip-axle distance [m]
HIP_AXLE_HIGH = 0.24         # thrust target [m]
HIP_AXLE_FLIGHT = 0.17
T_SQUAT = 0.8
T_THRUST_MAX = 0.30
T_LANDING = 1.0
T_SETTLE = 1.5

PHASE_HOLD = 'HOLD'
PHASE_BALANCE = 'BALANCE'
PHASE_SQUAT = 'PRELOAD'
PHASE_THRUST = 'THRUST'
PHASE_FLIGHT = 'FLIGHT'
PHASE_LANDING = 'LANDING'
STANCE_PHASES = (PHASE_BALANCE, PHASE_SQUAT, PHASE_THRUST, PHASE_LANDING)

PARAMS = {
    'mpc_enable': True,
    'jump_enable': False, 'jump_start_time': 4.0, 'jump_velocity': 1.2,
    'height_enable': False, 'height_amplitude': 0.03, 'height_period': 2.2,
    'velocity_enable': False, 'velocity_max': 0.44, 'accel_max': 0.6, 'velocity_distance': 2.0,
    'drive_enable': False, 'drive_distance': 1.0, 'drive_period': 8.0,
    'planar_enable': False, 'traj_length': 3.0, 'traj_lateral': 0.6, 'traj_duration': 12.0,
    'push_enable': False, 'push_time': 4.0, 'push_impulse': 2.7,
    'wait_start': False,    # hold the scripted motions until /rotino/cmd_start
}

K_LAT = 2.0
LAT_FULL_SPEED = 0.10

# ---------------- live commands (/rotino/cmd_*) ----------------
CMD_TIMEOUT = 0.5            # s without /rotino/cmd_vel -> velocity and turn rate go back to zero
CMD_V_MAX = 0.6              # m/s
CMD_W_MAX = 1.5              # rad/s
CMD_YAW_ACC = 3.0            # rad/s^2 (forward acceleration uses accel_max)
CMD_HEIGHT_MIN, CMD_HEIGHT_MAX = -0.05, 0.04   # m, offset of the CoM height from the nominal pose
CMD_HEIGHT_RATE = 0.08       # m/s
CMD_PUSH_MAX = 6.0           # N s


def smoothstep(t, T):
    tau = min(max(t / T, 0.0), 1.0) if T > 0.0 else 1.0
    return 3.0 * tau ** 2 - 2.0 * tau ** 3, (6.0 * tau - 6.0 * tau ** 2) / T if T > 0.0 else 0.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class WBRController(Node):

    def __init__(self):
        super().__init__('rotino_mpc_controller')
        self.declare_parameter('robot_description', '')
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        self.cfg = {name: self.get_parameter(name).value for name in PARAMS}
        urdf_xml = self.get_parameter('robot_description').value
        if not urdf_xml:
            raise RuntimeError('Parameter robot_description is empty.')

        self.model = WBRModel(urdf_xml)
        p = self.model.p
        self.weight = p.m_b * G

        # height <-> hip angle (hip = -knee/2), used for the pendulum-length grid and the jump heights
        samples = []
        self.height_table = []
        for hip in np.linspace(-0.45, 0.45, 25):
            e = self.model.equivalent_centroid(hip, -2.0 * hip)
            samples.append((e['l'], e['I_y']))
            self.height_table.append((e['z_b'], e['Z_C']))
        self.height_table.sort()
        self.lqr = LQRSchedule(self.model, samples, LQR_Q, LQR_R)
        self.mpc = UpperBodyMPC(p.m_b, MPC_HORIZON, MPC_DT, MPC_S_H, MPC_W_H, MPC_S_V, MPC_W_V)
        self.z_nom = self.model.equivalent_centroid(0.0, 0.0)['Z_C']
        self.z_low = self._z_for_hip_axle(HIP_AXLE_LOW)
        self.z_high = self._z_for_hip_axle(HIP_AXLE_HIGH)
        self.z_flight = self._z_for_hip_axle(HIP_AXLE_FLIGHT)
        self.trajectory = (CubicSTrajectory(self.cfg['traj_length'], self.cfg['traj_lateral'],
                                            self.cfg['traj_duration']) if self.cfg['planar_enable'] else None)

        self.get_logger().info(
            f"m_b={p.m_b:.3f} kg, z_nom={self.z_nom:.4f} m, K(l_nom) theta={self.lqr.gain(0.163)[0, 1]:+.2f} Nm/rad, "
            f"squat/thrust CoM heights {self.z_low:.3f}/{self.z_high:.3f} m, modes: "
            + ', '.join(k for k, v in self.cfg.items() if k.endswith('_enable') and v))

        self.wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_effort_controller/commands', 10)
        self.leg_pub = self.create_publisher(Float64MultiArray, '/leg_effort_controller/commands', 10)
        self.release_pub = self.create_publisher(Empty, '/rotino/release', 10)
        self.state_pub = self.create_publisher(String, '/rotino/jump_state', 10)
        self.wrench_pub = self.create_publisher(EntityWrench, '/world/rotino_world/wrench', 10)
        # [t, theta, theta_dot, s, s_dot, wheel_u, com_z, com_z_vel, Fn_total, min_wheel_gap, loaded]
        self.debug_pub = self.create_publisher(Float64MultiArray, '/rotino/debug', 10)
        # [t, s, s_ref, theta, theta_ref, phi, phi_ref, s_dot, s_dot_ref, z, z_ref, delta_s, F_z, tau_l, tau_r, l,
        #  tau_hip_l, tau_knee_l]
        self.wbr_pub = self.create_publisher(Float64MultiArray, '/rotino/wbr_state', 10)
        # [ex, ey, ez, evx, evy, evz]: estimated - ground-truth torso position/velocity (world)
        self.est_err_pub = self.create_publisher(Float64MultiArray, '/rotino/estimation_error', 10)
        self.planar_pub = self.create_publisher(Float64MultiArray, '/rotino/planar', 10)
        self.tracking_error_pub = self.create_publisher(Float64MultiArray, '/rotino/tracking_error', 10)

        self.imu = None
        self.odom = None
        self.contact = {'left': (-math.inf, 0.0), 'right': (-math.inf, 0.0)}
        # depth 1: only the newest sample matters; late control ticks use the real dt
        self.create_subscription(Imu, '/rotino/imu', lambda m: setattr(self, 'imu', m), 1)
        self.create_subscription(Odometry, '/rotino/odom', lambda m: setattr(self, 'odom', m), 1)
        self.create_subscription(Contacts, '/rotino/left_wheel_contact', lambda m: self._contact_cb(m, 'left'), 1)
        self.create_subscription(Contacts, '/rotino/right_wheel_contact', lambda m: self._contact_cb(m, 'right'), 1)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 1)
        self.create_subscription(Twist, '/rotino/cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(Float64, '/rotino/cmd_height', self._cmd_height_cb, 10)
        self.create_subscription(Empty, '/rotino/cmd_jump', self._cmd_jump_cb, 10)
        self.create_subscription(Float64, '/rotino/cmd_push', self._cmd_push_cb, 10)
        # wait_start: the scripted motions stay on hold until a message on /rotino/cmd_start
        self.create_subscription(Empty, '/rotino/cmd_start', self._cmd_start_cb, 10)
        self.motion_start = math.inf if self.cfg['wait_start'] else MOTION_START_TIME
        self.start_request = False

        self.kf = [AxisKalman(KF_Q_ACC, KF_R_POS, KF_R_VEL) for _ in range(3)]
        self.phase = PHASE_HOLD
        self.phase_t0 = 0.0
        self.hold_start = None
        self.t_release = None
        self.last_stamp = None
        self.tick = 0
        self.cpu_ms = 0.0
        self.max_dt = 0.0
        self.last_print = -math.inf
        self.jump_done = False
        self.jump_request = False
        self.jump_wait_logged = False
        self.push_request = None
        self.push_ticks_left = 0
        self.push_force = 0.0
        self.scripted_push_done = False
        self.heading_hold = 0.0
        self.t_now = 0.0

        # live commands: once one arrives the scripted motion profiles are replaced by the teleop reference
        self.teleop = False
        self.teleop_ready = False
        self.cmd_v = self.cmd_w = self.cmd_height = 0.0
        self.cmd_wall = -math.inf
        self.tele_s = self.tele_v = self.tele_yaw = self.tele_w = 0.0
        self.tele_z = None
        self.tele_zd = 0.0
        self.yaw_i = 0.0

        self.loaded = True
        self.unloaded_timer = 0.0
        self.loaded_timer = 0.0
        self.P_w = np.zeros(3)
        self.s = 0.0
        self.s_offset = 0.0
        self.yaw0 = 0.0
        self.prev_theta = None
        self.theta_dot = 0.0
        self.delta_s = 0.0
        self.delta_s_f = 0.0
        self.mpc_t0 = None
        self.mpc_x0 = None
        self.s_dot_f = 0.0
        self.F_z = self.weight
        self.z_land = self.z_nom
        self.z_ref_last = self.z_nom
        self.odom0 = None
        self.get_logger().info('Waiting for /joint_states and /rotino/imu ...')

    # -----------------------------------------------------------------
    def _z_for_hip_axle(self, zb):
        zbs, zcs = zip(*self.height_table)
        return float(np.interp(zb, zbs, zcs))

    def _contact_cb(self, msg, side):
        force, touching = 0.0, False
        for c in msg.contacts:
            if 'ground' not in c.collision1.name and 'ground' not in c.collision2.name:
                continue
            touching = True
            for w in c.wrenches:
                f = w.body_1_wrench.force
                force += math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
            if not c.wrenches:
                force = 0.5 * self.weight
        if touching:
            self.contact[side] = (stamp_to_sec(msg.header.stamp), force)

    # -----------------------------------------------------------------
    # Live commands
    # -----------------------------------------------------------------
    def _start_teleop(self):
        if not self.teleop:
            self.teleop = True
            self.get_logger().info('Live commands received: scripted motion profiles disabled')

    def _cmd_start_cb(self, _msg):
        self.start_request = True

    def _update_start(self, t):
        """wait_start: the scripted motions begin START_LEAD after the start command (never before the
        usual MOTION_START_TIME, if the command came during the release)."""
        if self.start_request and math.isinf(self.motion_start):
            self.motion_start = max(t + START_LEAD, MOTION_START_TIME)
            self.get_logger().info(f'Start command: the motion begins at t={self.motion_start:.2f} s')
        self.start_request = False

    def _cmd_vel_cb(self, msg):
        self.cmd_v = float(np.clip(msg.linear.x, -CMD_V_MAX, CMD_V_MAX))
        self.cmd_w = float(np.clip(msg.angular.z, -CMD_W_MAX, CMD_W_MAX))
        self.cmd_wall = time.monotonic()
        self._start_teleop()

    def _cmd_height_cb(self, msg):
        self.cmd_height = float(np.clip(msg.data, CMD_HEIGHT_MIN, CMD_HEIGHT_MAX))
        self._start_teleop()

    def _cmd_jump_cb(self, _msg):
        if self.phase == PHASE_BALANCE and not self.jump_request:
            self.jump_request = True
            self.jump_wait_logged = False
            self.get_logger().info('Jump requested')

    def _cmd_push_cb(self, msg):
        self.push_request = float(np.clip(msg.data, -CMD_PUSH_MAX, CMD_PUSH_MAX))

    def _update_teleop(self, t, dt, yaw):
        if not self.teleop or self.phase == PHASE_HOLD:
            return
        if not self.teleop_ready:
            # start from the current state so that taking over is bumpless
            self.tele_s = self.s
            self.tele_yaw = wrap(yaw - self.yaw0)
            self.tele_z = self.z_ref_last
            self.teleop_ready = True
        balance = self.phase == PHASE_BALANCE
        fresh = time.monotonic() - self.cmd_wall < CMD_TIMEOUT
        v_target = self.cmd_v if fresh and balance and not self.jump_request else 0.0
        w_target = self.cmd_w if fresh and balance and not self.jump_request else 0.0
        a = self.cfg['accel_max'] * dt
        self.tele_v += float(np.clip(v_target - self.tele_v, -a, a))
        self.tele_w += float(np.clip(w_target - self.tele_w, -CMD_YAW_ACC * dt, CMD_YAW_ACC * dt))
        if balance:
            self.tele_s += self.tele_v * dt
            self.tele_yaw += self.tele_w * dt
            # anti-windup: the heading reference never runs further than YAW_ERROR_MAX ahead of the robot
            lag = wrap(self.tele_yaw - (yaw - self.yaw0))
            if abs(lag) > YAW_ERROR_MAX:
                self.tele_yaw -= lag - math.copysign(YAW_ERROR_MAX, lag)
            step = float(np.clip(self.z_nom + self.cmd_height - self.tele_z, -CMD_HEIGHT_RATE * dt, CMD_HEIGHT_RATE * dt))
            self.tele_z += step
            self.tele_zd = step / dt

    @property
    def z_stand(self):
        return self.tele_z if self.teleop and self.tele_z is not None else self.z_nom

    def _set_phase(self, phase, t):
        self.phase, self.phase_t0 = phase, t
        self.state_pub.publish(String(data=phase))
        self.get_logger().info(f'>>> PHASE {phase} t={t:.3f}s')

    # -----------------------------------------------------------------
    # Operator / trajectory references (Fig. 4, blue blocks)
    # -----------------------------------------------------------------
    def _motion_reference(self, t):
        """Returns (s_ref, s_dot_ref, yaw_offset, yaw_rate_ref) of the stance trajectory at time t after release."""
        c = self.cfg
        tm = t - self.motion_start
        if self.phase != PHASE_BALANCE:
            return 0.0, 0.0, self.heading_hold, 0.0
        if self.teleop and self.teleop_ready:
            ahead = max(t - self.t_now, 0.0)   # MPC horizon: constant commanded rates
            return (self.tele_s + self.tele_v * ahead - self.s_offset, self.tele_v,
                    self.tele_yaw + self.tele_w * ahead, self.tele_w)
        if self.jump_done or tm <= 0.0:
            return 0.0, 0.0, self.heading_hold, 0.0
        if c['velocity_enable']:
            v, a, D = c['velocity_max'], c['accel_max'], c['velocity_distance']
            v = min(v, math.sqrt(a * D))
            ta = v / a
            tc = max(0.0, (D - v * ta) / v)
            if tm < ta:
                return 0.5 * a * tm * tm, a * tm, 0.0, 0.0
            if tm < ta + tc:
                return 0.5 * v * ta + v * (tm - ta), v, 0.0, 0.0
            td = min(tm - ta - tc, ta)
            return 0.5 * v * ta + v * tc + v * td - 0.5 * a * td * td, v - a * td, 0.0, 0.0
        if c['drive_enable']:
            w = 2.0 * math.pi / c['drive_period']
            return (0.5 * c['drive_distance'] * (1.0 - math.cos(w * tm)),
                    0.5 * c['drive_distance'] * w * math.sin(w * tm), 0.0, 0.0)
        if self.trajectory is not None:
            s, v = self.trajectory._time_law(tm)
            _, _, heading, _, yaw_rate = self.trajectory.sample(tm)
            return s, v, heading, yaw_rate
        return 0.0, 0.0, 0.0, 0.0

    def _height_reference(self, t):
        """(z_ref, z_dot_ref, z_ddot_ref) of the CoM above the axle at time t after release."""
        tp = t - self.phase_t0
        if self.phase == PHASE_SQUAT:
            a, da = smoothstep(tp, T_SQUAT)
            z0 = self.z_stand
            return z0 + (self.z_low - z0) * a, (self.z_low - z0) * da, 0.0
        if self.phase == PHASE_THRUST:
            # constant acceleration so that the CoM leaves full extension at jump_velocity
            v = self.cfg['jump_velocity']
            acc = v * v / (2.0 * (self.z_high - self.z_low))
            T = v / acc
            if tp < T:
                return self.z_low + 0.5 * acc * tp * tp, acc * tp, acc
            return self.z_high + v * (tp - T), v, 0.0
        if self.phase == PHASE_LANDING:
            a, da = smoothstep(tp, T_LANDING)
            z1 = self.z_stand
            return self.z_land + (z1 - self.z_land) * a, (z1 - self.z_land) * da, 0.0
        if self.teleop and self.tele_z is not None and self.phase == PHASE_BALANCE:
            target = self.z_nom + self.cmd_height
            z = self.tele_z + self.tele_zd * max(t - self.t_now, 0.0)
            z = min(z, target) if self.tele_zd > 0.0 else max(z, target) if self.tele_zd < 0.0 else z
            return z, self.tele_zd, 0.0
        th = t - self.motion_start
        if self.cfg['height_enable'] and th > 0.0 and self.phase == PHASE_BALANCE:
            w = 2.0 * math.pi / self.cfg['height_period']
            A = self.cfg['height_amplitude']
            return self.z_nom + A * math.sin(w * th), A * w * math.cos(w * th), -A * w * w * math.sin(w * th)
        return self.z_nom, 0.0, 0.0

    # -----------------------------------------------------------------
    def _joint_state_cb(self, msg):
        if self.imu is None:
            return
        t_abs = stamp_to_sec(msg.header.stamp)
        if self.last_stamp is None:
            self.last_stamp = t_abs
            return
        dt = t_abs - self.last_stamp
        if dt <= 0.0:
            return
        self.last_stamp = t_abs
        q = dict(zip(msg.name, msg.position))
        qd = dict(zip(msg.name, msg.velocity))
        if any(n not in q for n in LEG_JOINTS + WHEEL_JOINTS):
            return
        self.tick += 1
        t_cpu = time.perf_counter()
        if self.phase == PHASE_HOLD:
            self._hold_step(t_abs, q, qd)
        else:
            self._control_step(t_abs - self.t_release, dt, q, qd)
        self.cpu_ms = 0.99 * self.cpu_ms + 0.01 * (time.perf_counter() - t_cpu) * 1e3
        self.max_dt = max(self.max_dt, dt)

    # -----------------------------------------------------------------
    def _legs(self, q, qd):
        pl, Jl = self.model.leg_fk(q['left_hip'], q['left_knee'])
        pr, Jr = self.model.leg_fk(q['right_hip'], q['right_knee'])
        vl = Jl @ np.array([qd['left_hip'], qd['left_knee']])
        vr = Jr @ np.array([qd['right_hip'], qd['right_knee']])
        self.leg_qd = (np.array([qd['left_hip'], qd['left_knee']]), np.array([qd['right_hip'], qd['right_knee']]))
        return (pl, Jl, vl), (pr, Jr, vr)

    def _vmc(self, legs, p_d, F_ff, Kp, Kd, v_d=np.zeros(2)):
        """Eq. (19) for both legs: tau = J^T [Kp (p_d - p_f) + Kd (v_d - v_f) + F_ff] -> [hip_L, hip_R, knee_L, knee_R]."""
        taus = []
        for (p_f, J, v_f), qd in zip(legs, self.leg_qd):
            F = Kp @ (p_d - p_f) + Kd @ (v_d - v_f) + F_ff
            tau_ff = self.model.p.leg_damping * qd   # feedforward: cancels the URDF joint damping
            taus.append(np.clip(J.T @ F + tau_ff, -LEG_TORQUE_MAX, LEG_TORQUE_MAX))
        tau = [taus[0][0], taus[1][0], taus[0][1], taus[1][1]]
        self.leg_pub.publish(Float64MultiArray(data=[float(x) for x in tau]))
        return tau

    def _publish_wheels(self, tau_l, tau_r):
        self.wheel_pub.publish(Float64MultiArray(data=[float(tau_l), float(tau_r)]))

    def _hold_step(self, t_abs, q, qd):
        if self.hold_start is None:
            self.hold_start = t_abs
            self.get_logger().info('Holding on the anchor, legs to the nominal pose ...')
        legs = self._legs(q, qd)
        c_b = self.model.upper_body_com_base(q)
        p_d = np.array([c_b[0], c_b[2] - self.z_nom])
        self._vmc(legs, p_d, np.zeros(2), VMC_KP_HOLD, VMC_KD_HOLD)
        self._publish_wheels(0.0, 0.0)
        if t_abs - self.hold_start < HOLD_TIME:
            return

        R = quat_to_matrix(*self._quat())
        axle_b = np.array([0.5 * (legs[0][0][0] + legs[1][0][0]), 0.0, 0.5 * (legs[0][0][1] + legs[1][0][1])])
        self.P_w = np.array([0.0, 0.0, self.model.p.r])
        P_b = self.P_w - R @ axle_b
        for i in range(3):
            self.kf[i].reset(P_b[i], 0.0)
        heading = R[:, 0]
        self.yaw0 = math.atan2(heading[1], heading[0])
        if self.odom is not None:
            o = self.odom.pose.pose.position
            self.odom0 = np.array([o.x, o.y, o.z]) - P_b
        self.release_pub.publish(Empty())
        self.t_release = t_abs
        self._set_phase(PHASE_BALANCE, 0.0)

    def _quat(self):
        o = self.imu.orientation
        return o.x, o.y, o.z, o.w

    # -----------------------------------------------------------------
    def _control_step(self, t, dt, q, qd):
        p = self.model.p
        self.t_now = t
        if t < RELEASE_REPEAT_TIME:
            self.release_pub.publish(Empty())

        # ---------------- IMU ----------------
        R = quat_to_matrix(*self._quat())
        gyro_b = np.array([self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z])
        omega_w = R @ gyro_b
        f_b = np.array([self.imu.linear_acceleration.x, self.imu.linear_acceleration.y,
                        self.imu.linear_acceleration.z])
        acc_w = R @ f_b - np.array([0.0, 0.0, G])
        heading = R[:, 0].copy()
        heading[2] = 0.0
        heading /= max(np.linalg.norm(heading), 1e-9)
        yaw = math.atan2(heading[1], heading[0])

        # ---------------- leg kinematics, equivalent centroid (eqs. 2-3) ----------------
        legs = self._legs(q, qd)
        axle_b = np.array([0.5 * (legs[0][0][0] + legs[1][0][0]), 0.0, 0.5 * (legs[0][0][1] + legs[1][0][1])])
        v_rel_b = np.array([0.5 * (legs[0][2][0] + legs[1][2][0]), 0.0, 0.5 * (legs[0][2][1] + legs[1][2][1])])
        c_b = self.model.upper_body_com_base(q)
        com_rel_w = R @ (c_b - axle_b)
        S_C = float(np.dot(com_rel_w[:2], heading[:2]))
        Z_C = float(com_rel_w[2])
        l = math.hypot(S_C, Z_C)
        theta = math.atan2(S_C, Z_C)
        if self.prev_theta is None:
            self.prev_theta = theta
        self.theta_dot = THETA_DOT_FILTER * self.theta_dot + (1 - THETA_DOT_FILTER) * (theta - self.prev_theta) / dt
        self.prev_theta = theta

        # ---------------- contact state ----------------
        fn = sum(f if t_now_ok else 0.0 for f, t_now_ok in
                 ((self.contact[s][1], self.last_stamp - self.contact[s][0] <= CONTACT_STALE_TIME)
                  for s in ('left', 'right')))
        if self.loaded:
            self.unloaded_timer = self.unloaded_timer + dt if fn < F_CONTACT_OFF else 0.0
            if self.unloaded_timer >= CONTACT_LOSS_DEBOUNCE:
                self.loaded, self.loaded_timer = False, 0.0
        else:
            self.loaded_timer = self.loaded_timer + dt if fn > F_CONTACT_ON else 0.0
            if self.loaded_timer >= CONTACT_GAIN_DEBOUNCE:
                self.loaded, self.unloaded_timer = True, 0.0
        stance = self.phase in STANCE_PHASES and self.loaded

        # ---------------- Kalman filter (eqs. 20-21) ----------------
        axle_w_rel = R @ axle_b
        wP_b = -axle_w_rel
        wV_b = -np.cross(omega_w, axle_w_rel) - R @ v_rel_b
        for i in range(3):
            self.kf[i].predict(acc_w[i], dt)
        P_b = np.array([k.x[0] for k in self.kf])
        V_b = np.array([k.x[1] for k in self.kf])
        if stance:
            pitch_rate = gyro_b[1]
            w_l = qd['left_wheel_joint'] + pitch_rate + qd['left_hip'] + qd['left_knee']
            w_r = qd['right_wheel_joint'] + pitch_rate + qd['right_hip'] + qd['right_knee']
            V_w = 0.5 * p.r * (w_l + w_r) * heading
            self.P_w = self.P_w + V_w * dt
            self.P_w[2] = p.r
            for i in range(3):
                self.kf[i].correct(self.P_w[i] + wP_b[i], V_w[i] + wV_b[i])
            P_b = np.array([k.x[0] for k in self.kf])
            V_b = np.array([k.x[1] for k in self.kf])
        else:
            self.P_w = P_b - wP_b
        V_axle = V_b - wV_b
        self.s_dot_f = S_DOT_FILTER * self.s_dot_f + (1 - S_DOT_FILTER) * float(np.dot(V_axle[:2], heading[:2]))
        s_dot = self.s_dot_f
        self.s += float(np.dot(V_axle[:2], heading[:2])) * dt
        V_com = V_b + np.cross(omega_w, R @ c_b)
        s_com = self.s + S_C
        s_com_dot = float(np.dot(V_com[:2], heading[:2]))
        com_height = float(P_b[2] + (R @ c_b)[2])
        wheel_gap = float(P_b[2] + axle_w_rel[2]) - p.r

        # ---------------- phase machine (jump: Fig. 11 stages) ----------------
        tp = t - self.phase_t0
        if self.phase == PHASE_BALANCE:
            scripted = self.cfg['jump_enable'] and not self.jump_done and t > self.cfg['jump_start_time']
            if scripted or self.jump_request:
                if abs(self.theta_dot) < 0.3 and abs(s_dot) < 0.08:
                    self.jump_request = False
                    self.s_offset = self.s
                    self.heading_hold = wrap(yaw - self.yaw0)
                    self.tele_v = self.tele_w = 0.0
                    self._set_phase(PHASE_SQUAT, t)
                elif self.jump_request and not self.jump_wait_logged:
                    self.jump_wait_logged = True
                    self.get_logger().info('Jump: stopping the robot first')
        elif self.phase == PHASE_SQUAT and tp >= T_SQUAT:
            self._set_phase(PHASE_THRUST, t)
        elif self.phase == PHASE_THRUST:
            if not self.loaded:
                self._set_phase(PHASE_FLIGHT, t)
            elif tp >= T_THRUST_MAX:
                self.get_logger().warn('No take-off detected')
                self.z_land = Z_C
                self.s_offset = self.tele_s = self.s
                self._set_phase(PHASE_LANDING, t)
        elif self.phase == PHASE_FLIGHT:
            if self.loaded:
                self.z_land = Z_C
                self.s_offset = self.tele_s = self.s
                self.mpc.reset()
                self.get_logger().info(f'Landed after {tp:.3f}s of flight')
                self._set_phase(PHASE_LANDING, t)
        elif self.phase == PHASE_LANDING and tp >= T_LANDING + T_SETTLE:
            self.jump_done = True
            self._set_phase(PHASE_BALANCE, t)

        # ---------------- references ----------------
        self._update_start(t)
        self._update_teleop(t, dt, yaw)
        z_ref, zd_ref, zdd_ref = self._height_reference(t)
        self.z_ref_last = z_ref
        s_ref, sd_ref, heading_ref, yaw_rate_ref = self._motion_reference(t)
        s_ref += self.s_offset
        yaw_ref = self.yaw0 + heading_ref
        if self.trajectory is not None and self.phase == PHASE_BALANCE and not self.jump_done and not self.teleop:
            yaw_ref = self._planar_correction(t, P_b, wP_b, yaw, yaw_ref, sd_ref)

        tau_l = tau_r = 0.0
        tau_legs = [0.0] * 4
        if self.phase == PHASE_FLIGHT:
            # wheels held still, legs retracted under the CoM (flight task-space controller)
            p_d = np.array([c_b[0], c_b[2] - self.z_flight])
            tau_legs = self._vmc(legs, p_d, np.zeros(2), VMC_KP_FLIGHT, VMC_KD_FLIGHT)
            tau_l = -FLIGHT_WHEEL_DAMPING * qd['left_wheel_joint']
            tau_r = -FLIGHT_WHEEL_DAMPING * qd['right_wheel_joint']
        else:
            # ---------------- upper-body MPC, 100 Hz ----------------
            if not self.cfg['mpc_enable']:
                self.delta_s, self.F_z = 0.0, self.weight
            elif self.tick % MPC_EVERY == 0:
                self.delta_s, self.F_z = self._run_mpc(t, s_com, s_com_dot, Z_C, float(V_com[2]), axle_b)
                self.mpc_t0, self.mpc_x0 = t, np.array([s_com, s_com_dot])
            self.delta_s_f = DELTA_S_FILTER * self.delta_s_f + (1 - DELTA_S_FILTER) * self.delta_s
            ds = self.delta_s_f

            # ---------------- TV-LQR ----------------
            theta_ref = math.atan2(ds, z_ref)
            X = np.array([self.s, theta, yaw, s_dot, self.theta_dot, omega_w[2]])
            s_plan, sd_plan = s_ref, sd_ref
            if self.cfg['mpc_enable'] and self.mpc.x_h is not None and self.mpc_t0 is not None:
                # the wheels follow the CoM motion planned by the MPC (virtual CoM of Fig. 4)
                a = min(max((t - self.mpc_t0) / MPC_DT, 0.0), 1.0)
                s_plan, sd_plan = (1.0 - a) * self.mpc_x0 + a * self.mpc.x_h[0]
            X_ref = np.array([s_plan - ds, theta_ref, yaw_ref, sd_plan, 0.0, yaw_rate_ref])
            err = X - X_ref
            err[2] = float(np.clip(wrap(err[2]), -YAW_ERROR_MAX, YAW_ERROR_MAX))
            U = -self.lqr.gain(l) @ err
            common = 0.5 * (U[0] + U[1])
            if self.phase == PHASE_BALANCE:
                if abs(err[2]) > YAW_I_DEADBAND or yaw_rate_ref != 0.0:
                    self.yaw_i = float(np.clip(self.yaw_i - YAW_KI * err[2] * dt, -YAW_I_MAX, YAW_I_MAX))
            else:
                self.yaw_i = 0.0
            diff_ff = (YAW_FRICTION * math.tanh(yaw_rate_ref / YAW_FRICTION_SMOOTH) + YAW_VISCOUS * yaw_rate_ref)
            diff = float(np.clip(0.5 * (U[1] - U[0]) + diff_ff + self.yaw_i, -DIFF_TORQUE_MAX, DIFF_TORQUE_MAX))
            common = float(np.clip(common, -(WHEEL_TORQUE_MAX - abs(diff)), WHEEL_TORQUE_MAX - abs(diff)))
            tau_l, tau_r = common - diff, common + diff

            # ---------------- VMC ----------------
            up_b = R.T @ np.array([0.0, 0.0, 1.0])
            F_ff = -0.5 * self.F_z * np.array([up_b[0], up_b[2]])
            p_d = np.array([c_b[0] - ds, c_b[2] - z_ref])
            v_d = np.array([0.0, -zd_ref])
            if self.cfg['mpc_enable']:
                tau_legs = self._vmc(legs, p_d, F_ff, VMC_KP, VMC_KD, v_d)
            else:
                tau_legs = self._vmc(legs, p_d, F_ff, VMC_KP_HOLD, VMC_KD_HOLD, v_d)

        self._publish_wheels(tau_l, tau_r)
        self._push(t)

        # ---------------- telemetry ----------------
        if self.tick % 50 == 0:   # 10 Hz: late subscribers (dashboard) also learn the current phase
            self.state_pub.publish(String(data=self.phase))
        self.debug_pub.publish(Float64MultiArray(data=[
            t, theta, self.theta_dot, self.s, s_dot, 0.5 * (tau_l + tau_r) / p.wheel_torque_max,
            com_height, float(V_b[2]), fn, wheel_gap, float(self.loaded)]))
        self.wbr_pub.publish(Float64MultiArray(data=[
            t, self.s, s_ref - self.delta_s, theta, math.atan2(self.delta_s, z_ref), yaw, yaw_ref, s_dot, sd_ref,
            Z_C, z_ref, self.delta_s, self.F_z, float(tau_l), float(tau_r), l, tau_legs[0], tau_legs[2]]))
        if self.odom is not None and self.odom0 is not None:
            o, v = self.odom.pose.pose.position, self.odom.twist.twist.linear
            Ro = quat_to_matrix(self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w)
            gt_p = np.array([o.x, o.y, o.z]) - self.odom0
            gt_v = Ro @ np.array([v.x, v.y, v.z])
            self.est_err_pub.publish(Float64MultiArray(data=[*(P_b - gt_p), *(V_b - gt_v)]))

        if t - self.last_print >= 0.5:
            self.last_print = t
            self.get_logger().info(
                f't={t:6.2f} {self.phase:>8} th={math.degrees(theta):+6.2f}deg thref={math.degrees(math.atan2(self.delta_s, z_ref)):+5.2f} '
                f's={self.s:+6.3f}/{s_ref:+6.3f} v={s_dot:+5.2f}/{sd_ref:+5.2f} z={Z_C:.3f}/{z_ref:.3f} '
                f'yaw={math.degrees(wrap(yaw - self.yaw0)):+6.1f} ds={self.delta_s:+.4f} Fz={self.F_z:5.1f} '
                f'tau={tau_l:+5.2f}/{tau_r:+5.2f} legs={tau_legs[0]:+5.1f}/{tau_legs[2]:+5.1f} '
                f'gap={wheel_gap:+.3f} loaded={int(self.loaded)} cpu={self.cpu_ms:.2f}ms max_dt={self.max_dt * 1e3:.0f}ms')
            self.max_dt = 0.0

    # -----------------------------------------------------------------
    def _run_mpc(self, t, s_com, s_com_dot, z, z_dot, axle_b):
        p = self.model.p
        times = t + MPC_DT * np.arange(1, MPC_HORIZON + 1)
        ref_h = np.zeros((MPC_HORIZON, 2))
        ref_v = np.zeros((MPC_HORIZON, 2))
        for i, ti in enumerate(times):
            s_i, v_i, _, _ = self._motion_reference(ti)
            z_i, zd_i, _ = self._height_reference(ti)
            ref_h[i] = (s_i + self.s_offset, v_i)
            ref_v[i] = (z_i, zd_i)
        _, _, zdd = self._height_reference(t)
        z_b = -axle_b[2]
        ds_max = min(p.mu * z, math.sqrt(max(p.L_max ** 2 - z_b ** 2, 0.0)), DS_MAX_CAP)
        f_max = (F_MAX_THRUST_RATIO if self.phase == PHASE_THRUST else F_MAX_RATIO) * self.weight
        return self.mpc.solve([s_com, s_com_dot], ref_h, z, zdd, ds_max,
                              [z, z_dot], ref_v, F_MIN_RATIO * self.weight, f_max)

    def _planar_correction(self, t, P_b, wP_b, yaw, yaw_ref, v_ref):
        tm = max(t - self.motion_start, 0.0)
        px, py, _, _, _ = self.trajectory.sample(tm)
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        p_ref = np.array([c0 * px - s0 * py, s0 * px + c0 * py])
        axle_xy = (P_b - wP_b)[:2]
        err = axle_xy - p_ref
        normal = np.array([-math.sin(yaw_ref), math.cos(yaw_ref)])
        lateral = float(np.dot(err, normal))
        yaw_cmd = yaw_ref - math.atan(K_LAT * lateral) * min(1.0, abs(v_ref) / LAT_FULL_SPEED)
        self.planar_pub.publish(Float64MultiArray(data=[
            t, float(p_ref[0]), float(p_ref[1]), yaw_ref, float(axle_xy[0]), float(axle_xy[1]), yaw,
            float(np.dot(err, [math.cos(yaw_ref), math.sin(yaw_ref)])), lateral, 0.0]))
        self.tracking_error_pub.publish(Float64MultiArray(data=[float(err[0]), float(err[1])]))
        return yaw_cmd

    def _push(self, t):
        """Horizontal impulse on the torso: positive = forward along the heading."""
        c = self.cfg
        if c['push_enable'] and not self.scripted_push_done and t >= c['push_time']:
            self.scripted_push_done = True
            self.push_request = -c['push_impulse']
        if self.push_request is not None and self.push_ticks_left == 0:
            self.push_force = self.push_request / (PUSH_TICKS * PHYSICS_DT)
            self.push_ticks_left = PUSH_TICKS
            self.get_logger().info(f'>>> PUSH {self.push_request:+.2f} N s at t={t:.3f}s')
            self.push_request = None
        if self.push_ticks_left == 0:
            return
        self.push_ticks_left -= 1
        msg = EntityWrench()
        msg.entity.name = 'rotino::base_link'
        msg.entity.type = Entity.LINK
        heading = quat_to_matrix(*self._quat())[:, 0]
        msg.wrench.force.x = float(self.push_force * heading[0])
        msg.wrench.force.y = float(self.push_force * heading[1])
        self.wrench_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WBRController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
