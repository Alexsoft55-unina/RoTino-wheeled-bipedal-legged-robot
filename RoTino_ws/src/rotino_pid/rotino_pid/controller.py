"""
RoTino balance + vertical jump controller for ROS 2 Humble / Gazebo Fortress: cascaded PID with the
Zero Moment Point as the balance variable (rotino_pid/zmp_balance.py).

Port of RoTino_Ctrl_BalJumpAchieve.py (MuJoCo, position_servo leg mode), redesigned for torque control:
  - wheels: capture-point PI -> desired ZMP -> ZMP-offset PD on the wheel torque, with LIPM preview of
    the planned path (the CoM leans before the wheels accelerate) and model feedforward
  - lateral ZMP: the torso leans into turns (one leg shorter) so that the ZMP stays at the centre of
    the wheel segment; feedforward h a_y / g plus a correction on the measured multibody ZMP
  - legs: Cartesian PD (wheel centre w.r.t. the hip) with weight feedforward. The joint-space PD of the
    MuJoCo port (160/200 Nm/rad) entered a bang-bang limit cycle at 500 Hz (diagnostics/06_diagnosi.txt)
  - seven-state hybrid jump supervisor with phase-dependent gains
  - contact-force hysteresis + wheel clearance + vertical COM velocity for take-off/landing

MuJoCo -> ROS 2 mapping:
  data.qpos / qvel (joints)      -> /joint_states               (joint_state_broadcaster)
  torso freejoint pose           -> /rotino/odom                (gz OdometryPublisher, ground truth)
  robot_com / wheel framepos     -> forward kinematics on the URDF (kinematics.py)
  mj_contactForce wheel/ground   -> /rotino/{left,right}_wheel_contact (gz Contact sensor)
  wheel motor ctrl (gear 18)     -> /wheel_effort_controller/commands  [Nm]
  hip/knee                       -> /leg_effort_controller/commands torques [Nm]
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

from rotino_description.kinematics import RobotKinematics, quat_to_matrix
from rotino_description.model import WBRModel
from rotino_description.planar_trajectory import CubicSTrajectory
from rotino_description.zmp import ZmpEstimator
from rotino_pid.zmp_balance import (LipmPreview, ZmpLateralCompensation, ZmpSagittalBalance,
                                    trapezoid_path)

# =========================================================
# Geometry / actuation (RoTino_BOT(II).xml)
# =========================================================
WHEEL_RADIUS = 0.06
WHEEL_GEAR = 18.0            # the MuJoCo motor gear: wheel_u = torque / WHEEL_GEAR in the debug topic
WHEEL_TORQUE_MAX = 10.0      # Nm per wheel, as the MPC
G = 9.81

LEG_JOINTS = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
WHEEL_JOINTS = ['left_wheel_joint', 'right_wheel_joint']
LEFT_WHEEL_LINK = 'left_wheel_link'
RIGHT_WHEEL_LINK = 'right_wheel_link'

# Cartesian PD of the wheel centre in the hip frame (x forward, z up), per leg: tau = J^T F.
# In joint space the balance gains are ~50 / 30 Nm/rad on hip / knee (the joint PD had 160 / 200) and the
# damping is ~1 Nm s/rad, so a single 2 ms step can no longer overshoot the reflected leg inertia.
LEG_KP_STAND = np.diag([1500.0, 5000.0])
LEG_KD_STAND = np.diag([40.0, 80.0])
LEG_KP_JUMP = np.diag([3000.0, 4000.0])
LEG_KD_JUMP = np.diag([60.0, 80.0])
LEG_KP_FLIGHT = np.diag([800.0, 800.0])
LEG_KD_FLIGHT = np.diag([20.0, 20.0])
LEG_TORQUE_MAX = 50.0
LEG_FORCE_LIMIT = 60.0
HIP_CTRL_MIN, HIP_CTRL_MAX = -0.60, 0.60
KNEE_CTRL_MIN, KNEE_CTRL_MAX = -0.80, 0.80
LEG_LENGTH_MIN, LEG_LENGTH_MAX = 0.12, 0.24   # hip-axle vertical distance [m]

# Time the robot stays attached to the anchor with the legs held, before release.
HOLD_TIME = 1.0
RELEASE_REPEAT_TIME = 0.05
MOTION_START_TIME = 2.0      # reference motions start this long after release
# wait_start: the motions start this long after /rotino/cmd_start instead, so that the preview of the PID
# (about 1.1 s ahead) and the horizon of the MPC (0.5 s) see them coming exactly as after a release
START_LEAD = 1.0
PHYSICS_DT = 0.0005          # rotino_world.sdf max_step_size (push impulse)
PUSH_TICKS = 25

# =========================================================
# Balance (gains in rotino_pid/zmp_balance.py)
# =========================================================
COMZ_VEL_FILTER = 0.40
AY_FILTER = 0.05             # s, measured centripetal acceleration (no reference trajectory)
LAT_LEAD = 0.05              # s: lag of the legs (vertical PD ~33 rad/s) in rolling the torso

# =========================================================
# Yaw / planar trajectory tracking (differential wheel torque)
# =========================================================
K_PSI = 1.8       # Nm per rad of heading error (wheel scrub friction needs ~5x the inertia-only value)
K_OMEGA = 0.45    # Nm per rad/s of yaw-rate error
K_LAT = 2.0       # rad of heading correction per m of lateral error (through atan)
MAX_YAW = 2.7     # Nm
# Wheel scrub while turning makes the yaw stick-slip: without these terms the robot turned in jerks
# (yaw rate 0.06 -> 0.8 rad/s in 0.1 s) and its real centripetal acceleration, hence the lateral ZMP,
# no longer followed the planned turn. Coulomb + viscous feedforward from the reference yaw rate
# (identified in Gazebo for the MPC) and the integral of the heading error.
YAW_FRICTION = 0.25          # Nm per wheel
YAW_VISCOUS = 0.10           # Nm per wheel per rad/s
YAW_FRICTION_SMOOTH = 0.01   # rad/s (acts on the reference only: no chattering)
YAW_KI = 0.6                 # Nm per wheel per rad s
YAW_I_MAX = 0.6              # Nm per wheel
YAW_I_DEADBAND = 0.035       # rad: no integration on small static errors (avoids stick-slip hunting)

# Roll integral on the leg length difference: with 2000 N/m stance legs the load transfer of the turn
# ate ~80 % of the commanded lean (1.5 of 7.7 mm), and a proportional roll term (2.0) excited a 5 Hz roll
# mode that shook the lateral ZMP by +-40 mm. Stiffer legs (5000 N/m) and a slow integral instead.
ROLL_KP = 0.0                # rad per rad of roll error
ROLL_KI = 5.0                # 1/s
ROLL_I_MAX = 0.10            # rad
LAT_FULL_SPEED = 0.10  # m/s of reference speed at which lateral correction is fully active

# =========================================================
# Live commands (/rotino/cmd_*, rotino_dashboard): same semantics as the MPC
# =========================================================
CMD_TIMEOUT = 0.5            # s without /rotino/cmd_vel -> velocity and turn rate go back to zero
CMD_V_MAX = 1.5              # m/s
CMD_W_MAX = 1.5              # rad/s
CMD_YAW_ACC = 3.0            # rad/s^2 (forward acceleration uses accel_max)
CMD_HEIGHT_MIN, CMD_HEIGHT_MAX = -0.05, 0.04   # m, offset of the standing height (leg length)
CMD_HEIGHT_RATE = 0.08       # m/s
CMD_PUSH_MAX = 6.0           # N s
CMD_ACC_FILTER = 0.10        # s: the ZMP feedforward of a step in commanded acceleration is smoothed
YAW_ERROR_MAX = 0.35         # rad: the commanded heading never runs further ahead of the robot

# =========================================================
# Jump state machine
# =========================================================
STATE_BALANCE = 'BALANCE'
STATE_PRELOAD = 'PRELOAD'
STATE_THRUST = 'THRUST'
STATE_FLIGHT = 'FLIGHT'
STATE_LANDING = 'LANDING'
STATE_RECOVERY = 'RECOVERY'
STATE_SETTLE = 'SETTLE'
JUMP_ACTIVE_STATES = (STATE_PRELOAD, STATE_THRUST, STATE_FLIGHT, STATE_LANDING, STATE_RECOVERY)
POST_LANDING_STATES = (STATE_LANDING, STATE_RECOVERY, STATE_SETTLE)
# share of the capture-point correction per phase (the old per-phase position gains)
OUTER_SCALE = {STATE_PRELOAD: 0.5, STATE_THRUST: 0.1, STATE_LANDING: 0.3, STATE_RECOVERY: 0.5}
FLIGHT_WHEEL_DAMPING = 0.05  # Nm per rad/s

JUMP_ONCE = True
T_PRELOAD = 0.80
T_THRUST = 0.02
T_THRUST_MAX = 0.16
T_LANDING = 0.80
T_RECOVERY = 2.00

F_CONTACT_OFF_RATIO = 0.03
F_CONTACT_ON_RATIO = 0.15
F_CONTACT_OFF_MIN = 1.5
F_CONTACT_ON_MIN = 6.0
CONTACT_LOSS_DEBOUNCE = 0.004
CONTACT_GAIN_DEBOUNCE = 0.012
# The gz contact sensor only publishes while touching; older messages mean "no contact".
CONTACT_STALE_TIME = 0.006
TAKEOFF_VZ_MIN = 0.08

WHEEL_CLEARANCE_TAKEOFF_ABS = 0.0005
WHEEL_CLEARANCE_TAKEOFF_REL = 0.0030

HIP_PRELOAD_DELTA = +0.12
KNEE_PRELOAD_DELTA = -0.22
HIP_THRUST_DELTA = -0.27
KNEE_THRUST_DELTA = +0.44
HIP_LAND_DELTA = +0.08
KNEE_LAND_DELTA = -0.20

SETTLE_LEAN_ERR = math.radians(2.0)
SETTLE_RATE = math.radians(8.0)
SETTLE_VEL = 0.04
SETTLE_WHEEL = 0.06
SETTLE_X_ERR = 0.015
SETTLE_HOLD = 0.30
MAX_SETTLE_WAIT = 12.0

PRINT_DT = {
    STATE_BALANCE: 0.50,
    STATE_PRELOAD: 0.40,
    STATE_THRUST: 0.01,
    STATE_FLIGHT: 0.04,
    STATE_LANDING: 0.08,
    STATE_RECOVERY: 0.20,
    STATE_SETTLE: 0.25,
}

PARAMS = {
    'jump_enable': True, 'jump_start_time': 5.0, 'return_to_start_after_jump': False,
    'drive_enable': False, 'drive_distance': 1.0, 'drive_period': 8.0, 'drive_start_time': 2.0,
    'velocity_enable': False, 'velocity_max': 0.44, 'accel_max': 0.6, 'velocity_distance': 2.0,
    'height_enable': False, 'height_amplitude': 0.03, 'height_period': 2.2,
    'planar_enable': False, 'traj_length': 3.0, 'traj_lateral': 0.6, 'traj_duration': 12.0,
    'push_enable': False, 'push_time': 4.0, 'push_impulse': 2.7,
    'wait_start': False,    # hold the scripted motions until /rotino/cmd_start
    'zmp_lateral': False,   # lean into turns: off, the cylindrical wheels flip contact edge (docs/PID_ZMP.md)
}


def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def smoothstep01(t, T):
    if T <= 0.0:
        return 1.0
    tau = clamp(t / T, 0.0, 1.0)
    return 3.0 * tau ** 2 - 2.0 * tau ** 3


def fast_thrust01(t, T):
    if T <= 0.0:
        return 1.0
    tau = clamp(t / T, 0.0, 1.0)
    return 1.0 - (1.0 - tau) ** 3


def blend(a, b, alpha):
    return a + (b - a) * alpha


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


SYNC_BUFFER_SIZE = 50


class BalanceJumpController(Node):

    def __init__(self):
        super().__init__('rotino_pid_controller')
        self.declare_parameter('robot_description', '')
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        self.cfg = {name: self.get_parameter(name).value for name in PARAMS}

        urdf_xml = self.get_parameter('robot_description').value
        if not urdf_xml:
            raise RuntimeError('Parameter robot_description is empty.')
        self.jump_enable = bool(self.cfg['jump_enable'])
        self.jump_start_time = float(self.cfg['jump_start_time'])
        self.return_to_start = bool(self.cfg['return_to_start_after_jump'])
        self.planar_enable = bool(self.cfg['planar_enable'])
        self.trajectory = CubicSTrajectory(
            float(self.cfg['traj_length']), float(self.cfg['traj_lateral']),
            float(self.cfg['traj_duration'])) if self.planar_enable else None

        self.kin = RobotKinematics(urdf_xml)
        self.wbr = WBRModel(urdf_xml)
        self.zmp_est = ZmpEstimator(urdf_xml)
        self.balance = ZmpSagittalBalance.from_model(self.wbr)
        self.lateral = ZmpLateralCompensation(self.wbr.p.d)
        self.preview = LipmPreview()
        self.robot_weight = self.kin.total_mass * G
        self.upper_weight = self.wbr.p.m_b * G
        self.f_contact_off = max(F_CONTACT_OFF_MIN, F_CONTACT_OFF_RATIO * self.robot_weight)
        self.f_contact_on = max(F_CONTACT_ON_MIN, F_CONTACT_ON_RATIO * self.robot_weight)
        # hip-axle vertical distance of the XML standing pose (all joints 0)
        self.leg_length0 = float(-self.wbr.leg_fk(0.0, 0.0)[0][1])

        self.get_logger().info(
            f'Robot mass = {self.kin.total_mass:.3f} kg, weight = {self.robot_weight:.3f} N, '
            f'contact OFF/ON = {self.f_contact_off:.3f}/{self.f_contact_on:.3f} N, '
            f'leg length = {self.leg_length0:.4f} m, ZMP lean per m/s^2 = {self.balance.lean_per_acc * 1e3:.1f} mm, '
            f'lateral ZMP compensation {"ON" if self.cfg["zmp_lateral"] else "OFF"}, modes: '
            + ', '.join(k for k, v in self.cfg.items() if k.endswith('_enable') and v))

        self.leg_pub = self.create_publisher(Float64MultiArray, '/leg_effort_controller/commands', 10)
        self.wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_effort_controller/commands', 10)
        self.release_pub = self.create_publisher(Empty, '/rotino/release', 10)
        self.state_pub = self.create_publisher(String, '/rotino/jump_state', 10)
        self.wrench_pub = self.create_publisher(EntityWrench, '/world/rotino_world/wrench', 10)
        # [t, theta, theta_dot, x, xdot, wheel_u, com_z, com_z_vel, Fn_total, min_wheel_gap, loaded]
        self.debug_pub = self.create_publisher(Float64MultiArray, '/rotino/debug', 10)
        # same layout as the MPC, so that the benchmark logger records references and errors:
        # [t, s, s_ref, theta, theta_ref, yaw, yaw_ref, s_dot, s_dot_ref, z, z_ref, s_des, F_z, tau_l, tau_r,
        #  l, tau_hip_l, tau_knee_l]
        self.wbr_pub = self.create_publisher(Float64MultiArray, '/rotino/wbr_state', 10)
        # [t, zmp_des, zmp, xi_err, acc_ref, a_y, lean_cmd, dz, y_zmp, margin, e_long] (m, m/s^2, rad):
        # longitudinal ZMP ahead of the CoM (desired / contact), lateral ZMP measured by the multibody estimator
        self.zmp_pub = self.create_publisher(Float64MultiArray, '/rotino/zmp_ctrl', 10)
        # [t, x_ref, y_ref, heading_ref, x, y, heading, along_err, lateral_err, yaw_u] (world frame, axle midpoint)
        self.planar_pub = self.create_publisher(Float64MultiArray, '/rotino/planar', 10)
        # [err_x, err_y]: actual - desired position, world frame x/y (axle midpoint), while following the planar trajectory
        self.tracking_error_pub = self.create_publisher(Float64MultiArray, '/rotino/tracking_error', 10)

        self.odom = None
        self.latest_odom = None
        self.latest_joint_state = None
        self.joint_state_buffer = {}
        self.odom_buffer = {}
        self.warned_no_sync = False
        self.sync_count = 0
        self.imu = None
        self.contact = {'left': (-math.inf, 0.0), 'right': (-math.inf, 0.0)}
        self.warned_no_wrench = False

        self.create_subscription(Odometry, '/rotino/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/rotino/imu', self._imu_cb, 1)
        # The contact sensor publishes every physics step (~2 kHz): with a deeper queue the node handled
        # messages up to 5 ms old, past CONTACT_STALE_TIME, and flagged the robot airborne ~30 % of the time.
        self.create_subscription(Contacts, '/rotino/left_wheel_contact',
                                 lambda msg: self._contact_cb(msg, 'left'), 1)
        self.create_subscription(Contacts, '/rotino/right_wheel_contact',
                                 lambda msg: self._contact_cb(msg, 'right'), 1)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(Twist, '/rotino/cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(Float64, '/rotino/cmd_height', self._cmd_height_cb, 10)
        self.create_subscription(Empty, '/rotino/cmd_jump', self._cmd_jump_cb, 10)
        self.create_subscription(Float64, '/rotino/cmd_push', self._cmd_push_cb, 10)
        # live commands: once one arrives the scripted motion profiles are replaced by the teleop reference
        self.teleop = False
        self.teleop_ready = False
        self.cmd_v = self.cmd_w = self.cmd_height = 0.0
        self.cmd_wall = -math.inf
        self.jump_request = False
        # wait_start: the scripted motions stay on hold until a message on /rotino/cmd_start
        self.create_subscription(Empty, '/rotino/cmd_start', self._cmd_start_cb, 10)
        self.motion_start = math.inf if self.cfg['wait_start'] else MOTION_START_TIME
        self.start_request = False

        self.phase = 'HOLD'
        self.hold_start = None
        self.t_release = None
        self.last_stamp = None
        self._init_control_state()
        self.get_logger().info('Waiting for /joint_states and /rotino/odom ...')

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------
    def _init_control_state(self):
        self.jump_state = STATE_BALANCE
        self.jump_done = False
        self.loaded_state = True
        self.state_t0 = 0.0

        self.prev_com_z = None
        self.com_z_vel_filtered = 0.0
        self.prev_s = None
        self.prev_theta = None
        self.x = 0.0
        self.xdot = 0.0
        self.yaw0 = None
        self.prev_yaw = 0.0
        self.prev_axle_xy = None
        self.axle_origin = None
        self.a_y_meas = 0.0
        self.yaw_i = 0.0
        self.roll_i = 0.0
        self.yaw_hold = 0.0
        self.tele_s = self.tele_v = self.tele_a = self.tele_w = self.tele_h = 0.0

        self.unloaded_timer = 0.0
        self.loaded_timer = 0.0

        self.x_ref_active = 0.0
        self.last_print = -math.inf
        self.push_request = None
        self.push_ticks_left = 0
        self.push_force = 0.0
        self.scripted_push_done = False
        self.zmp_meas = None

        self.hip_flight_hold = (0.0, 0.0)
        self.knee_flight_hold = (0.0, 0.0)
        self.flight_hold_valid = False
        self.landing_start_hip = (0.0, 0.0)
        self.landing_start_knee = (0.0, 0.0)
        self.thrust_start_time = None

        self._reset_jump_metrics(0.0, 0.0)

    def _reset_jump_metrics(self, com_z, min_wheel_gap):
        self.stand_com_z = com_z
        self.preload_min_com_z = com_z
        self.wheel_gap_ref = None
        self.peak_gap_abs = min_wheel_gap
        self.peak_gap_rel = 0.0
        self.peak_com_z = com_z
        self.peak_air_com_z = -math.inf
        self.takeoff_time = None
        self.takeoff_com_z = None
        self.takeoff_com_z_vel = None
        self.landing_time = None
        self.flight_time = 0.0
        self.jump_height_airborne = 0.0
        self.jump_height_above_stand_air = 0.0
        self.v_takeoff_from_air = 0.0
        self.thrust_force_sum = 0.0
        self.thrust_force_count = 0
        self.average_thrust_force = 0.0
        self.settle_timer = 0.0
        self.settle_time = None
        self.wheel_sat_count = 0
        self.leg_sat_count = 0
        self.max_abs_wheel_u = 0.0
        self.max_abs_xdot_after_landing = 0.0
        self.max_abs_theta_dot_after_landing = 0.0

    def _set_state(self, state, t):
        self.jump_state = state
        self.state_t0 = t
        self.state_pub.publish(String(data=state))

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------
    def _odom_cb(self, msg):
        key = stamp_to_ns(msg.header.stamp)
        self.latest_odom = msg
        self.odom_buffer[key] = msg
        self._try_control(key, from_joint_state=False)

    def _imu_cb(self, msg):
        self.imu = msg

    def _contact_cb(self, msg, side):
        normal_force = 0.0
        ground_contact = False
        has_wrench = False
        for contact in msg.contacts:
            if 'ground' not in contact.collision1.name and 'ground' not in contact.collision2.name:
                continue
            ground_contact = True
            for wrench in contact.wrenches:
                has_wrench = True
                f = wrench.body_1_wrench.force
                normal_force += math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        if not ground_contact:
            return
        if not has_wrench:
            # Physics engine did not report forces: treat a touching wheel as carrying half the weight.
            normal_force = 0.5 * self.robot_weight
            if not self.warned_no_wrench:
                self.warned_no_wrench = True
                self.get_logger().warn('Contact messages carry no wrench; using weight-based contact force.')
        stamp = stamp_to_sec(msg.header.stamp)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self.contact[side] = (stamp, normal_force)

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg
        key = stamp_to_ns(msg.header.stamp)
        self.joint_state_buffer[key] = msg
        self._try_control(key, from_joint_state=True)

    def _try_control(self, key, from_joint_state):
        # Joint states and odometry arrive through different transports: only pair samples of the same
        # sim step, otherwise the torso pose lags the joints by a step and the COM lean rate gets noisy.
        for buffer in (self.joint_state_buffer, self.odom_buffer):
            if len(buffer) > SYNC_BUFFER_SIZE:
                del buffer[min(buffer)]
                if not self.warned_no_sync:
                    self.warned_no_sync = True
                    self.get_logger().warn(
                        '/joint_states and /rotino/odom stamps do not coincide: the odometry publish '
                        'frequency must equal the controller_manager update_rate. '
                        f'JS buffer: {len(self.joint_state_buffer)}, Odom buffer: {len(self.odom_buffer)}')
        if key not in self.joint_state_buffer or key not in self.odom_buffer:
            # Fallback: stamps never coincide. Step only on the ground-truth odometry (500 Hz) so each pose is
            # processed once with its exact stamp, paired with the newest joint state.
            if from_joint_state:
                return
            if self.latest_joint_state is not None and self.latest_odom is not None:
                self.sync_count += 1
                if self.sync_count == 1:
                    self.get_logger().warn('Timestamp sync failed, falling back to latest-of-each')
                msg = self.latest_joint_state
                self.odom = self.latest_odom
            else:
                return
        else:
            # Perfect sync
            msg = self.joint_state_buffer.pop(key)
            self.odom = self.odom_buffer.pop(key)
            for buffer in (self.joint_state_buffer, self.odom_buffer):
                for old_key in [k for k in buffer if k < key]:
                    del buffer[old_key]

        t_abs = key * 1e-9
        if self.last_stamp is None:
            self.last_stamp = t_abs
            return
        dt = t_abs - self.last_stamp
        if dt <= 0.0:
            return
        self.last_stamp = t_abs

        q = dict(zip(msg.name, msg.position))
        qd = dict(zip(msg.name, msg.velocity))
        if any(name not in q for name in LEG_JOINTS + WHEEL_JOINTS):
            return

        if self.phase == 'HOLD':
            self._hold_step(t_abs, q, qd)
        else:
            self._control_step(t_abs, dt, q, qd)

    # -----------------------------------------------------------------
    # Actuation
    # -----------------------------------------------------------------
    def _publish_legs(self, targets, q, qd, kp, kd, support):
        """Cartesian PD of each wheel centre in the hip frame, sent as joint torques.

        targets: ((x_L, z_L), (x_R, z_R)) wheel centre w.r.t. the hip in base_link; support: vertical force
        each leg carries as feedforward [N] (weight share in stance, 0 in flight). Joint order of
        leg_effort_controller: [hip_L, hip_R, knee_L, knee_R]. The feedforward also cancels the URDF joint
        damping, as the MPC does.
        """
        up_b = self._base_pose()[1].T @ np.array([0.0, 0.0, 1.0])
        f_ff = -support * np.array([up_b[0], up_b[2]])
        taus = []
        for side, p_d in zip(('left', 'right'), targets):
            qs = np.array([q[f'{side}_hip'], q[f'{side}_knee']])
            qds = np.array([qd[f'{side}_hip'], qd[f'{side}_knee']])
            p, J = self.wbr.leg_fk(*qs)
            F = kp @ (np.asarray(p_d) - p) + kd @ (-(J @ qds)) + f_ff
            taus.append(np.clip(J.T @ F + self.wbr.p.leg_damping * qds, -LEG_TORQUE_MAX, LEG_TORQUE_MAX))
        tau = [taus[0][0], taus[1][0], taus[0][1], taus[1][1]]
        self.leg_pub.publish(Float64MultiArray(data=[float(x) for x in tau]))
        return tau

    def _joint_targets_to_feet(self, targets):
        """[hip_L, hip_R, knee_L, knee_R] joint targets of the jump sequence -> wheel-centre targets."""
        return (self.wbr.leg_fk(targets[0], targets[2])[0], self.wbr.leg_fk(targets[1], targets[3])[0])

    def _publish_wheels(self, tau_l, tau_r):
        self.wheel_pub.publish(Float64MultiArray(data=[float(tau_l), float(tau_r)]))

    def _hold_step(self, t_abs, q, qd):
        if self.hold_start is None:
            self.hold_start = t_abs
            self.get_logger().info('Holding legs on the anchor ...')
        stand = np.array([0.0, -self.leg_length0])
        self._publish_legs((stand, stand), q, qd, LEG_KP_STAND, LEG_KD_STAND, 0.0)
        self._publish_wheels(0.0, 0.0)
        if t_abs - self.hold_start < HOLD_TIME:
            return

        self.release_pub.publish(Empty())
        self.phase = 'RUN'
        self.t_release = t_abs
        self._set_state(STATE_BALANCE, 0.0)
        self.get_logger().info(f'Released at sim t={t_abs:.3f}s. Balance controller active.')

    def _push(self, t, heading):
        """Horizontal impulse on the torso, as in the MPC: push_impulse backwards along the heading."""
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
        msg.wrench.force.x = float(self.push_force * heading[0])
        msg.wrench.force.y = float(self.push_force * heading[1])
        self.wrench_pub.publish(msg)

    # -----------------------------------------------------------------
    # References
    # -----------------------------------------------------------------
    def _path(self, times):
        """Planned contact (= longitudinal ZMP) path along the heading: (s, s_dot) arrays, or None."""
        c = self.cfg
        tm = np.asarray(times, float) - self.motion_start
        if self.planar_enable:
            tau = np.clip(tm / self.trajectory.duration, 0.0, 1.0)
            L, T = self.trajectory.length, self.trajectory.duration
            return (L * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5),
                    L * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T)
        if c['velocity_enable']:
            return trapezoid_path(tm, c['velocity_max'], c['accel_max'], c['velocity_distance'])
        if c['drive_enable']:
            start = self.motion_start if c['wait_start'] else c['drive_start_time']
            ts = np.maximum(np.asarray(times, float) - start, 0.0)
            w = 2.0 * math.pi / c['drive_period']
            return 0.5 * c['drive_distance'] * (1.0 - np.cos(w * ts)), 0.5 * c['drive_distance'] * w * np.sin(w * ts)
        return None

    def _leg_length_ref(self, t):
        c = self.cfg
        th = t - self.motion_start
        if self.teleop:
            return self.leg_length0 + self.tele_h
        if c['height_enable'] and th > 0.0 and self.jump_state == STATE_BALANCE and not self.jump_done:
            return self.leg_length0 + c['height_amplitude'] * math.sin(2.0 * math.pi * th / c['height_period'])
        return self.leg_length0

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
        self.cmd_v = clamp(msg.linear.x, -CMD_V_MAX, CMD_V_MAX)
        self.cmd_w = clamp(msg.angular.z, -CMD_W_MAX, CMD_W_MAX)
        self.cmd_wall = time.monotonic()
        self._start_teleop()

    def _cmd_height_cb(self, msg):
        self.cmd_height = clamp(msg.data, CMD_HEIGHT_MIN, CMD_HEIGHT_MAX)
        self._start_teleop()

    def _cmd_jump_cb(self, _msg):
        if self.phase == 'RUN' and self.jump_state == STATE_BALANCE and not self.jump_request:
            self.jump_request = True
            self.get_logger().info('Jump requested: stopping the robot first')

    def _cmd_push_cb(self, msg):
        self.push_request = clamp(msg.data, -CMD_PUSH_MAX, CMD_PUSH_MAX)

    def _update_teleop(self, dt, x, yaw):
        """Integrates the commanded speed / turn rate into the axle-travel and heading references."""
        if not self.teleop:
            return
        if self.jump_state != STATE_BALANCE:
            self.teleop_ready = False                  # restart from where the robot lands
            return
        if not self.teleop_ready:                      # bumpless takeover
            self.tele_s, self.yaw_hold = x, yaw
            self.tele_v = self.tele_a = self.tele_w = 0.0
            self.teleop_ready = True
        fresh = time.monotonic() - self.cmd_wall < CMD_TIMEOUT
        v_target = self.cmd_v if fresh and not self.jump_request else 0.0
        w_target = self.cmd_w if fresh and not self.jump_request else 0.0
        a = self.cfg['accel_max'] * dt
        dv = clamp(v_target - self.tele_v, -a, a)
        self.tele_a += dt / (CMD_ACC_FILTER + dt) * (dv / dt - self.tele_a)
        self.tele_v += dv
        self.tele_s += self.tele_v * dt
        self.tele_w += clamp(w_target - self.tele_w, -CMD_YAW_ACC * dt, CMD_YAW_ACC * dt)
        self.yaw_hold += self.tele_w * dt
        lag = wrap_angle(self.yaw_hold - yaw)
        if abs(lag) > YAW_ERROR_MAX:
            self.yaw_hold = yaw + math.copysign(YAW_ERROR_MAX, lag)
        self.tele_h += clamp(self.cmd_height - self.tele_h, -CMD_HEIGHT_RATE * dt, CMD_HEIGHT_RATE * dt)

    def _yaw_control(self, dt, yaw_err, yaw_rate_ref, yaw_rate):
        """Heading PID with friction feedforward -> differential wheel torque [Nm] (+ = turn left)."""
        if abs(yaw_err) > YAW_I_DEADBAND or yaw_rate_ref != 0.0:
            self.yaw_i = clamp(self.yaw_i + YAW_KI * yaw_err * dt, -YAW_I_MAX, YAW_I_MAX)
        ff = YAW_FRICTION * math.tanh(yaw_rate_ref / YAW_FRICTION_SMOOTH) + YAW_VISCOUS * yaw_rate_ref
        return clamp(K_PSI * yaw_err + K_OMEGA * (yaw_rate_ref - yaw_rate) + ff + self.yaw_i, -MAX_YAW, MAX_YAW)

    def _planar_tracking(self, t, dt, axle_xy, yaw, forward_speed, yaw_rate):
        """Track the planar trajectory: returns (along-track pos error, speed error, differential torque)."""
        s = t - self.motion_start
        px, py, heading_ref, v_ref, yaw_rate_ref = self.trajectory.sample(max(s, 0.0))
        if s <= 0.0:
            v_ref = yaw_rate_ref = 0.0
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        p_ref = self.axle_origin[:2] + np.array([c0 * px - s0 * py, s0 * px + c0 * py])
        yaw_ref = self.yaw0 + heading_ref

        err = axle_xy - p_ref
        tangent = np.array([math.cos(yaw_ref), math.sin(yaw_ref)])
        normal = np.array([-math.sin(yaw_ref), math.cos(yaw_ref)])
        along_err = float(np.dot(err, tangent))
        lateral_err = float(np.dot(err, normal))

        # A differential-drive robot can only cancel lateral error while moving: fade the correction out near standstill.
        yaw_cmd = yaw_ref - math.atan(K_LAT * lateral_err) * min(1.0, abs(v_ref) / LAT_FULL_SPEED)
        yaw_u = self._yaw_control(dt, wrap_angle(yaw_cmd - yaw), yaw_rate_ref, yaw_rate)

        self.planar_pub.publish(Float64MultiArray(data=[
            t, float(p_ref[0]), float(p_ref[1]), yaw_ref, float(axle_xy[0]), float(axle_xy[1]), yaw,
            along_err, lateral_err, yaw_u]))
        self.tracking_error_pub.publish(Float64MultiArray(data=[float(err[0]), float(err[1])]))
        return along_err, forward_speed - v_ref, yaw_u, yaw_ref

    def _centripetal_path(self, times):
        """Leftward centripetal acceleration of the planned S-curve, curvature * v^2 (vectorised sample())."""
        traj = self.trajectory
        tau = np.clip((np.asarray(times, float) - self.motion_start) / traj.duration, 0.0, 1.0)
        s = traj.length * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        v = traj.length * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / traj.duration
        x = np.interp(s, traj.s_table, traj.x_table)
        dy = traj._dy(x)
        return traj._ddy(x) / (1.0 + dy * dy) ** 1.5 * v * v

    def _lateral_shift_ref(self, t, dt, h, forward_speed, yaw_rate):
        """CoM shift to the left that keeps the lateral ZMP centred (and the leftward acceleration it
        answers): LIPM preview of h a_y / g along the planned turn, or the measured v * yaw_rate
        (filtered, causal) when there is no trajectory."""
        a = dt / (AY_FILTER + dt)
        self.a_y_meas += a * (forward_speed * yaw_rate - self.a_y_meas)
        if self.planar_enable and not self.teleop:
            omega = math.sqrt(G / h)
            shift = self.preview.smooth(lambda ts: h * self._centripetal_path(ts) / G, t + LAT_LEAD, omega)
            return shift, float(self._centripetal_path(np.array([t]))[0])
        return h * self.a_y_meas / G, self.a_y_meas

    def _base_pose(self):
        p = self.odom.pose.pose.position
        o = self.odom.pose.pose.orientation
        return np.array([p.x, p.y, p.z]), quat_to_matrix(o.x, o.y, o.z, o.w)

    def _contact_force(self, side, t_abs):
        stamp, force = self.contact[side]
        return force if t_abs - stamp <= CONTACT_STALE_TIME else 0.0

    # -----------------------------------------------------------------
    # Main control step (one MuJoCo loop iteration)
    # -----------------------------------------------------------------
    def _control_step(self, t_abs, dt, q, qd):
        t = t_abs - self.t_release
        if t < RELEASE_REPEAT_TIME:
            self.release_pub.publish(Empty())

        base_pos, base_rot = self._base_pose()
        frames = self.kin.link_frames(base_pos, base_rot, q)
        com = self.kin.center_of_mass(frames)
        wl = frames[LEFT_WHEEL_LINK][1]
        wr = frames[RIGHT_WHEEL_LINK][1]

        # ---------------- measured ZMP (multibody, 25 ms behind) ----------------
        o = self.odom.pose.pose.orientation
        zmp_now = self.zmp_est.update(t_abs, base_pos, (o.x, o.y, o.z, o.w), q)
        if zmp_now is not None:
            self.zmp_meas = zmp_now

        # ---------------- vertical CoM velocity ----------------
        com_z = float(com[2])
        if self.prev_com_z is None:
            self.prev_com_z = com_z
            self._reset_jump_metrics(com_z, min(wl[2], wr[2]) - WHEEL_RADIUS)
        com_z_vel_raw = (com_z - self.prev_com_z) / dt
        self.prev_com_z = com_z
        self.com_z_vel_filtered = COMZ_VEL_FILTER * self.com_z_vel_filtered + (1.0 - COMZ_VEL_FILTER) * com_z_vel_raw

        if self.jump_state == STATE_PRELOAD:
            self.preload_min_com_z = min(self.preload_min_com_z, com_z)

        # ---------------- wheel clearance ----------------
        min_wheel_gap = min(float(wl[2]), float(wr[2])) - WHEEL_RADIUS
        gap_rel = 0.0 if self.wheel_gap_ref is None else min_wheel_gap - self.wheel_gap_ref
        if self.jump_state in JUMP_ACTIVE_STATES:
            self.peak_gap_abs = max(self.peak_gap_abs, min_wheel_gap)
            self.peak_gap_rel = max(self.peak_gap_rel, gap_rel)
        wheel_clear_air = (min_wheel_gap > WHEEL_CLEARANCE_TAKEOFF_ABS
                           and gap_rel > WHEEL_CLEARANCE_TAKEOFF_REL)

        # ---------------- longitudinal ZMP offset: CoM ahead of the wheel axle ----------------
        wheel_mid = 0.5 * (wl + wr)
        heading = base_rot[:, 0].copy()
        heading[2] = 0.0
        heading /= max(np.linalg.norm(heading), 1e-9)
        s_c = float(np.dot(com - wheel_mid, heading))
        dz = float(com[2] - wheel_mid[2])
        theta = math.atan2(s_c, max(dz, 1e-6))          # CoM lean from the vertical, + forward (as the MPC)
        if self.prev_s is None:
            self.prev_s, self.prev_theta = s_c, theta
        # Ideal ground-truth state: exact derivatives on simulator stamps, no filtering.
        s_c_dot = (s_c - self.prev_s) / dt
        theta_dot = (theta - self.prev_theta) / dt
        self.prev_s, self.prev_theta = s_c, theta

        # ---------------- axle travel along the heading, yaw ----------------
        yaw = math.atan2(heading[1], heading[0])
        axle_xy = wheel_mid[:2].copy()
        if self.yaw0 is None:
            self.yaw0 = yaw
            self.yaw_hold = yaw
            self.prev_yaw = yaw
            self.prev_axle_xy = axle_xy
            self.axle_origin = wheel_mid.copy()
        forward_speed = float(np.dot((axle_xy - self.prev_axle_xy) / dt, heading[:2]))
        self.prev_axle_xy = axle_xy
        self.x += forward_speed * dt
        self.xdot = forward_speed
        x, xdot = self.x, self.xdot
        yaw_rate = wrap_angle(yaw - self.prev_yaw) / dt
        self.prev_yaw = yaw

        # ---------------- contact force hysteresis ----------------
        fn_total = self._contact_force('left', t_abs) + self._contact_force('right', t_abs)
        if self.jump_state == STATE_THRUST:
            self.thrust_force_sum += fn_total
            self.thrust_force_count += 1

        if self.loaded_state:
            self.unloaded_timer = self.unloaded_timer + dt if fn_total < self.f_contact_off else 0.0
            if self.unloaded_timer >= CONTACT_LOSS_DEBOUNCE:
                self.loaded_state = False
                self.loaded_timer = 0.0
        else:
            self.loaded_timer = self.loaded_timer + dt if fn_total > self.f_contact_on else 0.0
            if self.loaded_timer >= CONTACT_GAIN_DEBOUNCE:
                self.loaded_state = True
                self.unloaded_timer = 0.0
        airborne = not self.loaded_state

        if self.jump_state in JUMP_ACTIVE_STATES:
            self.peak_com_z = max(self.peak_com_z, com_z)
        if self.jump_state == STATE_FLIGHT:
            self.peak_air_com_z = max(self.peak_air_com_z, com_z)

        # ---------------- references ----------------
        self._update_start(t)
        self._update_teleop(dt, x, yaw)
        teleop = self.teleop_ready and self.jump_state == STATE_BALANCE
        scripted = self.jump_state == STATE_BALANCE and not self.jump_done and not self.teleop
        x_ref = v_ref = acc_ref = jerk_ref = 0.0
        if self.jump_state != STATE_BALANCE or self.jump_done:
            x_ref = self.x_ref_active
        path = self._path if scripted else None
        h = com_z                                       # CoM height above the (flat) ground
        omega = math.sqrt(G / max(h, 0.03))
        if teleop:
            # live commands: the future is unknown, so no preview; the ZMP feedforward uses the current
            # (rate-limited) acceleration of the reference
            x_ref, v_ref, acc_ref = self.tele_s, self.tele_v, self.tele_a
        elif path is not None and path(np.array([t])) is not None:
            (x_ref,), (v_ref,) = path(np.array([t]))
            acc_ref, jerk_ref = self.preview(path, t, omega)
        pos_err = x - x_ref
        vel_err = xdot - v_ref

        yaw_ref = self.yaw_hold
        if self.planar_enable and scripted:
            pos_err, vel_err, yaw_u, yaw_ref = self._planar_tracking(t, dt, axle_xy, yaw, forward_speed, yaw_rate)
        elif teleop:
            yaw_u = self._yaw_control(dt, wrap_angle(self.yaw_hold - yaw), self.tele_w, yaw_rate)
        elif self.jump_state == STATE_BALANCE:
            yaw_u = self._yaw_control(dt, wrap_angle(self.yaw_hold - yaw), 0.0, yaw_rate)
        else:
            self.yaw_i = 0.0
            yaw_u = clamp(K_PSI * wrap_angle(self.yaw_hold - yaw) - K_OMEGA * yaw_rate, -MAX_YAW, MAX_YAW)

        # ---------------- ZMP balance on the wheels ----------------
        info = {'zmp_des': -s_c, 'zmp': -s_c, 'xi_err': 0.0, 's_des': s_c}
        if self.jump_state == STATE_FLIGHT:
            common = 0.0
            tau_l = -FLIGHT_WHEEL_DAMPING * qd['left_wheel_joint']
            tau_r = -FLIGHT_WHEEL_DAMPING * qd['right_wheel_joint']
            wheel_sat = False
        else:
            common, info = self.balance.step(
                dt, h, pos_err, vel_err, s_c, s_c_dot, acc_ref, jerk_ref,
                outer_scale=OUTER_SCALE.get(self.jump_state, 1.0),
                integrate=self.jump_state in (STATE_BALANCE, STATE_SETTLE))
            diff = clamp(yaw_u, -MAX_YAW, MAX_YAW)
            common = clamp(common, -(WHEEL_TORQUE_MAX - abs(diff)), WHEEL_TORQUE_MAX - abs(diff))
            tau_l, tau_r = common - diff, common + diff
            wheel_sat = info['saturated'] or abs(common) >= 0.98 * (WHEEL_TORQUE_MAX - abs(diff))
        wheel_u = common / WHEEL_GEAR

        # ---------------- lateral ZMP: lean into the turn ----------------
        lean_cmd = leg_dz = a_y = 0.0
        y_zmp = None
        if self.zmp_meas is not None and self.zmp_meas['fz'] > 0.2 * self.robot_weight:
            y_zmp = float(self.zmp_meas['y_rel']) * 0.5 * float(self.zmp_meas['track'])
        if self.cfg['zmp_lateral'] and self.jump_state == STATE_BALANCE and not airborne:
            shift, a_y = self._lateral_shift_ref(t, dt, h, forward_speed, yaw_rate)
            lean_cmd, leg_dz, _ = self.lateral.step(dt, h, shift, y_zmp)
            # roll PI: lean = left side down = negative roll about the heading
            lean_meas = -math.atan2(base_rot[2, 1], base_rot[2, 2])
            roll_err = lean_cmd - lean_meas
            self.roll_i = clamp(self.roll_i + ROLL_KI * roll_err * dt, -ROLL_I_MAX, ROLL_I_MAX)
            leg_dz = self.wbr.p.d * math.tan(lean_cmd + ROLL_KP * roll_err + self.roll_i)
        else:
            self.lateral.reset()
            self.roll_i = 0.0

        # ---------------- settling check ----------------
        if self.landing_time is not None and self.jump_state in (STATE_RECOVERY, STATE_SETTLE, STATE_BALANCE):
            settled_now = (self.loaded_state
                           and abs(theta) < SETTLE_LEAN_ERR
                           and abs(theta_dot) < SETTLE_RATE
                           and abs(xdot) < SETTLE_VEL
                           and abs(wheel_u) < SETTLE_WHEEL
                           and abs(x - self.x_ref_active) < SETTLE_X_ERR)
            self.settle_timer = self.settle_timer + dt if settled_now else 0.0
        else:
            self.settle_timer = 0.0

        # ---------------- jump state machine ----------------
        stable_for_jump = (abs(theta) < math.radians(2.5)
                           and abs(xdot) < 0.08
                           and self.loaded_state)
        elapsed = t - self.state_t0

        if self.jump_state == STATE_BALANCE:
            scripted_jump = self.jump_enable and not self.jump_done and t > self.jump_start_time
            if (scripted_jump or self.jump_request) and stable_for_jump:
                self.jump_request = False
                self.x_ref_active = x                   # jump where the robot stands
                self._set_state(STATE_PRELOAD, t)
                self._reset_jump_metrics(com_z, min_wheel_gap)
                self.wheel_gap_ref = min_wheel_gap
                self.get_logger().info(f'>>> JUMP STATE: PRELOAD t={t:.4f}s StandZ={com_z:.4f}')

        elif self.jump_state == STATE_PRELOAD:
            if elapsed >= T_PRELOAD:
                self._set_state(STATE_THRUST, t)
                self.thrust_start_time = t
                self.thrust_force_sum = 0.0
                self.thrust_force_count = 0
                self.get_logger().info(
                    f'>>> JUMP STATE: THRUST t={t:.4f}s COMz={com_z:.4f} PreloadMinZ={self.preload_min_com_z:.4f}')

        elif self.jump_state == STATE_THRUST:
            takeoff_detected = (airborne
                                and fn_total < self.f_contact_off
                                and wheel_clear_air
                                and com_z_vel_raw > TAKEOFF_VZ_MIN)
            if takeoff_detected:
                self._set_state(STATE_FLIGHT, t)
                self.takeoff_time = t
                self.takeoff_com_z = com_z
                self.takeoff_com_z_vel = com_z_vel_raw
                self.peak_air_com_z = com_z
                self.hip_flight_hold = (q['left_hip'], q['right_hip'])
                self.knee_flight_hold = (q['left_knee'], q['right_knee'])
                self.flight_hold_valid = True
                if self.thrust_force_count > 0:
                    self.average_thrust_force = self.thrust_force_sum / self.thrust_force_count
                self.get_logger().info(
                    f'>>> TAKE-OFF DETECTED t={t:.4f}s COMz={com_z:.4f} Vz_raw={com_z_vel_raw:.4f} '
                    f'Fn={fn_total:.2f}N GapAbs={min_wheel_gap:.4f}m GapRel={gap_rel:.4f}m')
            elif elapsed >= T_THRUST_MAX:
                self._set_state(STATE_LANDING, t)
                self.get_logger().info(
                    f'>>> NO TAKE-OFF: SAFE LANDING t={t:.4f}s COMz={com_z:.4f} Fn={fn_total:.2f}N')

        elif self.jump_state == STATE_FLIGHT:
            self.peak_air_com_z = max(self.peak_air_com_z, com_z)
            if self.loaded_state:
                self.landing_start_hip = (q['left_hip'], q['right_hip'])
                self.landing_start_knee = (q['left_knee'], q['right_knee'])
                self._set_state(STATE_LANDING, t)
                self.landing_time = t
                self.x_ref_active = 0.0 if self.return_to_start else x
                self.settle_timer = 0.0
                self.settle_time = None
                self.flight_time = t - self.takeoff_time
                self.jump_height_airborne = max(0.0, self.peak_air_com_z - self.takeoff_com_z)
                self.jump_height_above_stand_air = max(0.0, self.peak_air_com_z - self.stand_com_z)
                self.v_takeoff_from_air = math.sqrt(2.0 * G * self.jump_height_airborne)
                self.get_logger().info(
                    f'>>> LANDING DETECTED t={t:.4f}s FlightTime={self.flight_time:.4f}s '
                    f'TakeoffZ={self.takeoff_com_z:.4f} PeakAirZ={self.peak_air_com_z:.4f} '
                    f'LandingZ={com_z:.4f} H_airborne={self.jump_height_airborne:.4f}m '
                    f'Vto_air_est={self.v_takeoff_from_air:.4f}m/s Favg_thrust={self.average_thrust_force:.2f}N')

        elif self.jump_state == STATE_LANDING:
            if elapsed >= T_LANDING:
                self._set_state(STATE_RECOVERY, t)
                self.get_logger().info('>>> JUMP STATE: RECOVERY')

        elif self.jump_state == STATE_RECOVERY:
            if elapsed >= T_RECOVERY:
                self._set_state(STATE_SETTLE, t)
                self.get_logger().info('>>> JUMP STATE: SETTLE / WAITING FOR STABLE BALANCE')

        elif self.jump_state == STATE_SETTLE:
            if self.settle_time is None and self.settle_timer >= SETTLE_HOLD:
                self.settle_time = t - self.landing_time
            settle_timeout = self.landing_time is not None and (t - self.landing_time) >= MAX_SETTLE_WAIT
            if self.settle_time is not None or settle_timeout:
                self._finish_jump(t)

        # ---------------- leg targets ----------------
        elapsed = t - self.state_t0
        stand = (0.0, 0.0, 0.0, 0.0)  # hip_L, hip_R, knee_L, knee_R of the XML standing pose
        preload = (HIP_PRELOAD_DELTA, HIP_PRELOAD_DELTA, KNEE_PRELOAD_DELTA, KNEE_PRELOAD_DELTA)
        thrust = (HIP_THRUST_DELTA, HIP_THRUST_DELTA, KNEE_THRUST_DELTA, KNEE_THRUST_DELTA)
        land = (HIP_LAND_DELTA, HIP_LAND_DELTA, KNEE_LAND_DELTA, KNEE_LAND_DELTA)

        joint_targets = None
        if self.jump_state == STATE_PRELOAD:
            a = smoothstep01(elapsed, T_PRELOAD)
            joint_targets = [blend(s, p, a) for s, p in zip(stand, preload)]
        elif self.jump_state == STATE_THRUST:
            a = fast_thrust01(elapsed, T_THRUST)
            joint_targets = [blend(p, th, a) for p, th in zip(preload, thrust)]
        elif self.jump_state == STATE_FLIGHT:
            joint_targets = ([*self.hip_flight_hold, *self.knee_flight_hold] if self.flight_hold_valid
                             else list(thrust))
        elif self.jump_state == STATE_LANDING:
            a = smoothstep01(elapsed, T_LANDING)
            start = (*self.landing_start_hip, *self.landing_start_knee)
            joint_targets = [blend(s, ld, a) for s, ld in zip(start, land)]
        elif self.jump_state == STATE_RECOVERY:
            a = smoothstep01(elapsed, T_RECOVERY)
            joint_targets = [blend(ld, s, a) for ld, s in zip(land, stand)]

        if joint_targets is not None:
            joint_targets = [clamp(joint_targets[0], HIP_CTRL_MIN, HIP_CTRL_MAX),
                             clamp(joint_targets[1], HIP_CTRL_MIN, HIP_CTRL_MAX),
                             clamp(joint_targets[2], KNEE_CTRL_MIN, KNEE_CTRL_MAX),
                             clamp(joint_targets[3], KNEE_CTRL_MIN, KNEE_CTRL_MAX)]
            feet = self._joint_targets_to_feet(joint_targets)
            if self.jump_state == STATE_FLIGHT:
                kp, kd, support = LEG_KP_FLIGHT, LEG_KD_FLIGHT, 0.0
            else:
                kp, kd, support = LEG_KP_JUMP, LEG_KD_JUMP, 0.5 * self.upper_weight
        else:
            # standing: wheel under the hip, legs lengthened/shortened to lean into the turn
            z0 = self._leg_length_ref(t)
            feet = (np.array([0.0, -clamp(z0 - 0.5 * leg_dz, LEG_LENGTH_MIN, LEG_LENGTH_MAX)]),
                    np.array([0.0, -clamp(z0 + 0.5 * leg_dz, LEG_LENGTH_MIN, LEG_LENGTH_MAX)]))
            kp, kd, support = LEG_KP_STAND, LEG_KD_STAND, 0.5 * self.upper_weight
        leg_torques = self._publish_legs(feet, q, qd, kp, kd, support)
        self._publish_wheels(tau_l, tau_r)
        self._push(t, heading)
        leg_sat = any(abs(tau) > 0.98 * LEG_TORQUE_MAX for tau in leg_torques)

        if self.jump_state in POST_LANDING_STATES:
            self.wheel_sat_count += int(wheel_sat)
            self.leg_sat_count += int(leg_sat)
            self.max_abs_wheel_u = max(self.max_abs_wheel_u, abs(wheel_u))
            self.max_abs_xdot_after_landing = max(self.max_abs_xdot_after_landing, abs(xdot))
            self.max_abs_theta_dot_after_landing = max(self.max_abs_theta_dot_after_landing, abs(theta_dot))

        # ---------------- telemetry ----------------
        self.debug_pub.publish(Float64MultiArray(data=[
            t, theta, theta_dot, x, xdot, wheel_u, com_z, com_z_vel_raw, fn_total, min_wheel_gap,
            float(self.loaded_state)]))
        theta_ref = math.atan2(info['s_des'], max(dz, 1e-6))
        self.wbr_pub.publish(Float64MultiArray(data=[
            t, x, x_ref, theta, theta_ref, yaw, yaw_ref, xdot, v_ref, dz, dz, info['s_des'],
            self.upper_weight, float(tau_l), float(tau_r), math.hypot(s_c, dz),
            float(leg_torques[0]), float(leg_torques[2])]))
        zm = self.zmp_meas
        nan = float('nan')
        self.zmp_pub.publish(Float64MultiArray(data=[
            t, info['zmp_des'], info['zmp'], info['xi_err'], acc_ref, a_y, lean_cmd, leg_dz,
            nan if y_zmp is None else y_zmp,
            nan if zm is None else float(zm['margin']), nan if zm is None else float(zm['e_long'])]))

        if t - self.last_print >= PRINT_DT.get(self.jump_state, 0.2):
            self.last_print = t
            if wheel_sat:
                status = 'WHEEL_SAT'
            elif leg_sat:
                status = 'LEG_SAT'
            elif abs(theta) < 0.035 and abs(xdot) < 0.08:
                status = 'BALANCING'
            elif abs(xdot) > 1.0:
                status = 'FAST MOTION'
            else:
                status = 'correcting'
            self.get_logger().info(
                f't={t:6.2f} {self.jump_state:>8} CoMLean={math.degrees(theta):+6.2f}deg '
                f'Rate={math.degrees(theta_dot):+7.1f}deg/s x={x:+6.3f}/{x_ref:+6.3f} v={xdot:+6.3f} '
                f'ZMP={info["zmp"] * 1e3:+5.1f}/{info["zmp_des"] * 1e3:+5.1f}mm '
                f'yZMP={float("nan") if y_zmp is None else y_zmp * 1e3:+5.1f}mm lean={math.degrees(lean_cmd):+4.1f}deg '
                f'tau={tau_l:+5.2f}/{tau_r:+5.2f} COMz={com_z:.4f} Fn={fn_total:6.2f}N '
                f'loaded={int(self.loaded_state)} {status}')

    def _finish_jump(self, t):
        settle_failed = self.settle_time is None
        jump_height_total = max(0.0, self.peak_com_z - self.stand_com_z)
        jump_height_from_crouch = max(0.0, self.peak_com_z - self.preload_min_com_z)
        visible_jump = self.peak_gap_abs >= 0.010 and self.jump_height_above_stand_air >= 0.010

        self._set_state(STATE_BALANCE, t)
        self.jump_done = JUMP_ONCE
        self.balance.reset()
        self.get_logger().info('>>> JUMP STATE: BALANCE / SETTLED')
        self.get_logger().info(
            f'>>> FINAL JUMP SUMMARY FlightTime={self.flight_time:.4f}s '
            f'H_airborne={self.jump_height_airborne:.4f}m H_total={jump_height_total:.4f}m '
            f'H_from_crouch={jump_height_from_crouch:.4f}m PeakGapAbs={self.peak_gap_abs:.4f}m '
            f'PeakGapRel={self.peak_gap_rel:.4f}m Vto_air_est={self.v_takeoff_from_air:.4f}m/s '
            f'Favg_thrust={self.average_thrust_force:.2f}N '
            f'SettleTime={-1.0 if settle_failed else self.settle_time:.4f} TimedOut={int(settle_failed)} '
            f'WheelSatCount={self.wheel_sat_count} LegSatCount={self.leg_sat_count} '
            f'MaxWheelU={self.max_abs_wheel_u:.3f} MaxPostLandVel={self.max_abs_xdot_after_landing:.3f} '
            f'MaxPostLandRate={math.degrees(self.max_abs_theta_dot_after_landing):.1f}deg/s '
            f'SUCCESS_VISIBLE={int(visible_jump)}')


def main(args=None):
    rclpy.init(args=args)
    node = BalanceJumpController()
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
