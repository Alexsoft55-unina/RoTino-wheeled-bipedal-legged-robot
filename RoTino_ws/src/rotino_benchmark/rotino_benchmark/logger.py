"""RoTino test bench logger: one labeled CSV row per control step (500 Hz), ready for PlotJuggler."""

import csv
import math
import os
from datetime import datetime

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts, Entity, EntityWrench
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String

JOINT_NAMES = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
               'left_wheel_joint', 'right_wheel_joint']

CONTACT_STALE_TIME = 0.02

HEADER = [
    'time_s', 'jump_state',
    'theta_deg', 'theta_dot_degs', 'x_m', 'xdot_ms', 'wheel_u',
    'com_z_m', 'com_z_vel_ms', 'fn_total_N', 'fn_left_N', 'fn_right_N',
    'min_wheel_gap_m', 'loaded',
    'hip_L_pos', 'hip_R_pos', 'knee_L_pos', 'knee_R_pos',
    'wheel_L_pos', 'wheel_R_pos',
    'hip_L_vel', 'hip_R_vel', 'knee_L_vel', 'knee_R_vel',
    'wheel_L_vel', 'wheel_R_vel',
    'wheel_L_torque_cmd', 'wheel_R_torque_cmd',
    'hip_L_torque_cmd', 'hip_R_torque_cmd', 'knee_L_torque_cmd', 'knee_R_torque_cmd',
    'roll_deg', 'yaw_deg',
    'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',
    'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
    'theta_ref_deg', 'theta_err_deg',
    's_ref_m', 's_err_m',
    'xdot_ref_ms', 'xdot_err_ms',
    'com_z_ref_m', 'com_z_err_mm',
    'delta_s_m', 'push_force_N',
    'base_x_m', 'base_y_m', 'base_z_m', 'base_qx', 'base_qy', 'base_qz', 'base_qw',
    'base_vx', 'base_vy', 'base_vz', 'base_wx', 'base_wy', 'base_wz',
    'contact_L_x', 'contact_L_y', 'contact_L_z', 'contact_R_x', 'contact_R_y', 'contact_R_z',
    'odom_stamp_s', 'joints_stamp_s',
    'zmp_des_mm', 'zmp_ctrl_mm', 'dcm_err_mm', 'acc_ref_ms2', 'lean_cmd_deg',
    'step_force_N',
    'yaw_ref_deg', 'yaw_err_deg', 'com_z_axle_m', 'com_z_axle_err_mm',
    'est_err_x_m', 'est_err_y_m', 'est_err_z_m', 'est_err_vx_ms', 'est_err_vy_ms', 'est_err_vz_ms',
]

FLUSH_EVERY = 100


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def contact_force_from_msg(msg):
    """Sum of contact-wrench force magnitudes and mean contact point for wheel/ground contacts."""
    total = 0.0
    ground_contact = False
    points = []
    for contact in msg.contacts:
        if 'ground' not in contact.collision1.name and 'ground' not in contact.collision2.name:
            continue
        ground_contact = True
        for wrench in contact.wrenches:
            f = wrench.body_1_wrench.force
            total += math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
        points += [(p.x, p.y, p.z) for p in contact.positions]
    point = [sum(c) / len(points) for c in zip(*points)] if points else None
    return total, ground_contact, point


class TestBenchLogger(Node):

    def __init__(self):
        super().__init__('rotino_test_bench_logger')
        pkg_share = get_package_share_directory('rotino_description')
        self.declare_parameter('controller', 'unknown')
        self.declare_parameter('output_dir', os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_share)))),
            'test_bench_logs'))
        output_dir = str(self.get_parameter('output_dir').value)
        os.makedirs(output_dir, exist_ok=True)

        law = str(self.get_parameter('controller').value)
        filename = f"rotino_{law}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.csv_path = os.path.join(output_dir, filename)
        self._file = open(self.csv_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)
        self._row_count = 0

        self.jump_state = 'UNKNOWN'
        self.joint_pos = {}
        self.joint_vel = {}
        self.wheel_cmd = [float('nan'), float('nan')]
        self.leg_cmd = [float('nan'), float('nan'), float('nan'), float('nan')]
        self.roll = 0.0
        self.yaw = 0.0
        self.contact = {'left': (-math.inf, 0.0), 'right': (-math.inf, 0.0)}
        self.contact_point = {'left': (-math.inf, None), 'right': (-math.inf, None)}
        self.odom = None
        self.odom_stamp = float('nan')
        self.joints_stamp = float('nan')
        self.imu = None
        self.wbr_data = None
        self.zmp_ctrl = None
        self.est_err = None
        self.push_force = 0.0
        self.last_wrench_stamp = -math.inf
        self.step_force = 0.0

        self.create_subscription(Float64MultiArray, '/rotino/debug', self._debug_cb, 10)
        self.create_subscription(Float64MultiArray, '/rotino/wbr_state', self._wbr_cb, 10)
        self.create_subscription(Float64MultiArray, '/rotino/zmp_ctrl', self._zmp_ctrl_cb, 10)
        self.create_subscription(EntityWrench, '/world/rotino_world/wrench', self._wrench_cb, 10)
        self.create_subscription(EntityWrench, '/world/rotino_world/wrench/persistent', self._step_cb, 10)
        self.create_subscription(Entity, '/world/rotino_world/wrench/clear', self._step_clear_cb, 10)
        self.create_subscription(String, '/rotino/jump_state', self._jump_state_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(Float64MultiArray, '/wheel_effort_controller/commands',
                                 self._wheel_cmd_cb, 10)
        self.create_subscription(Float64MultiArray, '/leg_effort_controller/commands',
                                 self._leg_cmd_cb, 10)
        self.create_subscription(Contacts, '/rotino/left_wheel_contact',
                                 lambda msg: self._contact_cb(msg, 'left'), 10)
        self.create_subscription(Contacts, '/rotino/right_wheel_contact',
                                 lambda msg: self._contact_cb(msg, 'right'), 10)
        self.create_subscription(Imu, '/rotino/imu', self._imu_cb, 10)
        self.create_subscription(Odometry, '/rotino/odom', self._odom_cb, 10)
        self.declare_parameter('extra_topics', False)
        if self.get_parameter('extra_topics').value:
            self.create_subscription(Float64MultiArray, '/rotino/estimation_error', self._est_err_cb, 10)
            self.get_logger().info('Extra topic on: /rotino/estimation_error')

        self.get_logger().info(f'Test bench logging to: {self.csv_path}')

    def _wbr_cb(self, msg):
        if len(msg.data) >= 12:
            self.wbr_data = {
                's_ref': msg.data[2],
                'theta_ref': msg.data[4],
                'yaw': msg.data[5],
                'yaw_ref': msg.data[6],
                'sd_ref': msg.data[8],
                'z': msg.data[9],
                'z_ref': msg.data[10],
                'delta_s': msg.data[11],
            }

    def _zmp_ctrl_cb(self, msg):
        if len(msg.data) >= 7:
            d = msg.data
            self.zmp_ctrl = [1e3 * d[1], 1e3 * d[2], 1e3 * d[3], d[4], math.degrees(d[6])]

    def _est_err_cb(self, msg):
        if len(msg.data) >= 6:
            self.est_err = list(msg.data[:6])

    def _wrench_cb(self, msg):
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        self.push_force = math.hypot(fx, fy)
        self.last_wrench_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _step_cb(self, msg):
        self.step_force += math.hypot(msg.wrench.force.x, msg.wrench.force.y)

    def _step_clear_cb(self, _msg):
        self.step_force = 0.0

    def _jump_state_cb(self, msg):
        self.jump_state = msg.data

    def _joint_state_cb(self, msg):
        self.joints_stamp = stamp_to_sec(msg.header.stamp)
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            self.joint_pos[name] = pos
            self.joint_vel[name] = vel

    def _wheel_cmd_cb(self, msg):
        if len(msg.data) >= 2:
            self.wheel_cmd = [msg.data[0], msg.data[1]]

    def _leg_cmd_cb(self, msg):
        if len(msg.data) >= 4:
            self.leg_cmd = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]

    def _contact_cb(self, msg, side):
        force, ground_contact, point = contact_force_from_msg(msg)
        if not ground_contact:
            return
        stamp = stamp_to_sec(msg.header.stamp)
        self.contact[side] = (stamp, force)
        if point is not None:
            self.contact_point[side] = (stamp, point)

    def _odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        self.odom = [p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z]
        self.odom_stamp = stamp_to_sec(msg.header.stamp)

    def _imu_cb(self, msg):
        self.imu = msg
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)
        
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _contact_force(self, side, now_s):
        stamp, force = self.contact[side]
        return force if now_s - stamp <= CONTACT_STALE_TIME else 0.0

    def _contact_xyz(self, side, now_s):
        stamp, point = self.contact_point[side]
        return list(point) if point is not None and now_s - stamp <= CONTACT_STALE_TIME else [float('nan')] * 3

    def _debug_cb(self, msg):
        (t, theta, theta_dot, x, xdot, wheel_u, com_z, com_z_vel,
         fn_total, min_wheel_gap, loaded) = msg.data

        now_s = self.get_clock().now().nanoseconds * 1e-9
        fn_left = self._contact_force('left', now_s)
        fn_right = self._contact_force('right', now_s)

        jp = self.joint_pos
        jv = self.joint_vel
        nan = float('nan')

        if self.imu is not None:
            gyro = self.imu.angular_velocity
            accel = self.imu.linear_acceleration
            imu_row = [gyro.x, gyro.y, gyro.z, accel.x, accel.y, accel.z]
        else:
            imu_row = [nan] * 6

        if self.wbr_data is not None:
            th_ref_deg = math.degrees(self.wbr_data['theta_ref'])
            th_err_deg = math.degrees(theta) - th_ref_deg
            s_ref = self.wbr_data['s_ref']
            s_err = x - s_ref
            sd_ref = self.wbr_data['sd_ref']
            sd_err = xdot - sd_ref
            z_ref = self.wbr_data['z_ref']
            z_err = (com_z - z_ref) * 1000.0
            delta_s = self.wbr_data['delta_s']
        else:
            th_ref_deg = 0.0
            th_err_deg = math.degrees(theta)
            s_ref = 0.0
            s_err = x
            sd_ref = 0.0
            sd_err = xdot
            z_ref = nan
            z_err = nan
            delta_s = nan

        w = self.wbr_data
        if w is not None:
            yaw_err = math.atan2(math.sin(w['yaw'] - w['yaw_ref']), math.cos(w['yaw'] - w['yaw_ref']))
            wbr_row = [math.degrees(w['yaw_ref']), math.degrees(yaw_err), w['z'], (w['z'] - w['z_ref']) * 1000.0]
        else:
            wbr_row = [nan] * 4

        push_f = self.push_force if (now_s - self.last_wrench_stamp < 0.05) else 0.0
        errors_row = [
            th_ref_deg, th_err_deg,
            s_ref, s_err,
            sd_ref, sd_err,
            z_ref, z_err,
            delta_s, push_f
        ]

        row = [
            t, self.jump_state,
            math.degrees(theta), math.degrees(theta_dot), x, xdot, wheel_u,
            com_z, com_z_vel, fn_total, fn_left, fn_right,
            min_wheel_gap, int(loaded),
            jp.get('left_hip', nan), jp.get('right_hip', nan),
            jp.get('left_knee', nan), jp.get('right_knee', nan),
            jp.get('left_wheel_joint', nan), jp.get('right_wheel_joint', nan),
            jv.get('left_hip', nan), jv.get('right_hip', nan),
            jv.get('left_knee', nan), jv.get('right_knee', nan),
            jv.get('left_wheel_joint', nan), jv.get('right_wheel_joint', nan),
            self.wheel_cmd[0], self.wheel_cmd[1],
            self.leg_cmd[0], self.leg_cmd[1], self.leg_cmd[2], self.leg_cmd[3],
            math.degrees(self.roll), math.degrees(self.yaw),
            *imu_row,
            *errors_row,
            *(self.odom if self.odom is not None else [nan] * 13),
            *self._contact_xyz('left', now_s), *self._contact_xyz('right', now_s),
            self.odom_stamp, self.joints_stamp,
            *(self.zmp_ctrl if self.zmp_ctrl is not None else [nan] * 5),
            self.step_force,
            *wbr_row,
            *(self.est_err if self.est_err is not None else [nan] * 6),
        ]
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % FLUSH_EVERY == 0:
            self._file.flush()
        if self._row_count % 2500 == 0:
            self.get_logger().info(f'Logged {self._row_count} rows (t={t:.2f}s) -> {self.csv_path}')

    def close(self):
        self._file.flush()
        self._file.close()
        self.get_logger().info(
            f'Test bench log closed: {self._row_count} rows written to {self.csv_path}\n'
            f'Apri PlotJuggler e trascina il file per plottare tutti i risultati:\n'
            f'  ros2 run plotjuggler plotjuggler\n'
            f'  -> Drag & drop: {self.csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TestBenchLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
