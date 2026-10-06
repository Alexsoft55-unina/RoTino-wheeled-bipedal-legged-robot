"""ROS side of the dashboard.

The subscriptions run in a separate process (no GIL contention with the Qt thread) and write into
shared-memory ring buffers; the GUI process only reads the last time window of each stream.
"""

import math
import multiprocessing as mp
import os
import queue
import re
import time
from multiprocessing import shared_memory

import numpy as np

JOINTS = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_wheel_joint', 'right_wheel_joint']
CONTACT_STALE_TIME = 0.02
MIN_JACOBIAN_DET = 2e-3     # m^2; nominal pose ~8e-3
BUFFER_SECONDS = 60.0
RATE = 500

# /rotino/debug:        [t, theta, theta_dot, s, s_dot, wheel_u, com_z, com_z_vel, Fn_total, min_wheel_gap, loaded]
# /rotino/wbr_state:    [t, s, s_ref, theta, theta_ref, phi, phi_ref, s_dot, s_dot_ref, z, z_ref, delta_s, F_z,
#                        tau_l, tau_r, l, tau_hip_l, tau_knee_l]
# /rotino/estimation_error: [ex, ey, ez, evx, evy, evz]
# zmp (computed here, rotino_description.zmp, world frame unless noted):
#   [zmp_x, zmp_y, lipm_x, lipm_y, com_x, com_y, com_z, cL_x, cL_y, cR_x, cR_y,
#    y_rel (+1 = on the left wheel), lateral [m], e_long [m], margin [m], Fn_L, Fn_R]
STREAMS = {
    'debug': 11,
    'wbr': 18,
    'est': 6,
    'joints': 12,       # 6 positions + 6 velocities, JOINTS order
    'contact': 2,       # 1 = wheel touching the ground (left, right)
    'wheel_cmd': 2,
    'leg_cmd': 4,       # hip_L, hip_R, knee_L, knee_R
    'leg_force': 4,     # force pushing the body, world frame: Fx_L, Fz_L, Fx_R, Fz_R [N]  (F = -J^-T tau)
    'odom': 5,          # x, y, z, pitch, yaw
    'zmp': 17,
}
NAMES = list(STREAMS)
# status vector: sim time, wall time of that sim time, transport delay, counts..., wall_last...
ST_SIM, ST_WALL, ST_DELAY, ST_COUNT0 = 0, 1, 2, 3
ST_WALL0 = ST_COUNT0 + len(NAMES)


class SharedRing:
    """Single-writer ring buffer of rows [t, values...] in shared memory."""

    def __init__(self, name, cols, meta, index, create):
        self.capacity = int(BUFFER_SECONDS * RATE)
        size = self.capacity * (cols + 1) * 8
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self.data = np.ndarray((self.capacity, cols + 1), dtype=np.float64, buffer=self.shm.buf)
        self.meta = meta          # int64 array: [head, size] per stream
        self.i = index

    # writer ----------------------------------------------------------
    def append(self, t, values):
        head = int(self.meta[2 * self.i])
        row = self.data[head]
        row[0] = t
        row[1:1 + len(values)] = values
        self.meta[2 * self.i] = (head + 1) % self.capacity
        self.meta[2 * self.i + 1] = min(int(self.meta[2 * self.i + 1]) + 1, self.capacity)

    def clear(self):
        self.meta[2 * self.i + 1] = 0

    # reader ----------------------------------------------------------
    def _segments(self):
        head, size = int(self.meta[2 * self.i]), int(self.meta[2 * self.i + 1])
        if size == 0:
            return []
        start = (head - size) % self.capacity
        if start + size <= self.capacity:
            return [self.data[start:start + size]]
        return [self.data[start:], self.data[:head]]

    def last(self):
        head, size = int(self.meta[2 * self.i]), int(self.meta[2 * self.i + 1])
        if size == 0:
            return None
        return self.data[(head - 1) % self.capacity].copy()

    def window(self, t_from):
        parts = []
        for seg in self._segments():
            i = int(np.searchsorted(seg[:, 0], t_from))
            if i < len(seg):
                parts.append(seg[i:])
        if not parts:
            return None
        return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()


def remove_stale_segments():
    """Unlinks shared memory left by dashboards that were killed (their pid no longer exists)."""
    try:
        entries = os.listdir('/dev/shm')
    except OSError:
        return
    for entry in entries:
        match = re.match(r'rotino_dash_(\d+)_', entry)
        if match is None:
            continue
        try:
            os.kill(int(match.group(1)), 0)
        except ProcessLookupError:
            try:
                os.unlink(os.path.join('/dev/shm', entry))
            except OSError:
                pass
        except PermissionError:
            pass


class SharedState:
    """Shared memory blocks; created by the GUI process, attached by the ROS process."""

    def __init__(self, prefix=None, create=True):
        self.prefix = prefix or f'rotino_dash_{mp.current_process().pid}_{int(time.time())}'
        self.create = create
        self._meta_shm = shared_memory.SharedMemory(name=self.prefix + '_meta', create=create,
                                                    size=16 * len(NAMES))
        self.meta = np.ndarray((2 * len(NAMES),), dtype=np.int64, buffer=self._meta_shm.buf)
        self._status_shm = shared_memory.SharedMemory(name=self.prefix + '_status', create=create,
                                                      size=8 * (ST_WALL0 + len(NAMES)))
        self.status = np.ndarray((ST_WALL0 + len(NAMES),), dtype=np.float64, buffer=self._status_shm.buf)
        if create:
            self.meta[:] = 0
            self.status[:] = 0.0
            self.status[ST_DELAY] = np.nan
        self.buffers = {name: SharedRing(f'{self.prefix}_{name}', cols, self.meta, i, create)
                        for i, (name, cols) in enumerate(STREAMS.items())}

    def close(self):
        for shm in [b.shm for b in self.buffers.values()] + [self._meta_shm, self._status_shm]:
            shm.close()
            if self.create:
                shm.unlink()


# ============================================================================ ROS process
def _contact_touching(msg):
    return any('ground' in c.collision1.name or 'ground' in c.collision2.name for c in msg.contacts)


def run_ros_process(prefix, commands, events):
    import rclpy
    import rclpy.clock
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from ros_gz_interfaces.msg import Contacts
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, Float64, Float64MultiArray, String

    shared = SharedState(prefix, create=False)
    rclpy.init()
    node = Node('rotino_dashboard', parameter_overrides=[Parameter('use_sim_time', value=True)])
    st = shared.status
    state = {'joint_index': None, 'q': None, 'pitch': 0.0, 'model': None, 'zmp': None, 'phase': None,
             'contact': {'left': -math.inf, 'right': -math.inf}}

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def store(name, t, values):
        if t <= 0.0:
            return
        buf = shared.buffers[name]
        last = buf.last()
        if last is not None and t < last[0] - 1.0:
            buf.clear()   # simulation restarted: time went backwards
        buf.append(t, values)
        i = NAMES.index(name)
        st[ST_COUNT0 + i] += 1
        st[ST_WALL0 + i] = time.monotonic()
        st[ST_SIM] = t
        st[ST_WALL] = time.monotonic()

    def on_array(name, msg):
        store(name, now(), msg.data[:STREAMS[name]])

    def on_leg_cmd(msg):
        on_array('leg_cmd', msg)
        m, q = state['model'], state['q']
        if m is None or q is None or len(msg.data) < 4:
            return
        c, s = math.cos(state['pitch']), math.sin(state['pitch'])
        out = []
        for side, (tau_hip, tau_knee) in enumerate(((msg.data[0], msg.data[2]), (msg.data[1], msg.data[3]))):
            _, J = m.leg_fk(q[side], q[2 + side])
            if abs(np.linalg.det(J)) < MIN_JACOBIAN_DET:   # leg almost straight: force not observable
                F = np.array([np.nan, np.nan])
            else:
                F = -np.linalg.solve(J.T, np.array([tau_hip, tau_knee]))
            out += [c * F[0] + s * F[1], -s * F[0] + c * F[1]]
        store('leg_force', now(), out)

    def on_joints(msg):
        if state['joint_index'] is None:
            names = list(msg.name)
            if not all(j in names for j in JOINTS):
                return
            state['joint_index'] = [names.index(j) for j in JOINTS]
        idx = state['joint_index']
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        st[ST_DELAY] = now() - t
        pos = [msg.position[i] for i in idx]
        state['q'] = pos
        store('joints', t, pos + [msg.velocity[i] for i in idx])
        store('contact', t, [float(t - state['contact'][s] <= CONTACT_STALE_TIME) for s in ('left', 'right')])
        if state['zmp'] is not None:
            store_zmp(state['zmp'].push_joints(t, dict(zip(JOINTS, pos))))

    def on_contact(side, msg):
        if _contact_touching(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            state['contact'][side] = stamp if stamp > 0.0 else now()

    def on_odom(msg):
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (o.w * o.y - o.z * o.x))))
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        state['pitch'] = pitch
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        store('odom', t, [p.x, p.y, p.z, pitch, yaw])
        if state['zmp'] is not None:
            store_zmp(state['zmp'].push_base(t, (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)))

    def store_zmp(r):
        if r is not None:
            store('zmp', r['t'], [*r['zmp'], *r['lipm'], *r['com'], *r['contact_l'][:2], *r['contact_r'][:2],
                                  r['y_rel'], r['y_rel'] * 0.5 * r['track'], r['e_long'], r['margin'],
                                  r['fn_left'], r['fn_right']])

    def on_phase(msg):
        if msg.data != state['phase']:
            state['phase'] = msg.data
            events.put(('phase', now(), msg.data))

    def on_description(msg):
        events.put(('description', msg.data))
        try:
            from rotino_description.model import WBRModel
            from rotino_description.zmp import ZmpEstimator
            state['model'] = WBRModel(msg.data)
            state['zmp'] = ZmpEstimator(msg.data, rate=RATE)
        except Exception as exc:
            events.put(('error', f'robot_description: {exc}'))

    fast = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    for topic, name in (('/rotino/debug', 'debug'), ('/rotino/wbr_state', 'wbr'),
                        ('/rotino/estimation_error', 'est'), ('/wheel_effort_controller/commands', 'wheel_cmd')):
        node.create_subscription(Float64MultiArray, topic, lambda m, n=name: on_array(n, m), fast)
    node.create_subscription(Float64MultiArray, '/leg_effort_controller/commands', on_leg_cmd, fast)
    node.create_subscription(JointState, '/joint_states', on_joints, fast)
    node.create_subscription(Contacts, '/rotino/left_wheel_contact', lambda m: on_contact('left', m), fast)
    node.create_subscription(Contacts, '/rotino/right_wheel_contact', lambda m: on_contact('right', m), fast)
    node.create_subscription(Odometry, '/rotino/odom', on_odom, fast)
    node.create_subscription(String, '/rotino/jump_state', on_phase, 10)
    node.create_subscription(String, '/robot_description', on_description, latched)

    running = [True]
    parent = os.getppid()
    cmd_vel_pub = node.create_publisher(Twist, '/rotino/cmd_vel', 10)
    cmd_height_pub = node.create_publisher(Float64, '/rotino/cmd_height', 10)
    cmd_jump_pub = node.create_publisher(Empty, '/rotino/cmd_jump', 10)
    cmd_push_pub = node.create_publisher(Float64, '/rotino/cmd_push', 10)
    teleop = {'active': False, 'v': 0.0, 'w': 0.0, 'h': 0.0}

    def publish_teleop():
        if not teleop['active']:
            return
        msg = Twist()
        msg.linear.x, msg.angular.z = teleop['v'], teleop['w']
        cmd_vel_pub.publish(msg)
        cmd_height_pub.publish(Float64(data=teleop['h']))

    def poll_commands():
        try:
            while True:
                cmd = commands.get_nowait()
                if cmd == 'clear':
                    for b in shared.buffers.values():
                        b.clear()
                    if state['zmp'] is not None:
                        state['zmp'].reset()
                    state['phase'] = None
                elif cmd == 'stop':
                    running[0] = False
                elif cmd[0] == 'teleop':
                    was_active = teleop['active']
                    teleop.update(active=cmd[1], v=cmd[2], w=cmd[3], h=cmd[4])
                    if was_active and not cmd[1]:
                        cmd_vel_pub.publish(Twist())   # stop before handing back
                    publish_teleop()
                elif cmd[0] == 'jump':
                    cmd_jump_pub.publish(Empty())
                elif cmd[0] == 'push':
                    cmd_push_pub.publish(Float64(data=float(cmd[1])))
        except queue.Empty:
            pass

    wall = rclpy.clock.Clock()
    node.create_timer(0.02, poll_commands, clock=wall)
    node.create_timer(0.05, publish_teleop, clock=wall)   # 20 Hz keep-alive (controller timeout 0.5 s)
    try:
        while running[0] and rclpy.ok() and os.getppid() == parent:   # exit with the GUI, even if it was killed
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        shared.close()


# ============================================================================ GUI-side handle
class RosBridge:
    """Starts the ROS process and exposes the shared data to the GUI."""

    def __init__(self):
        remove_stale_segments()
        self.shared = SharedState(create=True)
        self.buffers = self.shared.buffers
        ctx = mp.get_context('spawn')
        self.commands = ctx.Queue()
        self.events = ctx.Queue()
        self.process = ctx.Process(target=run_ros_process, args=(self.shared.prefix, self.commands, self.events),
                                   daemon=True)
        self.process.start()
        self.phase = None
        self.phase_since = None
        self.phase_log = []
        self.robot_description = None
        self.error = None

    def poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == 'phase':
                    _, t, name = ev
                    self.phase, self.phase_since = name, t
                    self.phase_log.append((t, name))
                    del self.phase_log[:-50]
                elif ev[0] == 'description':
                    self.robot_description = ev[1]
                elif ev[0] == 'error':
                    self.error = ev[1]
        except queue.Empty:
            pass

    def now(self):
        st = self.shared.status
        if st[ST_SIM] <= 0.0:
            return 0.0
        return float(st[ST_SIM] + min(time.monotonic() - st[ST_WALL], 0.5))

    @property
    def transport_delay(self):
        return float(self.shared.status[ST_DELAY])

    @property
    def counts(self):
        return {n: float(self.shared.status[ST_COUNT0 + i]) for i, n in enumerate(NAMES)}

    @property
    def wall_last(self):
        return {n: float(self.shared.status[ST_WALL0 + i]) for i, n in enumerate(NAMES)}

    def send_teleop(self, active, v, w, h):
        self.commands.put(('teleop', bool(active), float(v), float(w), float(h)))

    def send_jump(self):
        self.commands.put(('jump',))

    def send_push(self, impulse):
        self.commands.put(('push', float(impulse)))

    def clear(self):
        self.commands.put('clear')
        self.phase, self.phase_since = None, None
        self.phase_log.clear()

    def close(self):
        self.commands.put('stop')
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()
        self.shared.close()
