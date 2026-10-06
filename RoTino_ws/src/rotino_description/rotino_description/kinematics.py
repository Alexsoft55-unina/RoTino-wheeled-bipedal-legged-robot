import numpy as np
from urdf_parser_py import xml_reflection
from urdf_parser_py.urdf import URDF

# Silence warnings about <gazebo>/<ros2_control> tags that urdf_parser_py does not know.
xml_reflection.core.on_error = lambda *args, **kwargs: None


def rpy_to_matrix(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_to_matrix(axis, angle):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class RobotKinematics:
    """Forward kinematics and whole-robot COM from the URDF (replaces MuJoCo framepos/subtreecom)."""

    def __init__(self, urdf_xml):
        robot = URDF.from_xml_string(urdf_xml)
        self.root = robot.get_root()
        self.child_joints = {}
        for joint in robot.joints:
            self.child_joints.setdefault(joint.parent, []).append(joint)

        self.masses = []
        for link in robot.links:
            if link.inertial is None or link.inertial.mass <= 0.0:
                continue
            origin = link.inertial.origin
            com_local = np.array(origin.xyz if origin is not None and origin.xyz else [0.0, 0.0, 0.0])
            self.masses.append((link.name, link.inertial.mass, com_local))
        self.total_mass = sum(m for _, m, _ in self.masses)

    def link_frames(self, base_pos, base_rot, joint_positions):
        """Return {link_name: (R_world, p_world)}."""
        frames = {}
        stack = [(self.root, np.asarray(base_rot, dtype=float), np.asarray(base_pos, dtype=float))]
        while stack:
            name, rot, pos = stack.pop()
            frames[name] = (rot, pos)
            for joint in self.child_joints.get(name, []):
                origin = joint.origin
                xyz = np.array(origin.xyz if origin is not None and origin.xyz else [0.0, 0.0, 0.0])
                rpy = origin.rpy if origin is not None and origin.rpy else [0.0, 0.0, 0.0]
                child_pos = pos + rot @ xyz
                child_rot = rot @ rpy_to_matrix(*rpy)
                if joint.type in ('revolute', 'continuous'):
                    child_rot = child_rot @ axis_angle_to_matrix(joint.axis, joint_positions.get(joint.name, 0.0))
                elif joint.type == 'prismatic':
                    child_pos = child_pos + child_rot @ (np.asarray(joint.axis) * joint_positions.get(joint.name, 0.0))
                stack.append((joint.child, child_rot, child_pos))
        return frames

    def center_of_mass(self, frames):
        weighted = np.zeros(3)
        for name, mass, com_local in self.masses:
            rot, pos = frames[name]
            weighted += mass * (pos + rot @ com_local)
        return weighted / self.total_mass
