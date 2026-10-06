"""Step disturbance for the benchmark: a constant horizontal force on the torso, identical for every law.

The impulsive push of the scenarios is applied by the controllers themselves; the step force is applied
from outside, through the persistent wrench of Gazebo's ApplyLinkWrench system
(/world/rotino_world/wrench/persistent, bridged by robot.launch.py): Gazebo applies it at every physics
step until it is cleared, so its value does not depend on the timing of the ROS messages.

Time base: the first field of /rotino/debug, the time since the release from the anchor that both laws
publish and the logger writes as time_s. The force is backwards along the heading the robot has when
it starts (positive force = backwards, like the impulsive push).

    ros2 run rotino_benchmark disturbance --ros-args -p use_sim_time:=true -p force:=3.0 -p start_time:=4.0
"""

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity, EntityWrench
from std_msgs.msg import Float64MultiArray

PERSISTENT_TOPIC = '/world/rotino_world/wrench/persistent'
CLEAR_TOPIC = '/world/rotino_world/wrench/clear'
TORSO = 'rotino::base_link'


class StepDisturbance(Node):

    def __init__(self):
        super().__init__('rotino_disturbance')
        self.force = float(self.declare_parameter('force', 3.0).value)          # N, + = backwards
        self.start_time = float(self.declare_parameter('start_time', 4.0).value)  # s after release
        self.duration = float(self.declare_parameter('duration', 0.0).value)      # s, 0 = until the end
        self.pub = self.create_publisher(EntityWrench, PERSISTENT_TOPIC, 10)
        self.clear_pub = self.create_publisher(Entity, CLEAR_TOPIC, 10)
        self.heading = None
        self.state = 'wait'
        self.create_subscription(Odometry, '/rotino/odom', self._odom_cb, 10)
        self.create_subscription(Float64MultiArray, '/rotino/debug', self._debug_cb, 10)
        end = 'fino alla fine' if self.duration <= 0.0 else f'per {self.duration:.1f} s'
        self.get_logger().info(f'Gradino di {self.force:+.2f} N (+ = indietro) a t={self.start_time:.1f} s, {end}')

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.state == 'wait':            # heading of the robot before the step, then frozen
            self.heading = (math.cos(yaw), math.sin(yaw))

    def _entity(self):
        e = Entity()
        e.name = TORSO
        e.type = Entity.LINK
        return e

    def _debug_cb(self, msg):
        if not msg.data:
            return
        t = msg.data[0]
        if self.state == 'wait' and t >= self.start_time and self.heading is not None:
            # published once: every persistent message adds a wrench, a repeated one would double the force
            w = EntityWrench()
            w.entity = self._entity()
            w.wrench.force.x = -self.force * self.heading[0]
            w.wrench.force.y = -self.force * self.heading[1]
            self.pub.publish(w)
            self.state = 'on'
            self.get_logger().info(f'>>> GRADINO {self.force:+.2f} N applicato a t={t:.3f} s')
        elif self.state == 'on' and self.duration > 0.0 and t >= self.start_time + self.duration:
            self.clear_pub.publish(self._entity())
            self.state = 'off'
            self.get_logger().info(f'>>> GRADINO rimosso a t={t:.3f} s')


def main(args=None):
    rclpy.init(args=args)
    node = StepDisturbance()
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
