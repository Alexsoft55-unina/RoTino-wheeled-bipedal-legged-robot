"""Brings up RoTino in Gazebo and starts whichever control law was asked for."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SPAWN_HEIGHT = 0.2439
LEFT_CONTACT_GZ_TOPIC = '/world/rotino_world/model/rotino/link/left_wheel_link/sensor/left_wheel_contact/contact'
RIGHT_CONTACT_GZ_TOPIC = '/world/rotino_world/model/rotino/link/right_wheel_link/sensor/right_wheel_contact/contact'

SCENARIO_ARGS = [
    ('jump_enable', 'false', bool, 'Jump (squat, thrust, flight, landing) at jump_start_time'),
    ('jump_start_time', '4.0', float, 'Seconds after release before the jump starts'),
    ('jump_velocity', '1.2', float, 'CoM vertical velocity at the end of the thrust [m/s]'),
    ('height_enable', 'false', bool, 'Sinusoidal standing-height change'),
    ('height_amplitude', '0.03', float, 'Height oscillation amplitude [m]'),
    ('height_period', '2.2', float, 'Height oscillation period [s]'),
    ('velocity_enable', 'false', bool, 'Trapezoidal velocity profile'),
    ('velocity_max', '0.44', float, 'Cruise velocity [m/s]'),
    ('accel_max', '0.6', float, 'Acceleration / deceleration [m/s^2]'),
    ('velocity_distance', '2.0', float, 'Total travelled distance [m]'),
    ('drive_enable', 'false', bool, 'Back-and-forth reference while balancing'),
    ('drive_distance', '1.0', float, 'Back-and-forth travel [m]'),
    ('drive_period', '8.0', float, 'Duration of one back-and-forth cycle [s]'),
    ('planar_enable', 'false', bool, 'Follow the planar polynomial S-trajectory'),
    ('traj_length', '3.0', float, 'Trajectory arc length [m]'),
    ('traj_lateral', '0.6', float, 'Lateral offset of the S-curve [m]'),
    ('traj_duration', '12.0', float, 'Trajectory duration [s]'),
    ('push_enable', 'false', bool, 'Horizontal impulse on the torso'),
    ('push_time', '4.0', float, 'Seconds after release when the impulse is applied'),
    ('push_impulse', '2.7', float, 'Impulse [N s]'),
    ('wait_start', 'false', bool, 'Hold the scripted motion until a message on /rotino/cmd_start (std_msgs/Empty)'),
    ('zmp_lateral', 'false', bool, 'PID: lean into turns to keep the lateral ZMP centred (see docs/PID_ZMP.md)'),
]


def generate_launch_description():
    pkg_share = get_package_share_directory('rotino_description')
    world_file = os.path.join(pkg_share, 'worlds', 'rotino_world.sdf')
    xacro_file = os.path.join(pkg_share, 'urdf', 'rotino.urdf.xacro')
    controllers_file = os.path.join(pkg_share, 'config', 'rotino_controllers.yaml')

    gui = LaunchConfiguration('gui')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' controllers_file:=', controllers_file]),
        value_type=str,
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': PythonExpression(
                ["'", world_file, " -r -v 1' + ('' if '", gui, "'.lower() == 'true' else ' -s')"]),
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'rotino', '-z', str(SPAWN_HEIGHT),
                   '-x', LaunchConfiguration('spawn_x'), '-y', LaunchConfiguration('spawn_y'),
                   '-Y', LaunchConfiguration('spawn_yaw')],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/rotino/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/rotino/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            LEFT_CONTACT_GZ_TOPIC + '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            RIGHT_CONTACT_GZ_TOPIC + '@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
            '/rotino/release@std_msgs/msg/Empty]gz.msgs.Empty',
            '/world/rotino_world/wrench@ros_gz_interfaces/msg/EntityWrench]gz.msgs.EntityWrench',
            '/world/rotino_world/wrench/persistent@ros_gz_interfaces/msg/EntityWrench]gz.msgs.EntityWrench',
            '/world/rotino_world/wrench/clear@ros_gz_interfaces/msg/Entity]gz.msgs.Entity',
        ],
        remappings=[
            (LEFT_CONTACT_GZ_TOPIC, '/rotino/left_wheel_contact'),
            (RIGHT_CONTACT_GZ_TOPIC, '/rotino/right_wheel_contact'),
        ],
    )

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '--controller-manager', '/controller_manager'])

    joint_state_broadcaster_spawner = spawner('joint_state_broadcaster')
    wheel_controller_spawner = spawner('wheel_effort_controller')
    leg_controller_spawner = spawner('leg_effort_controller')

    controller = Node(
        package=LaunchConfiguration('controller_pkg'),
        executable=LaunchConfiguration('controller_exe'),
        name=LaunchConfiguration('controller_name'),
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('start_controller')),
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
            **{name: ParameterValue(LaunchConfiguration(name), value_type=typ)
               for name, _, typ, _ in SCENARIO_ARGS},
        }],
    )

    dashboard = ExecuteProcess(
        cmd=['ros2', 'run', 'rotino_dashboard', 'dashboard'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('dashboard')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('controller_pkg', description='Package holding the control law to run'),
        DeclareLaunchArgument('controller_exe', description='Controller executable inside controller_pkg'),
        DeclareLaunchArgument('controller_name', default_value='rotino_controller',
                              description='Node name of the controller'),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument('dashboard', default_value='false',
                              description='Open the real-time rotino_dashboard GUI'),
        DeclareLaunchArgument('spawn_x', default_value='0.0', description='Spawn x [m]'),
        DeclareLaunchArgument('spawn_y', default_value='0.0', description='Spawn y [m]'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0', description='Spawn heading [rad]'),
        DeclareLaunchArgument('start_controller', default_value='true',
                              description='Start the controller here; set false to launch it by hand'),
        *[DeclareLaunchArgument(name, default_value=default, description=desc)
          for name, default, _, desc in SCENARIO_ARGS],
        gz_sim,
        bridge,
        dashboard,
        robot_state_publisher,
        spawn_robot,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot,
                                           on_exit=[joint_state_broadcaster_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=joint_state_broadcaster_spawner,
                                           on_exit=[wheel_controller_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=wheel_controller_spawner,
                                           on_exit=[leg_controller_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=leg_controller_spawner,
                                           on_exit=[controller])),
    ])
