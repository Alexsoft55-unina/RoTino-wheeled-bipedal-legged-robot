"""Runs RoTino with the cascaded PD/PID control law."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    robot = os.path.join(get_package_share_directory('rotino_description'), 'launch', 'robot.launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(robot),
            launch_arguments={
                'controller_pkg': 'rotino_pid',
                'controller_exe': 'controller',
                'controller_name': 'rotino_pid_controller',
            }.items(),
        ),
    ])
