"""Runs RoTino with the MPC + TV-LQR + VMC control law.

Everything except the controller comes from rotino_description/launch/robot.launch.py, so all the
scenario arguments (velocity_enable, push_enable, jump_enable, gui, dashboard, ...) work here too;
run with --show-args to list them.
"""

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
                'controller_pkg': 'rotino_mpc',
                'controller_exe': 'controller',
                'controller_name': 'rotino_mpc_controller',
            }.items(),
        ),
    ])
