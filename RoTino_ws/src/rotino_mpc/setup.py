import os
from glob import glob

from setuptools import setup

package_name = 'rotino_mpc'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessando Prisco',
    maintainer_email='enzogpt55@libero.it',
    description='RoTino control law: upper-body MPC + time-varying LQR on the wheels + virtual model control on the legs (Cui et al., Micromachines 2022).',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'controller = rotino_mpc.controller:main',
        ],
    },
)
