import os
from glob import glob

from setuptools import setup

package_name = 'rotino_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.xml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessando Prisco',
    maintainer_email='enzogpt55@libero.it',
    description='RoTino robot description and shared model library',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'model = rotino_description.model:main',
        ],
    },
)
