from setuptools import setup

package_name = 'rotino_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alessando Prisco',
    maintainer_email='enzogpt55@libero.it',
    description='Real-time dashboard for the RoTino wheeled biped',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'dashboard = rotino_dashboard.dashboard:main',
        ],
    },
)
