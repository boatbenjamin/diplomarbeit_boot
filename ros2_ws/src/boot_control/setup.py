from setuptools import find_packages, setup
import os

package_name = 'boot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/gnc.launch.py',
                                               'launch/gnc_test_no_perception.launch.py']),
        ('share/' + package_name + '/config',  ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benjamin',
    maintainer_email='benjaminnell1234@gmail.com',
    description='GNC Stack autonomes Boot',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'wave_filter = boot_control.wave_filter_node:main',
            'object_tracker = boot_control.object_tracker_node:main',
            'mission_manager = boot_control.mission_manager_node:main',
            'course_manager = boot_control.course_manager_node:main',
            'guidance_ilos = boot_control.guidance_ilos_node:main',
            'collision_avoidance = boot_control.collision_avoidance_node:main',
            'boat_control = boot_control.boat_control_node:main',  # Hier von mc_control_node geändert
            'thrust_to_vrx = boot_control.thrust_to_vrx_node:main',  # Hier von mc_schubaufteilung geändert
            'watchdog = boot_control.watchdog_node:main',
            'buoy_detector = boot_control.buoy_detector_node:main',
            'pose_to_odom = boot_control.pose_to_odom_node:main',
            'detection_bridge = boot_control.detection_bridge_node:main',
        ],
    },
)