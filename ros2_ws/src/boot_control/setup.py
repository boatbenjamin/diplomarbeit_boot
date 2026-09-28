import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'boot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Kopiert automatisch alle *.launch.py Dateien:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Kopiert automatisch alle Konfigurationsdateien (.yaml):
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benjamin',
    maintainer_email='benjaminnell1234@gmail.com',
    description='GNC-Stack (Guidance, Navigation, Control) fuer das autonome '
                'Boot, getestet in VRX mit dem WAM-V',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # --- Zustandsschaetzung: aus GPS und IMU wird Position, Kurs und Geschwindigkeit ---
            'wave_filter_node = boot_control.wave_filter_node:main',
            'pose_to_odom = boot_control.pose_to_odom_node:main',
            'figure8_path_publisher = boot_control.figure8_path_publisher:main',
            # --- Wahrnehmung: Kamerabild -> erkannte Bojen ---
            'buoy_detector = boot_control.buoy_detector_node:main',
            'detection_bridge = boot_control.detection_bridge_node:main',
            'object_tracker = boot_control.object_tracker_node:main',
            # --- Mission und Fuehrung: wohin soll das Boot fahren ---
            'mission_manager = boot_control.mission_manager_node:main',
            'course_manager = boot_control.course_manager_node:main',
            'guidance_ilos_node = boot_control.guidance_ilos_node:main',
            'collision_avoidance = boot_control.collision_avoidance_node:main',
            # --- Regelung und Aktorik: wie kommt es dorthin ---
            'boat_control = boot_control.boat_control_node:main',
            'thrust_to_vrx = boot_control.thrust_to_vrx_node:main',
            # --- Ueberwachung und Testhilfen ---
            'watchdog = boot_control.watchdog_node:main',
            'course_test = boot_control.course_test_node:main',
        ],
    },
)