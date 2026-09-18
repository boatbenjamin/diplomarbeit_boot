"""
gnc_test_no_perception.launch.py   (NEU, nur zum Testen)
=============================================================
Startet die GNC-Kette OHNE Wahrnehmung (buoy_detector, detection_bridge,
object_tracker fehlen absichtlich). Stattdessen speist man /objects von
Hand per `ros2 topic pub` mit sauberen, erfundenen Bojenpositionen —
damit laesst sich pruefen, ob mission_manager -> course_manager ->
guidance_ilos -> collision_avoidance -> boat_control -> thrust_to_vrx
korrekt zusammenspielen, unabhaengig davon ob die Kamera/YOLO/HSV echte
Bojen findet.

Odometrie (pose_to_odom, wave_filter) bleibt ECHT drin, weil die bereits
nachweislich funktioniert (/wamv/odom laeuft mit ~150 Hz).

Verwendung:
  ros2 launch boot_control gnc_test_no_perception.launch.py

Danach in einem zweiten Terminal die Fake-Bojen einspeisen, siehe
Anleitung im Chat (ros2 topic pub /objects ...).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('boot_control')
    params_file = os.path.join(pkg, 'config', 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    gz_world = LaunchConfiguration('gz_world').perform(context)

    def node(name, executable=None):
        return Node(
            package='boot_control',
            executable=executable or name,
            name=name,
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time == 'true'}],
        )

    base = f'/world/{gz_world}/model/wamv/link/wamv'
    gz_imu = f'{base}/imu_wamv_link/sensor/imu_wamv_sensor/imu'
    ros_imu = '/wamv/sensors/imu/imu/data'

    # Nur IMU wird gebridged -- Kamera/camera_info brauchen wir hier nicht,
    # da buoy_detector/detection_bridge in diesem Test nicht laufen.
    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gnc_imu_bridge_test',
        arguments=[f'{gz_imu}@sensor_msgs/msg/Imu[gz.msgs.IMU'],
        remappings=[(gz_imu, ros_imu)],
        output='screen',
    )

    return [
        imu_bridge,

        node('pose_to_odom_node', 'pose_to_odom'),
        node('wave_filter_node',  'wave_filter'),

        # ── object_tracker_node ABSICHTLICH WEGGELASSEN ────────────────────
        # -> /objects wird per Hand ueber `ros2 topic pub` gespeist.

        node('mission_manager_node',     'mission_manager'),
        node('course_manager_node',      'course_manager'),
        node('guidance_ilos_node',       'guidance_ilos'),
        node('collision_avoidance_node', 'collision_avoidance'),
        node('boat_control_node',        'boat_control'),
        node('thrust_to_vrx_node',       'thrust_to_vrx'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gz_world', default_value='sydney_regatta'),
        OpaqueFunction(function=launch_setup),
    ])