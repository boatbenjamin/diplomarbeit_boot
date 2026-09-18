"""
gnc.launch.py  —  KORRIGIERT
=============================================================
Aenderung gegenueber der alten Version:

  Die VRX-eigene Bridge (competition.launch.py) baut ihre gz-Topicnamen
  aus dem DATEINAMEN der World ("navigation_task"), die laufende World
  heisst intern aber "sydney_regatta" (siehe navigation_task.sdf:
  <world name="sydney_regatta">). Ergebnis: ALLE Sensor-Bridges von VRX
  (Kamera, camera_info, IMU, GPS, Lidar, joint_state) abonnieren ein
  gz-Topic, das es nicht gibt. Sie legen ihre ROS-Topics zwar an
  (darum tauchen sie in `ros2 topic list` auf), liefern aber nie Daten.

  Vorher wurde hier nur das Kamera-Bild ueberbrueckt. Jetzt zusaetzlich
  camera_info (sonst rechnet detection_bridge_node ewig mit den
  hartkodierten fx=530/cx=640) und die IMU (sonst publiziert
  wave_filter_node nie /state/filtered, und damit stehen Mission-Manager,
  Guidance und Regler komplett still).

Verwendung:
  ros2 launch boot_control gnc.launch.py
  ros2 launch boot_control gnc.launch.py gz_world:=sydney_regatta

World-Namen pruefen mit:
  gz topic -l | grep imu
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

    def node(name, executable=None, extra_params=None):
        p = [params_file]
        if extra_params:
            p.append(extra_params)
        return Node(
            package='boot_control',
            executable=executable or name,
            name=name,
            output='screen',
            parameters=p + [{'use_sim_time': use_sim_time == 'true'}],
        )

    # ── gz-Topics der tatsaechlich laufenden World ───────────────────────
    base = f'/world/{gz_world}/model/wamv/link/wamv'
    gz_img  = f'{base}/base_link/sensor/front_left_camera_sensor/image'
    gz_info = f'{base}/base_link/sensor/front_left_camera_sensor/camera_info'
    gz_imu  = f'{base}/imu_wamv_link/sensor/imu_wamv_sensor/imu'

    ros_img  = '/wamv/sensors/cameras/front_left_camera_sensor/image_raw'
    ros_info = '/wamv/sensors/cameras/front_left_camera_sensor/camera_info'
    ros_imu  = '/wamv/sensors/imu/imu/data'

    sensor_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gnc_sensor_bridge',
        arguments=[
            f'{gz_img}@sensor_msgs/msg/Image[gz.msgs.Image',
            f'{gz_info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            f'{gz_imu}@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        remappings=[
            (gz_img,  ros_img),
            (gz_info, ros_info),
            (gz_imu,  ros_imu),
        ],
        output='screen',
    )

    return [
        sensor_bridge,

        # ── Odometrie-Quelle ─────────────────────────────────────────────
        # /wamv/odom hatte bisher GAR KEINEN Publisher (es stand nur in
        # `ros2 topic list`, weil wave_filter_node es abonniert).
        node('pose_to_odom_node',        'pose_to_odom'),

        # ── Wahrnehmung ──────────────────────────────────────────────────
        node('buoy_detector_node',       'buoy_detector'),
        node('detection_bridge_node',    'detection_bridge'),
        node('wave_filter_node',         'wave_filter'),
        node('object_tracker_node',      'object_tracker'),

        # ── Mission ──────────────────────────────────────────────────────
        node('mission_manager_node',     'mission_manager'),
        node('course_manager_node',      'course_manager'),

        # ── Fuehrung ─────────────────────────────────────────────────────
        node('guidance_ilos_node',       'guidance_ilos'),

        # ── Kollisionsvermeidung ─────────────────────────────────────────
        node('collision_avoidance_node', 'collision_avoidance'),

        # ── Regelung ─────────────────────────────────────────────────────
        node('boat_control_node',        'boat_control'),

        # ── VRX-Interface ────────────────────────────────────────────────
        node('thrust_to_vrx_node',       'thrust_to_vrx'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gz_world', default_value='follow_path_task'),
        OpaqueFunction(function=launch_setup),
    ])