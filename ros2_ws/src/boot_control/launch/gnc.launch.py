"""
gnc.launch.py
=============================================================
Kompletter GNC-Stack:

  Wahrnehmung : buoy_detector -> detection_bridge -> object_tracker
  Mission     : course_manager, mission_manager
  Fuehrung    : guidance_ilos
  Ausweichen  : collision_avoidance
  Regelung    : wave_filter -> boat_control -> thrust_to_vrx

Wie im Testlaunch: use_sim_time ueberall, eigene Bridge nur auf Wunsch.
Ueber die Argumente kann jede Ebene einzeln abgeschaltet werden, um
Fehler zu isolieren (z.B. with_perception:=false und stattdessen
/objects von Hand publizieren).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('boot_control')
    params_file = os.path.join(pkg, 'config', 'params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    gz_world = LaunchConfiguration('gz_world').perform(context)
    start_bridge = LaunchConfiguration('start_bridge').perform(context).lower() == 'true'

    mit_ca = LaunchConfiguration('with_collision_avoidance').perform(context).lower() == 'true'

    def node(name, executable, cond=None, remappings=None):
        return Node(
            package='boot_control', executable=executable, name=name,
            output='screen', emulate_tty=True,
            parameters=[params_file, {'use_sim_time': use_sim_time}],
            remappings=remappings or [],
            condition=IfCondition(cond) if cond is not None else None,
        )

    aktionen = []

    if start_bridge:
        base = f'/world/{gz_world}/model/wamv/link/wamv'
        paare = [
            (f'{base}/base_link/sensor/front_left_camera_sensor/image',
             '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
             'sensor_msgs/msg/Image', 'gz.msgs.Image'),
            (f'{base}/base_link/sensor/front_left_camera_sensor/camera_info',
             '/wamv/sensors/cameras/front_left_camera_sensor/camera_info',
             'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
            (f'{base}/imu_wamv_link/sensor/imu_wamv_sensor/imu',
             '/wamv/sensors/imu/imu/data', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
            (f'{base}/gps_wamv_link/sensor/gps_wamv_sensor/navsat',
             '/wamv/sensors/gps/gps/fix', 'sensor_msgs/msg/NavSatFix', 'gz.msgs.NavSat'),
        ]
        aktionen.append(Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='gnc_sensor_bridge',
            arguments=[f'{gz}@{ros_typ}[{gz_typ}' for gz, _, ros_typ, gz_typ in paare],
            remappings=[(gz, ros) for gz, ros, _, _ in paare],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ))

    # --- Zustandsschaetzung (immer) ---
    aktionen.append(node('wave_filter_node', 'wave_filter_node'))

    # --- Wahrnehmung ---
    p = LaunchConfiguration('with_perception')
    aktionen += [
        node('buoy_detector_node',    'buoy_detector',    p),
        node('detection_bridge_node', 'detection_bridge', p),
        node('object_tracker_node',   'object_tracker',   p),
    ]

    # --- Mission + Fuehrung ---
    g = LaunchConfiguration('with_guidance')
    aktionen += [
        node('course_manager_node',  'course_manager',  g),
        node('mission_manager_node', 'mission_manager', g),
        node('guidance_ilos_node',   'guidance_ilos',   g),
    ]

    # --- Kollisionsvermeidung ---
    aktionen.append(node('collision_avoidance_node', 'collision_avoidance',
                         LaunchConfiguration('with_collision_avoidance')))

    # --- Regelung ---
    # Ohne Kollisionsvermeidung publiziert NIEMAND /cmd/course_safe
    # (guidance_ilos sendet auf /cmd/course). Dann wird direkt
    # umgemappt, sonst haengt die Regelung stumm im Warte-Zustand.
    ca_remap = [] if mit_ca else [('/cmd/course_safe', '/cmd/course')]
    aktionen += [
        node('boat_control_node',  'boat_control', remappings=ca_remap),
        node('thrust_to_vrx_node', 'thrust_to_vrx'),
    ]
    return aktionen


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gz_world', default_value='follow_path_task'),
        DeclareLaunchArgument('start_bridge', default_value='false'),
        DeclareLaunchArgument('with_perception', default_value='true'),
        DeclareLaunchArgument('with_guidance', default_value='true'),
        DeclareLaunchArgument('with_collision_avoidance', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
