"""
gnc_test_no_perception.launch.py
=============================================================
Reduzierter Stack zum Testen der REGELUNG allein:

    wave_filter_node  ->  boat_control_node  ->  thrust_to_vrx_node

Optional (start_test:=true) wird course_test_node mitgestartet, der
/cmd/course_safe sendet und den Kursfehler mitloggt.

WICHTIGE AENDERUNGEN ggue. der alten Version
--------------------------------------------
1. Die ros_gz_bridge wird jetzt nur noch auf Wunsch gestartet
   (start_bridge:=true, Default false). VRX bringt seine eigene
   Bridge mit; zwei Bridges auf denselben Topics erzeugen doppelte
   Nachrichten und dadurch scheinbar "springende" Sensordaten.
   Vor dem Einschalten pruefen:
       ros2 topic list | grep wamv
       ros2 topic hz /wamv/sensors/imu/imu/data
2. `use_sim_time` wird als echter Bool-Parameter uebergeben, ausserdem
   an ALLE Knoten. Ohne use_sim_time laufen die Timer auf Wall Clock,
   waehrend die Sim langsamer laeuft -- dann stimmt kein einziges dt.
3. Die Kamera-Topics sind raus: dieser Test braucht keine Kamera,
   und eine Bridge auf ein nicht existierendes Kamera-Topic haelt
   sonst die ganze Bridge auf.
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

    def node(name, executable):
        return Node(
            package='boot_control',
            executable=executable,
            name=name,
            output='screen',
            emulate_tty=True,
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        )

    aktionen = []

    # --- optionale Sensor-Bridge (nur wenn VRX keine eigene startet) ---
    if start_bridge:
        base = f'/world/{gz_world}/model/wamv/link/wamv'
        gz_imu = f'{base}/imu_wamv_link/sensor/imu_wamv_sensor/imu'
        gz_gps = f'{base}/gps_wamv_link/sensor/gps_wamv_sensor/navsat'
        ros_imu = '/wamv/sensors/imu/imu/data'
        ros_gps = '/wamv/sensors/gps/gps/fix'

        aktionen.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gnc_sensor_bridge',
            arguments=[
                f'{gz_imu}@sensor_msgs/msg/Imu[gz.msgs.IMU',
                f'{gz_gps}@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
            ],
            remappings=[(gz_imu, ros_imu), (gz_gps, ros_gps)],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ))

    aktionen += [
        node('wave_filter_node',   'wave_filter_node'),    # Zustandsschaetzung
        node('boat_control_node',  'boat_control'),        # Regelungskaskade
        node('thrust_to_vrx_node', 'thrust_to_vrx'),       # VRX-Interface
    ]

    # --- optionaler Testgeber fuer den Sollkurs ---
    aktionen.append(Node(
        package='boot_control',
        executable='course_test',
        name='course_test_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
            {'psi_c_deg': float(LaunchConfiguration('psi_c_deg').perform(context))},
            {'u_c': float(LaunchConfiguration('u_c').perform(context))},
        ],
        condition=IfCondition(LaunchConfiguration('start_test')),
    ))

    return aktionen


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gz_world', default_value='follow_path_task'),
        DeclareLaunchArgument('start_bridge', default_value='false',
                              description='Eigene ros_gz_bridge starten? '
                                          'Nur wenn VRX keine mitbringt.'),
        DeclareLaunchArgument('start_test', default_value='true',
                              description='course_test_node mitstarten'),
        DeclareLaunchArgument('psi_c_deg', default_value='90.0'),
        DeclareLaunchArgument('u_c', default_value='1.5'),
        OpaqueFunction(function=launch_setup),
    ])
