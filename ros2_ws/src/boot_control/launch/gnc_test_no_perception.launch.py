import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('boot_control')
    params_file = os.path.join(pkg, 'config', 'params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    gz_world = LaunchConfiguration('gz_world').perform(context)
    start_bridge = LaunchConfiguration('start_bridge').perform(context).lower() == 'true'


    def node(name, executable, extra_params=None):
        p = [params_file, {'use_sim_time': use_sim_time}]
        if extra_params:
            p.append(extra_params)
        return Node(
            package='boot_control',
            executable=executable,
            name=name,
            output='screen',
            emulate_tty=True,
            parameters=p,
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

    # Basis-Knoten
    aktionen += [
        node('wave_filter_node', 'wave_filter_node'),  # Zustandsschätzung
        node('boat_control_node', 'boat_control'),  # Regelungskaskade
        node('thrust_to_vrx_node', 'thrust_to_vrx'),  # VRX-Interface
        node('guidance_ilos_node', 'guidance_ilos_node'),  # ILOS Pfad-Regler
        node('figure8_path_publisher', 'figure8_path_publisher')  # <-- TIPPFEHLER BEHOBEN
    ]

    # --- optionaler Test-Knoten für Kurs-Sequenzen ---


    # WICHTIG: Die Aktionsliste MUSS zurückgegeben werden!
    return aktionen


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gz_world', default_value='follow_path_task'),
        DeclareLaunchArgument('start_bridge', default_value='false',
                              description='Eigene ros_gz_bridge starten?'),

        # HIER GEÄNDERT: default_value auf 'false' gesetzt, damit der ILOS Regler in Ruhe arbeiten kann


        DeclareLaunchArgument('sequenz_deg', default_value='[0.0, 90.0, 180.0, -90.0]'),
        DeclareLaunchArgument('sequenz_dauer', default_value='40.0'),
        DeclareLaunchArgument('u_c', default_value='1.5'),
        OpaqueFunction(function=launch_setup),
    ])