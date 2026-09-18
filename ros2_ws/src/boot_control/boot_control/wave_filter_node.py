"""
wave_filter_node.py
=============================================================
ROS 2 Node: Wellen-Kalman-Filter

Subscriptions:
  /wamv/sensors/imu/imu/data  (sensor_msgs/Imu)
  /wamv/odom                  (nav_msgs/Odometry)

Publications:
  /state/filtered             (nav_msgs/Odometry)
    - pose.pose.orientation -> yaw [rad] (gefilterter Kurs, LF-Anteil)
    - twist.twist.angular.z  -> r   [rad/s] (gefilterte Gierrate)
    - twist.twist.linear.x   -> u   [m/s]   (Geschwindigkeit aus Odometrie)
    - pose.pose.position.x/y -> x, y [m]    (Position aus Odometrie)

Wichtig: Der Wellenfilter (WellenKalmanFilterV2) arbeitet intern in Grad.
Konvertierung findet an den Topic-Grenzen statt:
  Eingang:  rad -> Grad
  Ausgang:  Grad -> rad
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from boot_control.wave_filter_algorithm import WellenKalmanFilterV2


class WaveFilterNode(Node):
    def __init__(self):
        super().__init__('wave_filter_node')

        # Parameter
        self.declare_parameter('f_wave', 0.5)          # Hz, Startwert Wellenfrequenz
        self.declare_parameter('q_wave', 250.0)        # Prozessrauschen Welle
        self.declare_parameter('r_heading_deg', 0.5)   # Messrauschen Kurs [Grad^2]
        self.declare_parameter('r_gyro_deg', 0.05)     # Messrauschen Gierrate [Grad^2/s^2]

        f_wave = self.get_parameter('f_wave').value
        q_wave = self.get_parameter('q_wave').value
        r_hdg = self.get_parameter('r_heading_deg').value
        r_gyro = self.get_parameter('r_gyro_deg').value

        self._filter = WellenKalmanFilterV2(
            f_wave=f_wave,
            q_wave=q_wave,
            r_heading=r_hdg,
            r_gyro=r_gyro,
        )

        self._last_imu_rate_deg: float = 0.0   # Gierrate aus IMU [Grad/s]
        self._last_u: float = 0.0              # Vorwärtsgeschwindigkeit [m/s]
        self._last_x: float = 0.0
        self._last_y: float = 0.0
        self._filter_initialized: bool = False
        self._last_stamp = None

        self.create_subscription(Imu, '/wamv/sensors/imu/imu/data', self._imu_cb, 10)
        self.create_subscription(Odometry, '/wamv/odom', self._odom_cb, 10)

        self._pub = self.create_publisher(Odometry, '/state/filtered', 10)

        self.get_logger().info('wave_filter_node gestartet.')

    # ------------------------------------------------------------------
    def _imu_cb(self, msg: Imu):
        """IMU-Callback: extrahiert Kurs und Gierrate, fuehrt Filterschritt aus."""
        q = msg.orientation
        _, _, yaw_rad = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_deg = math.degrees(yaw_rad)
        rate_deg = math.degrees(msg.angular_velocity.z)

        # dt aus Header-Zeitstempel
        now = msg.header.stamp
        if self._last_stamp is None:
            dt = 0.02
        else:
            dt = (now.sec - self._last_stamp.sec) + \
                 (now.nanosec - self._last_stamp.nanosec) * 1e-9
            dt = max(1e-4, min(dt, 0.5))   # Clamp: kein Sprung bei Jitter
        self._last_stamp = now

        if not self._filter_initialized:
            self._filter.reset(yaw_deg)
            self._filter_initialized = True
            return

        output = self._filter.step(z_heading=yaw_deg, z_rate=rate_deg, dt=dt)

        self._last_imu_rate_deg = output.rate_lf

        # Gefilterten Zustand publizieren
        filtered_psi_rad = math.radians(output.kurs_lf)
        filtered_r_rad   = math.radians(output.rate_lf)
        self._publish(filtered_psi_rad, filtered_r_rad)

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        """Odometrie-Callback: Geschwindigkeit und Position cachen."""
        self._last_u = msg.twist.twist.linear.x
        self._last_x = msg.pose.pose.position.x
        self._last_y = msg.pose.pose.position.y

    # ------------------------------------------------------------------
    def _publish(self, psi_rad: float, r_rad: float):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Kurs als Quaternion kodieren (nur Yaw relevant)
        half = psi_rad / 2.0
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)
        odom.pose.pose.position.x = self._last_x
        odom.pose.pose.position.y = self._last_y

        odom.twist.twist.angular.z = r_rad
        odom.twist.twist.linear.x = self._last_u

        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = WaveFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
