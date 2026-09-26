#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
import numpy as np

ERDRADIUS = 6378137.0


class Figure8PathPublisher(Node):
    def __init__(self):
        super().__init__('figure8_path_publisher')

        # Publisher fuer den Pfad
        self.publisher_ = self.create_publisher(Path, '/path', 10)

        # ==========================================
        # ECHTE GPS-KOORDINATEN DER BEIDEN PUNKTE
        # BITTE HIER DIE RICHTIGEN WERTE EINTRAGEN!
        # ==========================================
        self.lat_A = -33.72248916822321  # <--- HIER ECHTE LATITUDE VON PUNKT A
        self.lon_A = 150.67420518427178  # <--- HIER ECHTE LONGITUDE VON PUNKT A

        self.lat_B = -33.722027251041276  # <--- HIER ECHTE LATITUDE VON PUNKT B (Bsp-Werte)
        self.lon_B = 150.67455787110447  # <--- HIER ECHTE LONGITUDE VON PUNKT B
        # ==========================================

        self._lat0 = None
        self._lon0 = None
        self._path_cache = None
        self.path_msg = None

        # Subscriber, um exakt denselben Nullpunkt wie der Wave-Filter zu finden
        self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',
                                 self._gps_cb, qos_profile_sensor_data)

        # Timer, um den Pfad periodisch zu veroeffentlichen
        self.timer = self.create_timer(1.0, self.publish_path)

        self.get_logger().info('Warte auf ersten GPS-Fix, um den odom-Nullpunkt zu setzen...')

    def _gps_cb(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        if not math.isfinite(msg.latitude) or not math.isfinite(msg.longitude):
            return

        # Nur den ALLERERSTEN GPS-Fix als gemeinsamen Nullpunkt speichern
        if self._lat0 is None:
            self._lat0 = msg.latitude
            self._lon0 = msg.longitude
            self.get_logger().info(f'Gemeinsamer Nullpunkt gesetzt: lat={self._lat0:.7f}, lon={self._lon0:.7f}')

            # Jetzt die GPS-Koordinaten von A und B ins lokale odom-System umrechnen
            # (Exakt dieselbe Mathematik wie in wave_filter_node.py)
            lat_ref = math.radians(self._lat0)

            ax = ERDRADIUS * math.radians(self.lon_A - self._lon0) * math.cos(lat_ref)
            ay = ERDRADIUS * math.radians(self.lat_A - self._lat0)

            bx = ERDRADIUS * math.radians(self.lon_B - self._lon0) * math.cos(lat_ref)
            by = ERDRADIUS * math.radians(self.lat_B - self._lat0)

            A = np.array([ax, ay])
            B = np.array([bx, by])

            self.get_logger().info(f'Lokale Koordinaten im odom-Frame: A({ax:.2f}, {ay:.2f}), B({bx:.2f}, {by:.2f})')

            # Pfad generieren
            self.path_msg = self.generate_ros_path(A, B)
            self.get_logger().info(f'Pfad mit {len(self.path_msg.poses)} Punkten erstellt und bereit.')

    def compute_figure8_path(self,
                             A: np.ndarray,
                             B: np.ndarray,
                             num_points: int = 400,
                             backward_extension: float = 15.0,
                             loop_radius_factor: float = 0.5,
                             num_laps: int = 10):
        """
        Liefert (path_x, path_y) fuer die komplette Figure-8-Bahn inkl.
        Anlaufbahn. Startet am unteren Rand des unteren Kreises (B).
        """
        if self._path_cache is not None:
            return self._path_cache

        d = B - A
        dist_ab = float(np.hypot(d[0], d[1]))
        if dist_ab < 1e-3:
            return np.array([]), np.array([])

        R = loop_radius_factor * dist_ab

        angle_A0 = float(np.arctan2(d[1], d[0]))
        angle_B0 = angle_A0 + np.pi

        pts_circle = max(8, num_points // max(1, 2 * num_laps))
        if pts_circle % 2 != 0:
            pts_circle += 1
        pts_half = pts_circle // 2

        xs, ys = [], []
        for _ in range(num_laps):
            theta_B1 = np.linspace(angle_A0, angle_B0, pts_half, endpoint=False)
            xs.extend((B[0] + R * np.cos(theta_B1)).tolist())
            ys.extend((B[1] + R * np.sin(theta_B1)).tolist())

            theta_A = np.linspace(angle_A0, angle_A0 - 2 * np.pi, pts_circle, endpoint=False)
            xs.extend((A[0] + R * np.cos(theta_A)).tolist())
            ys.extend((A[1] + R * np.sin(theta_A)).tolist())

            theta_B2 = np.linspace(angle_B0, angle_A0 + 2 * np.pi, pts_half, endpoint=False)
            xs.extend((B[0] + R * np.cos(theta_B2)).tolist())
            ys.extend((B[1] + R * np.sin(theta_B2)).tolist())

        path_x = np.array(xs)
        path_y = np.array(ys)

        # Anlaufbahn: Tangente am allerersten Punkt zurueckverlaengern
        dx0 = path_x[1] - path_x[0]
        dy0 = path_y[1] - path_y[0]
        norm = float(np.hypot(dx0, dy0))
        if norm > 1e-6:
            dx0, dy0 = dx0 / norm, dy0 / norm
            n_ext = max(2, int(backward_extension * 2))
            ext_dists = np.linspace(-backward_extension, 0, n_ext, endpoint=False)
            ext_x = path_x[0] + ext_dists * dx0
            ext_y = path_y[0] + ext_dists * dy0
            path_x = np.concatenate((ext_x, path_x))
            path_y = np.concatenate((ext_y, path_y))

        self._path_cache = (path_x, path_y)
        return self._path_cache

    def generate_ros_path(self, A: np.ndarray, B: np.ndarray) -> Path:
        """ Wandelt die Numpy-Arrays in eine nav_msgs/Path Nachricht um. """
        path_x, path_y = self.compute_figure8_path(A, B)

        ros_path = Path()
        ros_path.header.frame_id = 'odom'

        for x, y in zip(path_x, path_y):
            pose = PoseStamped()
            pose.header.frame_id = ros_path.header.frame_id

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)

        return ros_path

    def publish_path(self):
        """ Published den vorgenerierten Pfad, sobald er berechnet wurde. """
        if self.path_msg is not None:
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = Figure8PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()