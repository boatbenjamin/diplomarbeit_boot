#!/usr/bin/env python3
"""
figure8_path_publisher.py
=============================================================
Publiziert eine Lemniskate (liegende Acht) um zwei GPS-Punkte A und B
als nav_msgs/Path im odom-Frame.

Geometrie
---------
Zwei Kreise mit Radius R = loop_radius_factor * |AB| um A und B.
Bei loop_radius_factor = 0.5 beruehren sich die Kreise genau in der
Mitte zwischen A und B -- dort geht die Bahn tangential von einem
Kreis in den anderen ueber (A im Uhrzeigersinn, B gegen den
Uhrzeigersinn). Die Bahn wird als EINE geschlossene Runde publiziert;
der ILOS-Knoten erkennt das (Anfang == Ende) und faehrt endlos.

Gemeinsamer Nullpunkt
---------------------
Der odom-Ursprung wird vom wave_filter_node uebernommen
(/state/gps_origin, latched). Frueher hat dieser Knoten seinen eigenen
ersten GPS-Fix genommen -- startet man ihn spaeter neu, waehrend das
Boot schon faehrt, war der ganze Pfad um diese Strecke verschoben.

Parameter
---------
  lat_A, lon_A, lat_B, lon_B   GPS-Koordinaten der Kreismittelpunkte
  loop_radius_factor           R = factor * |AB|  (0.5 = Kreise beruehren sich)
  num_points                   Stuetzpunkte pro Runde
  umkehren                     Fahrtrichtung umdrehen
  eigener_ursprung             true: eigenen ersten GPS-Fix nehmen (Fallback,
                               nur wenn wave_filter_node nicht laeuft)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix

ERDRADIUS = 6378137.0

LATCHED = QoSProfile(depth=1,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=QoSReliabilityPolicy.RELIABLE)


def lemniskate(A: np.ndarray, B: np.ndarray, R: float, n: int, umkehren: bool = False):
    """Eine geschlossene Runde: Halbkreis B -> Vollkreis A -> Halbkreis B."""
    d = B - A
    a0 = math.atan2(d[1], d[0])            # Richtung A -> B
    n_half = max(8, n // 4)
    n_full = 2 * n_half

    # B gegen Uhrzeigersinn, von der Aussenseite bis zum Beruehrpunkt
    th = np.linspace(a0, a0 + math.pi, n_half, endpoint=False)
    xs = [B[0] + R * np.cos(th)]
    ys = [B[1] + R * np.sin(th)]
    # A im Uhrzeigersinn, voller Kreis ab dem Beruehrpunkt
    th = np.linspace(a0, a0 - 2 * math.pi, n_full, endpoint=False)
    xs.append(A[0] + R * np.cos(th))
    ys.append(A[1] + R * np.sin(th))
    # B zweite Haelfte zurueck zur Aussenseite (Endpunkt == Startpunkt)
    th = np.linspace(a0 + math.pi, a0 + 2 * math.pi, n_half + 1, endpoint=True)
    xs.append(B[0] + R * np.cos(th))
    ys.append(B[1] + R * np.sin(th))

    x, y = np.concatenate(xs), np.concatenate(ys)
    if umkehren:
        x, y = x[::-1], y[::-1]
    return x, y


class Figure8PathPublisher(Node):
    def __init__(self):
        super().__init__('figure8_path_publisher')

        self.declare_parameter('lat_A', -33.7225690)
        self.declare_parameter('lon_A', 150.6739884)
        self.declare_parameter('lat_B', -33.7223690)
        self.declare_parameter('lon_B', 150.6739884)
        self.declare_parameter('loop_radius_factor', 0.5)
        self.declare_parameter('num_points', 400)
        self.declare_parameter('umkehren', False)
        self.declare_parameter('eigener_ursprung', False)

        g = lambda n: self.get_parameter(n).value
        self._A_gps = (float(g('lat_A')), float(g('lon_A')))
        self._B_gps = (float(g('lat_B')), float(g('lon_B')))

        self.path_msg = None
        self._pub = self.create_publisher(Path, '/path', LATCHED)

        if bool(g('eigener_ursprung')):
            self.get_logger().warn('eigener_ursprung=true: Nullpunkt = eigener erster GPS-Fix. '
                                   'Nur korrekt, wenn gleichzeitig mit wave_filter gestartet!')
            self._sub = self.create_subscription(NavSatFix, '/wamv/sensors/gps/gps/fix',
                                                 self._origin_cb, qos_profile_sensor_data)
        else:
            self._sub = self.create_subscription(NavSatFix, '/state/gps_origin',
                                                 self._origin_cb, LATCHED)

        self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f'Warte auf GPS-Ursprung ... A=({self._A_gps[0]:.7f}, {self._A_gps[1]:.7f})  '
            f'B=({self._B_gps[0]:.7f}, {self._B_gps[1]:.7f})')

    # ------------------------------------------------------------------
    def _origin_cb(self, msg: NavSatFix):
        if self.path_msg is not None:
            return
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        lat0, lon0 = msg.latitude, msg.longitude
        c = math.cos(math.radians(lat0))

        def to_xy(lat, lon):
            return np.array([ERDRADIUS * math.radians(lon - lon0) * c,
                             ERDRADIUS * math.radians(lat - lat0)])

        A, B = to_xy(*self._A_gps), to_xy(*self._B_gps)
        dist = float(np.hypot(*(B - A)))
        if dist < 2.0:
            self.get_logger().error(f'A und B liegen nur {dist:.2f} m auseinander -- abgebrochen.')
            return

        R = float(self.get_parameter('loop_radius_factor').value) * dist
        x, y = lemniskate(A, B, R, int(self.get_parameter('num_points').value),
                          bool(self.get_parameter('umkehren').value))

        path = Path()
        path.header.frame_id = 'odom'
        for xi, yi in zip(x, y):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(xi)
            ps.pose.position.y = float(yi)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_msg = path

        laenge = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        self.get_logger().info(
            f'Ursprung lat={lat0:.7f} lon={lon0:.7f} -> odom: A=({A[0]:.1f}, {A[1]:.1f}) '
            f'B=({B[0]:.1f}, {B[1]:.1f}), |AB|={dist:.1f} m, R={R:.1f} m, '
            f'Rundenlaenge={laenge:.0f} m, {len(x)} Punkte')
        self._publish()

    def _publish(self):
        if self.path_msg is None:
            return
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self.path_msg)


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