"""
wave_filter_node.py
=============================================================
Zustandsschaetzer: GPS + IMU -> /state/filtered (nav_msgs/Odometry)

DER HAUPTFEHLER DER ALTEN VERSION
---------------------------------
Der Kurs psi wurde aus dem GYROSKOP AUFINTEGRIERT und bei 0.0
gestartet:

    self.psi += self.r * self.dt

Damit war psi kein absoluter Kurs, sondern nur "Drehung seit
Knotenstart". Folge:
  1. Das Boot startet in VRX mit irgendeinem echten Kurs psi_0 != 0.
     Der Regler bringt den INTEGRIERTEN Kurs auf den Sollwert --
     der echte Kurs landet bei psi_c + psi_0. Genau das Symptom
     "Sollkurs wird nicht erreicht".
  2. Jeder Gyro-Bias driftet unbegrenzt auf. Nach ein paar Minuten
     stimmt gar nichts mehr, und zwar langsam wandernd, was beim
     Debuggen besonders unangenehm ist.
  3. Die 12-Zustands-Wellen-Kalman-Filter-Klasse in
     wave_filter_algorithm.py wurde dabei nie benutzt.

NEU
---
  - psi kommt aus der ABSOLUTEN IMU-Orientierung (VRX liefert sie in
    ENU). Ein Komplementaerfilter mischt die schnelle Gyro-Information
    dazu, damit Wellenrauschen den Kurs nicht zappeln laesst, aber
    ohne Driftanteil.
  - u/v aus GPS-Positionsdifferenz mit echten Zeitstempeln, gefiltert
    und in den Bootsrahmen gedreht (vorher: Ableitung eines stark
    verzoegerten Tiefpasses, bei GPS-Stillstand systematisch zu klein).
  - dt kommt aus der Uhr (use_sim_time-faehig), nicht als Konstante.
  - Fallback: liefert die IMU keine Orientierung (covariance[0] < 0),
    wird auf Kurs-ueber-Grund aus GPS umgeschaltet und gewarnt.

Subscriptions:
  /wamv/sensors/gps/gps/fix   (sensor_msgs/NavSatFix)
  /wamv/sensors/imu/imu/data  (sensor_msgs/Imu)
Publication:
  /state/filtered             (nav_msgs/Odometry)   frame_id = 'odom'
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry

from boot_control.mc_quaternion import yaw_from_quaternion, quaternion_from_yaw
from boot_control.mc_common import wrap_pi

ERDRADIUS = 6378137.0


class WaveFilterNode(Node):
    def __init__(self):
        super().__init__('wave_filter_node')

        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('tau_psi', 0.30)      # s, Komplementaerfilter-Zeitkonstante
        self.declare_parameter('tau_vel', 0.50)      # s, Tiefpass auf die Geschwindigkeit
        self.declare_parameter('tau_pos', 0.20)      # s, Tiefpass auf die Position
        self.declare_parameter('heading_offset_deg', 0.0)  # falls IMU verdreht montiert
        self.declare_parameter('gps_topic', '/wamv/sensors/gps/gps/fix')
        self.declare_parameter('imu_topic', '/wamv/sensors/imu/imu/data')

        self._rate = float(self.get_parameter('rate_hz').value)
        self._tau_psi = float(self.get_parameter('tau_psi').value)
        self._tau_vel = float(self.get_parameter('tau_vel').value)
        self._tau_pos = float(self.get_parameter('tau_pos').value)
        self._psi_offset = math.radians(float(self.get_parameter('heading_offset_deg').value))

        # --- Zustand -------------------------------------------------
        self._lat0 = None
        self._lon0 = None
        self._raw_x = 0.0
        self._raw_y = 0.0
        self._x = 0.0
        self._y = 0.0
        self._pos_init = False

        self._vx = 0.0          # Weltrahmen (ENU)
        self._vy = 0.0
        self._last_gps = None   # (t, x, y)

        self._psi = None        # gefilterter Kurs [rad]
        self._psi_imu = None    # absolute IMU-Orientierung [rad]
        self._r = 0.0           # Gierrate [rad/s]
        self._imu_hat_orientierung = True
        self._warned_no_orientation = False

        self._t_last = None

        self.create_subscription(NavSatFix, self.get_parameter('gps_topic').value,
                                 self._gps_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value,
                                 self._imu_cb, qos_profile_sensor_data)
        self._pub = self.create_publisher(Odometry, '/state/filtered', 10)
        self.create_timer(1.0 / self._rate, self._update)

        self.get_logger().info(
            f'wave_filter_node gestartet ({self._rate:.0f} Hz, '
            f'Kurs aus IMU-Orientierung + Gyro-Komplementaerfilter).')

    # ------------------------------------------------------------------
    def _jetzt(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def _gps_cb(self, msg: NavSatFix):
        if msg.status.status < 0:          # STATUS_NO_FIX
            return
        if not math.isfinite(msg.latitude) or not math.isfinite(msg.longitude):
            return

        if self._lat0 is None:
            self._lat0, self._lon0 = msg.latitude, msg.longitude
            self.get_logger().info(
                f'GPS-Ursprung gesetzt: lat={self._lat0:.7f}, lon={self._lon0:.7f}')

        lat_ref = math.radians(self._lat0)
        self._raw_x = ERDRADIUS * math.radians(msg.longitude - self._lon0) * math.cos(lat_ref)
        self._raw_y = ERDRADIUS * math.radians(msg.latitude - self._lat0)

        # Geschwindigkeit aus echten Zeitstempeln (nicht aus dem Timer-dt)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self._jetzt()
        if self._last_gps is not None:
            dt = t - self._last_gps[0]
            if 1e-3 < dt < 1.0:
                vx = (self._raw_x - self._last_gps[1]) / dt
                vy = (self._raw_y - self._last_gps[2]) / dt
                a = dt / max(self._tau_vel + dt, 1e-9)
                self._vx += a * (vx - self._vx)
                self._vy += a * (vy - self._vy)
        self._last_gps = (t, self._raw_x, self._raw_y)

        if not self._pos_init:
            self._x, self._y = self._raw_x, self._raw_y
            self._pos_init = True

    # ------------------------------------------------------------------
    def _imu_cb(self, msg: Imu):
        self._r = msg.angular_velocity.z

        # orientation_covariance[0] < 0 heisst laut REP 145: keine Orientierung
        if msg.orientation_covariance[0] < 0.0:
            self._imu_hat_orientierung = False
            if not self._warned_no_orientation:
                self.get_logger().warn(
                    'IMU liefert keine Orientierung -- Kurs wird aus GPS-Kurs '
                    'ueber Grund gestuetzt. Bei langsamer Fahrt ungenau!')
                self._warned_no_orientation = True
            return

        self._imu_hat_orientierung = True
        self._psi_imu = wrap_pi(yaw_from_quaternion(msg.orientation) + self._psi_offset)

    # ------------------------------------------------------------------
    def _kurs_stuetzung(self) -> float:
        """Absoluter Kursmesswert: IMU-Orientierung, sonst Kurs ueber Grund."""
        if self._imu_hat_orientierung and self._psi_imu is not None:
            return self._psi_imu
        v = math.hypot(self._vx, self._vy)
        if v > 0.3:                      # erst ab sinnvoller Fahrt brauchbar
            return math.atan2(self._vy, self._vx)
        return self._psi if self._psi is not None else 0.0

    # ------------------------------------------------------------------
    def _update(self):
        t = self._jetzt()
        if self._t_last is None:
            self._t_last = t
            return
        dt = t - self._t_last
        self._t_last = t
        if dt <= 0.0 or dt > 1.0:        # Sprung der Sim-Zeit abfangen
            return

        if not self._pos_init:
            self.get_logger().warn('Warte auf ersten GPS-Fix ...',
                                   throttle_duration_sec=5.0)
            return

        psi_mess = self._kurs_stuetzung()
        if self._psi is None:
            self._psi = psi_mess

        # --- Komplementaerfilter: Gyro schnell, IMU-Orientierung absolut ---
        psi_pred = wrap_pi(self._psi + self._r * dt)
        a = dt / max(self._tau_psi + dt, 1e-9)
        self._psi = wrap_pi(psi_pred + a * wrap_pi(psi_mess - psi_pred))

        # --- Position leicht glaetten ---
        ap = dt / max(self._tau_pos + dt, 1e-9)
        self._x += ap * (self._raw_x - self._x)
        self._y += ap * (self._raw_y - self._y)

        # --- Weltgeschwindigkeit in den Bootsrahmen drehen ---
        c, s = math.cos(self._psi), math.sin(self._psi)
        u = self._vx * c + self._vy * s          # surge
        v = -self._vx * s + self._vy * c         # sway

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation = quaternion_from_yaw(self._psi)
        odom.twist.twist.linear.x = u
        odom.twist.twist.linear.y = v
        odom.twist.twist.angular.z = self._r
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
