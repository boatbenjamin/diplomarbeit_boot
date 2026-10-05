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

GIERRATEN-AUFBEREITUNG (2026-10-05)
-----------------------------------
Messbefund aus VRX: Die von ros_gz_bridge gelieferte angular_velocity.z
enthaelt eine periodische Stoerung mit ~12.5 Hz Grundfrequenz und
ungeraden Harmonischen (37.5 / 62.5 Hz) und Amplituden bis +-3.9 rad/s,
waehrend die echte Gierrate im Bereich +-1.1 rad/s liegt. Die IMU-
Zeitstempel springen dabei zwischen 8 und 12 ms (Sensorrate 100 Hz ist
kein Teiler der Physikschrittweite 4 ms).

Die alte Loesung -- eine harte Plausibilitaetsumschaltung
    if |r_gyro - r_aus_orient| > 1.0: nimm r_aus_orient
-- war die eigentliche Fehlerquelle: Jede Umschaltung ist ein SPRUNG im
Filtereingang. Bei tau_r = 0.05 s und 50 Hz schlaegt ein Eingangssprung
von 3 rad/s mit 0.86 rad/s auf die Ausgangsgroesse durch. Mal
Kp_r = Izz*omega_i = 2450 N*m/(rad/s) sind das 2100 N*m Momentensprung
bzw. rund 1000 N Schubsprung je Motor -- gemessen wurden Spruenge bis
1372 N. Das Ergebnis war ein selbsterhaltender Grenzzyklus: Umschaltung
-> Schubschlag -> Boot giert wirklich -> Gyro meldet das korrekt ->
naechste Umschaltung. Deshalb hoerte das "Zittern" nie von selbst auf.

Neu, zweistufig und STETIG (keine Fallunterscheidung mehr):
  1. Gleitender Mittelwert ueber GYRO_MITTEL_N Rohwerte. Bei 100 Hz
     Sensorrate sind 8 Samples = 80 ms = genau eine Periode der 12.5-Hz-
     Stoerung; ein Mittelwert ueber eine volle Stoerperiode loescht sie
     vollstaendig. (Ein 25-ms-Fenster erfasst nur 3 Samples und wirkt
     deshalb praktisch nicht -- das war der erste, fehlgeschlagene
     Reparaturversuch. Ein Median hilft ebenfalls nicht: bei einem
     Zwei-Cluster-Signal liefert er den haeufigeren Cluster statt der
     Mitte.)
  2. Stetige Begrenzung der Abweichung gegen die unabhaengig berechnete
     Orientierungsableitung:
         r = r_orient + clip(r_mittel - r_orient, +-r_dev_max)
     Das ist dieselbe Schutzwirkung wie die alte Umschaltung, aber ohne
     Sprung: bei kleiner Abweichung wirkt voll der Gyro, bei grosser
     wird nur der Ueberschuss gekappt.

Verifikation an 49 s aufgezeichneten VRX-Rohdaten (4929 Samples),
Referenz = zentrierte Orientierungsableitung ueber +-100 ms:
     roh                     RMS 2.915 rad/s, 100.0 % ueber 1.0 rad/s
     25-ms-Fenster (alt)     RMS 0.949 rad/s,  33.8 % ueber 1.0 rad/s
     N=8 + Clamp 0.4 (neu)   RMS 0.101 rad/s,   0.0 % ueber 1.0 rad/s
Die Begrenzung deckelt den Maximalfehler bei beliebiger Stoerfrequenz
auf r_dev_max; der blanke Mittelwert ist nur bei 12.5 Hz optimal.
Kosten: 71 ms bis 90 % Sprungantwort gegenueber T_i = 215 ms der
inneren Schleife -- Faktor 3 schneller als die Schleife, die er speist.

Subscriptions:
  /wamv/sensors/gps/gps/fix   (sensor_msgs/NavSatFix)
  /wamv/sensors/imu/imu/data  (sensor_msgs/Imu)
Publication:
  /state/filtered             (nav_msgs/Odometry)   frame_id = 'odom'
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry

from boot_control.mc_quaternion import yaw_from_quaternion, quaternion_from_yaw
from boot_control.mc_common import clip, wrap_pi

ERDRADIUS = 6378137.0


class WaveFilterNode(Node):
    def __init__(self):
        super().__init__('wave_filter_node')

        # 100 Hz: der Regler laeuft mit 50 Hz. Laeuft der Schaetzer ebenfalls
        # mit 50 Hz, driften beide Timer gegeneinander und der Regler sieht
        # denselben Zustand mal doppelt, mal gar nicht (Schwebung). Mit dem
        # doppelten Takt liegt immer ein frischer Zustand an.
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('tau_psi', 0.30)      # s, Komplementaerfilter-Zeitkonstante
        self.declare_parameter('tau_vel', 0.50)      # s, Tiefpass auf die Geschwindigkeit
        self.declare_parameter('tau_pos', 0.20)      # s, Tiefpass auf die Position
        self.declare_parameter('heading_offset_deg', 0.0)  # falls IMU verdreht montiert
        self.declare_parameter('gps_topic', '/wamv/sensors/gps/gps/fix')
        self.declare_parameter('imu_topic', '/wamv/sensors/imu/imu/data')
        self.declare_parameter('r_plausibel_max', 5.0)   # rad/s, harte Notbremse (NaN/Unsinn)
        # Der Mittelwert soll genau eine Periode der Bridge-Stoerung ueberdecken:
        #   N = Sensorrate / Stoerfrequenz = 100 Hz / 12.5 Hz = 8
        self.declare_parameter('gyro_mittel_n', 8)
        # Maximal zugelassene Abweichung Gyro <-> Orientierungsableitung [rad/s].
        # Deckelt den Messfehler stetig, unabhaengig von der Stoerfrequenz.
        self.declare_parameter('r_dev_max', 0.4)
        # Der Mittelwert leistet die Hauptarbeit -> Tiefpass nur noch leicht,
        # damit die Gesamttotzeit klein bleibt (frueher 0.05).
        self.declare_parameter('tau_r', 0.02)            # s, Tiefpass auf die Gierrate

        self._rate = float(self.get_parameter('rate_hz').value)
        self._tau_psi = float(self.get_parameter('tau_psi').value)
        self._tau_vel = float(self.get_parameter('tau_vel').value)
        self._tau_pos = float(self.get_parameter('tau_pos').value)
        self._psi_offset = math.radians(float(self.get_parameter('heading_offset_deg').value))
        self._r_max = float(self.get_parameter('r_plausibel_max').value)
        self._mittel_n = max(1, int(self.get_parameter('gyro_mittel_n').value))
        self._r_dev_max = float(self.get_parameter('r_dev_max').value)
        self._tau_r = float(self.get_parameter('tau_r').value)

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
        self._r = 0.0           # Gierrate [rad/s], gefiltert -> /state/filtered
        self._r_roh = 0.0       # Gyro, gleitender Mittelwert ueber _mittel_n Samples
        self._r_roh_hist = []   # die letzten _mittel_n Gyro-Rohwerte
        self._psi_imu_alt = None  # (t, psi) fuer Gierrate aus Orientierungsableitung
        self._r_aus_orient = 0.0
        self._imu_hat_orientierung = True
        self._warned_no_orientation = False
        # Diagnose: wie oft greift die Begrenzung, und wie stark
        self._clamp_zaehler = 0
        self._takt_zaehler = 0
        self._dev_max_gesehen = 0.0

        self._t_last = None

        self.create_subscription(NavSatFix, self.get_parameter('gps_topic').value,
                                 self._gps_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value,
                                 self._imu_cb, qos_profile_sensor_data)
        self._pub = self.create_publisher(Odometry, '/state/filtered', 10)
        # GPS-Nullpunkt fuer alle anderen Knoten (Pfadplaner!) -- latched,
        # damit auch spaeter gestartete Knoten denselben Ursprung bekommen.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self._pub_origin = self.create_publisher(NavSatFix, '/state/gps_origin', latched)
        self.create_timer(1.0 / self._rate, self._update)

        self.get_logger().info(
            f'wave_filter_node gestartet ({self._rate:.0f} Hz, '
            f'Kurs aus IMU-Orientierung + Gyro-Komplementaerfilter; '
            f'Gyro: Mittelwert N={self._mittel_n}, Begrenzung '
            f'+-{self._r_dev_max:.2f} rad/s gegen Orientierungsableitung).')

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
            origin = NavSatFix()
            origin.header = msg.header
            origin.header.frame_id = 'odom'
            origin.status = msg.status
            origin.latitude, origin.longitude, origin.altitude = \
                msg.latitude, msg.longitude, msg.altitude
            self._pub_origin.publish(origin)

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
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self._jetzt()

        # --- Stufe 1: gleitender Mittelwert ueber genau _mittel_n Rohwerte ---
        # Bewusst SAMPLE-basiert, nicht zeitbasiert: die Bridge liefert
        # abwechselnd 8 und 12 ms Abstand, ein Zeitfenster erfasst deshalb mal
        # 3 und mal 4 Samples und loescht die Stoerung nur teilweise. Entscheidend
        # ist die ANZAHL der gemittelten Perioden, nicht die Fensterbreite.
        w = msg.angular_velocity.z
        if math.isfinite(w):
            self._r_roh_hist.append(w)
            if len(self._r_roh_hist) > self._mittel_n:
                del self._r_roh_hist[0]
            self._r_roh = sum(self._r_roh_hist) / len(self._r_roh_hist)

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

        # Zweite, unabhaengige Gierratenquelle: Ableitung der Orientierung.
        # Dient als Referenz, gegen die der Gyro stetig begrenzt wird.
        # (t ist oben bereits bestimmt -- die alte Version hat es hier ein
        #  zweites Mal aus derselben Nachricht gelesen.)
        if self._psi_imu_alt is not None:
            dt = t - self._psi_imu_alt[0]
            if 1e-3 < dt < 0.5:
                r_o = wrap_pi(self._psi_imu - self._psi_imu_alt[1]) / dt
                a = dt / (0.1 + dt)
                self._r_aus_orient += a * (r_o - self._r_aus_orient)
        self._psi_imu_alt = (t, self._psi_imu)

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

        # --- Stufe 2: Gierrate stetig gegen die Orientierungsableitung begrenzen ---
        # KEINE Fallunterscheidung mehr: jede Umschaltung waere ein Sprung im
        # Filtereingang, und genau solche Spruenge hat der Regler frueher zu
        # Schubschlaegen von ueber 1000 N verstaerkt (siehe Modulkopf).
        r_mittel = self._r_roh
        if not math.isfinite(r_mittel) or abs(r_mittel) > self._r_max:
            # harte Notbremse: komplett unbrauchbarer Wert (NaN, Ausreisser)
            r_mess = self._r_aus_orient
        elif self._psi_imu_alt is None:
            # noch keine Orientierungsreferenz -> Gyro ungefiltert uebernehmen
            r_mess = r_mittel
        else:
            dev = r_mittel - self._r_aus_orient
            r_mess = self._r_aus_orient + clip(dev, -self._r_dev_max, self._r_dev_max)
            self._takt_zaehler += 1
            if abs(dev) > self._r_dev_max:
                self._clamp_zaehler += 1
            self._dev_max_gesehen = max(self._dev_max_gesehen, abs(dev))

        ar = dt / max(self._tau_r + dt, 1e-9)
        self._r += ar * (r_mess - self._r)

        # Diagnose statt Warnflut: einmal pro 10 s eine Zeile mit der Quote.
        # Eine hohe Quote ist kein Fehler mehr, sondern nur der Hinweis, dass
        # die Bridge stark stoert -- der Regler bekommt trotzdem ein stetiges
        # Signal. Steigt sie ueber ~50 %, lohnt ein Blick auf gyro_mittel_n.
        if self._takt_zaehler >= int(10.0 * self._rate):
            quote = 100.0 * self._clamp_zaehler / max(self._takt_zaehler, 1)
            self.get_logger().info(
                f'Gyro-Begrenzung aktiv in {quote:.1f} % der Takte '
                f'(groesste Abweichung {self._dev_max_gesehen:.2f} rad/s, '
                f'Grenze {self._r_dev_max:.2f}).')
            self._takt_zaehler = 0
            self._clamp_zaehler = 0
            self._dev_max_gesehen = 0.0

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