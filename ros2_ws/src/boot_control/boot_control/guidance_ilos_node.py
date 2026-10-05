"""
guidance_ilos_node.py
=============================================================
ROS 2 Node: ILOS-Fuehrungsgesetz (Integral Line-of-Sight)

Subscriptions:
  /state/filtered  (nav_msgs/Odometry)  -- gefilterter Zustand (x, y, psi, u, v)
  /path            (nav_msgs/Path)      -- Referenzpfad im odom-Frame

Publications:
  /cmd/course_safe (geometry_msgs/Twist)
    linear.x  = u_d   [m/s]  Soll-Fahrt
    angular.z = psi_d [rad]  Soll-Heading
  /diag/ilos       (geometry_msgs/Vector3Stamped)
    x = y_e [m] Querablage, y = psi_d [deg], z = u_d [m/s]

HINWEIS: Publiziert direkt auf /cmd/course_safe (ohne
collision_avoidance). Fuer den reinen ILOS-Test gewollt; sobald die
Kollisionsvermeidung mitlaeuft, Topic-Parameter auf /cmd/course setzen.

AENDERUNGEN ggue. der alten Version
-----------------------------------
1. Das Integral wurde bei JEDER /path-Nachricht (1 Hz) zurueckgesetzt
   -> der I-Anteil konnte nie wirken. Jetzt nur, wenn sich der Pfad
   wirklich aendert.
2. Pfadverfolgung mit Fortschritt (PfadTracker) statt globaler
   argmin-Suche; geschlossene Pfade werden endlos gefahren.
3. Schwimmwinkel-Kompensation (v aus /state/filtered).
4. Wartet auf den ersten Zustand, bevor gerechnet wird.
5. Diagnose-Topic + 1-Hz-Log.
6. Kruemmungs-Vorhalt: Parameter t_vorhalt [s] (0 = aus), siehe
   guidance_ilos_algorithm.py.
7. Schwimmwinkel-Vorsteuerung: Parameter k_beta [s] (0 = aus).
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path

from boot_control.mc_quaternion import yaw_from_quaternion
from boot_control.guidance_ilos_algorithm import ILOSGuidance, PfadTracker


class GuidanceILOSNode(Node):
    def __init__(self):
        super().__init__('guidance_ilos_node')

        self.declare_parameter('delta', 4.0)           # m, Lookahead
        self.declare_parameter('sigma', 0.3)           # Integrator-Gewicht
        self.declare_parameter('i_max', 10.0)          # m, Integrator-Saettigung
        self.declare_parameter('u_max', 3.0)           # m/s
        self.declare_parameter('u_min', 0.8)           # m/s, Mindestfahrt
        self.declare_parameter('a_quer_max', 1.0)      # m/s^2
        self.declare_parameter('k3', 1.0)              # Fahrtreduktion bei Kursfehler
        self.declare_parameter('beta_komp', True)      # Schwimmwinkel kompensieren
        self.declare_parameter('beta_max_deg', 30.0)
        self.declare_parameter('tau_beta', 3.0)        # s, Tiefpass Schwimmwinkel
        self.declare_parameter('t_vorschau', 3.0)      # s, Kurven-Vorschau
        self.declare_parameter('t_vorhalt', 0.0)       # s, Kruemmungs-Vorhalt (0 = aus)
        self.declare_parameter('k_beta', 0.0)          # s, Schwimmwinkel-Vorsteuerung (0 = aus)
        self.declare_parameter('fenster_vor', 20.0)    # m, Suchfenster vorwaerts
        self.declare_parameter('fenster_zurueck', 5.0) # m, Suchfenster rueckwaerts
        self.declare_parameter('reacquire_dist', 20.0) # m, globale Neusuche
        self.declare_parameter('timer_hz', 20.0)
        self.declare_parameter('cmd_topic', '/cmd/course_safe')

        g = lambda n: self.get_parameter(n).value
        self._guidance = ILOSGuidance(
            delta=float(g('delta')), sigma=float(g('sigma')), i_max=float(g('i_max')),
            u_max=float(g('u_max')), u_min=float(g('u_min')),
            a_quer_max=float(g('a_quer_max')), k3=float(g('k3')),
            beta_komp=bool(g('beta_komp')), beta_max_deg=float(g('beta_max_deg')),
            t_vorschau=float(g('t_vorschau')), tau_beta=float(g('tau_beta')),
            t_vorhalt=float(g('t_vorhalt')), k_beta=float(g('k_beta')),
        )
        self._fenster_vor = float(g('fenster_vor'))
        self._fenster_zurueck = float(g('fenster_zurueck'))
        self._reacquire = float(g('reacquire_dist'))

        self._pos = np.zeros(2)
        self._psi = 0.0
        self._u = 0.0
        self._v = 0.0
        self._state_ok = False
        self._last_stamp = None

        self._tracker = None
        self._path_sig = None

        self.create_subscription(Odometry, '/state/filtered', self._state_cb, 10)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(Path, '/path', self._path_cb, latched)
        self._pub = self.create_publisher(Twist, str(g('cmd_topic')), 10)
        self._pub_diag = self.create_publisher(Vector3Stamped, '/diag/ilos', 10)

        self.create_timer(1.0 / float(g('timer_hz')), self._timer_cb)
        self.get_logger().info(
            f'guidance_ilos_node gestartet: Delta={self._guidance.delta} m, '
            f'sigma={self._guidance.sigma}, u_max={self._guidance.u_max} m/s, '
            f'beta_komp={self._guidance.beta_komp}, t_vorhalt={self._guidance.t_vorhalt} s, k_beta={self._guidance.k_beta} s')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        self._pos[0] = msg.pose.pose.position.x
        self._pos[1] = msg.pose.pose.position.y
        self._psi = yaw_from_quaternion(msg.pose.pose.orientation)
        self._u = msg.twist.twist.linear.x
        self._v = msg.twist.twist.linear.y
        self._state_ok = True

    def _path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            return
        xs = np.array([p.pose.position.x for p in msg.poses])
        ys = np.array([p.pose.position.y for p in msg.poses])
        # Signatur: nur bei echter Aenderung neu aufsetzen
        sig = (len(xs), round(float(xs[0]), 2), round(float(ys[0]), 2),
               round(float(xs[-1]), 2), round(float(ys[-1]), 2),
               round(float(xs.sum()), 1), round(float(ys.sum()), 1))
        if sig == self._path_sig:
            return
        try:
            tracker = PfadTracker(xs, ys)
        except ValueError as e:
            self.get_logger().error(f'Pfad ungueltig: {e}')
            return
        self._path_sig = sig
        self._tracker = tracker
        self._guidance.reset_integral()
        self.get_logger().info(
            f'Neuer Pfad: {len(xs)} Punkte, {tracker.laenge:.1f} m, '
            f'{"geschlossen (endlos)" if tracker.closed else "offen"}')

    # ------------------------------------------------------------------
    def _timer_cb(self):
        if self._tracker is None:
            self.get_logger().warn('Kein /path empfangen -- ILOS inaktiv.',
                                   throttle_duration_sec=5.0)
            return
        if not self._state_ok:
            self.get_logger().warn('Warte auf /state/filtered ...', throttle_duration_sec=5.0)
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_stamp is None:
            self._last_stamp = t_now
            return
        dt = t_now - self._last_stamp
        self._last_stamp = t_now
        if not (1e-4 < dt < 1.0):
            return

        y_e, pi_h, _ = self._tracker.update(
            self._pos, self._fenster_zurueck, self._fenster_vor, self._reacquire)
        # Kruemmungs-Vorhalt: Pfadrichtung (und Kruemmung) ein Stueck voraus
        s_vor = self._tracker.s + max(self._u, 0.0) * self._guidance.t_vorhalt
        if self._guidance.t_vorhalt > 0.0:
            pi_h = self._tracker.pfadwinkel_bei(s_vor)
        kappa_vor = self._tracker.kappa_bei(s_vor)

        if self._tracker.am_ende:
            u_d, psi_d = 0.0, self._psi
        else:
            psi_d = self._guidance.update_heading(y_e, pi_h, self._u, self._v, dt, kappa_vor)
            u_ref = max(abs(self._u), self._guidance.u_min)
            k_max = self._tracker.kappa_vorschau(max(5.0, u_ref * self._guidance.t_vorschau))
            u_d = self._guidance.calculate_speed(k_max, self._psi, psi_d)

        msg = Twist()
        msg.linear.x = float(u_d)
        msg.angular.z = float(psi_d)
        self._pub.publish(msg)

        d = Vector3Stamped()
        d.header.stamp = self.get_clock().now().to_msg()
        d.header.frame_id = 'odom'
        d.vector.x = float(y_e)
        d.vector.y = math.degrees(psi_d)
        d.vector.z = float(u_d)
        self._pub_diag.publish(d)

        self.get_logger().info(
            f'y_e={y_e:+5.2f} m  s={self._tracker.s:6.1f}/{self._tracker.laenge:.0f} m '
            f'(Runde {self._tracker.runden + 1})  psi_d={math.degrees(psi_d):+6.1f}°  '
            f'psi={math.degrees(self._psi):+6.1f}°  beta={math.degrees(self._guidance.beta):+5.1f}° '
            f'(ff {math.degrees(self._guidance.beta_ff):+5.1f}°, kappa={kappa_vor:+.3f})  '
            f'y_int={self._guidance.y_int:+5.2f}  u_d={u_d:.2f}  u={self._u:.2f}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = GuidanceILOSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()