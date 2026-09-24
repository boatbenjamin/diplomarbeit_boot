"""
boat_control_node.py
=============================================================
ROS 2 Node: Regelungskaskade (Kurs -> Gierrate -> Schub)

Subscriptions:
  /state/filtered   (nav_msgs/Odometry)   -- psi, r, u
  /cmd/course_safe  (geometry_msgs/Twist) -- linear.x = u_c, angular.z = psi_c
  /mission/status   (std_msgs/String)     -- "COMPLETED" haelt das Boot an

Publications:
  /cmd/thrust       (geometry_msgs/Vector3Stamped)  vector.x = F_L, vector.y = F_R
  /diag/control     (geometry_msgs/Vector3Stamped)  x = e_psi[deg], y = r_d[deg/s], z = r[deg/s]

Aenderungen ggue. der alten Version
-----------------------------------
1. `befehl_empfangen=True` wurde in JEDEM Timer-Takt gesetzt. Damit
   wurde der Watchdog in mc_safety_monitor bei jedem Zyklus
   zurueckgesetzt und konnte nie ausloesen -- bleibt /cmd/course_safe
   aus, fuhr das Boot ewig auf dem letzten Befehl weiter.
   Jetzt: neuer_befehl() nur im Subscriber-Callback.
2. stossfreie_initialisierung() lief auf dem Default-Zustand
   (psi=r=u=0), wenn der erste Fahrbefehl vor der ersten
   Zustandsnachricht ankam. Der Kursregler startete dann mit einem
   falschen Referenzkurs. Jetzt wird auf /state/filtered gewartet.
3. Plattformparameter (d_y, f_max, Traegheit, Daempfung,
   Reglerbandbreite) sind ROS-Parameter -> config/params.yaml.
4. Es wird aktiv 0 N publiziert, wenn nicht geregelt wird. Vorher
   wurde einfach nichts gesendet -- der VRX-Thruster haelt aber den
   letzten Wert und das Boot faehrt unkontrolliert weiter.
5. Kein tf_transformations mehr (siehe mc_quaternion.py).
6. /mission/status = COMPLETED stoppt das Boot (AI_REQ_10).
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from boot_control.mc_common import BootParameter, BootState, wrap_pi
from boot_control.mc_control_node import ControlNode
from boot_control.mc_quaternion import yaw_from_quaternion


class BoatControlNode(Node):
    def __init__(self):
        super().__init__('boat_control_node')

        # --- Regeltakt ---
        self.declare_parameter('timer_hz', 50.0)
        # --- Plattform (Defaults = VRX WAM-V) ---
        self.declare_parameter('d_y', 1.027135)
        self.declare_parameter('f_max', 500.0)
        self.declare_parameter('r_max_deg', 45.0)
        self.declare_parameter('u_max', 2.2)
        self.declare_parameter('izz', 700.0)
        self.declare_parameter('n_r', 800.0)
        self.declare_parameter('masse', 250.0)
        self.declare_parameter('x_u', 100.0)
        self.declare_parameter('x_uu', 150.0)
        self.declare_parameter('omega_i', 1.5)
        self.declare_parameter('omega_a_faktor', 5.0)
        self.declare_parameter('dpsi_max_deg', 25.0)
        self.declare_parameter('stoppe_bei_mission_completed', True)

        g = lambda n: self.get_parameter(n).value
        self._p = BootParameter(
            d_y=float(g('d_y')), f_max=float(g('f_max')),
            r_max=math.radians(float(g('r_max_deg'))), u_max=float(g('u_max')),
            izz=float(g('izz')), n_r=float(g('n_r')),
            masse=float(g('masse')), x_u=float(g('x_u')), x_uu=float(g('x_uu')),
            omega_i=float(g('omega_i')), omega_a_faktor=float(g('omega_a_faktor')),
            dpsi_max=math.radians(float(g('dpsi_max_deg'))),
        )
        self._stop_on_completed = bool(g('stoppe_bei_mission_completed'))
        self._ctrl = ControlNode(p=self._p)

        # --- Zustand ---
        self._state = BootState(0.0, 0.0, 0.0)
        self._psi_c = 0.0
        self._u_c = 0.0
        self._state_empfangen = False
        self._initialisiert = False
        self._mission_fertig = False

        self.create_subscription(Odometry, '/state/filtered', self._state_cb, 10)
        self.create_subscription(Twist, '/cmd/course_safe', self._course_cb, 10)
        self.create_subscription(String, '/mission/status', self._mission_cb, 10)

        self._pub = self.create_publisher(Vector3Stamped, '/cmd/thrust', 10)
        self._pub_diag = self.create_publisher(Vector3Stamped, '/diag/control', 10)

        self._dt = 1.0 / float(g('timer_hz'))
        self.create_timer(self._dt, self._timer_cb)

        self.get_logger().info(
            f'boat_control_node gestartet: d_y={self._p.d_y:.3f} m, '
            f'f_max={self._p.f_max:.0f} N, N_max={self._p.n_max:.0f} Nm, '
            f'Kp_r={self._p.izz * self._p.omega_i:.0f}, Kp_psi={self._p.omega_a:.2f}')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        self._state = BootState(
            psi=yaw_from_quaternion(msg.pose.pose.orientation),
            r=msg.twist.twist.angular.z,
            u=msg.twist.twist.linear.x,
        )
        self._state_empfangen = True
        self._ctrl.safety.neuer_zustand()

    def _course_cb(self, msg: Twist):
        self._u_c = msg.linear.x
        self._psi_c = wrap_pi(msg.angular.z)
        # NUR hier -- sonst kann der Watchdog nie ausloesen
        self._ctrl.safety.neuer_befehl(self._psi_c, self._u_c)

        if not self._initialisiert and self._state_empfangen:
            self._ctrl.stossfreie_initialisierung(self._state)
            self._initialisiert = True
            self.get_logger().info(
                f'Regler stossfrei initialisiert bei psi={math.degrees(self._state.psi):.1f}°')

    def _mission_cb(self, msg: String):
        if msg.data == 'COMPLETED' and not self._mission_fertig:
            self._mission_fertig = True
            self.get_logger().info('Mission abgeschlossen -- Boot wird gestoppt.')

    # ------------------------------------------------------------------
    def _publiziere(self, f_l: float, f_r: float):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x = float(f_l)
        msg.vector.y = float(f_r)
        self._pub.publish(msg)

    # ------------------------------------------------------------------
    def _timer_cb(self):
        if self._mission_fertig and self._stop_on_completed:
            self._publiziere(0.0, 0.0)
            return

        if not self._initialisiert:
            # Aktiv 0 N senden: der VRX-Thruster haelt sonst den letzten Wert
            self._publiziere(0.0, 0.0)
            if not self._state_empfangen:
                self.get_logger().warn('Warte auf /state/filtered ...',
                                       throttle_duration_sec=5.0)
            else:
                self.get_logger().warn('Warte auf /cmd/course_safe ...',
                                       throttle_duration_sec=5.0)
            return

        thrust, safety = self._ctrl.regelzyklus(
            x=self._state, psi_c_eingang=self._psi_c, u_c_eingang=self._u_c,
            dt=self._dt, motorstrom=None,
        )

        if not safety.ok:
            self.get_logger().warn(f'Safety: {safety.reason}', throttle_duration_sec=2.0)

        self._publiziere(thrust.F_L, thrust.F_R)

        diag = Vector3Stamped()
        diag.header.stamp = self.get_clock().now().to_msg()
        diag.vector.x = math.degrees(wrap_pi(self._psi_c - self._state.psi))
        diag.vector.y = math.degrees(self._ctrl.letztes_r_d)
        diag.vector.z = math.degrees(self._state.r)
        self._pub_diag.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = BoatControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
