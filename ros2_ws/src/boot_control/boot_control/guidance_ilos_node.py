"""
guidance_ilos_node.py
=============================================================
ROS 2 Node: ILOS-Fuehrungsgesetz (Integral Line-of-Sight)

Subscriptions:
  /state/filtered  (nav_msgs/Odometry)  -- gefilterter Zustand
  /path            (nav_msgs/Path)       -- B-Spline-Referenzpfad

Publications:
  /cmd/course  (geometry_msgs/Twist)
    linear.x  = u_d  [m/s]    Soll-Fahrt
    angular.z = psi_d [rad]   Soll-Kurs

Takt: 10 Hz Timer (veroeffentlicht letzten berechneten Wert)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from boot_control.mc_quaternion import euler_from_quaternion

from boot_control.guidance_ilos_algorithm import ILOSGuidance, get_path_state


class GuidanceILOSNode(Node):
    def __init__(self):
        super().__init__('guidance_ilos_node')

        self.declare_parameter('delta',       2.0)   # m, Lookahead-Distanz
        self.declare_parameter('sigma',       0.10)  # Integrator-Verstaerkung
        self.declare_parameter('i_max',       3.0)   # m, Integrator-Saettigung
        self.declare_parameter('u_max',       1.8)   # m/s
        self.declare_parameter('a_quer_max',  1.0)   # m/s^2
        self.declare_parameter('k3',          1.0)   # Kursfehler-Daempfung
        self.declare_parameter('u_min',       0.5)   # m/s, Mindestfahrt (Ruderwirkung)
        self.declare_parameter('timer_hz',    10.0)  # Hz

        self._guidance = ILOSGuidance(
            delta      = self.get_parameter('delta').value,
            sigma      = self.get_parameter('sigma').value,
            i_max      = self.get_parameter('i_max').value,
            u_max      = self.get_parameter('u_max').value,
            a_quer_max = self.get_parameter('a_quer_max').value,
            k3         = self.get_parameter('k3').value,
            u_min      = self.get_parameter('u_min').value,
        )

        # Zustand
        self._pos = np.zeros(2)
        self._psi = 0.0
        self._dt  = 1.0 / self.get_parameter('timer_hz').value
        self._last_stamp = None

        self._path_x: np.ndarray = np.empty(0)
        self._path_y: np.ndarray = np.empty(0)

        self.create_subscription(Odometry, '/state/filtered', self._state_cb, 10)
        self.create_subscription(Path,     '/path',           self._path_cb,  10)

        self._pub = self.create_publisher(Twist, '/cmd/course', 10)

        hz = self.get_parameter('timer_hz').value
        self.create_timer(1.0 / hz, self._timer_cb)

        self.get_logger().info('guidance_ilos_node gestartet.')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        self._pos[0] = msg.pose.pose.position.x
        self._pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self._psi = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ------------------------------------------------------------------
    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        self._path_x = np.array([ps.pose.position.x for ps in msg.poses])
        self._path_y = np.array([ps.pose.position.y for ps in msg.poses])
        self._guidance.reset_integral()

    # ------------------------------------------------------------------
    def _timer_cb(self):
        if len(self._path_x) < 2:
            self.get_logger().warn('Kein /path empfangen -- ILOS inaktiv.',
                                   throttle_duration_sec=5.0)
            return

        # echtes dt statt des nominalen Timer-dt (wichtig fuer den
        # ILOS-Integrator, wenn die Sim langsamer als Echtzeit laeuft)
        t_now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_stamp is None:
            self._last_stamp = t_now
            return
        dt_real = t_now - self._last_stamp
        self._last_stamp = t_now
        if not (1e-4 < dt_real < 1.0):
            return
        self._dt = dt_real

        y_e, pi_h, kappa, _ = get_path_state(self._pos, self._path_x, self._path_y)

        psi_d = self._guidance.update_heading(y_e, pi_h, self._dt)
        u_d   = self._guidance.calculate_speed(kappa, self._psi, psi_d)

        msg = Twist()
        msg.linear.x  = float(u_d)
        msg.angular.z = float(psi_d)
        self._pub.publish(msg)


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
