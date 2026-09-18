"""
boat_control_node.py
=============================================================
ROS 2 Node: Regelungskaskade (Kurs- + Gierrate- + Tempolagen)

Subscriptions:
  /state/filtered   (nav_msgs/Odometry)   -- psi, r, u (gefilterter Zustand)
  /cmd/course_safe  (geometry_msgs/Twist) -- psi_c, u_c (nach SB-MPC)

Publications:
  /cmd/thrust  (geometry_msgs/Vector3Stamped)
    vector.x = F_L [N]   Schub links
    vector.y = F_R [N]   Schub rechts

Takt: 50 Hz Timer
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from boot_control.mc_common import BootState, DT
from boot_control.mc_control_node import ControlNode


class BoatControlNode(Node):
    def __init__(self):
        super().__init__('boat_control_node')

        self.declare_parameter('timer_hz', 50.0)

        self._ctrl = ControlNode()

        # Zustand
        self._state = BootState(psi=0.0, r=0.0, u=0.0)
        self._psi_c = 0.0
        self._u_c   = 0.0
        self._befehl_empfangen = False

        self.create_subscription(Odometry, '/state/filtered',   self._state_cb,  10)
        self.create_subscription(Twist,    '/cmd/course_safe',  self._course_cb, 10)

        self._pub = self.create_publisher(Vector3Stamped, '/cmd/thrust', 10)

        hz = self.get_parameter('timer_hz').value
        self._dt = 1.0 / hz
        self.create_timer(self._dt, self._timer_cb)

        self.get_logger().info('boat_control_node gestartet.')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._state = BootState(
            psi = psi,
            r   = msg.twist.twist.angular.z,
            u   = msg.twist.twist.linear.x,
        )

    # ------------------------------------------------------------------
    def _course_cb(self, msg: Twist):
        self._u_c   = msg.linear.x
        self._psi_c = msg.angular.z
        if not self._befehl_empfangen:
            self._ctrl.stossfreie_initialisierung(self._state)
            self._befehl_empfangen = True

    # ------------------------------------------------------------------
    def _timer_cb(self):
        if not self._befehl_empfangen:
            return

        thrust_cmd, safety = self._ctrl.regelzyklus(
            x                  = self._state,
            psi_c_eingang      = self._psi_c,
            u_c_eingang        = self._u_c,
            dt                 = self._dt,
            motorstrom         = 0.0,   # kein Strommessung in VRX
            befehl_empfangen   = True,
        )

        if not safety.ok:
            self.get_logger().warn(f'Safety: {safety.reason}')

        msg = Vector3Stamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.vector.x = float(thrust_cmd.F_L)
        msg.vector.y = float(thrust_cmd.F_R)
        self._pub.publish(msg)


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
