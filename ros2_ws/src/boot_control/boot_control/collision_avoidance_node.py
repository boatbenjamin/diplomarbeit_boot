"""
collision_avoidance_node.py
=============================================================
ROS 2 Node: SB-MPC Kollisionsvermeidung

Subscriptions:
  /cmd/course      (geometry_msgs/Twist)     -- Soll-Kurs/-Fahrt von ILOS
  /objects         (geometry_msgs/PoseArray) -- bestaetigte Tracks
  /state/filtered  (nav_msgs/Odometry)        -- eigener Zustand

Publications:
  /cmd/course_safe  (geometry_msgs/Twist)
    linear.x  = u_safe  [m/s]
    angular.z = psi_safe [rad]

Takt: 10 Hz Timer
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

from boot_control.collision_avoidance_algorithm import (
    ScenarioBasedMPC, OwnState, Obstacle,
    KURSVERSAETZE_DEG, TEMPOFAKTOREN, T_HORIZON, DT_PRED,
)


class CollisionAvoidanceNode(Node):
    def __init__(self):
        super().__init__('collision_avoidance_node')

        self.declare_parameter('timer_hz', 10.0)

        self._sbmpc = ScenarioBasedMPC(
            kursversaetze_deg = KURSVERSAETZE_DEG,
            tempofaktoren     = TEMPOFAKTOREN,
            t_horizon         = T_HORIZON,
            dt_pred           = DT_PRED,
        )

        # Zustand
        self._psi_d = 0.0
        self._u_d   = 0.0
        self._own   = OwnState(x=0.0, y=0.0, psi=0.0, u=0.0)
        self._obstacles: list[Obstacle] = []

        self.create_subscription(Twist,    '/cmd/course',      self._course_cb,  10)
        self.create_subscription(PoseArray, '/objects',         self._objects_cb, 10)
        self.create_subscription(Odometry,  '/state/filtered',  self._state_cb,   10)

        self._pub = self.create_publisher(Twist, '/cmd/course_safe', 10)

        hz = self.get_parameter('timer_hz').value
        self.create_timer(1.0 / hz, self._timer_cb)

        self.get_logger().info('collision_avoidance_node gestartet.')

    # ------------------------------------------------------------------
    def _course_cb(self, msg: Twist):
        self._u_d   = msg.linear.x
        self._psi_d = msg.angular.z

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._own = OwnState(
            x   = msg.pose.pose.position.x,
            y   = msg.pose.pose.position.y,
            psi = psi,
            u   = msg.twist.twist.linear.x,
            r   = msg.twist.twist.angular.z,
        )

    # ------------------------------------------------------------------
    def _objects_cb(self, msg: PoseArray):
        self._obstacles = []
        for pose in msg.poses:
            self._obstacles.append(Obstacle(
                id      = int(pose.orientation.z),
                x       = pose.position.x,
                y       = pose.position.y,
                vx      = pose.orientation.x,
                vy      = pose.orientation.y,
                sigma_p = pose.position.z,
            ))

    # ------------------------------------------------------------------
    def _timer_cb(self):
        result = self._sbmpc.compute(
            own       = self._own,
            psi_d     = self._psi_d,
            u_d       = self._u_d,
            obstacles = self._obstacles,
        )

        msg = Twist()
        msg.linear.x  = float(result.u_safe)
        msg.angular.z = float(result.psi_safe)
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
