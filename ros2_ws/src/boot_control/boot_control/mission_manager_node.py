"""
mission_manager_node.py
=============================================================
ROS 2 Node: Missionsfortschritt & Abschlusserkennung.

Ersetzt die alte Halbebenen-Tor-Logik durch eine kursform-unabhaengige
Fortschrittsverfolgung entlang des vom course_manager gelieferten /path.

Subscriptions:
  /path             (nav_msgs/Path)      -- geplanter Kurs
  /state/filtered   (nav_msgs/Odometry)  -- gefilterte Bootsposition

Publications:
  /mission/status   (std_msgs/String)    -- "RUNNING" | "COMPLETED"

ENTFERNT ggue. der alten Version:
  - /mission/target_gate Publisher (course_manager braucht kein
    Zwischenziel mehr).

WICHTIG: Diese Node stoppt das Boot NICHT selbst! /mission/status
muss von boat_control_node (oder einer eigenen Safety-Node) ausgewertet
werden, um AI_REQ_10 zu erfuellen (Boot muss nach Rennende stoppen und
2 physische Aktionen vor Neustart verlangen). Ich kenne den Inhalt von
boat_control_node.py nicht -- das ist aktuell nicht verdrahtet, bitte
separat pruefen/ergaenzen.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String

from boot_control.mission_manager_algorithm import MissionManager


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        self.declare_parameter('acceptance_radius', 4.0)
        self.declare_parameter('min_progress_fraction', 0.95)

        self._manager = MissionManager(
            acceptance_radius=self.get_parameter('acceptance_radius').value,
            min_progress_fraction=self.get_parameter('min_progress_fraction').value,
        )

        self._path_x = np.array([])
        self._path_y = np.array([])

        self.create_subscription(Path, '/path', self._path_cb, 10)
        self.create_subscription(Odometry, '/state/filtered', self._state_cb, 10)

        self._pub_status = self.create_publisher(String, '/mission/status', 10)

        self.get_logger().info('mission_manager_node gestartet.')

    # ------------------------------------------------------------------
    def _path_cb(self, msg: Path):
        if len(msg.poses) == 0:
            return
        self._path_x = np.array([p.pose.position.x for p in msg.poses])
        self._path_y = np.array([p.pose.position.y for p in msg.poses])

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        boat_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        if len(self._path_x) > 0:
            self._manager.update(boat_pos, self._path_x, self._path_y)

        status_msg = String()
        status_msg.data = 'COMPLETED' if self._manager.mission_completed else 'RUNNING'
        self._pub_status.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()