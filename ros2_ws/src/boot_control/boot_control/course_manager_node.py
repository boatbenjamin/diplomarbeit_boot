"""
course_manager_node.py
=============================================================
ROS 2 Node: Bahnplanung fuer die offizielle Monaco-AI-Class-Kursvorgabe.

AKTUELL IMPLEMENTIERT: Figure 8 (2 feste Marken, siehe AI Course
Description Kap. 2). Slalom / Docking / Fleet sind eigene Algorithmen,
die spaeter in course_manager_algorithm.py ergaenzt werden -- dieser
Node muesste dafuer nur `race_type` auswerten und die passende
compute_*-Methode aufrufen.

Subscriptions:
  /objects  (geometry_msgs/PoseArray)  -- bestaetigte Bojen-Tracks aus
    object_tracker_node. WICHTIG: orientation.w (Farbe) wird NICHT mehr
    ausgewertet -- bei Figure 8 sind beide Marken weiss, es gibt keine
    Rot/Gruen-Unterscheidung mehr. Nur x/y werden verwendet.

Publications:
  /path  (nav_msgs/Path)
    Kompletter Figure-8-Pfad inkl. Anlaufbahn. Wird EINMAL berechnet,
    sobald beide Marken sicher erkannt + gesperrt sind, danach nicht
    mehr veraendert (Marken bewegen sich laut Kursvorgabe nicht).

ENTFERNT ggue. der alten Version:
  - /mission/target_gate Subscription. Es gibt kein Zwischenziel mehr,
    der komplette Pfad wird auf einmal geplant.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path

from boot_control.course_manager_algorithm import CourseManager, Buoy


class CourseManagerNode(Node):
    def __init__(self):
        super().__init__('course_manager_node')

        self.declare_parameter('num_path_points', 400)
        self.declare_parameter('backward_extension', 15.0)
        self.declare_parameter('loop_radius_factor', 0.5)
        self.declare_parameter('num_laps', 1)
        self.declare_parameter('expected_marks', 2)
        self.declare_parameter('mark_match_gate', 5.0)
        self.declare_parameter('mark_lock_after', 5)

        self._num_pts = self.get_parameter('num_path_points').value
        self._bwd_ext = self.get_parameter('backward_extension').value
        self._loop_radius_factor = self.get_parameter('loop_radius_factor').value
        self._num_laps = self.get_parameter('num_laps').value

        self._manager = CourseManager(
            expected_marks=self.get_parameter('expected_marks').value,
            match_gate=self.get_parameter('mark_match_gate').value,
            lock_after=self.get_parameter('mark_lock_after').value,
        )

        self._path_published = False

        self.create_subscription(PoseArray, '/objects', self._objects_cb, 10)
        self._pub = self.create_publisher(Path, '/path', 10)

        self.get_logger().info('course_manager_node gestartet (Figure-8-Modus).')

    # ------------------------------------------------------------------
    def _objects_cb(self, msg: PoseArray):
        if self._path_published:
            return  # Pfad steht bereits fest

        buoys = [Buoy(x=p.position.x, y=p.position.y) for p in msg.poses]
        self._manager.update_objects(buoys)

        if not self._manager.marks_ready:
            self.get_logger().warn(
                f'Warte auf Marken: {self._manager.num_marks_found}/'
                f'{self._manager.expected_marks} erkannt, noch nicht alle gesperrt.',
                throttle_duration_sec=2.0,
            )
            return

        self._compute_and_publish()

    # ------------------------------------------------------------------
    def _compute_and_publish(self):
        path_x, path_y = self._manager.compute_figure8_path(
            num_points=self._num_pts,
            backward_extension=self._bwd_ext,
            loop_radius_factor=self._loop_radius_factor,
            num_laps=self._num_laps,
        )

        if len(path_x) == 0:
            self.get_logger().warn('Kein Pfad berechnet (Marken zu nah beieinander?).')
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for x, y in zip(path_x, path_y):
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self._pub.publish(path_msg)
        self._path_published = True
        self.get_logger().info(f'Figure-8-Pfad berechnet und gesperrt: {len(path_x)} Punkte.')


def main(args=None):
    rclpy.init(args=args)
    node = CourseManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()