"""
object_tracker_node.py
=============================================================
ROS 2 Node: Objekt-Tracking (Kalman-Filter + Ungarische Methode)

Subscriptions:
  /detections  (geometry_msgs/PoseArray)
    - pose.position.x  -> x [m]
    - pose.position.y  -> y [m]
    - pose.position.z    -> sigma_p (Detektionsunsicherheit) [m]
    - pose.orientation.w -> Farbe: +1.0 = rot (Backbord), -1.0 = gruen (Steuerbord)

Publications:
  /objects  (geometry_msgs/PoseArray)
    Nur bestaetigte Tracks (M-aus-N). Pro Pose:
    - position.x  -> Track-X [m]
    - position.y  -> Track-Y [m]
    - position.z  -> sigma_p (Positionsunsicherheit) [m]
    - orientation.x -> vx [m/s]
    - orientation.y -> vy [m/s]
    - orientation.z -> track_id (float)
    - orientation.w -> 1.0 wenn rot, -1.0 wenn gruen

Takt: callback-getrieben mit /detections (typisch 10 Hz aus Kamerasystem)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np

from boot_control.object_tracker_algorithm import Detection, ObjectTracker


class ObjectTrackerNode(Node):
    def __init__(self):
        super().__init__('object_tracker_node')

        # Parameter
        self.declare_parameter('gate_threshold', 3.0)   # Mahalanobis-Schwelle
        self.declare_parameter('max_age_seconds', 2.0)  # Track-Lebenszeit ohne Update
        self.declare_parameter('m_confirm', 3)          # M aus N: Hits fuer Bestaetigung
        self.declare_parameter('n_window', 5)           # N: Fenstergröße
        self.declare_parameter('dt', 0.1)               # s, Nominaler Zeitschritt

        gate = self.get_parameter('gate_threshold').value
        max_age = self.get_parameter('max_age_seconds').value
        m_conf = self.get_parameter('m_confirm').value
        n_win = self.get_parameter('n_window').value
        self._dt = self.get_parameter('dt').value

        self._tracker = ObjectTracker(
            gate_threshold=gate,
            max_age_seconds=max_age,
            m_confirm=m_conf,
            n_window=n_win,
        )

        self._last_stamp = None

        self.create_subscription(PoseArray, '/detections', self._detections_cb, 10)
        self._pub = self.create_publisher(PoseArray, '/objects', 10)

        self.get_logger().info('object_tracker_node gestartet.')

    # ------------------------------------------------------------------
    def _detections_cb(self, msg: PoseArray):
        now = msg.header.stamp
        if self._last_stamp is None:
            dt = self._dt
        else:
            dt = (now.sec - self._last_stamp.sec) + \
                 (now.nanosec - self._last_stamp.nanosec) * 1e-9
            dt = max(1e-3, min(dt, 1.0))
        self._last_stamp = now

        # Rohmessungen dekodieren
        detections = []
        for pose in msg.poses:
            # FIX: Farbe steht in orientation.w (+1 rot / -1 gruen), so wie
            # detection_bridge_node sie schreibt. position.z ist sigma_p und
            # damit IMMER > 0 -> vorher wurde jede Boje als 'red' gelabelt,
            # es gab nie ein rot/gruen-Paar und damit nie ein Torzentrum.
            color = 'red' if pose.orientation.w > 0.0 else 'green'
            detections.append(Detection(
                x=pose.position.x,
                y=pose.position.y,
                label=color,
            ))

        confirmed_tracks = self._tracker.step(detections, dt)

        # Bestaetigte Tracks publizieren
        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id

        for track in confirmed_tracks:
            # track.x ist numpy-Array (4,1): [x, y, vx, vy]
            tx = float(track.x[0, 0])
            ty = float(track.x[1, 0])
            tvx = float(track.x[2, 0])
            tvy = float(track.x[3, 0])

            # Positionsunsicherheit: sqrt(Spur der 2x2-Positionskovarianz)
            sigma_p = float(np.sqrt(track.P[0, 0] + track.P[1, 1]))

            color_w = 1.0 if track.label == 'red' else -1.0

            p = Pose()
            p.position.x = tx
            p.position.y = ty
            p.position.z = sigma_p           # Unsicherheit
            p.orientation.x = tvx
            p.orientation.y = tvy
            p.orientation.z = float(track.track_id)
            p.orientation.w = color_w
            out.poses.append(p)

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()