"""
pose_to_odom_node.py   (NEU)
=============================================================
Schliesst die Luecke bei /wamv/odom.

Problem: wave_filter_node abonniert /wamv/odom, aber NIEMAND publiziert
darauf. In `ros2 topic list` taucht das Topic nur auf, weil ein Abonnent
existiert. Ohne Odometrie bleiben Position und Fahrt im Wellenfilter auf
0.0 stehen, /state/filtered wird nie sinnvoll -> Mission-Manager,
Guidance und Regler haben keinen Zustand.

Loesung: VRX bridget die Ground-Truth-Pose des Modells world-unabhaengig
auf /wamv/pose (tf2_msgs/TFMessage, aus gz /model/wamv/pose). Die wird
hier in nav_msgs/Odometry umgesetzt; die Geschwindigkeit wird numerisch
differenziert und in den Bootsrahmen (surge/sway) gedreht.

HINWEIS: Das ist Ground Truth aus der Simulation, kein echter Sensor.
Fuer die Diplomarbeit sauber als solche kennzeichnen. Fuer einen
realistischen Aufbau spaeter GPS (/wamv/sensors/gps/gps/fix) + IMU
fusionieren und diesen Node ersetzen.

Subscription : /wamv/pose  (tf2_msgs/TFMessage)
Publication  : /wamv/odom  (nav_msgs/Odometry)
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from boot_control.mc_quaternion import euler_from_quaternion


class PoseToOdomNode(Node):
    def __init__(self):
        super().__init__('pose_to_odom_node')

        self.declare_parameter('child_frame', 'wamv/base_link')
        self.declare_parameter('odom_frame', 'odom')

        self._child = self.get_parameter('child_frame').value
        self._odom_frame = self.get_parameter('odom_frame').value

        self._last = None          # (t, x, y)
        self._u = 0.0
        self._v = 0.0
        self._last_psi = None
        self._r = 0.0

        self.create_subscription(TFMessage, '/wamv/pose', self._tf_cb, 10)
        self._pub = self.create_publisher(Odometry, '/wamv/odom', 10)

        self.get_logger().info(
            f'pose_to_odom_node gestartet (child_frame="{self._child}").')

    # ------------------------------------------------------------------
    def _tf_cb(self, msg: TFMessage):
        tf = None
        for t in msg.transforms:
            if t.child_frame_id == self._child:
                tf = t
                break
        if tf is None:
            # Beim ersten Mal die verfuegbaren Frames zeigen, damit ein
            # falscher child_frame-Parameter sofort auffaellt.
            self.get_logger().warn(
                f'child_frame "{self._child}" nicht in /wamv/pose. '
                f'Vorhanden: {[t.child_frame_id for t in msg.transforms]}',
                throttle_duration_sec=10.0)
            return

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        _, _, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])

        t_now = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9

        if self._last is not None:
            dt = t_now - self._last[0]
            if 1e-4 < dt < 1.0:
                vx_w = (x - self._last[1]) / dt      # Weltrahmen
                vy_w = (y - self._last[2]) / dt
                # in den Bootsrahmen drehen: surge / sway
                self._u = vx_w * math.cos(psi) + vy_w * math.sin(psi)
                self._v = -vx_w * math.sin(psi) + vy_w * math.cos(psi)
                if self._last_psi is not None:
                    dpsi = math.atan2(math.sin(psi - self._last_psi),
                                      math.cos(psi - self._last_psi))
                    self._r = dpsi / dt
        self._last = (t_now, x, y)
        self._last_psi = psi

        odom = Odometry()
        odom.header.stamp = tf.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._child
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = tf.transform.translation.z
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = self._u
        odom.twist.twist.linear.y = self._v
        odom.twist.twist.angular.z = self._r
        self._pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()