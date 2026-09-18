#!/usr/bin/env python3
"""
thrust_to_vrx_node.py
Leitet Schubkräfte [N] direkt an VRX-Thruster weiter.

Subscription:
  /cmd/thrust   (geometry_msgs/Vector3Stamped)
    vector.x = F_L [N]
    vector.y = F_R [N]

Publications:
  /wamv/thrusters/left/thrust   (std_msgs/Float64)  [N]
  /wamv/thrusters/right/thrust  (std_msgs/Float64)  [N]

VRX erwartet Newtonwerte direkt (Float64), kein normiertes [-1,1].
Korrekte Topics laut VRX-Bridge-Log:
  wamv/thrusters/left/thrust  (std_msgs/msg/Float64) -> gz.msgs.Double
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64

# Maximale Schubkraft zur Sättigung (WAM-V VRX default ~189 N, wir begrenzen auf F_MAX)
from boot_control.mc_common import F_MAX

# VRX WAM-V Thruster-Maximum (aus VRX SDF, sicher höher als F_MAX)
_VRX_THRUST_MAX = 189.0  # N


class ThrustToVrxNode(Node):
    def __init__(self):
        super().__init__('thrust_to_vrx_node')

        self.sub = self.create_subscription(
            Vector3Stamped, '/cmd/thrust', self._thrust_cb, 10)

        # Korrekte VRX-Topics: /wamv/thrusters/left/thrust (Float64)
        self.pub_left = self.create_publisher(
            Float64, '/wamv/thrusters/left/thrust', 10)
        self.pub_right = self.create_publisher(
            Float64, '/wamv/thrusters/right/thrust', 10)

        self.get_logger().info(
            f'thrust_to_vrx_node gestartet (F_MAX={F_MAX} N, VRX_MAX={_VRX_THRUST_MAX} N)'
        )

    def _thrust_cb(self, msg: Vector3Stamped):
        # Direkte Newton-Werte, gesättigt auf VRX-Maximum
        cmd_l = Float64()
        cmd_r = Float64()
        cmd_l.data = float(max(-_VRX_THRUST_MAX, min(_VRX_THRUST_MAX, msg.vector.x)))
        cmd_r.data = float(max(-_VRX_THRUST_MAX, min(_VRX_THRUST_MAX, msg.vector.y)))
        self.pub_left.publish(cmd_l)
        self.pub_right.publish(cmd_r)
        self.get_logger().debug(
            f'Thrust: L={cmd_l.data:.1f} N, R={cmd_r.data:.1f} N',
        )


def main(args=None):
    rclpy.init(args=args)
    node = ThrustToVrxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()