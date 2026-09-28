
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64


class ThrustToVrxNode(Node):
    def __init__(self):
        super().__init__('thrust_to_vrx_node')

        self.declare_parameter('vrx_thrust_max', 1000.0)   # N je Thruster,
        #                                                    VRX selbst laesst 2353 N zu
        self.declare_parameter('cmd_timeout', 0.5)         # s ohne Befehl -> 0 N
        self.declare_parameter('publish_hz', 50.0)
        self.declare_parameter('left_topic', '/wamv/thrusters/left/thrust')
        self.declare_parameter('right_topic', '/wamv/thrusters/right/thrust')

        self._max = float(self.get_parameter('vrx_thrust_max').value)
        self._timeout = float(self.get_parameter('cmd_timeout').value)

        self._f_l = 0.0
        self._f_r = 0.0
        self._t_last_cmd = None
        self._timeout_gemeldet = False

        self.create_subscription(Vector3Stamped, '/cmd/thrust', self._thrust_cb, 10)
        self._pub_left = self.create_publisher(
            Float64, self.get_parameter('left_topic').value, 10)
        self._pub_right = self.create_publisher(
            Float64, self.get_parameter('right_topic').value, 10)

        self.create_timer(1.0 / float(self.get_parameter('publish_hz').value), self._tick)

        self.get_logger().info(
            f'thrust_to_vrx_node gestartet (Limit {self._max:.0f} N je Thruster, '
            f'Timeout {self._timeout:.2f} s).')


    def _jetzt(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _clip(self, v: float) -> float:
        return max(-self._max, min(self._max, float(v)))


    def _thrust_cb(self, msg: Vector3Stamped):
        self._f_l = self._clip(msg.vector.x)
        self._f_r = self._clip(msg.vector.y)
        self._t_last_cmd = self._jetzt()
        if self._timeout_gemeldet:
            self.get_logger().info('/cmd/thrust wieder da.')
            self._timeout_gemeldet = False


    def _tick(self):
        t = self._jetzt()
        if self._t_last_cmd is None or (t - self._t_last_cmd) > self._timeout:
            if self._t_last_cmd is not None and not self._timeout_gemeldet:
                self.get_logger().error(
                    '/cmd/thrust seit %.2f s ausgeblieben -- Schub auf 0 N.'
                    % self._timeout)
                self._timeout_gemeldet = True
            f_l = f_r = 0.0
        else:
            f_l, f_r = self._f_l, self._f_r

        left, right = Float64(), Float64()
        left.data, right.data = f_l, f_r
        self._pub_left.publish(left)
        self._pub_right.publish(right)


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
