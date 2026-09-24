"""
course_test_node.py   (NEU)
=============================================================
Testknoten fuer genau den Fall "ich gebe einen Sollkurs vor und
pruefe, ob er erreicht wird".

Publiziert /cmd/course_safe (Twist) mit festem oder durchlaufendem
Sollkurs und loggt fortlaufend den Kursfehler. Ersetzt das
fehleranfaellige manuelle `ros2 topic pub` (dort wird gerne vergessen,
dass angular.z in RAD erwartet wird und dass ohne -r 10 nur EINMAL
gesendet wird -- dann greift der Watchdog nach 3 s und das Boot
stoppt, was wie ein Reglerfehler aussieht).

Parameter:
  psi_c_deg      Sollkurs in GRAD (wird intern nach rad gewandelt)
  u_c            Sollfahrt [m/s]
  rate_hz        Sendetakt (muss > 1/t_halt sein, sonst Watchdog!)
  sequenz_deg    Optional: Liste von Kursen, die nacheinander
                 angefahren werden, z.B. [0.0, 90.0, 180.0, -90.0]
  sequenz_dauer  s je Kurs der Sequenz

Aufruf:
  ros2 run boot_control course_test --ros-args -p psi_c_deg:=90.0 -p u_c:=1.5
  ros2 run boot_control course_test --ros-args \
      -p "sequenz_deg:=[0.0, 90.0, 180.0, -90.0]" -p sequenz_dauer:=40.0
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from boot_control.mc_quaternion import yaw_from_quaternion
from boot_control.mc_common import wrap_pi


class CourseTestNode(Node):
    def __init__(self):
        super().__init__('course_test_node')

        self.declare_parameter('psi_c_deg', 90.0)
        self.declare_parameter('u_c', 1.5)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('sequenz_deg', [0.0])
        self.declare_parameter('sequenz_dauer', 0.0)
        self.declare_parameter('log_periode', 2.0)

        self._u_c = float(self.get_parameter('u_c').value)
        self._psi_c = math.radians(float(self.get_parameter('psi_c_deg').value))
        self._seq = [math.radians(float(v)) for v in self.get_parameter('sequenz_deg').value]
        self._seq_dauer = float(self.get_parameter('sequenz_dauer').value)
        self._log_periode = float(self.get_parameter('log_periode').value)

        self._psi_ist = None
        self._u_ist = 0.0
        self._t0 = None
        self._t_log = None

        self.create_subscription(Odometry, '/state/filtered', self._state_cb, 10)
        self._pub = self.create_publisher(Twist, '/cmd/course_safe', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)

        if self._seq_dauer > 0.0 and len(self._seq) > 1:
            folge = ', '.join(f'{math.degrees(v):.0f}°' for v in self._seq)
            self.get_logger().info(
                f'course_test_node: Sequenz [{folge}], je {self._seq_dauer:.0f} s, '
                f'u_c={self._u_c} m/s')
        else:
            self.get_logger().info(
                f'course_test_node: Sollkurs {math.degrees(self._psi_c):.1f}°, '
                f'u_c={self._u_c} m/s')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        self._psi_ist = yaw_from_quaternion(msg.pose.pose.orientation)
        self._u_ist = msg.twist.twist.linear.x

    def _jetzt(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------
    def _aktueller_sollkurs(self, t: float) -> float:
        if self._seq_dauer > 0.0 and len(self._seq) > 1:
            idx = int(t / self._seq_dauer) % len(self._seq)
            return self._seq[idx]
        return self._psi_c

    def _tick(self):
        t = self._jetzt()
        if self._t0 is None:
            self._t0, self._t_log = t, t
            return
        dt_ges = t - self._t0

        psi_c = self._aktueller_sollkurs(dt_ges)

        msg = Twist()
        msg.linear.x = self._u_c
        msg.angular.z = psi_c          # RAD
        self._pub.publish(msg)

        if t - self._t_log >= self._log_periode:
            self._t_log = t
            if self._psi_ist is None:
                self.get_logger().warn('Noch kein /state/filtered empfangen.')
            else:
                e = math.degrees(wrap_pi(psi_c - self._psi_ist))
                self.get_logger().info(
                    f't={dt_ges:6.1f}s  soll={math.degrees(psi_c):7.1f}°  '
                    f'ist={math.degrees(self._psi_ist):7.1f}°  '
                    f'Fehler={e:+6.1f}°  u={self._u_ist:.2f} m/s')


def main(args=None):
    rclpy.init(args=args)
    node = CourseTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
