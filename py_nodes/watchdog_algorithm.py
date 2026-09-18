import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, Bool
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition


class SystemWatchdog(Node):
    def __init__(self):
        super().__init__('watchdog_node')

        # Konfiguration
        self.timeout_sec = 2.0
        self.max_retries = 3
        self.fail_count = 0
        self.last_heartbeat = time.time()
        self.target_node_name = '/guidance_ilos'  # Beispielhafter Zielknoten

        # Abonnieren des Lebenszeichen-Topics
        self.heartbeat_sub = self.create_subscription(
            Header,
            '/watchdog',
            self.heartbeat_callback,
            rclpy.qos.qos_profile_sensor_data  # Geringe Latenz für Systemüberwachung
        )

        # Publisher für den Abbruch der Mission bei wiederholtem Fehler
        self.mission_abort_pub = self.create_publisher(Bool, '/system/abort', 10)

        # Service Client für Lifecycle-Übergänge des Zielknotens
        self.change_state_client = self.create_client(
            ChangeState,
            f'{self.target_node_name}/change_state'
        )

        # Überwachungstakt (0.5s)
        self.timer = self.create_timer(0.5, self.monitor_loop)
        self.get_logger().info('Watchdog initialisiert. Überwache Lebenszeichen...')

    def heartbeat_callback(self, msg):
        """Aktualisiert den Zeitstempel bei jedem eintreffenden Heartbeat."""
        self.last_heartbeat = time.time()
        if self.fail_count > 0:
            self.get_logger().info('Knoten hat sich erholt, Zähler zurückgesetzt.')
            self.fail_count = 0

    def monitor_loop(self):
        """Prüft, ob das Lebenszeichen-Topic innerhalb der Frist ausblieb."""
        if (time.time() - self.last_heartbeat) > self.timeout_sec:
            self.get_logger().error(
                f'Timeout! Kein Lebenszeichen von {self.target_node_name} seit {self.timeout_sec}s.'
            )
            self.handle_failure()
            # Setze Zeitstempel voraus, um Spam während der Reset-Phase zu vermeiden
            self.last_heartbeat = time.time() + 5.0

    def handle_failure(self):
        """Führt Knoten-Neustart durch oder bricht bei Wiederholung die Mission ab."""
        self.fail_count += 1

        if self.fail_count <= self.max_retries:
            self.get_logger().warn(
                f'Versuch {self.fail_count}/{self.max_retries}: Starte Knoten neu.'
            )
            self.trigger_lifecycle_transition(Transition.TRANSITION_DEACTIVATE)
        else:
            self.get_logger().fatal('Wiederholter Ausfall! Mission wird abgebrochen.')
            abort_msg = Bool()
            abort_msg.data = True
            self.mission_abort_pub.publish(abort_msg)

    def trigger_lifecycle_transition(self, transition_id):
        """Sendet einen asynchronen Request für einen Lifecycle-Zustandswechsel."""
        if not self.change_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Lifecycle-Service des Zielknotens nicht erreichbar!')
            return

        req = ChangeState.Request()
        req.transition.id = transition_id
        self.change_state_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    watchdog_node = SystemWatchdog()

    try:
        rclpy.spin(watchdog_node)
    except KeyboardInterrupt:
        pass
    finally:
        watchdog_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()