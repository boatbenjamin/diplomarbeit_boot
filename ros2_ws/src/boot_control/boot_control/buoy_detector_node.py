#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO


class BuoyDetectorNode(Node):
    def __init__(self):
        super().__init__(
            'buoy_detector_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        self.camera_sub_topic = '/wamv/sensors/cameras/front_left_camera_sensor/image_raw'
        self.detections_pub_topic = '/detections/raw'

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.model_path = self.get_parameter('model_path').value
        self.get_logger().info(f"Lade YOLO-Modell von: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # RELIABLE subscriber — passt zum ros_gz_bridge Publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_image = self.create_subscription(
            Image,
            self.camera_sub_topic,
            self.image_callback,
            qos,
        )
        self.get_logger().info(f"Subscribed to: {self.camera_sub_topic}")

        self.pub_detections = self.create_publisher(
            Detection2DArray,
            self.detections_pub_topic,
            10,
        )
        self.get_logger().info(f"Publishing to: {self.detections_pub_topic}")

    def image_callback(self, msg: Image):
        self.get_logger().info('Bild empfangen', throttle_duration_sec=5.0)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Fehler bei Bildkonvertierung: {e}")
            return

        try:
            results = self.model(cv_image, stream=True, verbose=False)
        except Exception as e:
            self.get_logger().error(f"YOLO-Fehler: {e}")
            return

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        for r in results:
            for box in r.boxes:
                det = Detection2D()
                det.header = msg.header

                xywh = box.xywh[0].cpu().numpy()
                det.bbox.center.position.x = float(xywh[0])
                det.bbox.center.position.y = float(xywh[1])
                det.bbox.size_x = float(xywh[2])
                det.bbox.size_y = float(xywh[3])

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(box.cls[0].item()))
                hyp.hypothesis.score = float(box.conf[0].item())
                det.results.append(hyp)

                detections_msg.detections.append(det)

        n = len(detections_msg.detections)
        if n:
            self.get_logger().info(f"{n} Detektionen", throttle_duration_sec=2.0)

        self.pub_detections.publish(detections_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
