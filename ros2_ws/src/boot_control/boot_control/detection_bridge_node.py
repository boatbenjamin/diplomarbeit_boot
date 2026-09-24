"""
detection_bridge_node.py
=============================================================
ROS 2 Node: Konvertiert YOLO Detection2DArray → PoseArray (3D)

Subscriptions:
  /detections/raw   (vision_msgs/Detection2DArray)  -- YOLO Bounding Boxes
  /wamv/sensors/cameras/front_left_camera_sensor/camera_info  (CameraInfo)
  /wamv/sensors/cameras/front_left_camera_sensor/image_raw    (Image)

Publications:
  /detections  (geometry_msgs/PoseArray)   frame_id = 'odom'
    pose.position.x = x [m]  WELTKOORDINATE (odom)
    pose.position.y = y [m]  WELTKOORDINATE (odom)
    pose.position.z = Detektionsunsicherheit (sigma_p)
    pose.orientation.x = vx (0.0 — keine Geschwindigkeit hier)
    pose.orientation.y = vy (0.0)
    pose.orientation.z = float(detection_id)
    pose.orientation.w = +1.0 (rote Boje) | -1.0 (grüne Boje)

Tiefenschätzung: d = BOJE_DURCHMESSER_M * fx / bbox_breite_px
Farberkennung:   HSV-Analyse im erkannten Bildausschnitt
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from boot_control.mc_quaternion import euler_from_quaternion


# Bekannte Bojengrösse in VRX (Kanalmarker ~0.35 m Durchmesser)
BOJE_DURCHMESSER_M = 0.35


class DetectionBridgeNode(Node):
    def __init__(self):
        super().__init__('detection_bridge_node')

        self.declare_parameter('boje_durchmesser', BOJE_DURCHMESSER_M)
        self._boje_d = self.get_parameter('boje_durchmesser').value

        self._bridge = CvBridge()

        # Kamera-Intrinsik (Standardwerte bis camera_info eintrifft)
        self._fx = 530.0
        self._fy = 530.0
        self._cx = 640.0
        self._cy = 360.0
        self._have_info = False

        # Letztes Kamerabild für Farbanalyse
        self._last_image: np.ndarray | None = None

        # FIX: Bootszustand für die Transformation Boot-Frame -> Weltframe.
        # Ohne das publiziert dieser Node Koordinaten RELATIV ZUM BOOT,
        # während course_manager_node sie als 'odom' deklariert und
        # mission_manager_node sie mit der Weltposition des Boots vergleicht.
        # Der Kalman-Filter im Tracker (Constant-Velocity) würde ausserdem
        # feste Bojen als bewegt schätzen, sobald das Boot fährt.
        self._boat_xy = np.zeros(2)
        self._boat_psi = 0.0
        self._have_state = False

        self.create_subscription(
            Odometry, '/state/filtered', self._state_cb, 10)

        self.create_subscription(
            CameraInfo,
            '/wamv/sensors/cameras/front_left_camera_sensor/camera_info',
            self._info_cb, 1)

        self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self._image_cb, qos_profile_sensor_data)

        self.create_subscription(
            Detection2DArray,
            '/detections/raw',
            self._det_cb, 10)

        self._pub = self.create_publisher(PoseArray, '/detections', 10)

        self.get_logger().info('detection_bridge_node gestartet.')

    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        if not self._have_info:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]
            self._have_info = True
            self.get_logger().info(
                f'Kamera-Intrinsik: fx={self._fx:.1f} fy={self._fy:.1f} '
                f'cx={self._cx:.1f} cy={self._cy:.1f}')

    # ------------------------------------------------------------------
    def _state_cb(self, msg: Odometry):
        self._boat_xy[0] = msg.pose.pose.position.x
        self._boat_xy[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self._boat_psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._have_state = True

    # ------------------------------------------------------------------
    def _image_cb(self, msg: Image):
        try:
            self._last_image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _det_cb(self, msg: Detection2DArray):
        if not self._have_state:
            self.get_logger().warn(
                'Noch kein /state/filtered — Detektionen werden verworfen, '
                'da sie nicht in Weltkoordinaten umgerechnet werden können.',
                throttle_duration_sec=5.0)
            return

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'

        cpsi = math.cos(self._boat_psi)
        spsi = math.sin(self._boat_psi)

        for i, det in enumerate(msg.detections):
            bbox = det.bbox
            w = bbox.size_x
            h = bbox.size_y
            u = bbox.center.position.x
            v = bbox.center.position.y

            if w < 1.0:
                continue  # Degenerierte Box überspringen

            # ── Tiefenschätzung ─────────────────────────────────────────────
            d = self._boje_d * self._fx / w          # Entfernung [m]
            d = max(0.5, min(d, 50.0))               # Clamp 0.5–50 m

            # ── 3D-Position im Boot-Frame (Kamera ≈ Bootsnase) ─────────────
            x_boat = d                               # vorwärts
            y_boat = -(u - self._cx) * d / self._fx  # seitlich (neg = BB)

            # ── Farbanalyse ─────────────────────────────────────────────────
            # ── Farbe aus YOLO-Klasse (Fine-Tuned-Modell kennt Rot/Grün direkt) ──
            if not det.results:
                continue
            try:
                cls_id = int(det.results[0].hypothesis.class_id)
            except (ValueError, IndexError):
                continue
            CLASS_TO_COLOR = {2: -1.0, 4: 1.0}  # Green=-1, Red=+1
            if cls_id not in CLASS_TO_COLOR:
                continue  # schwarze/weisse/orange Marker sind keine Gate-Bojen
            color_w = CLASS_TO_COLOR[cls_id]

            # ── FIX: Boot-Frame -> Weltframe (odom) ────────────────────────
            x_w = self._boat_xy[0] + x_boat * cpsi - y_boat * spsi
            y_w = self._boat_xy[1] + x_boat * spsi + y_boat * cpsi

            # ── Unsicherheit (wächst mit Entfernung) ───────────────────────
            sigma_p = 0.1 + 0.02 * d

            pose = Pose()
            pose.position.x  = float(x_w)
            pose.position.y  = float(y_w)
            pose.position.z  = float(sigma_p)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(i)           # Detektions-ID
            pose.orientation.w = float(color_w)     # +1=rot, -1=grün

            out.poses.append(pose)

        self._pub.publish(out)

    # ------------------------------------------------------------------
    def _detect_color(self, u, v, w, h) -> float:
        """Gibt +1.0 für rot, -1.0 für grün zurück (0.0 = unbekannt → rot)."""
        if self._last_image is None:
            return 1.0

        img_h, img_w = self._last_image.shape[:2]
        x1 = max(0, int(u - w / 2))
        x2 = min(img_w, int(u + w / 2))
        y1 = max(0, int(v - h / 2))
        y2 = min(img_h, int(v + h / 2))

        roi = self._last_image[y1:y2, x1:x2]
        if roi.size == 0:
            return 1.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Rot: H ∈ [0,10] ∪ [160,180]
        rot_mask  = cv2.inRange(hsv, (0,  80, 50), (10,  255, 255)) | \
                    cv2.inRange(hsv, (160, 80, 50), (180, 255, 255))
        # Grün: H ∈ [35,85]
        gruen_mask = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))

        n_rot   = int(cv2.countNonZero(rot_mask))
        n_gruen = int(cv2.countNonZero(gruen_mask))

        if n_rot >= n_gruen:
            return  1.0   # rot
        else:
            return -1.0   # grün


def main(args=None):
    rclpy.init(args=args)
    node = DetectionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()