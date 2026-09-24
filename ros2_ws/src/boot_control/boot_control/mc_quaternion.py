"""
mc_quaternion.py
=============================================================
Minimale Quaternion-Hilfsfunktionen.

Grund: Alle Knoten haben bisher `tf_transformations` importiert. Das
Paket ist in vielen ROS-2-Installationen NICHT vorhanden (es kommt
aus python3-transforms3d und ist keine Abhaengigkeit von ros-base).
Fehlt es, stirbt der Knoten beim Start mit ModuleNotFoundError --
in der Launch-Ausgabe sieht das aus wie "Knoten laeuft nicht", ohne
dass klar wird warum.

Diese zwei Funktionen ersetzen den kompletten Bedarf des Pakets.
"""

import math
from geometry_msgs.msg import Quaternion


def yaw_from_quaternion(q) -> float:
    """Gierwinkel [rad] aus einer ROS-Quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """ROS-Quaternion aus reinem Gierwinkel (roll = pitch = 0)."""
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def euler_from_quaternion(q_list):
    """Kompatibilitaets-Ersatz fuer tf_transformations.euler_from_quaternion.
    Erwartet [x, y, z, w], liefert (roll, pitch, yaw)."""
    x, y, z, w = q_list
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw
