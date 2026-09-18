import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import splprep, splev


# =====================================================================
# 1. MISSION MANAGER (VERBESSERT: Half-Plane Switching & Radius)
# =====================================================================
class MissionManager:
    def __init__(self, acceptance_radius: float = 4.0):
        self.current_gate = 0
        self.mission_completed = False
        self.acceptance_radius = acceptance_radius
        self.start_pos = None

    def check_gate_passage(self, boat_pos: np.ndarray, gate_centers: np.ndarray) -> bool:
        """
        Prüft mittels 'Half-Plane Switching' gekoppelt mit einem 'Acceptance Radius',
        ob das Tor gültig passiert wurde.
        """
        if self.current_gate >= len(gate_centers):
            self.mission_completed = True
            return False

        # Speichere die Startposition im ersten Zyklus für den ersten Anfahrtsvektor
        if self.start_pos is None:
            self.start_pos = np.copy(boat_pos)

        c_current = gate_centers[self.current_gate]

        # 1. Fahrtrichtungsvektor der EINKOMMENDEN Route bestimmen!
        # (Nicht der ausgehenden, da die Lichtschranke senkrecht zur Anfahrt stehen muss)
        if self.current_gate > 0:
            c_prev = gate_centers[self.current_gate - 1]
        else:
            c_prev = self.start_pos

        path_dir = c_current - c_prev
        norm = np.linalg.norm(path_dir)

        if norm > 1e-6:
            path_dir = path_dir / norm
        else:
            path_dir = np.array([1.0, 0.0])  # Fallback

        # 2. Vektor vom Torzentrum zur aktuellen Bootsposition
        v_boat = boat_pos - c_current
        dist_to_gate = np.linalg.norm(v_boat)

        # 3. Schaltlogik: Halbebene gekreuzt UND in der Nähe des Tores
        # Skalarprodukt > 0 bedeutet, die orthogonale Linie wurde überquert.
        crossed_half_plane = np.dot(v_boat, path_dir) > 0
        in_vicinity = dist_to_gate <= self.acceptance_radius

        # Das Tor gilt als passiert, wenn das Boot nah genug ist und die Linie kreuzt,
        # ODER wenn es das Zentrum exakt trifft (dist_to_gate < 0.5)
        if (crossed_half_plane and in_vicinity) or (dist_to_gate < 0.5):
            self.current_gate += 1
            if self.current_gate >= len(gate_centers):
                self.mission_completed = True
            return True

        return False