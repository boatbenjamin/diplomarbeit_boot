import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


class Detection:
    """Rohmessung aus der vorgelagerten 3D-Kamerageometrie/Perzeption."""

    def __init__(self, x: float, y: float, label: str = "unknown"):
        self.x = x
        self.y = y
        self.label = label


class Track:
    """
    Verfolgte Objektspur mit Kalman-Filter zur Zustandsschätzung [x, y, vx, vy]^T
    sowie M-aus-N-Bestätigungslogik.
    """
    _id_counter = 0

    def __init__(self, detection: Detection, q_std: float = 0.5, r_std: float = 0.8):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.label = detection.label

        # Zustand: x = [x, y, vx, vy]^T
        self.x = np.array([[detection.x], [detection.y], [0.0], [0.0]], dtype=np.float64)

        # Initialisierung der Zustandskovarianz P
        self.P = np.diag([1.0, 1.0, 4.0, 4.0])

        # Rausch-Kovarianzen
        self.q_std = q_std
        self.r_std = r_std
        self.R = np.diag([self.r_std ** 2, self.r_std ** 2])

        # Messmatrix H (gemessen werden nur Positionen x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Lebenszyklus-Parameter (M-aus-N)
        self.hit_history: List[bool] = [True]
        self.confirmed: bool = False
        self.time_since_update: float = 0.0

    def predict(self, dt: float):
        """Prädiktionsschritt des Kalman-Filters mit Constant-Velocity-Modell."""
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Prozessrauschmatrix Q
        Q = np.array([
            [0.25 * dt ** 4, 0.0, 0.5 * dt ** 3, 0.0],
            [0.0, 0.25 * dt ** 4, 0.0, 0.5 * dt ** 3],
            [0.5 * dt ** 3, 0.0, dt ** 2, 0.0],
            [0.0, 0.5 * dt ** 3, 0.0, dt ** 2]
        ], dtype=np.float64) * (self.q_std ** 2)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += dt

    def update(self, detection: Detection):
        """Korrekturschritt des Kalman-Filters nach erfolgreicher Assoziation."""
        z = np.array([[detection.x], [detection.y]], dtype=np.float64)

        y = z - (self.H @ self.x)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovationskovarianz
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman-Gain

        self.x = self.x + (K @ y)  # Zustandsupdate (Position & v_x, v_y)
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P  # Kovarianzupdate

        self.label = detection.label
        self.time_since_update = 0.0
        self.hit_history.append(True)

    def mark_missed(self):
        """Setzt eine fehlende Detektion im aktuellen Zyklus fest."""
        self.hit_history.append(False)

    def update_status(self, m_confirm: int = 3, n_window: int = 5):
        """Prüft die M-aus-N-Bedingung zur Bestätigung des Tracks."""
        if len(self.hit_history) > n_window:
            self.hit_history = self.hit_history[-n_window:]
        if self.hit_history.count(True) >= m_confirm:
            self.confirmed = True

    def mahalanobis_distance(self, detection: Detection) -> float:
        """Berechnet den Mahalanobis-Abstand unter Berücksichtigung von S = H*P*H^T + R."""
        z = np.array([[detection.x], [detection.y]], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            d2 = y.T @ np.linalg.inv(S) @ y
            return float(np.sqrt(d2[0, 0]))
        except np.linalg.LinAlgError:
            return float('inf')


class ObjectTracker:
    """Verwaltet Datenassoziation, Kalman-Filter-Tracking und Track-Lebenszyklen."""

    def __init__(self,
                 gate_threshold: float = 3.0,
                 max_age_seconds: float = 2.0,
                 m_confirm: int = 3,
                 n_window: int = 5):
        self.tracks: List[Track] = []
        self.gate_threshold = gate_threshold
        self.max_age_seconds = max_age_seconds
        self.m_confirm = m_confirm
        self.n_window = n_window

    def step(self, detections: List[Detection], dt: float) -> List[Track]:
        """Führt den vollen Tracking-Zyklus für einen Abtastschritt dt aus."""
        # 1. Prädiktion aller bestehenden Tracks
        for track in self.tracks:
            track.predict(dt)

        # 2. Datenassoziation (Mahalanobis-Kostenmatrix + Ungarische Methode)
        matched_t, matched_d, unmatched_t, unmatched_d = self._associate(detections)

        # 3. Kalman-Update für zugeordnete Spuren
        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(detections[d_idx])

        # 4. Verpasste Frames für unvollständige Zuordnungen vermerken
        for t_idx in unmatched_t:
            self.tracks[t_idx].mark_missed()

        # 5. Tentative Tracks für freie Detektionen erzeugen
        for d_idx in unmatched_d:
            self.tracks.append(Track(detections[d_idx]))

        # 6. M-aus-N Status prüfen & Veraltete Tracks löschen
        active_tracks = []
        for track in self.tracks:
            track.update_status(m_confirm=self.m_confirm, n_window=self.n_window)
            if track.time_since_update <= self.max_age_seconds:
                active_tracks.append(track)
        self.tracks = active_tracks

        # Nur bestätigte Spuren zur Steuerung/Pfadplanung zurückgeben
        return [t for t in self.tracks if t.confirmed]

    def _associate(self, detections: List[Detection]) -> Tuple[List[int], List[int], List[int], List[int]]:
        if not self.tracks or not detections:
            return [], [], list(range(len(self.tracks))), list(range(len(detections)))

        # Mahalanobis-Kostenmatrix aufbauen
        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float64)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                cost_matrix[t_idx, d_idx] = track.mahalanobis_distance(det)

        # Globale Zuordnung via Ungarischer Methode (Global Nearest Neighbor)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks, matched_dets = [], []
        unmatched_tracks = list(set(range(len(self.tracks))) - set(row_ind))
        unmatched_dets = list(set(range(len(detections))) - set(col_ind))

        # Gating über Mahalanobis-Schwellwert
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.gate_threshold:
                unmatched_tracks.append(r)
                unmatched_dets.append(c)
            else:
                matched_tracks.append(r)
                matched_dets.append(c)

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

