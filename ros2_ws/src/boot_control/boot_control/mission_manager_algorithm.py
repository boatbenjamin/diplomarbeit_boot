"""
mission_manager_algorithm.py
=============================================================
Reine Logik zur Fortschritts- und Abschlusserkennung entlang eines vom
course_manager gelieferten Pfades. Ersetzt die alte Halbebenen-Tor-
Logik (die feste Rot/Gruen-Torpaare voraussetzte).

Funktionsprinzip: "Furthest-progress-index"
  Es wird der Pfadpunkt gesucht, der der aktuellen Bootsposition am
  naechsten liegt. Der am WEITESTEN erreichte Index wird gespeichert
  und kann nur vorwaerts wandern (Sensorrauschen darf ihn nicht
  zuruecksetzen). Sobald dieser Index nahe genug am letzten Punkt des
  Pfades liegt, gilt die Mission als abgeschlossen.

  Vorteil: funktioniert fuer JEDEN Pfad (Figure 8, spaeter Slalom,
  Docking-Anfahrt, ...), unabhaengig von der konkreten Kursgeometrie --
  diese Klasse muss die Kursform selbst nicht kennen.
"""

import numpy as np


class MissionManager:
    def __init__(self, acceptance_radius: float = 4.0, min_progress_fraction: float = 0.95):
        self._acceptance_radius = acceptance_radius
        self._min_progress_fraction = min_progress_fraction
        self._furthest_idx = 0
        self.mission_completed = False

    # ------------------------------------------------------------------
    def reset(self):
        self._furthest_idx = 0
        self.mission_completed = False

    # ------------------------------------------------------------------
    def update(self, boat_pos: np.ndarray, path_x: np.ndarray, path_y: np.ndarray):
        if self.mission_completed or len(path_x) == 0:
            return

        dists = np.hypot(path_x - boat_pos[0], path_y - boat_pos[1])
        nearest_idx = int(np.argmin(dists))

        if nearest_idx > self._furthest_idx:
            self._furthest_idx = nearest_idx

        last_idx = len(path_x) - 1
        progress_fraction = self._furthest_idx / max(1, last_idx)
        dist_to_end = float(np.hypot(path_x[last_idx] - boat_pos[0],
                                      path_y[last_idx] - boat_pos[1]))

        if (progress_fraction >= self._min_progress_fraction
                and dist_to_end <= self._acceptance_radius):
            self.mission_completed = True

    @property
    def furthest_index(self) -> int:
        return self._furthest_idx