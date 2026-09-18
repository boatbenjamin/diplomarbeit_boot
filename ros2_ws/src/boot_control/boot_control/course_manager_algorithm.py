"""
course_manager_algorithm.py
=============================================================
Reine Berechnungslogik fuer die Figure-8-Bahnplanung (offizielle
Monaco-AI-Class-Kursvorgabe, Kap. 2). Ersetzt die alte Rot/Gruen-
Tor-Logik vollstaendig.

Kernidee:
  - Marken werden ueber ihre Position wiedererkannt (Nearest-Neighbour
    zu bereits bekannten Marken), NICHT mehr ueber eine Farbe. Das
    behebt das alte Problem, dass ein blindes zip() nach x-Sortierung
    bei jedem Frame eine andere Zuordnung liefern konnte.
  - Sobald eine Marke oft genug gesehen wurde, wird ihre Position
    eingefroren (laufender Mittelwert). Danach wird sie nicht mehr
    aktualisiert -- die Kursvorgabe sagt explizit "All positions
    known", die Marken bewegen sich nicht.
  - Die Bahn: zwei Kreise mit Radius R = 0.5*|AB| um die Marken A und
    B, die sich im Mittelpunkt M tangential beruehren. Ein Kreis wird
    im Uhrzeigersinn, der andere gegen den Uhrzeigersinn durchfahren --
    nur so ist der Pfad an der Kreuzungsstelle M richtungsstetig
    (keine Ecke, kein Sprung im Sollkurs). Nachgerechnet: an M zeigen
    beide Bewegungsrichtungen exakt in dieselbe Richtung.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Buoy:
    x: float
    y: float


@dataclass
class _TrackedMark:
    x: float
    y: float
    n_hits: int = 1
    locked: bool = False

    def update(self, x: float, y: float, lock_after: int):
        if self.locked:
            return
        # laufender Mittelwert (Welford), stabiler als neu ueberschreiben
        self.n_hits += 1
        self.x += (x - self.x) / self.n_hits
        self.y += (y - self.y) / self.n_hits
        if self.n_hits >= lock_after:
            self.locked = True


class CourseManager:
    """Figure-8-Bahnplanung um zwei feste Marken."""

    def __init__(self,
                 expected_marks: int = 2,
                 match_gate: float = 5.0,
                 lock_after: int = 5):
        self._expected_marks = expected_marks
        self._match_gate = match_gate
        self._lock_after = lock_after
        self._marks: list[_TrackedMark] = []   # Reihenfolge = Identitaet (A=[0], B=[1])
        self._path_cache = None                # (path_x, path_y), sobald berechnet

    # ------------------------------------------------------------------
    def update_objects(self, buoys: list[Buoy]):
        """Ordnet neue Detektionen bestehenden Marken zu (Nearest Neighbour)."""
        if self._path_cache is not None:
            return  # Marken sind bereits gesperrt, keine weitere Arbeit noetig

        for b in buoys:
            best_idx, best_dist = None, self._match_gate
            for i, m in enumerate(self._marks):
                d = float(np.hypot(b.x - m.x, b.y - m.y))
                if d < best_dist:
                    best_idx, best_dist = i, d

            if best_idx is not None:
                self._marks[best_idx].update(b.x, b.y, self._lock_after)
            elif len(self._marks) < self._expected_marks:
                self._marks.append(_TrackedMark(x=b.x, y=b.y))

    @property
    def marks_ready(self) -> bool:
        return (len(self._marks) == self._expected_marks
                and all(m.locked for m in self._marks))

    @property
    def expected_marks(self) -> int:
        return self._expected_marks

    @property
    def num_marks_found(self) -> int:
        return len(self._marks)

    # ------------------------------------------------------------------
    def compute_figure8_path(self,
                              num_points: int = 400,
                              backward_extension: float = 15.0,
                              loop_radius_factor: float = 0.5,
                              num_laps: int = 1):
        """
        Liefert (path_x, path_y) fuer die komplette Figure-8-Bahn inkl.
        Anlaufbahn. Wird nur EINMAL berechnet und danach gecacht.
        """
        if self._path_cache is not None:
            return self._path_cache

        if not self.marks_ready:
            return np.array([]), np.array([])

        A = np.array([self._marks[0].x, self._marks[0].y])
        B = np.array([self._marks[1].x, self._marks[1].y])
        d = B - A
        dist_ab = float(np.hypot(d[0], d[1]))
        if dist_ab < 1e-3:
            return np.array([]), np.array([])

        R = loop_radius_factor * dist_ab
        angle_A0 = float(np.arctan2(d[1], d[0]))   # Winkel von A Richtung M
        angle_B0 = angle_A0 + np.pi                # Winkel von B Richtung M

        pts_per_loop = max(8, num_points // max(1, 2 * num_laps))

        xs, ys = [], []
        for _ in range(num_laps):
            # Uhrzeigersinn um A (Winkel abnehmend)
            theta = np.linspace(angle_A0, angle_A0 - 2 * np.pi, pts_per_loop)
            xs.extend((A[0] + R * np.cos(theta)).tolist())
            ys.extend((A[1] + R * np.sin(theta)).tolist())

            # Gegen-Uhrzeigersinn um B (Winkel zunehmend)
            theta = np.linspace(angle_B0, angle_B0 + 2 * np.pi, pts_per_loop)
            xs.extend((B[0] + R * np.cos(theta)).tolist())
            ys.extend((B[1] + R * np.sin(theta)).tolist())

        path_x = np.array(xs)
        path_y = np.array(ys)

        # Anlaufbahn: Tangente am ersten Punkt zurueckverlaengern, damit der
        # ILOS-Regler nicht auf einen Sprungstart reagieren muss.
        dx0 = path_x[1] - path_x[0]
        dy0 = path_y[1] - path_y[0]
        norm = float(np.hypot(dx0, dy0))
        if norm > 1e-6:
            dx0, dy0 = dx0 / norm, dy0 / norm
            n_ext = max(2, int(backward_extension * 2))
            ext_dists = np.linspace(-backward_extension, 0, n_ext, endpoint=False)
            ext_x = path_x[0] + ext_dists * dx0
            ext_y = path_y[0] + ext_dists * dy0
            path_x = np.concatenate((ext_x, path_x))
            path_y = np.concatenate((ext_y, path_y))

        self._path_cache = (path_x, path_y)
        return self._path_cache

    def reset(self):
        self._marks.clear()
        self._path_cache = None