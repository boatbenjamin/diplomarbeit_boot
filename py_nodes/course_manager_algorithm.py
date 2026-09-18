from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


@dataclass
class Buoy:
    x: float
    y: float
    color: str


class CourseManager:
    def __init__(self):
        self.current_objects = []
        self.current_target_gate = 0

    def update_objects(self, buoys: list[Buoy]):
        self.current_objects = buoys

    def update_mission(self, target_gate_index: int):
        self.current_target_gate = target_gate_index

    def calculate_gate_centers(self) -> np.ndarray:
        """Paart rote und grüne Bojen und liefert die Torzentren."""
        reds = sorted([b for b in self.current_objects if b.color == 'red'], key=lambda b: b.x)
        greens = sorted([b for b in self.current_objects if b.color == 'green'], key=lambda b: b.x)

        centers = []
        for rb, gb in zip(reds, greens):
            centers.append([(rb.x + gb.x) / 2.0, (rb.y + gb.y) / 2.0])
        return np.array(centers)

    def compute_smoothed_path(self, num_points: int = 50, backward_extension: float = 15.0):
        """
        Berechnet den Referenzpfad durch die Torzentren und verlängert ihn
        geradlinig nach hinten, um dem ILOS-Regler eine Anlaufbahn zu bieten.
        """
        gate_centers = self.calculate_gate_centers()
        valid_gates = gate_centers[self.current_target_gate:]

        if len(valid_gates) == 0:
            return np.array([]), np.array([])

        if len(valid_gates) == 1:
            return np.array([valid_gates[0, 0]]), np.array([valid_gates[0, 1]])

        waypoints_x = valid_gates[:, 0]
        waypoints_y = valid_gates[:, 1]

        try:
            # 1. Spline-Interpolation durch die anstehenden Tore
            k_deg = min(3, len(waypoints_x) - 1)
            tck, u = splprep([waypoints_x, waypoints_y], s=0.0, k=k_deg)

            u_fine = np.linspace(0, 1, num_points)
            path_x, path_y = splev(u_fine, tck)

            # 2. Wissenschaftlicher Standard: Rückwärtsverlängerung (Anlaufbahn)
            # Berechne die 1. Ableitung (Tangente) des Splines exakt am Start (u=0)
            dx, dy = splev(0, tck, der=1)
            tangent_norm = np.hypot(dx, dy)

            if tangent_norm > 0:
                # Normieren auf einen Einheitsvektor
                dx, dy = dx / tangent_norm, dy / tangent_norm

                # Erzeuge Punkte nach hinten (entgegen der Tangentenrichtung)
                # endpoint=False verhindert, dass das erste Tor doppelt im Array liegt
                ext_dists = np.linspace(-backward_extension, 0, int(backward_extension * 2), endpoint=False)
                ext_x = path_x[0] + ext_dists * dx
                ext_y = path_y[0] + ext_dists * dy

                # Hänge die Anlaufbahn VOR den eigentlichen Kurvenpfad
                path_x = np.concatenate((ext_x, path_x))
                path_y = np.concatenate((ext_y, path_y))

            return path_x, path_y

        except Exception as e:
            print(f"Spline-Fehler: {e}")
            return np.array([]), np.array([])


def main():
    manager = CourseManager()

    buoys = [
        Buoy(4.0, 2.0, 'red'), Buoy(4.0, -2.0, 'green'),  # Tor 0
        Buoy(9.0, 6.0, 'red'), Buoy(11.0, 2.0, 'green'),  # Tor 1 (Schikane)
        Buoy(15.0, 5.0, 'red'), Buoy(15.0, 1.0, 'green'),  # Tor 2
        Buoy(20.0, 2.0, 'red'), Buoy(20.0, -2.0, 'green')  # Tor 3
    ]

    manager.update_objects(buoys)
    manager.update_mission(target_gate_index=0)

    # Pfadberechnung mit 15 Metern Anlaufbahn
    path_x, path_y = manager.compute_smoothed_path(backward_extension=15.0)
    centers = manager.calculate_gate_centers()

    # Visualisierung
    plt.figure(figsize=(12, 6))

    # Bojen plotten
    plt.plot([b.x for b in buoys if b.color == 'red'], [b.y for b in buoys if b.color == 'red'], 'ro', markersize=10,
             label='Backbord')
    plt.plot([b.x for b in buoys if b.color == 'green'], [b.y for b in buoys if b.color == 'green'], 'go',
             markersize=10, label='Steuerbord')

    # Boot simulieren
    boat_x, boat_y = -6.0, -3.0
    plt.plot(boat_x, boat_y, 'k^', markersize=12, label='Boot (wartet auf Kurs)')

    # ILOS-Veranschaulichung auf die NEUE Anlaufbahn
    # Sucht grob den Punkt auf dem Pfad, der dem Boot auf der X-Achse am nächsten ist
    closest_idx = np.argmin(np.abs(path_x - boat_x))
    plt.arrow(boat_x, boat_y, path_x[closest_idx] - boat_x, path_y[closest_idx] - boat_y,
              color='gray', linestyle='--', head_width=0.4, length_includes_head=True, label='ILOS Cross-Track-Fehler')

    # Zentren und gesamten Pfad plotten
    plt.plot(centers[:, 0], centers[:, 1], 'kx', label='Torzentren')

    # Markierung der Verlängerung zur besseren Sichtbarkeit
    plt.plot(path_x, path_y, 'b-', linewidth=2.5, label='Integrierter Pfad (inkl. Anlaufbahn)')
    plt.plot(path_x[0], path_y[0], 'bs', label='Virtueller Startpunkt (-15m)')

    plt.title('Wissenschaftlicher ILOS-Standard: Lineare Rückwärtsverlängerung (Backward Extrapolation)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()