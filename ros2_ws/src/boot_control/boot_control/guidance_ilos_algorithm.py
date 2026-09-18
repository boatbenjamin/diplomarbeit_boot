"""
guidance_ilos_algorithm.py
=============================================================
ILOS-Führungsgesetz (Integral Line-of-Sight) für den autonomen
Bojenkurs (Kapitel 9 des Konzeptpapiers).

    psi_d = pi_h - atan2(y_e + sigma*y_int, Delta)
    y_int_dot = Delta * y_e / (Delta^2 + (y_e + sigma*y_int)^2)

Rein rechenlogisch, ROS-unabhaengig. Einbettung in den
/guidance_ilos-Knoten erfolgt in guidance_ilos_node.py.
"""

import math
import numpy as np

# =====================================================================
# ILOS-TUNING-PARAMETER (Anhang B)
# =====================================================================
DELTA = 2.0         # m, Lookahead-Distanz
SIGMA = 0.10        # Integratorverstaerkung (verhindert stationaere Querablage)
I_MAX = 3.0         # m, Integrator-Saettigung
U_MAX = 1.8         # m/s, maximale Fahrt
A_QUER_MAX = 1.0    # m/s^2, maximale Querbeschleunigung (fuer Kurvengeschwindigkeit)
K3 = 1.0            # Reduktionsfaktor: Fahrt sinkt bei grossem Kursfehler


# =====================================================================
# ILOS-KLASSE
# =====================================================================
class ILOSGuidance:
    def __init__(self, delta: float = DELTA, sigma: float = SIGMA, i_max: float = I_MAX,
                 u_max: float = U_MAX, a_quer_max: float = A_QUER_MAX, k3: float = K3):
        self.delta = delta
        self.sigma = sigma
        self.i_max = i_max
        self.u_max = u_max
        self.a_quer_max = a_quer_max
        self.k3 = k3
        self.y_int: float = 0.0

    def reset_integral(self) -> None:
        self.y_int = 0.0

    def update_heading(self, y_e: float, pi_h: float, dt: float) -> float:
        """Berechnet den Sollkurs psi_d [rad] aus Querablage und Pfadwinkel."""
        y_int_dot = (self.delta * y_e) / (self.delta ** 2 + (y_e + self.sigma * self.y_int) ** 2)
        self.y_int += y_int_dot * dt
        self.y_int = max(-self.i_max, min(self.i_max, self.y_int))
        psi_d = pi_h - math.atan2(y_e + self.sigma * self.y_int, self.delta)
        return (psi_d + math.pi) % (2 * math.pi) - math.pi

    def calculate_speed(self, kappa: float, current_yaw: float, psi_d: float) -> float:
        """Berechnet die Sollfahrt u_d [m/s] in Abhaengigkeit von Kruemmung und Kursfehler."""
        e_psi = (psi_d - current_yaw + math.pi) % (2 * math.pi) - math.pi
        u_curve = math.sqrt(self.a_quer_max / abs(kappa)) if abs(kappa) > 1e-3 else self.u_max
        u_heading = self.u_max * max(0.0, (1.0 - self.k3 * abs(e_psi)))
        u_d = min(self.u_max, u_curve, u_heading)
        return max(0.0, u_d)


# =====================================================================
# HILFSFUNKTION FUER PATH-TRACKING
# =====================================================================
def get_path_state(pos: np.ndarray, path_x: np.ndarray, path_y: np.ndarray):
    """Berechnet Querablage, Pfadwinkel und lokale Kruemmung zum naechstgelegenen
    Spline-Punkt. Gibt (y_e, pi_h, kappa, idx) zurueck."""
    distances = np.hypot(path_x - pos[0], path_y - pos[1])
    idx = np.argmin(distances)

    idx_prev = max(idx - 1, 0)
    idx_next = min(idx + 1, len(path_x) - 1)

    if idx_prev == idx_next:  # Array zu klein
        return 0.0, 0.0, 0.0, idx

    # 1. Pfadwinkel (pi_h) per zentraler Differenz
    dx = path_x[idx_next] - path_x[idx_prev]
    dy = path_y[idx_next] - path_y[idx_prev]
    pi_h = math.atan2(dy, dx)

    # 2. Querablage (y_e) via Projektion auf den Normalenvektor
    v_x, v_y = pos[0] - path_x[idx], pos[1] - path_y[idx]
    n_x, n_y = -math.sin(pi_h), math.cos(pi_h)
    y_e = v_x * n_x + v_y * n_y

    # 3. Kruemmung (kappa) per finiter Differenz (2. Ableitung)
    if 0 < idx < len(path_x) - 1:
        dx1 = (path_x[idx + 1] - path_x[idx - 1]) / 2.0
        dy1 = (path_y[idx + 1] - path_y[idx - 1]) / 2.0
        dx2 = path_x[idx + 1] - 2 * path_x[idx] + path_x[idx - 1]
        dy2 = path_y[idx + 1] - 2 * path_y[idx] + path_y[idx - 1]
        denom = (dx1 ** 2 + dy1 ** 2) ** 1.5
        kappa = (dx1 * dy2 - dy1 * dx2) / denom if denom > 1e-6 else 0.0
    else:
        kappa = 0.0

    return y_e, pi_h, kappa, idx

