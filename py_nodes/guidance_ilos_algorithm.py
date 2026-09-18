import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import splprep, splev


# =====================================================================
# 1. WEGPLANUNG (COURSE MANAGER MIT B-SPLINES)
# =====================================================================
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
        reds = sorted([b for b in self.current_objects if b.color == 'red'], key=lambda b: b.x)
        greens = sorted([b for b in self.current_objects if b.color == 'green'], key=lambda b: b.x)
        centers = [[(rb.x + gb.x) / 2.0, (rb.y + gb.y) / 2.0] for rb, gb in zip(reds, greens)]
        return np.array(centers)

    def compute_smoothed_path(self, num_points: int = 500, backward_extension: float = 15.0):
        # num_points=500 für eine hohe Auflösung, um die Krümmung (kappa) sauber abzuleiten
        gate_centers = self.calculate_gate_centers()
        valid_gates = gate_centers[self.current_target_gate:]

        if len(valid_gates) < 2:
            return np.array([]), np.array([])

        waypoints_x = valid_gates[:, 0]
        waypoints_y = valid_gates[:, 1]

        try:
            k_deg = min(3, len(waypoints_x) - 1)
            tck, u = splprep([waypoints_x, waypoints_y], s=0.0, k=k_deg)
            u_fine = np.linspace(0, 1, num_points)
            path_x, path_y = splev(u_fine, tck)

            # Rückwärtsverlängerung für ILOS-Anlauf
            dx, dy = splev(0, tck, der=1)
            tangent_norm = np.hypot(dx, dy)
            if tangent_norm > 0:
                dx, dy = dx / tangent_norm, dy / tangent_norm
                ext_dists = np.linspace(-backward_extension, 0, int(backward_extension * 10), endpoint=False)
                ext_x = path_x[0] + ext_dists * dx
                ext_y = path_y[0] + ext_dists * dy
                path_x = np.concatenate((ext_x, path_x))
                path_y = np.concatenate((ext_y, path_y))

            return path_x, path_y
        except Exception as e:
            print(f"Spline-Fehler: {e}")
            return np.array([]), np.array([])


# =====================================================================
# 2. ZENTRALE TUNING- UND SIMULATIONSPARAMETER
# =====================================================================
DELTA = 2.0
SIGMA = 0.10
I_MAX = 3.0
U_MAX = 1.8
A_QUER_MAX = 1.0
K3 = 1.0

KP_YAW = 7.0
MAX_YAW_RATE = 1.2
KP_THRUST = 120.0
MAX_THRUST_FORCE = 25.0

M_BOAT = 8.0
RHO_AIR, CW_AIR, A_AIR = 1.225, 0.6, 0.08
RHO_WATER, CW_WATER, A_WATER = 1000.0, 0.4, 0.04

BASE_WIND = np.array([2.5, -1.5])
GUST_AMP_X, GUST_AMP_Y = 1.2, 0.8
BASE_CURRENT = np.array([0.2, 0.1])
VORTEX_CENTER = np.array([5.0, 2.0])
VORTEX_STRENGTH = 0.8

RANDOM_SEED = 42
POS_NOISE_MAG = 0.4
YAW_NOISE_MAG = 0.2
WIND_NOISE_STD = 0.4
CURRENT_NOISE_STD = 0.05
WIND_TURB_STD = 0.15
WATER_TURB_STD = 0.02

DT = 0.05
MAX_ZYKLEN = 800
PRINT_INTERVAL = 50

# Boot startet vor dem Tor-Aufbau, leicht versetzt zur Ideal-Mittellinie
POS_INIT = np.array([-8.0, 3.0])
VEL_INIT = np.array([0.0, 0.0])
YAW_INIT = 0.0

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

POS_START = POS_INIT + np.random.uniform(-POS_NOISE_MAG, POS_NOISE_MAG, size=2)
YAW_START = YAW_INIT + np.random.uniform(-YAW_NOISE_MAG, YAW_NOISE_MAG)
BASE_WIND_ACTIVE = BASE_WIND + np.random.normal(0, WIND_NOISE_STD, size=2)
BASE_CURRENT_ACTIVE = BASE_CURRENT + np.random.normal(0, CURRENT_NOISE_STD, size=2)


# =====================================================================
# 3. ILOS-KLASSE UND UMWELTFELDER (Unverändert)
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
        y_int_dot = (self.delta * y_e) / (self.delta ** 2 + (y_e + self.sigma * self.y_int) ** 2)
        self.y_int += y_int_dot * dt
        self.y_int = max(-self.i_max, min(self.i_max, self.y_int))
        psi_d = pi_h - math.atan2(y_e + self.sigma * self.y_int, self.delta)
        return (psi_d + math.pi) % (2 * math.pi) - math.pi

    def calculate_speed(self, kappa: float, current_yaw: float, psi_d: float) -> float:
        e_psi = (psi_d - current_yaw + math.pi) % (2 * math.pi) - math.pi
        u_curve = math.sqrt(self.a_quer_max / abs(kappa)) if abs(kappa) > 1e-3 else self.u_max
        u_heading = self.u_max * max(0.0, (1.0 - self.k3 * abs(e_psi)))
        u_d = min(self.u_max, u_curve, u_heading)
        return max(0.0, u_d)


class EnvironmentalFields:
    @staticmethod
    def get_wind_velocity(pos: np.ndarray, t: float) -> np.ndarray:
        gust_x = GUST_AMP_X * math.sin(0.5 * t + 0.3 * pos[0])
        gust_y = GUST_AMP_Y * math.cos(0.7 * t + 0.2 * pos[1])
        turb = np.random.normal(0, WIND_TURB_STD, size=2)
        return BASE_WIND_ACTIVE + np.array([gust_x, gust_y]) + turb

    @staticmethod
    def get_water_velocity(pos: np.ndarray, t: float) -> np.ndarray:
        diff = pos - VORTEX_CENTER
        r = np.linalg.norm(diff)
        vortex_vel = np.array([-diff[1], diff[0]]) * (VORTEX_STRENGTH / (r ** 2)) if r > 0.1 else np.zeros(2)
        turb = np.random.normal(0, WATER_TURB_STD, size=2)
        return BASE_CURRENT_ACTIVE + vortex_vel + turb


def compute_drag_force(v_fluid: np.ndarray, v_obj: np.ndarray, rho: float, C_w: float, A: float) -> np.ndarray:
    v_rel = v_fluid - v_obj
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-6: return np.zeros(2)
    return 0.5 * rho * C_w * A * v_rel_norm * v_rel


# =====================================================================
# 4. HILFSFUNKTION FÜR PATH-TRACKING (Neu)
# =====================================================================
def get_path_state(pos: np.ndarray, path_x: np.ndarray, path_y: np.ndarray):
    """Berechnet Querablage, Pfadwinkel und lokale Krümmung zum nächstgelegenen Spline-Punkt."""
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

    # 3. Krümmung (kappa) per finiter Differenz (2. Ableitung)
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


# =====================================================================
# 5. SIMULATIONSSCHLEIFE
# =====================================================================
if __name__ == "__main__":
    # Pfadgenerierung initiieren
    manager = CourseManager()
    buoys = [
        Buoy(0.0, 2.0, 'red'), Buoy(0.0, -2.0, 'green'),
        Buoy(7.0, 5.0, 'red'), Buoy(9.0, 1.0, 'green'),
        Buoy(15.0, 6.0, 'red'), Buoy(15.0, 2.0, 'green'),
        Buoy(22.0, 1.0, 'red'), Buoy(22.0, -3.0, 'green')
    ]
    manager.update_objects(buoys)
    manager.update_mission(0)

    # Referenzpfad generieren (inkl. 15m Rückwärtsverlängerung)
    path_x, path_y = manager.compute_smoothed_path(backward_extension=15.0)

    guidance = ILOSGuidance()
    time_total = 0.0
    pos, vel, current_yaw = POS_START.copy(), VEL_INIT.copy(), YAW_START

    history_x, history_y = [], []

    print("Start der dynamischen ILOS-Spline Simulation...")

    for zyklus in range(1, MAX_ZYKLEN):
        # 1. Pfadstatus geometrisch analysieren (ersetzt starre Geradengleichung)
        y_e, pi_h, kappa, current_idx = get_path_state(pos, path_x, path_y)

        # Stop-Bedingung: Wenn das Boot am Ende des Pfades ankommt
        if current_idx >= len(path_x) - 2:
            print(f"Ziel erreicht nach {time_total:.1f}s!")
            break

        # 2. ILOS berechnet Sollkurs und Sollfahrt
        psi_d = guidance.update_heading(y_e, pi_h, DT)
        u_d = guidance.calculate_speed(kappa, current_yaw, psi_d)

        # 3. Aktorik (Regler)
        yaw_rate = (psi_d - current_yaw + math.pi) % (2 * math.pi) - math.pi
        current_yaw += np.clip(yaw_rate * KP_YAW * DT, -MAX_YAW_RATE, MAX_YAW_RATE)

        current_speed = np.linalg.norm(vel)
        thrust_mag = np.clip(KP_THRUST * (u_d - current_speed), 0.0, MAX_THRUST_FORCE)
        F_thrust = thrust_mag * np.array([math.cos(current_yaw), math.sin(current_yaw)])

        # 4. Umwelteinflüsse auslesen
        v_wind = EnvironmentalFields.get_wind_velocity(pos, time_total)
        v_water = EnvironmentalFields.get_water_velocity(pos, time_total)

        # 5. Widerstandskräfte & Kinetik
        F_drag_wind = compute_drag_force(v_wind, vel, RHO_AIR, CW_AIR, A_AIR)
        F_drag_water = compute_drag_force(v_water, vel, RHO_WATER, CW_WATER, A_WATER)

        acc = (F_thrust + F_drag_wind + F_drag_water) / M_BOAT
        vel += acc * DT
        pos += vel * DT
        time_total += DT

        history_x.append(pos[0])
        history_y.append(pos[1])

        if zyklus % PRINT_INTERVAL == 0 or zyklus == 1:
            print(f"Z: {zyklus:03d} | Pos: ({pos[0]:+6.2f}, {pos[1]:+6.2f}) | "
                  f"y_e: {y_e:+.2f}m | pi_h: {math.degrees(pi_h):+4.0f}° | "
                  f"kappa: {kappa:+.3f} | V: {current_speed:4.2f}m/s")

    # =====================================================================
    # 6. VISUALISIERUNG
    # =====================================================================
    plt.figure(figsize=(12, 6))

    # Tore
    plt.plot([b.x for b in buoys if b.color == 'red'], [b.y for b in buoys if b.color == 'red'], 'ro', markersize=8,
             label='Backbord')
    plt.plot([b.x for b in buoys if b.color == 'green'], [b.y for b in buoys if b.color == 'green'], 'go', markersize=8,
             label='Steuerbord')

    # Geplanter Pfad vs. Gefahrene Trajektorie
    plt.plot(path_x, path_y, 'b--', linewidth=2, label='Geplanter B-Spline Referenzpfad')
    plt.plot(history_x, history_y, 'k-', linewidth=2.5, label='Gefahrene Bootstrajektorie')
    plt.plot(POS_START[0], POS_START[1], 'ks', markersize=8, label='Startposition')

    plt.title('Vollintegrierte Simulation: ILOS Guidance auf B-Spline mit Kinetik & Drift')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()