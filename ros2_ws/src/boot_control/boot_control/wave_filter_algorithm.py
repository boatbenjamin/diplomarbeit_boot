from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, block_diag

def wrap_to_180(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0

@dataclass
class FilterOutput2D:
    # Niederfrequenter (echter) Zustand für den Regler / die Navigation
    x_lf: float          # Glatte X-Position [m] (East)
    y_lf: float          # Glatte Y-Position [m] (North)
    vx_lf: float         # Geschwindigkeit X [m/s]
    vy_lf: float         # Geschwindigkeit Y [m/s]
    kurs_lf: float       # Glatter Kurs [°]
    rate_lf: float       # Kurs-Änderungsrate [°/s]
    
    # Hochfrequente Wellenanteile (zur Diagnose / Motion-Control)
    welle_x: float       # Wellenversatz X [m]
    welle_y: float       # Wellenversatz Y [m]
    welle_psi: float     # Wellendrehung [°]
    
    outlier_rejected: bool

class WellenKalmanFilterV3_2D:
    """
    Kombinierter 2D-Position & Heading Wellen-Kalman-Filter.
    
    Zustand x (12 Dimensionen):
      [0..3]  X-Achse:    [x_lf, vx_lf, welle_x, welle_vx]
      [4..7]  Y-Achse:    [y_lf, vy_lf, welle_y, welle_vy]
      [8..11] Kurs-Achse: [psi_lf, r_lf, welle_psi, welle_r]
    """

    def __init__(
        self,
        dt_nominal: float = 0.05,
        f_wave: float = 0.5,         # Dominante Wellenfrequenz [Hz]
        damping: float = 0.15,        # Wellendämpfung
        q_pos: float = 1e-4,          # Prozessrauschen Position
        q_vel: float = 1e-2,          # Prozessrauschen Geschwindigkeit
        q_wave_pos: float = 5.0,      # Wellenintensität Position [m]
        q_wave_psi: float = 200.0,    # Wellenintensität Kurs [°]
        r_gnss_pos: float = 0.05,     # Messrauschen GPS X/Y [m]
        r_gnss_heading: float = 0.1,  # Messrauschen GNSS Doppelantenne [°]
        r_gyro: float = 0.05,         # Messrauschen IMU Gyro [°/s]
    ):
        self.dt_nominal = dt_nominal
        self.f_wave = f_wave
        self.damping = damping
        
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_wave_pos = q_wave_pos
        self.q_wave_psi = q_wave_psi
        
        self.r_gnss_pos = r_gnss_pos
        self.r_gnss_heading = r_gnss_heading
        self.r_gyro = r_gyro

        # Zustand x (12x1) und Kovarianz P (12x12)
        self.x = np.zeros(12)
        self.P = np.eye(12) * 10.0
        self.initialized = False

    def reset(self, init_x: float, init_y: float, init_heading: float):
        """Initialisiert den Zustand auf die ersten gültigen Sensordaten."""
        self.x = np.zeros(12)
        self.x[0] = init_x
        self.x[4] = init_y
        self.x[8] = init_heading
        self.P = np.eye(12) * 1.0
        self.initialized = True

    def _build_block_matrices(self, dt: float, q_wave: float):
        """Erstellt System- und Rauschmatrix für einen 4D-Unterblock (LF + Welle)."""
        omega_0 = 2.0 * np.pi * self.f_wave
        
        # Niederfrequenter Teil (Constant Velocity)
        F_lf = np.array([[1.0, dt], [0.0, 1.0]])
        Q_lf = self.q_vel * np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt]
        ]) + np.diag([self.q_pos, 0.0])

        # Wellen-Oszillator
        A_wave = np.array([[0.0, 1.0], [-(omega_0**2), -2.0 * self.damping * omega_0]])
        F_wave = expm(A_wave * dt)
        Q_wave = np.array([[0.0, 0.0], [0.0, q_wave * dt]])

        F_block = block_diag(F_lf, F_wave)
        Q_block = block_diag(Q_lf, Q_wave)
        return F_block, Q_block

    def _discretize(self, dt: float):
        """Fügt die Blöcke für X, Y und Kurs zusammen."""
        Fx, Qx = self._build_block_matrices(dt, self.q_wave_pos)
        Fy, Qy = self._build_block_matrices(dt, self.q_wave_pos)
        Fpsi, Qpsi = self._build_block_matrices(dt, self.q_wave_psi)

        F = block_diag(Fx, Fy, Fpsi)
        Q = block_diag(Qx, Qy, Qpsi)
        return F, Q

    def step(
        self,
        z_x: float | None = None,
        z_y: float | None = None,
        z_heading: float | None = None,
        z_gyro: float | None = None,
        dt: float | None = None
    ) -> FilterOutput2D:
        
        if not self.initialized:
            if z_x is not None and z_y is not None and z_heading is not None:
                self.reset(z_x, z_y, z_heading)
            else:
                # Warten auf vollständigen ersten Messwert
                return FilterOutput2D(0,0,0,0,0,0,0,0,0, False)

        dt = max(dt if dt is not None else self.dt_nominal, 1e-4)

        # 1. PRÄDIKTION
        F, Q = self._discretize(dt)
        x_pred = F @ self.x
        x_pred[8] = wrap_to_180(x_pred[8])  # Kurs wrappen
        P_pred = F @ self.P @ F.T + Q

        # 2. MESSMODELL DYNAMISCH AUFBAUEN
        H_rows, z_vals, R_vals, is_angle = [], [], [], []

        if z_x is not None:
            row = np.zeros(12); row[0] = 1.0; row[2] = 1.0  # x_lf + welle_x
            H_rows.append(row); z_vals.append(z_x); R_vals.append(self.r_gnss_pos); is_angle.append(False)

        if z_y is not None:
            row = np.zeros(12); row[4] = 1.0; row[6] = 1.0  # y_lf + welle_y
            H_rows.append(row); z_vals.append(z_y); R_vals.append(self.r_gnss_pos); is_angle.append(False)

        if z_heading is not None:
            row = np.zeros(12); row[8] = 1.0; row[10] = 1.0 # psi_lf + welle_psi
            H_rows.append(row); z_vals.append(z_heading); R_vals.append(self.r_gnss_heading); is_angle.append(True)

        if z_gyro is not None:
            row = np.zeros(12); row[9] = 1.0; row[11] = 1.0 # r_lf + welle_r
            H_rows.append(row); z_vals.append(z_gyro); R_vals.append(self.r_gyro); is_angle.append(False)

        # Falls keine Messungen vorliegen (z.B. GPS-Ausfall): Nur Prädiktion übernehmen
        if len(z_vals) == 0:
            self.x, self.P = x_pred, P_pred
            return self._make_output(outlier=False)

        H = np.array(H_rows)
        R = np.diag(R_vals)
        z = np.array(z_vals)

        # Innovation berechnen
        y = z - H @ x_pred
        for i, angle_flag in enumerate(is_angle):
            if angle_flag:
                y[i] = wrap_to_180(y[i])

        # 3. KORREKTUR (Joseph-Form Update)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.x[8] = wrap_to_180(self.x[8])

        I = np.eye(12)
        IKH = I - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ R @ K.T

        return self._make_output(outlier=False)

    def _make_output(self, outlier: bool) -> FilterOutput2D:
        return FilterOutput2D(
            x_lf=self.x[0],
            y_lf=self.x[4],
            vx_lf=self.x[1],
            vy_lf=self.x[5],
            kurs_lf=self.x[8],
            rate_lf=self.x[9],
            welle_x=self.x[2],
            welle_y=self.x[6],
            welle_psi=self.x[10],
            outlier_rejected=outlier
        )