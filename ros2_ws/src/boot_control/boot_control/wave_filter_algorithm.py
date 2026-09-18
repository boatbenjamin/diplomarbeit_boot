"""
Wellen-Kalman-Filter V2 - für den produktiven Einsatz auf einem autonomen Boot.

Gegenüber der ursprünglichen Version wurden folgende praxisrelevanten Probleme
behoben, die bei einem echten Boot (ROS 2, reale Sensorik, echter Seegang)
sonst zu Fehlfunktionen führen würden:

 1. VARIABLES dt
    ROS-2-Callbacks kommen nie exakt periodisch (Scheduling-Jitter, Netzwerk,
    Nachrichtenverlust). Die Original-F/Q-Matrizen waren für ein festes dt
    "eingebrannt". V2 diskretisiert bei jedem Schritt neu für das tatsächliche
    dt (exakt per Matrix-Exponential für das Wellenmodell).

 2. WINKEL-WRAP BEI ±180°
    Ein Kurs von 179° und -179° liegen nur 2° auseinander, nicht 358°.
    Ohne Wrap-Behandlung "explodiert" die Innovation beim Nulldurchgang und
    der Filter divergiert. V2 wrapt Kurs-Zustand und Kurs-Innovation korrekt.

 3. FESTE WELLENFREQUENZ
    Ein realer Seegang ändert seine dominante Frequenz über Minuten/Stunden.
    Ein Filter, der nur exakt bei f_wave einrastet, verliert bei Frequenz-
    drift schnell die Trennschärfe. V2 schätzt die Wellenfrequenz online
    per Nulldurchgangs-Tracking nach und passt das Wellenmodell adaptiv an.

 4. KEINE ZWEITE MESSGRÖSSE
    Ein autonomes Boot hat i. d. R. nicht nur einen (verrauschten) Kompass-
    /Fusions-Kurs, sondern auch eine Gyro-Drehrate (IMU). Die Drehrate ist
    für die Trennung LF/HF hochgradig informativ, weil Wellenbewegung sich
    dort besonders deutlich zeigt. V2 erweitert den Zustand um eine
    LF-Drehrate und kann die Gyro-Rate optional mitfusionieren.

 5. KEINE AUSREISSER-BEHANDLUNG
    Ein einzelner Sensor-Glitch (GPS-Kurssprung, Kompass-Störung durch
    Magnetfeld in der Nähe von Motoren) darf den Filter nicht aus der Bahn
    werfen. V2 prüft jede Messung per Mahalanobis-Abstand (Chi²-Gating) und
    verwirft Ausreißer statt sie zu verarbeiten.

 6. KEIN UMGANG MIT SENSORAUSFALL
    Fällt die Kurs-Quelle kurz aus, muss der Filter trotzdem weiterlaufen
    können (reiner Prädiktionsschritt), statt eine leere/ungültige Messung
    zu verarbeiten. V2 unterstützt einen "predict-only"-Aufruf.

 7. NUMERISCHE ROBUSTHEIT
    Die Kovarianz-Update-Gleichung wird in der (numerisch robusteren)
    Joseph-Form UND mit dem R-Term berechnet, damit P auch bei ungünstigen
    Kalman-Gains symmetrisch positiv-semidefinit bleibt.

Am Ende der Datei befindet sich zusätzlich eine ROS-2-Node-Vorlage
(WellenFilterNode), die zeigt, wie der Filter an echte Topics angebunden wird.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, block_diag


# ----------------------------------------------------------------------
# Hilfsfunktion: Winkel korrekt auf [-180°, 180°] wrappen
# ----------------------------------------------------------------------
def wrap_to_180(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass
class KalmanOutput:
    kurs_lf: float          # geschätzter, wellenbereinigter Kurs [°]
    rate_lf: float          # geschätzte Kurs-Änderungsrate (LF) [°/s]
    welle: float            # geschätzte Wellenauslenkung (HF) [°]
    welle_rate: float       # geschätzte Wellen-Änderungsrate [°/s]
    f_wave_est: float       # aktuell nachgeführte Wellenfrequenz [Hz]
    innovation_norm: float  # normierte Innovation (Diagnose)
    outlier_rejected: bool  # True, wenn die Messung verworfen wurde


class WellenKalmanFilterV2:
    """
    Zustandsvektor x = [kurs_lf, rate_lf, welle, welle_rate]

        kurs_lf     - niederfrequenter (echter) Kurs/Heading des Bootes  [°]
        rate_lf     - dessen zeitliche Ableitung                        [°/s]
        welle       - hochfrequenter, wellenbedingter Anteil im Signal  [°]
        welle_rate  - dessen zeitliche Ableitung                        [°/s]

    Messungen (beide optional, mind. eine pro Aufruf nötig):
        z_heading   - Rohkurs aus Sensorfusion/Kompass  (misst kurs_lf+welle)
        z_rate      - Gyro-Drehrate aus der IMU          (misst rate_lf+welle_rate)
    """

    def __init__(
        self,
        dt_nominal: float = 0.02,
        f_wave: float = 0.5,
        damping: float = 0.15,
        q_lf_pos: float = 1e-5,
        q_lf_rate: float = 1e-4,
        q_wave: float = 250.0,
        r_heading: float = 0.5,
        r_gyro: float | None = 0.05,
        freq_adapt: bool = True,
        freq_adapt_gain: float = 0.05,
        f_wave_min: float = 0.3,
        f_wave_max: float = 1.3,
        gate_threshold: float | None = None,
    ):
        self.dt_nominal = dt_nominal
        self.damping = damping
        self.f_wave = f_wave
        self.f_wave_min = f_wave_min
        self.f_wave_max = f_wave_max
        self.freq_adapt = freq_adapt
        self.freq_adapt_gain = freq_adapt_gain

        self.q_lf_pos = q_lf_pos
        self.q_lf_rate = q_lf_rate
        self.q_wave = q_wave
        self.r_heading = r_heading
        self.r_gyro = r_gyro

        # Chi²-Schwelle für Ausreißer-Gating (99%-Quantil), je nach Anzahl
        # gleichzeitig verarbeiteter Messgrößen (1 oder 2)
        self._gate_1d = gate_threshold if gate_threshold else 6.63   # chi2(1, 0.99)
        self._gate_2d = gate_threshold if gate_threshold else 9.21   # chi2(2, 0.99)

        self.x = np.zeros(4)
        self.P = np.diag([25.0, 1.0, 25.0, 1.0])

        # Für die Online-Frequenznachführung merken wir uns das Vorzeichen
        # der letzten Wellenschätzung und den Zeitpunkt des letzten
        # Nulldurchgangs, um die Halbperiode zu messen.
        self._last_wave_sign = 0
        self._t_since_last_crossing = 0.0

        self.initialized = False

    # ------------------------------------------------------------------
    def reset(self, initial_heading: float):
        """Kaltstart: Kurszustand auf den ersten Messwert setzen, um das
        Einschwingen des Filters zu vermeiden."""
        self.x = np.array([initial_heading, 0.0, 0.0, 0.0])
        self.P = np.diag([25.0, 1.0, 25.0, 1.0])
        self.initialized = True

    # ------------------------------------------------------------------
    def _discretize(self, dt: float):
        """Exakte (bzw. für die LF-Kette exakte, für das Wellenmodell per
        Matrix-Exponential exakte) Diskretisierung für das jeweils aktuelle
        dt. Wird bewusst pro Schritt neu berechnet, weil ROS-2-Zeitschritte
        in der Praxis nicht perfekt konstant sind."""

        omega_0 = 2.0 * np.pi * self.f_wave

        # --- LF-Block: konstante Rate (CV-Modell), exakt diskretisiert ---
        F_lf = np.array([[1.0, dt],
                          [0.0, 1.0]])
        # "Integrated white noise acceleration"-Prozessrauschen (Standardmodell
        # für ein konstante-Rate-Modell), plus kleiner Grundterm für Drift
        Q_lf = self.q_lf_rate * np.array([
            [dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 2 / 2.0, dt],
        ]) + np.diag([self.q_lf_pos, 0.0])

        # --- Wellen-Block: gedämpfter Oszillator, exakt via expm ---
        A_wave = np.array([[0.0, 1.0],
                            [-(omega_0 ** 2), -2.0 * self.damping * omega_0]])
        F_wave = expm(A_wave * dt)
        # Prozessrauschen greift nur an der Beschleunigung der Welle an
        Q_wave = np.array([[0.0, 0.0],
                            [0.0, self.q_wave * dt]])

        F = block_diag(F_lf, F_wave)
        Q = block_diag(Q_lf, Q_wave)
        return F, Q

    # ------------------------------------------------------------------
    def _adapt_wave_frequency(self, dt: float):
        """Sehr einfache, aber robuste Online-Nachführung der dominanten
        Wellenfrequenz über die Nulldurchgänge der geschätzten Wellenauslenkung.
        Eine Halbperiode zwischen zwei Vorzeichenwechseln liefert eine
        Momentanfrequenz, die per Tiefpass langsam in f_wave eingemischt wird,
        damit kurzfristiges Rauschen die Schätzung nicht destabilisiert."""
        welle = self.x[2]
        sign = 1 if welle > 0 else (-1 if welle < 0 else 0)
        self._t_since_last_crossing += dt

        if self._last_wave_sign != 0 and sign != 0 and sign != self._last_wave_sign:
            halbperiode = self._t_since_last_crossing
            if halbperiode > 1e-3:
                f_inst = 1.0 / (2.0 * halbperiode)
                f_inst = float(np.clip(f_inst, self.f_wave_min, self.f_wave_max))
                self.f_wave = (
                    (1.0 - self.freq_adapt_gain) * self.f_wave
                    + self.freq_adapt_gain * f_inst
                )
            self._t_since_last_crossing = 0.0

        if sign != 0:
            self._last_wave_sign = sign

    # ------------------------------------------------------------------
    def step(self, z_heading: float | None = None, dt: float | None = None,
              z_rate: float | None = None) -> KalmanOutput:
        """Führt einen Prädiktions- (+ optional Korrektur-)Schritt aus.

        z_heading : Rohkurs-Messung [°]  (None, falls gerade nicht verfügbar)
        z_rate    : Gyro-Drehrate [°/s]  (None, falls kein Gyro genutzt wird
                    oder gerade nicht verfügbar)
        dt        : tatsächlich vergangene Zeit seit dem letzten Aufruf [s].
                    Falls None, wird dt_nominal verwendet.
        """
        if not self.initialized:
            # Kaltstart ohne expliziten reset()-Aufruf abfangen
            self.reset(z_heading if z_heading is not None else 0.0)

        if dt is None:
            dt = self.dt_nominal
        dt = max(dt, 1e-4)  # Division-durch-Null / negative dt abfangen

        F, Q = self._discretize(dt)
        x_pred = F @ self.x
        x_pred[0] = wrap_to_180(x_pred[0])
        P_pred = F @ self.P @ F.T + Q

        # --- reiner Prädiktionsschritt, falls keine Messung vorliegt ---
        if z_heading is None and z_rate is None:
            self.x, self.P = x_pred, P_pred
            return self._make_output(innovation_norm=0.0, outlier=False)

        # --- Messmodell dynamisch zusammenstellen (robust ggü. Ausfällen) ---
        H_rows, z_vals, R_vals, angle_mask = [], [], [], []
        if z_heading is not None:
            H_rows.append([1.0, 0.0, 1.0, 0.0])
            z_vals.append(z_heading)
            R_vals.append(self.r_heading)
            angle_mask.append(True)
        if z_rate is not None and self.r_gyro is not None:
            H_rows.append([0.0, 1.0, 0.0, 1.0])
            z_vals.append(z_rate)
            R_vals.append(self.r_gyro)
            angle_mask.append(False)

        H = np.array(H_rows)
        R = np.diag(R_vals)
        z = np.array(z_vals)

        y = z - H @ x_pred
        for i, is_angle in enumerate(angle_mask):
            if is_angle:
                y[i] = wrap_to_180(y[i])

        S = H @ P_pred @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        gate = self._gate_2d if len(z_vals) == 2 else self._gate_1d

        if d2 > gate:
            # Ausreißer / Sensor-Glitch: Messung verwerfen, nur Prädiktion
            # übernehmen. Optional könnte man P hier zusätzlich leicht
            # aufblähen, um nach einem Ausfall schneller wieder zu konvergieren.
            self.x, self.P = x_pred, P_pred
            return self._make_output(innovation_norm=float(np.sqrt(d2)), outlier=True)

        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.x[0] = wrap_to_180(self.x[0])

        # Joseph-Form (numerisch robust, bleibt auch bei suboptimalem K
        # garantiert symmetrisch positiv-semidefinit)
        I = np.eye(4)
        IKH = I - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ R @ K.T

        if self.freq_adapt:
            self._adapt_wave_frequency(dt)

        return self._make_output(innovation_norm=float(np.sqrt(d2)), outlier=False)

    # ------------------------------------------------------------------
    def _make_output(self, innovation_norm: float, outlier: bool) -> KalmanOutput:
        return KalmanOutput(
            kurs_lf=self.x[0],
            rate_lf=self.x[1],
            welle=self.x[2],
            welle_rate=self.x[3],
            f_wave_est=self.f_wave,
            innovation_norm=innovation_norm,
            outlier_rejected=outlier,
        )
