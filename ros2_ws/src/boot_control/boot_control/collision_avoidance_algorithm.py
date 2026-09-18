"""
collision_avoidance_sbmpc.py
=============================================================
Szenariobasierter Model Predictive Controller (SB-MPC) zur
Kollisionsvermeidung fuer den Bojenkurs.

Sitzt im Regelkreis zwischen /guidance_ilos und der Regelungs-
kaskade (Kapitel 3/9/10/11 des Konzeptpapiers):

    guidance_ilos  --(psi_d, u_d)-->  COLLISION AVOIDANCE  --(psi_safe, u_safe)--> control cascade

Wissenschaftliche Grundlage (siehe Anhang D des Konzeptpapiers):
  - T. A. Johansen, T. Perez, A. Cristofaro (2016): "Ship Collision
    Avoidance and COLREGS Compliance Using Simulation-Based Control
    Behavior Selection With Predictive Hazard Assessment", IEEE T-ITS.
    -> Grundprinzip: endliche Menge Verhaltenskandidaten (Kursversatz,
       Tempofaktor), pro Kandidat wird die Eigen- und Hindernisbahn ueber
       einen endlichen Horizont vorausberechnet, ein Kostenfunktional
       bewertet Kollisionsrisiko + Manoeverkosten, der guenstigste
       Kandidat wird an die Kursfuehrung zurueckgegeben (rezeding horizon,
       "single control step" ueber den ganzen Horizont gehalten).
  - I. B. Hagen, D. K. M. Kufoalor, T. A. Johansen, E. F. Brekke (2022):
    zeigen, dass ein Wechselkosten-/Traegheitsterm (Bestrafung eines
    Kandidatenwechsels ggue. dem zuletzt gewaehlten Verhalten) fuer
    ruhiges, nicht-zappelndes Verhalten unverzichtbar ist.
  - Fossen (Handbook of Marine Craft Hydrodynamics): Nomoto-Modell
    erster Ordnung fuer die Gierbewegung wird fuer die Vorausberechnung
    der Eigenschiffsbahn verwendet (wie in Kapitel 5.2/5.4 des
    Konzeptpapiers beschrieben: "wird an drei Stellen konkret
    gebraucht: Reglerauslegung, Vorausberechnung im SB-MPC, Simulation").

Dieses Modul ist bewusst ROS-unabhaengig gehalten (reine Funktionen/
Klassen mit Python-Datentypen), damit es sich 1:1 in einen ROS-2-Knoten
(/collision_avoidance, 5-10 Hz) einbetten laesst: der Knoten liest
/cmd/course (psi_d, u_d) und /objects, ruft SBMPC.compute(...) auf und
veroeffentlicht /cmd/course_safe.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =====================================================================
# ZENTRALE TUNING-PARAMETER SB-MPC
# =====================================================================

# --- Verhaltenskandidaten (Kapitel 10.3, Anhang B) --------------------
KURSVERSAETZE_DEG = [-60, -45, -30, -15, 0, 15, 30, 45, 60]
TEMPOFAKTOREN = [1.0, 0.5, 0.0, -0.3]

# --- Vorausschau (Kapitel 10.3, Anhang B) ------------------------------
T_HORIZON = 15.0       # s, Vorausschauhorizont
DT_PRED = 0.5          # s, Schrittweite der Vorausberechnung

# --- Sicherheitsabstand (Kapitel 10.4) ---------------------------------
D0_SAFE = 3.0          # m, Basissicherheitsabstand bei ruhigem Wasser
SIGMA_FACTOR = 3.0     # Faktor auf die Positions-Unsicherheit der Objektschaetzung
C_YE = 0.0             # optionale Kopplung an die eigene Querablage y_e
OWN_RADIUS = 0.6       # m, halbe Bootslaenge + Puffer
OBST_RADIUS_DEFAULT = 0.5  # m, angenommener Hindernisradius falls unbekannt

# --- Kostenfunktion (Kapitel 10.3, Formel J(alpha,f)) ------------------
W_KOLL = 5000.0        # Gewicht Kollisionsrisiko
W_ALPHA = 1.0          # Gewicht Kursversatz |alpha|
W_TEMPO = 3.0          # Gewicht Tempofaktor (1-f)
W_WECHSEL = 4.0        # Gewicht Verhaltenswechsel ggue. letztem Zyklus
A_RISK = 4.0           # Steilheit der logistischen Risikofunktion rho(d_min)

# --- COLREGS-inspirierte Steuerbord-Praeferenz -------------------------
W_COLREGS = 0.4        # Zusatzgewicht, das Backbord- (negative) Kursversaetze bestraft

# --- Vereinfachtes Eigenschiffsmodell fuer die Vorausberechnung --------
K_NOMOTO = 1.1         # 1/s, Nomoto-Verstaerkung
T_NOMOTO = 1.2         # s, Nomoto-Zeitkonstante
R_MAX = math.radians(50)  # rad/s, maximale Gierrate
KP_KURS_PRED = 2.0     # 1/s, P-Verstaerkung des im Praediktor nachgebildeten Kursreglers
TAU_SPEED_PRED = 1.5   # s, Ersatzzeitkonstante fuer die Geschwindigkeitsangleichung


# =====================================================================
# DATENSTRUKTUREN
# =====================================================================

@dataclass
class OwnState:
    """Eigenzustand des Bootes, wie er aus /state/filtered kommt."""
    x: float
    y: float
    psi: float      # Kurswinkel [rad]
    u: float        # Fahrt durchs Wasser [m/s]
    r: float = 0.0  # aktuelle Gierrate [rad/s] (optional, fuer bessere Praediktion)


@dataclass
class Obstacle:
    """Ein Hindernis aus /objects (Kapitel 8.3: Kalman-Filter pro Objekt)."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    sigma_p: float = 1.0          # m, 1-sigma Positionsunsicherheit (aus Kovarianz)
    radius: float = OBST_RADIUS_DEFAULT


@dataclass
class Candidate:
    """Ein Verhaltenskandidat (alpha, f) mit Bewertung."""
    alpha_deg: float
    f: float
    j_koll: float = 0.0
    j_alpha: float = 0.0
    j_tempo: float = 0.0
    j_wechsel: float = 0.0
    j_colregs: float = 0.0
    j_total: float = math.inf
    d_min: float = math.inf
    own_track: Optional[np.ndarray] = None  # zum Plotten/Debuggen


@dataclass
class SBMPCResult:
    psi_safe: float
    u_safe: float
    chosen: Candidate
    candidates: List[Candidate] = field(default_factory=list)  # fuer Logging/Diagnose


# =====================================================================
# EIGENSCHIFF-PRAEDIKTOR (Nomoto-Modell, siehe Kapitel 5.2/11.2)
# =====================================================================

def predict_own_track(state: OwnState, psi_target: float, u_target: float,
                       t_horizon: float, dt: float,
                       v_drift: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    Berechnet die Eigenschiffsbahn fuer einen Verhaltenskandidaten voraus.

    Nachgebildet wird die reale Kaskade aus Kapitel 11: ein P-Kursregler
    liefert eine Soll-Gierrate, die ueber das Nomoto-Modell 1. Ordnung
    (T*r_dot + r = K*delta, hier direkt in geschlossener Regelschleife
    approximiert) auf die tatsaechliche Gierrate wirkt; die Geschwindigkeit
    naehert sich u_target mit einer Ersatzzeitkonstante an.

    v_drift: geschaetzter Wind-/Stroemungsversatz [m/s] aus der Zustands-
    schaetzung (Kapitel 7).
    """
    n_steps = max(1, int(round(t_horizon / dt)))
    track = np.zeros((n_steps + 1, 2))

    x, y, psi, u, r = state.x, state.y, state.psi, state.u, state.r
    track[0] = (x, y)

    for k in range(1, n_steps + 1):
        # --- Kursregelschleife (P-Regler + ratenbegrenzte Soll-Gierrate) ---
        e_psi = (psi_target - psi + math.pi) % (2 * math.pi) - math.pi
        r_d = max(-R_MAX, min(R_MAX, KP_KURS_PRED * e_psi))

        # --- Nomoto-Dynamik 1. Ordnung: r naehert sich r_d an ---
        r += (K_NOMOTO * r_d - r) / T_NOMOTO * dt
        psi += r * dt
        psi = (psi + math.pi) % (2 * math.pi) - math.pi

        # --- Geschwindigkeit naehert sich u_target an (Ersatz-PT1) ---
        u += (u_target - u) / TAU_SPEED_PRED * dt

        # --- Kinematik inkl. Wind-/Stroemungsversatz ---
        x += (u * math.cos(psi) + v_drift[0]) * dt
        y += (u * math.sin(psi) + v_drift[1]) * dt

        track[k] = (x, y)

    return track


def predict_obstacle_track(obst: Obstacle, t_horizon: float, dt: float) -> np.ndarray:
    """
    Hindernisbahn mit Modell konstanter Geschwindigkeit (CV-Modell),
    konsistent mit dem Objekt-Kalman-Filter aus Kapitel 8.3.
    """
    n_steps = max(1, int(round(t_horizon / dt)))
    t = np.arange(n_steps + 1) * dt
    xs = obst.x + obst.vx * t
    ys = obst.y + obst.vy * t
    return np.stack([xs, ys], axis=1)


# =====================================================================
# KOSTENFUNKTION (Kapitel 10.3)
# =====================================================================

def collision_risk(d_min: float, d_safe: float, a: float = A_RISK) -> float:
    """Weich modelliertes Kollisionsrisiko rho(d_min), Kapitel 10.3:
    rho(d_min) = 1 / (1 + exp(a*(d_min - d_safe)))
    -> naeherungsweise 1 bei d_min << d_safe, naeherungsweise 0 bei d_min >> d_safe,
    stetig differenzierbar.
    """
    exponent = max(-700.0, min(700.0, a * (d_min - d_safe)))  # exp()-Overflow vermeiden
    return 1.0 / (1.0 + math.exp(exponent))


def safety_distance(obst: Obstacle, y_e: float = 0.0) -> float:
    """d_sicher = d0 + 3*sigma_p + c*|y_e|, Kapitel 10.4."""
    return D0_SAFE + SIGMA_FACTOR * obst.sigma_p + C_YE * abs(y_e)


# =====================================================================
# SB-MPC HAUPTKLASSE
# =====================================================================

class ScenarioBasedMPC:
    """
    Zustandsbehaftete SB-MPC-Instanz. Haelt das zuletzt gewaehlte
    Verhalten fuer den Wechselkosten-Term (Kapitel 10.3) und kann so
    direkt als Singleton im /collision_avoidance-Knoten instanziert werden.
    """

    def __init__(self,
                 kursversaetze_deg: List[float] = None,
                 tempofaktoren: List[float] = None,
                 t_horizon: float = T_HORIZON,
                 dt_pred: float = DT_PRED):
        self.kursversaetze_deg = kursversaetze_deg if kursversaetze_deg is not None else KURSVERSAETZE_DEG
        self.tempofaktoren = tempofaktoren if tempofaktoren is not None else TEMPOFAKTOREN
        self.t_horizon = t_horizon
        self.dt_pred = dt_pred
        self._prev_behavior: Optional[Tuple[float, float]] = None  # (alpha_deg, f)

    def reset(self) -> None:
        """Bei Missionswechsel / Wiederaufnahme nach Handbetrieb zuruecksetzen."""
        self._prev_behavior = None

    def compute(self,
                own: OwnState,
                psi_d: float,
                u_d: float,
                obstacles: List[Obstacle],
                v_drift: Tuple[float, float] = (0.0, 0.0),
                y_e: float = 0.0) -> SBMPCResult:
        """
        Ein SB-MPC-Zyklus. Wird vom ROS-Knoten mit 5-10 Hz aufgerufen.

        Rueckgabe: SBMPCResult mit dem freigegebenen (psi_safe, u_safe)
        fuer /cmd/course_safe, plus allen bewerteten Kandidaten fuer das Log.
        """
        candidates: List[Candidate] = []

        # Kein Hindernis in Reichweite -> Kursfuehrung unveraendert durchreichen.
        if not obstacles:
            chosen = Candidate(alpha_deg=0.0, f=1.0, j_total=0.0)
            self._prev_behavior = (0.0, 1.0)
            return SBMPCResult(psi_safe=psi_d, u_safe=u_d, chosen=chosen, candidates=[])

        for alpha_deg in self.kursversaetze_deg:
            alpha = math.radians(alpha_deg)
            psi_target = psi_d + alpha

            for f in self.tempofaktoren:
                u_target = u_d * f

                own_track = predict_own_track(own, psi_target, u_target,
                                               self.t_horizon, self.dt_pred, v_drift)

                # --- Kollisionsrisiko ueber alle Hindernisse aufsummieren ---
                j_koll = 0.0
                worst_d_min = math.inf
                for obst in obstacles:
                    obst_track = predict_obstacle_track(obst, self.t_horizon, self.dt_pred)
                    n = min(len(own_track), len(obst_track))
                    dists = np.linalg.norm(own_track[:n] - obst_track[:n], axis=1)
                    dists -= (OWN_RADIUS + obst.radius)
                    d_min = float(np.min(dists))
                    d_safe = safety_distance(obst, y_e)
                    j_koll += collision_risk(d_min, d_safe)
                    worst_d_min = min(worst_d_min, d_min)

                # --- Manoeverkosten ---
                j_alpha = W_ALPHA * abs(alpha_deg) / 60.0   # normiert auf groessten Versatz
                j_tempo = W_TEMPO * (1.0 - f)
                j_wechsel = 0.0
                if self._prev_behavior is not None:
                    prev_alpha, prev_f = self._prev_behavior
                    if not (math.isclose(prev_alpha, alpha_deg) and math.isclose(prev_f, f)):
                        j_wechsel = W_WECHSEL
                j_colregs = W_COLREGS * max(0.0, -alpha_deg) / 60.0  # bestraft Backbord (negativ)

                j_total = (W_KOLL * j_koll) + j_alpha + j_tempo + j_wechsel + j_colregs

                candidates.append(Candidate(
                    alpha_deg=alpha_deg, f=f,
                    j_koll=j_koll, j_alpha=j_alpha, j_tempo=j_tempo,
                    j_wechsel=j_wechsel, j_colregs=j_colregs,
                    j_total=j_total, d_min=worst_d_min, own_track=own_track,
                ))

        chosen = min(candidates, key=lambda c: c.j_total)
        self._prev_behavior = (chosen.alpha_deg, chosen.f)

        psi_safe = psi_d + math.radians(chosen.alpha_deg)
        psi_safe = (psi_safe + math.pi) % (2 * math.pi) - math.pi
        u_safe = u_d * chosen.f

        return SBMPCResult(psi_safe=psi_safe, u_safe=u_safe, chosen=chosen, candidates=candidates)
