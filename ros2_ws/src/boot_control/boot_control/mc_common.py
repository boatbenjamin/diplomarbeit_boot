"""
mc_common.py
=============================================================
Gemeinsame Datentypen und Hilfsfunktionen fuer die Echtzeitschicht
auf dem Mikrocontroller (Kapitel 11 des Konzeptpapiers).

Wird von allen mc_*.py-Modulen importiert. Bewusst ROS-unabhaengig
gehalten (reine Python-Datentypen), damit sich der Code 1:1 in die
spaetere micro-ROS-Firmware uebernehmen laesst.
"""

import math
from dataclasses import dataclass


# =====================================================================
# GEMEINSAME BOOTSPARAMETER (am Boot auszumessen, Kapitel 5.4/6/Anhang B)
# =====================================================================

K_NOMOTO = 1.1              # 1/s, Nomoto-Verstaerkung
T_NOMOTO = 1.2              # s, Nomoto-Zeitkonstante
D_Y = 0.15                  # m, halber Motorabstand
F_MAX = 35.0                # N, max. Schub je Motor (Software-Limit, 60-70% des physikalischen Maximums)
R_MAX = math.radians(50)    # rad/s, maximale Gierrate

RHO_WASSER = 1025.0         # kg/m^3
S_BENETZT = 0.20            # m^2, benetzte Flaeche (Kapitel 6.1)
C_T = 0.010                 # Gesamtwiderstandsbeiwert (ITTC + Zuschlaege)

DT = 0.02                   # s, feste Regelperiode (50 Hz), NICHT gemessen
                             # (Kapitel 11.5, Punkt 1: reproduzierbare Integratordynamik)


# =====================================================================
# DATENSTRUKTUREN
# =====================================================================

@dataclass
class BootState:
    """Vom Zustandsschaetzer gelieferter Zustand (reduziert auf das,
    was die Regelung auf dem Mikrocontroller braucht)."""
    psi: float      # Kurswinkel [rad]
    r: float        # Gierrate [rad/s]
    u: float        # Fahrt durchs Wasser [m/s]


@dataclass
class ThrustCommand:
    F_L: float
    F_R: float


@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""


# =====================================================================
# HILFSFUNKTIONEN
# =====================================================================

def wrap_pi(angle: float) -> float:
    """Faltet einen Winkel auf das Intervall (-pi, pi] zurueck.
    Kapitel 11.2: ein vergessener Umlauf hier ist 'ein Klassiker'."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rate_limit(target: float, previous: float, max_rate: float, dt: float) -> float:
    """Begrenzt die Aenderung von previous nach target auf max_rate pro dt."""
    delta = target - previous
    max_delta = max_rate * dt
    delta = clip(delta, -max_delta, max_delta)
    return previous + delta


def widerstand_vorsteuerung(u_d: float) -> float:
    """X_ff(u_d) = 0.5 * rho * S * Ct * u_d^2 (Kapitel 6.1/11.2),
    vorzeichenrichtig fuer Rueckwaertsfahrt."""
    return math.copysign(0.5 * RHO_WASSER * S_BENETZT * C_T * u_d ** 2, u_d)