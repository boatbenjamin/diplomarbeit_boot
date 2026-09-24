"""
mc_common.py
=============================================================
Gemeinsame Datentypen und Hilfsfunktionen fuer die Echtzeitschicht.

WICHTIGE AENDERUNG ggue. der alten Version
------------------------------------------
Die Bootsparameter stehen NICHT mehr als feste Modulkonstanten hier
drin. Sie waren auf das reale kleine Boot getunt (D_Y = 0.15 m) und
haben in VRX (WAM-V, D_Y = 1.027 m) die komplette Regelung verstimmt.

Stattdessen: dataclass BootParameter, die pro Plattform aus
config/params.yaml gefuellt wird. Die Defaults hier sind die
VRX-WAM-V-Werte (aus vrx_urdf/wamv_description + SimpleHydrodynamics).
Fuer das reale Boot einfach die Werte in params.yaml ueberschreiben --
kein Codeeingriff noetig.
"""

import math
from dataclasses import dataclass


# =====================================================================
# BOOTS- UND PLATTFORMPARAMETER
# =====================================================================

@dataclass
class BootParameter:
    """Alle plattformabhaengigen Groessen an EINER Stelle."""

    # --- Geometrie / Aktorik ------------------------------------------
    d_y: float = 1.027135      # m, halber Motorabstand
                               #    VRX WAM-V: 1.027135 (aus wamv_aft_thrusters.xacro)
                               #    reales Boot: 0.15
    f_max: float = 500.0       # N, Software-Limit je Motor
                               #    VRX laesst 2353 N zu (max_thrust_cmd),
                               #    500 N ist eine bewusst gesetzte Reserve
    r_max: float = math.radians(45.0)   # rad/s, maximale Soll-Gierrate
    u_max: float = 2.2         # m/s, maximale Soll-Fahrt
                               #    Grenze: 2*f_max = x_u*u + x_uu*u^2
                               #    -> mit f_max=500 N sind ca. 2.26 m/s drin

    # --- Gierdynamik (1. Ordnung: Izz*r_dot + n_r*r = N) --------------
    izz: float = 700.0         # kg*m^2  (WAM-V base 446 + Motoren/Anbauten)
    n_r: float = 800.0         # N*m/(rad/s), lineare Gierdaempfung (SimpleHydrodynamics nR)

    # --- Laengsdynamik (m*u_dot + x_u*u + x_uu*|u|u = X) --------------
    masse: float = 250.0       # kg
    x_u: float = 100.0         # N/(m/s)   (SimpleHydrodynamics xU)
    x_uu: float = 150.0        # N/(m/s)^2 (SimpleHydrodynamics xUU)

    # --- Reglerbandbreiten --------------------------------------------
    omega_i: float = 1.5       # rad/s, Bandbreite innere Gierratenschleife
    omega_a_faktor: float = 5.0  # aeussere Schleife = omega_i / faktor
    dpsi_max: float = math.radians(25.0)  # rad/s, Rate des Kurs-Referenzmodells
                                          # bewusst < r_max, damit die Referenz
                                          # fuer das Boot ueberhaupt fahrbar ist

    # --- abgeleitete Groessen ------------------------------------------
    @property
    def n_max(self) -> float:
        """Maximal erreichbares Giermoment [N*m] (beide Motoren gegenlaeufig)."""
        return 2.0 * self.f_max * self.d_y

    @property
    def t_nomoto(self) -> float:
        """Ersatz-Zeitkonstante der Gierdynamik [s]."""
        return self.izz / self.n_r

    @property
    def k_nomoto(self) -> float:
        """Ersatz-Verstaerkung der Gierdynamik [(rad/s)/(N*m)]."""
        return 1.0 / self.n_r

    @property
    def omega_a(self) -> float:
        return self.omega_i / self.omega_a_faktor


# Fester Regeltakt (Kapitel 11.5: reproduzierbare Integratordynamik)
DT = 0.02   # s, 50 Hz


# =====================================================================
# DATENSTRUKTUREN
# =====================================================================

@dataclass
class BootState:
    """Vom Zustandsschaetzer gelieferter Zustand."""
    psi: float      # Kurswinkel [rad], ENU (0 = Ost, +90 deg = Nord)
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
    """Faltet einen Winkel auf (-pi, pi] zurueck."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rate_limit(target: float, previous: float, max_rate: float, dt: float) -> float:
    delta = clip(target - previous, -max_rate * dt, max_rate * dt)
    return previous + delta


def widerstand_vorsteuerung(u_d: float, p: BootParameter) -> float:
    """Stationaer noetige Laengskraft fuer die Sollfahrt u_d.

    FEHLER IN DER ALTEN VERSION: dort wurde
        X_ff = 0.5*rho*S*Ct*u^2
    mit S = 0.20 m^2 und Ct = 0.010 gerechnet -> bei 2 m/s ganze 4 N.
    Der tatsaechliche Widerstand des WAM-V bei 2 m/s ist
        100*2 + 150*4 = 800 N.
    Die Vorsteuerung war also um Faktor 200 zu klein und der
    I-Anteil musste alles alleine hochziehen (ca. 50 s bis Sollfahrt).

    Jetzt: dasselbe Widerstandsmodell, das auch die Simulation
    benutzt -- X_ff(u) = x_u*u + x_uu*|u|*u.
    """
    return p.x_u * u_d + p.x_uu * abs(u_d) * u_d
