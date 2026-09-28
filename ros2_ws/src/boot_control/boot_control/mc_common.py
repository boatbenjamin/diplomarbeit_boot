"""
mc_common.py
=============================================================
Gemeinsame Datentypen und Hilfsfunktionen fuer die Regelung.

Alle Werte, die vom Boot selbst abhaengen (Groesse, Masse, Motoren,
Daempfung), stehen in der dataclass BootParameter und nicht als feste
Konstanten im Code. Gefuellt wird sie aus config/params.yaml.

Die Defaults hier sind die Werte des WAM-V aus der VRX-Simulation
(aus vrx_urdf/wamv_description und SimpleHydrodynamics). Fuer das
reale Boot muessen nur die Werte in params.yaml ueberschrieben werden,
am Code aendert sich nichts. Vorher standen die Werte des kleinen
realen Boots fest im Code (d_y = 0.15 m) -- in VRX mit d_y = 1.027 m
war damit die ganze Regelung verstimmt.
"""

import math
from dataclasses import dataclass




@dataclass
class BootParameter:
    """Alle plattformabhaengigen Groessen an EINER Stelle."""

    # --- Geometrie / Aktorik ------------------------------------------
    d_y: float = 1.027135      # m, halber Abstand der beiden Motoren
                               #    VRX WAM-V: 1.027135 (wamv_aft_thrusters.xacro)
                               #    reales Boot: 0.15
    f_max: float = 1500.0      # N, Software-Limit je Motor. VRX selbst laesst
                               #    2353 N zu, der Rest ist Reserve.
    r_max: float = math.radians(45.0)   # rad/s, maximale Soll-Gierrate
    u_max: float = 2.2         # m/s, maximale Soll-Fahrt. Physikalische Grenze
                               #    aus 2*f_max = x_u*u + x_uu*u^2, der gueltige
                               #    Wert kommt aber aus params.yaml.

    # --- Gierdynamik (1. Ordnung: Izz*r_dot + n_r*r = N) --------------
    izz: float = 700.0         # kg*m^2, Traegheit um die Hochachse
                               #    (WAM-V Rumpf 446 + Motoren und Anbauten)
    n_r: float = 800.0         # N*m/(rad/s), lineare Gierdaempfung (VRX nR)
    n_rr: float = 0.0          # N*m/(rad/s)^2, quadratische Gierdaempfung
                               #    VRX hat hier nAbsR = 800, wird ueber
                               #    params.yaml gesetzt.

    # --- Laengsdynamik (m*u_dot + x_u*u + x_uu*|u|u = X) --------------
    masse: float = 250.0       # kg
    x_u: float = 100.0         # N/(m/s)   (SimpleHydrodynamics xU)
    x_uu: float = 150.0        # N/(m/s)^2 (SimpleHydrodynamics xUU)

    # --- Reglerbandbreiten --------------------------------------------
    omega_i: float = 1.5       # rad/s, Bandbreite der inneren Gierratenschleife
    omega_a_faktor: float = 5.0  # aeussere Kursschleife = omega_i / faktor
                                 # (aussen langsamer als innen, sonst arbeiten
                                 #  die beiden Schleifen gegeneinander)
    dpsi_max: float = math.radians(25.0)  # rad/s, wie schnell der Sollkurs im
                                          # Referenzmodell nachgezogen wird.
                                          # Muss kleiner als r_max sein, sonst
                                          # verlangt die Referenz mehr, als das
                                          # Boot drehen kann.
    tau_r_ff: float = 0.3      # s, Tiefpass auf die Gierraten-Vorsteuerung


    @property
    def n_max(self) -> float:
        """Groesstes moegliches Giermoment [N*m]: beide Motoren gegenlaeufig."""
        return 2.0 * self.f_max * self.d_y

    @property
    def t_nomoto(self) -> float:
        """Zeitkonstante des Nomoto-Ersatzmodells [s]: T = Izz/n_r."""
        return self.izz / self.n_r

    @property
    def k_nomoto(self) -> float:
        """Verstaerkung des Nomoto-Ersatzmodells [(rad/s)/(N*m)]: K = 1/n_r."""
        return 1.0 / self.n_r

    @property
    def omega_a(self) -> float:
        return self.omega_i / self.omega_a_faktor


# Fester Regeltakt (siehe Kapitel 11.5). Feste Schrittweite, damit sich die
# Integratoren immer gleich verhalten und Messungen vergleichbar bleiben.
DT = 0.02   # s, entspricht 50 Hz




@dataclass
class BootState:
    """Der Zustand, den der Zustandsschaetzer liefert."""
    psi: float      # Kurswinkel [rad] im ENU-Frame (0 = Ost, +90 deg = Nord)
    r: float        # Gierrate [rad/s], also wie schnell sich das Boot dreht
    u: float        # Fahrt in Laengsrichtung [m/s]


@dataclass
class ThrustCommand:
    F_L: float
    F_R: float


@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""



def wrap_pi(angle: float) -> float:
    """Rechnet einen Winkel auf den Bereich (-pi, pi] zurueck.

    Damit z.B. 350 Grad als -10 Grad behandelt wird und der Regler den
    kurzen Weg nimmt statt einmal ganz herum zu drehen.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rate_limit(target: float, previous: float, max_rate: float, dt: float) -> float:
    delta = clip(target - previous, -max_rate * dt, max_rate * dt)
    return previous + delta


def gier_vorsteuerung(r_d: float, p: BootParameter) -> float:
    """Vorsteuerung: welches Giermoment braucht man dauerhaft fuer r_d.

    Gerechnet mit linearer und quadratischer Gierdaempfung, also gleich wie
    im Hydrodynamikmodell von VRX.
    """
    return p.n_r * r_d + p.n_rr * abs(r_d) * r_d


def widerstand_vorsteuerung(u_d: float, p: BootParameter) -> float:
    """Vorsteuerung: welche Laengskraft braucht man dauerhaft fuer u_d.

        X_ff(u) = x_u*u + x_uu*|u|*u

    Das ist dasselbe Widerstandsmodell, das auch die Simulation verwendet.
    Beispiel WAM-V bei 2 m/s: 100*2 + 150*4 = 800 N. Rechnet man hier zu
    klein, muss der I-Anteil die fehlende Kraft alleine aufbauen und das
    Boot braucht fast eine Minute bis zur Sollfahrt.
    """
    return p.x_u * u_d + p.x_uu * abs(u_d) * u_d