"""
mc_kursregler.py
=============================================================
Aeussere Regelschleife: Kurs -> Soll-Gierrate (Kapitel 11.1/11.2
des Konzeptpapiers).

    r_d = sat(Kp_psi * e_psi, +-r_max)
    e_psi = wrap_pi(psi_c - psi_hat)

Reiner P-Regler, ergaenzt um ein Referenzmodell: Aus einem
Kurssprung wird eine glatte, mit den Fahrzeuggrenzen vertraegliche
Sollkurve gebildet, damit aus einem Zielwechsel kein
Stellgroessensprung wird.

Laeuft mit 10-20 Hz, mindestiens Faktor 5 langsamer als die innere
Gierratenschleife (mc_gierratenregler.py), die mit 50 Hz laeuft.
"""

import math
from dataclasses import dataclass
from typing import Optional

from boot_control.mc_common import R_MAX, clip, wrap_pi

# --- Startwerte (Anhang B) ---------------------------------------------
OMEGA_I = 3.0                  # rad/s, Bandbreite innere Schleife (siehe mc_gierratenregler.py)
OMEGA_A = OMEGA_I / 5.0        # rad/s, mind. Faktor 5 langsamer
KP_PSI = OMEGA_A               # reiner P-Regler auf den Kursfehler
DPSI_MAX = math.radians(60.0)  # rad/s, max. Aenderungsrate des Kurs-Sollwerts (Referenzmodell)


@dataclass
class CourseController:
    kp_psi: float = KP_PSI
    r_max: float = R_MAX
    dpsi_max: float = DPSI_MAX
    psi_ref: Optional[float] = None  # geglaetteter Sollkurs (Referenzmodell)

    def reset(self, psi_aktuell: float):
        """Stossfreie Initialisierung, z.B. beim Uebergang nach 'aktiv'."""
        self.psi_ref = psi_aktuell

    def step(self, psi_c: float, psi_hat: float, dt: float) -> float:
        if self.psi_ref is None:
            self.psi_ref = psi_hat

        # Referenzmodell: geglaetteter Kurs naehert sich psi_c an,
        # begrenzt auf dpsi_max. Winkeldifferenz muss gewickelt werden.
        e_ref = wrap_pi(psi_c - self.psi_ref)
        max_step = self.dpsi_max * dt
        step_val = clip(e_ref, -max_step, max_step)
        self.psi_ref = wrap_pi(self.psi_ref + step_val)

        e_psi = wrap_pi(self.psi_ref - psi_hat)
        r_d = clip(self.kp_psi * e_psi, -self.r_max, self.r_max)
        return r_d
