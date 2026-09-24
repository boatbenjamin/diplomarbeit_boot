"""
mc_kursregler.py
=============================================================
Aeussere Regelschleife: Kurs -> Soll-Gierrate.

    r_d = sat(Kp_psi * e_psi, +-r_max),   e_psi = wrap_pi(psi_ref - psi_hat)

Aenderungen ggue. der alten Version
-----------------------------------
1. Kp_psi und dpsi_max kommen jetzt aus BootParameter statt aus fest
   verdrahteten Modulkonstanten.
2. Das Referenzmodell lief mit dpsi_max = 60 deg/s und war damit
   SCHNELLER als das Boot ueberhaupt drehen kann -- es hatte also
   keinerlei glaettende Wirkung. Jetzt default 25 deg/s (< r_max).
3. Neu: Der Referenzkurs wird an den Istkurs gefesselt (max. 1 rad
   Vorsprung). Sonst eilt die Referenz bei einem 180-Grad-Sprung dem
   Boot davon, laeuft in die falsche Drehrichtung weiter und der
   Regler nimmt den laengeren Weg.
"""

import math
from dataclasses import dataclass
from typing import Optional

from boot_control.mc_common import BootParameter, clip, wrap_pi


@dataclass
class CourseController:
    p: BootParameter = None
    psi_ref: Optional[float] = None
    max_vorsprung: float = 1.0        # rad, max. Vorlauf der Referenz vor dem Ist

    def __post_init__(self):
        if self.p is None:
            self.p = BootParameter()

    @property
    def kp_psi(self) -> float:
        return self.p.omega_a

    def reset(self, psi_aktuell: float):
        self.psi_ref = wrap_pi(psi_aktuell)

    def step(self, psi_c: float, psi_hat: float, dt: float) -> float:
        if self.psi_ref is None:
            self.psi_ref = wrap_pi(psi_hat)

        # Referenzmodell: glatte, fahrbare Annaeherung an psi_c
        e_ref = wrap_pi(psi_c - self.psi_ref)
        step_val = clip(e_ref, -self.p.dpsi_max * dt, self.p.dpsi_max * dt)
        psi_ref_neu = wrap_pi(self.psi_ref + step_val)

        # Referenz darf dem Istkurs nicht davonlaufen (Anti-Reference-Windup)
        vorsprung = wrap_pi(psi_ref_neu - psi_hat)
        if abs(vorsprung) > self.max_vorsprung:
            psi_ref_neu = wrap_pi(psi_hat + math.copysign(self.max_vorsprung, vorsprung))
        self.psi_ref = psi_ref_neu

        e_psi = wrap_pi(self.psi_ref - psi_hat)
        return clip(self.kp_psi * e_psi, -self.p.r_max, self.p.r_max)
