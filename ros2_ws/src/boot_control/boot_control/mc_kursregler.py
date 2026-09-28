

import math
from dataclasses import dataclass
from typing import Optional

from boot_control.mc_common import BootParameter, clip, wrap_pi


@dataclass
class CourseController:
    p: BootParameter = None
    psi_ref: Optional[float] = None
    max_vorsprung: float = 1.0        # rad, so weit darf die Referenz dem
                                      #      Istkurs maximal vorauslaufen
    r_ff: float = 0.0                 # rad/s, aktuelle Vorsteuerung (zum Mitloggen)

    def __post_init__(self):
        if self.p is None:
            self.p = BootParameter()

    @property
    def kp_psi(self) -> float:
        return self.p.omega_a

    def reset(self, psi_aktuell: float):
        """Referenzkurs auf den Istkurs setzen, damit es beim Einschalten
        keinen Sprung gibt."""
        self.psi_ref = wrap_pi(psi_aktuell)
        self.r_ff = 0.0

    def step(self, psi_c: float, psi_hat: float, dt: float) -> float:
        if self.psi_ref is None:
            self.psi_ref = wrap_pi(psi_hat)

        # Referenzmodell: der Sollkurs wird nur mit dpsi_max nachgezogen,
        # also nicht sprunghaft
        e_ref = wrap_pi(psi_c - self.psi_ref)
        step_val = clip(e_ref, -self.p.dpsi_max * dt, self.p.dpsi_max * dt)
        psi_ref_neu = wrap_pi(self.psi_ref + step_val)

        # Referenz darf dem Istkurs nicht davonlaufen, sonst dreht das Boot
        # am Ende in die falsche Richtung (Anti-Reference-Windup)
        vorsprung = wrap_pi(psi_ref_neu - psi_hat)
        if abs(vorsprung) > self.max_vorsprung:
            psi_ref_neu = wrap_pi(psi_hat + math.copysign(self.max_vorsprung, vorsprung))
        # Vorsteuerung = Ableitung der Referenz, tiefpassgefiltert. Der Sollkurs
        # kommt nur mit 10-20 Hz herein, die rohe Ableitung waere also eine
        # Treppenfunktion mit groben Spruengen.
        if dt > 0.0:
            r_roh = wrap_pi(psi_ref_neu - self.psi_ref) / dt
            a = dt / (self.p.tau_r_ff + dt)
            self.r_ff += a * (r_roh - self.r_ff)
        self.psi_ref = psi_ref_neu

        e_psi = wrap_pi(self.psi_ref - psi_hat)
        return clip(self.r_ff + self.kp_psi * e_psi, -self.p.r_max, self.p.r_max)