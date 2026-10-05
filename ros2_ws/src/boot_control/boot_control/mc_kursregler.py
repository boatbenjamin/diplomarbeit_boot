"""
mc_kursregler.py
=============================================================
Aeussere Regelschleife: Kurs -> Soll-Gierrate.

    r_d = sat(r_ref + T_i * a_ref + Kp_psi * e_psi, +-r_max)
    e_psi = wrap_pi(psi_ref - psi_hat)

Referenzmodell 2. Ordnung (psi_ref, r_ref, a_ref)
------------------------------------------------
Die Referenz faehrt ein trapezfoermiges Drehratenprofil:
  - Drehrate begrenzt auf      dpsi_max   [rad/s]
  - Winkelbeschleunigung auf   ddpsi_max  [rad/s^2]
  - Bremskurve: r_ref <= sqrt(2 * ddpsi_max * |Restwinkel|)
  - Endanflug:  r_ref <= K_END * |Restwinkel|, r_ref folgt weich (K_R)
Die Referenz bremst also VOR dem Ziel ab und kommt mit r_ref = 0 an.

Der Endanflug ist noetig: Mit reiner Bang-Bang-Beschleunigung (+-ddpsi_max)
springt a_ref am Ziel jeden Takt hin und her; ueber den Vorhalt T_i*a_ref
waren das +-13 deg/s Rauschen auf der Soll-Gierrate (VRX-Test 23:33).

WARUM (Messungen VRX, 2026-10-03)
---------------------------------
Das alte Referenzmodell war ein reiner Ratenbegrenzer: Die Referenz lief
mit voller dpsi_max bis zum Zielkurs und sprang dort auf r = 0. Das Boot
kam mit ~35 deg/s am Ziel an und schoss 7-10 Grad ueber (30/90-Grad-Sprung),
unabhaengig von Kp_psi, omega_i und tau_r_ff.

Gierraten-Vorhalt T_i * a_ref
-----------------------------
Die innere Gierratenschleife folgt mit der Zeitkonstante
    T_i = Izz / (n_r + Kp_r) = T_nomoto / (1 + T_nomoto * omega_i)
(Izz=700, n_r=800, omega_i=3.5 -> T_i = 0.215 s). Der Vorhalt gleicht
diese Verzoegerung aus, sonst bremst das Boot trotz guter Referenz zu spaet.

Weiterhin enthalten
-------------------
- Referenzkurs an den Istkurs gefesselt (max_vorsprung), damit sie bei
  grossen Spruengen nicht davonlaeuft (Anti-Reference-Windup).
- tau_r_ff glaettet nur noch den Vorhalt-Anteil (a_ref springt).
"""

import math
from dataclasses import dataclass
from typing import Optional

from boot_control.mc_common import BootParameter, clip, wrap_pi

DDPSI_MAX_DEFAULT = math.radians(60.0)   # rad/s^2, Fallback falls nicht in BootParameter

# Endanflug des Referenzmodells (gegen Rattern am Ziel, siehe Modulkopf)
K_END = 3.0      # 1/s,  r_wunsch <= K_END * |Restwinkel|  (lineare Zone < ~7 Grad)
K_R = 12.0       # 1/s,  r_ref folgt r_wunsch mit Zeitkonstante 1/K_R (kritisch gedaempft: K_R = 4*K_END)


@dataclass
class CourseController:
    p: BootParameter = None
    psi_ref: Optional[float] = None
    r_ref: float = 0.0                # rad/s, Referenz-Drehrate
    a_ref: float = 0.0                # rad/s^2, Referenz-Winkelbeschleunigung
    max_vorsprung: float = 1.0        # rad, max. Vorlauf der Referenz vor dem Ist
    r_ff: float = 0.0                 # rad/s, gesamte Vorsteuerung (Diagnose)
    _vorhalt: float = 0.0             # gefilterter T_i * a_ref

    def __post_init__(self):
        if self.p is None:
            self.p = BootParameter()

    # ------------------------------------------------------------------
    @property
    def kp_psi(self) -> float:
        return self.p.omega_a

    @property
    def ddpsi_max(self) -> float:
        return getattr(self.p, 'ddpsi_max', None) or DDPSI_MAX_DEFAULT

    @property
    def t_i(self) -> float:
        """Zeitkonstante der geschlossenen Gierratenschleife."""
        t = getattr(self.p, 't_nomoto', None)
        w = getattr(self.p, 'omega_i', None)
        if t is None or w is None:
            return 0.2
        return t / (1.0 + t * w)

    def reset(self, psi_aktuell: float):
        self.psi_ref = wrap_pi(psi_aktuell)
        self.r_ref = 0.0
        self.a_ref = 0.0
        self.r_ff = 0.0
        self._vorhalt = 0.0

    # ------------------------------------------------------------------
    def step(self, psi_c: float, psi_hat: float, dt: float) -> float:
        if self.psi_ref is None:
            self.reset(psi_hat)
        if dt <= 0.0:
            return clip(self.r_ff, -self.p.r_max, self.p.r_max)

        a_max = self.ddpsi_max
        r_max_ref = self.p.dpsi_max

        # --- Referenzmodell 2. Ordnung ---
        e_ref = wrap_pi(psi_c - self.psi_ref)
        # Wunsch-Drehrate: begrenzt durch dpsi_max und durch die Bremskurve
        r_wunsch = math.copysign(
            min(r_max_ref, math.sqrt(2.0 * a_max * abs(e_ref)), K_END * abs(e_ref)), e_ref)
        dr = clip(K_R * (r_wunsch - self.r_ref) * dt, -a_max * dt, a_max * dt)
        self.r_ref += dr
        self.a_ref = dr / dt
        psi_ref_neu = wrap_pi(self.psi_ref + self.r_ref * dt)

        # Referenz darf dem Istkurs nicht davonlaufen (Anti-Reference-Windup)
        vorsprung = wrap_pi(psi_ref_neu - psi_hat)
        if abs(vorsprung) > self.max_vorsprung:
            psi_ref_neu = wrap_pi(psi_hat + math.copysign(self.max_vorsprung, vorsprung))
        self.psi_ref = psi_ref_neu

        # --- Vorsteuerung: Referenz-Drehrate + Vorhalt fuer die innere Schleife ---
        vorhalt_roh = self.t_i * self.a_ref
        tau = max(getattr(self.p, 'tau_r_ff', 0.0), 0.0)
        self._vorhalt += (dt / (tau + dt)) * (vorhalt_roh - self._vorhalt)
        self.r_ff = self.r_ref + self._vorhalt

        e_psi = wrap_pi(self.psi_ref - psi_hat)
        return clip(self.r_ff + self.kp_psi * e_psi, -self.p.r_max, self.p.r_max)