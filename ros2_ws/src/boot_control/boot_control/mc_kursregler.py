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
  - Bremskurve: r_korr <= sqrt(2 * ddpsi_max * |Restwinkel|)
  - Endanflug:  r_korr <= K_END * |Restwinkel|, r_ref folgt weich (K_R)
Die Referenz bremst also VOR dem Ziel ab und kommt mit r_korr = 0 an.

Der Endanflug ist noetig: Mit reiner Bang-Bang-Beschleunigung (+-ddpsi_max)
springt a_ref am Ziel jeden Takt hin und her; ueber den Vorhalt T_i*a_ref
waren das +-13 deg/s Rauschen auf der Soll-Gierrate (VRX-Test 23:33).

PFAD-VORSTEUERUNG r_pfad (NEU 2026-10-05)
-----------------------------------------
Das Referenzmodell oben war fuer einen FESTEN Zielkurs ausgelegt: Es
bremst auf r = 0 ab. Auf einem gekruemmten Pfad dreht sich der Zielkurs
aber staendig weiter. Der Endanflug r_korr = K_END*|e_ref| erzwingt dann
einen bleibenden Nachlauf:

    stationaer gilt  r_ref = K_END * e_ref
    -> fuer eine Dauerdrehrate r braucht die Referenz e_ref = r / K_END

Zahlenbeispiel Lemniskate (R = 11.1 m, u = 3 m/s -> r = 0.27 rad/s,
K_END = 3.0):  e_ref = 0.09 rad = 5.2 Grad bleibender Kursnachlauf.
Bei einem Lookahead von 8 m sind das rund 0.7 m Querablage nach
kurvenaussen -- ein systematischer Anteil der in VRX gemessenen
1.3-1.7 m, und zwar einer, den der ILOS-Integrator muehsam wieder
herausregeln muss.

Behebung: Die Fuehrung kennt die Pfadkruemmung exakt und liefert die
Solldrehrate des Pfades mit,

    r_pfad = kappa * u      (verifizierte Beziehung, vgl. Johansen et al.)

Das Referenzmodell regelt dann nur noch die ABWEICHUNG aus:

    r_wunsch = r_pfad + r_korr(e_ref)

Auf einem Kreis ist e_ref damit stationaer null statt 5.2 Grad. Der
Bremszweig bleibt unveraendert wirksam, sobald ein echter Kurssprung
dazukommt. r_pfad = 0 -> exakt das alte Verhalten.

Gierraten-Vorhalt T_i * a_ref
-----------------------------
Die innere Gierratenschleife folgt mit der Zeitkonstante
    T_i = Izz / (n_r + Kp_r) = T_nomoto / (1 + T_nomoto * omega_i)
(Izz=700, n_r=800, omega_i=3.0 -> T_i = 0.23 s). Der Vorhalt gleicht
diese Verzoegerung aus, sonst bremst das Boot trotz guter Referenz zu spaet.

Weiterhin enthalten
-------------------
- Referenzkurs an den Istkurs gefesselt (max_vorsprung), damit sie bei
  grossen Spruengen nicht davonlaeuft (Anti-Reference-Windup).
- tau_r_ff glaettet nur noch den Vorhalt-Anteil (a_ref springt).
- Optionale Totzone auf e_psi (Konzept 11.6), Standard aus.
"""

import math
from dataclasses import dataclass
from typing import Optional

from boot_control.mc_common import BootParameter, clip, wrap_pi

DDPSI_MAX_DEFAULT = math.radians(60.0)   # rad/s^2, Fallback falls nicht in BootParameter

# Endanflug des Referenzmodells (gegen Rattern am Ziel, siehe Modulkopf)
K_END = 3.0      # 1/s,  r_korr <= K_END * |Restwinkel|  (lineare Zone < ~7 Grad)
K_R = 12.0       # 1/s,  r_ref folgt r_wunsch mit Zeitkonstante 1/K_R (kritisch gedaempft: K_R = 4*K_END)


def _totzone(e: float, breite: float) -> float:
    """Totzone OHNE Sprung: innerhalb der Breite null, ausserhalb um die
    Breite verschoben. Ein einfaches 'if |e|<b: e=0' waere an der Grenze
    unstetig und damit selbst eine Grenzzyklus-Quelle."""
    if breite <= 0.0:
        return e
    if e > breite:
        return e - breite
    if e < -breite:
        return e + breite
    return 0.0


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
    def step(self, psi_c: float, psi_hat: float, dt: float,
             r_pfad: float = 0.0) -> float:
        """psi_c: Sollkurs [rad].  psi_hat: Istkurs [rad].
        r_pfad: Solldrehrate des Pfades [rad/s] = kappa*u, aus der Fuehrung.
                0 -> Verhalten wie vor 2026-10-05."""
        if self.psi_ref is None:
            self.reset(psi_hat)
        if dt <= 0.0:
            return clip(self.r_ff, -self.p.r_max, self.p.r_max)

        a_max = self.ddpsi_max
        r_max_ref = self.p.dpsi_max

        # --- Referenzmodell 2. Ordnung mit Pfad-Vorsteuerung ---
        e_ref = wrap_pi(psi_c - self.psi_ref)
        # Korrekturanteil: bremst VOR dem Ziel ab und faellt am Ziel auf null
        r_korr = math.copysign(
            min(r_max_ref, math.sqrt(2.0 * a_max * abs(e_ref)), K_END * abs(e_ref)), e_ref)
        # Grundrate: die Drehrate, die der Pfad ohnehin verlangt
        r_pfad_eff = clip(getattr(self.p, 'k_r_ff', 1.0) * r_pfad, -r_max_ref, r_max_ref)
        r_wunsch = clip(r_pfad_eff + r_korr, -r_max_ref, r_max_ref)

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

        e_psi = _totzone(wrap_pi(self.psi_ref - psi_hat),
                         getattr(self.p, 'e_psi_totzone', 0.0))
        return clip(self.r_ff + self.kp_psi * e_psi, -self.p.r_max, self.p.r_max)
