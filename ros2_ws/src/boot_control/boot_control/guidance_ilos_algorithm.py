"""
guidance_ilos_algorithm.py
=============================================================
ILOS-Fuehrungsgesetz (Integral Line-of-Sight) + Pfadverfolgung.
Rein rechenlogisch, ROS-unabhaengig (-> offline testbar).

    chi_d = pi_h - atan2(y_e + sigma*y_int, Delta)      Soll-KURS ueber Grund
    y_int_dot = Delta*y_e / (Delta^2 + (y_e + sigma*y_int)^2)
    psi_d = chi_d - beta                                 Soll-HEADING

beta = atan2(v, u) ist der Schwimmwinkel (Boot rutscht in der Kurve
seitlich nach aussen). Ohne diese Kompensation zeigt der Bug zwar in
die richtige Richtung, das Boot faehrt aber schraeg daran vorbei.

AENDERUNGEN ggue. der alten Version
-----------------------------------
1. Pfadverfolgung mit FORTSCHRITT: Es wird nur in einem Fenster um die
   letzte Position auf dem Pfad gesucht. Die alte globale argmin-Suche
   konnte bei einer Lemniskate (Pfad beruehrt sich in der Mitte)
   zwischen den Aesten springen.
2. Projektion auf Segmente statt auf den naechsten Stuetzpunkt ->
   stetige Querablage, kein Rauschen durch die Punktabstaende.
3. Geschlossene Pfade (Anfang == Ende) werden erkannt und endlos
   abgefahren -- kein 4-Millionen-Punkte-Pfad mehr noetig.
4. Kruemmungs-VORSCHAU fuer die Kurvengeschwindigkeit: das Boot bremst
   VOR der Kurve, nicht erst drin.
5. Integrator-Grenze sinnvoll: die alte Kombination sigma=0.1,
   i_max=3 erlaubte maximal atan(0.3/Delta) ~ 3 Grad Korrektur.
"""

import math
from typing import Optional, Tuple

import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =====================================================================
# PFAD-TRACKER
# =====================================================================
class PfadTracker:
    """Haelt den Pfad und die aktuelle Position s [m] darauf."""

    def __init__(self, xs, ys, closed_tol: float = 1.0):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        # doppelte Punkte entfernen (Segmentlaenge 0 -> Division durch 0)
        keep = np.ones(len(xs), dtype=bool)
        keep[1:] = np.hypot(np.diff(xs), np.diff(ys)) > 1e-6
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 2:
            raise ValueError('Pfad braucht mindestens 2 Punkte')

        self.closed = len(xs) > 3 and math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < closed_tol
        if self.closed:
            # Schliessendes Segment explizit anhaengen
            if math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) > 1e-6:
                xs = np.append(xs, xs[0])
                ys = np.append(ys, ys[0])

        self.x, self.y = xs, ys
        self.dx, self.dy = np.diff(xs), np.diff(ys)
        self.seg_len = np.hypot(self.dx, self.dy)
        self.s0 = np.concatenate(([0.0], np.cumsum(self.seg_len)))   # s am Segmentanfang
        self.laenge = float(self.s0[-1])
        self.n_seg = len(self.seg_len)
        self.seg_psi = np.arctan2(self.dy, self.dx)

        # Kruemmung je Segment: Richtungsaenderung / Laenge, leicht geglaettet
        dpsi = np.array([wrap_pi(b - a) for a, b in zip(self.seg_psi[:-1], self.seg_psi[1:])])
        if self.closed:
            dpsi = np.append(dpsi, wrap_pi(self.seg_psi[0] - self.seg_psi[-1]))
            ds = 0.5 * (self.seg_len + np.roll(self.seg_len, -1))
        else:
            dpsi = np.append(dpsi, 0.0)
            ds = 0.5 * (self.seg_len + np.append(self.seg_len[1:], self.seg_len[-1]))
        kappa = dpsi / np.maximum(ds, 1e-6)
        k = np.ones(5) / 5.0
        if self.closed:
            kappa = np.convolve(np.concatenate((kappa[-2:], kappa, kappa[:2])), k, 'valid')
        else:
            kappa = np.convolve(np.pad(kappa, 2, mode='edge'), k, 'valid')
        self.seg_kappa = kappa

        self.idx: Optional[int] = None     # aktuelles Segment
        self.s: float = 0.0                # Bogenlaenge der Projektion
        self.runden: int = 0

    # ------------------------------------------------------------------
    def _seg_index(self, s: float) -> int:
        if self.closed:
            s = s % self.laenge
        s = min(max(s, 0.0), self.laenge)
        i = int(np.searchsorted(self.s0, s, side='right') - 1)
        return min(max(i, 0), self.n_seg - 1)

    def _kandidaten(self, zurueck: float, vor: float) -> np.ndarray:
        if self.idx is None:
            return np.arange(self.n_seg)
        s_lo, s_hi = self.s - zurueck, self.s + vor
        if self.closed:
            n = int(math.ceil((zurueck + vor) / max(np.mean(self.seg_len), 1e-3))) + 2
            n = min(n, self.n_seg)
            start = self._seg_index(s_lo)
            return (start + np.arange(n)) % self.n_seg
        i0, i1 = self._seg_index(s_lo), self._seg_index(s_hi)
        return np.arange(i0, i1 + 1)

    def _projiziere(self, px, py, segs):
        ax, ay = self.x[segs], self.y[segs]
        dx, dy, L = self.dx[segs], self.dy[segs], self.seg_len[segs]
        t = ((px - ax) * dx + (py - ay) * dy) / np.maximum(L * L, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        qx, qy = ax + t * dx, ay + t * dy
        d = np.hypot(px - qx, py - qy)
        j = int(np.argmin(d))
        return int(segs[j]), float(t[j]), float(d[j])

    # ------------------------------------------------------------------
    def update(self, pos, fenster_zurueck: float = 5.0, fenster_vor: float = 20.0,
               reacquire_dist: float = 20.0) -> Tuple[float, float, float]:
        """Liefert (y_e, pi_h, kappa).
        y_e > 0: Boot liegt LINKS vom Pfad (in Fahrtrichtung)."""
        px, py = float(pos[0]), float(pos[1])

        segs = self._kandidaten(fenster_zurueck, fenster_vor)
        i, t, d = self._projiziere(px, py, segs)
        if self.idx is not None and d > reacquire_dist:
            # weit weg vom Pfad (z.B. Pfad neu, Boot abgetrieben): global neu suchen
            i, t, d = self._projiziere(px, py, np.arange(self.n_seg))

        s_neu = float(self.s0[i] + t * self.seg_len[i])
        if self.closed and self.idx is not None:
            if s_neu - self.s < -0.5 * self.laenge:
                self.runden += 1
        elif self.closed and s_neu > self.laenge - 0.5:
            s_neu -= self.laenge      # Erstkontakt genau am Start/Ende: nicht als Runde zaehlen
        self.idx, self.s = i, s_neu

        # Pfadwinkel stetig zwischen den Segmentrichtungen interpolieren
        if t < 0.5:
            j = i - 1 if (i > 0 or self.closed) else i
            w = 0.5 + t
        else:
            j = (i + 1) % self.n_seg if (i < self.n_seg - 1 or self.closed) else i
            w = 1.5 - t
        pi_h = wrap_pi(self.seg_psi[j] + w * wrap_pi(self.seg_psi[i] - self.seg_psi[j]))

        qx = self.x[i] + t * self.dx[i]
        qy = self.y[i] + t * self.dy[i]
        y_e = -(px - qx) * math.sin(pi_h) + (py - qy) * math.cos(pi_h)
        return y_e, pi_h, float(self.seg_kappa[i])

    def kappa_vorschau(self, strecke: float) -> float:
        """Groesste |Kruemmung| auf den naechsten `strecke` Metern."""
        if self.idx is None:
            return 0.0
        n = int(math.ceil(strecke / max(np.mean(self.seg_len), 1e-3))) + 1
        if self.closed:
            segs = (self.idx + np.arange(n)) % self.n_seg
        else:
            segs = np.arange(self.idx, min(self.idx + n, self.n_seg))
        return float(np.max(np.abs(self.seg_kappa[segs])))

    @property
    def am_ende(self) -> bool:
        return (not self.closed) and self.idx is not None and \
            self.s >= self.laenge - 0.5


# =====================================================================
# ILOS
# =====================================================================
class ILOSGuidance:
    def __init__(self, delta: float = 4.0, sigma: float = 0.3, i_max: float = 10.0,
                 u_max: float = 3.0, u_min: float = 0.8, a_quer_max: float = 1.0,
                 k3: float = 1.0, beta_komp: bool = True, beta_max_deg: float = 30.0,
                 u_beta_min: float = 0.8, t_vorschau: float = 3.0, tau_beta: float = 3.0):
        self.delta = delta
        self.sigma = sigma
        self.i_max = i_max
        self.u_max = u_max
        self.u_min = u_min
        self.a_quer_max = a_quer_max
        self.k3 = k3
        self.beta_komp = beta_komp
        self.beta_max = math.radians(beta_max_deg)
        self.u_beta_min = u_beta_min
        self.t_vorschau = t_vorschau
        self.tau_beta = tau_beta
        self.y_int: float = 0.0
        self.beta: float = 0.0

    def reset_integral(self) -> None:
        self.y_int = 0.0

    def update_heading(self, y_e: float, pi_h: float, u: float, v: float, dt: float) -> float:
        """Soll-Heading psi_d [rad] aus Querablage, Pfadwinkel und Schwimmwinkel."""
        # Integrator nur nahe am Pfad -- beim Anfahren von weit weg wuerde er
        # sonst voll aufladen und danach ueberschwingen.
        if abs(y_e) < 2.0 * self.delta:
            y_int_dot = (self.delta * y_e) / (self.delta ** 2 + (y_e + self.sigma * self.y_int) ** 2)
            self.y_int = max(-self.i_max, min(self.i_max, self.y_int + y_int_dot * dt))

        chi_d = pi_h - math.atan2(y_e + self.sigma * self.y_int, self.delta)

        # Schwimmwinkel (Drift) -- nur bei Fahrt sinnvoll messbar.
        # STARK tiefpassgefiltert: Dreht der Rumpf schnell, bleibt der
        # Geschwindigkeitsvektor zunaechst stehen -> gemessenes beta aendert
        # sich sofort um -dpsi. Ungefiltert rueckgekoppelt ist das eine
        # Mitkopplung mit Verstaerkung ~1 (in der Simulation: Schlingern).
        if self.beta_komp and u > self.u_beta_min:
            b = max(-self.beta_max, min(self.beta_max, math.atan2(v, u)))
        else:
            b = 0.0
        a = dt / (self.tau_beta + dt)
        self.beta += a * (b - self.beta)
        return wrap_pi(chi_d - self.beta)

    def calculate_speed(self, kappa_max: float, psi: float, psi_d: float) -> float:
        """Soll-Fahrt aus Kurven-Vorschau und Kursfehler (nie unter u_min)."""
        u_curve = math.sqrt(self.a_quer_max / kappa_max) if kappa_max > 1e-3 else self.u_max
        e_psi = wrap_pi(psi_d - psi)
        f_heading = max(0.0, math.cos(min(abs(e_psi) * self.k3, math.pi / 2.0)))
        u_heading = self.u_min + (self.u_max - self.u_min) * f_heading
        return max(self.u_min, min(self.u_max, u_curve, u_heading))