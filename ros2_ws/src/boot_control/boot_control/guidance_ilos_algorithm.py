

import math
from typing import Optional, Tuple

import numpy as np


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi



class PfadTracker:


    def __init__(self, xs, ys, closed_tol: float = 1.0):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        # doppelte Punkte raus, ein Segment der Laenge 0 gibt sonst spaeter
        # eine Division durch 0
        keep = np.ones(len(xs), dtype=bool)
        keep[1:] = np.hypot(np.diff(xs), np.diff(ys)) > 1e-6
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 2:
            raise ValueError('Pfad braucht mindestens 2 Punkte')

        # Anfang == Ende: der Pfad ist eine geschlossene Runde und wird endlos
        # gefahren. Dann das letzte Segment zurueck zum Start anhaengen.
        self.closed = len(xs) > 3 and math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < closed_tol
        if self.closed:
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

        # Kruemmung je Segment = Richtungsaenderung pro Meter. Wird leicht
        # geglaettet (Mittelwert ueber 5 Segmente), sonst springt sie von
        # Segment zu Segment und das Boot bremst ruckartig.
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

        self.idx: Optional[int] = None     # auf welchem Segment wir gerade sind
        self.s: float = 0.0                # zurueckgelegte Strecke auf dem Pfad [m]
        self.runden: int = 0               # abgeschlossene Runden


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


    def update(self, pos, fenster_zurueck: float = 5.0, fenster_vor: float = 20.0,
               reacquire_dist: float = 20.0) -> Tuple[float, float, float]:
        """Position auf dem Pfad nachfuehren.

        Liefert (y_e, pi_h, kappa): Querablage [m], Pfadrichtung [rad] und
        Kruemmung [1/m] an der aktuellen Stelle.
        y_e > 0 heisst, das Boot liegt links vom Pfad (in Fahrtrichtung).
        """
        px, py = float(pos[0]), float(pos[1])

        segs = self._kandidaten(fenster_zurueck, fenster_vor)
        i, t, d = self._projiziere(px, py, segs)
        if self.idx is not None and d > reacquire_dist:
            # zu weit weg vom Pfad (neuer Pfad, Boot abgetrieben):
            # einmal ueber den ganzen Pfad suchen
            i, t, d = self._projiziere(px, py, np.arange(self.n_seg))

        s_neu = float(self.s0[i] + t * self.seg_len[i])
        if self.closed and self.idx is not None:
            if s_neu - self.s < -0.5 * self.laenge:
                self.runden += 1
        elif self.closed and s_neu > self.laenge - 0.5:
            # Erstkontakt genau am Start/Ende, das ist noch keine Runde
            s_neu -= self.laenge
        self.idx, self.s = i, s_neu

        # Pfadrichtung zwischen den beiden Nachbarsegmenten interpolieren,
        # damit sie beim Segmentwechsel nicht springt
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
        """Groesste Kruemmung auf den naechsten `strecke` Metern.

        Damit kann das Boot schon VOR der Kurve bremsen und nicht erst,
        wenn es mitten drin ist.
        """
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
        """Soll-Heading psi_d [rad] aus Querablage, Pfadrichtung und Schwimmwinkel."""
        # Der Integrator laeuft nur, wenn das Boot schon halbwegs am Pfad ist.
        # Faehrt es von weit weg an, wuerde er sich voll aufladen und das Boot
        # danach ueber den Pfad hinausschiessen.
        if abs(y_e) < 2.0 * self.delta:
            y_int_dot = (self.delta * y_e) / (self.delta ** 2 + (y_e + self.sigma * self.y_int) ** 2)
            self.y_int = max(-self.i_max, min(self.i_max, self.y_int + y_int_dot * dt))

        chi_d = pi_h - math.atan2(y_e + self.sigma * self.y_int, self.delta)

        # Schwimmwinkel. Nur bei Fahrt messbar, im Stand ist atan2(v, u) Unsinn.
        # Stark tiefpassgefiltert: dreht sich der Rumpf schnell, zeigt der
        # Geschwindigkeitsvektor noch in die alte Richtung, das gemessene beta
        # springt also sofort mit. Ungefiltert zurueckgefuehrt ergibt das eine
        # Mitkopplung mit Verstaerkung ~1 und das Boot schlingert.
        if self.beta_komp and u > self.u_beta_min:
            b = max(-self.beta_max, min(self.beta_max, math.atan2(v, u)))
        else:
            b = 0.0
        a = dt / (self.tau_beta + dt)
        self.beta += a * (b - self.beta)
        return wrap_pi(chi_d - self.beta)

    def calculate_speed(self, kappa_max: float, psi: float, psi_d: float) -> float:
        """Soll-Fahrt [m/s] aus Kurvenvorschau und Kursfehler.

        Zwei Gruende langsamer zu fahren: eine engere Kurve kommt (sonst
        rutscht das Boot nach aussen) oder der Kursfehler ist gerade gross
        (erst drehen, dann beschleunigen). Nie unter u_min, weil das Boot
        ohne Fahrt nicht mehr lenkbar ist.
        """
        u_curve = math.sqrt(self.a_quer_max / kappa_max) if kappa_max > 1e-3 else self.u_max
        e_psi = wrap_pi(psi_d - psi)
        f_heading = max(0.0, math.cos(min(abs(e_psi) * self.k3, math.pi / 2.0)))
        u_heading = self.u_min + (self.u_max - self.u_min) * f_heading
        return max(self.u_min, min(self.u_max, u_curve, u_heading))