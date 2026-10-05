"""
mc_antiwindup_pid.py
=============================================================
Generischer PID mit Vorsteuerung und Rueckrechnungs-Anti-Windup:

    I_dot = Ki*e + (1/Tt)*(u_sat - u_unsat)

Aenderungen ggue. der alten Version
-----------------------------------
1. set_limits(): die Stellgrenzen koennen zur Laufzeit gesetzt werden.
   Das braucht die Gierratenschleife, weil das verfuegbare Giermoment
   davon abhaengt, wieviel Schub gerade fuer den Vortrieb draufgeht.
2. back_calculate(): erlaubt es, dem Integrator den TATSAECHLICH
   gestellten Wert zurueckzumelden (z.B. nach der Schubaufteilung).
   Ohne das laeuft der Integrator weiter hoch, obwohl die Aktorik
   laengst am Anschlag ist -- genau das hat in der alten Version den
   Ueberschwinger von 45 Grad verursacht.
3. D-Anteil mit Tiefpass (N_filt), damit Kd nicht direkt auf das
   Wellenrauschen wirkt.
4. NEU (2026-10-05): Ratenbegrenzung der Stellgroesse (du_max).
   Kapitel 11.6 des Konzepts fordert sie ("Ratenbegrenzung der
   Motorbefehle"), implementiert war sie nie. Ohne sie schlaegt jeder
   Messausreisser mit voller Reglerverstaerkung auf die Aktorik durch:
   gemessen wurden Schubspruenge bis 1372 N von einem Takt zum naechsten,
   bei einem Stellbereich von 2600 N.
   Die Begrenzung wirkt NACH der Saettigung, und der Integrator bekommt
   den tatsaechlich gestellten Wert zurueckgerechnet -- sonst laeuft er
   gegen die Ratengrenze genauso hoch wie frueher gegen die Amplituden-
   grenze.
   ACHTUNG beim Auslegen: Eine zu enge Ratengrenze ist selbst eine
   Phasenverzoegerung und kann neue Grenzzyklen erzeugen (bekannt aus der
   Luftfahrt als rate-limiting-induzierte PIO). Sie ist hier als
   Sicherheitsnetz gedacht, nicht als Hauptmassnahme -- die Hauptmassnahme
   ist ein sauberes Messsignal (siehe wave_filter_node.py).
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AntiWindupPID:
    kff: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    t_t: float = 1.0                  # Rueckrechnungszeitkonstante
    tau_d: float = 0.05               # s, Tiefpass auf den D-Anteil
    u_min: float = float("-inf")
    u_max: float = float("inf")
    du_max: float = float("inf")      # Stellgroesse pro Sekunde (Ratenbegrenzung)
    integral: float = 0.0
    prev_error: Optional[float] = None
    _d_state: float = 0.0
    _u_unsat: float = 0.0
    _u_sat: float = 0.0
    _u_prev: Optional[float] = None   # letzter tatsaechlich gestellter Wert

    # ------------------------------------------------------------------
    def reset(self, integral: float = 0.0):
        """Stossfreie Umschaltung: Integrator auf aktuellen Stellwert setzen."""
        self.integral = integral
        self.prev_error = None
        self._d_state = 0.0
        self._u_unsat = 0.0
        self._u_prev = None           # Ratenbegrenzung startet am ersten Wert

    def set_limits(self, u_min: float, u_max: float):
        """Stellgrenzen zur Laufzeit anpassen (siehe Modulkopf)."""
        if u_max < u_min:
            u_min, u_max = u_max, u_min
        self.u_min, self.u_max = u_min, u_max

    # ------------------------------------------------------------------
    def step(self, setpoint: float, measurement: float, ff_input: float, dt: float) -> float:
        if dt <= 0.0:
            return max(self.u_min, min(self.u_max, self._u_unsat))

        error = setpoint - measurement

        # --- gefilterter D-Anteil ---
        d_term = 0.0
        if self.kd != 0.0:
            if self.prev_error is not None:
                d_raw = (error - self.prev_error) / dt
                alpha = dt / max(self.tau_d + dt, 1e-9)
                self._d_state += alpha * (d_raw - self._d_state)
                d_term = self.kd * self._d_state
        self.prev_error = error

        u_unsat = (self.kff * ff_input) + (self.kp * error) + self.integral + d_term

        # 1. Amplitudenbegrenzung
        u_sat = max(self.u_min, min(self.u_max, u_unsat))

        # 2. Ratenbegrenzung (siehe Modulkopf Punkt 4)
        if math.isfinite(self.du_max) and self._u_prev is not None:
            d_max = self.du_max * dt
            u_sat = self._u_prev + max(-d_max, min(d_max, u_sat - self._u_prev))
        self._u_prev = u_sat

        self._u_sat = u_sat
        # Integrator inkl. Rueckrechnung gegen den TATSAECHLICH gestellten Wert.
        # u_sat enthaelt hier bereits beide Begrenzungen -> kein Windup, weder
        # gegen die Amplituden- noch gegen die Ratengrenze.
        self.integral += dt * (self.ki * error + (1.0 / self.t_t) * (u_sat - u_unsat))
        self._u_unsat = u_unsat
        return u_sat

    # ------------------------------------------------------------------
    def back_calculate(self, u_wirklich: float, dt: float):
        """Meldet den nach der Aktorik-Aufteilung TATSAECHLICH gestellten
        Wert zurueck und korrigiert den Integrator entsprechend.
        Ohne diesen Schritt weiss der Regler nichts von Begrenzungen,
        die erst NACH ihm entstehen (Schubaufteilung, Motorlimit)."""
        if dt <= 0.0:
            return
        self.integral += dt * (1.0 / self.t_t) * (u_wirklich - self._u_sat)
        # Die Ratenbegrenzung muss ab jetzt vom TATSAECHLICH gestellten Wert
        # aus weiterrechnen, sonst begrenzt sie im naechsten Takt gegen einen
        # Wert, den die Aktorik nie erreicht hat.
        self._u_prev = u_wirklich