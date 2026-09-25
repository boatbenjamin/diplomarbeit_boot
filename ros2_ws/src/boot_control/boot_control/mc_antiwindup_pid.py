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
"""

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
    integral: float = 0.0
    prev_error: Optional[float] = None
    _d_state: float = 0.0
    _u_unsat: float = 0.0

    # ------------------------------------------------------------------
    def reset(self, integral: float = 0.0):
        """Stossfreie Umschaltung: Integrator auf aktuellen Stellwert setzen."""
        self.integral = integral
        self.prev_error = None
        self._d_state = 0.0
        self._u_unsat = 0.0

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
        u_sat = max(self.u_min, min(self.u_max, u_unsat))
        self._u_sat = u_sat
        # Integrator inkl. Rueckrechnung
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
