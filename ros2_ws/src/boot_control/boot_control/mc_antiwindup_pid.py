
from dataclasses import dataclass
from typing import Optional


@dataclass
class AntiWindupPID:
    kff: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    t_t: float = 1.0                  # s, wie schnell die Rueckrechnung wirkt
    tau_d: float = 0.05               # s, Tiefpass auf den D-Anteil
    u_min: float = float("-inf")
    u_max: float = float("inf")
    integral: float = 0.0
    prev_error: Optional[float] = None
    _d_state: float = 0.0
    _u_unsat: float = 0.0
    _u_sat: float = 0.0


    def reset(self, integral: float = 0.0):
        """Integrator setzen, damit beim Einschalten kein Sprung entsteht."""
        self.integral = integral
        self.prev_error = None
        self._d_state = 0.0
        self._u_unsat = 0.0

    def set_limits(self, u_min: float, u_max: float):
        """Stellgrenzen im laufenden Betrieb aendern.

        Die Gierratenschleife braucht das, weil das verfuegbare Giermoment
        davon abhaengt, wieviel Schub gerade fuer den Vortrieb draufgeht.
        """
        if u_max < u_min:
            u_min, u_max = u_max, u_min
        self.u_min, self.u_max = u_min, u_max


    def step(self, setpoint: float, measurement: float, ff_input: float, dt: float) -> float:
        if dt <= 0.0:
            return max(self.u_min, min(self.u_max, self._u_unsat))

        error = setpoint - measurement

        # --- D-Anteil, gefiltert ---
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
        # Integrator hochzaehlen und gleichzeitig um die Begrenzung korrigieren
        self.integral += dt * (self.ki * error + (1.0 / self.t_t) * (u_sat - u_unsat))
        self._u_unsat = u_unsat
        return u_sat


    def back_calculate(self, u_wirklich: float, dt: float):
        """Meldet dem Integrator den Wert zurueck, der wirklich gestellt wurde.

        Noetig, weil nach dem Regler noch weitere Begrenzungen kommen
        (Schubaufteilung, Motorlimit). Ohne diese Rueckmeldung weiss der
        Regler nichts davon und laeuft dagegen an.
        """
        if dt <= 0.0:
            return
        self.integral += dt * (1.0 / self.t_t) * (u_wirklich - self._u_sat)