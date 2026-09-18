"""
mc_antiwindup_pid.py
=============================================================
Generischer PID-Regler mit Vorsteuerung und Rueckrechnungs-
Anti-Windup (Kapitel 11.3 des Konzeptpapiers).

    I_dot = Ki*e + (1/Tt)*(u_sat - u_unsat)

Wird sowohl von der Gierratenschleife (mc_gierratenregler.py) als
auch von der Geschwindigkeitsschleife (mc_tempo_regler.py) verwendet.

Kd sollte i.d.R. 0 bleiben (Kapitel 11.2: wirkt auf die Ableitung
der Gierrate, also genau auf den Frequenzbereich der Wellen) und
nur bei Bedarf und gefiltert hinzugenommen werden.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AntiWindupPID:
    kff: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    t_t: float = 1.0          # Rueckrechnungszeitkonstante
    u_min: float = float("-inf")
    u_max: float = float("inf")
    integral: float = 0.0
    prev_error: Optional[float] = None

    def reset(self, integral: float = 0.0):
        """Fuer stossfreie Umschaltung (Kapitel 11.3): Integrator auf
        aktuellen Stellwert initialisieren, z.B. beim Uebergang eines
        Lebenszyklus-Knotens nach 'aktiv'."""
        self.integral = integral
        self.prev_error = None

    def step(self, setpoint: float, measurement: float, ff_input: float, dt: float) -> float:
        error = setpoint - measurement

        d_term = 0.0
        if self.kd != 0.0 and self.prev_error is not None:
            d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        u_unsat = (self.kff * ff_input) + (self.kp * error) + self.integral + d_term
        u_sat = max(self.u_min, min(self.u_max, u_unsat))

        # Integrator-Update inkl. Rueckrechnung (Anti-Windup)
        self.integral += dt * (self.ki * error + (1.0 / self.t_t) * (u_sat - u_unsat))

        return u_sat


if __name__ == "__main__":
    # Kleiner Selbsttest: Sprungantwort auf einen Sollwert mit
    # Stellgrenzen, die den Regler in die Saettigung zwingen.
    pid = AntiWindupPID(kp=2.0, ki=1.0, t_t=2.0, u_min=-1.0, u_max=1.0)
    measurement = 0.0
    print(f"{'t':>5} {'u_sat':>8} {'integral':>10} {'messung':>8}")
    dt = 0.1
    for i in range(50):
        t = i * dt
        u = pid.step(setpoint=5.0, measurement=measurement, ff_input=0.0, dt=dt)
        # simple Strecke 1. Ordnung
        measurement += (u - measurement) * dt
        if i % 5 == 0:
            print(f"{t:5.1f} {u:8.3f} {pid.integral:10.3f} {measurement:8.3f}")