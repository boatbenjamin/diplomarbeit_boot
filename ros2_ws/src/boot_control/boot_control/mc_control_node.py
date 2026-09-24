"""
mc_control_node.py
=============================================================
Verdrahtet Kursschleife, Gierratenschleife, Tempo-Schleife,
Schubaufteilung und Safety-Monitor zu einem Regelzyklus.

Entspricht dem 50-Hz-Timer-Callback auf dem Mikrocontroller.
Reine Rechenlogik, KEIN rclpy -- die alte Version hatte hier unten
ein main(), das rclpy.spin() auf eine dataclass losgelassen haette
(ControlNode ist kein rclpy.node.Node). Das ist entfernt.

Aenderungen ggue. der alten Version
-----------------------------------
1. Alle Plattformparameter kommen aus BootParameter.
2. Nach der Schubaufteilung wird beiden Reglern das TATSAECHLICH
   gestellte Moment / die tatsaechliche Kraft zurueckgemeldet
   (back_calculate) -- dadurch kein Windup mehr gegen die Aktorik.
3. Die Gierratenschleife bekommt bei Bedarf enger gesetzte Grenzen.
4. Kein impliziter neuer_befehl() mehr in jedem Zyklus (siehe
   boat_control_node.py): der Watchdog kann jetzt ueberhaupt ausloesen.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from boot_control.mc_common import (BootParameter, BootState, DT,
                                    SafetyStatus, ThrustCommand)
from boot_control.mc_kursregler import CourseController
from boot_control.mc_gierratenregler import make_gierraten_pid
from boot_control.mc_tempo_regler import make_tempo_pi, tempo_regelzyklus
from boot_control.mc_schubaufteilung import berechne_schubaufteilung
from boot_control.mc_safety_monitor import SafetyMonitor
from boot_control.mc_antiwindup_pid import AntiWindupPID


@dataclass
class ControlNode:
    p: BootParameter = field(default_factory=BootParameter)
    kurs_regler: CourseController = None
    gierraten_pid: AntiWindupPID = None
    tempo_pi: AntiWindupPID = None
    safety: SafetyMonitor = field(default_factory=SafetyMonitor)

    # Diagnose (fuer /diag-Topics oder Plots)
    letztes_r_d: float = 0.0
    letztes_N: float = 0.0
    letztes_X: float = 0.0

    def __post_init__(self):
        if self.kurs_regler is None:
            self.kurs_regler = CourseController(p=self.p)
        if self.gierraten_pid is None:
            self.gierraten_pid = make_gierraten_pid(self.p)
        if self.tempo_pi is None:
            self.tempo_pi = make_tempo_pi(self.p)

    # ------------------------------------------------------------------
    def stossfreie_initialisierung(self, x0: BootState):
        """Beim Uebergang nach 'aktiv' aufrufen."""
        self.kurs_regler.reset(x0.psi)
        # Integrator so setzen, dass der aktuelle Zustand gehalten wird
        self.gierraten_pid.reset(integral=0.0)
        self.tempo_pi.reset(integral=0.0)

    # ------------------------------------------------------------------
    def regelzyklus(self, x: BootState, psi_c_eingang: float, u_c_eingang: float,
                    dt: float = DT, motorstrom: Optional[float] = None
                    ) -> Tuple[ThrustCommand, SafetyStatus]:
        status = self.safety.tick(dt, motorstrom=motorstrom)
        if not status.ok:
            return ThrustCommand(0.0, 0.0), status

        psi_c, u_c = self.safety.effektiver_befehl(psi_c_eingang, u_c_eingang)
        u_c = max(-self.p.u_max, min(self.p.u_max, u_c))

        # --- Aeussere Schleife: Kurs -> Soll-Gierrate ---
        r_d = self.kurs_regler.step(psi_c, x.psi, dt)

        # --- Innere Schleife: Gierrate -> Giermoment ---
        N = self.gierraten_pid.step(setpoint=r_d, measurement=x.r,
                                    ff_input=r_d, dt=dt)

        # --- Tempo-Schleife -> Laengskraft ---
        X = tempo_regelzyklus(self.tempo_pi, u_c, x.u, dt, self.p)

        # --- Schubaufteilung MIT Momenten-Prioritaet ---
        f_l, f_r, n_wirklich, x_wirklich = berechne_schubaufteilung(X, N, self.p)

        # --- Anti-Windup gegen die Aktorikgrenzen ---
        self.gierraten_pid.back_calculate(n_wirklich, dt)
        self.tempo_pi.back_calculate(x_wirklich, dt)

        self.letztes_r_d, self.letztes_N, self.letztes_X = r_d, n_wirklich, x_wirklich
        return ThrustCommand(F_L=f_l, F_R=f_r), status
