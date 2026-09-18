"""
mc_control_node.py
=============================================================
Fasst Kursschleife, Gierratenschleife, Geschwindigkeitsschleife,
Schubaufteilung und Safety-Monitor zu einem einzigen Regelzyklus
zusammen (Kapitel 11 des Konzeptpapiers).

Entspricht dem, was spaeter im 50-Hz-Timer-Callback des
/control_node auf dem Mikrocontroller laeuft. Importiert bewusst
nur die anderen mc_*.py-Module - keine Logik wird hier neu
implementiert, nur verdrahtet.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from boot_control.mc_common import BootState, DT, ThrustCommand, SafetyStatus
from boot_control.mc_kursregler import CourseController
from boot_control.mc_gierratenregler import make_gierraten_pid
from boot_control.mc_tempo_regler import make_tempo_pi, tempo_regelzyklus
from boot_control.mc_schubaufteilung import schubaufteilung
from boot_control.mc_safety_monitor import SafetyMonitor
from boot_control.mc_antiwindup_pid import AntiWindupPID


@dataclass
class ControlNode:
    kurs_regler: CourseController = field(default_factory=CourseController)
    gierraten_pid: AntiWindupPID = field(default_factory=make_gierraten_pid)
    tempo_pi: AntiWindupPID = field(default_factory=make_tempo_pi)
    safety: SafetyMonitor = field(default_factory=SafetyMonitor)

    def stossfreie_initialisierung(self, x0: BootState):
        """Beim Uebergang in 'aktiv' (Lebenszyklus-Knoten, Kapitel 11.3)
        aufrufen, um Integratoren stossfrei zu setzen."""
        self.kurs_regler.reset(x0.psi)
        self.gierraten_pid.reset(integral=0.0)
        self.tempo_pi.reset(integral=0.0)

    def regelzyklus(self, x: BootState, psi_c_eingang: float, u_c_eingang: float,
                     dt: float = DT, motorstrom: Optional[float] = None,
                     befehl_empfangen: bool = True) -> Tuple[ThrustCommand, SafetyStatus]:
        """
        Ein Aufruf pro 50-Hz-Takt.

        befehl_empfangen=False simuliert einen Zyklus, in dem KEIN
        neuer Fahrbefehl vom Linux-Rechner angekommen ist (fuer den
        Watchdog-Test); im echten Knoten wird neuer_befehl() nur im
        Subscriber-Callback von /cmd/course_safe aufgerufen, tick()
        dagegen in jedem Timer-Takt.
        """
        if befehl_empfangen:
            self.safety.neuer_befehl(psi_c_eingang, u_c_eingang)

        status = self.safety.tick(dt, motorstrom=motorstrom)

        if not status.ok:
            # Not-Aus oder Fahrbefehl-Timeout mit Stopp -> keine Vortriebskraft
            return ThrustCommand(0.0, 0.0), status

        psi_c, u_c = self.safety.effektiver_befehl(psi_c_eingang, u_c_eingang)

        # --- Aeussere Schleife: Kurs -> Soll-Gierrate ---
        r_d = self.kurs_regler.step(psi_c, x.psi, dt)

        # --- Innere Schleife: Gierrate -> Giermoment ---
        N = self.gierraten_pid.step(setpoint=r_d, measurement=x.r, ff_input=r_d, dt=dt)

        # --- Geschwindigkeitsschleife -> Laengskraft ---
        X = tempo_regelzyklus(self.tempo_pi, u_c, x.u, dt)

        # --- Schubaufteilung mit Momenten-Prioritaet ---
        thrust = schubaufteilung(X, N)

        return thrust, status

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = ControlNode()  # Ersetze 'ControlNode' mit dem echten Namen deiner Klasse in dieser Datei!
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()