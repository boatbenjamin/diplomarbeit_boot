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

from mc_common import BootState, DT, ThrustCommand, SafetyStatus
from mc_kursregler import CourseController
from mc_gierratenregler import make_gierraten_pid
from mc_tempo_regler import make_tempo_pi, tempo_regelzyklus
from mc_schubaufteilung import schubaufteilung
from mc_safety_monitor import SafetyMonitor
from mc_antiwindup_pid import AntiWindupPID


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


if __name__ == "__main__":
    import math
    from mc_common import widerstand_vorsteuerung, K_NOMOTO, T_NOMOTO, D_Y, F_MAX, wrap_pi

    def simuliere_boot_schritt(x: BootState, thrust: ThrustCommand, dt: float) -> BootState:
        """Einfaches Nomoto-Modell + Laengsdynamik fuer den End-zu-End-Test."""
        X = thrust.F_L + thrust.F_R
        N = D_Y * (thrust.F_L - thrust.F_R)

        r_dot = (K_NOMOTO * (N / (2.0 * F_MAX * D_Y)) - x.r) / T_NOMOTO
        r_neu = x.r + r_dot * dt
        psi_neu = wrap_pi(x.psi + r_neu * dt)

        masse = 6.0  # kg (Kapitel 6.1)
        widerstand = widerstand_vorsteuerung(x.u)
        u_dot = (X - widerstand) / masse
        u_neu = x.u + u_dot * dt

        return BootState(psi=psi_neu, r=r_neu, u=u_neu)

    node = ControlNode()
    boot = BootState(psi=0.0, r=0.0, u=0.0)
    node.stossfreie_initialisierung(boot)

    psi_soll = math.radians(60.0)
    u_soll = 1.5
    dt = DT

    print("End-zu-End-Test: 60 deg Kurswechsel bei 1.5 m/s, danach Verbindungsabbruch bei t=10s\n")
    print(f"{'t[s]':>6} {'psi[deg]':>10} {'r[deg/s]':>10} {'u[m/s]':>8} {'F_L[N]':>8} {'F_R[N]':>8}  status")

    t = 0.0
    n_steps = int(18.0 / dt)
    for step in range(n_steps):
        verbindung_ok = not (10.0 <= t < 13.5)  # 3.5s Ausfall simulieren
        thrust, status = node.regelzyklus(
            boot, psi_soll, u_soll, dt=dt, befehl_empfangen=verbindung_ok
        )
        boot = simuliere_boot_schritt(boot, thrust, dt)

        if step % int(0.5 / dt) == 0:
            print(f"{t:6.2f} {math.degrees(boot.psi):10.2f} {math.degrees(boot.r):10.2f} "
                  f"{boot.u:8.3f} {thrust.F_L:8.2f} {thrust.F_R:8.2f}  {status.reason}")
        t += dt