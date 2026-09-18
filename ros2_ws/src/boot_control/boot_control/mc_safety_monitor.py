"""
mc_safety_monitor.py
=============================================================
Safety-Monitor / Watchdog auf der Mikrocontroller-Seite (Kapitel 15
des Konzeptpapiers).

Wichtig (Kasten 'Die Sicherheitskette darf nicht durch ROS 2 fuehren',
Kapitel 15): Not-Aus, Stellgroessengrenzen und Stromueberwachung
gehoeren auf den Mikrocontroller und duerfen NICHT davon abhaengen,
dass eine Nachricht rechtzeitig vom Linux-Rechner ankommt. DDS ist
zuverlaessig, aber nicht echtzeitgarantiert. Ein bereits vorhandener
Watchdog auf der Linux-Seite ersetzt diese Logik hier nicht - er
ergaenzt sie hoechstens (z.B. fuer Missionsabbruch), denn er faellt
ja gerade dann aus, wenn der Linux-Rechner ausfaellt.

Ablauf bei ausbleibenden Fahrbefehlen:
  - bis T_TIMEOUT_BEFEHL (0.5 s): Befehl normal ausfuehren
  - danach bis T_HALT_LETZTER_KURS (3 s): letzten Kurs/Tempo halten
  - danach: Stopp (F_L = F_R = 0), Position halten
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from boot_control.mc_common import SafetyStatus

# --- Startwerte (Anhang B) ---------------------------------------------
T_TIMEOUT_BEFEHL = 0.5         # s, ab hier gilt der Linux-Rechner als stumm
T_HALT_LETZTER_KURS = 3.0      # s, danach vollstaendiger Stopp
T_STROM_DAUER_GRENZE = 10.0    # s, Motorstrom-Ueberschreitung bis zum Abbruch (Kapitel 15, Tabelle)


@dataclass
class SafetyMonitor:
    t_timeout_befehl: float = T_TIMEOUT_BEFEHL
    t_halt: float = T_HALT_LETZTER_KURS
    t_strom_grenze: float = T_STROM_DAUER_GRENZE

    zeit_seit_letztem_befehl: float = 0.0
    letzter_psi_c: Optional[float] = None
    letzter_u_c: Optional[float] = None
    not_aus: bool = False
    strom_ueberschreitung_seit: float = 0.0

    def neuer_befehl(self, psi_c: float, u_c: float):
        """Bei jedem Empfang eines Fahrbefehls (/cmd/course_safe) aufrufen."""
        self.zeit_seit_letztem_befehl = 0.0
        self.letzter_psi_c = psi_c
        self.letzter_u_c = u_c

    def tick(self, dt: float, motorstrom: Optional[float] = None,
             strom_grenze: float = math.inf) -> SafetyStatus:
        """Einmal pro Regelzyklus aufrufen, unabhaengig davon, ob ein
        neuer Befehl angekommen ist."""
        self.zeit_seit_letztem_befehl += dt

        if self.not_aus:
            return SafetyStatus(ok=False, reason="Not-Aus aktiv")

        if motorstrom is not None:
            if motorstrom > strom_grenze:
                self.strom_ueberschreitung_seit += dt
            else:
                self.strom_ueberschreitung_seit = 0.0
            if self.strom_ueberschreitung_seit > self.t_strom_grenze:
                return SafetyStatus(ok=False, reason="Motorstrom dauerhaft zu hoch (Fremdkoerper?)")

        if self.zeit_seit_letztem_befehl > self.t_halt:
            return SafetyStatus(ok=False, reason="Fahrbefehl-Timeout: Stopp, Position halten")

        if self.zeit_seit_letztem_befehl > self.t_timeout_befehl:
            return SafetyStatus(ok=True, reason="Fahrbefehl-Timeout: letzten Kurs halten")

        return SafetyStatus(ok=True, reason="")

    def effektiver_befehl(self, psi_c_eingang: float, u_c_eingang: float) -> Tuple[float, float]:
        """Liefert (psi_c, u_c), die dem Regler tatsaechlich vorgegeben
        werden - je nach Watchdog-Zustand der frische Befehl oder der
        zuletzt bekannte."""
        if self.zeit_seit_letztem_befehl > self.t_timeout_befehl and self.letzter_psi_c is not None:
            return self.letzter_psi_c, self.letzter_u_c
        return psi_c_eingang, u_c_eingang

    def loese_not_aus_aus(self):
        self.not_aus = True

    def quittiere_not_aus(self):
        self.not_aus = False
        self.strom_ueberschreitung_seit = 0.0
