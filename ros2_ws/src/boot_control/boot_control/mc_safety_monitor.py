

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from boot_control.mc_common import SafetyStatus

T_TIMEOUT_BEFEHL = 0.5
T_HALT_LETZTER_KURS = 3.0
T_TIMEOUT_ZUSTAND = 1.0
T_STROM_DAUER_GRENZE = 10.0


@dataclass
class SafetyMonitor:
    t_timeout_befehl: float = T_TIMEOUT_BEFEHL
    t_halt: float = T_HALT_LETZTER_KURS
    t_timeout_zustand: float = T_TIMEOUT_ZUSTAND
    t_strom_grenze: float = T_STROM_DAUER_GRENZE

    zeit_seit_letztem_befehl: float = 0.0
    zeit_seit_letztem_zustand: float = 0.0
    letzter_psi_c: Optional[float] = None
    letzter_u_c: Optional[float] = None
    not_aus: bool = False
    strom_ueberschreitung_seit: float = 0.0

    # ------------------------------------------------------------------
    def neuer_befehl(self, psi_c: float, u_c: float):
        """Nur im Callback von /cmd/course_safe aufrufen.

        Wird das in jedem Regeltakt aufgerufen, ist die Uhr staendig auf 0
        und der Watchdog kann nie ausloesen.
        """
        self.zeit_seit_letztem_befehl = 0.0
        self.letzter_psi_c = psi_c
        self.letzter_u_c = u_c

    def neuer_zustand(self):
        """Nur im Callback von /state/filtered aufrufen (siehe neuer_befehl)."""
        self.zeit_seit_letztem_zustand = 0.0

    # ------------------------------------------------------------------
    def tick(self, dt: float, motorstrom: Optional[float] = None,
             strom_grenze: float = math.inf) -> SafetyStatus:
        self.zeit_seit_letztem_befehl += dt
        self.zeit_seit_letztem_zustand += dt

        if self.not_aus:
            return SafetyStatus(False, "Not-Aus aktiv")

        if motorstrom is not None:
            if motorstrom > strom_grenze:
                self.strom_ueberschreitung_seit += dt
            else:
                self.strom_ueberschreitung_seit = 0.0
            if self.strom_ueberschreitung_seit > self.t_strom_grenze:
                return SafetyStatus(False, "Motorstrom dauerhaft zu hoch (Fremdkoerper?)")

        if self.zeit_seit_letztem_zustand > self.t_timeout_zustand:
            return SafetyStatus(False, "Zustandsschaetzung ausgefallen: Stopp")

        if self.zeit_seit_letztem_befehl > self.t_halt:
            return SafetyStatus(False, "Fahrbefehl-Timeout: Stopp, Position halten")

        if self.zeit_seit_letztem_befehl > self.t_timeout_befehl:
            return SafetyStatus(True, "Fahrbefehl-Timeout: letzten Kurs halten")

        return SafetyStatus(True, "")

    # ------------------------------------------------------------------
    def effektiver_befehl(self, psi_c_eingang: float, u_c_eingang: float) -> Tuple[float, float]:
        if (self.zeit_seit_letztem_befehl > self.t_timeout_befehl
                and self.letzter_psi_c is not None):
            return self.letzter_psi_c, self.letzter_u_c
        return psi_c_eingang, u_c_eingang

    def loese_not_aus_aus(self):
        self.not_aus = True

    def quittiere_not_aus(self):
        self.not_aus = False
        self.strom_ueberschreitung_seit = 0.0
