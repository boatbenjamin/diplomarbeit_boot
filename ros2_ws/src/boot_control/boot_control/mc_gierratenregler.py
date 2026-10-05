"""
mc_gierratenregler.py
=============================================================
Innere Regelschleife: Gierrate -> Giermoment N [N*m].

Auslegung aus dem Ersatzmodell 1. Ordnung der Gierdynamik

    Izz * r_dot + n_r * r = N     <=>    T*r_dot + r = K*N
    T = Izz/n_r        K = 1/n_r

    Kff = 1            (die Vorsteuerung wird ausserhalb schon als
                        Moment gerechnet: n_r*r_d + n_rr*|r_d|*r_d)
    Kp  = T*omega_i/K  = Izz*omega_i
    Ki  = Kp*omega_i/5
    T_t = Kp/Ki        = 5/omega_i

Geschlossene Schleife:  T_cl = Izz / (n_r + Kp) = T / (1 + T*omega_i)

WARUM DIE ALTEN WERTE NICHT FUNKTIONIERT HABEN
----------------------------------------------
Alt: K_NOMOTO = 1.1, T_NOMOTO = 1.2 -> Kp = 3.27 N*m pro rad/s.
Die Gierdaempfung des WAM-V ist aber n_r = 800 N*m pro rad/s.
Der P-Anteil war damit rund 250-mal zu schwach; geregelt hat
praktisch nur der I-Anteil, der ueber ~50 s hochgelaufen ist und
dann 45 Grad ueberschwungen hat.

WARUM omega_i NICHT BELIEBIG GROSS SEIN DARF (2026-10-05)
---------------------------------------------------------
Kp = Izz*omega_i wirkt direkt auf das MESSRAUSCHEN der Gierrate.
Mit Izz = 700 und omega_i = 3.5 ist Kp = 2450 N*m pro rad/s, das
Momentenlimit liegt bei n_max = 2*f_max*d_y ~ 3081 N*m. Ein
Messfehler von nur 1.26 rad/s treibt die Aktorik also in die volle
Saettigung. Genau das ist in VRX passiert: Die alte, harte
Plausibilitaetsumschaltung im Zustandsschaetzer erzeugte Spruenge
dieser Groessenordnung, der Regler verstaerkte sie zu Schubschlaegen
von ueber 1000 N, und aus Saettigung + Rueckkopplung wurde ein
stabiler Grenzzyklus bei 12.5 Hz (Rechteckform, ungerade Harmonische
bei 37.5 und 62.5 Hz). Das Boot "zitterte" dann dauerhaft.

Zwei Konsequenzen, beide hier umgesetzt:
  1. omega_i bleibt bei ~3 rad/s (Empfehlung des Konzeptpapiers).
     Die geschlossene Schleife ist damit T_cl = 0.23 s schnell --
     reichlich fuer ein Boot, dessen Bahnkruemmung sich im
     Sekundenbereich aendert.
  2. Ratenbegrenzung du_max auf das Giermoment. Sie bremst das, was
     an Messrauschen uebrig bleibt, bevor es die Motoren erreicht.
     Auslegung: ein ehrlicher Manoeverbedarf ist
         dN = n_r * r_max = 800 * 0.785 = 628 N*m,
     und den soll die Aktorik in ~0.1 s stellen duerfen
     -> rund 6000 N*m/s. Das ist knapp 3-mal mehr als jedes reale
     Manoever braucht, kappt aber die gemessenen Rauschspruenge
     (bis 2100 N*m in EINEM 20-ms-Takt = 105000 N*m/s) um Faktor 17.
     du_max = 0 oder inf schaltet die Begrenzung ab.
"""

import math

from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import BootParameter


def make_gierraten_pid(p: BootParameter) -> AntiWindupPID:
    """Stellgroesse ist das Giermoment N [N*m]; die Umrechnung auf
    Motorschuebe passiert in mc_schubaufteilung.py."""
    # Vorsteuerung wird ausserhalb als Moment berechnet
    # (gier_vorsteuerung: n_r*r_d + n_rr*|r_d|*r_d) -> hier nur Faktor 1
    kff = 1.0
    kp = p.izz * p.omega_i                      # = T_nomoto*omega_i/K_nomoto
    ki = kp * p.omega_i / 5.0
    t_t = (kp / ki) if ki > 0 else 1.0

    du = getattr(p, 'dn_max', 0.0)
    du_max = float(du) if (du and math.isfinite(du) and du > 0.0) else float('inf')

    return AntiWindupPID(
        kff=kff, kp=kp, ki=ki,
        kd=getattr(p, 'kd_r', 0.0),             # Standard 0 -- siehe Konzept 11.2
        tau_d=getattr(p, 'tau_d_r', 0.1),
        t_t=t_t,
        u_min=-p.n_max, u_max=p.n_max,
        du_max=du_max,
    )
