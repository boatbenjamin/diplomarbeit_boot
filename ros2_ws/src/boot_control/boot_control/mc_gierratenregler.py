"""
mc_gierratenregler.py
=============================================================
Innere Regelschleife: Gierrate -> Giermoment (Kapitel 11.1/11.2
des Konzeptpapiers).

Reglertyp PID mit Vorsteuerung, aus dem Nomoto-Modell (T*r_dot + r
= K*delta) abgeleitet:

    Kff = 1/K
    Kp  = T*omega_i / K
    Ki  = Kp*omega_i / 5

Laeuft mit 50 Hz, mindestens Faktor 5 schneller als die aeussere
Kursschleife (mc_kursregler.py). Regelt ausschliesslich die
Gierrate und wirkt damit unmittelbar gegen jede Stoerung (Wellen,
Wind) - die aeussere Schleife gibt nur noch eine gewuenschte
Gierrate vor.

Kd startet bei 0 (siehe Modulkopf von mc_antiwindup_pid.py) und
sollte nur erhoeht werden, wenn die Sprungantwort ohne ihn
ueberschwingt - und dann gefiltert werden.
"""

from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import D_Y, F_MAX, K_NOMOTO, T_NOMOTO

# --- Startwerte (Anhang B, am realen Boot per Zickzack-Versuch pruefen) -
OMEGA_I = 3.0    # rad/s, gewuenschte Bandbreite innere Schleife

KFF_R = 1.0 / K_NOMOTO
KP_R = T_NOMOTO * OMEGA_I / K_NOMOTO
KI_R = KP_R * OMEGA_I / 5.0
KD_R = 0.0
T_T_R = (KP_R / KI_R) if KI_R > 0 else 1.0  # Rueckrechnungszeitkonstante ~ Ti


def make_gierraten_pid() -> AntiWindupPID:
    """Erzeugt den Gierraten-PID. Stellgroesse ist das (unsaettigte)
    Giermoment N [N*m]; die Umrechnung auf Motorschuebe passiert in
    mc_schubaufteilung.py."""
    return AntiWindupPID(
        kff=KFF_R,
        kp=KP_R,
        ki=KI_R,
        kd=KD_R,
        t_t=T_T_R,
        u_min=-2 * F_MAX * D_Y,
        u_max=2 * F_MAX * D_Y,
    )
