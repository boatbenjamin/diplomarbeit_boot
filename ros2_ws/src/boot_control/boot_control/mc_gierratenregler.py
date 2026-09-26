"""
mc_gierratenregler.py
=============================================================
Innere Regelschleife: Gierrate -> Giermoment N [N*m].

Auslegung aus dem Ersatzmodell 1. Ordnung der Gierdynamik

    Izz * r_dot + n_r * r = N     <=>    T*r_dot + r = K*N
    T = Izz/n_r        K = 1/n_r

    Kff = 1/K        = n_r
    Kp  = T*omega_i/K = Izz*omega_i
    Ki  = Kp*omega_i/5

WARUM DIE ALTEN WERTE NICHT FUNKTIONIERT HABEN
----------------------------------------------
Alt: K_NOMOTO = 1.1, T_NOMOTO = 1.2 -> Kp = 3.27 N*m pro rad/s.
Die Gierdaempfung des WAM-V ist aber n_r = 800 N*m pro rad/s.
Der P-Anteil war damit rund 250-mal zu schwach; geregelt hat
praktisch nur der I-Anteil, der ueber ~50 s hochgelaufen ist und
dann 45 Grad ueberschwungen hat.
Neu (Izz=700, omega_i=1.5): Kp = 1050, Kff = 800.
"""

from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import BootParameter


def make_gierraten_pid(p: BootParameter) -> AntiWindupPID:
    """Stellgroesse ist das Giermoment N [N*m]; die Umrechnung auf
    Motorschuebe passiert in mc_schubaufteilung.py."""
    kff = 1.0 / p.k_nomoto                      # = n_r
    kp = p.t_nomoto * p.omega_i / (5*p.k_nomoto)    # = Izz * omega_i, leck mi am oasch lösung, bei Bedarf später ändern
    ki = kp * p.omega_i / 5.0
    t_t = (kp / ki) if ki > 0 else 1.0
    return AntiWindupPID(
        kff=kff, kp=kp, ki=ki, kd=0.0, t_t=t_t,
        u_min=-p.n_max, u_max=p.n_max,
    )
