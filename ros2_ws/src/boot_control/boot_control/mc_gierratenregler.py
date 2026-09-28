"""
mc_gierratenregler.py
=============================================================
Innerer Regler der Kaskade: aus der Soll-Gierrate r_d wird das
Giermoment N [N*m].

Fuer die Auslegung wird die Gierdynamik als System 1. Ordnung
angenaehert (Nomoto-Modell):

    Izz * r_dot + n_r * r = N     <=>    T*r_dot + r = K*N
    T = Izz/n_r        K = 1/n_r

Daraus die Reglerparameter mit der gewuenschten Bandbreite omega_i:

    Kff = 1/K         = n_r
    Kp  = T*omega_i/K = Izz*omega_i
    Ki  = Kp*omega_i/5

Wichtig ist, dass Kp zur Gierdaempfung n_r des Boots passt. Ist Kp zu
klein, macht praktisch nur noch der I-Anteil die Arbeit. Der laeuft
langsam hoch und das Boot schwingt beim Kurswechsel stark ueber.
"""

from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import BootParameter


def make_gierraten_pid(p: BootParameter) -> AntiWindupPID:
    """Baut den PID fuer die Gierratenschleife auf.

    Stellgroesse ist das Giermoment N [N*m]. Auf die beiden Motorkraefte
    umgerechnet wird es erst in mc_schubaufteilung.py.
    """
    # Die Vorsteuerung wird schon ausserhalb als fertiges Moment berechnet
    # (gier_vorsteuerung: n_r*r + n_rr*|r|*r), hier also nur Faktor 1.
    kff = 1.0
    # ACHTUNG: das ergibt Izz*omega_i/5, nicht Izz*omega_i wie in der
    # Auslegung oben und in params.yaml angeschrieben. Der Faktor 5 ist
    # noch zu klaeren -- entweder die Formel oder die Kommentare anpassen.
    kp = p.t_nomoto * p.omega_i / (5 * p.k_nomoto)
    ki = kp * p.omega_i / 5.0
    t_t = (kp / ki) if ki > 0 else 1.0
    return AntiWindupPID(
        kff=kff, kp=kp, ki=ki, kd=0.0, t_t=t_t,
        u_min=-p.n_max, u_max=p.n_max,
    )