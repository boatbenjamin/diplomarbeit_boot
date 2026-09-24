"""
mc_tempo_regler.py
=============================================================
Geschwindigkeitsschleife: Sollfahrt -> Laengskraft X [N].

    X_ff(u_d) = x_u*u_d + x_uu*|u_d|*u_d     (echte Widerstandskurve)
    X         = X_ff + Kp*e_u + Ki*Integral(e_u)

Auslegung: m*u_dot + d(u) = X. Mit gewuenschter Zeitkonstante
tau_u wird Kp = m/tau_u, Ki = Kp/(5*tau_u).

Alt: Kp = 8 N/(m/s), Ki = 4 -- bei 250 kg Bootsmasse praktisch
wirkungslos; zusammen mit der um Faktor 200 zu kleinen Vorsteuerung
brauchte das Boot ueber eine Minute bis zur Sollfahrt.
"""

from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import BootParameter, widerstand_vorsteuerung

TAU_U = 3.0    # s, gewuenschte Anfahrzeitkonstante


def make_tempo_pi(p: BootParameter, tau_u: float = TAU_U) -> AntiWindupPID:
    kp = p.masse / tau_u
    ki = kp / (5.0 * tau_u)
    return AntiWindupPID(
        kff=1.0, kp=kp, ki=ki, kd=0.0,
        t_t=(kp / ki) if ki > 0 else 1.0,
        u_min=-2.0 * p.f_max, u_max=2.0 * p.f_max,
    )


def tempo_regelzyklus(pi: AntiWindupPID, u_c: float, u_hat: float,
                      dt: float, p: BootParameter) -> float:
    """Ein Regelzyklus. Liefert die Laengskraft X [N]."""
    return pi.step(setpoint=u_c, measurement=u_hat,
                   ff_input=widerstand_vorsteuerung(u_c, p), dt=dt)
