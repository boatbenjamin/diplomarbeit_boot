
from boot_control.mc_antiwindup_pid import AntiWindupPID
from boot_control.mc_common import BootParameter, widerstand_vorsteuerung

TAU_U = 3.0    # s, so lange soll das Boot etwa bis zur Sollfahrt brauchen


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
    """Ein Regeltakt. Liefert die noetige Laengskraft X [N]."""
    return pi.step(setpoint=u_c, measurement=u_hat,
                   ff_input=widerstand_vorsteuerung(u_c, p), dt=dt)
