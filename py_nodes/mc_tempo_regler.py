"""
mc_tempo_regler.py
=============================================================
Geschwindigkeitsschleife: Sollfahrt -> Laengskraft (Kapitel 11.2
des Konzeptpapiers).

PI mit Vorsteuerung aus der bekannten Widerstandskurve. Weil der
Zusammenhang zwischen Schub und Fahrt quadratisch ist, bildet die
Vorsteuerung diesen Zusammenhang ab - sonst muesste der
Integralanteil ihn staendig nachziehen:

    X_ff(u_d) = 0.5 * rho * S * Ct * u_d^2
    X = X_ff + Kp*e_u + Ki*Integral(e_u)

Laeuft mit 50 Hz, parallel zur Gierratenschleife.
"""

from mc_antiwindup_pid import AntiWindupPID
from mc_common import F_MAX, widerstand_vorsteuerung

# --- Startwerte -------------------------------------------------------
KP_U = 8.0                    # N/(m/s), P-Anteil
KI_U = 4.0                    # N/(m/s)/s, I-Anteil
T_T_U = KP_U / KI_U if KI_U > 0 else 1.0


def make_tempo_pi() -> AntiWindupPID:
    """Vorsteuerung wird bereits fertig berechnet als ff_input
    uebergeben (siehe step()-Aufruf unten), daher kff=1.0."""
    return AntiWindupPID(
        kff=1.0,
        kp=KP_U,
        ki=KI_U,
        kd=0.0,
        t_t=T_T_U,
        u_min=-2 * F_MAX,
        u_max=2 * F_MAX,
    )


def tempo_regelzyklus(pi: AntiWindupPID, u_c: float, u_hat: float, dt: float) -> float:
    """Ein Regelzyklus: berechnet die Vorsteuerung und ruft den PI auf.
    Liefert die Laengskraft X [N]."""
    x_ff = widerstand_vorsteuerung(u_c)
    return pi.step(setpoint=u_c, measurement=u_hat, ff_input=x_ff, dt=dt)


if __name__ == "__main__":
    pi = make_tempo_pi()
    u_hat = 0.0
    u_c = 1.5  # m/s, Marschfahrt (Anhang B)
    dt = 0.02
    masse = 6.0  # kg (Kapitel 6.1)

    print(f"KP_U={KP_U}  KI_U={KI_U}  T_T_U={T_T_U:.3f}")
    print(f"\nSprungantwort auf u_c = {u_c} m/s (Abnahme Kapitel 16, Schritt 6: <3s stationaer):")
    print(f"{'t[s]':>6} {'u[m/s]':>8} {'X[N]':>8}")
    for i in range(int(6.0 / dt)):
        t = i * dt
        X = tempo_regelzyklus(pi, u_c, u_hat, dt)
        widerstand = widerstand_vorsteuerung(u_hat)
        u_dot = (X - widerstand) / masse
        u_hat += u_dot * dt
        if i % 25 == 0:
            print(f"{t:6.2f} {u_hat:8.3f} {X:8.2f}")