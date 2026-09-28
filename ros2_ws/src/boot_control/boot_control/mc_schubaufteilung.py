

from typing import Tuple

from boot_control.mc_common import BootParameter, clip


def berechne_schubaufteilung(f_x: float, N: float,
                             p: BootParameter) -> Tuple[float, float, float, float]:
    """Liefert (F_L, F_R, N_wirklich, X_wirklich).

    N_wirklich / X_wirklich sind die nach der Begrenzung tatsaechlich
    erreichten Werte -- Eingang fuer das Anti-Windup der Regler.
    """
    # 1. Giermoment hat Vorrang

    # 2. Restspielraum fuer den gemeinsamen Vortriebsanteil
    # Nach Zeile: d_f = clip(...)
    F_SURGE_RESERVE = 200.0  # N, immer für Vortrieb reserviert
    d_f = clip(N / (2.0 * p.d_y), -(p.f_max - F_SURGE_RESERVE), (p.f_max - F_SURGE_RESERVE))
    spielraum = p.f_max - abs(d_f)  # ≥ 200 N garantiert
    f_x_lim = clip(f_x / 2.0, -spielraum, spielraum)

    f_l = f_x_lim - d_f
    f_r = f_x_lim + d_f

    # 3. numerische Sicherheit (darf jetzt nicht mehr greifen)
    f_l = clip(f_l, -p.f_max, p.f_max)
    f_r = clip(f_r, -p.f_max, p.f_max)

    n_wirklich = p.d_y * (f_r - f_l)
    x_wirklich = f_l + f_r
    return f_l, f_r, n_wirklich, x_wirklich
