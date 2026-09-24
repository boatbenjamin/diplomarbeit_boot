"""
mc_schubaufteilung.py
=============================================================
Rechnet Laengskraft X [N] und Giermoment N [N*m] in die Einzelschuebe
F_L / F_R um.

Vorzeichenkonvention (ROS/ENU, Body Frame):
  linker Motor bei y = +d_y, rechter bei y = -d_y
  N = d_y * (F_R - F_L)      -> N > 0 dreht nach links (Gegenuhrzeiger)
  F_L = f_x - N/(2*d_y)
  F_R = f_x + N/(2*d_y)

DER ENTSCHEIDENDE FEHLER DER ALTEN VERSION
------------------------------------------
Die alte Funktion hat F_L und F_R einfach unabhaengig voneinander auf
+-F_MAX geclippt. Sobald der Vortriebswunsch gross war, lagen BEIDE
Motoren am Anschlag -- die Differenz zwischen ihnen wurde null und
damit das Giermoment ebenfalls. Das Boot hatte in dem Moment
buchstaeblich keine Lenkung mehr und blieb mit konstantem Kursfehler
stehen (im Test: 8.2 Grad Dauerfehler bei F_L = F_R = 200 N).

Neu: Momenten-Prioritaet (genau das, was der Kommentar in
mc_control_node.py schon immer behauptet hat).
  1. Zuerst die Differenz dF = N/(2*d_y) sichern.
  2. Der verbleibende Spielraum f_max - |dF| geht an den Vortrieb.
  3. Das tatsaechlich gestellte Moment/Kraft wird zurueckgegeben,
     damit die Regler ihre Integratoren korrigieren koennen.
"""

from typing import Tuple

from boot_control.mc_common import BootParameter, clip


def berechne_schubaufteilung(f_x: float, N: float,
                             p: BootParameter) -> Tuple[float, float, float, float]:
    """Liefert (F_L, F_R, N_wirklich, X_wirklich).

    N_wirklich / X_wirklich sind die nach der Begrenzung tatsaechlich
    erreichten Werte -- Eingang fuer das Anti-Windup der Regler.
    """
    # 1. Giermoment hat Vorrang
    d_f = clip(N / (2.0 * p.d_y), -p.f_max, p.f_max)

    # 2. Restspielraum fuer den gemeinsamen Vortriebsanteil
    spielraum = p.f_max - abs(d_f)
    f_x_lim = clip(f_x / 2.0, -spielraum, spielraum)

    f_l = f_x_lim - d_f
    f_r = f_x_lim + d_f

    # 3. numerische Sicherheit (darf jetzt nicht mehr greifen)
    f_l = clip(f_l, -p.f_max, p.f_max)
    f_r = clip(f_r, -p.f_max, p.f_max)

    n_wirklich = p.d_y * (f_r - f_l)
    x_wirklich = f_l + f_r
    return f_l, f_r, n_wirklich, x_wirklich
