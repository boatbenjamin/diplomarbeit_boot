"""
mc_schubaufteilung.py
=============================================================
Schubaufteilung auf die beiden Motoren mit Momenten-Prioritaet
bei Saettigung (Kapitel 11.4 des Konzeptpapiers).

    F_L = X/2 + N/(2*d_y)
    F_R = X/2 - N/(2*d_y)

Wenn Kurs- und Geschwindigkeitswunsch zusammen mehr verlangen, als
die beiden Motoren liefern koennen, ist die naheliegende Loesung -
beides gleichmaessig herunterskalieren - falsch: Das Boot verliert
dann Kurs UND Fahrt und driftet unkontrolliert ab.

Richtig ist die Momenten-Prioritaet: Das geforderte Giermoment wird
vollstaendig gestellt, der Gesamtschub wird auf den verbleibenden
Rest reduziert. Das Boot wird dadurch langsamer, haelt aber Kurs -
genau das Verhalten, das in einer Boe gebraucht wird.
"""

from boot_control.mc_common import D_Y, F_MAX, ThrustCommand, clip


def schubaufteilung(X: float, N: float, f_max: float = F_MAX, d_y: float = D_Y) -> ThrustCommand:
    n_grenz = 2.0 * f_max * d_y
    n_stell = clip(N, -n_grenz, n_grenz)

    x_grenz = max(0.0, 2.0 * f_max - abs(n_stell) / d_y)
    x_stell = clip(X, -x_grenz, x_grenz)

    f_l = x_stell / 2.0 + n_stell / (2.0 * d_y)
    f_r = x_stell / 2.0 - n_stell / (2.0 * d_y)

    # Sicherheitsnetz: einzelne Motorwerte hart auf f_max begrenzen
    f_l = clip(f_l, -f_max, f_max)
    f_r = clip(f_r, -f_max, f_max)

    return ThrustCommand(F_L=f_l, F_R=f_r)

