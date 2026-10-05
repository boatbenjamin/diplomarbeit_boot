#!/usr/bin/env python3
"""
analyse_lauf.py
=============================================================
Auswertung eines VRX-Testlaufs -- lokal ausfuehren, statt CSVs
hin- und herzuschicken.

    python3 analyse_lauf.py lauf.csv [weitere.csv ...]
    python3 analyse_lauf.py lauf.csv --schub lauf.log

CSV-Spalten (wie vom course_test_node geschrieben):
    t, runde, s, y_e, u, x, y

Ausgegeben wird je Datei:
  - Rundenzeit (Median, ohne die Einfahrrunde)
  - Querablage y_e: RMS und Maximum, GETRENNT nach Kurve und Gerade
  - mittlere Fahrt
  - Stellrauschen, falls ein Launch-Log mit --schub angegeben wird

Warum getrennt nach Kurve und Gerade: Auf der Lemniskate entstehen die
beiden Fehlerarten aus verschiedenen Ursachen. Auf der Geraden zeigt y_e
die stationaere Genauigkeit (Integrator, Drift). In der Kurve zeigt es,
ob Vorsteuerung und Kruemmungsvorhalt stimmen. Ein gemeinsamer RMS-Wert
mischt beides und verdeckt, welche Schraube gewirkt hat.
"""

import sys
import csv
import math


def lies_csv(pfad):
    zeilen = []
    with open(pfad, newline='') as f:
        for r in csv.DictReader(f):
            try:
                zeilen.append({k: float(v) for k, v in r.items()
                               if v not in (None, '')})
            except (ValueError, TypeError):
                continue
    return zeilen


def kruemmung_aus_bahn(zeilen, fenster=5):
    """|kappa| naeherungsweise aus der gefahrenen Bahn (x, y).
    Dient nur dazu, Kurven- von Geradenabschnitten zu TRENNEN -- dafuer
    reicht die Genauigkeit."""
    n = len(zeilen)
    kap = [0.0] * n
    for i in range(fenster, n - fenster):
        x0, y0 = zeilen[i - fenster]['x'], zeilen[i - fenster]['y']
        x1, y1 = zeilen[i]['x'], zeilen[i]['y']
        x2, y2 = zeilen[i + fenster]['x'], zeilen[i + fenster]['y']
        a = math.hypot(x1 - x0, y1 - y0)
        b = math.hypot(x2 - x1, y2 - y1)
        c = math.hypot(x2 - x0, y2 - y0)
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            continue
        # Menger-Kruemmung ueber die Flaeche des Dreiecks
        flaeche = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0
        kap[i] = 4.0 * flaeche / (a * b * c)
    return kap


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else float('nan')


def auswerten(pfad):
    z = lies_csv(pfad)
    if len(z) < 20:
        print(f'{pfad}: zu wenig Daten ({len(z)} Zeilen)')
        return
    hat_runde = 'runde' in z[0]
    kap = kruemmung_aus_bahn(z)
    # Schwelle: Haelfte der groessten auftretenden Kruemmung
    k_sort = sorted(kap)
    k_hoch = k_sort[int(0.9 * len(k_sort))]
    schwelle = 0.5 * k_hoch

    # Einfahrrunde ueberspringen
    runden = sorted({int(r['runde']) for r in z}) if hat_runde else [0]
    erste = runden[0]
    nutz = [r for r in z if not hat_runde or int(r['runde']) > erste]
    if len(nutz) < 20:
        nutz = z
    idx = [i for i, r in enumerate(z) if r in nutz] if len(nutz) != len(z) else list(range(len(z)))

    ye_kurve, ye_gerade = [], []
    for i in idx:
        (ye_kurve if kap[i] > schwelle else ye_gerade).append(abs(z[i]['y_e']))
    ye_alle = [abs(z[i]['y_e']) for i in idx]
    u_alle = [z[i]['u'] for i in idx if 'u' in z[i]]

    # Rundenzeiten
    zeiten = []
    if hat_runde:
        start = {}
        for r in z:
            k = int(r['runde'])
            start.setdefault(k, r['t'])
        ks = sorted(start)
        zeiten = [start[b] - start[a] for a, b in zip(ks, ks[1:])]
        zeiten = [t for t in zeiten if t > 1.0]
    med = sorted(zeiten)[len(zeiten) // 2] if zeiten else float('nan')

    print(f'\n=== {pfad} ===')
    print(f'  Dauer {z[-1]["t"] - z[0]["t"]:.1f} s, {len(runden)} Runden, '
          f'Rundenzeit (Median, ohne Einfahrt) {med:.2f} s')
    print(f'  Fahrt im Mittel {sum(u_alle)/len(u_alle):.2f} m/s' if u_alle else '')
    print(f'  {"":16} {"RMS":>8} {"max":>8} {"n":>7}')
    print(f'  {"gesamt":16} {rms(ye_alle):8.3f} {max(ye_alle):8.3f} {len(ye_alle):7d}')
    print(f'  {"Kurve":16} {rms(ye_kurve):8.3f} '
          f'{max(ye_kurve) if ye_kurve else float("nan"):8.3f} {len(ye_kurve):7d}')
    print(f'  {"Gerade":16} {rms(ye_gerade):8.3f} '
          f'{max(ye_gerade) if ye_gerade else float("nan"):8.3f} {len(ye_gerade):7d}')


def schub_analyse(pfad):
    """Stellrauschen aus einem Launch-Log oder einem 'ros2 topic echo'-Mitschnitt.
    Das ist ab jetzt die WICHTIGSTE Kennzahl: sie zeigt sofort, ob der
    Grenzzyklus zurueck ist."""
    werte = []
    for zeile in open(pfad, errors='ignore'):
        zeile = zeile.strip()
        if zeile.startswith('data:'):
            try:
                werte.append(float(zeile.split(':', 1)[1]))
            except ValueError:
                pass
    if len(werte) < 10:
        print(f'\n{pfad}: keine Schubwerte gefunden '
              f'(erwartet Zeilen der Form "data: 123.4")')
        return
    spr = [abs(werte[i] - werte[i - 1]) for i in range(1, len(werte))]
    mittel = sum(spr) / len(spr)
    print(f'\n=== Stellrauschen {pfad} ===')
    print(f'  {len(werte)} Werte, Bereich [{min(werte):.0f}, {max(werte):.0f}] N')
    print(f'  Sprung je Takt: Mittel {mittel:.1f} N, max {max(spr):.1f} N')
    print(f'  Richtungswechsel: {100*sum(1 for i in range(1, len(spr)) if (werte[i+1]-werte[i])*(werte[i]-werte[i-1]) < 0)/max(len(spr)-1,1):.0f} %')
    if mittel > 60:
        print('  BEWERTUNG: zu hoch -- Grenzzyklus wahrscheinlich aktiv.')
    elif mittel > 25:
        print('  BEWERTUNG: grenzwertig, beobachten.')
    else:
        print('  BEWERTUNG: ruhig.')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    schub = None
    if '--schub' in args:
        i = args.index('--schub')
        schub = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]
    for p in args:
        auswerten(p)
    if schub:
        schub_analyse(schub)
