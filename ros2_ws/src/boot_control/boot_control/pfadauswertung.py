#!/usr/bin/env python3
"""
pfadauswertung.py
=============================================================
Misst die Bahnfolge-Guete unabhaengig vom ILOS-Knoten:
Querablage y_e = vorzeichenbehafteter Abstand Boot <-> /path,
pro Runde ausgewertet.

    python3 -m boot_control.pfadauswertung --runden 3
    python3 -m boot_control.pfadauswertung --runden 3 --name delta8

Abonniert
  /path            nav_msgs/Path (latched, geschlossene Runde)
  /state/filtered  nav_msgs/Odometry (gleicher odom-Frame wie /path)

Ausgabe je Runde
  RMS |y_e|, max |y_e| (+ wo auf dem Pfad), Rundenzeit, mittlere Fahrt
  und getrennt: Kreuzungszone (Pfadmitte, +-zone_m) vs. Rest.
Die Anfahrt (Runde 0) und --einlauf weitere Runden werden nicht gewertet.
CSV mit allen Messpunkten unter ~/pfadtests/.
"""

import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.utilities import remove_ros_args
from nav_msgs.msg import Odometry, Path

LATCHED = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=QoSReliabilityPolicy.RELIABLE)


# ----------------------------------------------------------------------
class PfadGeometrie:
    """Geschlossener Polygonzug mit Bogenlaenge und Projektion."""

    def __init__(self, xs, ys):
        if math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) < 1e-6:
            xs, ys = xs[:-1], ys[:-1]            # doppelten Endpunkt entfernen
        self.x, self.y, self.n = xs, ys, len(xs)
        self.s = [0.0]
        for i in range(1, self.n + 1):
            j = i % self.n
            self.s.append(self.s[-1] + math.hypot(xs[j] - xs[i - 1], ys[j] - ys[i - 1]))
        self.laenge = self.s[-1]
        # Kreuzungspunkt = Schwerpunkt (bei der Lemniskate die Mitte)
        self.mx, self.my = sum(xs) / self.n, sum(ys) / self.n

    def projiziere(self, px, py, i_start=None, fenster=40):
        """-> (y_e vorzeichenbehaftet, s, Segmentindex). Positiv = links vom Pfad."""
        if i_start is None:
            idx = range(self.n)
        else:
            idx = ((i_start + k) % self.n for k in range(-fenster // 4, fenster))
        best = None
        for i in idx:
            j = (i + 1) % self.n
            ax, ay, bx, by = self.x[i], self.y[i], self.x[j], self.y[j]
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            qx, qy = ax + t * dx, ay + t * dy
            d = math.hypot(px - qx, py - qy)
            if best is None or d < best[0]:
                seite = math.copysign(1.0, dx * (py - ay) - dy * (px - ax))
                best = (d, seite * d, self.s[i] + t * math.sqrt(l2), i)
        return best[1], best[2], best[3]


# ----------------------------------------------------------------------
class Pfadauswertung(Node):
    def __init__(self, a):
        super().__init__('pfadauswertung', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.a = a
        self.pfad = None
        self.i_last = None
        self.s_last = None
        self.runde = 0                 # 0 = Anfahrt (wird nicht gewertet)
        self.daten = []                # t, runde, s, y_e, u, x, y
        self.fertig = False
        self.t_letzte_meldung = 0.0
        self.t_wechsel = None            # Zeit des letzten Rundenwechsels
        self.create_subscription(Path, '/path', self._pfad_cb, LATCHED)
        self.create_subscription(Odometry, '/state/filtered', self._odom_cb, 20)
        self.get_logger().info(f'Warte auf /path und /state/filtered ... ({a.runden} Runden)')

    def _pfad_cb(self, msg):
        if self.pfad is not None or len(msg.poses) < 10:
            return
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        self.pfad = PfadGeometrie(xs, ys)
        self.get_logger().info(f'Pfad: {self.pfad.n} Punkte, Rundenlaenge {self.pfad.laenge:.1f} m')

    def _odom_cb(self, msg):
        if self.pfad is None or self.fertig:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        tw = msg.twist.twist.linear
        u = math.hypot(tw.x, tw.y)
        global_suche = self.i_last is None
        y_e, s, i = self.pfad.projiziere(px, py, None if global_suche else self.i_last)
        if not global_suche and abs(y_e) > 15.0:          # Fenster verloren -> global
            y_e, s, i = self.pfad.projiziere(px, py)
        # Rundenwechsel: s springt von ~Ende auf ~Anfang
        if self.t_wechsel is None:
            self.t_wechsel = t
        if (self.s_last is not None and self.s_last - s > 0.5 * self.pfad.laenge
                and t - self.t_wechsel > 10.0):          # kein Doppelzaehlen am Start
            self.runde += 1
            self.t_wechsel = t
            gewertet = self.runde > self.a.einlauf
            self.get_logger().info(f'--- Runde {self.runde} beginnt' +
                                   (' (gewertet)' if gewertet else ' (Einlauf, nicht gewertet)'))
            if self.runde > self.a.einlauf + self.a.runden:
                self.fertig = True
                return
        self.i_last, self.s_last = i, s
        self.daten.append((t, self.runde, s, y_e, u, px, py))
        if t - self.t_letzte_meldung > 5.0:
            self.t_letzte_meldung = t
            self.get_logger().info(f'Runde {self.runde}  s={s:5.1f}/{self.pfad.laenge:.0f} m  '
                                   f'y_e={y_e:+5.2f} m  u={u:4.2f} m/s')

    # ------------------------------------------------------------------
    def auswerten(self):
        a = self.a
        if not self.daten or self.pfad is None:
            print('Keine Daten.')
            return
        ordner = os.path.expanduser(a.ordner)
        os.makedirs(ordner, exist_ok=True)
        datei = os.path.join(ordner, f'{time.strftime("%Y%m%d_%H%M%S")}_{a.name}.csv')
        with open(datei, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'runde', 's', 'y_e', 'u', 'x', 'y'])
            for d in self.daten:
                w.writerow([f'{d[0]:.3f}', d[1], f'{d[2]:.2f}', f'{d[3]:.3f}',
                            f'{d[4]:.3f}', f'{d[5]:.2f}', f'{d[6]:.2f}'])

        P = self.pfad
        def in_mitte(x, y):
            return math.hypot(x - P.mx, y - P.my) <= a.zone_m

        def stat(werte):
            if not werte:
                return None, None
            return math.sqrt(sum(v * v for v in werte) / len(werte)), max(abs(v) for v in werte)

        f2 = lambda v, n=2: '  ---' if v is None else f'{v:5.{n}f}'
        print('\n================ PFADAUSWERTUNG ================')
        print(f'Name: {a.name}   Rundenlaenge {P.laenge:.1f} m   Kreuzungszone +-{a.zone_m:.0f} m')
        print(' Runde | RMS y_e | max y_e @ s   | Mitte RMS/max | Rest RMS/max | t_Runde | u_mittel')
        alle, alle_m, alle_r = [], [], []
        erste = self.a.einlauf + 1
        for rn in range(erste, self.runde + 1):
            D = [d for d in self.daten if d[1] == rn]
            if len(D) < 20:
                continue
            ye = [d[3] for d in D]
            m = [d[3] for d in D if in_mitte(d[5], d[6])]
            r = [d[3] for d in D if not in_mitte(d[5], d[6])]
            rms, mx = stat(ye)
            dmax = max(D, key=lambda d: abs(d[3]))
            rm, mm = stat(m)
            rr, mr = stat(r)
            dauer = D[-1][0] - D[0][0]
            u_m = sum(d[4] for d in D) / len(D)
            vollst = '' if D[-1][2] - D[0][2] > 0.9 * P.laenge else ' (unvollst.)'
            print(f'  {rn:3d}  |  {f2(rms)} | {f2(mx)} @ {dmax[2]:5.1f} | {f2(rm)} / {f2(mm)} '
                  f'| {f2(rr)} / {f2(mr)} | {dauer:6.1f} s | {u_m:4.2f}{vollst}')
            alle += ye; alle_m += m; alle_r += r
        if alle:
            rms, mx = stat(alle); rm, mm = stat(alle_m); rr, mr = stat(alle_r)
            print(f' GESAMT|  {f2(rms)} |  {f2(mx)}         | {f2(rm)} / {f2(mm)} | {f2(rr)} / {f2(mr)} |')

            # Verlauf ueber den Pfad (Mittel aller gewerteten Runden, 5-m-Abschnitte)
            print('\n Verlauf ueber den Pfad (alle gewerteten Runden), y_e in m:')
            print('   s [m]   y_e min   y_e mittel   y_e max   u')
            schritt = 5.0
            s0 = 0.0
            while s0 < P.laenge:
                B = [d for d in self.daten if d[1] >= erste and s0 <= d[2] < s0 + schritt]
                if B:
                    ye = [d[3] for d in B]
                    print(f'  {s0:5.0f}   {min(ye):+6.2f}    {sum(ye)/len(ye):+6.2f}     '
                          f'{max(ye):+6.2f}   {sum(d[4] for d in B)/len(B):4.2f}')
                s0 += schritt
        else:
            print('Keine vollstaendige gewertete Runde.')
        print(f'\nCSV: {datei}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runden', type=int, default=3, help='gewertete Runden (nach der Anfahrtsrunde)')
    p.add_argument('--name', default='ilos', help='Kennung im Dateinamen, z.B. delta8')
    p.add_argument('--einlauf', type=int, default=1,
                   help='zusaetzlich verworfene Runden nach der Anfahrt (Einschwingen)')
    p.add_argument('--zone_m', type=float, default=6.0, help='Radius der Kreuzungszone um die Pfadmitte [m]')
    p.add_argument('--ordner', default='~/pfadtests')
    a = p.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Pfadauswertung(a)
    try:
        while rclpy.ok() and not node.fertig:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.auswerten()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()