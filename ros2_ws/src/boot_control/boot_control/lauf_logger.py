#!/usr/bin/env python3
"""
lauf_logger.py
=============================================================
Zeichnet NUR die Rohdaten eines Testlaufs auf (t, runde, s, y_e, u, x, y)
als CSV -- die Auswertung selbst macht analyse_lauf.py. Datenerfassung
(Projektion auf /path, Rundenerkennung) uebernommen aus pfadauswertung.py,
aber ohne dessen eigene Tabellen-Ausgabe, damit es nur eine Quelle fuer
die Auswertung gibt.

Aufruf (parallel zum Launch, in einem zweiten Terminal):
    python3 lauf_logger.py --runden 5 --name testlauf0

Beendet sich automatisch nach `--runden` gewerteten Runden (+ Anfahrt)
und schreibt die CSV. Ctrl-C schreibt ebenfalls, mit dem bis dahin
aufgezeichneten Stand.

Abonniert
  /path            nav_msgs/Path (latched, geschlossene Runde)
  /state/filtered  nav_msgs/Odometry (gleicher odom-Frame wie /path)

CSV-Spalten (identisch zu analyse_lauf.py / course_test_node):
  t, runde, s, y_e, u, x, y
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


class LaufLogger(Node):
    def __init__(self, a):
        super().__init__('lauf_logger', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.a = a
        self.pfad = None
        self.i_last = None
        self.s_last = None
        self.runde = 0                 # 0 = Anfahrt (wird nicht mitgezaehlt)
        self.daten = []                 # t, runde, s, y_e, u, x, y
        self.fertig = False
        self.t_letzte_meldung = 0.0
        self.t_wechsel = None
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
        if self.t_wechsel is None:
            self.t_wechsel = t
        if (self.s_last is not None and self.s_last - s > 0.5 * self.pfad.laenge
                and t - self.t_wechsel > 10.0):           # kein Doppelzaehlen am Start
            self.runde += 1
            self.t_wechsel = t
            gewertet = self.runde > self.a.einlauf
            self.get_logger().info(f'--- Runde {self.runde} beginnt' +
                                   (' (gewertet)' if gewertet else ' (Einlauf)'))
            if self.runde > self.a.einlauf + self.a.runden:
                self.fertig = True
                return
        self.i_last, self.s_last = i, s
        self.daten.append((t, self.runde, s, y_e, u, px, py))
        if t - self.t_letzte_meldung > 5.0:
            self.t_letzte_meldung = t
            self.get_logger().info(f'Runde {self.runde}  s={s:5.1f}/{self.pfad.laenge:.0f} m  '
                                   f'y_e={y_e:+5.2f} m  u={u:4.2f} m/s')

    def schreiben(self):
        if not self.daten:
            print('Keine Daten -- CSV nicht geschrieben.')
            return None
        ordner = os.path.expanduser(self.a.ordner)
        os.makedirs(ordner, exist_ok=True)
        datei = os.path.join(ordner, f'{time.strftime("%Y%m%d_%H%M%S")}_{self.a.name}.csv')
        with open(datei, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'runde', 's', 'y_e', 'u', 'x', 'y'])
            for d in self.daten:
                w.writerow([f'{d[0]:.3f}', d[1], f'{d[2]:.2f}', f'{d[3]:.3f}',
                            f'{d[4]:.3f}', f'{d[5]:.2f}', f'{d[6]:.2f}'])
        print(f'\nCSV geschrieben: {datei}')
        print(f'Auswertung: python3 analyse_lauf.py {datei} --schub <schub.log>')
        return datei


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runden', type=int, default=5, help='gewertete Runden (nach der Anfahrt)')
    p.add_argument('--name', default='lauf', help='Kennung im Dateinamen, z.B. testlauf0')
    p.add_argument('--einlauf', type=int, default=1,
                   help='zusaetzlich verworfene Runden nach der Anfahrt (Einschwingen)')
    p.add_argument('--ordner', default='~/pfadtests')
    a = p.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = LaufLogger(a)
    try:
        while rclpy.ok() and not node.fertig:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.schreiben()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
