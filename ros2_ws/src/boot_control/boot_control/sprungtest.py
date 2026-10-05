#!/usr/bin/env python3
"""
sprungtest.py
=============================================================
Sprungantwort des Kurs- bzw. Fahrtreglers (boat_control_node) messen.

Ablauf
  1. Vorlauf  (--vor s):  Sollkurs = Startkurs, Sollfahrt = --u
  2. Sprung:              kurs  -> Sollkurs + --delta Grad
                          tempo -> Sollfahrt --u1
  3. Nachlauf (--nach s): aufzeichnen, dann Kennwerte ausgeben

Annahmen (beim ersten Lauf pruefen!)
  /cmd/course_safe  geometry_msgs/Twist: linear.x = Sollfahrt [m/s],
                                         angular.z = Sollkurs [rad]
  /state/filtered   nav_msgs/Odometry:   Orientierung -> psi,
                    twist.angular.z = r, twist.linear.x = u
  Kein anderer Knoten darf auf /cmd/course_safe publizieren
  (guidance_ilos_node und course_test_node AUS).

Liegt im Paket boot_control (boot_control/sprungtest.py) und wird
normalerweise von gnc_test_no_perception.launch.py gestartet:
  ros2 launch boot_control gnc_test_no_perception.launch.py sprungtest:=kurs delta:=30
Direkt (nach 'source install/setup.bash'):
  python3 -m boot_control.sprungtest --modus kurs  --delta 30 --u 2.0
  python3 -m boot_control.sprungtest --modus tempo --u 1.5 --u1 3.0
"""

import argparse
import csv
import math
import os
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def yaw_aus_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def kennwerte(t, y, t_sprung, y0, y1, band):
    """Sprungkennwerte fuer eine Antwort y(t) von y0 nach y1."""
    idx = [i for i, ti in enumerate(t) if ti >= t_sprung]
    if len(idx) < 10:
        return None
    tt = [t[i] - t_sprung for i in idx]
    yy = [y[i] for i in idx]
    h = y1 - y0
    norm = [(v - y0) / h for v in yy]

    def erste_zeit(schwelle):
        for ti, n in zip(tt, norm):
            if n >= schwelle:
                return ti
        return None

    t10, t90 = erste_zeit(0.1), erste_zeit(0.9)
    peak = max(norm)
    i_peak = norm.index(peak)
    t_aus = 0.0
    for ti, v in zip(tt, yy):
        if abs(v - y1) > band:
            t_aus = ti
    ende = [v for ti, v in zip(tt, yy) if ti >= tt[-1] - 3.0]
    return {
        't_anstieg_10_90': (t90 - t10) if (t10 is not None and t90 is not None) else None,
        't_90': t90,
        'ueberschwingen_pct': max(0.0, (peak - 1.0) * 100.0),
        'ueberschwingen_abs': max(0.0, (peak - 1.0) * h),
        't_peak': tt[i_peak],
        't_ausregel': t_aus if t_aus < tt[-1] - 0.5 else None,
        'bleibender_fehler': (sum(ende) / len(ende)) - y1,
    }


class Sprungtest(Node):
    def __init__(self, a):
        super().__init__('sprungtest', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.a = a
        self.pub = self.create_publisher(Twist, '/cmd/course_safe', 10)
        self.create_subscription(Odometry, '/state/filtered', self._odom_cb, 20)
        self.t0 = None
        self.psi_start = None
        self.t_last_pub = -1.0
        self.daten = []          # t, psi_c, psi, r, u_c, u, v
        self.fertig = False
        self.get_logger().info(f'Warte auf /state/filtered ... ({a.modus})')

    def _soll(self, t):
        a = self.a
        nach_sprung = t >= a.vor
        if a.modus == 'kurs':
            psi_c = self.psi_start + (math.radians(a.delta) if nach_sprung else 0.0)
            return wrap(psi_c), a.u
        return self.psi_start, (a.u1 if nach_sprung else a.u)

    def _odom_cb(self, msg):
        if self.fertig:
            return
        t_abs = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        psi = yaw_aus_quat(msg.pose.pose.orientation)
        if self.t0 is None:
            self.t0, self.psi_start = t_abs, psi
            self.get_logger().info(f'Start: psi={math.degrees(psi):.1f} Grad')
        t = t_abs - self.t0
        psi_c, u_c = self._soll(t)

        if t - self.t_last_pub >= 0.05:          # 20 Hz Sollwerte
            cmd = Twist()
            cmd.linear.x = float(u_c)
            cmd.angular.z = float(psi_c)
            self.pub.publish(cmd)
            self.t_last_pub = t

        tw = msg.twist.twist
        self.daten.append((t, psi_c, psi, tw.angular.z, u_c, tw.linear.x, tw.linear.y))
        if t >= self.a.vor + self.a.nach:
            self.fertig = True

    # ------------------------------------------------------------------
    def auswerten(self):
        a = self.a
        if not self.daten:
            print('Keine Daten empfangen.')
            return
        os.makedirs(os.path.dirname(os.path.abspath(a.datei)), exist_ok=True)
        with open(a.datei, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'psi_c_deg', 'psi_deg', 'r_degs', 'u_c', 'u', 'v'])
            for t, pc, p, r, uc, u, v in self.daten:
                w.writerow([f'{t:.3f}', f'{math.degrees(pc):.2f}', f'{math.degrees(p):.2f}',
                            f'{math.degrees(r):.2f}', f'{uc:.2f}', f'{u:.3f}', f'{v:.3f}'])

        t = [d[0] for d in self.daten]
        if a.modus == 'kurs':
            # Kurs relativ zum Start, ungewrappt (Sprung bis 170 Grad ok)
            y = [math.degrees(wrap(d[2] - self.psi_start)) for d in self.daten]
            vor = [v for ti, v in zip(t, y) if a.vor - 3.0 <= ti < a.vor]
            y0 = sum(vor) / len(vor) if vor else 0.0
            k = kennwerte(t, y, a.vor, 0.0, a.delta, band=2.0)
            einheit, band_txt = 'Grad', '+-2 Grad'
            r_peak = max(abs(math.degrees(d[3])) for d in self.daten if d[0] >= a.vor)
        else:
            y = [d[5] for d in self.daten]
            vor = [v for ti, v in zip(t, y) if a.vor - 3.0 <= ti < a.vor]
            y0 = sum(vor) / len(vor) if vor else a.u
            k = kennwerte(t, y, a.vor, a.u, a.u1, band=0.1)
            einheit, band_txt = 'm/s', '+-0.1 m/s'
            r_peak = None

        f = lambda x, n=2: '---' if x is None else f'{x:.{n}f}'
        print('\n================ SPRUNGTEST ================')
        if a.modus == 'kurs':
            print(f'Modus kurs: {a.delta:+.0f} Grad bei u={a.u} m/s')
            print(f'Abweichung im Vorlauf (letzte 3 s): {y0:+.2f} Grad')
        else:
            print(f'Modus tempo: {a.u} -> {a.u1} m/s')
            print(f'Ist-Fahrt im Vorlauf (letzte 3 s): {y0:.2f} m/s')
        if k is None:
            print('Zu wenige Daten nach dem Sprung.')
        else:
            print(f'Anstiegszeit 10-90 %   : {f(k["t_anstieg_10_90"])} s')
            print(f'90 % erreicht nach     : {f(k["t_90"])} s')
            print(f'Ueberschwingen         : {f(k["ueberschwingen_pct"], 1)} %  '
                  f'({f(k["ueberschwingen_abs"])} {einheit}) bei t={f(k["t_peak"])} s')
            print(f'Ausregelzeit ({band_txt}): {f(k["t_ausregel"])} s')
            print(f'Bleibender Fehler      : {f(k["bleibender_fehler"])} {einheit}')
        if r_peak is not None:
            print(f'max. Gierrate          : {r_peak:.1f} Grad/s')
        print('\n   t     soll     ist      r[d/s]   u_c    u')
        naechst = max(0.0, a.vor - 3.0)      # Tabelle erst kurz vor dem Sprung
        for (ti, pc, p, r, uc, u, v), yi in zip(self.daten, y):
            if ti >= naechst:
                soll = (a.delta if ti >= a.vor else 0.0) if a.modus == 'kurs' else uc
                print(f'{ti:6.1f} {soll:8.2f} {yi:8.2f} {math.degrees(r):8.1f} {uc:6.2f} {u:6.2f}')
                naechst += 0.5
        print(f'\nCSV: {a.datei}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modus', choices=['kurs', 'tempo'], default='kurs')
    p.add_argument('--delta', type=float, default=30.0, help='Kurssprung [Grad]')
    p.add_argument('--u', type=float, default=2.0, help='Sollfahrt (vor dem Sprung) [m/s]')
    p.add_argument('--u1', type=float, default=3.0, help='Sollfahrt nach dem Sprung (tempo)')
    p.add_argument('--vor', type=float, default=15.0, help='Vorlauf [s]')
    p.add_argument('--nach', type=float, default=20.0, help='Nachlauf [s]')
    p.add_argument('--datei', default='sprung.csv')
    a = p.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Sprungtest(a)
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