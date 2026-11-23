"""Druckt die Planzahlen (Vormessung mit numpy-Strom) neben die Neumessung mit SplitMix64 aus `rbe_band.json` (für Bericht und README, Tabelle „Befunde und Korrekturen“).

Aufruf:  python tools/rbe_planzahlen.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import rbe_band as BD  # noqa: E402
import rbe_befunde as BF  # noqa: E402
import rbe_budget as B  # noqa: E402
import rbe_evaluation as V  # noqa: E402


def zeile(name, plan, bau):
    print(f"| {name} | {plan} | {bau} |")


def main() -> int:
    band = BD.lade_band()
    de = V.de
    print("| Größe | Planzahl (Vormessung) | Bau |\n|---|---|---|")
    ez, em = BF.eine_tour(band, "zeiger"), BF.eine_tour(band, "merkmal")
    zeile("Zeigernetz Greedy, eine Tour (Mittel der 5 Seeds)", "9,76 %", f"{de(ez['gap_netz'])} % (Seed-Mittel {de(min(ez['seed_mittel']))} bis {de(max(ez['seed_mittel']))})")
    zeile("Merkmalsnetz Greedy, eine Tour", "8,60 %", f"{de(em['gap_netz'])} % (Seed-Mittel {de(min(em['seed_mittel']))} bis {de(max(em['seed_mittel']))})")
    zeile("Nächster Nachbar, eine Tour", "17,40 %", f"{de(ez['gap_nn'])} %")
    zeile("Zufallstour (Mittel einer Tour)", "173,30 %", f"{de(BD.mittel(BD.klass_zeile('zufall', 1, band)['gap']))} %")
    zeile("Faktor in O: Zeigernetz / Merkmalsnetz je Tour", "613 / 119", f"{de(ez['faktor_o'], 0)} / {de(em['faktor_o'], 0)}")
    zeile("Wechselkurs (Deltas je Netz-Bewertung): Zeigernetz / Merkmalsnetz", "161 / 31,3", f"{de(ez['wechselkurs'], 1)} / {de(em['wechselkurs'], 1)}")
    for netz, plan_m, plan_n in (("zeiger", "2,04", "5,69"),):
        g = BF.gleiche_konstruktionen(band, netz, "mehrfach", 20)
        zeile("Mehrfach-Greedy aus 20 Startstädten (Zeigernetz) / Nächster Nachbar aus 20 Starts", f"{plan_m} / {plan_n} %", f"{de(g['gap_netz'])} / {de(g['gap_nn'])} %")
    for k, plan in ((8, "6,62"), (32, "3,94"), (128, "2,54"), (512, "1,67"), (2048, "1,05")):
        zeile(f"Sampling k = {k} (Zeigernetz)", f"{plan} %", f"{de(BD.mittel(BD.netz_zeile('zeiger', 'sampling', k, band)['gap']))} %")
    for k, plan in ((8, "5,26"), (32, "3,53"), (128, "2,25"), (512, "1,41")):
        zeile(f"Strahlsuche Breite {k} (Zeigernetz)", f"{plan} %", f"{de(BD.mittel(BD.netz_zeile('zeiger', 'strahl', k, band)['gap']))} % bei {V.zahl(B.k_strahl(20, k))} K")
    xs, ys = BD.neustart_mittelkurve(band)
    for b, plan in ((128, "43,57"), (512, "4,22"), (2048, "1,75"), (8192, "0,22"), (32768, "0,00")):
        i = int(np.searchsorted(xs, b))
        zeile(f"Neustart-Suche bei {V.zahl(b)} K", f"{plan} %", f"{de(float(ys[i]))} %")
    for nm, kx, plan in (("Nächster Nachbar + Verbesserung (S4)", "S4", "0,469 % bei 3 632 K"), ("Farthest Insertion + Verbesserung (S6)", "S6", "0,415 % bei 3 097 K")):
        z = BF.klass_start(band, kx)
        zeile(nm, plan, f"{de(z['gap'], 3)} % bei {V.zahl(z['K'])} K ({V.zahl(z['ev'])} Zugauswertungen)")
    for netz, plan in (("zeiger", "0,257 % bei 6 516 K"), ("merkmal", "0,125 % bei 6 341 K")):
        z = BF.start_einzel(band, netz, "S2")
        zeile(f"{V.NETZ_NAME[netz]} S2 (beste aus 20, dann Verbesserung)", plan, f"{de(z['gap'], 3)} % bei {V.zahl(z['K'])} K ({V.zahl(z['ev'])} Zugauswertungen, {de(z['O'] / 1e6, 3)} Mio O)")
    sl = BF.startloesung(band, "merkmal")
    zeile("Anteil der Prüfrunde an den Zugauswertungen von S4 (n = 20)", "64 % (2 210, doppelt gezählt)", f"{de(sl['anteil_pruefrunde'], 0)} % (1 850 Züge)")
    for steps, plan in ((8, "2,05 gegen 2,58 %"), (32, "1,11 gegen 1,66 %"), (128, "0,78 gegen 1,04 % (2 048 Touren)")):
        a = BF.aktive_suche(band, steps, "0.001")
        zeile(f"Aktive Suche {a['touren']} Touren, Lernrate 0,001", plan + " (Plan: 128 / 512 / 2 048 Touren)", f"{de(a['gap_aktiv'])} gegen {de(a['gap_sampling'])} %")
    a = BF.aktive_suche(band, 128, "0.01")
    zeile("Aktive Suche 2 048 Touren, Lernrate 0,01", "6,60 gegen 1,93 % (Abstimmung, 20 Instanzen)", f"{de(a['gap_aktiv'])} gegen {de(a['gap_sampling'])} %")
    for netz, verf, k, plan in (("zeiger", "sampling", 8, "rund 1 240"), ("zeiger", "mehrfach", 20, "rund 560"), ("zeiger", "sampling", 512, "rund 8 400")):
        w = BF.wechselkurs(band, netz, verf, k)
        zeile(f"c* Zeigernetz, {BF.VERFAHREN_NAME[verf]} k = {k}", plan, "außerhalb des gemessenen Bereichs" if w["c_stern"] is None else f"{V.zahl(w['c_stern'])}")
    sy = BF.symmetrien(band)
    zeile("Symmetrien ×8 (Zeigernetz, Mehrfachstart 20)", "nicht gemessen", f"{de(sy['gap_acht'])} gegen {de(sy['gap_eins'])} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
