"""Das Handbeispiel (fünf Städte) von Hand nachgerechnet: Wahrheit, Nächster Nachbar, Mini-Netz, Zählung K, exaktes Sampling, Strahlsuche, exakter Gradient der aktiven Suche.

Die Prüfungen stammen aus der Handrechnung der Planung (`handbeispiele_03.py`, gleiche enge Toleranzen); geändert ist nur die Verbesserung: die Zugmenge zählt hier verschiedene Züge
(M(5) = 50), nicht die Doppelzählung der Vormessung (65)."""

import math
from fractions import Fraction

import numpy as np

import rbe_budget as B
import rbe_hand as H
import rbe_net as N  # noqa: F401
from rbe_rng import SplitMix64


def test_wahrheit_held_karp_gleich_aufzaehlung():
    assert H.optimum() == 16 == H.optimum_held_karp()
    opt = sorted(t for t, L in H.laengen().items() if L == H.optimum())
    assert opt == [(0, 1, 4, 3, 2), (0, 2, 3, 4, 1)]
    assert 2 + 4 + 1 + 4 + 5 == 16


def test_naechster_nachbar():
    nn = H.naechster_nachbar(0)
    assert nn == [0, 2, 4, 3, 1] and H.laenge(nn) == 19 == 2 + 3 + 1 + 8 + 5
    assert min(H.laenge(H.naechster_nachbar(s)) for s in range(5)) == 19


def test_mini_netz_greedy_und_schritt_wsk():
    assert H.mini_greedy(0) == [0, 2, 3, 4, 1] and H.laenge(H.mini_greedy(0)) == 16
    assert [H.exponent(0, j, 0) for j in (2, 3, 1, 4)] == [6, 9, 15, 24]
    assert [H.laenge(H.mini_greedy(s)) for s in range(5)] == [16, 19, 19, 19, 16]
    p = H.schritt_wsk(0, {1, 2, 3, 4}, 0)
    assert sum(p.values()) == 1
    q = Fraction(2, 3)
    assert p[2] == q ** 6 / (q ** 6 + q ** 9 + q ** 15 + q ** 24)
    for j, soll in ((2, 0.7559), (3, 0.2240), (1, 0.0197), (4, 0.0005)):
        assert abs(float(p[j]) - soll) < 5e-5


def test_zaehlung_k_von_hand_und_instrumentiert():
    z = B.Zaehler()
    H.konstruktion_gezaehlt(0, z)
    assert z.k == B.k_konstruktion(5) == 10 == 4 + 3 + 2 + 1
    z = B.Zaehler()
    for s in range(5):
        H.konstruktion_gezaehlt(s, z)
    assert z.k == 50 == B.k_mehrfach(5, 5)
    assert (B.k_konstruktion(20), B.k_konstruktion(50), B.k_konstruktion(10)) == (190, 1225, 45)
    assert (B.k_mehrfach(20, 20), B.k_mehrfach(20, 8), B.k_mehrfach(20, 32)) == (3800, 1520, 6080)
    assert (B.k_mehrfach(20, 128), B.k_mehrfach(20, 512), B.k_mehrfach(20, 2048)) == (24320, 97280, 389120)


def test_strahlsuche_zaehlung_von_hand():
    assert B.k_strahl(5, 1) == 10
    assert B.k_strahl(5, 2) == 4 + 2 * 3 + 2 * 2 + 2 * 1 == 16
    assert B.k_strahl(5, 24) == 4 + 4 * 3 + 12 * 2 + 24 * 1 == 64
    assert B.k_strahl(20, 8) == 19 + 8 * sum(range(1, 19)) == 1387
    assert sum(range(1, 19)) == 171
    assert all(B.k_strahl(n, 1) == B.k_konstruktion(n) for n in range(2, 30))
    assert all(B.k_strahl(n, b) <= b * B.k_konstruktion(n) for n in range(2, 25) for b in (1, 2, 5, 40))
    assert [B.k_strahl(20, b) for b in (2, 8, 32, 128, 512)] == [361, 1387, 5257, 19945, 75807]
    assert (B.k_strahl(10, 8), B.k_strahl(50, 8)) == (297, 9457)


def test_strahlsuche_gegen_aufzaehlung_und_instrumentiert():
    h1, c1 = H.strahl_exakt(1)
    assert h1[0][0] == H.mini_greedy(0) and c1 == 10
    h2, c2 = H.strahl_exakt(2)
    assert c2 == B.k_strahl(5, 2) == 16
    h24, c24 = H.strahl_exakt(24)
    assert c24 == B.k_strahl(5, 24) == 64
    wahrscheinlichste = max(H.wsk(), key=lambda t: H.wsk()[t])
    assert tuple(h24[0][0]) == wahrscheinlichste and h24[0][1] == H.wsk()[wahrscheinlichste]
    assert sum(p for _, p in h24) == 1
    assert min(H.laenge(t) for t, _ in h2) == H.optimum()
    assert [t for t, _ in h2] == [[0, 2, 3, 4, 1], [0, 2, 4, 3, 1]]
    assert H.laenge([0, 2, 4, 3, 1]) == 19
    assert all(sorted(t) == list(range(5)) for t, _ in h2)


def test_sampling_exakt_in_bruechen():
    assert sum(H.wsk().values()) == 1 and len(H.wsk()) == 24
    J0 = H.erwartete_laenge()
    assert abs(float(J0) - 17.5773) < 5e-5
    assert abs(float(H.p_optimum()) - 0.5832) < 5e-5
    assert abs(float(H.wsk()[(0, 2, 3, 4, 1)]) - 0.5730) < 5e-5
    E1, E2, E4, E8 = (H.erwartetes_beste(k) for k in (1, 2, 4, 8))
    assert E1 == J0 and E1 > E2 > E4 > E8 >= H.optimum()
    assert [round(float(e), 4) for e in (E2, E4, E8)] == [16.5503, 16.0909, 16.0027]
    assert H.erwartetes_beste_paare() == E2      # Aufzählung aller 576 Paare gleich der Formel


def test_beobachtete_haeufigkeiten_mit_splitmix64():
    rng = SplitMix64(12345)
    anz = 60000
    haeufig, sc1, sc2, ls = {}, [], [], []
    for _ in range(anz):
        t, (g1, g2) = H.stichprobe(rng)
        haeufig[tuple(t)] = haeufig.get(tuple(t), 0) + 1
        sc1.append(g1)
        sc2.append(g2)
        ls.append(H.laengen()[tuple(t)])
    abw = max(abs(haeufig.get(t, 0) / anz - float(w)) / math.sqrt(max(float(w) * (1 - float(w)), 1e-12) / anz) for t, w in H.wsk().items())
    assert abw < 4.5
    assert abs(np.mean(ls) - float(H.erwartete_laenge())) < 4.5 * np.std(ls) / math.sqrt(anz)
    assert len(haeufig) >= 8
    g1, g2 = H.gradient_exakt()
    est1 = (np.array(ls) - np.mean(ls)) * np.array(sc1)
    est2 = (np.array(ls) - np.mean(ls)) * np.array(sc2)
    assert abs(est1.mean() - float(g1)) < 4.5 * est1.std() / math.sqrt(anz)
    assert abs(est2.mean() - float(g2)) < 4.5 * est2.std() / math.sqrt(anz)


def test_exakter_gradient_der_aktiven_suche():
    g1, g2 = H.gradient_exakt()
    assert (round(float(g1), 4), round(float(g2), 4)) == (-0.9727, -2.5068)
    fd1, fd2 = H.gradient_zentral()
    assert abs(fd1 - float(g1)) < 1e-6 * max(1.0, abs(fd1)) and abs(fd2 - float(g2)) < 1e-6 * max(1.0, abs(fd2))
    # der Referenzwert ändert den erwarteten Gradienten nicht (Erwartung der Score-Funktion exakt 0)
    g1_ohne = sum(w * H.laengen()[t] * H.score_exakt(t)[0] for t, w in H.wsk().items())
    assert g1 == g1_ohne
    assert sum(w * H.score_exakt(t)[0] for t, w in H.wsk().items()) == 0
    assert sum(w * H.score_exakt(t)[1] for t, w in H.wsk().items()) == 0
    assert abs(H.j_float(H.W1, H.W2) - float(H.erwartete_laenge())) < 1e-12
    klein = H.j_float(H.W1 - 0.02 * float(g1), H.W2 - 0.02 * float(g2))
    gross = H.j_float(H.W1 - 5.0 * float(g1), H.W2 - 5.0 * float(g2))
    assert abs(klein - 17.4374) < 5e-5 and abs(gross - 18.8803) < 5e-5 and klein < float(H.erwartete_laenge()) < gross


def test_verbesserung_handbeispiel_verschiedene_zuege():
    """Von 0-2-4-3-1 (19) aus endet die Suche im Optimum 16; die Prüfrunde bewertet M(5) = 50 verschiedene Züge (nicht 65 wie in der Vormessung mit doppelt gezählten Einzelorten)."""
    r = H.verbesserung([0, 2, 4, 3, 1])
    assert r["laenge"] == 16 and r["lokalmin"] and r["ev_pruef"] == 50 == B.K0.m_verschiedene_zuege(5)
    assert r["ev"] == r["ev"] and r["zuege"] >= 1
    opt = H.verbesserung([0, 2, 3, 4, 1])
    assert opt["ev"] == 50 and opt["zuege"] == 0              # Beweis des Lokalminimums allein
    # K: S4 = 10 + ev; S2 mit dem Mini-Netz (beste aus 5, schon optimal) = 50 + 50; S5 = 50 + ev der besten NN-Tour
    assert 50 + opt["ev"] == 100
    s4 = B.k_konstruktion(5) + r["ev"]
    assert s4 > 10 + 50
    best_nn = min((H.naechster_nachbar(s) for s in range(5)), key=H.laenge)
    assert H.verbesserung(best_nn)["laenge"] == 16
