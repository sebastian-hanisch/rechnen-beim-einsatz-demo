"""Zählregel: K (Kandidatenbewertungen) und O (grobe Operationen) von Hand, aus den Formen der Matrixprodukte und aus instrumentierten Läufen."""

import math

import numpy as np
from conftest import zufalls_instanz

import rbe_budget as B
import rbe_construct as CO
import rbe_modelle as MO


def _matmul_zeiger_kodierer(n, d=32):
    """Matrixprodukte des Kodierers (eine Schicht, ein Kopf) einzeln aufgeführt (zweiter Rechenweg zu `rbe_cost.ops_kodierer`)."""
    produkte = [(n, 2, d), (n, d, d), (n, d, d), (n, d, d), (n, d, n), (n, n, d), (n, d, d), (n, d, 2 * d), (n, 2 * d, d), (n, d, d)]
    return sum(m * k * r for m, k, r in produkte)


def _dekodierer(n, d=32):
    """Eine Tour: zwei Produkte für den Kontext der Startstadt (je d^2), je Schritt ein Produkt mit Wl (d^2) und ein Skalarprodukt je Ort (n d), dazu 4 n Operationen je Schritt."""
    return 2 * d * d + (n - 1) * (d * d + n * d) + (n - 1) * 4 * n


def test_k_von_hand_n20():
    assert B.k_konstruktion(20) == 190 == sum(range(1, 20))
    assert B.k_aktiv(20, 8, 16) == 24320 and B.k_aktiv(20, 32, 16) == 97280


def test_hypothesen_und_strahl_formel_n_klein():
    assert B.hypothesen_je_schritt(5, 2) == [1, 2, 2, 2]
    assert B.hypothesen_je_schritt(5, 24) == [1, 4, 12, 24]
    assert B.summe_hypothesen(5, 1) == 4
    assert B.summe_hypothesen(5, 24) == 1 + 4 + 12 + 24


def test_summe_hypothesen_n20_breite8_von_hand():
    h = B.hypothesen_je_schritt(20, 8)
    assert h[:3] == [1, 8, 8] and all(x == 8 for x in h[1:]) and len(h) == 19
    assert B.summe_hypothesen(20, 8) == 1 + 18 * 8


def test_o_kodierer_und_dekodierer_aus_den_produkten():
    assert B.o_kodierer(20) == _matmul_zeiger_kodierer(20) == 211200
    assert B.o_kodierer(50) == _matmul_zeiger_kodierer(50) == 624000
    assert B.o_dekodierer(20) == _dekodierer(20) == 35184
    assert B.o_dekodierer(50) == _dekodierer(50)
    assert B.o_zeiger(20, 1) == 246384
    assert B.o_zeiger(20, 8) == 211200 + 8 * 35184


def test_strahl_o_mit_breite_1_gleich_einer_tour():
    for n in (6, 20, 50):
        assert B.o_zeiger_strahl(n, 1) == B.o_zeiger(n, 1)
        assert math.isclose(B.o_merkmal_strahl(n, 1), B.o_merkmal(n, 1), rel_tol=1e-12)
    assert B.o_zeiger_strahl(20, 8) > B.o_zeiger(20, 8) - 7 * 35184 + 0     # Strahl 8 kostet weniger als 8 Touren (erster Schritt hat nur eine Hypothese)
    assert B.o_zeiger_strahl(20, 8) < B.o_zeiger(20, 8)


def test_merkmalsnetz_o_von_hand():
    n = 20
    bau = 2 * n * n + n * (8 + math.log2(n))
    assert math.isclose(B.o_merkmalsbildung_schritt(n), bau)
    je_tour = (n - 1) * bau + 190 * 144
    assert round(je_tour) == 47242                      # Merkmalsbildung 19 882 + 190 Zeilen zu 144 = 27 360
    assert round(B.o_merkmal(20, 1)) == 47642           # plus n^2 für den Maßstab s
    assert round(B.o_merkmal(50, 1) - 2500) == 454827


def test_nn_und_andere_gegner_in_o():
    assert B.o_nn(20) == 1520 and B.o_nn(50) == 8 * 1225
    assert B.o_farthest(20) == 1600
    assert B.o_zufall(1) == 8
    assert B.o_suche(1000) == 8000 and B.o_suche(1000, 3) == 3000


def test_wechselkurs_und_c_stern():
    w = B.wechselkurs(B.o_zeiger(20, 1), 190)
    assert round(w, 1) == 162.1                          # eine Netz-Bewertung kostet 162 Deltas (Zeigernetz, eine Tour mit Kodierer)
    assert round(B.wechselkurs(B.o_zeiger(20, 1000) - B.o_kodierer(20), 190 * 1000), 1) == 23.1   # Kodierer auf viele Touren verteilt: 35 184 / 1 520
    assert round(B.wechselkurs(B.o_merkmal(20, 1), 190), 1) == 31.3
    assert B.c_stern(484480, 400) == 1211.2


def test_o_aktiv_von_hand():
    assert B.o_aktiv(20, 1, 16) == 3 * (211200 + 16 * 35184)
    assert B.o_aktiv(20, 8, 16) == 8 * B.o_aktiv(20, 1, 16)


def test_instrumentierter_lauf_zaehlt_jede_zeile_des_merkmalsnetzes():
    """Die Zahl der Merkmalszeilen aus `rollout` (jede bewertete Zeile) ist n (n - 1) / 2 je Tour."""
    P, _ = MO.lade("merkmal", 0)
    X = np.stack([zufalls_instanz(8, s) for s in range(3)])
    import rbe_instances as I
    D = I.distance_matrix_np(X)
    tours, zeilen = CO.merkmal_touren(P, D, CO.alle_starts(3, 8, 8))
    assert zeilen == 3 * 8 * B.k_konstruktion(8)
