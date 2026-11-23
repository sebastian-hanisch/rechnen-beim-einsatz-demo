"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Größen, Breite größer als die Zahl der Folgen, k = 1, Budget kleiner als eine Konstruktion,
Instanzzahl unter 4, Lücke 0 und negative Lücke, Gleichstände."""

import numpy as np
import pytest
from conftest import zufalls_D

import rbe_budget as B
import rbe_construct as CO
import rbe_evaluation as V
import rbe_experiments as EX
import rbe_hand as H
import rbe_instances as I
import rbe_modelle as MO
import rbe_stats as ST
import rbe_suche as SU
from rbe_rngvec import UStream


@pytest.mark.parametrize("n", [3, 4, 5])
def test_kleinste_groessen_alle_verfahren_liefern_touren(n):
    """Die Netze sind bei n = 20 trainiert, rechnen aber auf jeder Größe; n = 3 hat nur eine Rundtour bis auf Umkehrung, nie ein Fehler."""
    X = np.stack([I.instance_unit("gleich", n, s) for s in range(3)])
    D = I.distance_matrix_np(X)
    for netz in MO.NETZE:
        P, _ = MO.lade(netz, 0)
        g = CO.zeiger_greedy(P, X, CO.alle_starts(3, n)) if netz == "zeiger" else CO.merkmal_touren(P, D, CO.alle_starts(3, n))[0]
        st, leb, k, h = CO.zeiger_strahl(P, X, 4) if netz == "zeiger" else CO.merkmal_strahl(P, D, 4)
        for t in list(g.reshape(-1, n)) + [st[i, j] for i in range(3) for j in range(st.shape[1]) if leb[i, j]]:
            assert sorted(t.tolist()) == list(range(n))
        assert k == B.k_strahl(n, 4) and h == B.summe_hypothesen(n, 4)


def test_strahl_breite_groesser_als_zahl_der_folgen():
    """n = 4 hat 3! = 6 Folgen: Breite 50 hält nur 6 Hypothesen am Leben, der Rest ist nicht lebend (nie in der Auswahl der besten)."""
    X = np.stack([I.instance_unit("gleich", 4, 1)])
    P, _ = MO.lade("zeiger", 0)
    st, leb, k, h = CO.zeiger_strahl(P, X, 50)
    assert int(leb.sum()) == 6 and k == B.k_strahl(4, 50) == 3 + 3 * 2 + 6 * 1 and h == 1 + 3 + 6
    L, t = CO.bester_der_strahlen(X, st, leb)
    assert sorted(t[0].tolist()) == [0, 1, 2, 3] and np.isfinite(L).all()


def test_sampling_k_gleich_1_und_mehrfachstart_k_gleich_1_sind_je_eine_tour():
    X = np.stack([I.instance_unit("gleich", 20, s) for s in range(3)])
    D = I.distance_matrix_np(X)
    r = EX.netz_laengen("zeiger", 0, X, D, [0, 1, 2], "sampling", [1])
    assert r[1][1].shape == (3, 20)
    g = EX.netz_laengen("zeiger", 0, X, D, [0, 1, 2], "mehrfach", [1])
    assert g[1][1][:, 0].tolist() == [0, 0, 0]


def test_budget_kleiner_als_eine_konstruktion_die_erste_tour_wird_gewertet():
    ctx = SU.kontext(zufalls_D(12, 3))
    ks, ls = SU.neustart_suche(ctx, UStream(1), 0.5)
    assert list(ks) == [1] and SU.beste_bei(ks, ls, 0.5) == float("inf") and SU.beste_bei(ks, ls, 1) == ls[0]


def test_luecke_null_und_negative_luecke():
    assert EX.luecke(np.array([5.0]), np.array([5.0]))[0] == 0.0
    assert EX.luecke(np.array([4.9]), np.array([5.0]))[0] == pytest.approx(-2.0)        # Referenz unterboten (bei n = 50 ist die Referenz nur eine Untergrenze der Lücke)
    assert V.de(-0.0) == "0,00"
    cm = ST.compare(np.zeros(6), np.zeros(6))
    assert cm["v"] == 0 and cm["klasse"] == "gleich"                                      # Differenz 0: kein Nenner, kein Fehler


def test_zu_wenig_instanzen_fuer_ein_urteil():
    for fn_args in ((V.verdict_eine_tour, (-1.0, -2.0, -0.5, 3, "zeiger", 190, 2.0, 1.0)),):
        art, text = fn_args[0](*fn_args[1])
        assert art == "info" and "Mit 3 Instanzen" in text
    cm = ST.compare(np.array([1.0]), np.array([2.0]))
    assert cm["lo"] is None                                                              # eine Instanz: kein Intervall


def test_gleichstaende_im_zeigernetz_greedy_waehlen_den_kleinsten_index():
    """Zwei gleiche Orte: das Netz bewertet sie identisch, die Greedy-Tour nimmt den kleineren Index."""
    X = np.array([[[0.1, 0.1], [0.5, 0.5], [0.5, 0.5], [0.9, 0.2], [0.3, 0.8]]])
    P, _ = MO.lade("zeiger", 0)
    t = CO.zeiger_greedy(P, X, np.zeros((1, 1), dtype=np.int64))[0, 0].tolist()
    assert t.index(1) < t.index(2)


def test_handbeispiel_ohne_instanz_zufall():
    """n = 5 braucht keine Instanzen aus dem Strom: alles Brüche und Aufzählung."""
    assert H.optimum() == H.optimum_held_karp() == 16
    assert H.verbesserung([0, 1, 4, 3, 2])["ev"] == 50           # die optimale Tour in der anderen Richtung: nur die Prüfrunde
