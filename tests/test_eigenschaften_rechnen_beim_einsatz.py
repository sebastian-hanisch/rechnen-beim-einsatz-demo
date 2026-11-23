"""Eigenschaftstests (feste Zufalls-Seeds): Wiederholbarkeit, Unabhängigkeit von der Reihenfolge der Instanzen, Monotonie, Zähler, Zweige (keine tote Spalte)."""

import numpy as np
import pytest
from conftest import zufalls_D

import rbe_budget as B
import rbe_experiments as EX
import rbe_live as LV
import rbe_suche as SU
from rbe_rngvec import UStream


def test_gleicher_zufalls_seed_gleiches_ergebnis_anderer_zufalls_seed_ein_anderes():
    a = LV.e2((0, 1, 2, 3), "zeiger", 0, "sampling", 8)
    b = LV.e2((0, 1, 2, 3), "zeiger", 0, "sampling", 8)
    c = LV.e2((4, 5, 6, 7), "zeiger", 0, "sampling", 8)
    d = LV.e2((0, 1, 2, 3), "zeiger", 1, "sampling", 8)
    assert (a["gap_netz"] == b["gap_netz"]).all() and not np.allclose(a["gap_netz"], c["gap_netz"]) and not np.allclose(a["gap_netz"], d["gap_netz"])


@pytest.mark.parametrize("netz", ["zeiger", "merkmal"])
def test_die_reihenfolge_der_instanzen_aendert_nichts(netz):
    vor = LV.e2((0, 1, 2, 3), netz, 0, "sampling", 16)
    rueck = LV.e2((3, 2, 1, 0), netz, 0, "sampling", 16)
    assert np.allclose(vor["gap_netz"], rueck["gap_netz"][::-1], atol=1e-9)
    assert np.allclose(vor["gap_nn"], rueck["gap_nn"][::-1])


def test_mehr_sampling_touren_nie_schlechter_und_mehr_strahlbreite_meist_besser():
    X, D, _ = LV.daten(range(12))
    for netz in ("zeiger", "merkmal"):
        r = EX.netz_laengen(netz, 0, X, D, list(range(12)), "sampling", [1, 2, 4, 8, 16, 32])
        ks = sorted(r)
        assert all((r[b][0] <= r[a][0] + 1e-12).all() for a, b in zip(ks, ks[1:]))
        s = EX.netz_laengen(netz, 0, X, D, list(range(12)), "strahl", [1, 4, 32])
        assert s[32][0].mean() < s[1][0].mean()                      # Breite 32 baut im Mittel kürzere Touren als Breite 1 (kein Gesetz: hier gezählt über 12 Instanzen)


def test_mehrfachstart_ist_nie_schlechter_als_greedy_und_k_gleich_m_mal_190():
    X, D, _ = LV.daten(range(8))
    r = EX.netz_laengen("zeiger", 0, X, D, list(range(8)), "mehrfach", [1, 5, 20])
    assert (r[5][0] <= r[1][0] + 1e-12).all() and (r[20][0] <= r[5][0] + 1e-12).all()
    for k in (1, 5, 20):
        assert EX.kosten_netz("zeiger", "mehrfach", k)[0] == k * B.k_konstruktion(20)
    assert EX.kosten_netz("zeiger", "mehrfach", 20, symm=8)[0] == 8 * 3800
    assert EX.kosten_netz("zeiger", "mehrfach", 20, symm=8)[1] == 8 * B.o_zeiger(20, 20)


def test_suche_mit_groesserem_budget_nie_schlechter_je_instanz():
    for i in range(5):
        ctx = SU.kontext(zufalls_D(20, 40 + i))
        ks, ls = SU.neustart_suche(ctx, UStream(i), 20000)
        werte = SU.beste_bei_raster(ks, ls, [1, 100, 1000, 5000, 20000])
        assert (np.diff(werte) <= 1e-12).all()


def test_zufallstour_als_gegner_liegt_weit_ueber_dem_nachsten_nachbarn():
    """Eine Zufallstour (1 Kandidatenbewertung) ist als Gegner ohne Verbesserung weit schlechter als der Nächste Nachbar; Mittel über 24 Instanzen."""
    zeilen = LV.e1(tuple(range(24)), 0)
    g = {z["art"]: float(z["gap"].mean()) for z in zeilen}
    assert g["zufall"] > 100 and 5 < g["nn"] < 25 and g["zufall"] > 5 * g["nn"]


def test_die_zeiger_und_merkmalsnetze_weichen_voneinander_und_von_nn_ab():
    zeilen = LV.e1(tuple(range(8)), 0)
    gaps = [z["gap"] for z in zeilen]
    for i in range(len(gaps)):
        for j in range(i):
            assert not np.allclose(gaps[i], gaps[j]), (zeilen[i]["name"], zeilen[j]["name"])
