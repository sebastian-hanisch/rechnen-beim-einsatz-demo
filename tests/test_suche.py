"""Die klassische Seite: Verbesserung (2-opt + Or-opt), Prüfrunde, Neustart-Suche, Startlösungen. Prüfungen an Mini-Instanzen und gegen einen zweiten Rechenweg (Zugliste in Listen)."""

import numpy as np
import pytest
from conftest import kanon, tour_len, zuege_listen, zufalls_D

import rbe_budget as B
import rbe_hand as H
import rbe_suche as SU
from rbe_rngvec import UStream


def _lokalminimum_unabhaengig(D, t) -> bool:
    """Weg 1: jede Nachbartour der Zugliste mit voller Neuberechnung der Länge."""
    D = D.tolist()
    L0 = tour_len(D, list(t))
    return all(tour_len(D, z) >= L0 - 1e-12 for z in zuege_listen(list(t)))


@pytest.mark.parametrize("n", [5, 6, 8, 12, 20, 50])
def test_pruefrunde_ist_m_von_n_verschiedenen_zuegen(n):
    ctx = SU.kontext(zufalls_D(n, n))
    assert ctx.nb.M == B.K0.m_verschiedene_zuege(n) == n * (11 * n - 35) // 2 == SU.pruefrunde(n)
    if n <= 20:
        t = list(range(n))
        assert len({kanon(z) for z in zuege_listen(t)}) <= len(zuege_listen(t)) == ctx.nb.M


def test_pruefrunde_n20_und_n50_exakte_zahlen():
    assert SU.pruefrunde(20) == 1850 and SU.pruefrunde(50) == 12875
    assert SU.pruefrunde(5) == 50


@pytest.mark.parametrize("n,seed", [(6, 1), (8, 2), (12, 3), (20, 4)])
def test_verbesserung_endet_im_lokalminimum_mit_gleicher_fortgeschriebener_laenge(n, seed):
    D = zufalls_D(n, seed)
    ctx = SU.kontext(D)
    t0 = UStream(seed).random_order(1, n)[0]
    r = SU.verbessere(ctx, t0)
    assert sorted(r["tour"].tolist()) == list(range(n))
    assert abs(r["laenge"] - tour_len(D.tolist(), r["tour"].tolist())) < 1e-9
    assert r["lokalmin"] and _lokalminimum_unabhaengig(D, r["tour"])
    assert r["laenge"] <= tour_len(D.tolist(), t0.tolist()) + 1e-12
    assert r["ev_pruef"] == ctx.nb.M
    assert r["ev"] >= r["ev_pruef"]


def test_gegenprobe_eine_zufallstour_ist_kein_lokalminimum():
    D = zufalls_D(12, 7)
    t = UStream(7).random_order(1, 12)[0]
    assert not _lokalminimum_unabhaengig(D, t)


def test_zweite_runde_findet_nichts_und_kostet_genau_eine_pruefrunde():
    D = zufalls_D(10, 9)
    ctx = SU.kontext(D)
    r = SU.verbessere(ctx, UStream(9).random_order(1, 10)[0])
    r2 = SU.verbessere(ctx, r["tour"])
    assert r2["zuege"] == 0 and r2["ev"] == ctx.nb.M and (r2["tour"] == r["tour"]).all()


def test_naechster_nachbar_und_farthest_insertion_sind_touren():
    D = zufalls_D(15, 5)
    for t in (SU.nn_tour(D, 3), SU.farthest_insertion(D, 0)):
        assert sorted(t.tolist()) == list(range(15))
    assert SU.nn_tour(D, 3)[0] == 3


def test_neustart_suche_ist_monoton_zaehlt_k_und_haelt_das_budget():
    D = zufalls_D(12, 11)
    ctx = SU.kontext(D)
    ks, ls = SU.neustart_suche(ctx, UStream(1), 3000)
    assert (np.diff(ks) > 0).all() and ks[0] == 1
    grid = [1, 50, 200, 1000, 3000]
    werte = [SU.beste_bei(ks, ls, b) for b in grid]
    assert all(a >= b - 1e-12 for a, b in zip(werte, werte[1:]))                # mehr Budget nie schlechter
    assert np.allclose(SU.beste_bei_raster(ks, ls, grid), werte)
    # jeder Ereignis-Zeitpunkt gehört zu einer Tour, deren Länge nachgerechnet stimmt: die Neustarts sind Zufallstouren (1 K), die Züge verkürzen
    assert ks[0] == 1 and ls[0] > 0


def test_neustart_suche_kleiner_als_eine_tour_budget():
    D = zufalls_D(8, 1)
    ctx = SU.kontext(D)
    ks, ls = SU.neustart_suche(ctx, UStream(2), 1)
    assert list(ks) == [1] and len(ls) == 1                                         # die erste Zufallstour wird gewertet
    assert SU.beste_bei(ks, ls, 0) == float("inf")


def test_handbeispiel_ev_zaehlung_von_hand():
    """Hand: von 0-2-4-3-1 aus; die Prüfrunde auf der optimalen Tour bewertet M(5) = 50 Züge; die Endtour hat Länge 16."""
    ctx = SU.kontext(H.distanzen())
    r = SU.verbessere(ctx, np.array([0, 2, 3, 4, 1]))
    assert r["zuege"] == 0 and r["ev"] == 50 and r["laenge"] == 16.0
    assert _lokalminimum_unabhaengig(H.distanzen(), r["tour"])


def _durchlauf_referenz(D, t0):
    """Zweiter Rechenweg: der Durchlauf in reinem Python mit der Zugliste aus Listenoperationen und voller Neuberechnung jeder Nachbartour (kein numpy-Indexieren, keine Kantenformel).
    2-opt-Phase: Durchläufe über die 2-opt-Züge, der Durchlauf geht hinter dem angenommenen Zug weiter; Or-opt-Phase: von vorn, danach wieder 2-opt."""
    Dl = D.tolist()
    t = list(t0)
    n = len(t)
    m2 = n * (n - 3) // 2
    ev, p, verb, phase = 0, 0, False, "2opt"
    while True:
        zl = zuege_listen(t)
        l0 = tour_len(Dl, t)
        von, bis = (p, m2) if phase == "2opt" else (m2, len(zl))
        gefunden = None
        for idx in range(von, bis):
            ev += 1
            if tour_len(Dl, zl[idx]) < l0 - 1e-12:
                gefunden = idx
                break
        if gefunden is not None:
            t = zl[gefunden]
            if phase == "2opt":
                p, verb = gefunden + 1, True
            else:
                phase, p, verb = "2opt", 0, False
            continue
        if phase == "or":
            return t, ev
        if verb:
            p, verb = 0, False
        else:
            phase = "or"


@pytest.mark.parametrize("n,seed", [(6, 2), (8, 5), (10, 7), (12, 9)])
def test_durchlauf_gleicht_der_referenz_in_reinem_python(n, seed):
    D = zufalls_D(n, seed)
    t0 = UStream(seed).random_order(1, n)[0]
    ref_tour, ref_ev = _durchlauf_referenz(D, t0)
    r = SU.durchlauf(SU.kontext(D), t0)
    assert r["ev"] == ref_ev and abs(r["laenge"] - tour_len(D.tolist(), ref_tour)) < 1e-9
    assert r["tour"].tolist() == ref_tour                                           # sogar dieselbe Reihenfolge der Orte (gleiche Zugliste, gleiche Drehung)
    assert r["lokalmin"] and not r["abgebrochen"] and _lokalminimum_unabhaengig(D, r["tour"])


def test_durchlauf_ereignisse_stimmen_mit_der_neuberechnung_ueberein():
    D = zufalls_D(14, 3)
    ctx = SU.kontext(D)
    t0 = UStream(3).random_order(1, 14)[0]
    prot = []
    r = SU.durchlauf(ctx, t0, protokoll=prot)
    laenge = tour_len(D.tolist(), t0.tolist())
    for e in prot:
        laenge += e["delta"]
        assert abs(laenge - tour_len(D.tolist(), e["tour"].tolist())) < 1e-9 and e["delta"] < 0
        assert sorted(e["tour"].tolist()) == list(range(14))
    assert abs(laenge - r["laenge"]) < 1e-9 and [e["ev"] for e in prot] == sorted(e["ev"] for e in prot)
    assert prot[-1]["ev"] + (r["ev"] - prot[-1]["ev"]) == r["ev"] and r["ev"] - prot[-1]["ev"] >= r["ev_pruef"]       # nach dem letzten Zug: Rest des Durchlaufs und ein voller Durchlauf


def test_durchlauf_grenze_bricht_ab_und_zaehlt_hoechstens_die_grenze():
    D = zufalls_D(12, 4)
    ctx = SU.kontext(D)
    t0 = UStream(4).random_order(1, 12)[0]
    voll = SU.durchlauf(ctx, t0)
    for grenze in (0, 5, 50, voll["ev"] - 1):
        r = SU.durchlauf(ctx, t0, grenze=grenze)
        assert r["abgebrochen"] and r["ev"] == grenze and not r["lokalmin"]
    assert not SU.durchlauf(ctx, t0, grenze=voll["ev"])["abgebrochen"]


def test_beweis_auf_lokalminimum_kostet_genau_eine_pruefrunde():
    D = zufalls_D(10, 11)
    ctx = SU.kontext(D)
    r = SU.verbessere(ctx, UStream(11).random_order(1, 10)[0])
    r2 = SU.verbessere(ctx, r["tour"])
    assert r2["zuege"] == 0 and r2["ev"] == ctx.nb.M == SU.pruefrunde(10)


@pytest.mark.parametrize("n", [5, 8, 20, 50])
def test_die_2opt_zuege_stehen_am_anfang_der_zugliste(n):
    """Der Durchlauf trennt 2-opt und Or-opt an der Stelle n (n - 3) / 2 der Zugliste; die Arten der Zugliste (0 = 2-opt) müssen dazu passen."""
    ctx = SU.kontext(zufalls_D(n, n))
    m2 = n * (n - 3) // 2
    assert (ctx.nb.art[:m2] == 0).all() and (ctx.nb.art[m2:] != 0).all() and int((ctx.nb.art == 0).sum()) == m2
