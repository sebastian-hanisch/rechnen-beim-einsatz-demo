"""Die Verfahren des Einsatzes: Greedy, Mehrfachstart, Sampling, Strahlsuche, Symmetrien. Je Verfahren: Tour ist eine Permutation, Länge gleich Neuberechnung, Sonderfälle
(Strahl mit Breite 1 gleich Greedy, Mehrfachstart mit einer Startstadt gleich Greedy), und jedes Verfahren weicht auf mindestens einer Instanz von der Nachbarvariante ab."""

import numpy as np
import pytest
from conftest import tour_len

import rbe_construct as CO
import rbe_instances as I
import rbe_modelle as MO
import rbe_net as N


@pytest.fixture(scope="module")
def inst20():
    seeds = list(range(8))
    X = np.stack([I.instance_unit("gleich", 20, s) for s in seeds])
    return seeds, X, I.distance_matrix_np(X)


def _permutation(t, n):
    return sorted(int(v) for v in t) == list(range(n))


@pytest.mark.parametrize("netz", ["zeiger", "merkmal"])
def test_greedy_mehrfach_sampling_strahl_sind_touren_mit_stimmender_laenge(netz, inst20):
    seeds, X, D = inst20
    P, _ = MO.lade(netz, 0)
    n = 20
    starts = CO.alle_starts(len(seeds), n, 5)
    tours = CO.zeiger_greedy(P, X, starts) if netz == "zeiger" else CO.merkmal_touren(P, D, starts)[0]
    samp = CO.zeiger_gezogen(P, X, 6, seeds, 0) if netz == "zeiger" else CO.merkmal_touren(P, D, np.zeros((len(seeds), 6), dtype=np.int64), "sample", seeds, 0)[0]
    st, leb, _, _ = CO.zeiger_strahl(P, X, 4) if netz == "zeiger" else CO.merkmal_strahl(P, D, 4)
    for t_all in (tours, samp, st):
        L = CO.laenge(X, t_all)
        for i in range(len(seeds)):
            for j in range(t_all.shape[1]):
                if t_all is st and not leb[i, j]:
                    continue
                assert _permutation(t_all[i, j], n)
                assert abs(L[i, j] - tour_len(D[i].tolist(), t_all[i, j].tolist())) < 1e-9
    assert (tours[:, :, 0] == starts).all()


@pytest.mark.parametrize("netz", ["zeiger", "merkmal"])
def test_mehrfachstart_mit_einer_startstadt_und_strahl_breite_1_sind_greedy(netz, inst20):
    seeds, X, D = inst20
    P, _ = MO.lade(netz, 1)
    n = 20
    g = CO.zeiger_greedy(P, X, CO.alle_starts(len(seeds), n, 1)) if netz == "zeiger" else CO.merkmal_touren(P, D, CO.alle_starts(len(seeds), n, 1))[0]
    st, leb, k, h = CO.zeiger_strahl(P, X, 1) if netz == "zeiger" else CO.merkmal_strahl(P, D, 1)
    assert leb.all() and (st == g).all() and k == 190 and h == 19


def test_mehrfachstart_k_touren_sind_die_ersten_k_startstaedte(inst20):
    seeds, X, D = inst20
    P, _ = MO.lade("zeiger", 0)
    alle = CO.zeiger_greedy(P, X, CO.alle_starts(len(seeds), 20))
    drei = CO.zeiger_greedy(P, X, CO.alle_starts(len(seeds), 20, 3))
    assert (alle[:, :3] == drei).all()
    assert (alle[:, :, 0] == np.arange(20)).all()


@pytest.mark.parametrize("netz", ["zeiger", "merkmal"])
def test_sampling_ist_verschachtelt_und_haengt_nicht_von_den_anderen_instanzen_ab(netz, inst20):
    seeds, X, D = inst20
    P, _ = MO.lade(netz, 2)

    def ziehe(idx, k):
        if netz == "zeiger":
            return CO.zeiger_gezogen(P, X[idx], k, [seeds[i] for i in idx], 2)
        return CO.merkmal_touren(P, D[idx], np.zeros((len(idx), k), dtype=np.int64), "sample", [seeds[i] for i in idx], 2)[0]

    acht = ziehe(list(range(8)), 16)
    vier = ziehe([3, 4, 5, 6], 8)
    assert (acht[[3, 4, 5, 6], :8] == vier).all()          # dieselben Touren, egal welche Instanzen mitgerechnet werden und wie viele Touren
    ein = ziehe([5], 16)
    assert (ein[0] == acht[5]).all()


def test_verschiedene_verfahren_weichen_voneinander_ab(inst20):
    """Zweig-Tests (keine Nullspalte): jedes Verfahren mit Wirkung weicht auf mindestens einer Instanz von der Nachbarvariante ab."""
    from rbe_experiments import netz_laengen
    seeds, X, D = inst20
    for netz in MO.NETZE:
        g = netz_laengen(netz, 0, X, D, seeds, "greedy", [1])[1][0]
        s8 = netz_laengen(netz, 0, X, D, seeds, "sampling", [8, 32])
        m20 = netz_laengen(netz, 0, X, D, seeds, "mehrfach", [20])[20][0]
        b8 = netz_laengen(netz, 0, X, D, seeds, "strahl", [8])[8][0]
        assert (s8[8][0] != g).any() and (s8[32][0] != s8[8][0]).any() and (m20 != g).any() and (b8 != g).any()
        assert (s8[8][0] <= s8[8][0] + 1e-12).all()
        assert (s8[32][0] <= s8[8][0] + 1e-12).all()        # mehr Sampling-Touren nie schlechter (gleicher Strom)
        assert (m20 <= g + 1e-12).all()


def test_symmetrien_sind_acht_verschiedene_isometrien_des_einheitsquadrats(inst20):
    seeds, X, D = inst20
    abb = CO.symmetrien(X)
    assert abb.shape == (8,) + X.shape
    for j in range(8):
        assert (abb[j] >= 0).all() and (abb[j] <= 1).all()
        assert np.allclose(I.distance_matrix_np(abb[j]), D, atol=1e-12)
        for k in range(j):
            assert not np.allclose(abb[j], abb[k])


def test_merkmalsnetz_ist_gegen_alle_acht_symmetrien_invariant_zeigernetz_nicht(inst20):
    seeds, X, D = inst20
    abb = CO.symmetrien(X)
    Pm, _ = MO.lade("merkmal", 0)
    Pz, _ = MO.lade("zeiger", 0)
    starts = CO.alle_starts(len(seeds), 20, 3)
    ref_m = CO.merkmal_touren(Pm, D, starts)[0]
    ref_z = CO.zeiger_greedy(Pz, X, starts)
    zeiger_weicht_ab = False
    for j in range(1, 8):
        Dj = I.distance_matrix_np(abb[j])
        assert (CO.merkmal_touren(Pm, Dj, starts)[0] == ref_m).all()
        if (CO.zeiger_greedy(Pz, abb[j], starts) != ref_z).any():
            zeiger_weicht_ab = True
    assert zeiger_weicht_ab                                   # sonst wäre der Regler „Symmetrien“ tot


def test_normalverteilte_gewichte_eines_kleinen_netzes_geben_wahrscheinlichkeiten_die_sich_zu_eins_summieren():
    """Fünf Orte, Mini-Netz von Stück 1: die Wahrscheinlichkeiten der Zeigerschritte summieren sich zu 1, besuchte Orte haben Wahrscheinlichkeit 0."""
    P = N.hand_params()
    x = I.hand_instance()
    p = N.probabilities_along(P, x, [0, 1, 2, 3, 4])
    assert p.shape == (4, 5) and np.allclose(p.sum(axis=1), 1.0)
    assert all(p[k, :k + 1].sum() == 0 for k in range(4))
