"""Orakel: unabhängige Rechenwege für die Kernroutinen (ORACLE-PLAYBOOK §2): Gradient der aktiven Suche gegen endliche Differenzen, Held-Karp gegen Aufzählung, Strahlsuche gegen eine
zweite Umsetzung und gegen die Aufzählung aller Folgen, Sampling gegen die exakte Verteilung, Verbesserung gegen Zugliste und volle Neuberechnung.

Die Orakel sind klein (je unter 10 Sekunden); die Messungen mit vielen Instanzen stehen in der eingefrorenen Messreihe."""

import itertools
import math

import numpy as np
import pytest
from conftest import tour_len, zufalls_D, zufalls_instanz

import rbe_active as AK
import rbe_budget as B
import rbe_construct as CO
import rbe_exact as E
import rbe_instances as I
import rbe_modelle as MO
import rbe_net as N
import rbe_suche as SU
from rbe_rngvec import UStream


# ---------------------------------------------------------------------------------------------------------------- O1 Gradient
def test_gradient_der_aktiven_suche_gegen_zentrale_differenzen():
    """Form der aktiven Suche (eine Instanz, 16 gezogene Touren ab Stadt 0, d = 8, n = 6): der Rückwärtspass gleicht zentralen Differenzen (relative Abweichung < 1e-5)."""
    n, d, K = 6, 8, 16
    P = N.init_params(d, UStream(5))
    X = zufalls_instanz(n, 3)[None]
    starts = np.zeros((1, K), dtype=np.int64)
    H, _ = N.encode(P, X)
    tours = N.decode(P, H, starts, mode="sample", u_stream=UStream(7))["tours"]
    L = N.tour_lengths(X, tours)
    w = (L - L.mean()) / L.size
    F, G = N.loss_and_grad(P, X, starts, w, tours)

    def f(Pn):
        Hn, _ = N.encode(Pn, X, keep=False)
        return float((w * N.decode(Pn, Hn, starts, mode="forced", forced=tours, keep=False)["logp"]).sum())

    assert abs(F - f(P)) < 1e-12
    groesste = max(float(np.abs(g).max()) for g in G.values())
    maxrel = 0.0
    rng = UStream(11)
    for name, g in G.items():
        idx = np.unravel_index(int(rng.uniform(1)[0] * g.size) % g.size, g.shape) if g.ndim else ()
        h = 1e-6
        Pp, Pm = N.copy_params(P), N.copy_params(P)
        Pp[name][idx] += h
        Pm[name][idx] -= h
        fd = (f(Pp) - f(Pm)) / (2 * h)
        maxrel = max(maxrel, abs(fd - g[idx]) / (groesste + 1e-30))
    assert maxrel < 1e-5, maxrel


def test_aktive_suche_veraendert_alle_gewichte_und_das_anfangsmodell_bleibt():
    P0 = MO.lade_zeiger(0)[0]
    kopie = N.copy_params(P0)
    x = I.instance_unit("gleich", 20, 4)
    r = AK.aktive_suche(P0, x, 3, 16, 1e-3, UStream(2))
    assert all((P0[k] == kopie[k]).all() for k in P0)                          # Anfangsmodell unverändert
    assert r["gezogen"] == 48 and len(r["beste"]) == 3 and (np.diff(r["beste"]) <= 1e-15).all()
    assert abs(r["laenge"] - tour_len(I.distance_matrix_np(x).tolist(), r["tour"].tolist())) < 1e-9
    assert sorted(r["tour"].tolist()) == list(range(20))


def test_sampling_gleicher_tourzahl_ist_der_vergleich():
    P0 = MO.lade_zeiger(0)[0]
    x = I.instance_unit("gleich", 20, 4)
    s = AK.sampling_gleiche_touren(P0, x, 4, 8, UStream(3))
    assert s["gezogen"] == 32 and len(s["beste"]) == 4 and (np.diff(s["beste"]) <= 1e-15).all()


# ---------------------------------------------------------------------------------------------------------------- O2 Held-Karp
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_held_karp_gleich_aufzaehlung(n):
    D = zufalls_D(n, n)
    Dl = D.tolist()
    best = min(tour_len(Dl, [0] + list(p)) for p in itertools.permutations(range(1, n)))
    assert abs(E.held_karp(D) - best) < 1e-9


def test_held_karp_gegen_cp_sat_an_instanzen_n20():
    ortools = pytest.importorskip("ortools.sat.python.cp_model")
    cp = ortools
    for seed in (0, 1, 2):
        x = I.instance_unit("gleich", 20, seed)
        D = I.distance_matrix_np(x)
        skala = 10 ** 6
        m = cp.CpModel()
        n = 20
        lit = {}
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    lit[i, j] = m.NewBoolVar(f"a{i}_{j}")
                    arcs.append((i, j, lit[i, j]))
        m.AddCircuit(arcs)
        m.Minimize(sum(int(round(D[i, j] * skala)) * lit_var for (i, j), lit_var in lit.items()))
        s = cp.CpSolver()
        s.parameters.max_time_in_seconds = 20
        s.parameters.num_workers = 1
        assert s.Solve(m) == cp.OPTIMAL
        assert abs(s.ObjectiveValue() / skala - E.held_karp(D)) < 5e-5


def test_wahrheit_datei_stimmt_mit_held_karp_ueberein():
    import rbe_band as BD
    w = BD.wahrheit_n20()
    for i in (0, 31, 150):
        assert abs(E.held_karp(I.distance_matrix_np(I.instance_unit("gleich", 20, i))) - w[i]) < 1e-8


# ---------------------------------------------------------------------------------------------------------------- O5 Strahlsuche
def _politik_tabelle(n: int, seed: int) -> np.ndarray:
    """Eine feste Politik: Gewichte w[i][j] aus dem Strom; p(j | i, besucht) proportional zu exp(w[i][j]) über die offenen Orte."""
    return UStream(seed).normal((n, n))


def _strahl_referenz(W: np.ndarray, n: int, breite: int):
    """Zweite Umsetzung der Strahlsuche in reinem Python (Listen, keine Vektoren): (Endhypothesen [(Tour, logp)], K, Summe der Hypothesen)."""
    hyp = [([0], 0.0)]
    k = h = 0
    for t in range(1, n):
        neu = []
        for tour, lp in hyp:
            offen = [j for j in range(n) if j not in tour]
            k += len(offen)
            z = math.log(sum(math.exp(W[tour[-1]][j]) for j in offen))
            for j in offen:
                neu.append((tour + [j], lp + W[tour[-1]][j] - z))
        h += len(hyp)
        neu.sort(key=lambda e: -e[1])          # stabil: bei Gleichstand bleibt die Reihenfolge der Erzeugung (Hypothese, dann Ort)
        hyp = neu[:breite]
    return hyp, k, h


@pytest.mark.parametrize("n,breite", [(5, 1), (5, 2), (6, 3), (6, 7), (7, 50), (6, 5000)])
def test_strahlsuche_gleicht_der_zweiten_umsetzung(n, breite):
    W = _politik_tabelle(n, 100 + n)

    def logp(cur, besucht, t):
        roh = W[cur]                                                             # (m, B, n)
        roh = np.where(besucht, -np.inf, roh)
        mx = np.where(np.isfinite(roh.max(-1, keepdims=True)), roh.max(-1, keepdims=True), 0.0)
        return roh - (np.log(np.exp(roh - mx).sum(-1, keepdims=True)) + mx)

    tours, lebend, k, h = CO.strahl(logp, 1, n, breite)
    ref, k_ref, h_ref = _strahl_referenz(W.tolist(), n, breite)
    assert (k, h) == (k_ref, h_ref) == (B.k_strahl(n, breite), B.summe_hypothesen(n, breite))
    mein = sorted((tuple(int(v) for v in tours[0, b]) for b in range(breite) if lebend[0, b]))
    assert mein == sorted(tuple(t) for t, _ in ref)
    assert int(lebend[0].sum()) == len(ref) == min(breite, math.factorial(n - 1))


@pytest.mark.parametrize("netz", ["zeiger", "merkmal"])
def test_strahl_ab_breite_n_minus_1_fakultaet_gleich_wahrscheinlichster_folge(netz):
    """Mit Breite (n-1)! = 120 (n = 6) enthält der Strahl alle Folgen; die erste ist die wahrscheinlichste, gerechnet über das Produkt der Schrittwahrscheinlichkeiten."""
    n = 6
    x = I.instance_unit("gleich", n, 9)[None]
    D = I.distance_matrix_np(x)
    P, _ = MO.lade(netz, 0)
    if netz == "zeiger":
        tours, lebend, _, _ = CO.zeiger_strahl(P, x, 120)
        logp = {tuple(p): float(np.log(N.probabilities_along(P, x[0], [0] + list(p))[np.arange(n - 1), list(p)]).sum()) for p in itertools.permutations(range(1, n))}
    else:
        import rbek_featnet as F
        tours, lebend, _, _ = CO.merkmal_strahl(P, D, 120)
        logp = {}
        s = CO.massstab(D)
        for p in itertools.permutations(range(1, n)):
            tour = np.array([0] + list(p))[None]
            _, _, _, rec = F.rollout(D, s, np.array([0]), P, teacher=tour, record=True)
            import rbek_mlp as MLP
            lp = 0.0
            for (X, a, unv) in rec:
                z = np.where(unv, MLP.forward(P, X), -np.inf)
                lp += float(z[0, a[0]] - np.log(np.exp(z[0][unv[0]] - z[0][unv[0]].max()).sum()) - z[0][unv[0]].max())
            logp[tuple(p)] = lp
    assert lebend[0].all()
    bester = max(logp, key=logp.get)
    assert tuple(int(v) for v in tours[0, 0][1:]) == bester
    assert len({tuple(t) for t in tours[0].tolist()}) == 120


# ---------------------------------------------------------------------------------------------------------------- O6 exaktes Sampling
def test_sampling_haeufigkeiten_gleichen_der_exakten_verteilung():
    """Zeigernetz (Mini-Netz von Stück 1, fünf Orte, 24 Folgen): Häufigkeit je Folge im Band von 4,5 Standardfehlern, 60 000 Ziehungen aus SplitMix64."""
    P = N.hand_params()
    x = I.hand_instance()
    exakt = {}
    for p in itertools.permutations(range(1, 5)):
        t = [0] + list(p)
        exakt[tuple(t)] = float(np.prod(N.probabilities_along(P, x, t)[np.arange(4), list(p)]))
    assert abs(sum(exakt.values()) - 1.0) < 1e-12
    anz = 60000
    tours = CO.zeiger_gezogen(P, x[None], anz, [0], 0)[0]
    schluessel, zaehl = np.unique(tours, axis=0, return_counts=True)
    haeufig = {tuple(int(v) for v in s): c for s, c in zip(schluessel, zaehl)}
    for t, p in exakt.items():
        se = math.sqrt(max(p * (1 - p), 1e-12) / anz)
        assert abs(haeufig.get(t, 0) / anz - p) < 4.5 * se, (t, p, haeufig.get(t, 0) / anz)


# ---------------------------------------------------------------------------------------------------------------- O8 Verbesserung
def test_verbesserung_delta_jedes_zugs_gegen_volle_neuberechnung():
    """Der Abstieg (`rbek_search`) verkürzt mit jedem angenommenen Zug um genau das Delta der Kantenformel: die fortgeschriebene Länge gleicht der Neuberechnung (1e-9)."""
    import rbek_moves as M
    import rbek_search as S
    D = zufalls_D(14, 21)
    ctx = SU.kontext(D)
    t = UStream(21).random_order(1, 14)[0]
    res = S.descend(ctx, t, "aufzaehlung", protokoll=True)
    laenge = M.tour_length(D, t)
    for z in res["protokoll"]:
        laenge += z["delta"]
        assert abs(laenge - M.tour_length(D, z["tour"])) < 1e-9
        assert sorted(z["tour"].tolist()) == list(range(14))
    assert abs(laenge - M.tour_length(D, res["tour"])) < 1e-9
