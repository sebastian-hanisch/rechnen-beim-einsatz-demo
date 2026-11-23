"""Die Live-Rechnungen der App (`rbe_live`): Rechengrenze, Instanzen, Experiment 1 bis 5 in kleiner Fassung, Übereinstimmung mit den Messfunktionen."""

import numpy as np

import rbe_band as BD
import rbe_budget as B
import rbe_experiments as EX
import rbe_live as LV
import rbe_net as N

SEEDS = (0, 1, 2, 3)


def test_instanz_seeds_und_grenzen():
    assert LV.instanz_seeds(0, 4) == (0, 1, 2, 3) and LV.instanz_seeds(11, 16)[-1] == 191
    assert LV.instanz_seeds(3, 2) == (48, 49)


def test_plan_runs_kuerzt_die_instanzen_und_sagt_es():
    assert LV.plan_runs(8, 190) == (8, None)
    m, text = LV.plan_runs(16, 75807)           # Strahlsuche 512 bei n = 20
    assert m == 13 and "von 16 auf 13 gekürzt" in text and "1 000 000" in text
    assert LV.plan_runs(16, 5_000_000)[0] == 1                 # mindestens eine Instanz
    assert LV.plan_runs(4, 1) == (4, None)


def test_daten_exakte_laengen_passen_zu_den_instanzen():
    X, D, ref = LV.daten(SEEDS)
    assert X.shape == (4, 20, 2) and D.shape == (4, 20, 20) and ref.shape == (4,)
    assert (ref > 0).all() and (ref < 20).all()


def test_e1_zeilen_kosten_und_reihenfolge_der_gaps():
    zeilen = LV.e1(SEEDS, 0)
    namen = [z["name"] for z in zeilen]
    assert namen == ["Zufallstour", "Nächster Nachbar", "Farthest Insertion", "Zeigernetz", "Merkmalsnetz"]
    k = {z["art"]: z["K"] for z in zeilen}
    assert k == {"zufall": 1, "nn": 190, "fi": 190, "zeiger": 190, "merkmal": 190}
    o = {z["art"]: z["O"] for z in zeilen}
    assert o["nn"] == 1520 and o["zeiger"] == 246384 and round(o["merkmal"]) == 47642
    gap = {z["art"]: z["gap"].mean() for z in zeilen}
    assert all((z["gap"] >= -1e-6).all() for z in zeilen)          # nie unter dem Optimum (die exakten Längen sind auf 9 Stellen gerundet)
    assert gap["zufall"] > gap["nn"] > 0 and gap["zeiger"] < gap["nn"]


def test_e2_gleicht_den_messfunktionen_und_haelt_die_konstruktionen_gleich():
    r = LV.e2(SEEDS, "zeiger", 0, "sampling", 8)
    X, D, ref = LV.daten(SEEDS)
    direkt = EX.netz_laengen("zeiger", 0, X, D, list(SEEDS), "sampling", [8])[8][0]
    assert np.allclose(r["gap_netz"], EX.luecke(direkt, ref))
    assert r["K"] == B.k_mehrfach(20, 8) == 1520 and r["k_nn"] == 8 and r["K_nn"] == 1520
    assert r["gap_zufall"] is not None
    m = LV.e2(SEEDS, "merkmal", 1, "mehrfach", 512)             # Mehrfachstart: höchstens 20 Startstädte
    assert m["k_eff"] == 20 and m["K"] == 3800 and m["k_nn"] == 20
    s = LV.e2(SEEDS, "zeiger", 0, "sampling", 512)
    assert s["k_nn"] == 20 and s["K"] == 97280 and s["gap_zufall"] is None            # 97 280 Zufallstouren: über der Grenze von 65 536, keine Spalte
    assert LV.e2(SEEDS, "zeiger", 0, "sampling", 256)["gap_zufall"] is not None
    sy = LV.e2(SEEDS, "zeiger", 0, "mehrfach", 20, 8)
    assert sy["symm"] == 8 and sy["k_nn"] == 20 and sy["K"] == 8 * 3800
    ohne = LV.e2(SEEDS, "merkmal", 0, "mehrfach", 20, 8)            # Symmetrien wirken beim Merkmalsnetz nicht
    assert ohne["symm"] == 1


def test_strahl_hypothesen_zeigen_die_lebenden_je_schritt():
    h = LV.strahl_hypothesen("zeiger", 0, 0, 8)
    assert len(h) == 19
    assert [len(t) for t, _ in h][:3] == [8, 8, 8]
    tours, score = h[-1]
    assert tours.shape[1] == 20 and (np.diff(score) <= 1e-12).all()             # nach Wahrscheinlichkeit sortiert


def test_e3_live_nur_bei_kleinem_budget():
    r = LV.e2(SEEDS, "zeiger", 0, "sampling", 8)
    ok = LV.e3_live(SEEDS, "zeiger", r["gap_netz"], r["K"], r["O"], 8.0, k_max=100_000)
    assert ok is not None and ok["anzahl"] == 4 and ok["budget_k"] == 1520 and ok["budget_o"] == int(r["O"] // 8)
    assert (ok["gap_suche_k"] >= -1e-6).all()
    assert LV.e3_live(SEEDS, "zeiger", r["gap_netz"], r["K"], r["O"], 8.0, k_max=1000) is None


def test_e4_startloesungen_kosten_und_endluecken():
    r = LV.e4(SEEDS, "zeiger", 0)
    st = r["starts"]
    assert set(st) == {"s4", "s5", "s6", "netz1", "netz2", "netz3"}
    assert st["s4"]["k_bau"] == 190 and st["s5"]["k_bau"] == 3800 and st["netz3"]["k_bau"] == 1520
    for z in st.values():
        assert (z["gap_end"] <= z["gap_start"] + 1e-9).all()               # die Verbesserung verkürzt nie
        assert (z["ev"] >= B.K0.m_verschiedene_zuege(20)).all()            # mindestens die Prüfrunde
        assert np.allclose(z["K"], z["k_bau"] + z["ev"])
    assert (st["s5"]["gap_start"] <= st["s4"]["gap_start"] + 1e-9).all()      # beste aus 20 ist nie schlechter als die erste


def test_abstieg_protokoll_ist_stimmig():
    X, D, _ = LV.daten([0])
    from rbe_suche import nn_tour
    a = LV.abstieg(D[0], nn_tour(D[0], 0))
    assert len(a["touren"]) == len(a["deltas"]) + 1 and a["lokalmin"] and a["ev_pruef"] == B.K0.m_verschiedene_zuege(20) and sum(a["stellen"]) + a["ev_pruef"] <= a["ev"]
    from rbek_moves import tour_length
    assert abs(tour_length(D[0], a["touren"][0]) + sum(a["deltas"]) - a["laenge"]) < 1e-9


def test_e5_kleine_fassung_und_zeitwaechter():
    r = LV.e5((0, 1), 0, 4, 4, 1e-3)
    assert r["instanzen"] == 2 and not r["abgebrochen"] and r["gap_aktiv"].shape == (2, 4) and r["gap_sampling"].shape == (2, 4)
    assert (np.diff(r["gap_aktiv"], axis=1) <= 1e-9).all() and (r["gap_aktiv"] >= -1e-6).all()    # beste Lücke bisher fällt nie
    assert r["touren_je_instanz"] == 16 and r["K_je_instanz"] == B.k_aktiv(20, 4, 4)
    r0 = LV.e5((0, 1, 2), 0, 4, 4, 1e-3, max_sekunden=0.0)
    assert r0["abgebrochen"] and r0["instanzen"] == 1                      # eine Instanz wird immer gerechnet


def test_e5_aktive_suche_veraendert_das_modell_der_app_nicht():
    from rbe_modelle import lade_zeiger
    vorher = {k: v.copy() for k, v in lade_zeiger(0)[0].items()}
    LV.e5((0,), 0, 2, 4, 1e-3)
    nachher = lade_zeiger(0)[0]
    assert all((vorher[k] == nachher[k]).all() for k in vorher)
    assert N.n_params(nachher) == 12480


def test_wahrheit_ist_unter_der_nn_laenge():
    ref = BD.wahrheit_n20(10)
    X, D, _ = LV.daten(range(10))
    nn = EX.nn_laengen(D, [1])[1][0]
    assert (nn >= ref - 1e-9).all()
