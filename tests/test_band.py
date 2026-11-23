"""Die eingefrorene Messung `rbe_band.json`: Konsistenz (Form, Kosten, Monotonie, Raster) und Neu-Rechnen eines Ausschnitts als Band (nie bytegleich: Plattformen)."""

import numpy as np
import pytest

import rbe_band as BD
import rbe_budget as B
import rbe_experiments as EX
import rbe_live as LV
import rbe_modelle as MO

BAND = BD.lade_band()
N_INST = BAND["kopf"]["instanzen"]
SEEDS = BAND["kopf"]["modell_seeds"]


def test_kopf_nennt_groesse_und_umgebung():
    k = BAND["kopf"]
    assert k["format"] == "rbe-band-1" and k["n"] == 20 and k["familie"] == "gleich" and N_INST == 300 and SEEDS == 5
    assert k["aktiv_instanzen"] == 100 and k["symm_instanzen"] == 100 and k["n50_instanzen"] == 10
    assert k["python"] and k["numpy"] and k["rechner"] and set(k["modell_hashes_sha256_16"]) == {"zeiger", "merkmal"}


def test_netz_zeilen_form_kosten_und_untergrenze():
    for netz in MO.NETZE:
        for verf, zeilen in BAND["netz"][netz].items():
            for k, z in zeilen.items():
                g = np.array(z["gap"])
                assert g.shape == (SEEDS, N_INST), (netz, verf, k)
                assert (g >= -1e-6).all()                                          # nie unter dem Optimum (Held-Karp)
                kost, o = EX.kosten_netz(netz, verf, int(k))
                assert z["K"] == kost and z["O"] == pytest.approx(o)


def test_sampling_und_mehrfachstart_sind_je_instanz_verschachtelt_nie_schlechter():
    for netz in MO.NETZE:
        for verf in ("sampling", "mehrfach"):
            ks = BD.netz_ks(netz, verf, BAND)
            for a, b in zip(ks, ks[1:]):
                ga, gb = BD.netz_zeile(netz, verf, a, BAND)["gap"], BD.netz_zeile(netz, verf, b, BAND)["gap"]
                assert (gb <= ga + 1e-5).all(), (netz, verf, a, b)


def test_strahlbreite_1_gleich_greedy_und_k_aus_der_formel():
    for netz in MO.NETZE:
        assert np.allclose(BD.netz_zeile(netz, "strahl", 1, BAND)["gap"], BD.netz_zeile(netz, "mehrfach", 1, BAND)["gap"], atol=1e-5)
    assert BD.netz_zeile("zeiger", "strahl", 8, BAND)["K"] == 1387


def test_klassische_zeilen_kosten_und_monotonie():
    nn = BAND["klass"]["nn_mehrfach"]
    assert [nn[str(k)]["K"] for k in (1, 20)] == [190, 3800] and nn["1"]["O"] == 1520
    ks = sorted(int(k) for k in nn)
    for a, b in zip(ks, ks[1:]):
        assert (np.array(nn[str(b)]["gap"]) <= np.array(nn[str(a)]["gap"]) + 1e-5).all()
    z = BAND["klass"]["zufall"]
    zk = sorted(int(k) for k in z)
    assert zk[0] == 1 and zk[-1] == 65536 and z["1"]["K"] == 1 and z["1"]["O"] == 8
    for a, b in zip(zk, zk[1:]):
        assert (np.array(z[str(b)]["gap"]) <= np.array(z[str(a)]["gap"]) + 1e-5).all()
    assert BAND["klass"]["fi"]["K"] == 190 and BAND["klass"]["fi"]["O"] == 1600


def test_neustart_raster_enthaelt_alle_netz_budgets_in_k_und_o():
    raster = set(BAND["klass"]["neustart"]["raster"])
    fehlt = []
    for netz in MO.NETZE:
        for verf, zeilen in BAND["netz"][netz].items():
            for k, z in zeilen.items():
                for b in (int(z["K"]), int(z["O"] // B.O_JE_DELTA)):
                    if b <= max(raster) and b not in raster:        # darüber ist die Suche nicht gemessen: es gilt ihr Wert bei 400 000
                        fehlt.append((netz, verf, k, b))
    assert not fehlt, fehlt[:5]
    assert BAND["klass"]["neustart"]["raster"] == sorted(raster)
    g = np.array(BAND["klass"]["neustart"]["gap"])
    assert g.shape == (len(raster), N_INST) and (np.diff(g, axis=0) <= 1e-5).all() and (g >= -1e-6).all()   # nie schlechter bei größerem Budget, nie unter dem Optimum
    assert BD.neustart_gap_bei(190, BAND).shape == (N_INST,)
    with pytest.raises(ValueError):
        BD.neustart_gap_bei(1, BAND)


def test_mittelkurve_log_log_interpolation_und_b_stern():
    xs, ys = BD.neustart_mittelkurve(BAND)
    assert (np.diff(ys) <= 1e-9).all()
    mitte = BD.suche_luecke_bei(float(np.sqrt(xs[10] * xs[11])), BAND)
    assert mitte is None or min(ys[10], ys[11]) - 1e-9 <= mitte <= max(ys[10], ys[11]) + 1e-9
    assert BD.suche_luecke_bei(0.5, BAND) is None and BD.suche_luecke_bei(1e12, BAND) is None
    g = BD.mittel(BD.netz_zeile("zeiger", "mehrfach", 1, BAND)["gap"])
    b = BD.b_stern(g, BAND)
    assert b is not None and BD.suche_luecke_bei(b, BAND) == pytest.approx(g, rel=1e-6)
    assert BD.b_stern(1e6, BAND) is None and BD.b_stern(-1.0, BAND) is None


def test_folge_luecke_interpolation_von_hand():
    gap = np.array([[10.0, 10.0], [4.0, 6.0], [2.0, 6.0]])
    k = np.array([[100.0, 100.0], [300.0, 400.0], [500.0, 800.0]])
    out = BD.folge_luecke_bei(gap, k, np.array([200.0, 100.0]))
    assert out[0] == pytest.approx(7.0) and out[1] == pytest.approx(10.0)
    assert np.isnan(BD.folge_luecke_bei(gap, k, np.array([50.0, 50.0]))).all()
    assert BD.folge_luecke_bei(gap, k, np.array([10_000.0, 10_000.0])).tolist() == [2.0, 6.0]


def test_startloesungen_kosten_sind_konstruktion_plus_zugauswertungen():
    kk = B.k_konstruktion(20)
    for netz in MO.NETZE:
        for name, bau in (("S1", kk), ("S2", 20 * kk), ("S3", 8 * kk)):
            z = BAND["start"][netz][name]
            assert np.allclose(np.array(z["K"]), bau + np.array(z["ev"]))
            assert (np.array(z["ev"]) >= B.K0.m_verschiedene_zuege(20)).all()      # mindestens die Prüfrunde
            assert (np.array(z["gap"]) >= -1e-6).all()
    for name, bau in (("S4", kk), ("S5", 20 * kk)):
        z = BAND["start"]["klass"][name]
        assert np.allclose(np.array(z["K"]), bau + np.array(z["ev"]))
    f = BAND["folgen"]["nn"]
    assert np.array(f["gap"]).shape == (20, N_INST) and (np.diff(np.array(f["gap"]), axis=0) <= 1e-5).all() and (np.diff(np.array(f["K"]), axis=0) > 0).all()


def test_aktive_suche_und_symmetrien_zeilen():
    a = BAND["aktiv"]
    assert a["touren_je_schritt"] == 16 and a["schritte"] == [1, 2, 4, 8, 16, 32, 64, 128] and a["instanzen"] == 100
    for lr in ("0.001", "0.01"):
        g = np.array(a["aktiv"][lr])
        assert g.shape == (SEEDS, 8, 100) and (np.diff(g, axis=1) <= 1e-5).all()            # beste Lücke bisher fällt nie
    assert np.array(a["sampling"]).shape == (SEEDS, 8, 100)
    assert a["K"]["128"] == B.k_aktiv(20, 128, 16) and a["O"]["8"] == B.o_aktiv(20, 8, 16)
    s = BAND["symm"]
    assert s["K_acht"] == 8 * s["K_eins"] and np.array(s["eins"]).shape == (SEEDS, 100)


def test_neu_rechnen_eines_ausschnitts_als_band():
    """Zwölf Instanzen, Gewichts-Zufalls-Seed 0, je Verfahren: das Mittel der Lücke weicht höchstens 0,3 Prozentpunkte von der Messreihe ab (nie bytegleich: Linux und BLAS rechnen im letzten Bit anders)."""
    seeds = list(range(12))
    X, D, ref = LV.daten(seeds)
    for netz, verf, ks in (("zeiger", "mehrfach", [1, 20]), ("merkmal", "mehrfach", [1, 20]), ("zeiger", "sampling", [8]), ("merkmal", "sampling", [8]), ("zeiger", "strahl", [8]), ("merkmal", "strahl", [8])):
        neu = EX.netz_laengen(netz, 0, X, D, seeds, verf, ks)
        for k in ks:
            alt = np.array(BAND["netz"][netz][verf][str(k)]["gap"])[0, :12]
            assert abs(EX.luecke(neu[k][0], ref).mean() - alt.mean()) < 0.3, (netz, verf, k)
    nn = EX.nn_laengen(D, [20])[20][0]
    assert abs(EX.luecke(nn, ref).mean() - np.array(BAND["klass"]["nn_mehrfach"]["20"]["gap"])[:12].mean()) < 1e-4
