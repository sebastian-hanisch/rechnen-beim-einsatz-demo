"""Vorgerechnete Modelle: Format, Parameterzahlen, Meta, Gleichheit mit der Quelle und die Güte (Band) der Netze."""

import json
from pathlib import Path

import numpy as np
import pytest

import rbe_band as BD
import rbe_budget as B
import rbe_experiments as EX
import rbe_modelle as MO
import rbe_net as N

ROOT = Path(__file__).resolve().parent.parent
QUELLE_1 = ROOT.parent / "zeigernetz-demo"


@pytest.mark.parametrize("seed", range(5))
def test_zeigernetz_format_parameter_und_meta(seed):
    P, meta = MO.lade_zeiger(seed)
    assert N.n_params(P) == MO.ZEIGER_PARAMS == 12480
    assert N.describe(P) == {"d": 32, "layers": 1, "attn": True, "ffn": True}
    assert meta["format"] == "znz-modell-1" and meta["n_train"] == 20 and meta["steps"] == 8000 and meta["batch"] == 120 and meta["init_seed"] == seed == meta["data_seed"]
    assert meta["steps"] * meta["batch"] == MO.trainingstouren("zeiger") == 960000
    assert 8.0 < meta["val_gap_greedy"] < 13.0                     # Validierungslücke (100 Instanzen, Zufalls-Seed ab 1000)


@pytest.mark.parametrize("seed", range(5))
def test_merkmalsnetz_format_und_parameter(seed):
    params, meta = MO.lade_merkmal(seed)
    assert sum(p.size for p in params) == MO.MERKMAL_PARAMS == 145 == 7 * 16 + 16 + 16 + 1
    assert meta["format"] == "zwl-mlp-1" and meta["kind"] == "K"
    assert MO.trainingstouren("merkmal") == 960000 == 3750 * 256
    assert 7.0 < meta["val_gap_greedy"] < 10.0 and meta["n"] == 20 and meta["n_train_tours"] == 960000


def test_unbekanntes_netz_und_fehlende_datei():
    with pytest.raises(ValueError):
        MO.lade("gibt-es-nicht", 0)
    with pytest.raises(FileNotFoundError):
        N.load_model(MO.zeiger_pfad(99))


def test_zeigernetz_seed_0_gleicht_dem_modell_aus_stueck_1():
    quelle = QUELLE_1 / "znz_models" / "znz_n20_voll_d32_s0.npz"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    P, _ = MO.lade_zeiger(0)
    Q, _ = N.load_model(quelle)
    assert set(P) == set(Q) and all(np.allclose(P[k], Q[k], atol=1e-9, rtol=0) for k in P)


def test_alle_fuenf_zeigernetze_reproduzieren_die_greedy_laengen_aus_stueck_1():
    """Stück 1 hat nur Seed 0 als Datei abgelegt; die Seeds 1 bis 4 wurden hier mit demselben Rezept neu trainiert und liefern auf den 200 Testinstanzen dieselben Greedy-Längen (9 Stellen)."""
    quelle = QUELLE_1 / "znz_frozen.json"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    frozen = json.loads(quelle.read_text(encoding="utf-8"))
    X = EX.instanzen(range(200))
    D = EX.distanzen(X)
    for seed in range(5):
        ref = [r for r in frozen["netze"] if r["n"] == 20 and r["variante"] == "voll" and r["d"] == 32 and r["init_seed"] == seed][0]["greedy_laengen"]
        mine = EX.netz_laengen("zeiger", seed, X, D, list(range(200)), "greedy", [1])[1][0]
        assert np.max(np.abs(mine - np.array(ref))) < 1e-7, seed


def test_greedy_luecken_der_netze_im_band_der_planung():
    """Mittel über die fünf Gewichts-Zufalls-Seeds auf den 300 Testinstanzen (aus der Messreihe): Zeigernetz im Band 8,5 bis 11,5 %, Merkmalsnetz 7 bis 10,5 %, Nächster Nachbar 15,5 bis 19,5 %."""
    band = BD.lade_band()
    assert 8.5 <= BD.mittel(BD.netz_zeile("zeiger", "mehrfach", 1, band)["gap"]) <= 11.5
    assert 7.0 <= BD.mittel(BD.netz_zeile("merkmal", "mehrfach", 1, band)["gap"]) <= 10.5
    assert 15.5 <= BD.mittel(BD.klass_zeile("nn_mehrfach", 1, band)["gap"]) <= 19.5
    assert B.k_konstruktion(20) == band["netz"]["zeiger"]["mehrfach"]["1"]["K"]
