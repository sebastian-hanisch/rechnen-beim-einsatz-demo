"""Export für `netz-gegen-gegner-demo` (Schema `ngg_export_v1`, `data/rbe_ngg_export.json`): Schema, Instanzen (Hash gegen die Kopie des Erzeugers), Kosten nach der Zählregel, Längen gegen das Optimum."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import rbe_budget as B
import rbe_scenario as SC

ROOT = Path(__file__).resolve().parent.parent
PFAD = ROOT / "data" / "rbe_ngg_export.json"
EXPORT = json.loads(PFAD.read_text(encoding="utf-8"))


def _hash(punkte) -> str:
    return hashlib.sha256(";".join(f"{round(float(x), 12):.12f},{round(float(y), 12):.12f}" for x, y in punkte).encode("utf-8")).hexdigest()


def test_oberer_rahmen_und_zwei_exporte():
    assert EXPORT["schema"] == "ngg_export_v1" and EXPORT["quelle"] == "rbe"
    assert [e["netz"] for e in EXPORT["exporte"]] == ["zeigernetz", "merkmalsnetz"]
    assert all(e["schema"] == "ngg_export_v1" and e["quelle"] == "rbe" for e in EXPORT["exporte"])


def test_training_kopf():
    z, m = (e["training"] for e in EXPORT["exporte"])
    assert z["parameter"] == 12480 and z["schritte"] == 8000 and z["batch"] == 120 and z["touren"] == 960000 and z["n_train"] == 20 and z["kandidaten_je_tour"] == 190
    assert z["kandidatenbewertungen"] == 182_400_000 and z["modell_seeds"] == 5 and z["operationen_vorwaerts_je_tour"] == float(B.o_zeiger(20, 1))
    assert m["parameter"] == 145 and m["schritte"] == 3750 and m["batch"] == 256 and m["touren"] == 960000 and m["kandidatenbewertungen"] == 182_400_000
    assert m["operationen_vorwaerts_je_tour"] == float(B.o_merkmal(20, 1))


@pytest.mark.parametrize("idx", [0, 1])
def test_zellen_instanzen_hash_und_form(idx):
    zellen = EXPORT["exporte"][idx]["zellen"]
    assert [z["familie"] for z in zellen] == ["gleich", "cluster", "gitter"]
    for z in zellen:
        assert z["n"] == 20 and z["zweck"] == 11 and z["indizes"] == list(range(30)) and len(z["instanz_hash"]) == 30 and len(z["optimum"]) == 30
        for s in (0, 7, 29):
            assert z["instanz_hash"][s] == _hash(SC.make_instance(z["familie"], 20, s)[0])
        assert len(set(z["instanz_hash"])) == 30


@pytest.mark.parametrize("idx", [0, 1])
def test_modi_kosten_und_laengen(idx):
    netz = ["zeiger", "merkmal"][idx]
    o1 = B.o_zeiger(20, 1) if netz == "zeiger" else B.o_merkmal(20, 1)
    for z in EXPORT["exporte"][idx]["zellen"]:
        opt = np.array(z["optimum"])
        modi = z["modi"]
        g = modi["greedy"]
        assert g["k"] == 1 and g["kosten_K"] == 190 and g["kosten_O"] == pytest.approx(float(o1)) and not g["mit_verbesserung"] and np.array(g["laengen"]).shape == (5, 30)
        assert set(modi["sampling"]) == {"2", "3", "4", "6", "8", "32", "128"}
        for k, s in modi["sampling"].items():
            assert s["kosten_K"] == int(k) * 190 and np.array(s["laengen"]).shape == (5, 30)
        ks = [int(k) for k in modi["sampling"]]
        for a, b in zip(sorted(ks), sorted(ks)[1:]):
            assert (np.array(modi["sampling"][str(b)]["laengen"]) <= np.array(modi["sampling"][str(a)]["laengen"]) + 1e-5).all()       # verschachtelt: mehr Touren nie schlechter
        assert modi["mehrfach_n"]["kosten_K"] == 3800 and modi["mehrfach_n"]["k"] == 20
        v = modi["mehrfach_n_verbessert"]
        assert v["mit_verbesserung"] and np.array(v["kosten_K"]).shape == (5, 30) and np.array(v["laengen"]).shape == (5, 30)
        assert (np.array(v["kosten_K"]) >= 3800 + B.K0.m_verschiedene_zuege(20)).all()
        for zeile in (g["laengen"], modi["mehrfach_n"]["laengen"], v["laengen"]):
            assert (np.array(zeile) >= opt - 1e-3).all()                                    # nie unter dem Optimum (1000er-Quadrat)
        assert (np.array(v["laengen"]) <= np.array(modi["mehrfach_n"]["laengen"]) + 1e-6).all()   # die Verbesserung verkürzt nie
