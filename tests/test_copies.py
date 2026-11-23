"""Kopier-Tests: die kopierten Dateien gleichen ihren Quellen nach dem Ersetzen des Präfixes (übersprungen, wenn die Quelle fehlt, z. B. in der CI), und die Konstantendateien
führen dieselben Namen und Werte wie die Quellen.

Quellen: `zeigernetz-demo` (Präfix znz_ -> rbe_) und `zugwahl-lernen-demo` (zwl_ -> rbek_), die Vorlage `rng_splitmix64_exact.py` im Website-Repo, die Modelle der Merkmalsnetze."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUELLE_1 = ROOT.parent / "zeigernetz-demo"
QUELLE_4 = ROOT.parent / "zugwahl-lernen-demo"
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def _norm(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("name", ["rngblock", "rngvec", "scenario", "instances", "exact", "stats", "net", "train", "cost"])
def test_kopie_aus_stueck_1_gleicht_der_quelle(name):
    quelle = QUELLE_1 / f"znz_{name}.py"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    assert _norm(ROOT / f"rbe_{name}.py") == _norm(quelle).replace(b"znz_", b"rbe_")


@pytest.mark.parametrize("name", ["cost", "moves", "search", "mlp", "featnet", "construct"])
def test_kopie_aus_stueck_4_gleicht_der_quelle(name):
    quelle = QUELLE_4 / f"zwl_{name}.py"
    if not quelle.exists():
        pytest.skip("zugwahl-lernen-demo nicht vorhanden")
    assert _norm(ROOT / f"rbek_{name}.py") == _norm(quelle).replace(b"zwl_", b"rbek_")


def test_rng_ist_die_unveraenderte_vorlage():
    if not VORLAGE.exists():
        pytest.skip("Vorlage nicht vorhanden (CI ohne Website-Repo)")
    assert _norm(ROOT / "rbe_rng.py") == _norm(VORLAGE)


def test_modul_frisch_gleicht_der_quelle():
    quelle = QUELLE_1 / "modul_frisch.py"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    assert _norm(ROOT / "modul_frisch.py") == _norm(quelle)


def test_modelle_des_merkmalsnetzes_gleichen_der_quelle():
    for k in range(5):
        quelle = QUELLE_4 / "zwl_modelle" / f"featnet_s{k}.npz"
        if not quelle.exists():
            pytest.skip("zugwahl-lernen-demo nicht vorhanden")
        assert (ROOT / "rbek_modelle" / f"featnet_s{k}.npz").read_bytes() == quelle.read_bytes()


def _modul(pfad: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, pfad)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def _gebrauchte_namen(dateien) -> set:
    """Alle Namen `C.NAME`, die die kopierten Dateien aus der Konstantendatei lesen."""
    import re
    namen = set()
    for d in dateien:
        namen |= set(re.findall(r"(?<![A-Za-z_])C\.([A-Z][A-Z_0-9]*)", (ROOT / d).read_text(encoding="utf-8")))
    return namen


def _konstanten_gleich(eigene: str, quelle_pfad: Path, dateien, ausnahmen=()):
    """Jeder Name, den die kopierten Dateien brauchen, steht in der eigenen Konstantendatei und hat denselben Wert wie in der Quelle."""
    import importlib
    mine = importlib.import_module(eigene)
    quelle = _modul(quelle_pfad, "quelle_" + eigene)
    fehlt, verschieden = [], []
    for name in sorted(_gebrauchte_namen(dateien) - set(ausnahmen)):
        if not hasattr(mine, name):
            fehlt.append(name)
        elif getattr(mine, name) != getattr(quelle, name):
            verschieden.append(name)
    return fehlt, verschieden


KOPIEN_1 = [f"rbe_{n}.py" for n in ("rngblock", "rngvec", "scenario", "instances", "exact", "stats", "net", "train", "cost")]
KOPIEN_4 = [f"rbek_{n}.py" for n in ("cost", "moves", "search", "mlp", "featnet", "construct")]


def test_konstanten_aus_stueck_1_gleichen_der_quelle():
    q = QUELLE_1 / "znz_constants.py"
    if not q.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    fehlt, verschieden = _konstanten_gleich("rbe_constants", q, KOPIEN_1)
    assert not fehlt and not verschieden, (fehlt, verschieden)
    gebraucht = _gebrauchte_namen(KOPIEN_1)
    for name in ("C_CLIP", "BATCH", "LERNRATE", "ZWECK_INIT", "ZWECK_TRAIN_X", "ZWECK_TRAIN_S", "MODELL_FORMAT", "SEITE", "FAMILIEN", "MARKE_KLAR", "BOOTSTRAP_ZIEHUNGEN", "OPS_JE_KANDIDAT"):
        assert name in gebraucht


def test_konstanten_aus_stueck_4_gleichen_der_quelle():
    q = QUELLE_4 / "zwl_constants.py"
    if not q.exists():
        pytest.skip("zugwahl-lernen-demo nicht vorhanden")
    fehlt, verschieden = _konstanten_gleich("rbek_constants", q, KOPIEN_4, ausnahmen=("MODELL_ORDNER",))
    assert not fehlt and not verschieden, (fehlt, verschieden)
    gebraucht = _gebrauchte_namen(KOPIEN_4)
    for name in ("EPS", "OR_LAENGEN", "RANG_KAPPE", "H", "K_PARAMS", "Z_PARAMS", "O_JE_ZEILE_K", "O_JE_ZEILE_Z", "MODELL_FORMAT", "SICHT_GEWICHT_Z"):
        assert name in gebraucht


def test_zwecke_dieser_demo_liegen_im_reservierten_bereich_und_sind_verschieden():
    import rbe_constants as C
    eigene = [C.ZWECK_SAMPLE_EINSATZ, C.ZWECK_ZUFALLSTOUREN, C.ZWECK_AKTIV, C.ZWECK_EXPORT, C.ZWECK_LIVE]
    assert all(200 <= z <= 249 for z in eigene) and len(set(eigene)) == len(eigene)
    quellen_zwecke = [C.ZWECK_INIT, C.ZWECK_TRAIN_X, C.ZWECK_TRAIN_S, C.ZWECK_BOOT, C.ZWECK_INSTANZ, C.ZWECK_START, C.ZWECK_LAUF]
    assert not set(eigene) & set(quellen_zwecke)


def test_formeln_der_zaehlung_gleichen_denen_von_stueck_4():
    """`rbe_cost` (Kopie von znz_cost) und `rbek_cost` (Kopie von zwl_cost) führen dieselben Namen mit denselben Werten."""
    import rbe_cost as A
    import rbek_cost as K
    for n in (5, 6, 10, 20, 50):
        assert A.k_konstruktion(n) == K.k_konstruktion(n) and A.m_verschiedene_zuege(n) == K.m_verschiedene_zuege(n)
    assert A.ops_kandidaten(190) == K.ops_kandidaten(190) == 1520


def test_kein_numpy_random_im_kern():
    """Eigene Dateien (ohne Kopien) importieren weder numpy.random noch random; die Kopien rufen es nicht auf (SplitMix64 überall)."""
    import re
    kopien = set(KOPIEN_1 + KOPIEN_4 + ["rbe_rng.py", "modul_frisch.py"])
    for datei in ROOT.glob("*.py"):
        text = datei.read_text(encoding="utf-8")
        assert not re.search(r"(np|numpy)\.random\.|default_rng\(|^\s*import random|^\s*from random import", text, re.M) or datei.name in kopien, datei.name
    for name in kopien:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"(np|numpy)\.random\.\w+\(|default_rng\(|^\s*import random|^\s*from random import", text, re.M), name
