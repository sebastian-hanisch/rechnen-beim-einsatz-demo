"""Erzeugt die Kopien aus den beiden Quellen (zeigernetz-demo, Präfix znz_ -> rbe_; zugwahl-lernen-demo, Präfix zwl_ -> rbek_) und die Modelle.

Aufruf (Wurzelordner der Demo):  python tools/rbe_kopieren.py            # schreibt alle Kopien neu
                                  python tools/rbe_kopieren.py --check    # meldet Abweichungen, schreibt nichts

Warum zwei Präfixe: beide Quellen haben eine `*_constants.py` und das Merkmalsnetz K prüft ein anderes Modellformat (`zwl-mlp-1`) als das Zeigernetz (`znz-modell-1`);
unter einem gemeinsamen Präfix wäre `C.MODELL_FORMAT` doppelt belegt. `rbe_` führt die Dateien aus Stück 1 (Zeigernetz, Zufall, Instanzen, exakte Wege, Zählung),
`rbek_` die aus Stück 4 (Zugmenge, lokale Suche, Merkmalsnetz K). Die Konstantendateien sind keine Kopien: sie führen dieselben Namen und Werte (Test `test_copies.py`).
Zeilenenden werden nicht verändert; der Kopier-Test vergleicht nach Umwandlung von CRLF in LF."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJEKTE = ROOT.parent
QUELLE_1 = PROJEKTE / "zeigernetz-demo"
QUELLE_4 = PROJEKTE / "zugwahl-lernen-demo"
VORLAGE_RNG = PROJEKTE / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"

AUS_STUECK_1 = ["rngblock", "rngvec", "scenario", "instances", "exact", "stats", "net", "train", "cost"]
AUS_STUECK_4 = ["cost", "moves", "search", "mlp", "featnet", "construct"]
MODELLE_K = [f"featnet_s{k}.npz" for k in range(5)]


def _lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def ziel_bytes(quelle: Path, alt: bytes, neu: bytes) -> bytes:
    return _lf(quelle.read_bytes()).replace(alt, neu)


def aufgaben():
    """(Ziel, Bytes) aller Textkopien."""
    out = [(ROOT / "rbe_rng.py", _lf(VORLAGE_RNG.read_bytes())), (ROOT / "modul_frisch.py", _lf((QUELLE_1 / "modul_frisch.py").read_bytes()))]
    for n in AUS_STUECK_1:
        out.append((ROOT / f"rbe_{n}.py", ziel_bytes(QUELLE_1 / f"znz_{n}.py", b"znz_", b"rbe_")))
    for n in AUS_STUECK_4:
        out.append((ROOT / f"rbek_{n}.py", ziel_bytes(QUELLE_4 / f"zwl_{n}.py", b"zwl_", b"rbek_")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fehler = 0
    for ziel, inhalt in aufgaben():
        if a.check:
            if not ziel.exists() or _lf(ziel.read_bytes()) != inhalt:
                print("ABWEICHUNG:", ziel.name)
                fehler += 1
        else:
            ziel.write_bytes(inhalt)
    modell_ziel = ROOT / "rbek_modelle"
    for m in MODELLE_K:
        quelle = QUELLE_4 / "zwl_modelle" / m
        if a.check:
            if not (modell_ziel / m).exists() or (modell_ziel / m).read_bytes() != quelle.read_bytes():
                print("ABWEICHUNG:", m)
                fehler += 1
        else:
            modell_ziel.mkdir(exist_ok=True)
            shutil.copyfile(quelle, modell_ziel / m)
    print("fertig" if not fehler else f"{fehler} Abweichungen")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
