"""Setzt die Befundtabelle der README aus der eingefrorenen Messung (`rbe_band.json`) zusammen.

    python tools/rbe_readme_befunde.py              # druckt den Block
    python tools/rbe_readme_befunde.py --schreiben  # ersetzt den Block zwischen den Markern in README.md

Die Sätze sind dieselben wie die Verdikte der App (`rbe_evaluation`), gerechnet auf den 300 Instanzen der Messung; `tests/test_claims.py` vergleicht den Block mit der README."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rbe_band as BD  # noqa: E402
import rbe_claims as CL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schreiben", action="store_true")
    a = ap.parse_args()
    block = CL.readme_block(BD.lade_band())
    if not a.schreiben:
        print(block)
        return 0
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    i, j = readme.find(CL.MARKER_START), readme.find(CL.MARKER_ENDE)
    if i < 0 or j < 0:
        print("Marker fehlen in der README")
        return 1
    (ROOT / "README.md").write_text(readme[:i] + block + readme[j + len(CL.MARKER_ENDE):], encoding="utf-8", newline="\n")
    print("README.md aktualisiert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
