"""Erzeugt `rbe_truth.json`: die exakten Tourlängen (Held-Karp) der Instanzen dieser Demo.

* `n20_test`: die 300 Testinstanzen n = 20 gleichverteilt (Zufalls-Seed 0 bis 299, Einheitsquadrat). Die ersten 200 sind dieselben wie in `zeigernetz-demo`; liegt dessen `znz_frozen.json` vor,
  werden deren Längen von dort übernommen und an drei Instanzen mit `held_karp` neu gerechnet; die Instanzen 200 bis 299 sind selbst gerechnet (Held-Karp).
* `export`: 30 Instanzen je Familie (gleich, cluster, gitter; Zufalls-Seed 0 bis 29) für den Export an `netz-gegen-gegner-demo`.
* `n50`: die 10 Instanzen n = 50 gleichverteilt (Zufalls-Seed 0 bis 9) aus `zugwahl-lernen-demo/zwl_refs.json` (CP-SAT, bewiesen oder beste bekannte Länge; Untergrenzen im Feld `bewiesen`).

Aufruf (ein Prozess, BLAS-Threads 1):  python tools/rbe_make_truth.py"""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import rbe_exact as E  # noqa: E402
import rbe_instances as I  # noqa: E402

AUS_1 = ROOT.parent / "zeigernetz-demo" / "znz_frozen.json"
AUS_4 = ROOT.parent / "zugwahl-lernen-demo" / "zwl_refs.json"
ZIEL = ROOT / "rbe_truth.json"
N = 20
ANZAHL_TEST = 300


def exakt(X: np.ndarray) -> list:
    D = I.distance_matrix_np(X)
    return [round(float(E.held_karp(D[i])), 9) for i in range(X.shape[0])]


def main() -> int:
    t0 = time.perf_counter()
    X = I.test_set(N, ANZAHL_TEST)
    if AUS_1.exists():
        test = list(json.loads(AUS_1.read_text(encoding="utf-8"))["instanzen"]["20"]["wahrheit_test"])
        for i in (0, 77, 199):
            assert abs(exakt(X[i:i + 1])[0] - test[i]) < 1e-8, f"Instanz {i}: Held-Karp weicht von Stück 1 ab"
        test += exakt(X[len(test):])
        quelle = f"Instanzen 0 bis {len(test) - 101} aus zeigernetz-demo/znz_frozen.json (drei neu gerechnet), 200 bis {len(test) - 1} selbst gerechnet (Held-Karp)"
    else:
        test, quelle = exakt(X), "selbst gerechnet (Held-Karp)"
    export = {f: exakt(I.test_set(N, 30, f)) for f in ("gleich", "cluster", "gitter")}
    n50 = None
    if AUS_4.exists():
        refs = json.loads(AUS_4.read_text(encoding="utf-8"))["refs"]["gleich"]["50"]
        n50 = {"seeds": list(range(len(refs))), "laengen": [round(r["laenge"] / 1000.0, 9) for r in refs], "bewiesen": [bool(r["bewiesen"]) for r in refs],
               "quelle": "zugwahl-lernen-demo/zwl_refs.json (CP-SAT; nicht bewiesene Längen sind die besten gefundenen: Lücken sind dann Untergrenzen)"}
    ZIEL.write_text(json.dumps({"format": "rbe-truth-1", "n20_test": test, "n20_test_quelle": quelle, "export": export, "n50": n50}, ensure_ascii=False), encoding="utf-8")
    print(f"fertig in {time.perf_counter() - t0:.0f} s: {len(test)} Testinstanzen, Export {[len(v) for v in export.values()]}, n = 50: {None if n50 is None else len(n50['laengen'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
