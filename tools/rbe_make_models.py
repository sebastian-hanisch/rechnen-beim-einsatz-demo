"""Trainiert die fünf Zeigernetze (n = 20, eine Schicht, d = 32, Gewichts-Zufalls-Seed 0 bis 4) mit dem Rezept von Stück 1 und legt sie nach `rbe_models/`.

Das Rezept ist das von `zeigernetz-demo/tools/znz_make_frozen.py` (8 000 Schritte x 120 Touren, Lernrate 1e-3, Referenzwert Mittel, init_seed = data_seed = Seed). Stück 1 hat nur
Seed 0 als Datei abgelegt; hier werden alle fünf neu trainiert und gegen die eingefrorene Messung von Stück 1 (`znz_frozen.json`, Greedy-Längen auf 200 Testinstanzen) geprüft, wenn die
Quelle vorhanden ist. Aufruf (ein Prozess, BLAS-Threads 1):
    python tools/rbe_make_models.py            # trainiert fehlende Modelle (wiederaufnehmbar)
    python tools/rbe_make_models.py --only 3   # nur Seed 3"""

import argparse
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

import rbe_constants as C  # noqa: E402
import rbe_instances as I  # noqa: E402
import rbe_net as N  # noqa: E402
import rbe_train as T  # noqa: E402

N_STADT, D, SCHRITTE, SEEDS = 20, 32, 8000, 5
MODELLE = ROOT / "rbe_models"
QUELLE_FROZEN = ROOT.parent / "zeigernetz-demo" / "znz_frozen.json"


def modell_pfad(seed: int) -> Path:
    return MODELLE / f"rbe_n{N_STADT}_voll_d{D}_s{seed}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    frozen = json.loads(QUELLE_FROZEN.read_text(encoding="utf-8")) if QUELLE_FROZEN.exists() else None
    inst = frozen["instanzen"][str(N_STADT)] if frozen else None
    if inst is None:
        print("Quelle znz_frozen.json fehlt: keine exakten Längen für die Validierung, Meta ohne Lücke")
    Xval = I.val_set(N_STADT, 100)
    opt_val = np.array(inst["wahrheit_val"]) if inst else None
    Xtest = I.test_set(N_STADT, 200)
    MODELLE.mkdir(exist_ok=True)
    for seed in range(SEEDS):
        if a.only is not None and seed != a.only:
            continue
        pfad = modell_pfad(seed)
        if pfad.exists():
            print(f"seed {seed}: vorhanden", flush=True)
            continue
        t0 = time.perf_counter()
        P, hist, counts = T.train(N_STADT, SCHRITTE, d=D, init_seed=seed, data_seed=seed, val=(Xval, opt_val) if inst else None, eval_every=0)
        val_gap = T.gap_greedy(P, Xval, opt_val) if inst else None
        starts = np.zeros((Xtest.shape[0], 1), dtype=np.int64)
        greedy = N.tour_lengths(Xtest, N.greedy_tours(P, Xtest, starts))[:, 0]
        abw = None
        if frozen:
            ref = [r for r in frozen["netze"] if r["n"] == N_STADT and r["variante"] == "voll" and r["d"] == D and r["init_seed"] == seed][0]
            abw = float(np.max(np.abs(greedy - np.array(ref["greedy_laengen"]))))
        meta = {"n_train": N_STADT, "steps": counts["schritte"], "batch": C.BATCH, "lr": C.LERNRATE, "baseline": "mean", "init_seed": seed, "data_seed": seed,
                "val_gap_greedy": round(val_gap, 4) if val_gap is not None else None, "val_instances": 100, "variante": "voll", "erzeugt_von": "tools/rbe_make_models.py"}
        N.save_model(pfad, P, meta)
        print(f"seed {seed}: {counts['touren']} Touren, Validierungslücke {val_gap}, größte Abweichung der Greedy-Längen zu Stück 1: {abw}, {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
