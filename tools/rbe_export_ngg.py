"""Schreibt `data/rbe_ngg_export.json` (Schema `ngg_export_v1`) für `netz-gegen-gegner-demo`: Zeigernetz und Merkmalsnetz K auf denselben 30 Instanzen je Familie
(gleich, cluster, gitter; n = 20, Zufalls-Seed 0 bis 29, Zweck 11 der Quelle `hhw_scenario`), je Gewichts-Zufalls-Seed (5).

Aufbau der Datei: oben `schema`, `quelle`, `beschreibung` und die Liste `exporte`; **jedes Element von `exporte` ist ein vollständiger Export im Schema `ngg_export_v1`** (eigenes `netz`:
"zeigernetz" oder "merkmalsnetz", `training`, `zellen`). Längen im 1000er-Quadrat der Quelle (Einheitsquadrat mal 1000), Lücken rechnet der Leser gegen `optimum`.
`instanz_hash` je Instanz: sha256 über die auf 12 Stellen gerundeten Koordinaten der Quelle (1000er-Quadrat), formatiert als "x,y;x,y;..." mit je 12 Nachkommastellen.

Modi (Plan von `netz-gegen-gegner-demo`): `greedy` (eine Tour ab Stadt 0), `sampling` (k = 2, 3, 4, 6, 8, 32, 128: erste Tour greedy, weitere gezogen bei Temperatur 1, beste Tour, **Startstadt
je Instanz zufällig**, ein Strom je Instanz), `mehrfach_n` (Greedy ab allen 20 Startstädten, beste Tour), `mehrfach_n_verbessert` (beste aus 20, dann 2-opt + Or-opt mit erster Verbesserung
bis zum Lokalminimum; kosten_K und kosten_O je Instanz einschließlich der Zugauswertungen).

Aufruf (ein Prozess, BLAS-Threads 1):  python tools/rbe_export_ngg.py"""

import hashlib
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

import rbe_budget as B  # noqa: E402
import rbe_constants as C  # noqa: E402
import rbe_construct as CO  # noqa: E402
import rbe_experiments as EX  # noqa: E402
import rbe_modelle as MO  # noqa: E402
import rbe_scenario as SC  # noqa: E402
from rbe_rngvec import UStream  # noqa: E402
from rbe_scenario import derive  # noqa: E402

ZIEL = ROOT / "data" / "rbe_ngg_export.json"
FAMILIEN = ("gleich", "cluster", "gitter")
ANZAHL = 30
N = 20
SAMPLING_KS = (2, 3, 4, 6, 8, 32, 128)
SEITE = C.SEITE
NETZ_NAME = {"zeiger": "zeigernetz", "merkmal": "merkmalsnetz"}


def instanz_hash(punkte) -> str:
    text = ";".join(f"{round(float(x), 12):.12f},{round(float(y), 12):.12f}" for x, y in punkte)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trainingskopf(netz: str) -> dict:
    n_inst_tour = B.k_konstruktion(N)
    touren = MO.trainingstouren(netz)
    o = B.o_zeiger(N, 1) if netz == "zeiger" else B.o_merkmal(N, 1)
    if netz == "zeiger":
        return {"parameter": MO.ZEIGER_PARAMS, "schritte": MO.ZEIGER_TRAININGSSCHRITTE, "batch": MO.ZEIGER_BATCH, "touren": touren, "n_train": N, "kandidaten_je_tour": n_inst_tour,
                "kandidatenbewertungen": touren * n_inst_tour, "operationen_vorwaerts_je_tour": float(o), "modell_seeds": MO.MODELL_SEEDS, "referenzwert": "mittel"}
    return {"parameter": MO.MERKMAL_PARAMS, "schritte": MO.MERKMAL_TRAININGSSCHRITTE, "batch": MO.MERKMAL_BATCH_TOUREN, "touren": touren, "n_train": N, "kandidaten_je_tour": n_inst_tour,
            "kandidatenbewertungen": touren * n_inst_tour, "operationen_vorwaerts_je_tour": float(o), "modell_seeds": MO.MODELL_SEEDS,
            "referenzwert": "Mittel der 8 Touren (verschiedene Startstädte) derselben Instanz"}


def modus_netz(netz: str, modell: int, X, D, idx, startstadt) -> dict:
    """Rohlängen (Einheitsquadrat) aller Modi für ein Netz und einen Gewichts-Zufalls-Seed: dict Modus -> Längen (M,) bzw. (M, ...)."""
    P, _ = MO.lade(netz, modell)
    out = {}
    g = EX.netz_laengen(netz, modell, X, D, idx, "greedy", [1])[1]
    out["greedy"] = g[0]
    # Sampling: erste Tour greedy ab der zufälligen Startstadt, weitere gezogen ab derselben Startstadt
    st = startstadt[:, None]
    if netz == "zeiger":
        erste = CO.zeiger_greedy(P, X, st)
        gez = CO.zeiger_gezogen(P, X, max(SAMPLING_KS) - 1, idx, modell, start=startstadt)
    else:
        erste = CO.merkmal_touren(P, D, st)[0]
        gez = CO.merkmal_touren(P, D, np.repeat(st, max(SAMPLING_KS) - 1, axis=1), "sample", idx, modell)[0]
    alle = np.concatenate([erste, gez], axis=1)
    L = CO.laenge(X, alle)
    out["sampling"] = {k: L[:, :k].min(axis=1) for k in SAMPLING_KS}
    m = EX.netz_laengen(netz, modell, X, D, idx, "mehrfach", [N])[N]
    out["mehrfach_n"] = m[0]
    s2 = EX.verbessere_start(D, m[1], N * B.k_konstruktion(N))
    out["verbessert"] = s2
    return out


def main() -> int:
    t0 = time.perf_counter()
    wahr = json.loads((ROOT / "rbe_truth.json").read_text(encoding="utf-8"))["export"]
    exporte = []
    for netz in MO.NETZE:
        zellen = []
        for f_idx, fam in enumerate(FAMILIEN):
            seeds = list(range(ANZAHL))
            punkte = [SC.make_instance(fam, N, s)[0] for s in seeds]
            X = EX.instanzen(seeds, fam)
            D = EX.distanzen(X)
            idx = [1000 * f_idx + s for s in seeds]
            startstadt = np.array([int(UStream(derive(C.ZWECK_EXPORT, f_idx, s)).uniform(1)[0] * N) for s in seeds], dtype=np.int64)
            je_modell = [modus_netz(netz, m, X, D, idx, startstadt) for m in range(MO.MODELL_SEEDS)]
            kk = B.k_konstruktion(N)
            o1 = B.o_zeiger(N, 1) if netz == "zeiger" else B.o_merkmal(N, 1)
            o20 = B.o_zeiger(N, N) if netz == "zeiger" else B.o_merkmal(N, N)

            def mod(k, kosten_o, l_je):
                return {"k": k, "mit_verbesserung": False, "kosten_K": k * kk, "kosten_O": float(kosten_o), "laengen": [np.round(SEITE * laenge, 6).tolist() for laenge in l_je]}

            sampling = {}
            for k in SAMPLING_KS:
                o_k = B.o_zeiger(N, k) if netz == "zeiger" else B.o_merkmal(N, k)
                sampling[str(k)] = mod(k, o_k, [r["sampling"][k] for r in je_modell])
            modi = {"greedy": mod(1, o1, [r["greedy"] for r in je_modell]), "sampling": sampling, "mehrfach_n": mod(N, o20, [r["mehrfach_n"] for r in je_modell]),
                    "mehrfach_n_verbessert": {"mit_verbesserung": True, "k": N,
                                              "kosten_K": [np.round(r["verbessert"]["k_gesamt"], 1).tolist() for r in je_modell],
                                              "kosten_O": [np.round(o20 + B.O_JE_DELTA * r["verbessert"]["ev"], 1).tolist() for r in je_modell],
                                              "laengen": [np.round(SEITE * r["verbessert"]["laenge"], 6).tolist() for r in je_modell]}}
            zellen.append({"familie": fam, "n": N, "zweck": C.ZWECK_INSTANZ, "indizes": seeds, "instanz_hash": [instanz_hash(p) for p in punkte],
                           "optimum": [round(SEITE * v, 6) for v in wahr[fam]], "laengen_einheit": "1000er-Quadrat", "modi": modi})
            print(f"  {netz} {fam}: {time.perf_counter() - t0:.0f} s", flush=True)
        exporte.append({"schema": "ngg_export_v1", "netz": NETZ_NAME[netz], "quelle": "rbe", "training": trainingskopf(netz), "zellen": zellen})
    ZIEL.parent.mkdir(exist_ok=True)
    ZIEL.write_text(json.dumps({"schema": "ngg_export_v1", "quelle": "rbe", "beschreibung": "Enthält zwei Exporte (Zeigernetz und Merkmalsnetz K) in `exporte`; jeder ist ein Export im Schema ngg_export_v1.",
                                "exporte": exporte}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"fertig in {time.perf_counter() - t0:.0f} s: {ZIEL.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
