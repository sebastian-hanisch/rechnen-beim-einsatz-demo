"""Erzeugt die eingefrorene Messung `rbe_band.json` (Format `rbe-band-1`) in Stufen mit Zwischenspeichern in `tools/_zwischen/`.

Aufruf (Wurzelordner der Demo; ein rechnender Prozess, BLAS-Threads 1; wiederaufnehmbar, fertige Zwischenstände werden übersprungen):
    python tools/rbe_make_band.py --stage netz        # beide Netze, 5 Gewichts-Zufalls-Seeds: Greedy, Mehrfachstart, Sampling, Strahlsuche
    python tools/rbe_make_band.py --stage klass       # Nächster Nachbar, Farthest Insertion, Zufallstouren, Neustart-Suche, Nächster Nachbar und Farthest Insertion mit Verbesserung
    python tools/rbe_make_band.py --stage start       # Netz baut, Suche verbessert (n = 20): Startlösungen S1 bis S6 und Folgen mit Verbesserung je Start
    python tools/rbe_make_band.py --stage aktiv       # aktive Suche gegen Sampling gleicher Tourzahl
    python tools/rbe_make_band.py --stage symm        # Symmetrien (acht Abbildungen) beim Zeigernetz
    python tools/rbe_make_band.py --stage n50         # Ausblick n = 50 (nur Merkmalsnetz, 10 Instanzen)
    python tools/rbe_make_band.py --stage auswertung  # setzt rbe_band.json zusammen

Größe: siehe KOPF unten und die README. Die Instanzen sind die 300 Testinstanzen (Zufalls-Seed 0 bis 299; die ersten 200 wie in `zeigernetz-demo`); exakte Längen aus `rbe_truth.json`."""

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

import rbe_active as AK  # noqa: E402
import rbe_budget as B  # noqa: E402
import rbe_constants as C  # noqa: E402
import rbe_construct as CO  # noqa: E402
import rbe_experiments as EX  # noqa: E402
import rbe_modelle as MO  # noqa: E402
import rbe_suche as SU  # noqa: E402
from rbe_rngvec import UStream  # noqa: E402
from rbe_scenario import derive  # noqa: E402

ZW = ROOT / "tools" / "_zwischen"
BAND = ROOT / "rbe_band.json"
WAHRHEIT = ROOT / "rbe_truth.json"

N_STADT = 20
N_INST = 300                        # Testinstanzen n = 20 (Zufalls-Seed 0 bis 299)
SEEDS = tuple(range(MO.MODELL_SEEDS))
MEHRFACH_KS = tuple(range(1, 21))
SAMPLING_KS = {"zeiger": tuple(2 ** j for j in range(0, 12)), "merkmal": tuple(2 ** j for j in range(0, 12))}      # bis 2 048 Touren
STRAHL_KS = {"zeiger": tuple(2 ** j for j in range(0, 10)), "merkmal": tuple(2 ** j for j in range(0, 10))}         # Breite bis 512
ZUFALL_B = tuple(2 ** j for j in range(0, 17))                                                                       # beste von bis zu 65 536 Zufallstouren
NEUSTART_MAX = 400_000                                                                                              # Kandidatenbewertungen je Instanz
START_FOLGE_ZUFALL = 30             # Zufallsstarts mit Verbesserung je Folge
AKTIV_INST, AKTIV_SCHRITTE, AKTIV_TOUREN, AKTIV_LRS = 100, 128, 16, (1e-3, 1e-2)
SYMM_INST = 100
N50_INST = 10
SAMPLING_START_K = 8                # S3: Sampling mit 8 Touren als Startlösung


def rund(a, st: int = 9) -> list:
    return np.round(np.asarray(a, dtype=np.float64), st).tolist()


def lade(pfad: Path, standard=None):
    return json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else standard


def schreibe(pfad: Path, obj) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(pfad)


def daten(m: int = N_INST):
    seeds = list(range(m))
    X = EX.instanzen(seeds)
    return seeds, X, EX.distanzen(X)


def raster_budgets() -> list:
    """K-Raster der Neustart-Suche: die Budgets der Netz-Zeilen in K, die Budgets in O (geteilt durch 8) und ein dichtes geometrisches Raster."""
    ks = set()
    n = N_STADT
    for netz in MO.NETZE:
        ks.add(B.k_konstruktion(n))
        for k in MEHRFACH_KS:
            ks.add(EX.kosten_netz(netz, "mehrfach", k)[0])
            ks.add(int(EX.kosten_netz(netz, "mehrfach", k)[1] // B.O_JE_DELTA))
        for k in SAMPLING_KS[netz]:
            ks.add(EX.kosten_netz(netz, "sampling", k)[0])
            ks.add(int(EX.kosten_netz(netz, "sampling", k)[1] // B.O_JE_DELTA))
        for k in STRAHL_KS[netz]:
            ks.add(EX.kosten_netz(netz, "strahl", k)[0])
            ks.add(int(EX.kosten_netz(netz, "strahl", k)[1] // B.O_JE_DELTA))
    x = 8.0
    while x < NEUSTART_MAX:
        ks.add(int(round(x)))
        x *= 1.25
    ks.update((128, 512, 2048, 8192, 32768))
    return sorted(k for k in ks if 1 <= k <= NEUSTART_MAX)


# ---------------------------------------------------------------------------------------------------------------- Stufe netz
def stufe_netz():
    seeds, X, D = daten()
    for netz in MO.NETZE:
        for modell in SEEDS:
            pfad = ZW / f"netz_{netz}_s{modell}.json"
            if pfad.exists():
                continue
            t0 = time.perf_counter()
            rec = {"mehrfach": {}, "sampling": {}, "strahl": {}}
            for verf, ks in (("mehrfach", MEHRFACH_KS), ("sampling", SAMPLING_KS[netz]), ("strahl", STRAHL_KS[netz])):
                t1 = time.perf_counter()
                res = EX.netz_laengen(netz, modell, X, D, seeds, verf, ks)
                rec[verf] = {str(k): rund(v[0]) for k, v in res.items()}
                print(f"  {netz} s{modell} {verf}: {time.perf_counter() - t1:.0f} s", flush=True)
            rec["sekunden"] = round(time.perf_counter() - t0, 1)
            schreibe(pfad, rec)
            print(f"netz {netz} s{modell}: {rec['sekunden']:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe klass
def stufe_klass():
    pfad = ZW / "klass.json"
    if pfad.exists():
        return
    seeds, X, D = daten()
    t0 = time.perf_counter()
    rec = {}
    nn = EX.nn_laengen(D, MEHRFACH_KS)
    rec["nn_mehrfach"] = {str(k): rund(v[0]) for k, v in nn.items()}
    fi_l, fi_t = EX.fi_laengen(D, 0)
    rec["fi"] = rund(fi_l)
    rec["zufall"] = {str(b): rund(v) for b, v in EX.zufall_beste(D, ZUFALL_B, lambda i: UStream(derive(C.ZWECK_ZUFALLSTOUREN, 0, seeds[i]))).items()}
    print(f"  Nächster Nachbar, Farthest Insertion, Zufall: {time.perf_counter() - t0:.0f} s", flush=True)
    raster = raster_budgets()
    neu = np.empty((N_INST, len(raster)))
    t1 = time.perf_counter()
    for i in range(N_INST):
        ctx = SU.kontext(D[i])
        ks, ls = SU.neustart_suche(ctx, UStream(derive(C.ZWECK_ZUFALLSTOUREN, 1, seeds[i])), NEUSTART_MAX)
        neu[i] = SU.beste_bei_raster(ks, ls, raster)
        if i % 25 == 24:
            print(f"  Neustart-Suche {i + 1}/{N_INST}: {time.perf_counter() - t1:.0f} s", flush=True)
    rec["neustart"] = {"raster": raster, "laenge": [rund(neu[:, j]) for j in range(len(raster))]}
    # Nächster Nachbar + Verbesserung (S4) und Farthest Insertion + Verbesserung (S6)
    kk = B.k_konstruktion(N_STADT)
    s4 = EX.verbessere_start(D, nn[1][1], kk)
    s6 = EX.verbessere_start(D, fi_t, kk)
    rec["nn_v"] = {"laenge": rund(s4["laenge"]), "ev": rund(s4["ev"], 1), "zuege": rund(s4["zuege"], 1)}
    rec["fi_v"] = {"laenge": rund(s6["laenge"]), "ev": rund(s6["ev"], 1), "zuege": rund(s6["zuege"], 1)}
    rec["sekunden"] = round(time.perf_counter() - t0, 1)
    schreibe(pfad, rec)
    print(f"klass: {rec['sekunden']:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe start
def folge(D: np.ndarray, tours: np.ndarray, konstr_k: int) -> dict:
    """Verbesserung ab jeder der Starttouren (M, m, n): Länge und Zugauswertungen je Start."""
    M, m, n = tours.shape
    L, EV = np.empty((M, m)), np.empty((M, m))
    for i in range(M):
        ctx = SU.kontext(D[i])
        for j in range(m):
            r = SU.verbessere(ctx, tours[i, j])
            L[i, j], EV[i, j] = r["laenge"], r["ev"]
    return {"laenge": L, "ev": EV, "konstr_k": konstr_k}


def stufe_start():
    seeds, X, D = daten()
    kk = B.k_konstruktion(N_STADT)
    for netz in MO.NETZE:
        for modell in SEEDS:
            pfad = ZW / f"start_{netz}_s{modell}.json"
            if pfad.exists():
                continue
            t0 = time.perf_counter()
            P, _ = MO.lade(netz, modell)
            starts = CO.alle_starts(N_INST, N_STADT)
            if netz == "zeiger":
                tours = CO.zeiger_greedy(P, X, starts)
            else:
                tours, _ = CO.merkmal_touren(P, D, starts)
            fol = folge(D, tours, kk)
            L0 = CO.laenge(X, tours)
            beste = L0.argmin(axis=1)
            mi = np.arange(N_INST)
            s2 = EX.verbessere_start(D, tours[mi, beste], N_STADT * kk)
            samp = EX.netz_laengen(netz, modell, X, D, seeds, "sampling", [SAMPLING_START_K])[SAMPLING_START_K]
            s3 = EX.verbessere_start(D, samp[1], SAMPLING_START_K * kk)
            rec = {"folge_laenge": rund(fol["laenge"]), "folge_ev": rund(fol["ev"], 1), "start_laenge": rund(L0), "s2": {"laenge": rund(s2["laenge"]), "ev": rund(s2["ev"], 1)},
                   "s3": {"laenge": rund(s3["laenge"]), "ev": rund(s3["ev"], 1), "start": rund(samp[0])}, "sekunden": round(time.perf_counter() - t0, 1)}
            schreibe(pfad, rec)
            print(f"start {netz} s{modell}: {rec['sekunden']:.0f} s", flush=True)
    pfad = ZW / "start_klass.json"
    if not pfad.exists():
        t0 = time.perf_counter()
        nn_t = np.stack([np.stack([SU.nn_tour(D[i], s) for s in range(N_STADT)]) for i in range(N_INST)])
        fol = folge(D, nn_t, kk)
        L0 = np.array([[D[i][t, np.roll(t, -1)].sum() for t in nn_t[i]] for i in range(N_INST)])
        beste = L0.argmin(axis=1)
        mi = np.arange(N_INST)
        s5 = EX.verbessere_start(D, nn_t[mi, beste], N_STADT * kk)
        zuf = np.stack([UStream(derive(C.ZWECK_ZUFALLSTOUREN, 2, seeds[i])).random_order(START_FOLGE_ZUFALL, N_STADT) for i in range(N_INST)])
        fz = folge(D, zuf, 1)
        rec = {"nn": {"folge_laenge": rund(fol["laenge"]), "folge_ev": rund(fol["ev"], 1), "start_laenge": rund(L0)}, "s5": {"laenge": rund(s5["laenge"]), "ev": rund(s5["ev"], 1)},
               "zufall": {"folge_laenge": rund(fz["laenge"]), "folge_ev": rund(fz["ev"], 1)}, "sekunden": round(time.perf_counter() - t0, 1)}
        schreibe(pfad, rec)
        print(f"start klass: {rec['sekunden']:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe aktiv
def stufe_aktiv():
    seeds, X, D = daten(AKTIV_INST)
    for modell in SEEDS:
        pfad = ZW / f"aktiv_s{modell}.json"
        if pfad.exists():
            continue
        t0 = time.perf_counter()
        P, _ = MO.lade("zeiger", modell)
        rec = {"aktiv": {}, "sampling": []}
        for lr in AKTIV_LRS:
            beste = np.empty((AKTIV_INST, AKTIV_SCHRITTE))
            for i in range(AKTIV_INST):
                us = UStream(derive(C.ZWECK_AKTIV, 0, modell, seeds[i]))
                beste[i] = AK.aktive_suche(P, X[i], AKTIV_SCHRITTE, AKTIV_TOUREN, lr, us)["beste"]
            rec["aktiv"][str(lr)] = [rund(beste[:, s - 1]) for s in range(1, AKTIV_SCHRITTE + 1)]
        beste = np.empty((AKTIV_INST, AKTIV_SCHRITTE))
        for i in range(AKTIV_INST):
            us = UStream(derive(C.ZWECK_AKTIV, 1, modell, seeds[i]))
            beste[i] = AK.sampling_gleiche_touren(P, X[i], AKTIV_SCHRITTE, AKTIV_TOUREN, us)["beste"]
        rec["sampling"] = [rund(beste[:, s - 1]) for s in range(1, AKTIV_SCHRITTE + 1)]
        rec["sekunden"] = round(time.perf_counter() - t0, 1)
        schreibe(pfad, rec)
        print(f"aktiv s{modell}: {rec['sekunden']:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe symm
def stufe_symm():
    seeds, X, D = daten(SYMM_INST)
    for modell in SEEDS:
        pfad = ZW / f"symm_s{modell}.json"
        if pfad.exists():
            continue
        t0 = time.perf_counter()
        eins = EX.netz_laengen("zeiger", modell, X, D, seeds, "mehrfach", [N_STADT], symm=1)[N_STADT][0]
        acht = EX.netz_laengen("zeiger", modell, X, D, seeds, "mehrfach", [N_STADT], symm=8)[N_STADT][0]
        mk = EX.netz_laengen("merkmal", modell, X, D, seeds, "mehrfach", [N_STADT])[N_STADT][0]
        # Merkmalsnetz ist invariant: dieselbe Tourlänge unter allen acht Abbildungen (Eigenschaftstest in den Tests, hier nur die Zählung)
        schreibe(pfad, {"eins": rund(eins), "acht": rund(acht), "merkmal": rund(mk), "sekunden": round(time.perf_counter() - t0, 1)})
        print(f"symm s{modell}: {time.perf_counter() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe n50
def stufe_n50():
    pfad = ZW / "n50.json"
    if pfad.exists():
        return
    t0 = time.perf_counter()
    n = 50
    seeds = list(range(N50_INST))
    X = EX.instanzen(seeds, n=n)
    D = EX.distanzen(X)
    kk = B.k_konstruktion(n)
    rec = {"merkmal": {}, "klass": {}}
    for modell in SEEDS:
        P, _ = MO.lade("merkmal", modell)
        tours, _ = CO.merkmal_touren(P, D, CO.alle_starts(N50_INST, n))
        L0 = CO.laenge(X, tours)
        mi = np.arange(N50_INST)
        s1 = EX.verbessere_start(D, tours[:, 0], kk)
        s2 = EX.verbessere_start(D, tours[mi, L0.argmin(axis=1)], n * kk)
        rec["merkmal"][str(modell)] = {"greedy": rund(L0[:, 0]), "mehrfach": rund(L0.min(axis=1)), "s1": {"laenge": rund(s1["laenge"]), "ev": rund(s1["ev"], 1)},
                                       "s2": {"laenge": rund(s2["laenge"]), "ev": rund(s2["ev"], 1)}}
        print(f"  n50 merkmal s{modell}: {time.perf_counter() - t0:.0f} s", flush=True)
    nn_t = np.stack([np.stack([SU.nn_tour(D[i], s) for s in range(n)]) for i in range(N50_INST)])
    fol = folge(D, nn_t, kk)
    L0 = np.array([[D[i][t, np.roll(t, -1)].sum() for t in nn_t[i]] for i in range(N50_INST)])
    mi = np.arange(N50_INST)
    s5 = EX.verbessere_start(D, nn_t[mi, L0.argmin(axis=1)], n * kk)
    fi_l, fi_t = EX.fi_laengen(D, 0)
    s6 = EX.verbessere_start(D, fi_t, kk)
    rec["klass"] = {"nn_start_laenge": rund(L0), "folge_laenge": rund(fol["laenge"]), "folge_ev": rund(fol["ev"], 1), "s5": {"laenge": rund(s5["laenge"]), "ev": rund(s5["ev"], 1)},
                    "fi": rund(fi_l), "s6": {"laenge": rund(s6["laenge"]), "ev": rund(s6["ev"], 1)}}
    rec["sekunden"] = round(time.perf_counter() - t0, 1)
    schreibe(pfad, rec)
    print(f"n50: {rec['sekunden']:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- Stufe auswertung
AKTIV_STUFEN = (1, 2, 4, 8, 16, 32, 64, 128)       # gespeicherte Schrittzahlen der aktiven Suche


def _gap(laenge, ref):
    return np.round(100.0 * (np.asarray(laenge) - np.asarray(ref)) / np.asarray(ref), 5)


def stufe_auswertung() -> int:
    import hashlib
    import platform
    wahr = json.loads(WAHRHEIT.read_text(encoding="utf-8"))
    ref = np.array(wahr["n20_test"])
    n = N_STADT
    kk = B.k_konstruktion(n)
    band = {"format": "rbe-band-1"}
    sek = {}
    netz = {}
    for nz in MO.NETZE:
        recs = [lade(ZW / f"netz_{nz}_s{m}.json") for m in SEEDS]
        if any(r is None for r in recs):
            print(f"Zwischenstand netz_{nz} unvollständig")
            return 1
        sek[f"netz_{nz}"] = round(sum(r["sekunden"] for r in recs), 1)
        netz[nz] = {}
        for verf in ("mehrfach", "sampling", "strahl"):
            netz[nz][verf] = {}
            for k in recs[0][verf]:
                kost, op = EX.kosten_netz(nz, verf, int(k))
                netz[nz][verf][k] = {"K": kost, "O": op, "gap": [_gap(r[verf][k], ref).tolist() for r in recs]}
    band["netz"] = netz
    kl = lade(ZW / "klass.json")
    sek["klass"] = kl["sekunden"]
    klass = {"nn_mehrfach": {k: {"K": int(k) * kk, "O": B.o_nn(n) * int(k), "gap": _gap(v, ref).tolist()} for k, v in kl["nn_mehrfach"].items()},
             "fi": {"K": kk, "O": B.o_farthest(n), "gap": _gap(kl["fi"], ref).tolist()},
             "zufall": {k: {"K": int(k), "O": B.o_zufall(int(k)), "gap": _gap(v, ref).tolist()} for k, v in kl["zufall"].items()},
             "neustart": {"raster": kl["neustart"]["raster"], "gap": [_gap(v, ref).tolist() for v in kl["neustart"]["laenge"]]},
             "nn_v": {"K": (kk + np.array(kl["nn_v"]["ev"])).tolist(), "O": (B.o_nn(n) + B.O_JE_DELTA * np.array(kl["nn_v"]["ev"])).tolist(), "ev": kl["nn_v"]["ev"],
                      "gap": _gap(kl["nn_v"]["laenge"], ref).tolist()},
             "fi_v": {"K": (kk + np.array(kl["fi_v"]["ev"])).tolist(), "O": (B.o_farthest(n) + B.O_JE_DELTA * np.array(kl["fi_v"]["ev"])).tolist(), "ev": kl["fi_v"]["ev"],
                      "gap": _gap(kl["fi_v"]["laenge"], ref).tolist()}}
    band["klass"] = klass
    # Netz baut, Suche verbessert (n = 20)
    start, folgen = {}, {}
    for nz in MO.NETZE:
        recs = [lade(ZW / f"start_{nz}_s{m}.json") for m in SEEDS]
        if any(r is None for r in recs):
            print(f"Zwischenstand start_{nz} unvollständig")
            return 1
        sek[f"start_{nz}"] = round(sum(r["sekunden"] for r in recs), 1)
        o_netz = (lambda m: B.o_zeiger(n, m)) if nz == "zeiger" else (lambda m: B.o_merkmal(n, m))
        o_bau = {"S1": o_netz(1), "S2": o_netz(n), "S3": o_netz(SAMPLING_START_K)}
        k_bau = {"S1": kk, "S2": n * kk, "S3": SAMPLING_START_K * kk}
        start[nz] = {}
        for s_name, quelle in (("S1", None), ("S2", "s2"), ("S3", "s3")):
            lens, evs = [], []
            for r in recs:
                if s_name == "S1":
                    lens.append(np.array(r["folge_laenge"])[:, 0])
                    evs.append(np.array(r["folge_ev"])[:, 0])
                else:
                    lens.append(np.array(r[quelle]["laenge"]))
                    evs.append(np.array(r[quelle]["ev"]))
            start[nz][s_name] = {"K": [(k_bau[s_name] + e).tolist() for e in evs], "O": [(o_bau[s_name] + B.O_JE_DELTA * e).tolist() for e in evs], "ev": [e.tolist() for e in evs],
                                 "gap": [_gap(x, ref).tolist() for x in lens]}
        # Folge: Verbesserung ab den Greedy-Touren aus Startstadt 0, 1, ..., n - 1; laufend beste Lücke und kumuliertes K
        fol_gap, fol_k = [], []
        for r in recs:
            L, EV = np.array(r["folge_laenge"]), np.array(r["folge_ev"])
            fol_gap.append(np.minimum.accumulate(_gap(L, ref[:, None]), axis=1).T.tolist())
            fol_k.append(np.cumsum(kk + EV, axis=1).T.tolist())
        folgen[nz] = {"gap": fol_gap, "K": fol_k}
    kc = lade(ZW / "start_klass.json")
    sek["start_klass"] = kc["sekunden"]
    nnL, nnEV = np.array(kc["nn"]["folge_laenge"]), np.array(kc["nn"]["folge_ev"])
    zfL, zfEV = np.array(kc["zufall"]["folge_laenge"]), np.array(kc["zufall"]["folge_ev"])
    start["klass"] = {
        "S4": {"K": (kk + nnEV[:, 0]).tolist(), "O": (B.o_nn(n) + B.O_JE_DELTA * nnEV[:, 0]).tolist(), "ev": nnEV[:, 0].tolist(), "gap": _gap(nnL[:, 0], ref).tolist()},
        "S5": {"K": (n * kk + np.array(kc["s5"]["ev"])).tolist(), "O": (n * B.o_nn(n) + B.O_JE_DELTA * np.array(kc["s5"]["ev"])).tolist(), "ev": kc["s5"]["ev"],
               "gap": _gap(kc["s5"]["laenge"], ref).tolist()},
        "S6": klass["fi_v"]}
    folgen["nn"] = {"gap": np.minimum.accumulate(_gap(nnL, ref[:, None]), axis=1).T.tolist(), "K": np.cumsum(kk + nnEV, axis=1).T.tolist()}
    folgen["zufall"] = {"gap": np.minimum.accumulate(_gap(zfL, ref[:, None]), axis=1).T.tolist(), "K": np.cumsum(1 + zfEV, axis=1).T.tolist()}
    band["start"], band["folgen"] = start, folgen
    # aktive Suche
    recs = [lade(ZW / f"aktiv_s{m}.json") for m in SEEDS]
    if any(r is None for r in recs):
        print("Zwischenstand aktiv unvollständig")
        return 1
    sek["aktiv"] = round(sum(r["sekunden"] for r in recs), 1)
    refa = ref[:AKTIV_INST]
    akt = {"instanzen": AKTIV_INST, "touren_je_schritt": AKTIV_TOUREN, "schritte": list(AKTIV_STUFEN), "aktiv": {}, "sampling": {}}
    for lr in AKTIV_LRS:
        akt["aktiv"][str(lr)] = [[_gap(r["aktiv"][str(lr)][s - 1], refa).tolist() for s in AKTIV_STUFEN] for r in recs]
    akt["sampling"] = [[_gap(r["sampling"][s - 1], refa).tolist() for s in AKTIV_STUFEN] for r in recs]
    akt["K"] = {str(s): B.k_aktiv(n, s, AKTIV_TOUREN) for s in AKTIV_STUFEN}
    akt["O"] = {str(s): B.o_aktiv(n, s, AKTIV_TOUREN) for s in AKTIV_STUFEN}
    akt["O_sampling"] = {str(s): B.o_zeiger(n, s * AKTIV_TOUREN) for s in AKTIV_STUFEN}
    band["aktiv"] = akt
    # Symmetrien
    recs = [lade(ZW / f"symm_s{m}.json") for m in SEEDS]
    if any(r is None for r in recs):
        print("Zwischenstand symm unvollständig")
        return 1
    sek["symm"] = round(sum(r["sekunden"] for r in recs), 1)
    refs_ = ref[:SYMM_INST]
    band["symm"] = {"instanzen": SYMM_INST, "K_eins": n * kk, "K_acht": 8 * n * kk, "O_eins": B.o_zeiger(n, n), "O_acht": 8 * B.o_zeiger(n, n),
                    "eins": [_gap(r["eins"], refs_).tolist() for r in recs], "acht": [_gap(r["acht"], refs_).tolist() for r in recs],
                    "merkmal": [_gap(r["merkmal"], refs_).tolist() for r in recs]}
    # n = 50 (Ausblick)
    r50 = lade(ZW / "n50.json")
    if r50 is not None:
        t50 = np.array(wahr["n50"]["laengen"])
        n50 = 50
        k50 = B.k_konstruktion(n50)
        sek["n50"] = r50["sekunden"]
        m50 = {}
        for s_name, kb, ob in (("S1", k50, B.o_merkmal(n50, 1)), ("S2", n50 * k50, B.o_merkmal(n50, n50))):
            key = s_name.lower()
            evs = [np.array(r50["merkmal"][str(m)][key]["ev"]) for m in SEEDS]
            m50[s_name] = {"K": [(kb + e).tolist() for e in evs], "O": [(ob + B.O_JE_DELTA * e).tolist() for e in evs], "ev": [e.tolist() for e in evs],
                           "gap": [_gap(r50["merkmal"][str(m)][key]["laenge"], t50).tolist() for m in SEEDS]}
        kl50 = r50["klass"]
        fL, fEV = np.array(kl50["folge_laenge"]), np.array(kl50["folge_ev"])
        band["n50"] = {"instanzen": N50_INST, "bewiesen": wahr["n50"]["bewiesen"], "merkmal": m50,
                       "klass": {"S4": {"K": (k50 + fEV[:, 0]).tolist(), "O": (B.o_nn(n50) + B.O_JE_DELTA * fEV[:, 0]).tolist(), "ev": fEV[:, 0].tolist(), "gap": _gap(fL[:, 0], t50).tolist()},
                                 "S5": {"K": (n50 * k50 + np.array(kl50["s5"]["ev"])).tolist(), "O": (n50 * B.o_nn(n50) + B.O_JE_DELTA * np.array(kl50["s5"]["ev"])).tolist(),
                                        "ev": kl50["s5"]["ev"], "gap": _gap(kl50["s5"]["laenge"], t50).tolist()},
                                 "S6": {"K": (k50 + np.array(kl50["s6"]["ev"])).tolist(), "O": (B.o_farthest(n50) + B.O_JE_DELTA * np.array(kl50["s6"]["ev"])).tolist(),
                                        "ev": kl50["s6"]["ev"], "gap": _gap(kl50["s6"]["laenge"], t50).tolist()}},
                       "folge": {"gap": np.minimum.accumulate(_gap(fL, t50[:, None]), axis=1).T.tolist(), "K": np.cumsum(k50 + fEV, axis=1).T.tolist()}}
    hashes = {"zeiger": [hashlib.sha256(MO.zeiger_pfad(s).read_bytes()).hexdigest()[:16] for s in SEEDS],
              "merkmal": [hashlib.sha256((ROOT / "rbek_modelle" / f"featnet_s{s}.npz").read_bytes()).hexdigest()[:16] for s in SEEDS]}
    band["kopf"] = {"format": "rbe-band-1", "n": n, "familie": "gleich", "instanzen": N_INST, "instanz_seeds": f"0 bis {N_INST - 1}", "modell_seeds": len(SEEDS),
                    "aktiv_instanzen": AKTIV_INST, "symm_instanzen": SYMM_INST, "n50_instanzen": N50_INST, "rechner": platform.platform(), "python": platform.python_version(),
                    "numpy": np.__version__, "sekunden_je_stufe": sek, "modell_hashes_sha256_16": hashes,
                    "hinweis": "Lücken in Prozent über der exakten Länge (Held-Karp, n = 50: CP-SAT). Netz-Zeilen: je Gewichts-Zufalls-Seed ein Vektor über die Instanzen."}
    BAND.write_text(json.dumps(band, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"rbe_band.json geschrieben: {BAND.stat().st_size / 1e6:.2f} MB")
    return 0


STUFEN = {"netz": stufe_netz, "klass": stufe_klass, "start": stufe_start, "aktiv": stufe_aktiv, "symm": stufe_symm, "n50": stufe_n50}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STUFEN) + ["auswertung", "alle"], default=None)
    a = ap.parse_args()
    if a.stage is None:
        ap.print_help()
        return 2
    if a.stage == "auswertung":
        return stufe_auswertung()
    for name in (STUFEN if a.stage == "alle" else [a.stage]):
        print(f"== Stufe {name}", flush=True)
        STUFEN[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
