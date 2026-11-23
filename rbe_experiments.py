"""Die Messungen der Demo als Funktionen (dieselben Funktionen rechnen die eingefrorene Messreihe `tools/rbe_make_band.py` und die Klicks der App).

Instanzen: 20 Orte gleichverteilt im Einheitsquadrat (Quelle `rbe_scenario.make_instance("gleich", 20, Zufalls-Seed)`, durch 1000 geteilt); Test-Zufalls-Seeds 0 bis 199 (App und Messung),
Zufalls-Seeds ab 1000 gehören zur Validierung des Trainings. Gaps in Prozent über der exakten Länge (Held-Karp). Zeit ist kein Befund; gezählt wird in K und O (`rbe_budget`)."""

import numpy as np

import rbe_budget as B
import rbe_construct as CO
import rbe_instances as I
import rbe_modelle as MO
import rbe_stats as ST
import rbe_suche as SU
from rbe_net import tour_lengths

N_STADT = MO.N_STADT
FAMILIE = "gleich"
VERFAHREN = ("greedy", "mehrfach", "sampling", "strahl")
VERFAHREN_ANZEIGE = {"greedy": "Eine Tour (Greedy)", "mehrfach": "Mehrfachstart (Greedy ab k Startstädten)", "sampling": "Sampling (k gezogene Touren)", "strahl": "Strahlsuche (Beam Search) der Breite k"}


def instanzen(seeds, familie: str = FAMILIE, n: int = N_STADT) -> np.ndarray:
    """Instanzen (M, n, 2) im Einheitsquadrat für die Zufalls-Seeds `seeds`."""
    return np.stack([I.instance_unit(familie, n, int(s)) for s in seeds])


def distanzen(X: np.ndarray) -> np.ndarray:
    return I.distance_matrix_np(X)


def luecke(laengen, opt) -> np.ndarray:
    """Lücke in Prozent über der exakten Länge, je Instanz."""
    return ST.gap(laengen, np.asarray(opt, dtype=np.float64))


# ---------------------------------------------------------------------------------------------------------------- Kosten der Netz-Verfahren
def kosten_netz(netz: str, verfahren: str, k: int, n: int = N_STADT, symm: int = 1) -> tuple[int, float]:
    """(K, O) eines Netz-Verfahrens je Instanz. greedy: eine Tour; mehrfach: k Startstädte (mal `symm` Abbildungen, je Abbildung ein Kodierer); sampling: k Touren, ein Kodierer;
    strahl: Breite k. K zählt die bewerteten Kandidaten (Konstruktion n (n - 1) / 2), O die groben Operationen des Netzes."""
    if verfahren == "greedy":
        k, verfahren = 1, "mehrfach"
    if verfahren in ("mehrfach", "sampling"):
        m = k * symm if verfahren == "mehrfach" else k
        if netz == "zeiger":
            o = B.o_zeiger(n, m) if verfahren == "sampling" or symm == 1 else symm * B.o_zeiger(n, k)
        else:
            o = B.o_merkmal(n, m)
        return B.k_mehrfach(n, m), float(o)
    if verfahren == "strahl":
        o = B.o_zeiger_strahl(n, k) if netz == "zeiger" else B.o_merkmal_strahl(n, k)
        return B.k_strahl(n, k), float(o)
    raise ValueError(verfahren)


# ---------------------------------------------------------------------------------------------------------------- Netz-Verfahren auf Instanzen
def _prefix_beste(L: np.ndarray, tours: np.ndarray, ks):
    """Für jedes k die kürzeste unter den ersten k Touren je Instanz: {k: (Länge (M,), Tour (M, n))}."""
    mi = np.arange(L.shape[0])
    out = {}
    for k in ks:
        j = L[:, :k].argmin(axis=1)
        out[k] = (L[mi, j], tours[mi, j])
    return out


def netz_laengen(netz: str, modell: int, X: np.ndarray, D: np.ndarray, seeds, verfahren: str, ks, symm: int = 1) -> dict:
    """Beste Länge und Tour je Instanz für ein Netz-Verfahren und jedes k in `ks`: {k: (Länge (M,), Tour (M, n))}.

    `seeds`: Zufalls-Seeds der Instanzen (bestimmen die Ziehungen). Mehrfachstart und Sampling sind verschachtelt (die ersten k von kmax Touren: mehr ist nie schlechter),
    die Strahlsuche rechnet jede Breite für sich. `symm` = 8 nur beim Zeigernetz im Mehrfachstart: dieselbe Rechnung für die acht Abbildungen des Einheitsquadrats."""
    P, _ = MO.lade(netz, modell)
    M, n, _ = X.shape
    ks = sorted(set(int(k) for k in ks))
    if verfahren == "greedy":
        return netz_laengen(netz, modell, X, D, seeds, "mehrfach", [1])
    if verfahren == "strahl":
        out = {}
        for k in ks:
            tours, lebend, _, _ = CO.zeiger_strahl(P, X, k) if netz == "zeiger" else CO.merkmal_strahl(P, D, k)
            out[k] = CO.bester_der_strahlen(X, tours, lebend)
        return out
    kmax = ks[-1]
    if verfahren == "mehrfach":
        starts = CO.alle_starts(M, n, kmax)
        if netz == "zeiger":
            if symm == 1:
                tours = CO.zeiger_greedy(P, X, starts)
                return _prefix_beste(CO.laenge(X, tours), tours, ks)
            abb = CO.symmetrien(X)
            alle = [CO.zeiger_greedy(P, abb[j], starts) for j in range(symm)]
            tours = np.concatenate(alle, axis=1)                     # (M, symm * kmax, n): Abbildung 0 zuerst
            L = CO.laenge(X, tours).reshape(M, symm, kmax)
            mi = np.arange(M)
            out = {}
            for k in ks:
                flach = L[:, :, :k].reshape(M, -1)
                j = flach.argmin(axis=1)
                tt = tours.reshape(M, symm, kmax, n)[:, :, :k].reshape(M, -1, n)
                out[k] = (flach[mi, j], tt[mi, j])
            return out
        tours, _ = CO.merkmal_touren(P, D, starts)
        return _prefix_beste(CO.laenge(X, tours), tours, ks)
    if verfahren == "sampling":
        if netz == "zeiger":
            tours = CO.zeiger_gezogen(P, X, kmax, seeds, modell)
        else:
            starts = np.zeros((M, kmax), dtype=np.int64)
            tours, _ = CO.merkmal_touren(P, D, starts, modus="sample", instanzen=seeds, modell=modell)
        return _prefix_beste(CO.laenge(X, tours), tours, ks)
    raise ValueError(verfahren)


# ---------------------------------------------------------------------------------------------------------------- Klassische Verfahren auf Instanzen
def nn_laengen(D: np.ndarray, ks) -> dict:
    """Nächster Nachbar aus den ersten k Startstädten, beste Länge je Instanz: {k: (Länge (M,), Tour (M, n))}."""
    M, n = D.shape[0], D.shape[-1]
    kmax = max(ks)
    L = np.empty((M, kmax))
    T = np.empty((M, kmax, n), dtype=np.int64)
    for i in range(M):
        for s in range(kmax):
            t = SU.nn_tour(D[i], s)
            T[i, s] = t
            L[i, s] = D[i][t, np.roll(t, -1)].sum()
    return _prefix_beste(L, T, sorted(set(int(k) for k in ks)))


def fi_laengen(D: np.ndarray, start: int = 0):
    """Farthest Insertion ab `start`: (Länge (M,), Tour (M, n))."""
    T = np.stack([SU.farthest_insertion(D[i], start) for i in range(D.shape[0])])
    L = np.array([D[i][T[i], np.roll(T[i], -1)].sum() for i in range(D.shape[0])])
    return L, T


def zufall_beste(D: np.ndarray, budgets, strom) -> dict:
    """Beste von b Zufallstouren (je 1 Kandidatenbewertung) für jedes b in `budgets`: {b: Länge (M,)}. Die ersten b von bmax Touren je Instanz (verschachtelt)."""
    bmax = max(budgets)
    M, n = D.shape[0], D.shape[-1]
    out = {int(b): np.empty(M) for b in budgets}
    for i in range(M):
        T = strom(i).random_order(bmax, n)
        L = D[i][T, np.roll(T, -1, axis=1)].sum(axis=1)
        lauf = np.minimum.accumulate(L)
        for b in budgets:
            out[int(b)][i] = lauf[int(b) - 1]
    return out


def verbessere_start(D: np.ndarray, starts: np.ndarray, konstruktionen_k: int) -> dict:
    """Verbesserung (2-opt + Or-opt) ab einer Starttour je Instanz: starts (M, n). Rückgabe: laenge (M,), ev (M, Zugauswertungen einschließlich Prüfrunde), k_gesamt
    (M, = Konstruktion `konstruktionen_k` + ev), zuege (M,)."""
    M = D.shape[0]
    laenge, ev, zuege = np.empty(M), np.empty(M), np.empty(M)
    for i in range(M):
        r = SU.verbessere(SU.kontext(D[i]), starts[i])
        laenge[i], ev[i], zuege[i] = r["laenge"], r["ev"], r["zuege"]
    return {"laenge": laenge, "ev": ev, "k_gesamt": konstruktionen_k + ev, "zuege": zuege}


# ---------------------------------------------------------------------------------------------------------------- Vergleich
def vergleich(gap_a, gap_b, seed: int = 6) -> dict:
    """Gepaarter Vergleich A gegen B über Instanzen (Mittel der Differenz in Prozentpunkten, Bootstrap-Intervall, Klasse): negativ heißt A besser."""
    return ST.compare(np.asarray(gap_a), np.asarray(gap_b), seed)


def bester_tour(X: np.ndarray, tours: np.ndarray) -> np.ndarray:
    return tour_lengths(X, tours[:, None, :])[:, 0]
