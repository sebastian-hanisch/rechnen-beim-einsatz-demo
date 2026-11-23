"""Die Rechnungen, die die App bei jedem Klick live macht (kleine Fassungen der Messungen in `rbe_experiments`): Instanzen, Rechengrenze, Experiment 1 bis 5.

Jede Funktion hat explizite Ein- und Ausgaben (keine Streamlit-Aufrufe), die App legt die Ergebnisse mit `st.cache_data` ab. Instanzen: 16 Instanzen je Zufalls-Seed der App
(Test-Zufalls-Seed 16 s bis 16 s + 15), exakte Längen aus `rbe_truth.json`. Rechengrenze: höchstens `RECHENGRENZE_K` Kandidatenbewertungen je Klick (Netz und Suche zusammen)."""

import numpy as np

import rbe_active as AK
import rbe_band as BD
import rbe_budget as B
import rbe_constants as C
import rbe_experiments as EX
import rbe_modelle as MO
import rbe_suche as SU
from rbe_rngvec import UStream
from rbe_scenario import derive

N = EX.N_STADT


def instanz_seeds(seed: int, anzahl: int) -> tuple:
    """Test-Zufalls-Seeds der live gerechneten Instanzen: 16 seed, ..., 16 seed + anzahl - 1."""
    return tuple(C.INSTANZEN_JE_SEED * int(seed) + i for i in range(int(anzahl)))


def daten(seeds) -> tuple:
    """(X (M, n, 2), D (M, n, n), exakte Längen (M,)) der Instanzen mit den Test-Zufalls-Seeds `seeds`."""
    seeds = list(seeds)
    X = EX.instanzen(seeds)
    return X, EX.distanzen(X), BD.wahrheit_n20()[seeds]


def plan_runs(wunsch: int, k_je_instanz: float, grenze: int = C.RECHENGRENZE_K) -> tuple[int, str | None]:
    """Wie viele Instanzen dürfen bei `k_je_instanz` Kandidatenbewertungen je Instanz gerechnet werden (höchstens `wunsch`, mindestens 1)? Rückgabe (Zahl, Hinweis oder None)."""
    erlaubt = max(1, int(grenze // max(1.0, k_je_instanz)))
    m = min(int(wunsch), erlaubt)
    if m < wunsch:
        return m, (f"Rechenaufwand: {int(k_je_instanz):,} Kandidatenbewertungen je Instanz, Instanzen von {wunsch} auf {m} gekürzt (Grenze {grenze:,} je Klick).".replace(",", " "))
    return m, None


def _zufall(D, seeds, anzahl: int = 1):
    return EX.zufall_beste(D, [anzahl], lambda i: UStream(derive(C.ZWECK_ZUFALLSTOUREN, 0, seeds[i])))[anzahl]


# ---------------------------------------------------------------------------------------------------------------- Experiment 1
def e1(seeds, modell: int) -> list:
    """Eine Tour jedes Verfahrens auf den Instanzen: Liste von {name, art, K, O, gap (M,), tour (M, n) oder None}. Zufallstour: Mittel einer Zufallstour (1 Kandidatenbewertung)."""
    X, D, ref = daten(seeds)
    kk = B.k_konstruktion(N)
    nn_l, nn_t = EX.nn_laengen(D, [1])[1]
    fi_l, fi_t = EX.fi_laengen(D)
    zeile = [{"name": "Zufallstour", "art": "zufall", "K": 1, "O": B.o_zufall(1), "gap": EX.luecke(_zufall(D, seeds), ref), "tour": None},
             {"name": "Nächster Nachbar", "art": "nn", "K": kk, "O": B.o_nn(N), "gap": EX.luecke(nn_l, ref), "tour": nn_t},
             {"name": "Farthest Insertion", "art": "fi", "K": kk, "O": B.o_farthest(N), "gap": EX.luecke(fi_l, ref), "tour": fi_t}]
    for netz in MO.NETZE:
        lg, t = EX.netz_laengen(netz, modell, X, D, list(seeds), "greedy", [1])[1]
        k, o = EX.kosten_netz(netz, "greedy", 1)
        zeile.append({"name": MO.NETZ_ANZEIGE[netz], "art": netz, "K": k, "O": o, "gap": EX.luecke(lg, ref), "tour": t})
    return zeile


# ---------------------------------------------------------------------------------------------------------------- Experiment 2
def e2(seeds, netz: str, modell: int, verfahren: str, k: int, symm: int = 1) -> dict:
    """Netz-Verfahren mit k Konstruktionen gegen den Nächsten Nachbarn aus k Startstädten (höchstens 20) und, wenn die Zahl der Zufallstouren (k x 190) nicht über 65 536 liegt,
    gegen die beste von k x 190 Zufallstouren. Mehrfachstart: k höchstens 20."""
    X, D, ref = daten(seeds)
    k_eff = min(k, N) if verfahren == "mehrfach" else k
    symm_eff = symm if (netz == "zeiger" and verfahren == "mehrfach") else 1
    res = EX.netz_laengen(netz, modell, X, D, list(seeds), verfahren, [k_eff], symm=symm_eff)[k_eff]
    kost_k, kost_o = EX.kosten_netz(netz, verfahren, k_eff, symm=symm_eff)
    k_nn = min(k_eff * symm_eff, N) if verfahren == "mehrfach" else min(k_eff, N)
    nn_l, nn_t = EX.nn_laengen(D, [k_nn])[k_nn]
    out = {"gap_netz": EX.luecke(res[0], ref), "tour_netz": res[1], "K": kost_k, "O": kost_o, "k_eff": k_eff, "symm": symm_eff, "gap_nn": EX.luecke(nn_l, ref), "tour_nn": nn_t,
           "k_nn": k_nn, "K_nn": k_nn * B.k_konstruktion(N), "O_nn": B.o_nn(N) * k_nn, "ref": ref, "gap_zufall": None}
    b = kost_k  # Zufallstouren mit demselben K (1 je Tour)
    if b <= 65536:
        out["gap_zufall"] = EX.luecke(EX.zufall_beste(D, [b], lambda i: UStream(derive(C.ZWECK_ZUFALLSTOUREN, 0, seeds[i])))[b], ref)
    return out


def strahl_hypothesen(netz: str, modell: int, seed: int, breite: int) -> list:
    """Lebende Hypothesen der Strahlsuche je Schritt für eine Instanz: Liste je Schritt t = 1 .. n - 1 von (Touren (B, t + 1), Log-Wahrscheinlichkeiten (B,)); nicht lebende fehlen."""
    import rbe_construct as CO
    X, D, _ = daten([seed])
    P, _ = MO.lade(netz, modell)
    prot: list = []
    if netz == "zeiger":
        CO.zeiger_strahl(P, X, breite, protokoll=prot)
    else:
        CO.merkmal_strahl(P, D, breite, protokoll=prot)
    out = []
    for tours, score in prot:
        leb = np.isfinite(score[0])
        out.append((tours[0][leb], score[0][leb]))
    return out


# ---------------------------------------------------------------------------------------------------------------- Experiment 3 (kleine Fassung live)
def e3_live(seeds, netz: str, gap_netz: np.ndarray, kost_k: int, kost_o: float, c: float, anzahl: int = 4, k_max: int = 100_000) -> dict | None:
    """Die Neustart-Suche auf den ersten `anzahl` Instanzen bei demselben K und bei demselben O (geteilt durch c Operationen je Delta). None, wenn ein Budget über `k_max` liegt."""
    b_k, b_o = int(kost_k), int(kost_o // c)
    if max(b_k, b_o) > k_max:
        return None
    X, D, ref = daten(seeds[:anzahl])
    gk, go = np.empty(len(ref)), np.empty(len(ref))
    for i in range(len(ref)):
        ks, ls = SU.neustart_suche(SU.kontext(D[i]), UStream(derive(C.ZWECK_ZUFALLSTOUREN, 1, seeds[i])), max(b_k, b_o))
        gk[i] = SU.beste_bei(ks, ls, b_k)
        go[i] = SU.beste_bei(ks, ls, b_o)
    return {"gap_suche_k": EX.luecke(gk, ref), "gap_suche_o": EX.luecke(go, ref), "gap_netz": gap_netz[:len(ref)], "budget_k": b_k, "budget_o": b_o, "anzahl": len(ref)}


# ---------------------------------------------------------------------------------------------------------------- Experiment 4
START_NAMEN = {"s4": "Nächster Nachbar ab Ort 0", "s5": "Nächster Nachbar, beste aus 20", "s6": "Farthest Insertion", "netz1": "Greedy ab Ort 0", "netz2": "beste aus 20 Greedy-Touren",
               "netz3": "Sampling, beste aus 8"}


def e4(seeds, netz: str, modell: int) -> dict:
    """Netz baut, Suche verbessert (2-opt + Or-opt) auf den Instanzen: für jede Startlösung Start-Lücke, End-Lücke, Zugauswertungen, Gesamt-K und Gesamt-O je Instanz sowie die Starttouren."""
    import rbe_construct as CO
    X, D, ref = daten(seeds)
    M = len(ref)
    kk = B.k_konstruktion(N)
    P, _ = MO.lade(netz, modell)
    o_netz = (lambda m: B.o_zeiger(N, m)) if netz == "zeiger" else (lambda m: B.o_merkmal(N, m))
    starts_all = CO.alle_starts(M, N)
    tours_all = CO.zeiger_greedy(P, X, starts_all) if netz == "zeiger" else CO.merkmal_touren(P, D, starts_all)[0]
    L_all = CO.laenge(X, tours_all)
    mi = np.arange(M)
    nn_all = np.stack([np.stack([SU.nn_tour(D[i], s) for s in range(N)]) for i in range(M)])
    nn_L = np.array([[D[i][t, np.roll(t, -1)].sum() for t in nn_all[i]] for i in range(M)])
    fi = np.stack([SU.farthest_insertion(D[i], 0) for i in range(M)])
    samp = EX.netz_laengen(netz, modell, X, D, list(seeds), "sampling", [8])[8]
    quellen = {
        "s4": (nn_all[:, 0], kk, B.o_nn(N), nn_L[:, 0]),
        "s5": (nn_all[mi, nn_L.argmin(axis=1)], N * kk, N * B.o_nn(N), nn_L.min(axis=1)),
        "s6": (fi, kk, B.o_farthest(N), np.array([D[i][fi[i], np.roll(fi[i], -1)].sum() for i in range(M)])),
        "netz1": (tours_all[:, 0], kk, o_netz(1), L_all[:, 0]),
        "netz2": (tours_all[mi, L_all.argmin(axis=1)], N * kk, o_netz(N), L_all.min(axis=1)),
        "netz3": (samp[1], 8 * kk, o_netz(8), samp[0]),
    }
    out = {"ref": ref, "starts": {}}
    for name, (touren, k_bau, o_bau, l_start) in quellen.items():
        v = EX.verbessere_start(D, touren, k_bau)
        out["starts"][name] = {"tour_start": touren, "gap_start": EX.luecke(l_start, ref), "gap_end": EX.luecke(v["laenge"], ref), "ev": v["ev"], "zuege": v["zuege"],
                               "K": v["k_gesamt"], "O": o_bau + B.O_JE_DELTA * v["ev"], "k_bau": k_bau, "o_bau": o_bau}
    return out


def abstieg(D: np.ndarray, start_tour) -> dict:
    """Die Verbesserung (Durchläufe) Zug für Zug auf einer Instanz: {touren (Start und nach jedem Zug), deltas, stellen (Zugauswertungen seit dem vorigen Zug), ev, ev_pruef, laenge, lokalmin}."""
    import rbek_moves as M
    ctx = SU.kontext(D)
    prot: list = []
    r = SU.durchlauf(ctx, np.asarray(start_tour), protokoll=prot)
    return {"touren": [np.asarray(start_tour)] + [e["tour"] for e in prot], "deltas": [e["delta"] for e in prot], "stellen": [e["stelle"] for e in prot], "ev": r["ev"],
            "ev_pruef": r["ev_pruef"], "laenge": float(M.tour_length(D, r["tour"])), "lokalmin": r["lokalmin"]}


# ---------------------------------------------------------------------------------------------------------------- Experiment 5
def e5(seeds, modell: int, schritte: int, touren: int, lr: float, max_sekunden: float = C.ZEITWAECHTER_SEKUNDEN, fortschritt=None) -> dict:
    """Aktive Suche (Zeigernetz wird auf der Instanz weitertrainiert) gegen Sampling gleicher Tourzahl auf den Instanzen `seeds`. Zeitwächter: nach `max_sekunden` Sekunden bricht die
    Schleife über die Instanzen ab (die schon gerechneten gelten)."""
    import time
    t0 = time.perf_counter()
    X, D, ref = daten(seeds)
    P, _ = MO.lade_zeiger(modell)
    beste_a, beste_s, fertig = [], [], 0
    for i in range(len(ref)):
        if time.perf_counter() - t0 > max_sekunden and fertig >= 1:
            break
        a = AK.aktive_suche(P, X[i], schritte, touren, lr, UStream(derive(C.ZWECK_AKTIV, 0, modell, seeds[i])))
        s = AK.sampling_gleiche_touren(P, X[i], schritte, touren, UStream(derive(C.ZWECK_AKTIV, 1, modell, seeds[i])))
        beste_a.append(a["beste"])
        beste_s.append(s["beste"])
        fertig += 1
        if fortschritt is not None:
            fortschritt(fertig, len(ref))
    ref = ref[:fertig]
    ba, bs = np.array(beste_a), np.array(beste_s)
    return {"gap_aktiv": 100.0 * (ba - ref[:, None]) / ref[:, None], "gap_sampling": 100.0 * (bs - ref[:, None]) / ref[:, None], "instanzen": fertig, "gewuenscht": len(seeds),
            "abgebrochen": fertig < len(seeds), "touren_je_instanz": schritte * touren, "K_je_instanz": B.k_aktiv(N, schritte, touren), "O_je_instanz": B.o_aktiv(N, schritte, touren),
            "O_sampling_je_instanz": B.o_zeiger(N, schritte * touren)}
