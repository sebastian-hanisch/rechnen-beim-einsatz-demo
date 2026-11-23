"""Die klassische Seite: Startlösungen mit Verbesserung (2-opt + Or-opt mit erster Verbesserung, ohne Nachbarlisten) und die Neustart-Suche als einfacher Gegner.

Die Zugmenge und ihre Deltas kommen unverändert aus der Kopie `rbek_moves` (Stück 4); der Abstieg ist hier der **Durchlauf** der klassischen Umsetzung (nicht der Abstieg mit Neubeginn der Liste
nach jedem Zug aus `rbek_search`, der für die Frage der Reihenfolge gebaut ist). Zählung: eine **Zugauswertung** ist ein bewertetes Delta (K = 1); die letzte Runde eines Abstiegs, die **Prüfrunde**, bewertet alle
M(n) = n (11 n - 35) / 2 verschiedenen Züge (1 850 bei n = 20, 12 875 bei n = 50) und beweist das Lokalminimum. Eine Zufallstour zählt 1 Kandidatenbewertung.

Gegner im Sinne von DEMO-PLAYBOOK §4: nicht abgestimmt (keine Nachbarlisten, keine Don't-look-Bits, kein Annealing); die Aussagen gelten für diese einfachste lokale Suche."""

import numpy as np

import rbe_budget as B
import rbek_constants as KC
import rbek_construct as KCON
import rbek_moves as M

S_EPS = -KC.EPS      # ein Zug verbessert, wenn Delta < -EPS


def kontext(D) -> M.Kontext:
    """Zugmenge, Maßstab und Rangtabelle für eine Instanz (D: (n, n))."""
    return M.Kontext(np.asarray(D, dtype=np.float64))


def durchlauf(ctx: M.Kontext, t0, grenze: float | None = None, protokoll: list | None = None) -> dict:
    """Verbesserung mit erster Verbesserung in Durchläufen, 2-opt vor Or-opt (die Umsetzung der klassischen Seite der Linie, wie `znz_gegner.two_opt_k` erweitert um Or-opt):

    * **2-opt-Phase:** die 2-opt-Züge der Zugliste werden der Reihe nach bewertet; der erste verbessernde wird angenommen und der Durchlauf geht **hinter diesem Zug weiter** (die Deltas werden
      für die neue Tour neu gerechnet, die Liste beginnt nicht von vorn). Wurde in einem Durchlauf ein Zug angenommen, folgt ein neuer Durchlauf; ein voller Durchlauf ohne Verbesserung beendet die Phase.
    * **Or-opt-Phase:** die Or-opt-Züge werden von vorn bewertet; der erste verbessernde wird angenommen, danach beginnt wieder die 2-opt-Phase. Findet die Phase nichts, ist die Tour ein Lokalminimum
      der ganzen Zugmenge: die letzten beiden Durchläufe zusammen sind die **Prüfrunde** und bewerten alle M(n) = n (11 n - 35) / 2 verschiedenen Züge.

    Jede bewertete Stelle der Zugliste ist eine Zugauswertung (K = 1). `grenze`: höchstens so viele Zugauswertungen; bei Überschreitung bricht der Lauf ab (`abgebrochen`), die bis dahin angenommenen
    Züge gelten. Mit einer Liste als `protokoll` wird je angenommenem Zug ein Eintrag (Tour danach, Delta, Zugauswertungen seit dem vorigen Zug, Zugauswertungen insgesamt) angehängt.
    Rückgabe: tour, laenge, zuege, ev (alle Zugauswertungen), ev_pruef (M), lokalmin, abgebrochen."""
    t = np.asarray(t0).copy()
    nb = ctx.nb
    m = nb.M
    m2 = int(np.count_nonzero(nb.art == 0))            # die 2-opt-Züge stehen am Anfang der Zugliste
    ev, zuege, p, verbessert, seit, phase = 0, 0, 0, False, 0, "2opt"

    def abbruch():
        return {"tour": t, "laenge": float(M.tour_length(ctx.D, t)), "zuege": zuege, "ev": int(grenze), "ev_pruef": 0, "lokalmin": False, "abgebrochen": True}

    while True:
        delta = M.deltas(ctx.D, t, nb)
        von, bis = (p, m2) if phase == "2opt" else (m2, m)
        treffer = np.nonzero(delta[von:bis] < S_EPS)[0]
        if treffer.size:
            idx = von + int(treffer[0])
            kosten = idx - von + 1
            if grenze is not None and ev + kosten > grenze:
                return abbruch()
            ev += kosten
            seit += kosten
            t = M.apply_move(t, idx, nb)
            zuege += 1
            if protokoll is not None:
                protokoll.append({"tour": t.copy(), "delta": float(delta[idx]), "stelle": seit, "ev": ev})
            seit = 0
            if phase == "2opt":
                p, verbessert = idx + 1, True
            else:
                phase, p, verbessert = "2opt", 0, False
            continue
        rest = bis - von
        if grenze is not None and ev + rest > grenze:
            return abbruch()
        ev += rest
        seit += rest
        if phase == "or":
            return {"tour": t, "laenge": float(M.tour_length(ctx.D, t)), "zuege": zuege, "ev": ev, "ev_pruef": m, "lokalmin": True, "abgebrochen": False}
        if verbessert:
            p, verbessert = 0, False
        else:
            phase = "or"


def verbessere(ctx: M.Kontext, tour) -> dict:
    """Verbesserung bis zum Lokalminimum (`durchlauf`): tour, laenge, zuege, ev (alle Zugauswertungen einschließlich des Beweises), ev_pruef (M(n)), lokalmin (immer wahr)."""
    r = durchlauf(ctx, tour)
    return {k: r[k] for k in ("tour", "laenge", "zuege", "ev", "ev_pruef", "lokalmin")}


def nn_tour(D, start: int = 0) -> np.ndarray:
    return KCON.nn_tour_np(D, start)


def farthest_insertion(D, start: int = 0) -> np.ndarray:
    return KCON.farthest_insertion(D, start)


def neustart_suche(ctx: M.Kontext, us, budget: float):
    """Neustart-Suche: Zufallstour (1 Kandidatenbewertung), Verbesserung in Durchläufen bis zum Lokalminimum, neue Zufallstour, ... bis `budget` Kandidatenbewertungen verbraucht sind.

    Rückgabe (K, L): die kumulierten Kandidatenbewertungen und die Länge der aktuellen Tour nach jedem Ereignis (neue Zufallstour, angenommener Zug). Ein angenommener Zug gehört zu dem K,
    bei dem er gefunden wurde; der Beweis am Ende eines Abstiegs ändert die Tour nicht und steht nur in K. Zufall aus `us` (UStream)."""
    n = ctx.n
    ks, ls = [], []
    k = 0
    while k < budget:
        t = us.random_order(1, n)[0]
        k += 1
        ks.append(k)
        ls.append(float(M.tour_length(ctx.D, t)))
        prot: list = []
        r = durchlauf(ctx, t, grenze=budget - k, protokoll=prot)
        for e in prot:
            ks.append(k + e["ev"])
            ls.append(float(M.tour_length(ctx.D, e["tour"])))
        k += r["ev"]
    return np.array(ks, dtype=np.int64), np.array(ls)


def beste_bei(ks: np.ndarray, ls: np.ndarray, budget: float) -> float:
    """Beste Länge unter den Ereignissen mit K <= budget (`inf`, wenn noch keines)."""
    m = ks <= budget
    return float(ls[m].min()) if m.any() else float("inf")


def beste_bei_raster(ks: np.ndarray, ls: np.ndarray, raster) -> np.ndarray:
    """`beste_bei` für viele Budgets: (len(raster),)."""
    lauf = np.minimum.accumulate(ls)
    pos = np.searchsorted(ks, np.asarray(raster), side="right") - 1
    return np.where(pos >= 0, lauf[np.maximum(pos, 0)], np.inf)


def pruefrunde(n: int) -> int:
    """Zugauswertungen einer vollen Prüfrunde (verschiedene Züge): n (11 n - 35) / 2."""
    return B.K0.m_verschiedene_zuege(n)
