"""Das Handbeispiel: fünf Städte, eine ganzzahlige symmetrische Fahrzeitmatrix und ein Mini-Netz mit **zwei** Parametern, alles in Brüchen exakt nachzurechnen.

Das Mini-Netz hat keine versteckte Schicht: die Wahrscheinlichkeit, von Stadt i nach j zu gehen, ist proportional zu exp(-w1 D[i][j] - w2 D[j][s]) (Entfernung zur aktuellen Stadt und
zur Startstadt s) mit w1 = 2 ln 1,5 und w2 = ln 1,5, also proportional zu (2/3) hoch (2 D[i][j] + D[j][s]): rational. Damit lassen sich Sampling, Strahlsuche und der exakte
Gradient der aktiven Suche ohne Näherung bestimmen. Reine Funktionen, keine Zustände; Zufall nur über SplitMix64 (`rbe_rng`)."""

import itertools
import math
from fractions import Fraction
from functools import lru_cache

import numpy as np

import rbe_budget as B
import rbe_exact as E
import rbe_suche as SU
from rbe_rng import SplitMix64

D_HAND = ((0, 5, 2, 3, 8), (5, 0, 9, 8, 4), (2, 9, 0, 4, 3), (3, 8, 4, 0, 1), (8, 4, 3, 1, 0))
N_HAND = 5
Q = Fraction(2, 3)
A_, B_ = 2, 1                      # Gewichte der beiden Merkmale im Exponenten: 2 D[i][j] + D[j][s]
W1, W2 = A_ * math.log(1.5), B_ * math.log(1.5)


def distanzen() -> np.ndarray:
    return np.array(D_HAND, dtype=np.float64)


def laenge(t) -> int:
    return sum(D_HAND[t[i]][t[(i + 1) % N_HAND]] for i in range(N_HAND))


@lru_cache(maxsize=1)
def alle_touren() -> tuple:
    """Die (n - 1)! = 24 Folgen ab Stadt 0."""
    return tuple((0,) + p for p in itertools.permutations(range(1, N_HAND)))


@lru_cache(maxsize=1)
def laengen() -> dict:
    return {t: laenge(t) for t in alle_touren()}


def optimum() -> int:
    """Kürzeste Rundtour (Aufzählung aller 24 Folgen)."""
    return min(laengen().values())


def optimum_held_karp() -> float:
    """Dasselbe mit dem Held-Karp-Verfahren der Kopie `rbe_exact` (zweiter Rechenweg)."""
    return E.held_karp(distanzen())


def naechster_nachbar(start: int) -> list:
    t, gesehen = [start], {start}
    while len(t) < N_HAND:
        c = t[-1]
        j = min((j for j in range(N_HAND) if j not in gesehen), key=lambda j: (D_HAND[c][j], j))
        t.append(j)
        gesehen.add(j)
    return t


def exponent(i: int, j: int, s: int) -> int:
    """Ganzzahliger Exponent 2 D[i][j] + D[j][s] des Mini-Netzes."""
    return A_ * D_HAND[i][j] + B_ * D_HAND[j][s]


def schritt_wsk(i: int, offen, s: int) -> dict:
    """Exakte Wahrscheinlichkeiten (Brüche) des nächsten Ortes über die offenen Städte."""
    gew = {j: Q ** exponent(i, j, s) for j in offen}
    z = sum(gew.values())
    return {j: g / z for j, g in gew.items()}


def mini_greedy(start: int) -> list:
    t, offen = [start], set(range(N_HAND)) - {start}
    while offen:
        c = t[-1]
        j = min(offen, key=lambda j: (exponent(c, j, start), j))
        t.append(j)
        offen.remove(j)
    return t


def folge_wsk(t, start: int = 0) -> Fraction:
    """Exakte Wahrscheinlichkeit der Folge t."""
    offen = set(range(N_HAND)) - {start}
    p = Fraction(1)
    for i in range(1, N_HAND):
        p *= schritt_wsk(t[i - 1], offen, start)[t[i]]
        offen.remove(t[i])
    return p


@lru_cache(maxsize=1)
def wsk() -> dict:
    return {t: folge_wsk(t) for t in alle_touren()}


def erwartete_laenge() -> Fraction:
    """Erwartete Länge einer gezogenen Tour (exakt)."""
    return sum(w * laengen()[t] for t, w in wsk().items())


def p_optimum() -> Fraction:
    return sum(w for t, w in wsk().items() if laengen()[t] == optimum())


def erwartetes_beste(k: int) -> Fraction:
    """E[kürzeste von k unabhängigen Stichproben] (exakt): P(min >= L) = P(Länge >= L)^k."""
    werte = sorted(set(laengen().values()))
    e = Fraction(0)
    for i, L in enumerate(werte):
        p_ge = sum(w for t, w in wsk().items() if laengen()[t] >= L)
        naechste = werte[i + 1] if i + 1 < len(werte) else 10 ** 9
        p_ge_next = sum(w for t, w in wsk().items() if laengen()[t] >= naechste)
        e += L * (p_ge ** k - p_ge_next ** k)
    return e


def erwartetes_beste_paare() -> Fraction:
    """k = 2 durch Aufzählung aller 24 x 24 Paare (zweiter Rechenweg)."""
    w, lg = wsk(), laengen()
    return sum(w[a] * w[b] * min(lg[a], lg[b]) for a in w for b in w)


def strahl_exakt(breite: int, start: int = 0):
    """Strahlsuche über die exakten Wahrscheinlichkeiten: (Endhypothesen [(Folge, Wahrscheinlichkeit)], gezählte Bewertungen K)."""
    hyp, gezaehlt = [([start], Fraction(1))], 0
    for _ in range(N_HAND - 1):
        neu = []
        for t, p in hyp:
            offen = set(range(N_HAND)) - set(t)
            gezaehlt += len(offen)
            for j, pj in schritt_wsk(t[-1], offen, start).items():
                neu.append((t + [j], p * pj))
        neu.sort(key=lambda e: (-e[1], e[0]))
        hyp = neu[:breite]
    return hyp, gezaehlt


def konstruktion_gezaehlt(start: int, zaehler: B.Zaehler) -> list:
    """Greedy-Konstruktion mit dem Mini-Netz, instrumentiert: jede bewertete Zeile (offener Kandidat je Schritt) wird gezählt."""
    t, offen = [start], set(range(N_HAND)) - {start}
    while offen:
        c = t[-1]
        zaehler.zeilen(len(offen))
        p = schritt_wsk(c, offen, start)
        j = max(sorted(p), key=lambda j: p[j])
        t.append(j)
        offen.remove(j)
    return t


def score_exakt(t, start: int = 0) -> tuple:
    """Exakte Score-Funktion d log P / d (w1, w2) der Folge t am Punkt (w1, w2) = (2, 1) ln 1,5 (Brüche): -x_gewählt + Erwartung von x unter p, summiert über die Schritte."""
    offen = set(range(N_HAND)) - {start}
    s1 = s2 = Fraction(0)
    for i in range(1, N_HAND):
        c, j = t[i - 1], t[i]
        p = schritt_wsk(c, offen, start)
        s1 += -D_HAND[c][j] + sum(p[k] * D_HAND[c][k] for k in offen)
        s2 += -D_HAND[j][start] + sum(p[k] * D_HAND[k][start] for k in offen)
        offen.remove(j)
    return s1, s2


def gradient_exakt() -> tuple:
    """Gradient der erwarteten Länge nach (w1, w2), exakt als Score-Funktion über alle 24 Folgen: Summe P(t) (L(t) - J) score(t). Der Referenzwert J ändert ihn nicht."""
    J0 = erwartete_laenge()
    g1 = sum(w * (laengen()[t] - J0) * score_exakt(t)[0] for t, w in wsk().items())
    g2 = sum(w * (laengen()[t] - J0) * score_exakt(t)[1] for t, w in wsk().items())
    return g1, g2


def j_float(w1: float, w2: float) -> float:
    """Erwartete Länge als Funktion der Parameter (Gleitkomma, Aufzählung aller 24 Folgen)."""
    gesamt = 0.0
    for t in alle_touren():
        p, offen = 1.0, [j for j in range(N_HAND) if j != 0]
        for i in range(1, N_HAND):
            c = t[i - 1]
            gew = {j: math.exp(-w1 * D_HAND[c][j] - w2 * D_HAND[j][0]) for j in offen}
            p *= gew[t[i]] / sum(gew.values())
            offen.remove(t[i])
        gesamt += p * laenge(t)
    return gesamt


def gradient_zentral(h: float = 1e-6) -> tuple:
    """Zentrale Differenz der erwarteten Länge (dritter Rechenweg)."""
    return ((j_float(W1 + h, W2) - j_float(W1 - h, W2)) / (2 * h), (j_float(W1, W2 + h) - j_float(W1, W2 - h)) / (2 * h))


def stichprobe(rng: SplitMix64, start: int = 0):
    """Eine gezogene Folge (Temperatur 1, Gleitkomma) und ihre Score-Funktion (Summe über die Schritte)."""
    t, offen, s1, s2 = [start], [j for j in range(N_HAND) if j != start], 0.0, 0.0
    while offen:
        c = t[-1]
        gew = [math.exp(-W1 * D_HAND[c][j] - W2 * D_HAND[j][start]) for j in offen]
        z = sum(gew)
        u, cum, wahl = rng.random() * z, 0.0, len(offen) - 1
        for idx, g in enumerate(gew):
            cum += g
            if u < cum:
                wahl = idx
                break
        j = offen[wahl]
        s1 += -D_HAND[c][j] + sum(g * D_HAND[c][k] for g, k in zip(gew, offen)) / z
        s2 += -D_HAND[j][start] + sum(g * D_HAND[k][start] for g, k in zip(gew, offen)) / z
        t.append(j)
        offen.pop(wahl)
    return t, (s1, s2)


def verbesserung(tour) -> dict:
    """Suche (2-opt + Or-opt, erste Verbesserung) ab `tour` auf der Handinstanz: Endtour, Länge, angenommene Züge, alle Zugauswertungen (letzte Runde = M(5) = 50)."""
    ctx = SU.kontext(distanzen())
    return SU.verbessere(ctx, np.array(tour))
