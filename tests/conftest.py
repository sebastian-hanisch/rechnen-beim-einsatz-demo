"""Gemeinsame Fixtures: das Handbeispiel (fünf Städte, ganzzahlige Fahrzeitmatrix) und die Zugliste als zweiter Rechenweg.

Die Zugliste (Weg 1, Listenoperationen) und die Positionsfelder (Weg 2, `rbek_moves`) werden gegeneinander geprüft; die Zugliste liegt nur hier in den Tests."""
# ruff: noqa: E402

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rbe_hand as H


@pytest.fixture
def hand_D():
    return H.distanzen()


def tour_len(D, t) -> float:
    """Länge einer Tour in reinem Python (zweiter Weg, kein numpy-Indexieren)."""
    return sum(D[t[k]][t[(k + 1) % len(t)]] for k in range(len(t)))


def kanon(t):
    """Kanonische Form eines Rundwegs (Start bei 0, kleinere Richtung)."""
    k = list(t).index(0)
    u = list(t)[k:] + list(t)[:k]
    v = [u[0]] + u[1:][::-1]
    return tuple(min(u, v))


def zuege_listen(t):
    """Alle verschiedenen 2-opt- und Or-opt-Züge (Länge 1 bis 3) als neue Touren (Listenoperationen); die Verschiebung eines Einzelorts zählt nur in einer Richtung."""
    t = list(t)
    n = len(t)
    out = []
    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            out.append(t[: i + 1] + t[i + 1: j + 1][::-1] + t[j + 1:])
    for L in (1, 2, 3):
        for i in range(n):
            seg = [t[(i + k) % n] for k in range(L)]
            rest = [t[(i + L + k) % n] for k in range(n - L)]
            for pos in range(len(rest) - 1):
                for rev in (0, 1):
                    if L == 1 and rev == 1:
                        continue
                    s = seg[::-1] if rev else seg
                    out.append(rest[: pos + 1] + s + rest[pos + 1:])
    return out


def zufalls_instanz(n: int, seed: int) -> np.ndarray:
    """Zufallsinstanz im Einheitsquadrat aus dem Quellstrom (SplitMix64, nicht numpy.random): (n, 2)."""
    import rbe_instances as I
    from rbe_rngvec import UStream
    return I.train_batch(UStream(seed), 1, n)[0]


def zufalls_D(n: int, seed: int) -> np.ndarray:
    import rbe_instances as I
    return I.distance_matrix_np(zufalls_instanz(n, seed))
