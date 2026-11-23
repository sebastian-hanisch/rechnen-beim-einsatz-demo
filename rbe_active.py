"""Aktive Suche (Active Search, Bello u. a. 2016): das Zeigernetz wird beim Einsatz auf **einer** Instanz weitertrainiert.

Je Schritt werden k Touren aus dem Netz gezogen (ab Stadt 0, Temperatur 1), der Referenzwert ist das Mittel der k Längen, ein Adam-Schritt entgegen dem REINFORCE-Gradienten
verändert alle Gewichte des Netzes. Ergebnis ist die kürzeste Tour aller bisher gezogenen Touren. Der Vergleich ist das Sampling mit derselben Zahl Touren aus dem unveränderten
Anfangsmodell. Der Rückwärtspass (`rbe_net.loss_and_grad` bzw. `rbe_train.reinforce_step`) ist der des Trainings (Gradienten-Check in den Tests).

Nur das Zeigernetz: das Merkmalsnetz K hat in der Quelle keinen Rückwärtspass für die Konstruktion in dieser Demo (nicht gemessen)."""

import numpy as np

import rbe_net as N
import rbe_train as T
from rbe_rngvec import UStream


def aktive_suche(P0: dict, x: np.ndarray, schritte: int, touren: int, lr: float, us: UStream) -> dict:
    """Aktive Suche auf der Instanz x (n, 2) mit `touren` Touren je Schritt.

    Rückgabe: beste (kürzeste Länge bis einschließlich Schritt s, Form (schritte,)), mittel (mittlere Länge der Touren des Schritts), tour (beste Tour), laenge (beste Länge),
    gezogen (Zahl Touren = schritte x touren). `P0` bleibt unverändert."""
    P = N.copy_params(P0)
    opt = T.Adam(P, lr)
    X = np.asarray(x, dtype=np.float64)[None]
    starts = np.zeros((1, touren), dtype=np.int64)
    beste, mittel = [], []
    best_len, best_tour = np.inf, None
    for _ in range(schritte):
        res = T.reinforce_step(P, opt, X, starts, us)
        L = res["L"][0]
        j = int(L.argmin())
        if L[j] < best_len:
            best_len, best_tour = float(L[j]), res["tours"][0, j].copy()
        beste.append(best_len)
        mittel.append(float(L.mean()))
    return {"beste": np.array(beste), "mittel": np.array(mittel), "tour": best_tour, "laenge": best_len, "gezogen": schritte * touren}


def sampling_gleiche_touren(P0: dict, x: np.ndarray, schritte: int, touren: int, us: UStream) -> dict:
    """Sampling mit derselben Zahl Touren aus dem unveränderten Anfangsmodell (schritte x touren Touren ab Stadt 0), beste Länge bis einschließlich Schritt s."""
    X = np.asarray(x, dtype=np.float64)[None]
    k = schritte * touren
    tours = N.sample_tours(P0, X, k, us)
    L = N.tour_lengths(X, tours)[0].reshape(schritte, touren)
    beste = np.minimum.accumulate(L.min(axis=1))
    return {"beste": beste, "mittel": L.mean(axis=1), "laenge": float(beste[-1]), "gezogen": k}
