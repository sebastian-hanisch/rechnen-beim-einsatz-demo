"""Training: Adam, Referenzwert „Mittel“, ein REINFORCE-Schritt, die Trainingsschleife mit Rückmeldung, Zeitwächter und Validierungskurve.

Je Schritt `batch` frische Instanzen mit je einer gezogenen Tour (Start Stadt 0); Verlust F = Summe (L - b)/B * log p(Tour), Schritt gegen den Gradienten.
Zufall: Gewichte `UStream(derive(ZWECK_INIT, d, kodierercode, init_seed))`, Instanzen `UStream(derive(ZWECK_TRAIN_X, n, data_seed))`,
Ziehungen `UStream(derive(ZWECK_TRAIN_S, n, data_seed))`. Der Referenzwert ist über `baseline_fn(X, starts, L)` austauschbar (Stück 2 baut darauf)."""

import math
import time

import numpy as np

import rbe_constants as C
import rbe_net as N
from rbe_instances import train_batch
from rbe_rngvec import UStream
from rbe_scenario import derive


class Adam:
    """Adam mit Begrenzung der Euklid-Norm des Gesamtgradienten auf `clip`."""

    def __init__(self, P: dict, lr: float = C.LERNRATE, b1: float = C.ADAM_B1, b2: float = C.ADAM_B2, eps: float = C.ADAM_EPS, clip: float | None = C.NORM_BEGRENZUNG):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    def step(self, P: dict, G: dict) -> float:
        """Ein Schritt gegen den Gradienten G (verändert P an Ort und Stelle); gibt die Norm des Gradienten vor der Begrenzung zurück."""
        gn = math.sqrt(sum(float((g ** 2).sum()) for g in G.values()))
        f = self.clip / gn if (self.clip and gn > self.clip) else 1.0
        self.t += 1
        for k in P:
            g = G[k] * f
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return gn


def mean_baseline(X, starts, L):
    """Referenzwert „Mittel“: das Mittel aller Tourlängen des Schritts (ein Skalar, unabhängig von der einzelnen Tour)."""
    return float(L.mean())


def reinforce_step(P: dict, opt: Adam, X: np.ndarray, starts: np.ndarray, ustream: UStream, baseline_fn=mean_baseline) -> dict:
    """Ein REINFORCE-Schritt: Touren ziehen, Längen messen, Referenzwert abziehen, Gradient, Adam-Schritt.

    Rückgabe {"L" (M, K), "b", "grad_norm", "logp", "tours"}. Die Gewichte w = (L - b) / (M K)."""
    H, ecache = N.encode(P, X)
    out = N.decode(P, H, starts, mode="sample", u_stream=ustream)
    L = N.tour_lengths(X, out["tours"])
    b = baseline_fn(X, starts, L)
    w = (L - b) / L.size
    Gd, dH = N.decode_backward(P, out["cache"], w)
    gn = opt.step(P, {**N.encode_backward(P, ecache, dH), **Gd})
    return {"L": L, "b": b, "grad_norm": gn, "logp": out["logp"], "tours": out["tours"]}


def gap_greedy(P: dict, X: np.ndarray, opt: np.ndarray) -> float:
    """Mittlere Lücke der Greedy-Touren (Start Stadt 0) zur exakten Länge `opt` in Prozent (Mittel der Lücken je Instanz)."""
    starts = np.zeros((X.shape[0], 1), dtype=np.int64)
    L = N.tour_lengths(X, N.greedy_tours(P, X, starts))[:, 0]
    return float(np.mean(L / opt - 1.0) * 100.0)


def end_gap(history: list, last: int = C.LETZTE_MESSPUNKTE) -> float | None:
    """Endwert der Lernkurve: Mittel der Lücken der letzten `last` Messpunkte (None ohne Messpunkte)."""
    gaps = [h["val_gap"] for h in history if h.get("val_gap") is not None]
    return float(np.mean(gaps[-last:])) if gaps else None


def train(n: int, steps: int, batch: int = C.BATCH, d: int = 32, layers: int = 1, attn: bool = True, ffn: bool = True, lr: float = C.LERNRATE,
          init_seed: int = 0, data_seed: int = 0, val=None, eval_every: int = C.EVAL_ALLE, progress=None, max_seconds: float | None = None,
          baseline_fn=mean_baseline, progress_every: int = 25):
    """Trainiert ein Zeigernetz mit REINFORCE. Rückgabe (P, history, counts).

    val: None oder (X_val, opt_val) mit exakten Längen; die Kurve `history` enthält dann {"step", "val_gap", "touren"} bei Schritt 0, alle `eval_every` Schritte und am Ende.
    progress(step, steps, touren) wird alle `progress_every` Schritte und am Ende aufgerufen. max_seconds: Zeitwächter; nach Ablauf endet die Schleife,
    das Zwischenmodell wird zurückgegeben (counts["abgebrochen"]). counts: {"touren", "entscheidungen", "schritte", "abgebrochen", "sekunden"}."""
    if not (attn or ffn):
        layers = 0
    P = N.init_params(d, UStream(derive(C.ZWECK_INIT, d, N.variant_code(layers, attn, ffn), init_seed)), layers=layers, attn=attn, ffn=ffn)
    opt = Adam(P, lr)
    xs = UStream(derive(C.ZWECK_TRAIN_X, n, data_seed))
    ss = UStream(derive(C.ZWECK_TRAIN_S, n, data_seed))
    starts = np.zeros((batch, 1), dtype=np.int64)
    history = []
    t0 = time.perf_counter()

    def messen(step):
        if val is not None and len(val[0]) > 0:          # ohne Validierungsinstanzen keine Kurve
            history.append({"step": step, "val_gap": gap_greedy(P, val[0], val[1]), "touren": step * batch})

    messen(0)
    erledigt = 0
    abgebrochen = False
    for step in range(1, steps + 1):
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            abgebrochen = True
            break
        reinforce_step(P, opt, train_batch(xs, batch, n), starts, ss, baseline_fn)
        erledigt = step
        if eval_every and step % eval_every == 0:
            messen(step)
        if progress is not None and (step % progress_every == 0 or step == steps):
            progress(step, steps, step * batch)
    if eval_every and erledigt % eval_every != 0:
        messen(erledigt)
    touren = erledigt * batch
    counts = {"touren": touren, "entscheidungen": touren * (n - 1), "schritte": erledigt, "abgebrochen": abgebrochen, "sekunden": time.perf_counter() - t0}
    return P, history, counts
