"""Das Zeigernetz (Pointer Network) von Hand in numpy: Kodierer, Dekodierer mit Maske, Vorwärts- und Rückwärtspass, Modelldatei.

Aufbau (PLAN-01, Z2). Eingabe X der Form (M, n, 2) im Einheitsquadrat, absolute Koordinaten (der Aufrufer normiert mit `rbe_instances.normalize_unit`, wenn er will).
Starts der Form (M, K), Touren (M, K, n), Längen (M, K): M Instanzen, K Touren je Instanz.

Kodierer: h_i = x_i W0 + b0; je Schicht Attention (Q, K, V, Wo; ein Kopf, Rest-Verbindung) und Vorwärtsnetz (W1, b1, ReLU, W2, b2; Rest-Verbindung); beide Teile einzeln abschaltbar
(fehlen ihre Schlüssel in P, entfällt der Teil: die Variante steckt in den Schlüsseln, `describe(P)` liest sie ab). Dekodierer: Kontext c_t = mittlerer Stadtvektor Wg + Startstadt Wf +
aktuelle Stadt Wl (`Wc` ist Wg, Wf, Wl übereinander, 3d x d), Schlüssel k_j = H_j Wkd, Wert s = c . k, u = C tanh(s / sqrt d), Softmax nur über die freien Orte (Maske).

Zufall nur über einen übergebenen `UStream` (`rbe_rngvec`); kein numpy.random. Alle Gradienten von Hand; der Gradienten-Check gegen endliche Differenzen steht in tests/."""

import json
import math
import re

import numpy as np

import rbe_constants as C

C_CLIP = C.C_CLIP


# ----------------------------------------------------------------------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------------------------------------------------------------------
def init_params(d: int, rng, layers: int = 1, attn: bool = True, ffn: bool = True, dtype=np.float64) -> dict:
    """Gewichte: Normalverteilung mit Streuung 1/sqrt(Eingangszahl), Biases null. `rng` hat `normal(shape)` (z. B. `UStream`).

    Reihenfolge der Ziehungen: W0, je Schicht (Wq, Wk, Wv, Wo, dann W1, W2), Wc, Wkd. Ohne Attention und ohne Vorwärtsnetz gibt es keine Schicht (layers = 0)."""
    def lin(i, o):
        return (rng.normal((i, o)) * (1.0 / math.sqrt(i))).astype(dtype)

    if not (attn or ffn):
        layers = 0
    P = {"W0": lin(2, d), "b0": np.zeros(d, dtype)}
    for layer in range(layers):
        if attn:
            for name in ("Wq", "Wk", "Wv", "Wo"):
                P[f"{name}{layer}"] = lin(d, d)
        if ffn:
            P[f"W1{layer}"] = lin(d, 2 * d)
            P[f"b1{layer}"] = np.zeros(2 * d, dtype)
            P[f"W2{layer}"] = lin(2 * d, d)
            P[f"b2{layer}"] = np.zeros(d, dtype)
    P["Wc"] = lin(3 * d, d)
    P["Wkd"] = lin(d, d)
    return P


def n_params(P: dict) -> int:
    return int(sum(v.size for v in P.values()))


def copy_params(P: dict) -> dict:
    return {k: v.copy() for k, v in P.items()}


_SCHICHT_RE = re.compile(r"^(Wq|W1)(\d+)$")


def describe(P: dict) -> dict:
    """Aufbau aus den Schlüsseln: {"d", "layers", "attn", "ffn"}."""
    layers, attn, ffn = 0, False, False
    for k in P:
        m = _SCHICHT_RE.match(k)
        if m:
            layers = max(layers, int(m.group(2)) + 1)
            attn = attn or m.group(1) == "Wq"
            ffn = ffn or m.group(1) == "W1"
    return {"d": int(P["W0"].shape[1]), "layers": layers, "attn": attn, "ffn": ffn}


def expected_shapes(d: int, layers: int, attn: bool, ffn: bool) -> dict:
    """Formen aller Parameter einer Variante (für die Prüfung beim Laden)."""
    shapes = {"W0": (2, d), "b0": (d,)}
    for layer in range(layers):
        if attn:
            for name in ("Wq", "Wk", "Wv", "Wo"):
                shapes[f"{name}{layer}"] = (d, d)
        if ffn:
            shapes.update({f"W1{layer}": (d, 2 * d), f"b1{layer}": (2 * d,), f"W2{layer}": (2 * d, d), f"b2{layer}": (d,)})
    shapes["Wc"] = (3 * d, d)
    shapes["Wkd"] = (d, d)
    return shapes


def param_count(d: int, layers: int = 1, attn: bool = True, ffn: bool = True) -> int:
    """Zahl der Parameter einer Variante aus den Formen (12 480 für d = 32 mit Schicht)."""
    if not (attn or ffn):
        layers = 0
    return int(sum(int(np.prod(s)) for s in expected_shapes(d, layers, attn, ffn).values()))


def variant_code(layers: int, attn: bool, ffn: bool) -> int:
    """Kodierercode für die Zufalls-Seeds: 0 Einbettung, 1 nur Attention, 2 nur Vorwärtsnetz, 3 beides; je weitere Schicht + 4."""
    if not (attn or ffn) or layers == 0:
        return 0
    return int(attn) + 2 * int(ffn) + 4 * (layers - 1)


def softmax_last(a: np.ndarray) -> np.ndarray:
    a = a - a.max(-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(-1, keepdims=True)


# Das Mini-Netz des Handbeispiels: d = 2, keine Schicht, 22 Parameter (PLAN-01, Teil 5; Wc = Wg, Wf, Wl übereinander)
_HAND = {
    "W0": [[0.5, -0.5], [0.5, 1.0]],
    "b0": [0.1, -0.1],
    "Wg": [[0.2, 0.1], [-0.1, 0.3]],
    "Wf": [[0.4, -0.2], [0.1, 0.2]],
    "Wl": [[1.0, -0.5], [0.5, 1.0]],
    "Wkd": [[1.0, 0.5], [-0.5, 1.0]],
}


def hand_params() -> dict:
    """Die festen Gewichte des Mini-Netzes (Handbeispiel), als Parameterwörterbuch (float64)."""
    return {
        "W0": np.array(_HAND["W0"]), "b0": np.array(_HAND["b0"]),
        "Wc": np.array(_HAND["Wg"] + _HAND["Wf"] + _HAND["Wl"]), "Wkd": np.array(_HAND["Wkd"]),
    }


def _n_layers(P: dict) -> int:
    return describe(P)["layers"]


# ----------------------------------------------------------------------------------------------------------------------------------------
# Kodierer
# ----------------------------------------------------------------------------------------------------------------------------------------
def encode(P: dict, X: np.ndarray, keep: bool = True):
    """X (M, n, 2) -> H (M, n, d) und Cache für den Rückwärtspass."""
    d = P["W0"].shape[1]
    sc = 1.0 / math.sqrt(d)
    H = X @ P["W0"] + P["b0"]
    caches = []
    for layer in range(_n_layers(P)):
        c = {"Hin": H}
        if f"Wq{layer}" in P:
            Q = H @ P[f"Wq{layer}"]
            Kk = H @ P[f"Wk{layer}"]
            V = H @ P[f"Wv{layer}"]
            A = softmax_last((Q @ Kk.transpose(0, 2, 1)) * sc)
            AV = A @ V
            H1 = H + AV @ P[f"Wo{layer}"]
            c.update(Q=Q, Kk=Kk, V=V, A=A, AV=AV)
        else:
            H1 = H
        c["H1"] = H1
        if f"W1{layer}" in P:
            Z = H1 @ P[f"W1{layer}"] + P[f"b1{layer}"]
            R = np.maximum(Z, 0.0)
            H = H1 + R @ P[f"W2{layer}"] + P[f"b2{layer}"]
            c.update(Z=Z, R=R)
        else:
            H = H1
        if keep:
            caches.append(c)
    return H, (X, caches)


def _wgrad(Xin: np.ndarray, dY: np.ndarray) -> np.ndarray:
    """Gewichtsgradient: Summe über (M, n) von Xin^T dY."""
    return np.einsum("mni,mnj->ij", Xin, dY)


def encode_backward(P: dict, cache, dH: np.ndarray) -> dict:
    """Gradienten der Kodierer-Parameter (alle Schlüssel des Kodierers) für gegebenes dF/dH."""
    X, caches = cache
    d = P["W0"].shape[1]
    sc = 1.0 / math.sqrt(d)
    G = {}
    for layer in reversed(range(len(caches))):
        c = caches[layer]
        if f"W1{layer}" in P:
            G[f"W2{layer}"] = _wgrad(c["R"], dH)
            G[f"b2{layer}"] = dH.sum((0, 1))
            dZ = (dH @ P[f"W2{layer}"].T) * (c["Z"] > 0)
            G[f"W1{layer}"] = _wgrad(c["H1"], dZ)
            G[f"b1{layer}"] = dZ.sum((0, 1))
            dH1 = dH + dZ @ P[f"W1{layer}"].T
        else:
            dH1 = dH
        if f"Wq{layer}" in P:
            Q, Kk, V, A, AV = c["Q"], c["Kk"], c["V"], c["A"], c["AV"]
            G[f"Wo{layer}"] = _wgrad(AV, dH1)
            dO = dH1 @ P[f"Wo{layer}"].T
            dA = dO @ V.transpose(0, 2, 1)
            dV = A.transpose(0, 2, 1) @ dO
            dS = A * (dA - (dA * A).sum(-1, keepdims=True))
            dQ = (dS @ Kk) * sc
            dK = (dS.transpose(0, 2, 1) @ Q) * sc
            G[f"Wq{layer}"] = _wgrad(c["Hin"], dQ)
            G[f"Wk{layer}"] = _wgrad(c["Hin"], dK)
            G[f"Wv{layer}"] = _wgrad(c["Hin"], dV)
            dH = dH1 + dQ @ P[f"Wq{layer}"].T + dK @ P[f"Wk{layer}"].T + dV @ P[f"Wv{layer}"].T
        else:
            dH = dH1
    G["W0"] = np.einsum("mni,mnj->ij", X, dH)
    G["b0"] = dH.sum((0, 1))
    return G


# ----------------------------------------------------------------------------------------------------------------------------------------
# Dekodierer
# ----------------------------------------------------------------------------------------------------------------------------------------
def _split_wc(P: dict, d: int):
    Wc = P["Wc"]
    return Wc[:d], Wc[d:2 * d], Wc[2 * d:]


def decoder_context(P: dict, H: np.ndarray, starts: np.ndarray) -> dict:
    """Alles, was der Dekodierer über die Schritte hinweg gleich lässt: Schlüssel, Mittel der Stadtvektoren, Kontext der Startstadt (M, K, d)."""
    M, n, d = H.shape
    Wg, Wf, _ = _split_wc(P, d)
    mi = np.arange(M)[:, None]
    kp = H @ P["Wkd"]
    g = H.mean(1)
    hf = H[mi, starts]
    Cg = (g @ Wg)[:, None, :] + hf @ Wf
    return {"H": H, "kp": kp, "g": g, "hf": hf, "Cg": Cg, "starts": starts, "d": d, "n": n}


def _step(P: dict, ctx: dict, cur: np.ndarray, visited: np.ndarray):
    """Ein Zeigerschritt: Kontext c, begrenzter Wert u, Wahrscheinlichkeiten p (0 bei besuchten Orten), Log-Wahrscheinlichkeiten (-inf bei besuchten)."""
    H, d = ctx["H"], ctx["d"]
    _, _, Wl = _split_wc(P, d)
    M = H.shape[0]
    mi = np.arange(M)[:, None]
    c = ctx["Cg"] + H[mi, cur] @ Wl
    s = c @ ctx["kp"].transpose(0, 2, 1)
    u = C_CLIP * np.tanh(s * (1.0 / math.sqrt(d)))
    ul = np.where(visited, -np.inf, u)
    m = ul.max(-1, keepdims=True)
    e = np.exp(ul - m)
    S = e.sum(-1, keepdims=True)
    return c, u, e / S, (ul - m) - np.log(S)


def step_logp(P: dict, ctx: dict, cur: np.ndarray, visited: np.ndarray) -> np.ndarray:
    """Log-Wahrscheinlichkeiten des nächsten Ortes (M, K, n); -inf bei besuchten Orten (cur (M, K), visited (M, K, n) boolesch)."""
    return _step(P, ctx, cur, visited)[3]


def decode(P: dict, H: np.ndarray, starts: np.ndarray, mode: str = "greedy", u_stream=None, forced=None, keep: bool = True) -> dict:
    """Baut Touren. mode: "greedy" (größte Wahrscheinlichkeit, Gleichstand kleinster Index), "sample" (Inversionsverfahren, liest je Schritt
    `u_stream.uniform((M, K, 1))`), "forced" (vorgegebene Touren `forced` (M, K, n), nur Log-Wahrscheinlichkeiten).

    Rückgabe {"tours" (M, K, n), "logp" (M, K) Summe der Log-Wahrscheinlichkeiten, "cache"} (cache None bei keep=False)."""
    M, n, d = H.shape
    K = starts.shape[1]
    ctx = decoder_context(P, H, starts)
    mi = np.arange(M)[:, None]
    ki = np.arange(K)[None, :]
    visited = np.zeros((M, K, n), dtype=bool)
    visited[mi, ki, starts] = True
    cur = starts.copy()
    tours = np.empty((M, K, n), dtype=np.int64)
    tours[:, :, 0] = starts
    logp = np.zeros((M, K))
    cs, us, ps = [], [], []
    for t in range(n - 1):
        c, u, p, lp = _step(P, ctx, cur, visited)
        if mode == "forced":
            a = forced[:, :, t + 1]
        elif mode == "greedy":
            a = p.argmax(-1)
        elif mode == "sample":
            cum = p.cumsum(-1)
            r = (1.0 - u_stream.uniform((M, K, 1))) * cum[..., -1:]
            a = np.minimum((cum < r).sum(-1), n - 1)
        else:
            raise ValueError(f"unbekannter Modus {mode!r}")
        logp += np.take_along_axis(lp, a[..., None], -1)[..., 0]
        visited[mi, ki, a] = True
        cur = a
        tours[:, :, t + 1] = a
        if keep:
            cs.append(c)
            us.append(u)
            ps.append(p)
    cache = None
    if keep:
        cache = {"ctx": ctx, "tours": tours, "cs": np.array(cs), "us": np.array(us), "ps": np.array(ps)}
    return {"tours": tours, "logp": logp, "cache": cache}


def decode_backward(P: dict, cache: dict, w: np.ndarray):
    """Gradient von F = Summe w[m, k] * logp[m, k] nach den Dekodierer-Parametern (Wc, Wkd) und nach H; w (M, K).

    Je Zeigerschritt: dF/ds_j = w (delta_ja - p_j) (C - u_j^2 / C) / sqrt(d) für die freien Orte j; danach Produktregel durch s = c . k, die Kontextsumme und H.
    Rückgabe (G, dH). REINFORCE: w = (L - b) / (M K)."""
    ctx = cache["ctx"]
    H, kp, g, hf, starts = ctx["H"], ctx["kp"], ctx["g"], ctx["hf"], ctx["starts"]
    tours, cs, us, ps = cache["tours"], cache["cs"], cache["us"], cache["ps"]
    M, n, d = H.shape
    T = n - 1
    Wg, Wf, Wl = _split_wc(P, d)
    sc = 1.0 / math.sqrt(d)
    eye = np.eye(n)
    acts = tours[:, :, 1:]
    onehot = eye[acts].transpose(2, 0, 1, 3)                      # (T, M, K, n)
    dlog = (onehot - ps) * w[None, :, :, None]
    DS = dlog * (C_CLIP - us ** 2 / C_CLIP) * sc                  # (T, M, K, n): nach s = c . kp
    dkp = np.einsum("tmkn,tmkd->mnd", DS, cs)
    dc = np.einsum("tmkn,mnd->tmkd", DS, kp)
    curs = tours[:, :, :T]
    mi = np.arange(M)[:, None, None]
    hcs = H[mi, curs].transpose(2, 0, 1, 3)                       # (T, M, K, d)
    G = {"Wkd": np.einsum("mnd,mne->de", H, dkp)}
    dH = dkp @ P["Wkd"].T
    dWl = np.einsum("tmkd,tmke->de", hcs, dc)
    dhc = dc @ Wl.T
    oh_cur = eye[curs].transpose(2, 0, 1, 3)
    dH += np.einsum("tmkn,tmkd->mnd", oh_cur, dhc)
    dCg = dc.sum(0)                                               # (M, K, d)
    dWg = np.einsum("md,mke->de", g, dCg)
    dg = np.einsum("mke,de->md", dCg, Wg)
    dH += dg[:, None, :] / n
    dWf = np.einsum("mkd,mke->de", hf, dCg)
    dhf = dCg @ Wf.T
    dH += np.einsum("mkn,mkd->mnd", eye[starts], dhf)
    G["Wc"] = np.concatenate([dWg, dWf, dWl], axis=0)
    return G, dH


def loss_and_grad(P: dict, X: np.ndarray, starts: np.ndarray, w: np.ndarray, tours: np.ndarray):
    """F(theta) = Summe w * logp(vorgegebene Touren) und der Gradient aller Parameter."""
    H, ecache = encode(P, X)
    out = decode(P, H, starts, mode="forced", forced=tours)
    F = float((w * out["logp"]).sum())
    Gd, dH = decode_backward(P, out["cache"], w)
    Ge = encode_backward(P, ecache, dH)
    return F, {**Ge, **Gd}


# ----------------------------------------------------------------------------------------------------------------------------------------
# Einsatz und Hilfen
# ----------------------------------------------------------------------------------------------------------------------------------------
def tour_lengths(X: np.ndarray, tours: np.ndarray) -> np.ndarray:
    """Geschlossene Tourlängen: X (M, n, 2), tours (M, K, n) -> (M, K)."""
    mi = np.arange(X.shape[0])[:, None, None]
    P = X[mi, tours]
    Q = np.roll(P, -1, axis=2)
    return np.sqrt(((P - Q) ** 2).sum(-1)).sum(-1)


def is_permutation(tours: np.ndarray) -> bool:
    """Ist jede Tour (letzte Achse) eine Permutation von 0..n-1?"""
    n = tours.shape[-1]
    return bool(np.all(np.sort(tours, axis=-1) == np.arange(n)))


def greedy_tours(P: dict, X: np.ndarray, starts: np.ndarray, chunk: int = 250) -> np.ndarray:
    """Greedy-Touren (M, K, n); die Instanzen werden in Blöcken zu `chunk` gerechnet (Ergebnis unabhängig vom Block)."""
    out = []
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        H, _ = encode(P, Xc, keep=False)
        out.append(decode(P, H, starts[i:i + chunk], mode="greedy", keep=False)["tours"])
    return np.concatenate(out)


def sample_tours(P: dict, X: np.ndarray, k: int, u_stream, start: int = 0) -> np.ndarray:
    """k gezogene Touren je Instanz ab Stadt `start`, Form (M, k, n). Je Schritt eine Ziehung (M, k, 1) aus `u_stream` (Reihenfolge: Schritt, Instanz, Tour)."""
    H, _ = encode(P, X, keep=False)
    starts = np.full((X.shape[0], k), start, dtype=np.int64)
    return decode(P, H, starts, mode="sample", u_stream=u_stream, keep=False)["tours"]


def probabilities_along(P: dict, x: np.ndarray, tour, with_u: bool = False):
    """Wahrscheinlichkeiten p_t(j) der Schritte 1 bis n-1 entlang der Tour `tour` für eine Instanz x (n, 2): Form (n-1, n), besuchte Orte 0.
    Mit `with_u=True` zusätzlich die begrenzten Werte u_t(j) (gleiche Form, bei besuchten Orten der Wert vor der Maske)."""
    tours = np.asarray(tour, dtype=np.int64)[None, None, :]
    H, _ = encode(P, np.asarray(x, dtype=np.float64)[None], keep=False)
    cache = decode(P, H, tours[:, :, 0], mode="forced", forced=tours, keep=True)["cache"]
    ps = cache["ps"][:, 0, 0, :]
    return (ps, cache["us"][:, 0, 0, :]) if with_u else ps


# ----------------------------------------------------------------------------------------------------------------------------------------
# Modelldatei (Format znz-modell-1)
# ----------------------------------------------------------------------------------------------------------------------------------------
def save_model(path, P: dict, meta: dict) -> None:
    """Speichert Parameter (float64) und `meta` (JSON-Text) in einer .npz-Datei; `meta` bekommt format, d, layers, attn, ffn und n_params, wenn sie fehlen."""
    info = describe(P)
    meta = {"format": C.MODELL_FORMAT, **info, "n_params": n_params(P), **meta}
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in P.items()}
    np.savez_compressed(path, **arrays, meta=np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False)))


def load_model(path):
    """Lädt (P, meta). Prüft Format, Formen der Parameter gegen `describe` und `n_params` gegen `meta`; sonst ValueError."""
    with np.load(path, allow_pickle=False) as z:
        if "meta" not in z.files:
            raise ValueError("Modelldatei ohne meta")
        meta = json.loads(str(z["meta"]))
        P = {k: np.array(z[k], dtype=np.float64) for k in z.files if k != "meta"}
    if meta.get("format") != C.MODELL_FORMAT:
        raise ValueError(f"unbekanntes Format {meta.get('format')!r}")
    if "W0" not in P or P["W0"].ndim != 2:
        raise ValueError("Parameter W0 fehlt oder hat falsche Form")
    info = describe(P)
    soll = expected_shapes(info["d"], info["layers"], info["attn"], info["ffn"])
    if set(P) != set(soll):
        raise ValueError(f"Parameter {sorted(set(P) ^ set(soll))} passen nicht zur Variante")
    for k, shape in soll.items():
        if P[k].shape != shape:
            raise ValueError(f"Parameter {k} hat die Form {P[k].shape}, erwartet {shape}")
    if int(meta.get("n_params", -1)) != n_params(P):
        raise ValueError("n_params in meta stimmt nicht mit den Parametern überein")
    return P, meta
