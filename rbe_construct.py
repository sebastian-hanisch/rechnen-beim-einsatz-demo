"""Die Verfahren, mit denen ein Netz beim Einsatz mehr rechnen kann: Greedy, Mehrfachstart, Sampling (gezogene Touren), Strahlsuche (Beam Search), Symmetrien des Einheitsquadrats.

Jede Einheit ist eine Funktion mit expliziten Ein- und Ausgaben. Beide Netze (Zeigernetz aus `rbe_net`, Merkmalsnetz K aus `rbek_featnet`) bauen Touren Schritt für Schritt aus
Wahrscheinlichkeiten über die noch offenen Orte; die Strahlsuche ist für beide dieselbe Funktion `strahl` mit einer Funktion der Log-Wahrscheinlichkeiten.

Zufall nur über SplitMix64 (`UStream`). **Je Instanz ein eigener Strom** (`strom(netz, modell, instanz)`): die Ziehungen einer Instanz hängen nicht davon ab, welche anderen Instanzen im selben
Aufruf mitgerechnet werden, und die ersten k von K gezogenen Touren sind dieselben (verschachtelt: mehr Sampling-Touren sind nie schlechter)."""

import numpy as np

import rbe_constants as C
import rbe_net as N
import rbek_featnet as F
import rbek_mlp as MLP
from rbe_rngvec import UStream
from rbe_scenario import derive

NETZ_CODE = {"zeiger": 0, "merkmal": 1}
MAX_ZEILEN = 4096        # höchstens so viele Touren (Zeilen) gleichzeitig im Merkmalsnetz (Speicher: je Zeile n x n Zahlen)
CHUNK_ZEIGER = 10        # Instanzen je Block beim Zeigernetz (Ergebnis hängt nicht vom Block ab)


def strom(netz: str, modell: int, instanz: int) -> UStream:
    """Der Strom der Ziehungen einer Instanz für ein Netz und einen Gewichts-Zufalls-Seed."""
    return UStream(derive(C.ZWECK_SAMPLE_EINSATZ, NETZ_CODE[netz], int(modell), int(instanz)))


class _Schichten:
    """Ersatz für den Strom in `decode(mode="sample")`: liefert je Aufruf die nächste Schicht der vorab gezogenen Gleichverteilten (m, k, T) als (m, k, 1)."""

    def __init__(self, u: np.ndarray):
        self.u, self.t = u, 0

    def uniform(self, shape):
        out = self.u[:, :, self.t:self.t + 1]
        self.t += 1
        assert out.shape == tuple(shape)
        return out


class _Gumbel:
    """Ersatz für den Strom in `rollout(mode="sample")`: je Aufruf die nächste Schicht der vorab gezogenen Gumbel-Zahlen (R, T, n) als (R, n)."""

    def __init__(self, g: np.ndarray):
        self.g, self.t = g, 0

    def gumbel(self, shape):
        out = self.g[:, self.t, :]
        self.t += 1
        assert out.shape == tuple(shape)
        return out


def massstab(D: np.ndarray) -> np.ndarray:
    """Maßstab s des Merkmalsnetzes je Instanz: mittlerer Abstand zum nächsten Nachbarn; D (M, n, n) oder (n, n)."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[-1]
    return (D + np.diag(np.full(n, np.inf))).min(axis=-1).mean(axis=-1)


def laenge(X: np.ndarray, tours: np.ndarray) -> np.ndarray:
    """Geschlossene Tourlängen: X (M, n, 2), tours (M, K, n) -> (M, K)."""
    return N.tour_lengths(X, tours)


def laufendes_minimum(L: np.ndarray) -> np.ndarray:
    """Beste Länge unter den ersten 1, 2, ..., K Touren je Instanz: (M, K) -> (M, K)."""
    return np.minimum.accumulate(L, axis=1)


# ---------------------------------------------------------------------------------------------------------------- Zeigernetz
def zeiger_greedy(P: dict, X: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Greedy-Touren des Zeigernetzes: X (M, n, 2), starts (M, K) -> Touren (M, K, n). Mehrfachstart = alle n Startstädte als `starts`."""
    return N.greedy_tours(P, X, starts)


def alle_starts(m: int, n: int, k: int | None = None) -> np.ndarray:
    """Startstädte 0 .. k - 1 für m Instanzen (Mehrfachstart mit k Startstädten, Standard alle n): Form (m, k)."""
    k = n if k is None else k
    return np.tile(np.arange(k, dtype=np.int64), (m, 1))


def zeiger_gezogen(P: dict, X: np.ndarray, k: int, instanzen, modell: int, start=0) -> np.ndarray:
    """k gezogene Touren je Instanz ab Stadt `start` (Temperatur 1; eine Zahl oder eine Stadt je Instanz): X (M, n, 2) -> (M, k, n). `instanzen`: Nummern der Instanzen (bestimmen die Ströme)."""
    M, n, _ = X.shape
    start_je = np.broadcast_to(np.asarray(start, dtype=np.int64), (M,))
    out = []
    for a in range(0, M, CHUNK_ZEIGER):
        sl = slice(a, a + CHUNK_ZEIGER)
        u = np.stack([strom("zeiger", modell, i).uniform((k, n - 1)) for i in instanzen[sl]])
        H, _ = N.encode(P, X[sl], keep=False)
        starts = np.repeat(start_je[sl][:, None], k, axis=1)
        out.append(N.decode(P, H, starts, mode="sample", u_stream=_Schichten(u), keep=False)["tours"])
    return np.concatenate(out)


def zeiger_strahl(P: dict, X: np.ndarray, breite: int, start: int = 0, protokoll: list | None = None):
    """Strahlsuche mit dem Zeigernetz: X (M, n, 2) -> (Touren (M, B, n), lebend (M, B) boolesch, gezählte Bewertungen K je Instanz, Summe der Hypothesen je Instanz)."""
    M, n, _ = X.shape
    tours, lebend, k_ges, h_sum = [], [], 0, 0
    for a in range(0, M, max(1, CHUNK_ZEIGER // 2)):
        Xc = X[a:a + max(1, CHUNK_ZEIGER // 2)]
        H, _ = N.encode(P, Xc, keep=False)
        ctx = N.decoder_context(P, H, np.full((Xc.shape[0], breite), start, dtype=np.int64))

        def logp(cur, visited, t, ctx=ctx):
            return N.step_logp(P, ctx, cur, visited)

        t_, l_, k_ges, h_sum = strahl(logp, Xc.shape[0], n, breite, start, protokoll)
        tours.append(t_)
        lebend.append(l_)
    return np.concatenate(tours), np.concatenate(lebend), k_ges, h_sum


# ---------------------------------------------------------------------------------------------------------------- Merkmalsnetz K
def _zeilen_blocks(m: int, je_instanz: int):
    """Blöcke von Instanzen mit höchstens MAX_ZEILEN Zeilen (mindestens eine Instanz)."""
    schritt = max(1, MAX_ZEILEN // max(1, je_instanz))
    return [slice(a, min(a + schritt, m)) for a in range(0, m, schritt)]


def merkmal_touren(params, D: np.ndarray, starts: np.ndarray, modus: str = "greedy", instanzen=None, modell: int = 0):
    """Touren des Merkmalsnetzes K: D (M, n, n), starts (M, K) -> (Touren (M, K, n), Zahl der Merkmalszeilen insgesamt).

    modus "greedy" oder "sample" (Gumbel-Max-Ziehung aus dem Strom je Instanz, `instanzen` nötig; die K Touren einer Instanz gehören zu den K Startstädten in `starts`)."""
    M, K = starts.shape
    n = D.shape[-1]
    s_inst = massstab(D)
    tours = np.empty((M, K, n), dtype=np.int64)
    zeilen = 0
    for sl in _zeilen_blocks(M, K):
        m = len(range(*sl.indices(M)))
        DD = np.repeat(D[sl], K, axis=0)
        s = np.repeat(s_inst[sl], K)
        st = starts[sl].reshape(-1)
        us = None
        if modus == "sample":
            g = np.concatenate([strom("merkmal", modell, i).gumbel((K, n - 1, n)) for i in instanzen[sl]])
            us = _Gumbel(g)
        T, _, z, _ = F.rollout(DD, s, st, params, us=us, mode=modus)
        tours[sl] = T.reshape(m, K, n)
        zeilen += z
    return tours, zeilen


def merkmal_strahl(params, D: np.ndarray, breite: int, start: int = 0, protokoll: list | None = None):
    """Strahlsuche mit dem Merkmalsnetz K: D (M, n, n) -> (Touren (M, B, n), lebend (M, B), K je Instanz, Summe der Hypothesen je Instanz)."""
    M, n, _ = D.shape
    s_inst = massstab(D)
    tours, lebend, k_ges, h_sum = [], [], 0, 0
    for sl in _zeilen_blocks(M, breite):
        m = len(range(*sl.indices(M)))
        R = m * breite
        Dd = np.repeat(D[sl], breite, axis=0).copy()
        ii = np.arange(n)
        Dd[:, ii, ii] = np.inf
        s = np.repeat(s_inst[sl], breite)
        st = np.full(R, start, dtype=np.int64)

        def logp(cur, visited, t, Dd=Dd, s=s, st=st, R=R, m=m):
            unv = ~visited.reshape(R, n)
            X = F.k_features(Dd, s, cur.reshape(R), st, unv, t, n)
            z = np.where(unv, MLP.forward(params, X), -np.inf)
            mx = z.max(axis=1, keepdims=True)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            lse = np.log(np.exp(z - mx).sum(axis=1, keepdims=True)) + mx
            return (z - lse).reshape(m, breite, n)

        t_, l_, k_ges, h_sum = strahl(logp, m, n, breite, start, protokoll)
        tours.append(t_)
        lebend.append(l_)
    return np.concatenate(tours), np.concatenate(lebend), k_ges, h_sum


# ---------------------------------------------------------------------------------------------------------------- Strahlsuche (für beide Netze)
def strahl(logp_fn, m: int, n: int, breite: int, start: int = 0, protokoll: list | None = None):
    """Strahlsuche der Breite B für m Instanzen. `logp_fn(cur (m, B), besucht (m, B, n), t) -> (m, B, n)`: Log-Wahrscheinlichkeiten des nächsten Ortes (-inf bei besuchten).

    Je Schritt wachsen alle lebenden Hypothesen um jeden offenen Ort; behalten werden die B mit der größten Summe der Log-Wahrscheinlichkeiten (Gleichstand: kleinerer Index der
    Hypothese, dann kleinerer Ort). Rückgabe (Touren (m, B, n), lebend (m, B), K je Instanz = Summe lebende Hypothesen x offene Orte, Summe der lebenden Hypothesen je Instanz).
    Hypothesen ohne Fortsetzung bleiben als nicht lebend (Wahrscheinlichkeit -inf) stehen. Mit einer Liste als `protokoll` wird nach jedem Schritt (Touren (m, B, t + 1), Summe der
    Log-Wahrscheinlichkeiten (m, B)) angehängt (für die Anzeige der lebenden Hypothesen)."""
    B = breite
    mi = np.arange(m)[:, None]
    bi = np.arange(B)[None, :]
    tours = np.zeros((m, B, n), dtype=np.int64)
    tours[:, :, 0] = start
    besucht = np.zeros((m, B, n), dtype=bool)
    besucht[:, :, start] = True
    cur = np.full((m, B), start, dtype=np.int64)
    score = np.full((m, B), -np.inf)
    score[:, 0] = 0.0
    k_ges, h_sum = 0, 0
    for t in range(1, n):
        lebend_je = int(np.isfinite(score[0]).sum())
        k_ges += lebend_je * (n - t)
        h_sum += lebend_je
        lp = logp_fn(cur, besucht, t)
        flat = (score[:, :, None] + lp).reshape(m, B * n)
        order = np.argsort(-flat, axis=1, kind="stable")[:, :B]
        score = np.take_along_axis(flat, order, axis=1)
        par, city = order // n, order % n
        tours = np.take_along_axis(tours, par[:, :, None], axis=1)
        tours[:, :, t] = city
        besucht = np.take_along_axis(besucht, par[:, :, None], axis=1)
        besucht[mi, bi, city] = True
        cur = city
        if protokoll is not None:
            protokoll.append((tours[:, :, :t + 1].copy(), score.copy()))
    return tours, np.isfinite(score), k_ges, h_sum


def bester_der_strahlen(X: np.ndarray, tours: np.ndarray, lebend: np.ndarray):
    """Die kürzeste lebende Endhypothese je Instanz: (Länge (M,), Tour (M, n))."""
    L = np.where(lebend, laenge(X, tours), np.inf)
    j = L.argmin(axis=1)
    mi = np.arange(X.shape[0])
    return L[mi, j], tours[mi, j]


# ---------------------------------------------------------------------------------------------------------------- Symmetrien
def symmetrien(X: np.ndarray) -> np.ndarray:
    """Die acht Abbildungen des Einheitsquadrats (Drehungen und Spiegelungen) auf die Instanzen X (M, n, 2): Form (8, M, n, 2). Entfernungen bleiben gleich; das Zeigernetz
    sieht absolute Koordinaten und kann je Abbildung eine andere Tour bauen."""
    x, y = X[..., 0], X[..., 1]
    paare = [(x, y), (1 - x, y), (x, 1 - y), (1 - x, 1 - y), (y, x), (1 - y, x), (y, 1 - x), (1 - y, 1 - x)]
    return np.stack([np.stack(p, axis=-1) for p in paare])
