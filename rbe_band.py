"""Zugriff auf die eingefrorene Messung `rbe_band.json` (Format `rbe-band-1`, erzeugt von `tools/rbe_make_band.py`).

Alle Lücken sind Prozent über der exakten Tourlänge (Held-Karp; n = 50: CP-SAT). Netz-Zeilen haben je Gewichts-Zufalls-Seed einen Vektor über die Instanzen (Form (5, Instanzen));
klassische Zeilen einen Vektor. Das Mittel über die Gewichts-Zufalls-Seeds je Instanz ist die Größe, über die verglichen wird (gepaart über die Instanzen)."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

BAND_PFAD = Path(__file__).resolve().parent / "rbe_band.json"
TRUTH_PFAD = Path(__file__).resolve().parent / "rbe_truth.json"


@lru_cache(maxsize=2)
def lade_band(pfad: str = str(BAND_PFAD)) -> dict:
    return json.loads(Path(pfad).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def lade_truth(pfad: str = str(TRUTH_PFAD)) -> dict:
    return json.loads(Path(pfad).read_text(encoding="utf-8"))


def wahrheit_n20(anzahl: int | None = None) -> np.ndarray:
    """Exakte Längen der Testinstanzen n = 20 (Zufalls-Seed 0 bis 199) im Einheitsquadrat."""
    w = np.array(lade_truth()["n20_test"])
    return w if anzahl is None else w[:anzahl]


def netz_zeile(netz: str, verfahren: str, k: int, band: dict | None = None) -> dict:
    """{"K", "O", "gap" (Seeds, Instanzen)} einer Netz-Zeile; KeyError, wenn nicht gemessen."""
    band = band or lade_band()
    z = band["netz"][netz][verfahren][str(int(k))]
    return {"K": z["K"], "O": z["O"], "gap": np.array(z["gap"])}


def netz_ks(netz: str, verfahren: str, band: dict | None = None) -> list[int]:
    band = band or lade_band()
    return sorted(int(k) for k in band["netz"][netz][verfahren])


def klass_zeile(art: str, k: int | None = None, band: dict | None = None) -> dict:
    """{"K", "O", "gap" (Instanzen)} einer klassischen Zeile: art "nn_mehrfach" und "zufall" brauchen k (Zahl Startstädte bzw. Zufallstouren), "fi" nicht."""
    band = band or lade_band()
    z = band["klass"][art] if k is None else band["klass"][art][str(int(k))]
    return {"K": z["K"], "O": z["O"], "gap": np.array(z["gap"])}


def je_instanz(gap: np.ndarray) -> np.ndarray:
    """Mittel über die Gewichts-Zufalls-Seeds je Instanz (ein Vektor bleibt unverändert)."""
    g = np.asarray(gap, dtype=np.float64)
    return g.mean(axis=0) if g.ndim == 2 else g


def mittel(gap: np.ndarray) -> float:
    return float(np.mean(gap))


# ---------------------------------------------------------------------------------------------------------------- Neustart-Suche
def neustart_raster(band: dict | None = None) -> np.ndarray:
    return np.array((band or lade_band())["klass"]["neustart"]["raster"], dtype=np.int64)


def neustart_gap_bei(budget: float, band: dict | None = None) -> np.ndarray:
    """Beste Lücke der Neustart-Suche je Instanz bei `budget` Kandidatenbewertungen (K): der Wert am größten Rasterpunkt, der nicht über dem Budget liegt (die Raster enthalten alle
    Budgets der Netz-Zeilen in K und O/8 exakt). Unter dem kleinsten Rasterpunkt gibt es keinen Wert (ValueError)."""
    band = band or lade_band()
    raster = band["klass"]["neustart"]["raster"]
    pos = np.searchsorted(raster, budget, side="right") - 1
    if pos < 0:
        raise ValueError("Budget unter dem kleinsten Rasterpunkt der Neustart-Suche")
    return np.array(band["klass"]["neustart"]["gap"][pos])


def neustart_mittelkurve(band: dict | None = None):
    """(Budgets, mittlere Lücke über die Instanzen) der Neustart-Suche; die Kurve ist nicht steigend."""
    band = band or lade_band()
    return neustart_raster(band).astype(np.float64), np.array([float(np.mean(g)) for g in band["klass"]["neustart"]["gap"]])


def suche_luecke_bei(budget: float, band: dict | None = None) -> float | None:
    """Mittlere Lücke der Neustart-Suche bei einem beliebigen Budget: log-log-Interpolation zwischen zwei gemessenen Punkten; None außerhalb des gemessenen Bereichs oder wo die
    Lücke 0 ist (Logarithmus nicht definiert)."""
    xs, ys = neustart_mittelkurve(band)
    if budget < xs[0] or budget > xs[-1]:
        return None
    i = int(np.searchsorted(xs, budget, side="right") - 1)
    if xs[i] == budget:
        return float(ys[i]) if ys[i] > 0 else None
    if i + 1 >= len(xs) or ys[i] <= 0 or ys[i + 1] <= 0:
        return None
    s = math.log(ys[i + 1] / ys[i]) / math.log(xs[i + 1] / xs[i])
    return float(ys[i] * (budget / xs[i]) ** s)


def b_stern(luecke: float, band: dict | None = None) -> float | None:
    """Kleinstes Budget (Kandidatenbewertungen = Deltas), bei dem die Neustart-Suche im Mittel die Lücke `luecke` erreicht: log-log-Interpolation zwischen zwei gemessenen Punkten.
    None, wenn die Lücke außerhalb des gemessenen Bereichs liegt (über der Lücke beim kleinsten oder unter der Lücke beim größten Budget mit Lücke über 0)."""
    xs, ys = neustart_mittelkurve(band)
    ok = ys > 0
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 2 or luecke > ys[0] or luecke < ys[-1]:
        return None
    # ys ist nicht steigend: erster Index mit ys[i + 1] <= luecke
    for i in range(len(xs) - 1):
        if ys[i + 1] <= luecke <= ys[i]:
            if ys[i] == ys[i + 1]:
                return float(xs[i])
            s = math.log(ys[i + 1] / ys[i]) / math.log(xs[i + 1] / xs[i])
            return float(xs[i] * (luecke / ys[i]) ** (1.0 / s))
    return None


# ---------------------------------------------------------------------------------------------------------------- Folgen mit Verbesserung je Start
def folge_luecke_bei(folge_gap: np.ndarray, folge_k: np.ndarray, k_ziel: np.ndarray) -> np.ndarray:
    """Lücke eines klassischen Mehrfachstarts mit Verbesserung je Start bei dem Gesamt-K `k_ziel` (je Instanz): lineare Interpolation zwischen der Lücke nach m und nach m + 1 Starts;
    folge_gap, folge_k: (Starts, Instanzen) (laufend beste Lücke, kumuliertes K). NaN, wenn k_ziel unter dem K nach dem ersten Start liegt; über dem letzten K gilt die letzte Lücke."""
    g, k = np.asarray(folge_gap), np.asarray(folge_k)
    out = np.full(k_ziel.shape, np.nan)
    for i in range(k.shape[1]):
        ki = k[:, i]
        if k_ziel[i] < ki[0]:
            continue
        if k_ziel[i] >= ki[-1]:
            out[i] = g[-1, i]
            continue
        j = int(np.searchsorted(ki, k_ziel[i], side="right") - 1)
        w = (k_ziel[i] - ki[j]) / (ki[j + 1] - ki[j])
        out[i] = g[j, i] + w * (g[j + 1, i] - g[j, i])
    return out
