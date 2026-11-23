"""Vorgerechnete Modelle: fünf Zeigernetze (n = 20, eine Schicht, d = 32, Gewichts-Zufalls-Seed 0 bis 4, Format `znz-modell-1`) und fünf Merkmalsnetze K (145 Parameter,
Format `zwl-mlp-1`, aus `zugwahl-lernen-demo`). Beide Netze sind einmal trainiert (bei n = 20, gleichverteilt) und werden hier nur eingesetzt; das Training steht als dritter Posten daneben.

Dateien: `rbe_models/rbe_n20_voll_d32_s<Seed>.npz` (Zeigernetz, erzeugt von `tools/rbe_make_models.py`) und `rbek_modelle/featnet_s<Seed>.npz` (Merkmalsnetz K, Kopie)."""

from functools import lru_cache
from pathlib import Path

import rbe_net as N
import rbek_mlp as MLP

WURZEL = Path(__file__).resolve().parent
MODELL_ORDNER_ZEIGER = WURZEL / "rbe_models"
NETZE = ("zeiger", "merkmal")
NETZ_ANZEIGE = {"zeiger": "Zeigernetz", "merkmal": "Merkmalsnetz"}
MODELL_SEEDS = 5
N_STADT = 20            # Training und Einsatz bei 20 Orten
D_ZEIGER = 32
ZEIGER_PARAMS = 12480
MERKMAL_PARAMS = 145
ZEIGER_TRAININGSSCHRITTE, ZEIGER_BATCH = 8000, 120      # 960 000 Touren
MERKMAL_TRAININGSSCHRITTE, MERKMAL_BATCH_TOUREN = 3750, 256   # 960 000 Touren (Stück 4)


def zeiger_pfad(seed: int, root: Path | None = None) -> Path:
    return Path(root if root is not None else MODELL_ORDNER_ZEIGER) / f"rbe_n{N_STADT}_voll_d{D_ZEIGER}_s{int(seed)}.npz"


@lru_cache(maxsize=16)
def lade_zeiger(seed: int):
    """(P, meta) des Zeigernetzes mit Gewichts-Zufalls-Seed `seed`; wirft ValueError bei einer fehlerhaften Datei, FileNotFoundError ohne Datei."""
    return N.load_model(zeiger_pfad(seed))


@lru_cache(maxsize=16)
def lade_merkmal(seed: int):
    """(params, meta) des Merkmalsnetzes K mit Gewichts-Zufalls-Seed `seed`."""
    return MLP.load_model(MLP.model_path("K", int(seed), WURZEL))


def lade(netz: str, seed: int):
    """(Parameter, meta) des Netzes `netz` ("zeiger" oder "merkmal")."""
    if netz == "zeiger":
        return lade_zeiger(seed)
    if netz == "merkmal":
        return lade_merkmal(seed)
    raise ValueError(f"unbekanntes Netz {netz!r}")


def anzahl_parameter(netz: str) -> int:
    return {"zeiger": ZEIGER_PARAMS, "merkmal": MERKMAL_PARAMS}[netz]


def trainingstouren(netz: str) -> int:
    """Touren des einmaligen Trainings: Zeigernetz 8 000 Schritte x 120 = 960 000; Merkmalsnetz K 3 750 Schritte x 256 = 960 000."""
    if netz == "zeiger":
        return ZEIGER_TRAININGSSCHRITTE * ZEIGER_BATCH
    return MERKMAL_TRAININGSSCHRITTE * MERKMAL_BATCH_TOUREN
