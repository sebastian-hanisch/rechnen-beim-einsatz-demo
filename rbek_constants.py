"""Konstanten der Kopien aus Stück 4 (`zugwahl-lernen-demo`, Präfix zwl_, hier rbek_): nur die Namen, die `rbek_moves`, `rbek_search`, `rbek_cost`, `rbek_mlp`, `rbek_featnet` brauchen,
mit **denselben Namen und Werten** wie in `zwl_constants.py` (Test `tests/test_copies.py`); einzige Ausnahme ist `MODELL_ORDNER` (hier `rbek_modelle`).

Das Merkmalsnetz K (sieben Merkmale, 145 Parameter) ist die Quelle der Konstruktion; das Zugnetz Z (177 Parameter) wird hier nicht benutzt, seine Namen stehen nur, weil die kopierten
Dateien sie beim Import nennen."""

# ---------------------------------------------------------------- Zugmenge und Zählung (Q2, Q5)
OR_LAENGEN = (1, 2, 3)
EPS = 1e-12                                     # ein Zug verbessert, wenn Delta < -EPS
RANG_KAPPE = 8                                  # Ränge werden bei 8 gekappt (Merkmale)
LAENGEN_KAPPE = 6.0                             # Längen durch s werden bei 6 gekappt
OPS_JE_KANDIDAT = 8                             # Annahme: grobe Operationen je Zugauswertung (Delta aus Kanten)
O_JE_ZEILE_Z = 9 * 16 + 16 + 16                 # Zugnetz Z 9-16-1 (176)
O_JE_ZEILE_K = 7 * 16 + 16 + 16                 # Merkmalsnetz K 7-16-1 (144), Merkmalsbildung nicht mitgezählt
O_JE_ZEILE_HAND = OPS_JE_KANDIDAT
O_FARTHEST_FAKTOR = 4                           # Annahme: Farthest Insertion 4 n^2 grobe Operationen
REGELN_MIT_NETZ = ("gelernt", "gelernt_det")
REGELN_MIT_ZEILEN = ("nachbarliste", "gelernt", "gelernt_det")
SICHT_GEWICHT_Z = {"frei": 0.0, "K": 1.0, "O": O_JE_ZEILE_Z / OPS_JE_KANDIDAT}
SICHT_GEWICHT_HAND = {"frei": 0.0, "K": 1.0, "O": O_JE_ZEILE_HAND / OPS_JE_KANDIDAT}

# ---------------------------------------------------------------- Netze
H = 16                                          # versteckte Einheiten
Z_MERKMALE, K_MERKMALE = 9, 7
Z_PARAMS = H * Z_MERKMALE + H + H + 1           # 177
K_PARAMS = H * K_MERKMALE + H + H + 1           # 145
MODELL_FORMAT = "zwl-mlp-1"
MODELL_ORDNER = "rbek_modelle"                  # in der Quelle zwl_modelle
