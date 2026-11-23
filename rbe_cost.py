"""Zählung der Linie (Quelle der K/O-Formeln, Q5; Zuschnitt 2, Abschnitt 10).

K = Kandidatenbewertungen: eine Konstruktion kostet n(n-1)/2 (nie 1), eine Prüfrunde 2-opt n(n-3)/2, ein Zeigerwert des Netzes ist eine Kandidatenbewertung.
O = grobe Operationen: Multiplikationen und Additionen der Matrixprodukte, aus den Formen einzeln gezählt; die Kandidatenbewertungen der klassischen Verfahren
mit 8 je Stück (Annahme, nur auf Zehnerpotenzen belastbar). K begünstigt das Netz (ein Zeigerwert ist teurer als ein Vergleich), O benachteiligt es.
Das Training steht als dritter Posten daneben (Touren, Entscheidungen). Das Wort der Vormessung für die Konstruktion „Tourlängen-Auswertung“ gilt hier nicht."""

import rbe_constants as C


def k_konstruktion(n: int) -> int:
    """Kandidatenbewertungen einer Konstruktion (Nächster Nachbar oder Netz): n (n - 1) / 2."""
    return n * (n - 1) // 2


def k_pruefrunde_2opt(n: int) -> int:
    """Kandidatenbewertungen einer vollen 2-opt-Prüfrunde: n (n - 3) / 2 (Paare i < j ohne benachbarte Kanten)."""
    return n * (n - 3) // 2


def m_verschiedene_zuege(n: int) -> int:
    """Verschiedene Züge einer vollen Prüfrunde (2-opt und Or-opt, duplikatfrei, Q2): n (11 n - 35) / 2; n = 20: 1 850, n = 50: 12 875. In Stück 1 nur als Formel, nicht gebaut."""
    return n * (11 * n - 35) // 2


def matmul_ops(*formen) -> int:
    """Multiplikationen und Additionen der Matrixprodukte: Summe a * b * c für Produkte (a x b) mal (b x c)."""
    return sum(a * b * c for a, b, c in formen)


def ops_kodierer(n: int, d: int, attn: bool = True, ffn: bool = True, layers: int = 1) -> int:
    """O des Kodierers für eine Instanz: Einbettung, je Schicht Attention (Q, K, V, Q K^T, A V, Wo) und Vorwärtsnetz, Schlüssel des Dekodierers. n = 20, d = 32: 211 200."""
    if not (attn or ffn):
        layers = 0
    formen = [(n, 2, d)]
    for _ in range(layers):
        if attn:
            formen += [(n, d, d)] * 3 + [(n, d, n), (n, n, d), (n, d, d)]
        if ffn:
            formen += [(n, d, 2 * d), (n, 2 * d, d)]
    formen += [(n, d, d)]
    return matmul_ops(*formen)


def ops_dekodierer_tour(n: int, d: int) -> int:
    """O des Dekodierers für eine Tour ab fester Startstadt: zwei Produkte für den Kontext der Startstadt, je Schritt ein Produkt mit Wl und ein Skalarprodukt je Ort,
    dazu 4 n elementweise Operationen je Schritt. n = 20, d = 32: 35 184 (nicht 34 160 der Vormessung: das zweite Kontextprodukt fehlte dort)."""
    formen = [(1, d, d)] * 2 + [(1, d, d), (1, d, n)] * (n - 1)
    return matmul_ops(*formen) + (n - 1) * 4 * n


def ops_netz_tour(n: int, d: int, attn: bool = True, ffn: bool = True, layers: int = 1) -> int:
    """O einer Greedy-Tour des Netzes (Kodierer + Dekodierer). n = 20, d = 32: 246 384; n = 10: 113 704."""
    return ops_kodierer(n, d, attn, ffn, layers) + ops_dekodierer_tour(n, d)


def ops_kandidaten(k: float) -> float:
    """O für k Kandidatenbewertungen der klassischen Verfahren: 8 je Stück (Annahme)."""
    return C.OPS_JE_KANDIDAT * k


def training_touren(schritte: int, batch: int = C.BATCH) -> int:
    """Touren des Trainings: Schritte mal Batch."""
    return schritte * batch


def training_entscheidungen(schritte: int, n: int, batch: int = C.BATCH) -> int:
    """Zeigerentscheidungen des Trainings: Touren mal (n - 1)."""
    return training_touren(schritte, batch) * (n - 1)
