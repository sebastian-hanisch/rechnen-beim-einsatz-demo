"""Zählregel dieser Demo (Q5, Zuschnitt 2 Abschnitt 10): Kandidatenbewertungen K und grobe Operationen O nebeneinander, Training als dritter Posten, Wechselkurs.

Gebaut auf `rbe_cost` (Kopie von `znz_cost`, Quelle der Formeln der Linie) und `rbek_constants` (Zeilen des Merkmalsnetzes K).

K = Kandidatenbewertungen: eine Konstruktion kostet n(n-1)/2 (Nächster Nachbar, Zeigernetz, Merkmalsnetz K), nie 1; ein Delta der Suche, eine Merkmalszeile und ein Zeigerwert
zählen je 1; eine Zufallstour zählt 1. K begünstigt das Netz (eine Netz-Bewertung kostet viele Operationen). O = grobe Operationen (Multiplikationen und Additionen der Matrixprodukte,
einzeln aus den Formen gezählt; klassische Bewertungen 8 je Stück, Annahme, nur auf Zehnerpotenzen belastbar). O benachteiligt das Netz. Zeit ist kein Befund.

Jede Funktion hat nur explizite Ein- und Ausgaben (DEMO-PLAYBOOK 1b); `Zaehler` zählt in den instrumentierten Läufen der Tests jede bewertete Zeile nach."""

import math

import rbe_constants as C
import rbe_cost as K0
import rbek_constants as KC

O_JE_DELTA = C.OPS_JE_KANDIDAT          # 8: grobe Operationen je Delta der Suche (Regler „Operationen je Delta“ der App)
O_JE_ZEILE_MERKMAL = KC.O_JE_ZEILE_K    # 144: Merkmalsnetz 7-16-1 je Zeile
D_ZEIGER = 32                           # Einbettungsdimension des Zeigernetzes


# ---------------------------------------------------------------------------------------------------------------- K
def k_konstruktion(n: int) -> int:
    """K einer Konstruktion: Summe der offenen Kandidaten über alle Schritte, n (n - 1) / 2."""
    return K0.k_konstruktion(n)


def k_mehrfach(n: int, m: int) -> int:
    """K von m Konstruktionen (Mehrfachstart, Sampling): m n (n - 1) / 2."""
    return m * K0.k_konstruktion(n)


def hypothesen_je_schritt(n: int, breite: int) -> list[int]:
    """Lebende Hypothesen vor jedem Schritt t = 1 .. n - 1: h_1 = 1, h_(t+1) = min(Breite, h_t (n - t))."""
    h, out = 1, []
    for t in range(1, n):
        out.append(h)
        h = min(breite, h * (n - t))
    return out


def summe_hypothesen(n: int, breite: int) -> int:
    """Summe der lebenden Hypothesen über alle Schritte (Breite 1: n - 1)."""
    return sum(hypothesen_je_schritt(n, breite))


def k_strahl(n: int, breite: int) -> int:
    """K der Strahlsuche: je Schritt t bewertet jede lebende Hypothese ihre n - t offenen Kandidaten."""
    return sum(h * (n - t) for t, h in enumerate(hypothesen_je_schritt(n, breite), start=1))


def k_aktiv(n: int, schritte: int, touren: int) -> int:
    """K der aktiven Suche, nur Vorwärtsrechnung gezählt (begünstigt das Netz zusätzlich): Schritte x Touren x n (n - 1) / 2."""
    return schritte * touren * K0.k_konstruktion(n)


def k_start_mit_verbesserung(n: int, konstruktionen: int, zugauswertungen: float) -> float:
    """Gesamt-K einer Startlösung mit Verbesserung: Konstruktionen x n (n - 1) / 2 plus Zugauswertungen (Deltas)."""
    return konstruktionen * K0.k_konstruktion(n) + zugauswertungen


# ---------------------------------------------------------------------------------------------------------------- O
def o_nn(n: int) -> float:
    """O einer Nächster-Nachbar-Tour: 8 je Kandidat (Konvention von `rbe_cost`), n (n - 1) / 2 Kandidaten: n = 20: 1 520."""
    return K0.ops_kandidaten(K0.k_konstruktion(n))


def o_farthest(n: int) -> float:
    """O von Farthest Insertion: 4 n^2 (Annahme)."""
    return float(KC.O_FARTHEST_FAKTOR * n * n)


def o_zufall(anzahl: int = 1) -> float:
    """O von Zufallstouren: eine Tour zählt 1 Kandidatenbewertung zu 8."""
    return K0.ops_kandidaten(anzahl)


def o_kodierer(n: int, d: int = D_ZEIGER) -> int:
    """O des Kodierers (Zeigernetz, eine Schicht): aus den Matrixprodukten einzeln gezählt; n = 20, d = 32: 211 200."""
    return K0.ops_kodierer(n, d)


def o_dekodierer(n: int, d: int = D_ZEIGER) -> int:
    """O des Dekodierers für eine Tour: n = 20, d = 32: 35 184."""
    return K0.ops_dekodierer_tour(n, d)


def o_zeiger(n: int, m: int = 1, d: int = D_ZEIGER) -> int:
    """O von m Touren des Zeigernetzes für eine Instanz: ein Kodierer, m Dekodierer-Läufe (n = 20: 246 384 für m = 1)."""
    return o_kodierer(n, d) + m * o_dekodierer(n, d)


def o_zeiger_strahl(n: int, breite: int, d: int = D_ZEIGER) -> int:
    """O der Strahlsuche mit dem Zeigernetz: Kodierer, Kontext der Startstadt (2 d^2) und je lebender Hypothese und Schritt ein Zeigerschritt (d^2 + n d + 4 n)."""
    return o_kodierer(n, d) + 2 * d * d + summe_hypothesen(n, breite) * (d * d + n * d + 4 * n)


def o_merkmalsbildung_schritt(n: int) -> float:
    """O der Merkmalsbildung des Merkmalsnetzes K je Schritt und Hypothese: 2 n^2 + n (8 + log2 n) (Annahme der Vormessung; Stück 4 zählt sie nicht mit)."""
    return 2.0 * n * n + n * (8 + math.log2(n))


def o_merkmal(n: int, m: int = 1) -> float:
    """O von m Touren des Merkmalsnetzes K: n^2 (Maßstab s, einmal) + m ((n - 1) Merkmalsbildungen + n (n - 1) / 2 Zeilen zu 144). n = 20, m = 1: 47 642."""
    return n * n + m * ((n - 1) * o_merkmalsbildung_schritt(n) + K0.k_konstruktion(n) * O_JE_ZEILE_MERKMAL)


def o_merkmal_strahl(n: int, breite: int) -> float:
    """O der Strahlsuche mit dem Merkmalsnetz K: n^2 + Summe der Hypothesen x Merkmalsbildung + K der Strahlsuche x 144."""
    return n * n + summe_hypothesen(n, breite) * o_merkmalsbildung_schritt(n) + k_strahl(n, breite) * O_JE_ZEILE_MERKMAL


def o_aktiv(n: int, schritte: int, touren: int, d: int = D_ZEIGER) -> int:
    """O der aktiven Suche (Annahme: Rückwärtspass = 2 x Vorwärtspass, zusammen 3 x): 3 x Schritte x (Kodierer + Touren x Dekodierer); n = 20, 16 Touren: 2 273 280 je Schritt
    in der Vormessung (34 160 je Tour), hier mit 35 184 je Tour."""
    return 3 * schritte * (o_kodierer(n, d) + touren * o_dekodierer(n, d))


def o_suche(k: float, o_je_delta: float = O_JE_DELTA) -> float:
    """O der Suche für k Deltas bei `o_je_delta` Operationen je Delta."""
    return o_je_delta * k


# ---------------------------------------------------------------------------------------------------------------- Wechselkurs
def wechselkurs(o_netz: float, k_netz: float, o_je_delta: float = O_JE_DELTA) -> float:
    """Wie viele Deltas der Suche eine Netz-Bewertung kostet: O des Netzes geteilt durch (K des Netzes x O je Delta). Das ist das Maß, um das K das Netz begünstigt."""
    return o_netz / (k_netz * o_je_delta)


def c_stern(o_netz: float, b_stern: float) -> float:
    """Operationen je Delta, ab denen die Suche mit dem Budget der Netz-Güte (b_stern Deltas) so viel kostet wie das Netz: c* = O_Netz / B*. Das ist die Umkehrung der Annahme 8 je Delta."""
    return o_netz / b_stern


class Zaehler:
    """Zählt in instrumentierten Läufen (Tests) jede bewertete Zeile nach."""

    def __init__(self):
        self.k = 0

    def zeilen(self, anzahl: int) -> None:
        self.k += int(anzahl)
