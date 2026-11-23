"""Befunde aus der eingefrorenen Messung (`rbe_band.json`) als reine Funktionen: dieselben Zahlen stehen in der App, in der README und in `tests/test_claims.py`.

Alle Vergleiche sind gepaart über die Instanzen (je Instanz Mittel über die Gewichts-Zufalls-Seeds), mit 95-%-Bootstrap-Intervall (SplitMix64, 2 000 Ziehungen) und der Marke 0,25 Prozentpunkte
(`rbe_evaluation.klasse_paar`). Die Neustart-Suche bekommt dasselbe Budget wie das Netz-Verfahren, in K und in O (O geteilt durch `c` Operationen je Delta)."""

import numpy as np

import rbe_band as BD
import rbe_budget as B
import rbe_evaluation as V
import rbe_stats as ST

VERFAHREN_NAME = {"mehrfach": "Mehrfachstart", "sampling": "Sampling", "strahl": "Strahlsuche"}


def vergleich(a: np.ndarray, b: np.ndarray) -> dict:
    """Gepaarter Vergleich (Netz minus Gegner): {"v", "lo", "hi", "klasse", "m"}; a und b sind Vektoren über die Instanzen."""
    return ST.compare(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------------------------------------------- Frage 1 und 2
def eine_tour(band: dict, netz: str) -> dict:
    """Eine Tour (Greedy ab Ort 0) gegen den Nächsten Nachbarn mit derselben Zahl Kandidatenbewertungen: Lücken, Vergleich, Kosten, Faktor in O, Seeds unter dem Nächsten Nachbarn."""
    z = BD.netz_zeile(netz, "mehrfach", 1, band)
    nn = BD.klass_zeile("nn_mehrfach", 1, band)
    g = BD.je_instanz(z["gap"])
    cm = vergleich(g, nn["gap"])
    seeds = [ST.unter_gegner(z["gap"][s], nn["gap"]) for s in range(z["gap"].shape[0])]
    return {"gap_netz": BD.mittel(z["gap"]), "gap_nn": BD.mittel(nn["gap"]), "cmp": cmp_dict(cm), "K": z["K"], "O": z["O"], "O_nn": nn["O"], "faktor_o": z["O"] / nn["O"],
            "seeds_unter": sum(seeds), "seeds": len(seeds), "instanzen": len(g), "wechselkurs": B.wechselkurs(z["O"], z["K"]),
            "seed_mittel": [float(np.mean(z["gap"][s])) for s in range(z["gap"].shape[0])]}


def cmp_dict(cm: dict) -> dict:
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cm.items()}


def gleiche_konstruktionen(band: dict, netz: str, verfahren: str, k: int) -> dict:
    """Netz-Verfahren mit k Konstruktionen gegen den Nächsten Nachbarn aus min(k, 20) Startstädten (gleiche Zahl Konstruktionen, soweit es sie gibt)."""
    z = BD.netz_zeile(netz, verfahren, k, band)
    k_nn = min(k, 20)
    nn = BD.klass_zeile("nn_mehrfach", k_nn, band)
    cm = vergleich(BD.je_instanz(z["gap"]), nn["gap"])
    seeds = [ST.unter_gegner(z["gap"][s], nn["gap"]) for s in range(z["gap"].shape[0])]
    return {"gap_netz": BD.mittel(z["gap"]), "gap_nn": BD.mittel(nn["gap"]), "cmp": cmp_dict(cm), "K": z["K"], "O": z["O"], "k_nn": k_nn, "K_nn": nn["K"], "O_nn": nn["O"],
            "seeds_unter": sum(seeds), "seeds": len(seeds), "instanzen": z["gap"].shape[1]}


def je_konstruktion_besser_als_sampling(band: dict, netz: str) -> tuple[int, int, list]:
    """Mehrfachstart gegen Sampling bei gleicher Zahl Konstruktionen k = 2, 4, 8, 16: in wie vielen dieser k ist die mittlere Lücke des Mehrfachstarts kleiner? (Anzahl, Zahl der k, Liste (k, Lücke Mehrfachstart, Lücke Sampling))"""
    zeilen = []
    for k in (2, 4, 8, 16):
        zeilen.append((k, BD.mittel(BD.netz_zeile(netz, "mehrfach", k, band)["gap"]), BD.mittel(BD.netz_zeile(netz, "sampling", k, band)["gap"])))
    return sum(1 for _, a, b in zeilen if a < b), len(zeilen), zeilen


# ---------------------------------------------------------------------------------------------------------------- Frage 3
def kreuzung_zeilen(band: dict, netz: str, verfahren: str, c: float = B.O_JE_DELTA) -> tuple[list, list]:
    """(Zeilen in K, Zeilen in O) für `verdict_kreuzung`: (Budget, d, lo, hi) mit d = Lücke(Netz) - Lücke(Neustart-Suche) bei gleichem Budget; in O ist das Budget der Suche O / c Deltas.
    Die Suche ist bis 400 000 Kandidatenbewertungen gemessen; für größere Budgets gilt ihr Wert bei 400 000 (sie ist anytime, bei mehr Budget nie schlechter: das ist eine Obergrenze ihrer Lücke,
    der Vergleich also für die Suche ungünstig)."""
    raster0 = band["klass"]["neustart"]["raster"][0]
    zk, zo = [], []
    for k in BD.netz_ks(netz, verfahren, band):
        z = BD.netz_zeile(netz, verfahren, k, band)
        g = BD.je_instanz(z["gap"])
        b_k, b_o = int(z["K"]), int(z["O"] // c)
        if raster0 <= b_k:
            cm = ST.compare(g, BD.neustart_gap_bei(b_k, band))
            zk.append((b_k, cm["v"], cm["lo"], cm["hi"]))
        if raster0 <= b_o:
            cm = ST.compare(g, BD.neustart_gap_bei(b_o, band))
            zo.append((b_o, cm["v"], cm["lo"], cm["hi"]))
    return zk, zo


def kreuzung_zaehlung(zeilen: list) -> dict:
    """Zählung über die Budgets: {"suche": Zahl Budgets mit Suche vorn, "netz": Netz vorn, "gleich": kein belegter Unterschied, "gesamt": Zahl}."""
    kl = [V.klasse_paar(d, lo, hi) for _, d, lo, hi in zeilen]
    return {"suche": kl.count("b_vorn"), "netz": kl.count("a_vorn"), "gleich": kl.count("gleich"), "gesamt": len(kl)}


def kreuzungspunkt(zeilen: list) -> tuple[int | None, int | None]:
    """(größtes Budget, bei dem das Netz noch vorn liegt, kleinstes Budget, ab dem die Suche vorn liegt); None, wenn es das nicht gibt."""
    netz = [b for b, d, lo, hi in zeilen if V.klasse_paar(d, lo, hi) == "a_vorn"]
    suche = [b for b, d, lo, hi in zeilen if V.klasse_paar(d, lo, hi) == "b_vorn"]
    return (max(netz) if netz else None), (min(suche) if suche else None)


# ---------------------------------------------------------------------------------------------------------------- Frage 4
def startloesung(band: dict, netz: str) -> dict:
    """Netz (beste aus 20 Greedy-Touren, dann Verbesserung) gegen den Nächsten Nachbarn aus 20 Startstädten (gleiche Zahl Konstruktionen) und die Gesamtkosten gegen die günstigste klassische Startlösung."""
    s2 = band["start"][netz]["S2"]
    s5 = band["start"]["klass"]["S5"]
    s4, s6 = band["start"]["klass"]["S4"], band["start"]["klass"]["S6"]
    g_net, g_s5 = np.mean(np.array(s2["gap"]), axis=0), np.array(s5["gap"])
    cm = vergleich(g_net, g_s5)
    klass_k = {"S4": float(np.mean(s4["K"])), "S5": float(np.mean(s5["K"])), "S6": float(np.mean(s6["K"]))}
    klass_o = {"S4": float(np.mean(s4["O"])), "S5": float(np.mean(s5["O"])), "S6": float(np.mean(s6["O"]))}
    ev_s4 = float(np.mean(s4["ev"]))
    ev_net = float(np.mean(np.array(s2["ev"])))
    return {"gap_end_netz": float(np.mean(g_net)), "gap_end_s5": float(np.mean(g_s5)), "cmp": cmp_dict(cm), "ev_netz": ev_net, "ev_s4": ev_s4, "ev_s5": float(np.mean(s5["ev"])),
            "z_spar": 100.0 * (1.0 - ev_net / ev_s4), "K_netz": float(np.mean(np.array(s2["K"]))), "O_netz": float(np.mean(np.array(s2["O"]))), "K_klass": klass_k, "O_klass": klass_o,
            "guenstig_k": min(klass_k, key=klass_k.get), "instanzen": len(g_net), "anteil_pruefrunde": 100.0 * B.K0.m_verschiedene_zuege(20) / ev_s4}


def start_einzel(band: dict, netz: str, name: str) -> dict:
    z = band["start"][netz][name]
    return {"gap": float(np.mean(np.array(z["gap"]))), "ev": float(np.mean(np.array(z["ev"]))), "K": float(np.mean(np.array(z["K"]))), "O": float(np.mean(np.array(z["O"])))}


def klass_start(band: dict, name: str) -> dict:
    z = band["start"]["klass"][name]
    return {"gap": float(np.mean(np.array(z["gap"]))), "ev": float(np.mean(np.array(z["ev"]))), "K": float(np.mean(np.array(z["K"]))), "O": float(np.mean(np.array(z["O"])))}


def mehrfachstart_gesamt(daten: dict, folge: dict, netz_start: str = "S2", n: int = 20, o_nn_bau: float | None = None) -> dict | None:
    """Netz (beste aus n Touren, Verbesserung) gegen den klassischen Mehrfachstart mit Verbesserung (Nächster Nachbar ab Startstadt 0, 1, ...) bei gleichem Gesamt-K (je Instanz linear interpoliert).

    `daten`: {"gap": (Seeds, Instanzen), "K": ..., "O": ...} des Netz-Starts; `folge`: {"gap": (Starts, Instanzen), "K": ...} der klassischen Folge. Gibt None zurück bei weniger als zwei vergleichbaren Instanzen."""
    gap_net = np.mean(np.array(daten["gap"]), axis=0)
    k_ziel = np.mean(np.array(daten["K"]), axis=0)
    gk = BD.folge_luecke_bei(np.array(folge["gap"]), np.array(folge["K"]), k_ziel)
    ok = np.isfinite(gk)
    if ok.sum() < 2:
        return None
    cm = vergleich(gap_net[ok], gk[ok])
    seeds_gap = np.array(daten["gap"])
    k_seeds = np.array(daten["K"])
    klar = 0
    for s_i in range(seeds_gap.shape[0]):
        gk_s = BD.folge_luecke_bei(np.array(folge["gap"]), np.array(folge["K"]), k_seeds[s_i])
        ok_s = np.isfinite(gk_s)
        if ok_s.sum() >= 2:
            c_s = vergleich(seeds_gap[s_i][ok_s], gk_s[ok_s])
            klar += int(c_s["lo"] is not None and c_s["v"] <= -V.MARKE and c_s["hi"] < 0)
    kk = B.k_konstruktion(n)
    k_erster = float(np.mean(np.array(folge["K"])[0]))
    o_start = (B.o_nn(n) if o_nn_bau is None else o_nn_bau) + B.O_JE_DELTA * (k_erster - kk)       # O eines klassischen Starts (Konstruktion plus Verbesserung)
    k_netz = float(np.mean(k_ziel))
    o_klass = k_netz / k_erster * o_start
    return {"gap_netz": float(np.mean(gap_net[ok])), "gap_klass": float(np.mean(gk[ok])), "cmp": cmp_dict(cm), "K_netz": k_netz, "O_netz": float(np.mean(np.array(daten["O"]))), "O_klass": o_klass,
            "o_faktor": float(np.mean(np.array(daten["O"]))) / o_klass, "instanzen": int(ok.sum()), "ausgelassen": int((~ok).sum()),
            "seeds_klar": (klar, seeds_gap.shape[0])}


# ---------------------------------------------------------------------------------------------------------------- Frage 5
def aktive_suche(band: dict, schritte: int, lr: str) -> dict | None:
    """Aktive Suche gegen Sampling gleicher Tourzahl (16 Touren je Schritt) nach `schritte` Schritten: Lücken, Vergleich, Faktor in O, Seeds unter dem Sampling."""
    akt = band["aktiv"]
    if schritte not in akt["schritte"] or lr not in akt["aktiv"]:
        return None
    si = akt["schritte"].index(schritte)
    a = np.array(akt["aktiv"][lr])[:, si, :]
    s = np.array(akt["sampling"])[:, si, :]
    cm = vergleich(a.mean(axis=0), s.mean(axis=0))
    seeds = [ST.unter_gegner(a[i], s[i]) for i in range(a.shape[0])]
    return {"gap_aktiv": float(a.mean()), "gap_sampling": float(s.mean()), "cmp": cmp_dict(cm), "touren": schritte * akt["touren_je_schritt"], "o_faktor": akt["O"][str(schritte)] / akt["O_sampling"][str(schritte)],
            "K": akt["K"][str(schritte)], "O": akt["O"][str(schritte)], "instanzen": akt["instanzen"], "seeds_unter": sum(seeds), "seeds": len(seeds)}


# ---------------------------------------------------------------------------------------------------------------- Frage 6
def wechselkurs(band: dict, netz: str, verfahren: str, k: int) -> dict:
    """c* eines Netz-Verfahrens: Operationen je Delta, ab denen die Neustart-Suche mit demselben O so teuer ist wie das Netz (None außerhalb des gemessenen Bereichs der Suche)."""
    z = BD.netz_zeile(netz, verfahren, k, band)
    g = BD.mittel(z["gap"])
    b = BD.b_stern(g, band)
    return {"gap": g, "K": z["K"], "O": z["O"], "b_stern": b, "c_stern": None if b is None else B.c_stern(z["O"], b)}


def symmetrien(band: dict) -> dict:
    sy = band["symm"]
    eins, acht, mk = np.array(sy["eins"]), np.array(sy["acht"]), np.array(sy["merkmal"])
    cm = vergleich(acht.mean(axis=0), eins.mean(axis=0))
    return {"gap_eins": float(eins.mean()), "gap_acht": float(acht.mean()), "cmp": cmp_dict(cm), "K_eins": sy["K_eins"], "K_acht": sy["K_acht"], "O_eins": sy["O_eins"], "O_acht": sy["O_acht"],
            "gap_merkmal": float(mk.mean()), "instanzen": sy["instanzen"]}
