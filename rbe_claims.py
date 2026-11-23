"""Die Befundtabelle der README: jede Zeile ist der Verdikt-Satz der App, gerechnet auf der eingefrorenen Messung (300 Instanzen, Mittel über 5 Gewichts-Zufalls-Seeds).

`befund_tabelle(band)` liefert den Markdown-Block, den `tools/rbe_readme_befunde.py --schreiben` zwischen die Marker der README setzt und den `tests/test_claims.py` Zeichen für Zeichen
mit der README vergleicht. Die Sätze sind also dieselben wie in der App: was die App sagt, steht mit Voraussetzung in der README und ist an die Messung gebunden."""

import numpy as np

import rbe_band as BD
import rbe_befunde as BF
import rbe_budget as B
import rbe_constants as C
import rbe_evaluation as V
import rbe_texts as TX

MARKER_START = "<!-- BEFUNDE:START (erzeugt von tools/rbe_readme_befunde.py, nicht von Hand ändern) -->"
MARKER_ENDE = "<!-- BEFUNDE:ENDE -->"

VOR_1 = "n = 20 gleichverteilt, {m} Instanzen, Mittel über {s} Gewichts-Zufalls-Seeds; Lücke gegen Held-Karp; Intervall: 95-%-Bootstrap über die Instanzen"


def _zeilen(band: dict) -> list:
    """Liste von (Frage, Befund, Voraussetzung, Einheit, Test)."""
    m, s = band["kopf"]["instanzen"], band["kopf"]["modell_seeds"]
    vor = VOR_1.format(m=m, s=s)
    z = []
    # Frage 1
    for netz in C.NETZE:
        e = BF.eine_tour(band, netz)
        art, text = V.verdict_eine_tour(e["cmp"]["v"], e["cmp"]["lo"], e["cmp"]["hi"], e["instanzen"], netz, e["K"], e["O"], e["O_nn"], eingefroren=m)
        z.append(("1", text + f" In {e['seeds_unter']} von {e['seeds']} Gewichts-Zufalls-Seeds liegt das Netz unter dem Nächsten Nachbarn (gepaart, Intervall ohne 0).", vor, "K und O", "test_eine_tour"))
    # Frage 2
    for netz in C.NETZE:
        for verf, k in (("mehrfach", 20), ("sampling", 8), ("sampling", 32), ("strahl", 8)):
            g = BF.gleiche_konstruktionen(band, netz, verf, k)
            art, text = V.verdict_gleiche_konstruktionen(g["cmp"]["v"], g["cmp"]["lo"], g["cmp"]["hi"], g["instanzen"], netz, BF.VERFAHREN_NAME[verf], k, g["K"], g["gap_netz"], g["gap_nn"], g["O"], g["O_nn"], eingefroren=m,
                                                         k_nn=g["k_nn"], k_nn_gesamt=g["K_nn"])
            z.append(("2", text + f" In {g['seeds_unter']} von {g['seeds']} Gewichts-Zufalls-Seeds liegt das Netz-Verfahren unter dem Nächsten Nachbarn.", vor, "K und O", "test_gleiche_konstruktionen"))
        anz, von, zeilen = BF.je_konstruktion_besser_als_sampling(band, netz)
        z.append(("2", f"{V.NETZ_NAME[netz]}: bei gleicher Zahl Konstruktionen (k = 2, 4, 8, 16) hat der Mehrfachstart in {anz} von {von} Fällen die kleinere mittlere Lücke als das Sampling ("
                       + "; ".join(f"k = {k}: {V.de(a)} gegen {V.de(b)} %" for k, a, b in zeilen) + ").", vor, "K", "test_je_konstruktion"))
    # Frage 3
    for netz in C.NETZE:
        for verf in C.VERFAHREN:
            zk, zo = BF.kreuzung_zeilen(band, netz, verf)
            art, text = V.verdict_kreuzung(zk, zo, BF.VERFAHREN_NAME[verf], netz, m)
            z.append(("3", text, vor + f"; Neustart-Suche (2-opt + Or-opt, erste Verbesserung, nicht abgestimmt) mit demselben Budget; {B.O_JE_DELTA} Operationen je Delta", "K und O", "test_kreuzung"))
    # Frage 4
    for netz in C.NETZE:
        sl = BF.startloesung(band, netz)
        art, text = V.verdict_startloesung(f"{V.NETZ_NAME[netz]}, beste aus 20 Touren", sl["cmp"]["v"], sl["cmp"]["lo"], sl["cmp"]["hi"], sl["instanzen"], sl["gap_end_netz"], sl["gap_end_s5"], sl["z_spar"], sl["K_netz"],
                                           min(sl["K_klass"].values()), sl["O_netz"], min(sl["O_klass"].values()), {"S4": "Nächster Nachbar ab Ort 0", "S5": "Nächster Nachbar, beste aus 20", "S6": "Farthest Insertion"}[sl["guenstig_k"]], eingefroren=m)
        z.append(("4", text, vor + "; Verbesserung: 2-opt + Or-opt, erste Verbesserung, ohne Nachbarlisten", "K und O", "test_startloesung"))
    for netz in C.NETZE:
        mg = BF.mehrfachstart_gesamt(band["start"][netz]["S2"], band["folgen"]["nn"])
        art, text = V.verdict_mehrfachstart_gesamt(netz, mg["K_netz"], mg["gap_netz"], mg["gap_klass"], mg["cmp"]["v"], mg["cmp"]["lo"], mg["cmp"]["hi"], mg["instanzen"], mg["o_faktor"], 20, mg["seeds_klar"], mg["ausgelassen"])
        z.append(("4", text, vor + "; klassischer Mehrfachstart: Nächster Nachbar ab Startstadt 0, 1, ... mit Verbesserung je Start, bei gleichem Gesamt-K linear interpoliert", "K und O", "test_mehrfachstart_gesamt"))
    if "n50" in band:
        n50 = band["n50"]
        mg = BF.mehrfachstart_gesamt(n50["merkmal"]["S2"], n50["folge"], n=50)
        if mg is not None:
            art, text = V.verdict_mehrfachstart_gesamt("merkmal", mg["K_netz"], mg["gap_netz"], mg["gap_klass"], mg["cmp"]["v"], mg["cmp"]["lo"], mg["cmp"]["hi"], mg["instanzen"], mg["o_faktor"], 50, mg["seeds_klar"])
            z.append(("4", text + " Ausblick, nicht mit den App-Strömen und nicht für das Zeigernetz gemessen.", f"n = 50 gleichverteilt, {n50['instanzen']} Instanzen (CP-SAT-Referenz, Lücken sind Untergrenzen, wo nicht bewiesen), Mittel über {s} Gewichts-Zufalls-Seeds", "K und O", "test_ausblick_n50"))
    # Frage 5
    akt = band["aktiv"]
    for schritte in (8, 32, 128):
        for lr in ("0.001", "0.01"):
            a = BF.aktive_suche(band, schritte, lr)
            art, text = V.verdict_aktive_suche(a["cmp"]["v"], a["cmp"]["lo"], a["cmp"]["hi"], a["instanzen"], a["touren"], float(lr), a["o_faktor"])
            z.append(("5", text + f" In {a['seeds_unter']} von {a['seeds']} Gewichts-Zufalls-Seeds liegt die aktive Suche unter dem Sampling.", f"Zeigernetz, n = 20 gleichverteilt, {akt['instanzen']} Instanzen, {akt['touren_je_schritt']} Touren je Schritt, Mittel über {s} Gewichts-Zufalls-Seeds", "K und O", "test_aktive_suche"))
    # Frage 6
    for netz in C.NETZE:
        for verf, k in (("mehrfach", 20), ("sampling", 8), ("sampling", 512)):
            w = BF.wechselkurs(band, netz, verf, k)
            art, text = V.verdict_wechselkurs(w["c_stern"], float(B.O_JE_DELTA))
            z.append(("6", f"{V.NETZ_NAME[netz]}, {BF.VERFAHREN_NAME[verf]}, k = {k} (Lücke {V.de(w['gap'])} %, {V.zahl(w['K'])} K, {V.zahl(w['O'])} O): " + text, vor + "; Umkehrung der Annahme, keine neue Messung", "O", "test_wechselkurs"))
    # Zusatz: Symmetrien, Prüfrunde
    sy = BF.symmetrien(band)
    z.append(("2", f"Zeigernetz, Mehrfachstart aus 20 Startstädten mit acht Abbildungen des Einheitsquadrats gegen eine Abbildung ({sy['instanzen']} Instanzen): Lücke {V.de(sy['gap_acht'])} gegen {V.de(sy['gap_eins'])} % "
                f"(Differenz {V.de_vz(sy['cmp']['v'])} Prozentpunkte, Intervall {V.de(sy['cmp']['lo'])} bis {V.de(sy['cmp']['hi'])}), {V.zahl(sy['K_acht'])} gegen {V.zahl(sy['K_eins'])} K, {V.zahl(sy['O_acht'])} gegen {V.zahl(sy['O_eins'])} O; "
                f"Merkmalsnetz (invariant) {V.de(sy['gap_merkmal'])} %.", f"n = 20 gleichverteilt, {sy['instanzen']} Instanzen, Mittel über {s} Gewichts-Zufalls-Seeds", "K und O", "test_symmetrien"))
    sl = BF.startloesung(band, "zeiger")
    z.append(("4", f"Beweis des Lokalminimums: eine volle Prüfrunde bewertet {V.zahl(B.K0.m_verschiedene_zuege(20))} Züge ({TX.ZUG_HINWEIS}), das sind {V.de(sl['anteil_pruefrunde'], 0)} % der {V.zahl(sl['ev_s4'])} Zugauswertungen "
                   f"(Mittel über {m} Instanzen) des Nächsten Nachbarn mit Verbesserung (Durchlauf, 2-opt vor Or-opt); eine perfekte Startlösung spart in diesem Abstieg höchstens etwa ein Drittel "
                   f"({V.de(100 - sl['anteil_pruefrunde'], 0)} %).", vor, "K", "test_pruefrunde"))
    return z


def befund_tabelle(band: dict) -> str:
    kopf = "| Frage | Befund (gemessen) | Voraussetzung | Einheit | Test |\n|---|---|---|---|---|\n"
    zeilen = [f"| {f} | {b} | {v} | {e} | `tests/test_claims.py::{t}` |" for f, b, v, e, t in _zeilen(band)]
    return kopf + "\n".join(zeilen)


def readme_block(band: dict) -> str:
    return f"{MARKER_START}\n{befund_tabelle(band)}\n{MARKER_ENDE}"


def zahlen_ueberblick(band: dict) -> dict:
    """Die wichtigsten Zahlen für Bericht und Texte (nur Anzeige)."""
    out = {"eine_tour": {n: BF.eine_tour(band, n) for n in C.NETZE}}
    out["nn_mehrfach_20"] = BD.mittel(BD.klass_zeile("nn_mehrfach", 20, band)["gap"])
    out["neustart"] = {int(k): float(np.mean(g)) for k, g in zip(*[band["klass"]["neustart"]["raster"], band["klass"]["neustart"]["gap"]]) if int(k) in (128, 512, 2048, 8192, 32768, 131072, 389120)}
    return out
