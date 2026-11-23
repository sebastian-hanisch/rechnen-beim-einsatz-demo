"""Auswertung: Zahlenformate und alle Verdikte als reine Funktionen (die App ruft `getattr(st, art)(text)`).

Jedes Verdikt gibt `(art, text)` zurück; art: "success", "info" oder "warning". Alle Schwellen sind gesetzt, nicht abgeleitet (`rbe_constants`): A schlägt B, wenn der gepaarte
Unterschied d = Lücke(A) minus Lücke(B) mindestens 0,25 Prozentpunkte beträgt und das 95-%-Bootstrap-Intervall über die Instanzen 0 ausschließt; sonst „nicht unterscheidbar“.
Jeder Text nennt Instanzzahl, Netz, Einheit (K oder O) und, wo es sie gibt, die benannte Ausnahme im selben Satz. Keiner sagt „immer“ oder „alle“ ohne Zählung."""

import rbe_constants as C

MARKE = C.MARKE_KLAR
MIN_INSTANZEN = 4
C_STERN_FAKTOR = 10         # gesetzt, nicht abgeleitet: „weit“ ab dem Zehnfachen der Annahme
NETZ_NAME = {"zeiger": "Zeigernetz", "merkmal": "Merkmalsnetz"}


# ---------------------------------------------------------------------------------------------------------------- Zahlenformate (deutsch)
def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma (eine Zahl, die auf null rundet, ohne Minuszeichen)."""
    s = f"{x:.{stellen}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",")


def de_vz(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma und Vorzeichen (eine Zahl, die auf null rundet, ohne Vorzeichen)."""
    s = f"{x:+.{stellen}f}"
    if float(s) == 0:
        s = s[1:]
    return s.replace(".", ",")


def zahl(n) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner (1 850)."""
    return f"{int(round(n)):,}".replace(",", " ")


def mio(x: float, stellen: int = 2) -> str:
    """Operationen in Millionen mit Dezimalkomma."""
    return de(x / 1e6, stellen)


def faktor(x: float) -> str:
    """Faktor „das 162-Fache“: ab 10 ohne Nachkommastelle, darunter mit einer."""
    return de(x, 0) if x >= 10 else de(x, 1)


# ---------------------------------------------------------------------------------------------------------------- Klassen
def klasse_paar(d: float, lo, hi) -> str:
    """Einordnung eines Unterschieds d = Lücke(A) - Lücke(B) mit Intervall (lo, hi): "a_vorn" (d <= -0,25 und Intervall unter 0), "b_vorn" (d >= 0,25 und Intervall über 0), "a_klein" bzw. "b_klein"
    (Betrag mindestens 0,10, unter 0,25, Intervall ohne 0: ein kleiner Unterschied unter der Marke), "gleich" (Betrag unter 0,10 oder Intervall mit 0), "zu_wenig" (kein Intervall)."""
    if lo is None or hi is None:
        return "zu_wenig"
    if d <= -MARKE and hi < 0:
        return "a_vorn"
    if d >= MARKE and lo > 0:
        return "b_vorn"
    if d <= -C.MARKE_GLEICH and hi < 0:
        return "a_klein"
    if d >= C.MARKE_GLEICH and lo > 0:
        return "b_klein"
    return "gleich"


KLEIN = f"kleiner Unterschied unter der Marke von {de(C.MARKE_KLAR)} Prozentpunkten"


def _zu_wenig(n_inst: int, eingefroren: int) -> tuple:
    return "info", f"Mit {n_inst} Instanzen ist kein Urteil möglich (mindestens {MIN_INSTANZEN}); die eingefrorene Messung hat {eingefroren} Instanzen."


def _o_satz(o_netz: float, o_gegner: float) -> str:
    f = o_netz / o_gegner if o_gegner else float("inf")
    if f > 1:
        return f"In Operationen (O) kostet der Durchlauf das {faktor(f)}-Fache"
    return "In Operationen (O) kostet der Durchlauf nicht mehr"


# ---------------------------------------------------------------------------------------------------------------- Frage 1: eine Tour
def verdict_eine_tour(d: float, lo, hi, n_inst: int, netz: str, k: int, o_netz: float, o_nn: float, eingefroren: int = 200) -> tuple:
    """Eine Tour des Netzes gegen den Nächsten Nachbarn bei derselben Zahl Kandidatenbewertungen k (K); d = Lücke(Netz) - Lücke(Nächster Nachbar)."""
    name = NETZ_NAME[netz]
    kl = klasse_paar(d, lo, hi)
    if n_inst < MIN_INSTANZEN or kl == "zu_wenig":
        return _zu_wenig(n_inst, eingefroren)
    iv = f"Intervall {de(lo)} bis {de(hi)}, {n_inst} Instanzen"
    if kl == "a_vorn":
        return "success", (f"{name} baut mit {zahl(k)} Kandidatenbewertungen (K) im Mittel eine um {de(-d)} Prozentpunkte kürzere Tour als der Nächste Nachbar mit denselben {zahl(k)} ({iv}). "
                           f"{_o_satz(o_netz, o_nn)} ({zahl(o_netz)} gegen {zahl(o_nn)}): der Vorsprung gilt in {'K, nicht in O' if o_netz > o_nn else 'K und in O'}.")
    if kl == "b_vorn":
        return "warning", (f"{name} baut mit {zahl(k)} Kandidatenbewertungen (K) im Mittel eine um {de(d)} Prozentpunkte längere Tour als der Nächste Nachbar mit denselben {zahl(k)} ({iv}). "
                           f"{_o_satz(o_netz, o_nn)} ({zahl(o_netz)} gegen {zahl(o_nn)}).")
    if kl in ("a_klein", "b_klein"):
        return "info", (f"{name}: {KLEIN} zum Nächsten Nachbarn bei {zahl(k)} Kandidatenbewertungen (K), Differenz {de_vz(d)} Prozentpunkte ({iv}). "
                        f"{_o_satz(o_netz, o_nn)} ({zahl(o_netz)} gegen {zahl(o_nn)}).")
    return "info", (f"{name}: kein belegter Unterschied zum Nächsten Nachbarn bei {zahl(k)} Kandidatenbewertungen (K), Differenz {de_vz(d)} Prozentpunkte ({iv}). "
                    f"{_o_satz(o_netz, o_nn)} ({zahl(o_netz)} gegen {zahl(o_nn)}).")


# ---------------------------------------------------------------------------------------------------------------- Frage 2: mehr Touren bauen
def verdict_gleiche_konstruktionen(d: float, lo, hi, n_inst: int, netz: str, verfahren_name: str, k: int, k_gesamt: float, gap_netz: float, gap_nn: float,
                                   o_netz: float, o_nn: float, eingefroren: int = 200, k_nn: int | None = None, k_nn_gesamt: float | None = None) -> tuple:
    """Netz-Verfahren mit k Konstruktionen gegen den Nächsten Nachbarn aus k Startstädten (gleiche Zahl Konstruktionen, daher gleiches K); d = Lücke(Netz) - Lücke(Nächster Nachbar).
    Hat der Nächste Nachbar nur `k_nn` < k verschiedene Touren (eine je Startstadt), gilt der Vergleich nicht bei gleichem K: der Satz nennt `k_nn` Startstädte (mit `k_nn_gesamt` K)."""
    name = NETZ_NAME[netz]
    kl = klasse_paar(d, lo, hi)
    if n_inst < MIN_INSTANZEN or kl == "zu_wenig":
        return _zu_wenig(n_inst, eingefroren)
    weniger = k_nn is not None and k_nn < k
    if weniger:
        kosten_nn = f" ({zahl(k_nn_gesamt)} K)" if k_nn_gesamt is not None else ""
        kopf = (f"{name}, {verfahren_name} mit {zahl(k)} {'Konstruktion' if k == 1 else 'Konstruktionen'} ({zahl(k_gesamt)} Kandidatenbewertungen, K): Lücke {de(gap_netz)} gegen {de(gap_nn)} % beim Nächster-Nachbar-"
                f"Mehrfachstart aus {k_nn} Startstädten{kosten_nn} (Differenz {de_vz(d)} Prozentpunkte, Intervall {de(lo)} bis {de(hi)}, {n_inst} Instanzen). "
                f"Der Nächste Nachbar hat nur {k_nn} verschiedene Touren (eine je Startstadt), das Netz bekommt hier mehr K. ")
    else:
        kopf = (f"{name}, {verfahren_name} mit {zahl(k)} {'Konstruktion' if k == 1 else 'Konstruktionen'} ({zahl(k_gesamt)} Kandidatenbewertungen, K): Lücke {de(gap_netz)} gegen {de(gap_nn)} % beim Nächster-Nachbar-"
                f"Mehrfachstart mit gleich vielen Konstruktionen (Differenz {de_vz(d)} Prozentpunkte, Intervall {de(lo)} bis {de(hi)}, {n_inst} Instanzen). ")
    ende = f"In Operationen (O): {zahl(o_netz)} gegen {zahl(o_nn)}."
    art = {"a_vorn": "success", "b_vorn": "warning", "gleich": "info", "a_klein": "info", "b_klein": "info"}[kl]
    zusatz = {"a_vorn": "Das Netz liegt vorn. ", "b_vorn": "Der Nächste Nachbar liegt vorn. ", "gleich": "Kein belegter Unterschied. ",
              "a_klein": f"Das Netz liegt knapp vorn ({KLEIN}). ", "b_klein": f"Der Nächste Nachbar liegt knapp vorn ({KLEIN}). "}[kl]
    return art, kopf + zusatz + ende


# ---------------------------------------------------------------------------------------------------------------- Frage 3: gleiches Gesamtbudget gegen die Suche
def _kreuzung_satz(zeilen: list, einheit: str) -> tuple:
    """(Satz, Zählung) für eine Einheit; zeilen: Liste (Budget, d, lo, hi) aufsteigend, d = Lücke(Netz) - Lücke(Suche) (positiv: Suche besser). Unterschiede unter der Marke zählen als „kein klares
    Urteil“; der Satz beschreibt die Läufe aufeinanderfolgender Budgets mit gleicher Einordnung."""
    klassen = []
    for b, d, lo, hi in zeilen:
        kl = klasse_paar(d, lo, hi)
        klassen.append("suche" if kl == "b_vorn" else "netz" if kl == "a_vorn" else "gleich")
    z = {"suche": klassen.count("suche"), "netz": klassen.count("netz"), "gleich": klassen.count("gleich")}
    m = len(zeilen)
    if m == 0:
        return f"In {einheit} liegt keine Messung vor.", z
    budgets = [b for b, *_ in zeilen]
    in_o = einheit.startswith("Operationen")           # in O ist das Budget der Suche die Zahl ihrer Deltas (O geteilt durch Operationen je Delta), nicht O selbst
    if z["suche"] == m:
        ab = f"kleinstes Budget {zahl(budgets[0])} Deltas der Suche" if in_o else f"ab {zahl(budgets[0])}"
        return f"In {einheit} liegt die Suche bei allen {m} gemessenen Budgets vorn ({ab}).", z
    if z["netz"] == m:
        return f"In {einheit} liegt das Netz bei allen {m} gemessenen Budgets vorn.", z
    laeufe, start = [], 0
    for i in range(1, m + 1):
        if i == m or klassen[i] != klassen[start]:
            laeufe.append((klassen[start], budgets[start], budgets[i - 1], i - start))
            start = i
    teile = []
    for art, von, bis, anzahl in laeufe:
        bereich = zahl(von) if von == bis else f"{zahl(von)} bis {zahl(bis)}"
        if in_o:
            bereich += " Deltas der Suche"
        teile.append({"netz": f"liegt das Netz bei {bereich} vorn", "suche": f"liegt die Suche bei {bereich} vorn", "gleich": f"gibt es bei {bereich} kein klares Urteil ({anzahl} Budgets)"}[art])
    return f"In {einheit} " + "; dann ".join(teile) + ".", z


def verdict_kreuzung(k_zeilen: list, o_zeilen: list, verfahren_name: str, netz: str, n_inst: int) -> tuple:
    """Netz-Verfahren gegen die Neustart-Suche (2-opt + Or-opt, nicht abgestimmt) bei gleichem Budget in K und in O. Die Zeilen sind (Budget, d, lo, hi), d = Lücke(Netz) - Lücke(Suche)."""
    satz_k, zk = _kreuzung_satz(k_zeilen, "Kandidatenbewertungen (K)")
    satz_o, zo = _kreuzung_satz(o_zeilen, "Operationen (O)")
    name = NETZ_NAME[netz]
    kopf = f"{name}, {verfahren_name}, {n_inst} Instanzen: "
    if k_zeilen and o_zeilen and zk["suche"] == len(k_zeilen) and zo["suche"] == len(o_zeilen):
        gesamt = "Die Suche liegt in beiden Einheiten bei jedem gemessenen Budget vorn."
        art = "warning"
    elif zk["netz"] > 0 and zo["netz"] == 0:
        gesamt = "Das Netz liegt in Kandidatenbewertungen bei einem Teil der Budgets vorn, in Operationen bei keinem: die Entscheidung liegt zwischen den beiden Einheiten."
        art = "info"
    elif zo["netz"] > 0:
        gesamt = "Das Netz liegt auch in Operationen bei einem Teil der Budgets vorn."
        art = "success"
    else:
        gesamt = "Kein einheitliches Bild über die gemessenen Budgets."
        art = "info"
    return art, f"{kopf}{satz_k} {satz_o} {gesamt}"


# ---------------------------------------------------------------------------------------------------------------- Frage 4: Netz als Startlösung
def verdict_startloesung(variante: str, d_end: float, lo, hi, n_inst: int, luecke_netz: float, luecke_klass: float, z_spar: float, k_gesamt: float, k_guenstig: float,
                         o_gesamt: float, o_guenstig: float, name_guenstig: str, eingefroren: int = 200) -> tuple:
    """Netz als Startlösung (Verbesserung 2-opt + Or-opt) gegen den Nächsten Nachbarn aus gleich vielen Startstädten (gleiche Zahl Konstruktionen); d_end = Endlücke(Netz) - Endlücke(Nächster Nachbar).
    `k_guenstig`, `o_guenstig`: Gesamtkosten der günstigsten klassischen Startlösung (Nächster Nachbar, bester aus n Startstädten, Farthest Insertion)."""
    kl = klasse_paar(d_end, lo, hi)
    if n_inst < MIN_INSTANZEN or kl == "zu_wenig":
        return _zu_wenig(n_inst, eingefroren)
    sp = f"{de(z_spar, 0)} % {'weniger' if z_spar >= 0 else 'mehr'} Zugauswertungen als" if abs(z_spar) >= 0.5 else "etwa gleich viele Zugauswertungen wie"
    # zwei Vergleichsbasen: die Endlücke steht gegen den Nächsten Nachbarn aus 20 Startstädten (gleiche Zahl Konstruktionen), die Zugauswertungen gegen die Verbesserung ab einem einzelnen Nächsten Nachbarn (Ort 0)
    kopf = (f"{variante}: Endlücke {de(luecke_netz)} gegen {de(luecke_klass)} % beim Nächsten Nachbarn aus 20 Startstädten (gleiche Zahl Konstruktionen, {n_inst} Instanzen); "
            f"die Verbesserung ab dem Netz braucht {sp} ab einem einzelnen Nächsten Nachbarn. "
            f"Gesamtkosten {zahl(k_gesamt)} gegen {zahl(k_guenstig)} Kandidatenbewertungen (K) und {mio(o_gesamt)} gegen {mio(o_guenstig, 3)} Mio Operationen (O) ({name_guenstig}, die günstigste klassische Startlösung). ")
    if k_gesamt > k_guenstig or o_gesamt > o_guenstig:
        return "warning", kopf + f"Das Netz als Startlösung ist teurer als {name_guenstig} und spart weniger Zugauswertungen, als es kostet."
    if kl == "a_vorn":
        return "success", kopf + "Das Netz als Startlösung kostet nicht mehr als die günstigste klassische Startlösung und liegt in der Endlücke vorn."
    if kl == "a_klein":
        return "info", kopf + f"Das Netz als Startlösung kostet nicht mehr, der Vorsprung in der Endlücke ist klein ({KLEIN})."
    return "info", kopf + "Das Netz als Startlösung kostet nicht mehr, aber ein belegter Vorsprung in der Endlücke fehlt."


def verdict_mehrfachstart_gesamt(netz: str, k_gesamt: float, luecke_netz: float, luecke_klass: float, d: float, lo, hi, n_inst: int, o_faktor: float, n: int, seeds_klar: tuple | None = None, ausgelassen: int = 0) -> tuple:
    """Netz mit der besten aus n Touren und einer Verbesserung gegen den klassischen Mehrfachstart mit Verbesserung bei **gleichem Gesamt-K** (Interpolation der klassischen Folge);
    d = Lücke(Netz) - Lücke(klassisch bei diesem K). `ausgelassen`: Instanzen ohne vergleichbaren Wert (Ziel-K unter dem K des ersten klassischen Starts), sie stehen im Satz."""
    name = NETZ_NAME[netz]
    kl = klasse_paar(d, lo, hi)
    if n_inst < MIN_INSTANZEN or kl == "zu_wenig":
        return _zu_wenig(n_inst, n_inst)
    fehlen = f" ({ausgelassen} {'Instanz' if ausgelassen == 1 else 'Instanzen'} ohne vergleichbaren Wert)" if ausgelassen else ""
    kopf = (f"n = {n}, {n_inst} Instanzen{fehlen}, {name}: bei {zahl(k_gesamt)} Kandidatenbewertungen (K) Lücke {de(luecke_netz)} % mit der besten aus {n} Touren und einer Verbesserung, "
            f"gegen {de(luecke_klass)} % beim klassischen Mehrfachstart mit Verbesserung und gleichem Gesamt-K (Differenz {de_vz(d)} Prozentpunkte, Intervall {de(lo)} bis {de(hi)}). ")
    if seeds_klar is not None:
        kopf += f"In {seeds_klar[0]} von {seeds_klar[1]} Gewichts-Zufalls-Seeds liegt das Netz um mindestens {de(MARKE)} Prozentpunkte vorn. "
    if kl == "a_vorn":
        return "info", kopf + f"In K liegt das Netz vorn; in Operationen kostet es das {faktor(o_faktor)}-Fache der klassischen Rechnung für dieselbe Güte: die Entscheidung liegt zwischen den Einheiten."
    if kl == "a_klein":
        return "info", kopf + f"In K liegt das Netz knapp vorn ({KLEIN}); in Operationen kostet es das {faktor(o_faktor)}-Fache der klassischen Rechnung."
    if kl == "b_vorn":
        return "warning", kopf + "In K liegt der klassische Mehrfachstart vorn."
    if kl == "b_klein":
        return "info", kopf + f"In K liegt der klassische Mehrfachstart knapp vorn ({KLEIN})."
    return "info", kopf + "In K kein belegter Unterschied."


# ---------------------------------------------------------------------------------------------------------------- Frage 5: aktive Suche
def verdict_aktive_suche(d: float, lo, hi, n_inst: int, touren: int, lr: float, o_faktor: float, eingefroren: int = 100) -> tuple:
    """Aktive Suche (Zeigernetz wird auf der Instanz weitertrainiert) gegen Sampling mit gleich vielen Touren aus dem Anfangsmodell; d = Lücke(aktiv) - Lücke(Sampling)."""
    kl = klasse_paar(d, lo, hi)
    if n_inst < MIN_INSTANZEN or kl == "zu_wenig":
        return _zu_wenig(n_inst, eingefroren)
    iv = f"Intervall {de(lo)} bis {de(hi)}, {n_inst} Instanzen"
    lr_text = de(lr, 4).rstrip("0").rstrip(",") if lr < 1e-2 else de(lr, 2)
    if kl == "a_vorn":
        return "success", (f"Mit {zahl(touren)} Touren auf der Instanz (Lernrate {lr_text}) liegt die aktive Suche {de(-d)} Prozentpunkte vor dem Sampling mit gleich vielen Touren ({iv}). "
                           f"Sie kostet in Operationen (O) das {faktor(o_faktor)}-Fache (Rückwärtspass als Doppeltes des Vorwärtspasses angenommen).")
    if kl == "b_vorn":
        return "warning", (f"Mit {zahl(touren)} Touren auf der Instanz (Lernrate {lr_text}) liegt die aktive Suche {de(d)} Prozentpunkte hinter dem Sampling mit gleich vielen Touren ({iv}). "
                           f"Sie kostet in Operationen (O) das {faktor(o_faktor)}-Fache (Rückwärtspass als Doppeltes des Vorwärtspasses angenommen).")
    if kl in ("a_klein", "b_klein"):
        return "info", (f"Mit {zahl(touren)} Touren auf der Instanz (Lernrate {lr_text}): {KLEIN} zwischen aktiver Suche und Sampling mit gleich vielen Touren "
                        f"(Differenz {de_vz(d)} Prozentpunkte, {iv}).")
    return "info", (f"Mit {zahl(touren)} Touren auf der Instanz (Lernrate {lr_text}): kein belegter Unterschied zwischen aktiver Suche und Sampling mit gleich vielen Touren "
                    f"(Differenz {de_vz(d)} Prozentpunkte, {iv}).")


# ---------------------------------------------------------------------------------------------------------------- Frage 6: Wechselkurs
def verdict_wechselkurs(c_stern: float | None, c0: float) -> tuple:
    """Ab wie vielen Operationen je Delta (c*) gleicht die Suche in Operationen mit dem Netz gleich; c0 ist die Annahme. Das ist keine neue Messung, sondern die Umkehrung der Annahme."""
    if c_stern is None:
        return "info", "Die Lücke dieses Netz-Verfahrens liegt außerhalb des gemessenen Bereichs der Suche: kein Wechselkurs bestimmbar."
    if c_stern >= C_STERN_FAKTOR * c0:
        return "success", f"Das Netz erreicht die Suche in Operationen erst, wenn ein Delta mehr als {zahl(c_stern)} Operationen kostet; angenommen sind {zahl(c0)}."
    if c_stern > c0:
        return "info", (f"Das Netz erreicht die Suche in Operationen bei {zahl(c_stern)} Operationen je Delta, das ist nur das {de(c_stern / c0, 1)}-Fache der Annahme von {zahl(c0)}: "
                        f"das Ergebnis hängt an der Annahme.")
    return "warning", f"Bei {zahl(c0)} Operationen je Delta liegt das Netz auch in Operationen vorn (Schwelle c* = {zahl(c_stern)})."
