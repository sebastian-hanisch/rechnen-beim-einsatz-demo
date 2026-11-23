"""Verdikte als reine Funktionen: Randfälle, Schwellen, Nenner 0, Texte (Instanzzahl, Netz, Einheit, benannte Ausnahme im selben Satz)."""

import pytest

import rbe_evaluation as V

# ---------------------------------------------------------------------------------------------------------------- Zahlenformate


def test_zahlenformate():
    assert V.de(1.234) == "1,23" and V.de(-0.0001) == "0,00" and V.de_vz(-0.0001) == "0,00" and V.de_vz(0.5) == "+0,50" and V.de_vz(-2.345) == "-2,35"
    assert V.zahl(1850) == "1 850" and V.zahl(1520.4) == "1 520" and V.mio(246384) == "0,25"
    assert V.faktor(162.1) == "162" and V.faktor(3.14) == "3,1"


# ---------------------------------------------------------------------------------------------------------------- Klassen
@pytest.mark.parametrize("d,lo,hi,soll", [
    (-1.0, -1.5, -0.5, "a_vorn"), (-0.25, -0.4, -0.1, "a_vorn"), (-0.24, -0.4, -0.1, "a_klein"), (-1.0, -1.5, 0.1, "gleich"), (0.15, 0.05, 0.25, "b_klein"), (-0.05, -0.1, -0.01, "gleich"), (-0.10, -0.2, -0.01, "a_klein"),
    (1.0, 0.5, 1.5, "b_vorn"), (0.25, 0.1, 0.4, "b_vorn"), (0.3, -0.1, 0.5, "gleich"), (0.0, -0.1, 0.1, "gleich"), (0.5, None, None, "zu_wenig"),
])
def test_klasse_paar(d, lo, hi, soll):
    assert V.klasse_paar(d, lo, hi) == soll


# ---------------------------------------------------------------------------------------------------------------- Frage 1
def test_eine_tour_vier_faelle_und_ausnahme_im_selben_satz():
    kw = dict(n_inst=8, netz="zeiger", k=190, o_netz=246384, o_nn=1520)
    art, text = V.verdict_eine_tour(-7.6, -9.0, -6.0, **kw)
    assert art == "success" and "Zeigernetz" in text and "7,60 Prozentpunkte kürzere" in text and "190" in text and "8 Instanzen" in text
    assert "das 162-Fache" in text and "gilt in K, nicht in O" in text
    art, text = V.verdict_eine_tour(2.0, 1.0, 3.0, **kw)
    assert art == "warning" and "längere" in text
    art, text = V.verdict_eine_tour(0.1, -0.5, 0.7, **kw)
    assert art == "info" and "kein belegter Unterschied" in text
    art, text = V.verdict_eine_tour(-7.6, -9.0, -6.0, **{**kw, "n_inst": 3})
    assert art == "info" and "kein Urteil" in text and "mindestens 4" in text and "200" in text
    art, text = V.verdict_eine_tour(-7.6, None, None, **kw)
    assert art == "info" and "kein Urteil" in text


def test_eine_tour_kleiner_unterschied():
    kw = dict(n_inst=8, netz="zeiger", k=190, o_netz=246384, o_nn=1520)
    art, text = V.verdict_eine_tour(-0.15, -0.25, -0.05, **kw)
    assert art == "info" and "kleiner Unterschied unter der Marke von 0,25 Prozentpunkten" in text


def test_eine_tour_o_satz_wenn_das_netz_in_o_nicht_teurer_ist():
    art, text = V.verdict_eine_tour(-3.0, -4.0, -2.0, 8, "merkmal", 190, 1000.0, 1520.0)
    assert "kostet der Durchlauf nicht mehr" in text and "Merkmalsnetz" in text and "gilt in K und in O" in text


# ---------------------------------------------------------------------------------------------------------------- Frage 2
def test_gleiche_konstruktionen():
    art, text = V.verdict_gleiche_konstruktionen(-3.65, -4.5, -2.8, 8, "zeiger", "Mehrfachstart", 20, 3800, 2.04, 5.69, 894400, 30400)
    assert art == "success" and "Mehrfachstart mit 20 Konstruktionen (3 800 Kandidatenbewertungen, K)" in text and "2,04 gegen 5,69" in text and "Das Netz liegt vorn" in text
    art, text = V.verdict_gleiche_konstruktionen(1.0, 0.5, 1.5, 8, "merkmal", "Sampling", 8, 1520, 6.0, 5.0, 100, 50)
    assert art == "warning" and "Der Nächste Nachbar liegt vorn" in text
    art, text = V.verdict_gleiche_konstruktionen(0.0, -0.4, 0.4, 8, "merkmal", "Sampling", 8, 1520, 6.0, 6.0, 100, 50)
    assert art == "info" and "Kein belegter Unterschied" in text
    assert V.verdict_gleiche_konstruktionen(0.0, -0.4, 0.4, 2, "merkmal", "Sampling", 8, 1520, 6.0, 6.0, 100, 50)[0] == "info"
    art, text = V.verdict_gleiche_konstruktionen(-1.0, -1.5, -0.5, 8, "zeiger", "Sampling", 512, 97280, 1.67, 2.04, 100, 50, k_nn=20)
    assert "nur 20 verschiedene Touren" in text and "20 Startstädten" in text
    assert "mit gleich vielen Konstruktionen" not in text and "das Netz bekommt hier mehr K" in text                         # k > 20: nicht dasselbe K
    art, text = V.verdict_gleiche_konstruktionen(-1.0, -1.5, -0.5, 8, "zeiger", "Sampling", 32, 6080, 4.08, 5.49, 100, 50, k_nn=20, k_nn_gesamt=3800)
    assert "Mehrfachstart aus 20 Startstädten (3 800 K)" in text and "(6 080 Kandidatenbewertungen, K)" in text
    assert "verschiedene Touren" not in V.verdict_gleiche_konstruktionen(-1.0, -1.5, -0.5, 8, "zeiger", "Mehrfachstart", 20, 3800, 1.0, 2.0, 100, 50, k_nn=20)[1]


# ---------------------------------------------------------------------------------------------------------------- Frage 3
def _z(budgets, klassen):
    """Zeilen (Budget, d, lo, hi) mit der gewünschten Klasse: 's' Suche vorn, 'n' Netz vorn, 'g' gleich."""
    out = []
    for b, k in zip(budgets, klassen):
        out.append((b, {"s": 1.0, "n": -1.0, "g": 0.0}[k], {"s": 0.5, "n": -1.5, "g": -0.2}[k], {"s": 1.5, "n": -0.5, "g": 0.2}[k]))
    return out


def test_kreuzung_alle_suche_vorn_in_beiden_einheiten():
    art, text = V.verdict_kreuzung(_z([190, 1520, 6080], "sss"), _z([30, 300, 760], "sss"), "Sampling", "zeiger", 200)
    assert art == "warning" and "bei allen 3 gemessenen Budgets vorn (ab 190)" in text and "(kleinstes Budget 30 Deltas der Suche)" in text and "in beiden Einheiten bei jedem gemessenen Budget vorn" in text and "200 Instanzen" in text


def test_kreuzung_netz_bis_b1_dann_suche_nur_in_k():
    art, text = V.verdict_kreuzung(_z([190, 380, 1520, 6080], "nnss"), _z([30, 300], "ss"), "Sampling", "zeiger", 200)
    assert art == "info" and "liegt das Netz bei 190 bis 380 vorn; dann liegt die Suche bei 1 520 bis 6 080 vorn" in text and "in Operationen bei keinem" in text


def test_kreuzung_gemischt_und_netz_vorn_auch_in_o():
    art, text = V.verdict_kreuzung(_z([1, 2, 3, 4], "sgns"), _z([1, 2], "gg"), "Strahlsuche", "merkmal", 100)
    assert "liegt die Suche bei 1 vorn; dann gibt es bei 2 kein klares Urteil (1 Budgets); dann liegt das Netz bei 3 vorn; dann liegt die Suche bei 4 vorn" in text
    art, text = V.verdict_kreuzung(_z([1, 2], "nn"), _z([1, 2], "ns"), "Sampling", "zeiger", 100)
    assert art == "success" and "bei allen 2 gemessenen Budgets vorn" in text and "auch in Operationen" in text


def test_kreuzung_leere_zeilen():
    art, text = V.verdict_kreuzung([], [], "Sampling", "zeiger", 10)
    assert "keine Messung" in text


# ---------------------------------------------------------------------------------------------------------------- Frage 4
def test_startloesung_ersetzt_nicht_wenn_teurer_in_k_oder_o():
    art, text = V.verdict_startloesung("Merkmalsnetz, beste aus 20", -0.3, -0.5, -0.1, 200, 0.13, 0.48, 26.0, 6341, 3097, 0.97, 0.025, "Farthest Insertion")
    assert art == "warning" and "ist teurer als Farthest Insertion" in text and "spart weniger Zugauswertungen, als es kostet" in text and "6 341 gegen 3 097" in text and "26 % weniger Zugauswertungen als ab einem einzelnen Nächsten Nachbarn" in text
    assert "beim Nächsten Nachbarn aus 20 Startstädten (gleiche Zahl Konstruktionen, 200 Instanzen)" in text              # zwei Vergleichsbasen, beide genannt
    art, text = V.verdict_startloesung("Netz", -0.5, -0.8, -0.2, 200, 0.1, 0.6, 20.0, 3000, 3097, 0.5, 0.5, "Farthest Insertion")
    assert art == "success" and "kostet nicht mehr" in text
    art, text = V.verdict_startloesung("Netz", -0.5, -0.8, -0.2, 200, 0.1, 0.6, 20.0, 3000, 3097, 0.9, 0.5, "Farthest Insertion")
    assert art == "warning"
    art, text = V.verdict_startloesung("Netz", 0.0, -0.2, 0.2, 200, 0.5, 0.5, 0.2, 3000, 3097, 0.4, 0.5, "x")
    assert art == "info" and "etwa gleich viele Zugauswertungen" in text and "fehlt" in text
    assert V.verdict_startloesung("Netz", -0.5, -0.8, -0.2, 2, 0.1, 0.6, 20.0, 1, 2, 1, 2, "x")[1].startswith("Mit 2 Instanzen")
    art, text = V.verdict_startloesung("Netz", -0.15, -0.25, -0.05, 200, 0.4, 0.55, 20.0, 3000, 3097, 0.4, 0.5, "x")
    assert art == "info" and "Vorsprung in der Endlücke ist klein" in text


def test_mehrfachstart_gesamt():
    art, text = V.verdict_mehrfachstart_gesamt("merkmal", 87433, 0.92, 1.47, -0.55, -0.9, -0.2, 10, 24.0, 50)
    assert art == "info" and "n = 50, 10 Instanzen, Merkmalsnetz" in text and "das 24-Fache" in text and "zwischen den Einheiten" in text
    assert V.verdict_mehrfachstart_gesamt("merkmal", 1, 1.0, 0.5, 0.5, 0.3, 0.7, 10, 1.0, 50)[0] == "warning"
    assert "In K kein belegter Unterschied" in V.verdict_mehrfachstart_gesamt("merkmal", 1, 1.0, 1.0, 0.0, -0.3, 0.3, 10, 1.0, 50)[1]
    art, text = V.verdict_mehrfachstart_gesamt("merkmal", 5890, 0.10, 0.21, -0.11, -0.21, -0.03, 297, 14.0, 20, (0, 5))
    assert art == "info" and "knapp vorn (kleiner Unterschied unter der Marke von 0,25 Prozentpunkten)" in text and "In 0 von 5 Gewichts-Zufalls-Seeds liegt das Netz um mindestens 0,25 Prozentpunkte vorn" in text
    assert "297 Instanzen (3 Instanzen ohne vergleichbaren Wert)" in V.verdict_mehrfachstart_gesamt("merkmal", 5890, 0.10, 0.21, -0.11, -0.21, -0.03, 297, 14.0, 20, (0, 5), 3)[1]
    assert "(" not in V.verdict_mehrfachstart_gesamt("merkmal", 5890, 0.10, 0.21, -0.11, -0.21, -0.03, 297, 14.0, 20, (0, 5))[1].split(":")[0]
    art, text = V.verdict_mehrfachstart_gesamt("zeiger", 6112, 0.31, 0.20, 0.11, 0.03, 0.2, 297, 14.0, 20)
    assert art == "info" and "klassische Mehrfachstart knapp vorn" in text
    assert "kein Urteil" in V.verdict_mehrfachstart_gesamt("merkmal", 1, 1.0, 1.0, 0.0, -0.3, 0.3, 3, 1.0, 50)[1]


# ---------------------------------------------------------------------------------------------------------------- Frage 5
def test_aktive_suche():
    art, text = V.verdict_aktive_suche(-0.5, -0.8, -0.2, 50, 512, 1e-3, 3.4)
    assert art == "success" and "512 Touren" in text and "Lernrate 0,001" in text and "0,50 Prozentpunkte vor dem Sampling" in text and "das 3,4-Fache" in text
    art, text = V.verdict_aktive_suche(4.7, 3.0, 6.0, 50, 1024, 1e-2, 3.4)
    assert art == "warning" and "Lernrate 0,01" in text and "hinter dem Sampling" in text
    art, text = V.verdict_aktive_suche(0.1, -0.3, 0.4, 50, 128, 3e-3, 3.4)
    assert art == "info" and "kein belegter Unterschied" in text and "Lernrate 0,003" in text
    art, text = V.verdict_aktive_suche(-0.14, -0.2, -0.05, 100, 2048, 1e-3, 4.1)
    assert art == "info" and "kleiner Unterschied unter der Marke von 0,25 Prozentpunkten" in text
    assert V.verdict_aktive_suche(-0.5, -0.8, -0.2, 2, 512, 1e-3, 3.4)[1].startswith("Mit 2 Instanzen")


# ---------------------------------------------------------------------------------------------------------------- Frage 6
def test_wechselkurs_drei_faelle_und_ausserhalb():
    art, text = V.verdict_wechselkurs(1240.0, 8)
    assert art == "success" and "mehr als 1 240 Operationen" in text and "angenommen sind 8" in text
    art, text = V.verdict_wechselkurs(40.0, 8)
    assert art == "info" and "das 5,0-Fache der Annahme" in text and "hängt an der Annahme" in text
    art, text = V.verdict_wechselkurs(5.0, 8)
    assert art == "warning" and "auch in Operationen vorn" in text
    art, text = V.verdict_wechselkurs(80.0, 8)
    assert art == "success"                               # genau das Zehnfache gilt als weit
    assert "außerhalb des gemessenen Bereichs" in V.verdict_wechselkurs(None, 8)[1]
