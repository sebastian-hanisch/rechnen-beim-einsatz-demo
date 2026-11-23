"""Jede Zahl der README gegen die eingefrorene Messung (`rbe_band.json`), die Module und die Rohdaten.

Die Befundtabelle der README ist die Ausgabe von `rbe_claims.befund_tabelle` (dieselben Sätze wie die Verdikte der App) und steht wortgleich in der README. Zusätzlich rechnen die Tests unten die
Kernzahlen **unabhängig** aus den Rohvektoren der Messreihe (numpy-Mittel, eigene Zählung) und prüfen die Einordnung (Marke 0,25 Prozentpunkte, Intervall ohne 0). Zahlen ohne Test stehen in der README
als „Messung dieser README, nicht im Test“ (Zeiten, Speicher, die Spalte „Plan“ der Tabelle der Korrekturen)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import rbe_band as BD
import rbe_budget as B
import rbe_claims as CL
import rbe_evaluation as V
import rbe_modelle as MO
import rbe_stats as ST
import rbe_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
BAND = BD.lade_band()
de, zahl = V.de, V.zahl


def steht(*texte):
    for t in texte:
        assert t in README, f"nicht in der README: {t!r}"


def _g(netz, verf, k):
    """Rohvektor der Lücken (Seeds, Instanzen) einer Netz-Zeile."""
    return np.array(BAND["netz"][netz][verf][str(k)]["gap"])


def _klass(art, k=None):
    z = BAND["klass"][art] if k is None else BAND["klass"][art][str(k)]
    return np.array(z["gap"])


def _suche_bei(b):
    """Lücke der Neustart-Suche je Instanz beim größten Rasterpunkt unter b (roh aus dem Raster)."""
    raster = BAND["klass"]["neustart"]["raster"]
    i = max(j for j, r in enumerate(raster) if r <= b)
    return np.array(BAND["klass"]["neustart"]["gap"][i])


def _klasse(d_vektor):
    """Marke und Intervall wie in der Messreihe: gepaart über die Instanzen."""
    c = ST.compare(d_vektor, np.zeros_like(d_vektor))
    return V.klasse_paar(c["v"], c["lo"], c["hi"]), c


# ------------------------------------------------------------------------------------------------------------- Struktur
def test_befundtabelle_steht_wortgleich_in_der_readme():
    assert CL.readme_block(BAND) in README


def test_kopf_der_messung_steht_in_der_readme():
    k = BAND["kopf"]
    assert k["instanzen"] == 300 and k["aktiv_instanzen"] == 100 and k["symm_instanzen"] == 100 and k["n50_instanzen"] == 10
    steht("300 Testinstanzen n = 20", "aktive Suche auf 100 Instanzen", "Ausblick n = 50 auf 10 Instanzen")


def test_annahmenblock_der_app_steht_wortgleich_in_der_readme():
    assert TX.GRENZEN.strip() in README


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    m = re.search(r"(\d+) Tests in (\d+) Dateien", README)
    assert m, "Testzahl fehlt in der README"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=170).stdout
    zeilen = [z for z in out.splitlines() if "::" in z]
    assert int(m.group(1)) == len(zeilen)
    assert int(m.group(2)) == len(list((ROOT / "tests").glob("test_*.py")))


def test_jeder_testverweis_der_readme_gibt_es():
    verweise = re.findall(r"`tests/(test_[a-z_]+\.py)::(test_[a-z0-9_]+)`|`(test_[a-z_]+\.py)`", README)
    assert len(verweise) >= 15
    for datei, funktion, nur_datei in verweise:
        pfad = ROOT / "tests" / (datei or nur_datei)
        assert pfad.exists(), datei or nur_datei
        if funktion:
            assert f"def {funktion}(" in pfad.read_text(encoding="utf-8"), (datei, funktion)


def test_readme_hat_die_pflichtabschnitte_in_der_reihenfolge_und_keine_planungswoerter():
    kopf = [z for z in README.splitlines() if z.startswith("## ")]
    erwartet = ["## Kernfrage und Befund in Kürze", "## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Training und Zähleinheiten",
                "## Befunde und Korrekturen gegenüber dem Plan", "## Wo die Annahmen enden", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht enthalten",
                "## Quellen (mit Belegstufe)", "## Verweise", "## Lokal ausführen"]
    assert kopf == erwartet
    for wort in ("Stück", "Hook", "Preset", "Zuschnitt", "BAUPLAN", "Phase 0", "Welle", "Strohmann", "Schwanz", "Saatwert", "TLA", "läuft lokal"):
        assert wort not in README, wort
    assert not re.search(r"Kante(?![a-zäöü])", README), "„Kante“ ist ein Wort der Planung"
    assert not re.search(r"(?<![a-zA-Zäöüß])(trägt|tragen|tragend\w*)(?![a-zA-Zäöüß])", README), "„trägt/tragen“ vermeiden (beträgt ist gemeint und erlaubt)"
    assert "Zufalls-Seed" in README and "numpy.random" not in README.replace("kein `numpy.random`", "")


def test_modelldateien_stehen_in_der_readme_mit_namen():
    steht("`rbe_n20_voll_d32_s0..4.npz`", "`featnet_s0..4.npz`", "znz-modell-1", "zwl-mlp-1", "data/rbe_ngg_export.json", "ngg_export_v1")
    for s in range(5):
        assert MO.zeiger_pfad(s).exists() and (ROOT / "rbek_modelle" / f"featnet_s{s}.npz").exists()


def test_alle_dateien_der_dateistruktur_gibt_es():
    namen = re.findall(r"`((?:rbe|rbek)_[a-z_]+\.py|app\.py|modul_frisch\.py|tools/[a-z_]+\.py)`", README)
    assert len(set(namen)) >= 30
    for n in set(namen):
        assert (ROOT / n).exists(), n


# ------------------------------------------------------------------------------------------------------------- Frage 1
def test_eine_tour():
    nn = _klass("nn_mehrfach", 1)
    for netz, soll_o, soll_k in (("zeiger", 246384, 190), ("merkmal", 47642, 190)):
        g = _g(netz, "mehrfach", 1)
        je_inst = g.mean(axis=0)
        d = je_inst - nn
        klasse, c = _klasse(d)
        assert klasse == "a_vorn" and c["v"] < -7 and c["hi"] < 0
        assert all((g[s] - nn).mean() < 0 for s in range(5))                                  # in 5 von 5 Gewichts-Zufalls-Seeds im Mittel unter dem Nächsten Nachbarn
        assert BAND["netz"][netz]["mehrfach"]["1"]["K"] == soll_k and round(BAND["netz"][netz]["mehrfach"]["1"]["O"]) == soll_o
    z, m = _g("zeiger", "mehrfach", 1), _g("merkmal", "mehrfach", 1)
    steht(f"{de(z.mean())} %", f"{de(m.mean())} %", f"{de(nn.mean())} %", "das 162-Fache", "das 31-Fache", "246 384", "47 642", "In 5 von 5 Gewichts-Zufalls-Seeds liegt das Netz unter dem Nächsten Nachbarn")
    assert round(246384 / 1520) == 162 and round(47642 / 1520) == 31
    assert abs(z.mean() - 10.24) < 0.01 and abs(m.mean() - 8.69) < 0.01 and abs(nn.mean() - 17.46) < 0.01
    steht("Zufallstour")  # Zeile der Tabelle der Korrekturen oder Methodik
    assert round(float(_klass("zufall", 1).mean())) == 171


# ------------------------------------------------------------------------------------------------------------- Frage 2
def test_gleiche_konstruktionen():
    nn20 = _klass("nn_mehrfach", 20)
    z20, m20 = _g("zeiger", "mehrfach", 20), _g("merkmal", "mehrfach", 20)
    assert abs(nn20.mean() - 5.49) < 0.01 and abs(z20.mean() - 2.19) < 0.01 and abs(m20.mean() - 0.99) < 0.01
    for g in (z20, m20):
        assert _klasse(g.mean(axis=0) - nn20)[0] == "a_vorn"
    steht("2,19", "5,49", "0,99")
    # Sampling des Merkmalsnetzes bei k = 32 gegen den Nächsten Nachbarn aus 20 Startstädten: gleichauf
    s32 = _g("merkmal", "sampling", 32)
    klasse, c = _klasse(s32.mean(axis=0) - nn20)
    assert klasse == "gleich" and abs(c["v"]) < 0.1 and abs(s32.mean() - 5.55) < 0.01
    steht("5,55 gegen 5,49 %")
    # Sampling k = 8 des Zeigernetzes gegen NN aus 8 Startstädten, Strahl 8
    nn8 = _klass("nn_mehrfach", 8)
    assert _klasse(_g("zeiger", "sampling", 8).mean(axis=0) - nn8)[0] == "a_vorn" and _klasse(_g("zeiger", "strahl", 8).mean(axis=0) - nn8)[0] == "a_vorn"
    assert BAND["netz"]["zeiger"]["strahl"]["8"]["K"] == 1387


def test_je_konstruktion():
    for netz, soll in (("zeiger", 4), ("merkmal", 4)):
        zaehl = sum(1 for k in (2, 4, 8, 16) if _g(netz, "mehrfach", k).mean() < _g(netz, "sampling", k).mean())
        assert zaehl == soll
    steht("in 4 von 4 Fällen die kleinere mittlere Lücke als das Sampling")
    steht(f"k = 8: {de(_g('zeiger', 'mehrfach', 8).mean())} gegen {de(_g('zeiger', 'sampling', 8).mean())} %")


# ------------------------------------------------------------------------------------------------------------- Frage 3
def _klassen_je_budget(netz, verf, einheit, c=8):
    out = []
    for k in sorted(int(x) for x in BAND["netz"][netz][verf]):
        z = BAND["netz"][netz][verf][str(k)]
        b = int(z["K"]) if einheit == "K" else int(z["O"] // c)
        d = np.array(z["gap"]).mean(axis=0) - _suche_bei(b)
        cm = ST.compare(d, np.zeros_like(d))
        out.append((b, V.klasse_paar(cm["v"], cm["lo"], cm["hi"])))
    return out


def test_kreuzung():
    # Neustart-Suche (Mittel über die Instanzen) bei den Budgets der Planung
    xs = np.array(BAND["klass"]["neustart"]["raster"])
    ys = np.array([np.mean(g) for g in BAND["klass"]["neustart"]["gap"]])
    wert = {b: ys[list(xs).index(b)] for b in (128, 512, 2048, 8192)}
    assert abs(wert[128] - 43.66) < 0.01 and abs(wert[512] - 4.21) < 0.01 and abs(wert[2048] - 0.81) < 0.01 and abs(wert[8192] - 0.09) < 0.01
    steht("43,66", "4,21", "0,81", "0,09")
    assert (np.diff(ys) <= 1e-9).all()
    zaehlung = {}
    for netz in ("zeiger", "merkmal"):
        for verf in ("mehrfach", "sampling", "strahl"):
            k_kl = _klassen_je_budget(netz, verf, "K")
            o_kl = _klassen_je_budget(netz, verf, "O")
            assert [kl for b, kl in o_kl].count("b_vorn") == len(o_kl)                 # in O liegt die Suche bei jedem gemessenen Budget vorn
            assert k_kl[0] == (190, "a_vorn")                                           # bei einer Tour liegt das Netz vorn
            zaehlung[(netz, verf)] = k_kl
    for verf in ("mehrfach", "sampling", "strahl"):
        rest = [kl for b, kl in zaehlung[("zeiger", verf)][1:]]
        assert rest.count("b_vorn") == len(rest)                                        # Zeigernetz: ab der zweiten Tour liegt die Suche in K vorn
    mk = zaehlung[("merkmal", "mehrfach")]
    assert [kl for b, kl in mk[1:5]] != ["b_vorn"] * 4 and [b for b, kl in mk[1:5]] == [380, 570, 760, 950]      # Merkmalsnetz, Mehrfachstart: bei 380 bis 950 kein klares Urteil
    assert all(kl == "b_vorn" for b, kl in mk[5:])
    assert all(kl == "b_vorn" for b, kl in zaehlung[("merkmal", "sampling")][1:])
    steht("liegt das Netz bei 190 vorn; dann gibt es bei 380 bis 950 kein klares Urteil (4 Budgets); dann liegt die Suche bei 1 140 bis 3 800 vorn", "liegt die Suche bei allen 20 gemessenen Budgets vorn")


# ------------------------------------------------------------------------------------------------------------- Frage 4
def test_startloesung():
    s5 = np.array(BAND["start"]["klass"]["S5"]["gap"])
    for netz, end, k in (("zeiger", 0.22, 6112), ("merkmal", 0.10, 5890)):
        z = BAND["start"][netz]["S2"]
        g = np.array(z["gap"]).mean(axis=0)
        assert abs(g.mean() - end) < 0.006 and abs(np.mean(z["K"]) - k) < 1
        assert g.mean() < s5.mean() and np.mean(z["K"]) > 2 * np.mean(BAND["start"]["klass"]["S6"]["K"])      # kostet mehr als das Doppelte der günstigsten klassischen Startlösung
        assert np.mean(z["O"]) > 40 * np.mean(BAND["start"]["klass"]["S6"]["O"])
    assert abs(s5.mean() - 0.36) < 0.006
    s6k = np.mean(BAND["start"]["klass"]["S6"]["K"])
    assert round(s6k) == 2492 and np.mean(BAND["start"]["klass"]["S6"]["O"]) < 25000
    steht("6 112", "5 890", "2 492", "0,020 Mio Operationen", "0,36")


def test_mehrfachstart_gesamt():
    import rbe_befunde as BF
    mgz = BF.mehrfachstart_gesamt(BAND["start"]["zeiger"]["S2"], BAND["folgen"]["nn"])
    mgm = BF.mehrfachstart_gesamt(BAND["start"]["merkmal"]["S2"], BAND["folgen"]["nn"])
    assert abs(mgz["cmp"]["v"]) < 0.1 and -0.25 < mgm["cmp"]["v"] < -0.05 and mgm["cmp"]["hi"] < 0
    assert mgz["seeds_klar"] == (0, 5) and mgm["seeds_klar"] == (0, 5)                      # die vorab gesetzte Regel (4 von 5 Seeds mit 0,25 Prozentpunkten) ist nicht erfüllt
    assert V.klasse_paar(mgm["cmp"]["v"], mgm["cmp"]["lo"], mgm["cmp"]["hi"]) == "a_klein"
    assert mgz["instanzen"] == 297 and mgm["instanzen"] == 297 and mgz["ausgelassen"] == 3
    steht("kein belegter Unterschied", "knapp vorn (kleiner Unterschied unter der Marke von 0,25 Prozentpunkten)", "In 0 von 5 Gewichts-Zufalls-Seeds liegt das Netz um mindestens 0,25 Prozentpunkte vorn")
    assert 15 < mgm["o_faktor"] < 25


def test_ausblick_n50():
    n50 = BAND["n50"]
    assert n50["instanzen"] == 10 and len(n50["bewiesen"]) == 10 and np.array(n50["merkmal"]["S2"]["gap"]).shape == (5, 10)
    steht(f"bei {sum(n50['bewiesen'])} von 10 Instanzen bewiesen", "n = 50, 10 Instanzen")
    import rbe_befunde as BF
    mg = BF.mehrfachstart_gesamt(n50["merkmal"]["S2"], n50["folge"], n=50)
    assert V.klasse_paar(mg["cmp"]["v"], mg["cmp"]["lo"], mg["cmp"]["hi"]) == "gleich" and abs(mg["cmp"]["v"]) < 0.25


def test_pruefrunde():
    ev4 = np.mean(BAND["start"]["klass"]["S4"]["ev"])
    assert abs(100 * 1850 / ev4 - 67) < 0.5 and B.K0.m_verschiedene_zuege(20) == 1850 and B.K0.m_verschiedene_zuege(50) == 12875
    assert round(ev4) == 2775
    steht("67 %", "2 775", "1 850", "33 %", "1 570 verschiedene Nachbartouren", "etwa ein Drittel")
    # die 1 850 Züge sind Zugbeschreibungen: als Nachbartouren (Kantenmengen) sind es bei n = 20 nur 1 570 verschiedene, bei n = 50 12 175 (Aufzählung der Prüfung, Zugmenge unverändert)
    import rbek_moves as MV
    import rbe_texts as TX
    for n, soll in ((20, 1570), (50, 12175)):
        nb = MV.Neighborhood(n)
        t = np.arange(n)
        touren = {frozenset(frozenset((int(u[i]), int(u[(i + 1) % n]))) for i in range(n)) for u in (MV.apply_move(t, k, nb) for k in range(nb.M))}
        assert nb.M == MV.neighborhood_size(n) and len(touren) == soll
    assert TX.NACHBARTOUREN_20 == 1570


# ------------------------------------------------------------------------------------------------------------- Frage 5
def test_aktive_suche():
    a = BAND["aktiv"]
    assert a["instanzen"] == 100 and a["touren_je_schritt"] == 16
    erwartet = {8: ("0.001", "a_vorn", 1.89, 2.45), 32: ("0.001", "a_vorn", 1.07, 1.51), 128: ("0.001", None, 0.79, 0.93), 128.01: ("0.01", "b_vorn", 2.69, 0.93)}
    for key, (lr, klasse_soll, ga, gs) in erwartet.items():
        schritte = int(key)
        i = a["schritte"].index(schritte)
        aktiv = np.array(a["aktiv"][lr])[:, i, :]
        samp = np.array(a["sampling"])[:, i, :]
        assert abs(aktiv.mean() - ga) < 0.006 and abs(samp.mean() - gs) < 0.006, (schritte, lr)
        klasse, c = _klasse(aktiv.mean(axis=0) - samp.mean(axis=0))
        if klasse_soll:
            assert klasse == klasse_soll
        else:
            assert klasse in ("gleich", "a_klein")                                        # 2 048 Touren: kein Unterschied von mindestens 0,25 Prozentpunkten
    assert 3.8 < a["O"]["8"] / a["O_sampling"]["8"] < 4.2 and 3.8 < a["O"]["128"] / a["O_sampling"]["128"] < 4.3
    steht("1,89 gegen 2,45 %", "1,07 gegen 1,51 %", "0,79 gegen 0,93 %", "2,69 gegen 0,93 %")


# ------------------------------------------------------------------------------------------------------------- Frage 6
def _b_stern(luecke):
    """Kleinstes Budget, bei dem die Neustart-Suche im Mittel die Lücke erreicht: log-log-Interpolation, eigene Umsetzung."""
    xs = np.array(BAND["klass"]["neustart"]["raster"], dtype=float)
    ys = np.array([np.mean(g) for g in BAND["klass"]["neustart"]["gap"]])
    ok = ys > 0
    xs, ys = xs[ok], ys[ok]
    for i in range(len(xs) - 1):
        if ys[i + 1] <= luecke <= ys[i]:
            s = np.log(ys[i + 1] / ys[i]) / np.log(xs[i + 1] / xs[i])
            return float(xs[i] * (luecke / ys[i]) ** (1.0 / s))
    return None


def test_wechselkurs():
    erwartet = (("zeiger", "sampling", 8, 1406), ("zeiger", "mehrfach", 20, 884), ("zeiger", "sampling", 512, 14352), ("merkmal", "mehrfach", 20, 520), ("merkmal", "sampling", 8, 1083), ("merkmal", "sampling", 512, 46008))
    for netz, verf, k, soll in erwartet:
        z = BAND["netz"][netz][verf][str(k)]
        b = _b_stern(np.mean(z["gap"]))
        assert abs(z["O"] / b - soll) < 1.5, (netz, verf, k)
        assert z["O"] / b >= 10 * B.O_JE_DELTA                                            # „weit“: mindestens das Zehnfache der Annahme
        steht(f"mehr als {zahl(soll)} Operationen")
    # Annahme 8 je Delta: das Netz ist in O nirgends gleichauf (alle sechs Reihen: weit)


# ------------------------------------------------------------------------------------------------------------- Symmetrien, Zählregel, Training
def test_symmetrien():
    sy = BAND["symm"]
    eins, acht, mk = np.array(sy["eins"]), np.array(sy["acht"]), np.array(sy["merkmal"])
    assert abs(eins.mean() - 2.50) < 0.01 and abs(acht.mean() - 0.72) < 0.01 and eins.mean() - acht.mean() > 1.5 and sy["K_acht"] == 8 * sy["K_eins"] == 30400
    assert (acht.mean(axis=0) < eins.mean(axis=0)).mean() > 0.9                              # auf über 90 % der Instanzen kürzer
    steht("0,72 gegen 2,50 %", "30 400 gegen 3 800 K", f"Merkmalsnetz (invariant) {de(mk.mean())} %")


def test_zaehlregel_und_trainingszahlen():
    steht("246 384", "211 200", "35 184", "47 642", "1 520", "1 600", "162,1", "23,1", "31,3", "182 400 000", "27 360")
    assert B.o_zeiger(20, 1) == 246384 and B.o_kodierer(20) == 211200 and B.o_dekodierer(20) == 35184 and round(B.o_merkmal(20, 1)) == 47642 and B.o_nn(20) == 1520 and B.o_farthest(20) == 1600
    assert round(B.wechselkurs(246384, 190), 1) == 162.1 and round(B.wechselkurs(35184 * 1000, 190 * 1000), 1) == 23.1 and round(B.wechselkurs(47642, 190), 1) == 31.3
    assert 960000 * 190 == 182_400_000 == MO.trainingstouren("zeiger") * 190 and MO.trainingstouren("merkmal") == 960000
    assert 144 * 190 == 27360
    steht("12 480", "145", "960 000 Touren", "8 000 Schritte × 120 Touren", "3 750 Schritte × 256 Touren")


def test_strahl_und_handbeispiel_zahlen_der_readme():
    assert B.k_strahl(20, 8) == 1387 and B.k_strahl(20, 8) < 8 * 190
    steht("1 387 K, nicht 1 520")
    assert B.K0.m_verschiedene_zuege(20) == 1850
    steht("1 850")
