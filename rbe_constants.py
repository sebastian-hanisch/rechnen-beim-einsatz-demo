"""Konstanten (Zwischenstand; wird mit der App erweitert).

Die Namen und Werte der Netz-, Trainings-, Instanz- und Zweck-Konstanten aus Stück 1 (`zeigernetz-demo`) sind Schnittstelle zu den kopierten Dateien `rbe_net`, `rbe_train`,
`rbe_scenario`, `rbe_instances`, `rbe_stats`, `rbe_cost` (Test `tests/test_copies.py` vergleicht sie mit der Quelle). Die Konstanten der Kopien aus Stück 4 stehen in `rbek_constants.py`."""

# ---------------------------------------------------------------- Netz und Training (Stück 1)
C_CLIP = 10.0
BATCH = 120
LERNRATE = 1e-3
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
NORM_BEGRENZUNG = 1.0
EVAL_ALLE = 100
LETZTE_MESSPUNKTE = 3
MODELL_FORMAT = "znz-modell-1"
OPS_JE_KANDIDAT = 8

# ---------------------------------------------------------------- Instanzen (Quelle: hyperheuristik-demo über Stück 1)
FAMILIEN = ("gleich", "cluster", "gitter")
FAMILIE_HAND = "hand"
SEITE = 1000.0
CLUSTER_ZAHL, CLUSTER_SIGMA = 5, 40.0
GITTER_FAKTOR, GITTER_RAUSCHEN = 1.5, 0.1
VAL_OFFSET = 1000
HAND_PUNKTE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5))
HAND_SKALA = 4.0

# ---------------------------------------------------------------- Verdikte (gesetzt, nicht abgeleitet; Werte von Stück 1)
MARKE_KLAR, MARKE_GLEICH = 0.25, 0.10      # Prozentpunkte
BOOTSTRAP_ZIEHUNGEN = 2000

# ---------------------------------------------------------------- Zwecke der Zufalls-Seeds (reservierter Bereich dieser Demo: 200 bis 249)
ZWECK_INIT, ZWECK_TRAIN_X, ZWECK_TRAIN_S, ZWECK_BOOT = 2, 3, 4, 6      # Werte von Stück 1 (Training und Statistik der Kopien)
ZWECK_INSTANZ, ZWECK_START, ZWECK_LAUF = 11, 33, 24                    # Werte der Quelle hhw_scenario
ZWECK_ZUFALL = 8                # Wert von Stück 1 (hier ungenutzt, damit die Namen gleich bleiben)
ZWECK_SAMPLE_EINSATZ = 200      # gezogene Touren der Messung (Zeigernetz und Merkmalsnetz K)
ZWECK_ZUFALLSTOUREN = 201       # Zufallstouren (Gegner, Neustart-Suche)
ZWECK_AKTIV = 202               # aktive Suche
ZWECK_EXPORT = 203              # Export für Stück 6
ZWECK_LIVE = 204                # Ziehungen der App

# ---------------------------------------------------------------- Regler und Rechengrenzen der App
N_HAND = 5                                  # n = 5: nur das Handbeispiel (feste Fahrzeitmatrix)
N_STUFEN = (5, 20)
DEFAULT_N = 20
INSTANZEN_STUFEN = (4, 8, 12, 16)           # Instanzen, die die Klicks live rechnen
DEFAULT_INSTANZEN = 8
SEED_MAX = 11                               # Zufalls-Seed 0 bis 11: Instanzen 16 s bis 16 s + 15 (höchstens Test-Zufalls-Seed 191)
INSTANZEN_JE_SEED = 16
NETZE = ("zeiger", "merkmal")
NETZ_ANZEIGE = {"zeiger": "Zeigernetz", "merkmal": "Merkmalsnetz"}
MODELL_SEEDS = 5                            # Gewichts-Zufalls-Seed 0 bis 4
VERFAHREN = ("mehrfach", "sampling", "strahl")
VERFAHREN_ANZEIGE = {"mehrfach": "Mehrfachstart (Startstädte)", "sampling": "Sampling (gezogene Touren)", "strahl": "Strahlsuche (Breite)"}
K_STUFEN = (1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512)
SYMM_STUFEN = (1, 8)
ACHSEN = ("K", "O", "beide")
ACHSE_ANZEIGE = {"K": "Kandidatenbewertungen (K)", "O": "Grobe Operationen (O)", "beide": "K und O nebeneinander"}
STARTARTEN = ("s4", "s5", "s6", "netz1", "netz2")
START_ANZEIGE = {"s4": "Nächster Nachbar ab Ort 0", "s5": "Nächster Nachbar, beste aus 20 Startstädten", "s6": "Farthest Insertion", "netz1": "Netz, Greedy ab Ort 0",
                 "netz2": "Netz, beste aus 20 Startstädten"}
AKTIV_SCHRITTE_STUFEN = (4, 8, 16, 32, 64, 128)
AKTIV_TOUREN_STUFEN = (4, 8, 16, 32)
AKTIV_LR_STUFEN = (3e-4, 1e-3, 3e-3, 1e-2)
AKTIV_INSTANZEN_STUFEN = (1, 2, 3, 4)
C_STUFEN = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
DEFAULT_C = 8
RECHENGRENZE_K = 1_000_000                  # Kandidatenbewertungen je Klick (Netz und Suche zusammen); die Instanzzahl wird gekürzt, die App sagt es
RECHENGRENZE_TOUREN = 8192                  # gezogene Touren der aktiven Suche je Klick
MIN_INSTANZEN = 4                           # darunter gibt es kein Urteil
ZEITWAECHTER_SEKUNDEN = 100.0               # aktive Suche in der App bricht danach ab

# ---------------------------------------------------------------- Farben
COLOR_ZEIGER = "#2F6B65"
COLOR_MERKMAL = "#8E5EA2"
COLOR_NN = "#e45756"
COLOR_SUCHE = "#f58518"

# ---------------------------------------------------------------- Presets (das erste ist die Voreinstellung beim Öffnen; jedes setzt mindestens einen Regler anders)
_BASIS = {
    "n": DEFAULT_N, "instanzen": DEFAULT_INSTANZEN, "seed": 0, "netz": "zeiger", "modell": 0, "verfahren": "mehrfach", "k": 1, "symm": 1, "achse": "beide", "start": "s4",
    "schritte": 32, "touren": 16, "lr": 1e-3, "aktiv_inst": 2, "c": DEFAULT_C,
}
PRESETS = {
    "Eine Tour: Netz gegen Nächster Nachbar": dict(_BASIS),
    "Alle Startstädte": {**_BASIS, "k": 20},
    "Acht gesampelte Touren": {**_BASIS, "verfahren": "sampling", "k": 8},
    "Strahlsuche, Breite 8": {**_BASIS, "verfahren": "strahl", "k": 8},
    "Gleiches Budget: wo kippt es?": {**_BASIS, "verfahren": "sampling", "k": 8, "achse": "K"},
    "Gleiche Operationen": {**_BASIS, "verfahren": "sampling", "k": 8, "achse": "O"},
    "Netz baut, Suche verbessert: Zeigernetz": {**_BASIS, "k": 20, "start": "netz2"},
    "Netz baut, Suche verbessert: Merkmalsnetz": {**_BASIS, "netz": "merkmal", "k": 20, "start": "netz2"},
    "Das Netz auf der Instanz trainieren": {**_BASIS, "schritte": 64},
    "Zu große Lernrate": {**_BASIS, "schritte": 64, "lr": 1e-2},
    "Wie teuer müsste ein Delta sein?": {**_BASIS, "verfahren": "sampling", "k": 8, "c": 512},
    "Handbeispiel: fünf Städte": {**_BASIS, "n": N_HAND},
}
PRESET_HELP = {
    "Eine Tour: Netz gegen Nächster Nachbar": "Frage 1: eine Tour des Netzes (190 Kandidatenbewertungen) gegen den Nächsten Nachbarn mit denselben 190, und was beide in Operationen kosten. Zeigernetz, 8 Instanzen, Zufalls-Seed 0.",
    "Alle Startstädte": "Frage 2: Mehrfachstart, Greedy ab allen 20 Startstädten (3 800 Kandidatenbewertungen), gegen den Nächsten Nachbarn aus allen 20 Startstädten.",
    "Acht gesampelte Touren": "Frage 2: acht nach den Wahrscheinlichkeiten des Netzes gezogene Touren (1 520 Kandidatenbewertungen), die kürzeste zählt.",
    "Strahlsuche, Breite 8": "Frage 2: Strahlsuche der Breite 8 (1 387 Kandidatenbewertungen, nicht 8 mal 190: der erste Schritt hat nur eine Hypothese).",
    "Gleiches Budget: wo kippt es?": "Frage 3 in K: Sampling mit 8 Touren gegen die Neustart-Suche bei gleicher Zahl Kandidatenbewertungen (eingefrorene Messung, 300 Instanzen).",
    "Gleiche Operationen": "Frage 3 in O: dieselbe Rechnung, aber das Budget der Suche ist das O des Netzes geteilt durch 8 Operationen je Delta.",
    "Netz baut, Suche verbessert: Zeigernetz": "Frage 4: das Zeigernetz baut die Startlösung (beste aus 20 Greedy-Touren), 2-opt + Or-opt verbessert sie; Kosten gegen die klassischen Startlösungen.",
    "Netz baut, Suche verbessert: Merkmalsnetz": "Frage 4 mit dem Merkmalsnetz als Startlösung (beste aus 20 Greedy-Touren); Ausblick n = 50 im Expander bei Frage 4.",
    "Das Netz auf der Instanz trainieren": "Frage 5: aktive Suche, 64 Schritte mit 16 Touren je Schritt, Lernrate 0,001, auf 2 Instanzen; der Knopf startet die Rechnung (2 048 Touren).",
    "Zu große Lernrate": "Frage 5 mit Lernrate 0,01: die aktive Suche kann dann hinter dem Sampling liegen.",
    "Wie teuer müsste ein Delta sein?": "Frage 6: Sampling mit 8 Touren, Annahme 512 statt 8 Operationen je Delta; zeigt, wie weit die Antwort an der Annahme hängt.",
    "Handbeispiel: fünf Städte": "Fünf Städte als Fahrzeitmatrix zum Nachrechnen: Mini-Netz mit zwei Parametern, exaktes Sampling, Strahlsuche, Gradient der aktiven Suche (Expander Handbeispiel).",
}
