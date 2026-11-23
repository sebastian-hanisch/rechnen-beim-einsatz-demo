"""Texte der App: Zahlen der Zählregel gegen die Module, Wortregeln (keine Planungswörter, kein Gedankenstrich im Fließtext, echte Umlaute), Hilfetexte der Schnellstart-Szenarien."""

import re

import rbe_budget as B
import rbe_constants as C
import rbe_texts as TX

ALLE = {name: getattr(TX, name) for name in ("TITEL", "INTRO", "EINFACHE_WORTE", "SO_FUNKTIONIERT", "ZAEHLREGEL", "ERKLAERTABELLE", "GRENZEN", "MATHE")}
ALLE["KERNFRAGEN"] = "\n".join(TX.KERNFRAGEN)
ALLE["PRESET_HELP"] = "\n".join(C.PRESET_HELP.values())


def test_zahlen_der_zaehlregel_stimmen_mit_den_modulen():
    assert "190" in TX.SO_FUNKTIONIERT and "1 850" in TX.ZAEHLREGEL
    assert "246 384" in TX.ZAEHLREGEL and "211 200" in TX.ZAEHLREGEL and "35 184" in TX.ZAEHLREGEL and "47 642" in TX.ZAEHLREGEL and "1 520" in TX.ZAEHLREGEL
    assert f"{B.o_zeiger(20, 1):,}".replace(",", " ") in TX.ZAEHLREGEL
    assert "1 387" in TX.ERKLAERTABELLE and "1 600" in TX.ERKLAERTABELLE and f"{B.o_aktiv(20, 1, 16):,}".replace(",", " ") in TX.ERKLAERTABELLE
    assert "960 000 Touren" in TX.ZAEHLREGEL


def test_keine_planungswoerter_und_keine_verbotenen_formulierungen():
    for name, text in ALLE.items():
        for wort in ("Stück", "Hook", "Preset", "Welle", "Zuschnitt", "Strohmann", "Saatwert", "TLA", "läuft lokal", "Phase 0"):
            if name == "ERKLAERTABELLE" and wort == "Stück":
                continue
            assert wort not in text, (name, wort)
        assert not re.search(r"(?<![a-zA-Zäöüß])(trägt|tragen)(?![a-zA-Zäöüß])", text), name
        assert "Schwanz" not in text and "Konfiguration" not in text.replace("Adresszeile", "")


def test_kein_gedankenstrich_im_fliesstext():
    for name, text in ALLE.items():
        assert "–" not in text and "—" not in text, name


def test_umlaute_sind_echte_umlaute():
    umschrift = re.compile(r"(?<![a-zA-Z])(fuer|ueber|Laenge|laenge|Loesung|loesung|Aenderung|waehl\w*|Waehl\w*|moeglich\w*|Groesse|groesse|zaehl\w*|Zaehl\w*|Staedte|Startstaedte|Luecke|luecke)(?![a-zA-Z])")
    for name, text in ALLE.items():
        assert not umschrift.search(text), (name, umschrift.search(text).group(0))
    assert "ä" in TX.INTRO or "ö" in TX.INTRO or "ü" in TX.INTRO


def test_jedes_schnellstart_szenario_hat_einen_hilfetext_mit_inhalt():
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, text in C.PRESET_HELP.items():
        assert len(text) > 40 and text.endswith(".") and "Preset" not in text


def test_zufalls_seed_schreibweise():
    for name, text in ALLE.items():
        assert "Saat" not in text and "Seed" not in text.replace("Zufalls-Seed", ""), name


def test_kernfragen_sind_sechs_und_nummeriert():
    assert len(TX.KERNFRAGEN) == 6 and all(f.startswith(f"{i + 1}. ") for i, f in enumerate(TX.KERNFRAGEN))
