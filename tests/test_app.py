"""End-to-end Rauchtests über Streamlits AppTest: die App lädt mit den Standardeinstellungen, jedes Preset läuft ohne Python-Fehler und zeigt Verdikte, der Knopf der aktiven Suche rechnet,
das Handbeispiel zeigt keine Experimente. Plotly-Fehler im Browser sieht AppTest nicht (die Browser-Prüfung steht in der README)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rbe_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last reichen 180 s nicht immer


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in (at.success, at.info, at.warning) for e in kind) + " " + " ".join(m.value for m in at.markdown)


def test_app_laedt_ohne_fehler_und_zeigt_die_sechs_kernfragen():
    at = _app()
    ueberschriften = [s.value for s in at.subheader]
    for frage in ("Wer baut mit einer Tour", "Was bringt es, mehr Touren", "gleicher Gesamtarbeit", "als Startlösung", "weiter zu trainieren", "Wie teuer müsste"):
        assert any(frage in u for u in ueberschriften), frage
    assert not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    text = _texte(at)
    if name.startswith("Handbeispiel"):
        assert "n = 5 ist nur das Handbeispiel" in text and "Wahrheit:" in text
        assert not [s for s in at.subheader if "Wer baut mit einer Tour" in s.value]
    else:
        assert "Kandidatenbewertungen (K)" in text and "Operationen" in text


def test_frage_1_verdikt_nennt_k_und_o_und_die_ausnahme_im_selben_satz():
    at = _app()
    verdikte = [str(e.value) for kind in (at.success, at.info, at.warning) for e in kind if "Nächsten Nachbarn" in str(e.value) and "Instanzen" in str(e.value)]
    assert verdikte, "kein Verdikt zu Frage 1"
    v = verdikte[0]
    assert "190" in v and "8 Instanzen" in v and "Zeigernetz" in v and ("gilt in K, nicht in O" in v or "kein belegter" in v or "längere" in v)


def test_aktive_suche_knopf_rechnet_und_zeigt_ein_verdikt():
    at = _app()
    at.sidebar.select_slider(key="schritte_select").set_value(4)
    at.sidebar.select_slider(key="touren_select").set_value(4)
    at.sidebar.select_slider(key="aktiv_inst_select").set_value(4)
    at.run()
    assert "Noch nicht gerechnet" in _texte(at)
    at.button(key="aktiv_knopf").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Sampling mit gleich vielen Touren" in _texte(at) and "Noch nicht gerechnet" not in _texte(at)
    assert "aktiv5" in at.session_state and at.session_state["aktiv5"]["res"]["instanzen"] == 4


def test_nach_aenderung_der_einstellung_verschwindet_das_alte_ergebnis_der_aktiven_suche():
    at = _app()
    at.sidebar.select_slider(key="schritte_select").set_value(4)
    at.sidebar.select_slider(key="touren_select").set_value(4)
    at.run()
    at.button(key="aktiv_knopf").click().run()
    at.sidebar.select_slider(key="lr_select").set_value(0.01)
    at.run()
    assert "Noch nicht gerechnet" in _texte(at)
