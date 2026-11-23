"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rbe_constants as C
import rbe_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


def _passendes_preset(state_key, wert):
    """Name des Presets, dessen Werte genau den Standardwerten mit diesem einen geänderten Regler entsprechen (sonst None)."""
    basis = {sk: P.SETTING_SPECS[sk].default for sk in P.SETTING_SPECS}
    preset_key_of = {sk: key for key, sk in P.PRESET_KEYS.items()}
    neu = {**basis, state_key: wert}
    for name, p in C.PRESETS.items():
        if all(neu[sk] == p.get(preset_key_of[sk], P.SETTING_SPECS[sk].default) for sk in neu):
            return name
    return None


def _kandidaten(state_key, aktuell):
    spec = P.SETTING_SPECS[state_key]
    return [o for o in spec.options if o != aktuell] if spec.options is not None else [spec.hi if aktuell != spec.hi else spec.lo]


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    """Ein geänderter Regler hebt die Markierung auf. Trifft der neue Wert zusammen mit den Standardwerten genau ein anderes Preset (z. B. n = 5 das Handbeispiel), ist dieses zu Recht markiert."""
    at = _app()
    regler = _regler(at, state_key)
    kandidaten = _kandidaten(state_key, regler.value)
    ohne = [w for w in kandidaten if _passendes_preset(state_key, w) is None]
    wert = ohne[0] if ohne else kandidaten[0]
    regler.set_value(wert).run()
    assert not at.exception, [str(e) for e in at.exception]
    markiert = [n for n, typ in _typen(at).items() if typ == "primary"]
    assert markiert == ([] if ohne else [_passendes_preset(state_key, wert)])
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
