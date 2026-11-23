"""Worst-Case-Test am Reglermaximum: Strahlsuche mit dem Merkmalsnetz bei Breite 512, 16 Instanzen und die größte aktive Suche. Statt einer Zeit-Assertion bricht der AppTest
mit `default_timeout` ab (DEMO-PLAYBOOK 1c): ein Klick darf nicht länger als das Zeitlimit dauern."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

import rbe_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
LIMIT = int(os.environ.get("RBE_WORSTCASE_SEKUNDEN", "90"))


def _app(**werte) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=LIMIT)
    at.run()
    for key, wert in werte.items():
        regler = at.sidebar.radio(key=key) if key in ("netz_select", "verfahren_select", "symm_select", "achse_select") else (
            at.sidebar.number_input(key=key) if key in ("seed_input", "modell_input") else at.sidebar.select_slider(key=key))
        regler.set_value(wert)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_strahlsuche_merkmalsnetz_breite_512_mit_16_instanzen():
    at = _app(netz_select="merkmal", verfahren_select="strahl", k_select=512, instanzen_select=16)
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "gekürzt" in " ".join(c.value for c in at.caption)      # 16 Instanzen x 75 807 K liegen über der Rechengrenze: die Zahl wird gekürzt, und die App sagt es
    assert "75 807" in text


def test_sampling_zeigernetz_512_symmetrien_und_groesste_aktive_suche():
    at = _app(verfahren_select="sampling", k_select=512, instanzen_select=16, c_select=1000, achse_select="beide")
    assert not at.exception
    at2 = _app(verfahren_select="mehrfach", k_select=20, symm_select=8, schritte_select=128, touren_select=32, aktiv_inst_select=4)
    at2.button(key="aktiv_knopf").click().run()
    assert not at2.exception, [str(e) for e in at2.exception]
    r = at2.session_state["aktiv5"]["res"]
    assert r["instanzen"] == 2 and r["touren_je_instanz"] == 128 * 32           # 8 192 Touren je Klick: aus 4 Instanzen werden 2
    assert 128 * 32 * 2 <= C.RECHENGRENZE_TOUREN
