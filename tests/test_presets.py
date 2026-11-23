"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset unterscheidbar, Permalink (Begrenzen und Einrasten)."""

import pytest

import rbe_constants as C
import rbe_presets as P


def test_jedes_preset_hat_hilfe_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_presetwerte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            wert = preset[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key, wert)


def test_standard_preset_ist_die_voreinstellung_und_alle_presets_unterscheiden_sich():
    ersten = next(iter(C.PRESETS))
    assert ersten == "Eine Tour: Netz gegen Nächster Nachbar" and P.DEFAULT_PRESET == ersten
    for key, state_key in P.PRESET_KEYS.items():
        assert C.PRESETS[ersten][key] == P.SETTING_SPECS[state_key].default, key
    kombis = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(kombis)) == len(kombis)                                 # keine zwei Presets mit denselben Reglerwerten
    for name, p in C.PRESETS.items():
        if name != ersten:
            assert any(p[k] != C.PRESETS[ersten][k] for k in p), name       # jedes weitere setzt mindestens einen Regler anders


def test_setting_specs_decken_jeden_preset_schluessel():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(urls)) == len(urls)


@pytest.mark.parametrize("state_key,eingabe,soll", [
    ("seed_input", 99, 11), ("seed_input", -3, 0), ("modell_input", 9, 4), ("k_select", 3, 2), ("k_select", 600, 512), ("k_select", 18, 16), ("n_select", 7, 5), ("n_select", 14, 20),
    ("instanzen_select", 10, 8), ("c_select", 700, 512), ("netz_select", "kein-netz", None), ("netz_select", "merkmal", "merkmal"), ("lr_select", 0.0028, 0.003),
])
def test_clamp_begrenzt_und_rastet_ein(state_key, eingabe, soll):
    assert P.clamp(P.SETTING_SPECS[state_key], eingabe) == soll


def test_rechenregeln_der_voreinstellungen_bleiben_unter_den_grenzen():
    """Voreinstellung und Presets rechnen höchstens 1 000 000 Kandidatenbewertungen je Klick (Netz-Verfahren) und höchstens 8 192 Touren der aktiven Suche."""
    import rbe_experiments as EX
    for name, p in C.PRESETS.items():
        if p["n"] != C.DEFAULT_N:
            continue
        verf = p["verfahren"]
        k = min(p["k"], 20) if verf == "mehrfach" else p["k"]
        K = EX.kosten_netz(p["netz"], verf, k, symm=p["symm"] if (p["netz"] == "zeiger" and verf == "mehrfach") else 1)[0]
        assert K * p["instanzen"] <= C.RECHENGRENZE_K, name
        assert p["schritte"] * p["touren"] * p["aktiv_inst"] <= C.RECHENGRENZE_TOUREN, name
