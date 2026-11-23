"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Button (Standardmuster aus dem Demo-Portfolio).

SETTING_SPECS: die Regler der Seitenleiste; sie stehen im Permalink und entscheiden, welches Preset hervorgehoben ist. Regler, die nur bei einer Wahl wirken (`symm_select` nur beim
Zeigernetz im Mehrfachstart), bleiben immer sichtbar und sind sonst abgeschaltet (disabled), sie verschwinden nicht."""

from dataclasses import dataclass

import streamlit as st

import rbe_constants as C
from rbe_rng import SplitMix64


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None       # Auswahlregler: erlaubte Werte


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, options=C.N_STUFEN),
    "instanzen_select": SettingSpec("instanzen", int, C.DEFAULT_INSTANZEN, options=C.INSTANZEN_STUFEN),
    "seed_input": SettingSpec("instanz", int, 0, 0, C.SEED_MAX),
    "netz_select": SettingSpec("netz", str, "zeiger", options=C.NETZE),
    "modell_input": SettingSpec("modell", int, 0, 0, C.MODELL_SEEDS - 1),
    "verfahren_select": SettingSpec("verfahren", str, "mehrfach", options=C.VERFAHREN),
    "k_select": SettingSpec("k", int, 1, options=C.K_STUFEN),
    "symm_select": SettingSpec("symmetrien", int, 1, options=C.SYMM_STUFEN),
    "achse_select": SettingSpec("achse", str, "beide", options=C.ACHSEN),
    "start_select": SettingSpec("start", str, "s4", options=C.STARTARTEN),
    "schritte_select": SettingSpec("schritte", int, 32, options=C.AKTIV_SCHRITTE_STUFEN),
    "touren_select": SettingSpec("touren", int, 16, options=C.AKTIV_TOUREN_STUFEN),
    "lr_select": SettingSpec("lernrate", float, 1e-3, options=C.AKTIV_LR_STUFEN),
    "aktiv_inst_select": SettingSpec("aktivinstanzen", int, 2, options=C.AKTIV_INSTANZEN_STUFEN),
    "c_select": SettingSpec("operationen", int, C.DEFAULT_C, options=C.C_STUFEN),
}
PRESET_KEYS = {
    "n": "n_select", "instanzen": "instanzen_select", "seed": "seed_input", "netz": "netz_select", "modell": "modell_input", "verfahren": "verfahren_select", "k": "k_select",
    "symm": "symm_select", "achse": "achse_select", "start": "start_select", "schritte": "schritte_select", "touren": "touren_select", "lr": "lr_select", "aktiv_inst": "aktiv_inst_select",
    "c": "c_select",
}
DEFAULT_PRESET = next(iter(C.PRESETS))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    """(Minimum, Maximum) eines Zahlenreglers."""
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key) -> tuple:
    """Erlaubte Werte eines Auswahlreglers."""
    return SETTING_SPECS[state_key].options


def clamp(spec: SettingSpec, value):
    """Wert aus der Adresszeile auf den Regler abbilden: Zahlen begrenzen, Auswahlwerte einrasten (Zahl: nächster Wert; Text: nur gültige)."""
    if spec.options is not None:
        if value in spec.options:
            return value
        if spec.caster in (int, float):
            return min(spec.options, key=lambda o: (abs(o - value), o))
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = clamp(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_instance():
    """Neue Instanz: der nächste Zufalls-Seed aus einem SplitMix64-Strom, der mit der Zahl der bisherigen Klicks beginnt (kein numpy.random)."""
    klicks = st.session_state.get("zufalls_klicks", 0) + 1
    st.session_state["zufalls_klicks"] = klicks
    st.session_state["seed_input"] = SplitMix64(klicks * 7919 + int(st.session_state.get("seed_input", 0))).below(C.SEED_MAX + 1)
