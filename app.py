"""Mehr Rechnen beim Einsatz: Wann holt die Suche das Netz ein? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei von Hand in numpy gebaute Netze (Zeigernetz, Merkmalsnetz) bauen Rundtouren mit 20 Orten und dürfen beim Einsatz mehr rechnen: mehrere Touren bauen (Mehrfachstart, Sampling,
Strahlsuche), als Startlösung für eine Suche dienen oder auf der Instanz weitertrainiert werden (aktive Suche). Die Gegenseite ist eine einfache klassische Suche. Gezählt wird in
Kandidatenbewertungen (K) und groben Operationen (O) nebeneinander. Aufbau nach DEMO-PLAYBOOK §2: Kopf, Einleitung, Erklär-Expander, Primäransicht (eine Instanz), sechs Kernfragen
mit Verdikt, Vergleich aller Verfahren, Handbeispiel, Grenzen, Formeln.

Lauffähig mit: streamlit run app.py
"""
# ruff: noqa: E402  (die BLAS-Umgebungsvariablen müssen vor dem ersten Import von numpy stehen)

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # vor dem ersten Import von numpy: ein BLAS-Thread

import numpy as np
import streamlit as st

import modul_frisch
import rbe_band as BD
import rbe_befunde as BF
import rbe_budget as B
import rbe_constants as C
import rbe_evaluation as V
import rbe_exact as EXA
import rbe_experiments as EX
import rbe_hand as H
import rbe_layout as L
import rbe_live as LV
import rbe_presets as PR
import rbe_stats as ST
import rbe_texts as TX
import rbe_visualization as VIS

modul_frisch.aktualisieren(__file__)

st.set_page_config(page_title="Mehr Rechnen beim Einsatz – Sebastian Hanisch", layout="wide")


# ---------------------------------------------------------------------------------------------------------------- Zwischenspeicher
@st.cache_resource(show_spinner=False)
def _band():
    try:
        return BD.lade_band()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _e1(seeds, modell):
    return LV.e1(seeds, modell)


@st.cache_data(show_spinner=False)
def _e2(seeds, netz, modell, verfahren, k, symm):
    return LV.e2(seeds, netz, modell, verfahren, k, symm)


@st.cache_data(show_spinner=False)
def _e3_live(seeds, netz, modell, verfahren, k, symm, c):
    r = LV.e2(seeds, netz, modell, verfahren, k, symm)
    return LV.e3_live(seeds, netz, r["gap_netz"], r["K"], r["O"], c)


@st.cache_data(show_spinner=False)
def _e4(seeds, netz, modell):
    return LV.e4(seeds, netz, modell)


@st.cache_data(show_spinner=False)
def _optimum_tour(seed):
    X, D, _ = LV.daten([seed])
    return EXA.held_karp_tour(D[0])[1]


@st.cache_data(show_spinner=False)
def _abstieg(seed, start_tour):
    X, D, _ = LV.daten([seed])
    return LV.abstieg(D[0], np.array(start_tour))


@st.cache_data(show_spinner=False)
def _hypothesen(netz, modell, seed, breite):
    return LV.strahl_hypothesen(netz, modell, seed, breite)


@st.cache_data(show_spinner=False)
def _punkte(seed):
    return LV.daten([seed])[0][0]


# ---------------------------------------------------------------------------------------------------------------- Kopf, Einleitung, Erklärung
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title(TX.TITEL)
st.markdown(TX.INTRO)

with st.expander("💬 In einfachen Worten"):
    st.markdown(TX.EINFACHE_WORTE)

with st.expander("So funktioniert das Rechnen beim Einsatz", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

with st.expander("Wie hier gezählt wird (K und O)"):
    st.markdown(TX.ZAEHLREGEL)

with st.expander("Benannte Verfahren kurz erklärt"):
    st.markdown(TX.ERKLAERTABELLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider, einfach kopieren, um ein Szenario zu teilen.")

PR.load_permalink_settings()
PR.init_session_state_defaults()

# ---------------------------------------------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    PR.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Instanzen")
    n = st.select_slider("Orte n", options=PR.options("n_select"), key="n_select", help="n = 20: Rundtouren mit 20 Orten gleichverteilt im Einheitsquadrat, alle Experimente. n = 5 ist nur das Handbeispiel (feste Fahrzeitmatrix).")
    hand = n == C.N_HAND
    instanzen = st.select_slider("Instanzen live", options=PR.options("instanzen_select"), key="instanzen_select", disabled=hand,
                                 help="Wie viele Instanzen die Klicks rechnen. Die Rechengrenze (1 000 000 Kandidatenbewertungen je Klick) kann die Zahl kürzen; unter 4 Instanzen gibt es kein Urteil.")
    seed = st.number_input("Zufalls-Seed der Instanzen", *PR.bounds("seed_input"), key="seed_input", step=1, disabled=hand,
                           help="Zufalls-Seed 0 bis 11: die Instanzen 16 s bis 16 s + 15 der 200 Testinstanzen (für sie sind die exakten Längen vorgerechnet).")
    st.button("🎲 Neue Instanzen", width="stretch", on_click=PR.randomize_instance, disabled=hand)
    st.header("⚙️ Netz")
    netz = st.radio("Netz", options=PR.options("netz_select"), format_func=C.NETZ_ANZEIGE.get, key="netz_select", horizontal=True, disabled=hand,
                    help="Zeigernetz: Kodierer und Dekodierer, 12 480 Parameter. Merkmalsnetz: sieben Merkmale, 145 Parameter. Beide sind vorgerechnet (je 960 000 Touren bei n = 20).")
    modell = st.number_input("Gewichts-Zufalls-Seed des Modells", *PR.bounds("modell_input"), key="modell_input", step=1, disabled=hand,
                             help="Welches der fünf vorgerechneten Modelle (0 bis 4) die Klicks benutzen; die eingefrorenen Tabellen mitteln über alle fünf.")
    st.header("⚙️ Mehr Rechnen")
    verfahren = st.radio("Verfahren", options=PR.options("verfahren_select"), format_func=C.VERFAHREN_ANZEIGE.get, key="verfahren_select", disabled=hand,
                         help="Mehrfachstart: Greedy ab den ersten k Startstädten. Sampling: k gezogene Touren ab Ort 0. Strahlsuche: Breite k.")
    k = st.select_slider("k", options=PR.options("k_select"), key="k_select", disabled=hand,
                         help="Zahl der Startstädte, der gezogenen Touren oder die Breite der Strahlsuche. Beim Mehrfachstart gibt es höchstens 20 Startstädte.")
    symm_aktiv = netz == "zeiger" and verfahren == "mehrfach" and not hand
    symm = st.radio("Symmetrien (nur Zeigernetz, Mehrfachstart)", options=PR.options("symm_select"), key="symm_select", horizontal=True, disabled=not symm_aktiv,
                    help="8: dieselbe Rechnung für die acht Abbildungen des Einheitsquadrats (Drehungen, Spiegelungen); der Kodierer läuft je Abbildung. Das Merkmalsnetz ist dagegen invariant.")
    achse = st.radio("Achse in Frage 3", options=PR.options("achse_select"), format_func=C.ACHSE_ANZEIGE.get, key="achse_select", disabled=hand, help="Welche Einheit Frage 3 als Diagramm zeigt; das Verdikt nennt immer beide.")
    start = st.selectbox("Startlösung in Frage 4", options=PR.options("start_select"), format_func=C.START_ANZEIGE.get, key="start_select", disabled=hand,
                         help="Welche Startlösung das Diagramm in Frage 4 Zug für Zug verbessert; die Tabelle zeigt alle.")
    c = st.select_slider("Operationen je Delta", options=PR.options("c_select"), key="c_select", disabled=hand,
                         help="Annahme für O: grobe Operationen je Delta der Suche (Standard 8). Wirkt auf die Umrechnung des Budgets in Frage 3 und auf Frage 6.")
    st.header("⚙️ Aktive Suche")
    schritte = st.select_slider("Schritte", options=PR.options("schritte_select"), key="schritte_select", disabled=hand, help="Schritte der aktiven Suche in Frage 5.")
    touren = st.select_slider("Touren je Schritt", options=PR.options("touren_select"), key="touren_select", disabled=hand, help="Gezogene Touren je Schritt (auch das Sampling zum Vergleich zieht so viele).")
    lr = st.select_slider("Lernrate", options=PR.options("lr_select"), format_func=lambda v: f"{v:g}".replace(".", ","), key="lr_select", disabled=hand,
                          help="Adam-Lernrate der aktiven Suche. Eingefroren sind 0,001 und 0,01 (16 Touren je Schritt).")
    aktiv_inst = st.select_slider("Instanzen der aktiven Suche", options=PR.options("aktiv_inst_select"), key="aktiv_inst_select", disabled=hand,
                                  help="Auf wie vielen Instanzen die aktive Suche live läuft; höchstens 8 192 gezogene Touren je Klick.")

n, instanzen, seed, modell, k, symm, c, schritte, touren, aktiv_inst = (int(v) for v in (n, instanzen, seed, modell, k, symm, c, schritte, touren, aktiv_inst))
PR.sync_query_params({"n_select": n, "instanzen_select": instanzen, "seed_input": seed, "netz_select": netz, "modell_input": modell, "verfahren_select": verfahren, "k_select": k,
                      "symm_select": symm, "achse_select": achse, "start_select": start, "schritte_select": schritte, "touren_select": touren, "lr_select": lr,
                      "aktiv_inst_select": aktiv_inst, "c_select": c})

band = _band()
name_netz = C.NETZ_ANZEIGE[netz]


def _tabelle(kopf: list, zeilen: list) -> str:
    return "| " + " | ".join(kopf) + " |\n|" + "---|" * len(kopf) + "\n" + "\n".join("| " + " | ".join(str(z) for z in zeile) + " |" for zeile in zeilen)


def _mit_ci(gap: np.ndarray) -> str:
    ci = ST.bootstrap_ci(gap)
    return f"{V.de(float(np.mean(gap)))} % [{V.de(ci[0])}; {V.de(ci[1])}]" if ci else f"{V.de(float(np.mean(gap)))} %"


# ---------------------------------------------------------------------------------------------------------------- Handbeispiel
def handbeispiel():
    st.markdown("**Fünf Städte, ganzzahlige symmetrische Fahrzeitmatrix.** Alles unten ist von Hand nachzurechnen (Brüche, Aufzählung aller 24 Folgen ab Stadt 0).")
    st.markdown(_tabelle(["", "0", "1", "2", "3", "4"], [[f"**{i}**"] + list(H.D_HAND[i]) for i in range(5)]))
    nn = H.naechster_nachbar(0)
    st.markdown(f"**Wahrheit:** Held-Karp und Aufzählung stimmen überein, das Optimum ist **{H.optimum()}** mit 0-2-3-4-1 (2 + 4 + 1 + 4 + 5), eindeutig bis auf die Umkehrung. **Nächster Nachbar** ab 0: "
                f"{'-'.join(map(str, nn))} mit Länge {H.laenge(nn)} ({V.de(100 * (H.laenge(nn) / H.optimum() - 1), 1)} % über dem Optimum), aus jeder Startstadt {min(H.laenge(H.naechster_nachbar(s)) for s in range(5))} oder mehr.")
    st.markdown("**Mini-Netz** mit zwei Parametern (ein Merkmalsnetz ohne versteckte Schicht, Merkmale: Entfernung zur aktuellen Stadt und zur Startstadt): die Wahrscheinlichkeit, von i nach j zu gehen, ist "
                "proportional zu (2/3) hoch (2 D[i][j] + D[j][s]).")
    p1 = H.schritt_wsk(0, {1, 2, 3, 4}, 0)
    st.markdown(_tabelle(["Erster Schritt ab 0 nach", "Exponent 2 D + D[j][0]", "Wahrscheinlichkeit"], [[j, H.exponent(0, j, 0), V.de(float(p1[j]), 4)] for j in (2, 3, 1, 4)]))
    g0 = H.mini_greedy(0)
    st.markdown(f"**Greedy** ab 0: {'-'.join(map(str, g0))} mit Länge {H.laenge(g0)} (das Optimum); aus den fünf Startstädten die Längen {', '.join(str(H.laenge(H.mini_greedy(s))) for s in range(5))}. "
                f"**Mehrfachstart** bei gleicher Zahl Konstruktionen: Mini-Netz {min(H.laenge(H.mini_greedy(s)) for s in range(5))}, Nächster Nachbar {min(H.laenge(H.naechster_nachbar(s)) for s in range(5))}.")
    z = B.Zaehler()
    H.konstruktion_gezaehlt(0, z)
    st.markdown(f"**Zählung K:** eine Konstruktion 4 + 3 + 2 + 1 = **{z.k}**; Mehrfachstart aus allen 5 Startstädten **{B.k_mehrfach(5, 5)}**; Strahlsuche Breite 2: Schritte mit 1, 2, 2, 2 lebenden "
                f"Hypothesen und 4, 3, 2, 1 offenen Kandidaten, also 4 + 6 + 4 + 2 = **{B.k_strahl(5, 2)}**; Breite 24 (alle Folgen) **{B.k_strahl(5, 24)}**; Breite 1 ist Greedy ({B.k_strahl(5, 1)}). "
                f"Bei n = 20: Konstruktion {V.zahl(B.k_konstruktion(20))}, 20 Startstädte {V.zahl(B.k_mehrfach(20, 20))}, Strahlsuche Breite 8 {V.zahl(B.k_strahl(20, 8))} (nicht 8 × {V.zahl(B.k_konstruktion(20))}).")
    E = [H.erwartetes_beste(kk) for kk in (1, 2, 4, 8)]
    st.markdown(f"**Sampling exakt** (Brüche über alle 24 Folgen): erwartete Länge einer Stichprobe {V.de(float(E[0]), 4)}; Wahrscheinlichkeit, in einer Stichprobe ein Optimum zu treffen, {V.de(float(H.p_optimum()), 4)} "
                f"(die Greedy-Tour allein {V.de(float(H.wsk()[(0, 2, 3, 4, 1)]), 4)}); erwartetes Beste aus k = 2, 4, 8 Stichproben {V.de(float(E[1]), 4)} / {V.de(float(E[2]), 4)} / {V.de(float(E[3]), 4)}.")
    h2, _ = H.strahl_exakt(2)
    st.markdown("**Strahlsuche** Breite 2: " + "; ".join(f"{'-'.join(map(str, t))} (Wahrscheinlichkeit {V.de(float(p), 4)}, Länge {H.laenge(t)})" for t, p in h2) + ".")
    g1, g2 = H.gradient_exakt()
    j0 = float(H.erwartete_laenge())
    st.markdown(f"**Aktive Suche exakt:** Gradient der erwarteten Länge nach den zwei Parametern als Score-Funktion über alle 24 Folgen **({V.de(float(g1), 4)}; {V.de(float(g2), 4)})**, gleich der zentralen "
                f"Differenz; der Referenzwert ändert ihn nicht. Ein Schritt der Weite 0,02 entgegen dem Gradienten senkt die erwartete Länge von {V.de(j0, 4)} auf "
                f"{V.de(H.j_float(H.W1 - 0.02 * float(g1), H.W2 - 0.02 * float(g2)), 4)}, ein zu großer (5,0) hebt sie auf {V.de(H.j_float(H.W1 - 5 * float(g1), H.W2 - 5 * float(g2)), 4)}.")
    r4, r_opt = H.verbesserung(nn), H.verbesserung([0, 2, 3, 4, 1])
    st.markdown(f"**Verbesserung** (2-opt + Or-opt, erste Verbesserung): von {'-'.join(map(str, nn))} ({H.laenge(nn)}) endet die Suche im Optimum {int(r4['laenge'])} nach {r4['ev']} Zugauswertungen und {r4['zuege']} Zug/Zügen; "
                f"die Prüfrunde auf einer lokal optimalen Tour bewertet M(5) = {r_opt['ev']} Züge. **Netz baut, Suche verbessert** in K: Nächster Nachbar + Verbesserung "
                f"{B.k_konstruktion(5)} + {r4['ev']} = {B.k_konstruktion(5) + r4['ev']}; Mini-Netz, beste aus 5 (schon optimal), {B.k_mehrfach(5, 5)} + {r_opt['ev']} = {B.k_mehrfach(5, 5) + r_opt['ev']}.")


# ---------------------------------------------------------------------------------------------------------------- Seite
if hand:
    st.info("n = 5 ist nur das Handbeispiel: alle Zahlen stehen unten im Expander **🧮 Handbeispiel** und sind von Hand nachzurechnen. Die Experimente brauchen n = 20; wähle ein anderes Szenario links.")
elif band is None:
    st.error("Die eingefrorene Messung `rbe_band.json` fehlt: die Experimente 3, 4 (Tabellen) und 6 sind nicht verfügbar.")

if not hand:
    seeds_all = LV.instanz_seeds(seed, instanzen)
    kk = B.k_konstruktion(C.DEFAULT_N)

    # ------------------------------------------------------------------------------------------------------------ Primäransicht: eine Instanz
    m_e2, hinweis = LV.plan_runs(instanzen, EX.kosten_netz(netz, verfahren, min(k, 20) if verfahren == "mehrfach" else k, symm=symm if symm_aktiv else 1)[0])
    seeds_e2 = seeds_all[:m_e2]
    if hinweis:
        st.caption(hinweis)
    e1 = _e1(seeds_all, modell)
    e2 = _e2(seeds_e2, netz, modell, verfahren, k, symm if symm_aktiv else 1)
    st.markdown("## 🗺️ Eine Instanz")
    wahl = st.selectbox("Instanz", options=list(seeds_e2), format_func=lambda s: f"Test-Zufalls-Seed {s}", key=f"anzeige_instanz_{seed}_{len(seeds_e2)}", help="Welche der live gerechneten Instanzen die Karte zeigt.")
    i_w = list(seeds_e2).index(wahl)
    punkte = _punkte(wahl)
    opt = _optimum_tour(wahl)
    ref_w = float(e2["ref"][i_w])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(VIS.karte(punkte, [("Optimum", opt, VIS.FARBE_OPT, True), (f"{name_netz}: {V.zahl(e2['K'])} K", e2["tour_netz"][i_w], VIS.FARBE_NETZ[netz], False),
                                           (f"Nächster Nachbar: {V.zahl(e2['K_nn'])} K", e2["tour_nn"][i_w], VIS.FARBE_NN, False)],
                                  titel=f"Test-Zufalls-Seed {wahl}"), width="stretch", key=f"karte_{wahl}_{netz}_{verfahren}_{k}")
    with c2:
        st.markdown(f"**{name_netz}, {C.VERFAHREN_ANZEIGE[verfahren]}, k = {e2['k_eff']}{' (8 Abbildungen)' if e2['symm'] == 8 else ''}, Gewichts-Zufalls-Seed {modell}**")
        st.metric("Lücke des Netz-Verfahrens (diese Instanz)", f"{V.de(float(e2['gap_netz'][i_w]))} %", delta=f"{V.de_vz(float(e2['gap_netz'][i_w] - e2['gap_nn'][i_w]))} Prozentpunkte gegen Nächster Nachbar", delta_color="inverse")
        st.metric("Lücke Nächster Nachbar aus gleich vielen Startstädten", f"{V.de(float(e2['gap_nn'][i_w]))} %", delta="Referenz", delta_color="off")
        st.caption(f"Optimum {V.de(ref_w, 3)} (Einheitsquadrat); Kosten des Netz-Verfahrens {V.zahl(e2['K'])} Kandidatenbewertungen (K), {V.zahl(e2['O'])} grobe Operationen (O).")

    # ------------------------------------------------------------------------------------------------------------ Frage 1
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[0]}")
    st.caption(f"Regler: Netz, Gewichts-Zufalls-Seed, Instanzen, Zufalls-Seed. Eine Tour jedes Verfahrens auf {len(seeds_all)} Instanzen; Lücke in % über dem Optimum (Held-Karp), Mittel mit 95-%-Bootstrap-Intervall über die Instanzen.")
    zeilen1 = []
    for z in e1:
        wk = f"{V.de(B.wechselkurs(z['O'], z['K']), 1)} Deltas" if z["art"] in C.NETZE else "-"
        zeilen1.append([z["name"], _mit_ci(z["gap"]), V.zahl(z["K"]), V.zahl(z["O"]), wk])
    st.markdown(_tabelle(["Verfahren (eine Tour)", "Lücke Mittel [95-%-Intervall]", "K", "O", "Wechselkurs: eine Bewertung kostet"], zeilen1))
    st.caption(f"K: eine Konstruktion kostet n (n − 1) / 2 = {V.zahl(kk)} Kandidatenbewertungen, eine Zufallstour 1. O: Zeigernetz {V.zahl(B.o_zeiger(20, 1))} (Kodierer {V.zahl(B.o_kodierer(20))} einmal je Instanz, Dekodierer "
               f"{V.zahl(B.o_dekodierer(20))} je Tour), Merkmalsnetz {V.zahl(B.o_merkmal(20, 1))}, Nächster Nachbar {V.zahl(B.o_nn(20))} (8 je Kandidat). Der Wechselkurs ist das Maß, um das K das Netz begünstigt.")
    pk = [(z["name"], z["K"], float(np.mean(z["gap"])), {"zufall": VIS.FARBE_ZUFALL, "nn": VIS.FARBE_NN, "fi": VIS.FARBE_FI, "zeiger": VIS.FARBE_ZEIGER, "merkmal": VIS.FARBE_MERKMAL}[z["art"]],
           "circle" if z["art"] in C.NETZE else "diamond") for z in e1 if z["art"] != "zufall"]
    po = [(z["name"], z["O"], float(np.mean(z["gap"])), {"zufall": VIS.FARBE_ZUFALL, "nn": VIS.FARBE_NN, "fi": VIS.FARBE_FI, "zeiger": VIS.FARBE_ZEIGER, "merkmal": VIS.FARBE_MERKMAL}[z["art"]],
           "circle" if z["art"] in C.NETZE else "diamond") for z in e1 if z["art"] != "zufall"]
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(VIS.punkte_luecke(pk, "K", "Lücke gegen K (eine Tour)"), width="stretch", key="e1_k")
    with d2:
        st.plotly_chart(VIS.punkte_luecke(po, "O", "Lücke gegen O (eine Tour)"), width="stretch", key="e1_o")
    z_netz = next(z for z in e1 if z["art"] == netz)
    z_nn = next(z for z in e1 if z["art"] == "nn")
    cmp1 = ST.compare(z_netz["gap"], z_nn["gap"])
    art, text = V.verdict_eine_tour(cmp1["v"], cmp1["lo"], cmp1["hi"], cmp1["m"], netz, z_netz["K"], z_netz["O"], z_nn["O"])
    getattr(st, art)(text)

    # ------------------------------------------------------------------------------------------------------------ Frage 2
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[1]}")
    st.caption("Regler: Netz, Gewichts-Zufalls-Seed, Verfahren, k, Symmetrien, Instanzen. Verglichen wird bei gleicher Zahl Konstruktionen: Nächster Nachbar aus k Startstädten (es gibt nur 20 verschiedene Touren).")
    zeilen2 = [[f"{name_netz}, {C.VERFAHREN_ANZEIGE[verfahren]}, k = {e2['k_eff']}", _mit_ci(e2["gap_netz"]), V.zahl(e2["K"]), V.zahl(e2["O"])],
               [f"Nächster Nachbar, {e2['k_nn']} {'Startstadt' if e2['k_nn'] == 1 else 'Startstädte'}", _mit_ci(e2["gap_nn"]), V.zahl(e2["K_nn"]), V.zahl(e2["O_nn"])]]
    if e2["gap_zufall"] is not None:
        zeilen2.append([f"Zufallstouren, beste aus {V.zahl(e2['K'])}", _mit_ci(e2["gap_zufall"]), V.zahl(e2["K"]), V.zahl(B.o_zufall(e2["K"]))])
    st.markdown(_tabelle(["Verfahren", "Lücke Mittel [95-%-Intervall]", "K", "O"], zeilen2))
    cmp2 = ST.compare(e2["gap_netz"], e2["gap_nn"])
    art, text = V.verdict_gleiche_konstruktionen(cmp2["v"], cmp2["lo"], cmp2["hi"], cmp2["m"], netz, C.VERFAHREN_ANZEIGE[verfahren].split(" (")[0], e2["k_eff"] * e2["symm"], e2["K"],
                                                 float(np.mean(e2["gap_netz"])), float(np.mean(e2["gap_nn"])), e2["O"], e2["O_nn"], k_nn=e2["k_nn"], k_nn_gesamt=e2["K_nn"])
    getattr(st, art)(text)
    if band is not None and str(e2["k_eff"]) in band["netz"][netz][verfahren] and e2["symm"] == 1:
        zb = BD.netz_zeile(netz, verfahren, e2["k_eff"], band)
        nn_k = str(e2["k_nn"])
        nb = BD.klass_zeile("nn_mehrfach", e2["k_nn"], band) if nn_k in band["klass"]["nn_mehrfach"] else None
        st.caption(f"Eingefroren ({band['kopf']['instanzen']} Instanzen, Mittel über {band['kopf']['modell_seeds']} Gewichts-Zufalls-Seeds): {name_netz} {V.de(BD.mittel(zb['gap']))} %"
                   + (f" gegen Nächster Nachbar {V.de(BD.mittel(nb['gap']))} %." if nb else "."))
    if verfahren == "strahl" and k > 1:
        hyp = _hypothesen(netz, modell, wahl, k)
        t_sel = st.slider("Konstruktionsschritt t (welche Hypothesen leben)", 1, len(hyp), len(hyp), key="strahl_schritt", help="Die lebenden Hypothesen der Strahlsuche nach t Schritten auf der oben gewählten Instanz.")
        tours_t, score_t = hyp[t_sel - 1]
        st.markdown(f"Nach Schritt {t_sel}: {len(tours_t)} lebende Hypothesen (von höchstens {k}); die besten fünf:")
        st.markdown(_tabelle(["Rang", "Teiltour", "Log-Wahrscheinlichkeit"], [[r + 1, "-".join(map(str, tours_t[r])), V.de(float(score_t[r]), 3)] for r in range(min(5, len(tours_t)))]))
    if symm_aktiv and symm == 1:
        st.caption("Symmetrien ×8 sind nur beim Zeigernetz im Mehrfachstart wählbar; die Messung steht im Expander „Alle Verfahren im Vergleich“.")

    # ------------------------------------------------------------------------------------------------------------ Frage 3
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[2]}")
    st.caption("Regler: Netz, Verfahren, Achse, Operationen je Delta. Eingefroren (300 Instanzen, Mittel über die Gewichts-Zufalls-Seeds); die Neustart-Suche (2-opt + Or-opt, nicht abgestimmt) bekommt dasselbe Budget in K und in O.")
    if band is None:
        st.info("Ohne die eingefrorene Messung gibt es hier nichts zu zeigen.")
    else:
        zk, zo = BF.kreuzung_zeilen(band, netz, verfahren, c)
        xs_k, xs_o, ys_k = [], [], []
        for kk_ in BD.netz_ks(netz, verfahren, band):
            z = BD.netz_zeile(netz, verfahren, kk_, band)
            xs_k.append(z["K"])
            xs_o.append(z["O"])
            ys_k.append(BD.mittel(z["gap"]))
        ys_o = ys_k
        rx, ry = BD.neustart_mittelkurve(band)
        nnv, fiv = band["klass"]["nn_v"], band["klass"]["fi_v"]
        kurven_k = [(f"{name_netz}, {C.VERFAHREN_ANZEIGE[verfahren].split(' (')[0]}", xs_k, ys_k, VIS.FARBE_NETZ[netz], True, "circle"), ("Neustart-Suche", rx, ry, VIS.FARBE_SUCHE, True, "circle-open"),
                    ("Nächster Nachbar + Verbesserung", [float(np.mean(nnv["K"]))], [BD.mittel(np.array(nnv["gap"]))], VIS.FARBE_NN, False, "diamond"),
                    ("Farthest Insertion + Verbesserung", [float(np.mean(fiv["K"]))], [BD.mittel(np.array(fiv["gap"]))], VIS.FARBE_FI, False, "diamond")]
        kurven_o = [(f"{name_netz}, {C.VERFAHREN_ANZEIGE[verfahren].split(' (')[0]}", xs_o, ys_o, VIS.FARBE_NETZ[netz], True, "circle"), ("Neustart-Suche", rx * c, ry, VIS.FARBE_SUCHE, True, "circle-open"),
                    ("Nächster Nachbar + Verbesserung", [float(np.mean(nnv["O"]))], [BD.mittel(np.array(nnv["gap"]))], VIS.FARBE_NN, False, "diamond"),
                    ("Farthest Insertion + Verbesserung", [float(np.mean(fiv["O"]))], [BD.mittel(np.array(fiv["gap"]))], VIS.FARBE_FI, False, "diamond")]
        if achse == "beide":
            e3a, e3b = st.columns(2)
            with e3a:
                st.plotly_chart(VIS.kurve_budget(kurven_k, "K", "Lücke gegen das Budget in K"), width="stretch", key="e3_k")
            with e3b:
                st.plotly_chart(VIS.kurve_budget(kurven_o, "O", f"Lücke gegen das Budget in O ({c} Operationen je Delta)"), width="stretch", key="e3_o")
        elif achse == "K":
            st.plotly_chart(VIS.kurve_budget(kurven_k, "K", "Lücke gegen das Budget in K"), width="stretch", key="e3_k")
        else:
            st.plotly_chart(VIS.kurve_budget(kurven_o, "O", f"Lücke gegen das Budget in O ({c} Operationen je Delta)"), width="stretch", key="e3_o")
        st.caption(f"Die senkrechte Achse endet bei 20 % (höhere Lücken, etwa die der Suche bei sehr kleinem Budget, sind abgeschnitten). Leserichtung: K begünstigt das Netz (eine Bewertung kostet {V.de(B.wechselkurs(B.o_zeiger(20, 1), kk), 0)} Deltas beim Zeigernetz mit einer Tour, {V.de(B.wechselkurs(B.o_zeiger(20, 1000) - B.o_kodierer(20), kk * 1000), 0)} bei vielen "
                   f"Touren je Kodierer, {V.de(B.wechselkurs(B.o_merkmal(20, 1), kk), 0)} beim Merkmalsnetz), O benachteiligt es. **Training als dritter Posten:** das Netz wurde einmal mit 960 000 Touren trainiert "
                   f"({V.zahl(960000 * kk)} Kandidatenbewertungen), nicht in K und O gerechnet.")
        art, text = V.verdict_kreuzung(zk, zo, C.VERFAHREN_ANZEIGE[verfahren].split(" (")[0], netz, band["kopf"]["instanzen"])
        getattr(st, art)(text)
        live3 = _e3_live(seeds_e2, netz, modell, verfahren, k, symm if symm_aktiv else 1, c)
        if live3 is None:
            st.caption("Live-Probe der Neustart-Suche: bei diesem Budget zu teuer (über 100 000 Kandidatenbewertungen); die eingefrorene Messung gilt.")
        else:
            st.caption(f"Live-Probe auf {live3['anzahl']} Instanzen (zu wenig für ein Urteil, nur zur Ansicht): {name_netz} {V.de(float(np.mean(live3['gap_netz'])))} %; Neustart-Suche mit demselben K "
                       f"({V.zahl(live3['budget_k'])}) {V.de(float(np.mean(live3['gap_suche_k'])))} %, mit demselben O ({V.zahl(live3['budget_o'])} Deltas bei {c} je Delta) {V.de(float(np.mean(live3['gap_suche_o'])))} %.")

    # ------------------------------------------------------------------------------------------------------------ Frage 4
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[3]}")
    st.caption("Regler: Netz, Gewichts-Zufalls-Seed, Instanzen, Startlösung. Das Netz baut die Startlösung, die Suche (2-opt + Or-opt, erste Verbesserung) verbessert sie bis zum Lokalminimum.")
    seeds4 = seeds_all[:8]
    e4 = _e4(seeds4, netz, modell)
    zeilen4 = []
    for sk in ("s4", "s5", "s6", "netz1", "netz2", "netz3"):
        s_ = e4["starts"][sk]
        name = LV.START_NAMEN[sk] if sk.startswith("s") else f"{name_netz}: {LV.START_NAMEN[sk]}"
        zeilen4.append([name, V.de(float(np.mean(s_["gap_start"]))) + " %", V.de(float(np.mean(s_["gap_end"]))) + " %", V.zahl(np.mean(s_["ev"])), V.zahl(s_["k_bau"] / kk), V.zahl(np.mean(s_["K"])),
                        V.mio(float(np.mean(s_["O"])), 3)])
    st.markdown(_tabelle(["Startlösung", "Lücke der Startlösung", "Lücke nach der Verbesserung", "Zugauswertungen (Mittel)", "Konstruktionen", "Gesamt-K", "Gesamt-O (Mio)"], zeilen4))
    ev4 = float(np.mean(e4["starts"]["s4"]["ev"]))
    M20 = B.K0.m_verschiedene_zuege(20)
    anteil = 100.0 * M20 / ev4
    st.markdown(f"**Beweis des Lokalminimums:** die Endtour ist ein Lokalminimum; eine volle Prüfrunde kostet **{V.zahl(M20)}** Zugauswertungen ({TX.ZUG_HINWEIS}), das sind {V.de(anteil, 0)} % der {V.zahl(ev4)} Zugauswertungen des Nächsten Nachbarn mit Verbesserung "
                f"({len(e4['ref'])} Instanzen, Durchlauf, 2-opt vor Or-opt); eine perfekte Startlösung spart in diesem Abstieg also höchstens etwa ein Drittel ({V.de(100 - anteil, 0)} %; Obergrenze, nicht erreichbar).")
    inst4 = len(e4["ref"])
    s2n, s5 = e4["starts"]["netz2"], e4["starts"]["s5"]
    cmp4 = ST.compare(s2n["gap_end"], s5["gap_end"])
    klass_k = {kx: float(np.mean(e4["starts"][kx]["K"])) for kx in ("s4", "s5", "s6")}
    klass_o = {kx: float(np.mean(e4["starts"][kx]["O"])) for kx in ("s4", "s5", "s6")}
    guenstig = min(klass_k, key=klass_k.get)
    art, text = V.verdict_startloesung(f"{name_netz}, beste aus 20 Touren", cmp4["v"], cmp4["lo"], cmp4["hi"], inst4, float(np.mean(s2n["gap_end"])), float(np.mean(s5["gap_end"])),
                                       100.0 * (1.0 - float(np.mean(s2n["ev"])) / ev4), float(np.mean(s2n["K"])), min(klass_k.values()), float(np.mean(s2n["O"])), min(klass_o.values()),
                                       LV.START_NAMEN[guenstig])
    getattr(st, art)(text)
    st.caption("Die Endlücke steht gegen den Nächsten Nachbarn mit gleich vielen Konstruktionen (20 Startstädte), die gesparten Zugauswertungen gegen die Verbesserung ab einem einzelnen Nächsten Nachbarn, die Gesamtkosten gegen die günstigste klassische Startlösung.")
    # die Verbesserung Zug für Zug auf einer Instanz
    inst_opts = list(seeds4[:inst4])
    wahl4 = st.selectbox("Instanz für die Verbesserung", options=inst_opts, format_func=lambda s: f"Test-Zufalls-Seed {s}", key=f"e4_instanz_{seed}_{len(seeds4)}", help="Auf welcher Instanz die gewählte Startlösung Zug für Zug verbessert wird.")
    sk = "netz1" if start == "netz1" else start
    tour_start = e4["starts"][sk]["tour_start"][inst_opts.index(wahl4)]
    ab = _abstieg(wahl4, tuple(int(v) for v in tour_start))
    nz = len(ab["deltas"])
    if nz > 0:
        schritt4 = st.slider("Verbesserungszug (0 = Startlösung)", 0, nz, min(1, nz), key=f"e4_zug_{wahl4}_{start}_{netz}_{modell}", help="Die Tour nach dem k-ten angenommenen Zug.")
    else:
        schritt4 = 0
        st.caption("Diese Startlösung ist auf dieser Instanz schon ein Lokalminimum: die Suche nimmt keinen Zug an und beweist nur noch das Lokalminimum.")
    pk4 = _punkte(wahl4)
    ca, cb = st.columns([1, 1])
    with ca:
        st.plotly_chart(VIS.karte(pk4, [(f"nach Zug {schritt4}", ab["touren"][schritt4], VIS.FARBE_NETZ[netz] if sk.startswith("netz") else VIS.FARBE_NN, False)], titel=f"{C.START_ANZEIGE[start]}: nach {schritt4} von {nz} Zügen"),
                        width="stretch", key=f"e4_karte_{wahl4}_{start}_{netz}_{modell}_{schritt4}")
    with cb:
        if schritt4 == 0:
            st.markdown("Startlösung. Der erste verbessernde Zug in der Reihenfolge der Zugliste wird angenommen.")
        else:
            st.markdown(f"**Zug {schritt4}:** Delta {V.de_vz(ab['deltas'][schritt4 - 1], 4)}; seit dem vorigen Zug wurden {V.zahl(ab['stellen'][schritt4 - 1])} Zugauswertungen bewertet (der Durchlauf geht hinter dem letzten Zug weiter, die Zugliste beginnt nicht von vorn).")
        st.markdown(f"Insgesamt {nz} Züge und {V.zahl(ab['ev'])} Zugauswertungen; der letzte volle Durchlauf ohne Verbesserung (Beweis des Lokalminimums, Prüfrunde) bewertet {V.zahl(ab['ev_pruef'])}; Endlänge {V.de(ab['laenge'], 4)} (Einheitsquadrat), Lokalminimum bewiesen: {'ja' if ab['lokalmin'] else 'nein'}.")
    if band is not None:
        ni = band["kopf"]["instanzen"]
        mg = BF.mehrfachstart_gesamt(band["start"][netz]["S2"], band["folgen"]["nn"])
        if mg is not None:
            art, text = V.verdict_mehrfachstart_gesamt(netz, mg["K_netz"], mg["gap_netz"], mg["gap_klass"], mg["cmp"]["v"], mg["cmp"]["lo"], mg["cmp"]["hi"], mg["instanzen"], mg["o_faktor"], 20, mg["seeds_klar"], mg["ausgelassen"])
            getattr(st, art)(text)
            st.caption(f"Eingefroren: {ni} Instanzen, Mittel über {band['kopf']['modell_seeds']} Gewichts-Zufalls-Seeds; der klassische Mehrfachstart (Nächster Nachbar ab Startstadt 0, 1, 2, ... mit Verbesserung je Start) "
                       f"ist zwischen zwei gemessenen Startzahlen linear interpoliert, bei {mg['ausgelassen']} Instanzen lag das Gesamt-K des Netzes unter dem K des ersten klassischen Starts (aus dem Vergleich ausgelassen).")
        if "n50" in band:
            with st.expander("Ausblick n = 50 (eingefroren, nur Merkmalsnetz, 10 Instanzen)"):
                n50 = band["n50"]
                mg50 = BF.mehrfachstart_gesamt(n50["merkmal"]["S2"], n50["folge"], n=50)
                if mg50 is not None:
                    art, text = V.verdict_mehrfachstart_gesamt("merkmal", mg50["K_netz"], mg50["gap_netz"], mg50["gap_klass"], mg50["cmp"]["v"], mg50["cmp"]["lo"], mg50["cmp"]["hi"], mg50["instanzen"], mg50["o_faktor"], 50, mg50["seeds_klar"])
                    getattr(st, art)(text)
                st.caption(f"Referenz: CP-SAT, bewiesen optimal bei {sum(n50['bewiesen'])} von {len(n50['bewiesen'])} Instanzen, sonst beste gefundene Länge (die Lücken sind dann Untergrenzen). Das Zeigernetz wurde bei n = 50 nicht neu gemessen.")


    # ------------------------------------------------------------------------------------------------------------ Frage 5
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[4]}")
    st.caption("Regler: Gewichts-Zufalls-Seed, Schritte, Touren je Schritt, Lernrate, Instanzen der aktiven Suche. Nur das Zeigernetz: alle seine Gewichte werden je Schritt auf der Instanz verändert (Bello u. a. 2016). Vergleich: Sampling mit gleich vielen Touren aus dem Anfangsmodell.")
    wunsch_touren = schritte * touren
    m5 = max(1, min(aktiv_inst, C.RECHENGRENZE_TOUREN // wunsch_touren))
    seeds5 = seeds_all[:m5]
    st.markdown(f"**Zählung vor dem Lauf:** {V.zahl(schritte)} Schritte × {V.zahl(touren)} Touren × {m5} Instanzen = **{V.zahl(wunsch_touren * m5)} Touren** "
                f"({V.zahl(B.k_aktiv(20, schritte, touren) * m5)} Kandidatenbewertungen K, nur Vorwärtsrechnung; {V.mio(B.o_aktiv(20, schritte, touren) * m5)} Mio O). Alle 12 480 Gewichte werden je Schritt verändert."
                + (f" Instanzen von {aktiv_inst} auf {m5} gekürzt (höchstens {V.zahl(C.RECHENGRENZE_TOUREN)} Touren je Klick)." if m5 < aktiv_inst else ""))
    schluessel5 = (seeds5, modell, schritte, touren, lr)
    if st.button(f"🏋️ Aktive Suche starten ({V.zahl(wunsch_touren * m5)} Touren)", key="aktiv_knopf"):
        balken = st.progress(0.0, text="Instanz 0")
        with st.spinner(f"Aktive Suche: {V.zahl(schritte)} Schritte × {V.zahl(touren)} Touren × {m5} Instanzen = {V.zahl(wunsch_touren * m5)} Touren"):
            st.session_state["aktiv5"] = {"key": schluessel5, "res": LV.e5(seeds5, modell, schritte, touren, lr, fortschritt=lambda a, b: balken.progress(a / b, text=f"Instanz {a} von {b}"))}
        st.rerun()
    r5 = st.session_state.get("aktiv5")
    if r5 is not None and r5["key"] == schluessel5:
        r = r5["res"]
        if r["abgebrochen"]:
            st.warning(f"Nach {r['instanzen']} von {r['gewuenscht']} Instanzen abgebrochen (Zeitwächter {int(C.ZEITWAECHTER_SEKUNDEN)} Sekunden); die gerechneten Instanzen gelten.")
        ga, gs = r["gap_aktiv"], r["gap_sampling"]
        xs = np.arange(1, ga.shape[1] + 1)
        st.plotly_chart(VIS.kurve_schritte([("Aktive Suche", xs, ga.mean(axis=0), VIS.FARBE_ZEIGER, False), ("Sampling gleicher Tourzahl", xs, gs.mean(axis=0), VIS.FARBE_ZUFALL, True)],
                                           f"Beste Lücke bisher (Mittel über {r['instanzen']} Instanzen)"), width="stretch", key="e5_kurve")
        cm5 = ST.compare(ga[:, -1], gs[:, -1])
        art, text = V.verdict_aktive_suche(cm5["v"], cm5["lo"], cm5["hi"], r["instanzen"], r["touren_je_instanz"], lr, r["O_je_instanz"] / r["O_sampling_je_instanz"])
        getattr(st, art)(text)
    else:
        st.info("Noch nicht gerechnet: starte die aktive Suche mit dem Knopf (Zählung und Spinner stehen oben).")
    if band is not None:
        akt = band["aktiv"]
        eingefroren = BF.aktive_suche(band, schritte, str(lr)) if touren == akt["touren_je_schritt"] else None
        if eingefroren is not None:
            cmb = eingefroren["cmp"]
            art, text = V.verdict_aktive_suche(cmb["v"], cmb["lo"], cmb["hi"], eingefroren["instanzen"], eingefroren["touren"], lr, eingefroren["o_faktor"])
            getattr(st, art)(f"Eingefroren ({akt['instanzen']} Instanzen, Mittel über {band['kopf']['modell_seeds']} Gewichts-Zufalls-Seeds): " + text)
        else:
            st.caption(f"Eingefroren sind {akt['touren_je_schritt']} Touren je Schritt, Lernrate 0,001 und 0,01 und Schritte {', '.join(map(str, akt['schritte']))}; die gewählte Kombination ist nicht eingefroren.")


    # ------------------------------------------------------------------------------------------------------------ Frage 6
    st.markdown("---")
    st.subheader(f"🔬 {TX.KERNFRAGEN[5]}")
    st.caption("Regler: Netz, Verfahren, k, Operationen je Delta. Das ist keine neue Messung, sondern die Umkehrung der Annahme (8 Operationen je Delta), gerechnet auf der eingefrorenen Messung.")
    if band is not None:
        ks6 = BD.netz_ks(netz, verfahren, band)
        k6 = min(ks6, key=lambda x: (abs(x - min(k, 20) if verfahren == "mehrfach" else abs(x - k)), x))
        z6 = BD.netz_zeile(netz, verfahren, k6, band)
        g6 = BD.mittel(z6["gap"])
        b6 = BD.b_stern(g6, band)
        c_st = None if b6 is None else B.c_stern(z6["O"], b6)
        st.markdown(f"**{name_netz}, {C.VERFAHREN_ANZEIGE[verfahren].split(' (')[0]}, k = {k6}** (nächstes gemessenes k): Lücke {V.de(g6)} %, {V.zahl(z6['K'])} K, {V.zahl(z6['O'])} O. "
                    + ("" if b6 is None else f"Die Neustart-Suche erreicht diese Lücke mit {V.zahl(b6)} Deltas; das Netz kostet so viele Operationen wie die Suche bei {V.zahl(c_st)} Operationen je Delta (Schwelle c*). "
                       f"Bei den angenommenen {c} Operationen je Delta hat die Suche für dasselbe O {V.zahl(z6['O'] / c)} Deltas."))
        cs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000], dtype=float)
        lk = np.array([np.nan if (v := BD.suche_luecke_bei(z6["O"] / cc, band)) is None else v for cc in cs])
        st.plotly_chart(VIS.kurve_wechselkurs(cs, lk, g6, float(c), "Lücke der Suche bei gleichem O, abhängig von den Operationen je Delta"), width="stretch", key="e6_kurve")
        art, text = V.verdict_wechselkurs(c_st, float(c))
        getattr(st, art)(text)

    # ------------------------------------------------------------------------------------------------------------ Vergleich (eingefroren)
    st.markdown("---")
    with st.expander("🔧 Alle Verfahren im Vergleich (eingefroren)"):
        if band is None:
            st.info("Die eingefrorene Messung fehlt.")
        else:
            kopf = band["kopf"]
            st.caption(f"{kopf['instanzen']} Instanzen n = 20 gleichverteilt (Test-Zufalls-Seed 0 bis {kopf['instanzen'] - 1}), Netze: Mittel über {kopf['modell_seeds']} Gewichts-Zufalls-Seeds; Lücke in % über dem Optimum. "
                       f"Gerechnet mit {kopf['python']}, numpy {kopf['numpy']}; Zeit ist kein Befund.")
            t1, t2, t3, t4 = st.tabs(["Netz-Verfahren", "Klassische Verfahren", "Netz baut, Suche verbessert", "Aktive Suche, Symmetrien"])
            with t1:
                zeilen = []
                for nz in C.NETZE:
                    for vf in C.VERFAHREN:
                        for kb in BD.netz_ks(nz, vf, band):
                            zz = BD.netz_zeile(nz, vf, kb, band)
                            zeilen.append({"Netz": C.NETZ_ANZEIGE[nz], "Verfahren": C.VERFAHREN_ANZEIGE[vf].split(" (")[0], "k": kb, "K": zz["K"], "O": zz["O"], "Lücke Mittel in %": BD.mittel(zz["gap"]),
                                           "Streuung der Seed-Mittel": float(np.std(zz["gap"].mean(axis=1))), "Seeds unter Nächster Nachbar (gleiche Konstruktionen)": ""})
                for z in zeilen:
                    if z["Verfahren"] == "Mehrfachstart" or z["Verfahren"] == "Sampling" or z["Verfahren"] == "Strahlsuche":
                        kn = str(min(z["k"], 20))
                        if kn in band["klass"]["nn_mehrfach"]:
                            nn_g = BD.je_instanz(BD.klass_zeile("nn_mehrfach", int(kn), band)["gap"])
                            zz = BD.netz_zeile("zeiger" if z["Netz"] == "Zeigernetz" else "merkmal", {"Mehrfachstart": "mehrfach", "Sampling": "sampling", "Strahlsuche": "strahl"}[z["Verfahren"]], z["k"], band)
                            z["Seeds unter Nächster Nachbar (gleiche Konstruktionen)"] = f"{sum(ST.unter_gegner(zz['gap'][s], nn_g) for s in range(zz['gap'].shape[0]))} von {zz['gap'].shape[0]}"
                st.dataframe(zeilen, hide_index=True, width="stretch", column_config={"K": st.column_config.NumberColumn(format="%d"), "O": st.column_config.NumberColumn(format="%d"),
                                                                                  "Lücke Mittel in %": st.column_config.NumberColumn(format="%.2f %%"), "Streuung der Seed-Mittel": st.column_config.NumberColumn(format="%.2f")})
            with t2:
                zeilen = [{"Verfahren": "Nächster Nachbar, k Startstädte", "k": int(kb), "K": zz["K"], "O": zz["O"], "Lücke Mittel in %": BD.mittel(zz["gap"])} for kb in sorted(band["klass"]["nn_mehrfach"], key=int)
                          for zz in [BD.klass_zeile("nn_mehrfach", int(kb), band)]]
                fz = BD.klass_zeile("fi", None, band)
                zeilen.append({"Verfahren": "Farthest Insertion", "k": 1, "K": fz["K"], "O": fz["O"], "Lücke Mittel in %": BD.mittel(fz["gap"])})
                zeilen += [{"Verfahren": "Zufallstouren, beste aus k", "k": int(kb), "K": zz["K"], "O": zz["O"], "Lücke Mittel in %": BD.mittel(zz["gap"])} for kb in sorted(band["klass"]["zufall"], key=int)
                           for zz in [BD.klass_zeile("zufall", int(kb), band)]]
                zeilen += [{"Verfahren": "Neustart-Suche", "k": int(rb), "K": int(rb), "O": float(rb) * B.O_JE_DELTA, "Lücke Mittel in %": float(np.mean(g))} for rb, g in zip(*[band["klass"]["neustart"]["raster"], band["klass"]["neustart"]["gap"]])
                           if int(rb) in (128, 512, 2048, 8192, 32768, 131072, 389120) or int(rb) <= 0]
                for nm, kx in (("Nächster Nachbar + Verbesserung", "nn_v"), ("Farthest Insertion + Verbesserung", "fi_v")):
                    zeilen.append({"Verfahren": nm, "k": 1, "K": float(np.mean(band["klass"][kx]["K"])), "O": float(np.mean(band["klass"][kx]["O"])), "Lücke Mittel in %": BD.mittel(np.array(band["klass"][kx]["gap"]))})
                st.dataframe(zeilen, hide_index=True, width="stretch", column_config={"K": st.column_config.NumberColumn(format="%d"), "O": st.column_config.NumberColumn(format="%d"),
                                                                                  "Lücke Mittel in %": st.column_config.NumberColumn(format="%.2f %%")})
            with t3:
                zeilen = []
                for nz in C.NETZE:
                    for sn, titel in (("S1", "Greedy ab Ort 0"), ("S2", "beste aus 20 Greedy-Touren"), ("S3", "Sampling, beste aus 8")):
                        z = band["start"][nz][sn]
                        zeilen.append({"Startlösung": f"{C.NETZ_ANZEIGE[nz]}: {titel}", "Lücke nach Verbesserung in %": BD.mittel(np.array(z["gap"])), "Zugauswertungen": float(np.mean(z["ev"])),
                                       "Gesamt-K": float(np.mean(z["K"])), "Gesamt-O (Mio)": float(np.mean(z["O"])) / 1e6})
                for sn, titel in (("S4", "Nächster Nachbar ab Ort 0"), ("S5", "Nächster Nachbar, beste aus 20"), ("S6", "Farthest Insertion")):
                    z = band["start"]["klass"][sn]
                    zeilen.append({"Startlösung": titel, "Lücke nach Verbesserung in %": BD.mittel(np.array(z["gap"])), "Zugauswertungen": float(np.mean(z["ev"])), "Gesamt-K": float(np.mean(z["K"])),
                                   "Gesamt-O (Mio)": float(np.mean(z["O"])) / 1e6})
                st.dataframe(zeilen, hide_index=True, width="stretch", column_config={"Lücke nach Verbesserung in %": st.column_config.NumberColumn(format="%.3f %%"), "Zugauswertungen": st.column_config.NumberColumn(format="%d"),
                                                                                  "Gesamt-K": st.column_config.NumberColumn(format="%d"), "Gesamt-O (Mio)": st.column_config.NumberColumn(format="%.3f")})
            with t4:
                akt = band["aktiv"]
                zeilen = []
                for si, s_ in enumerate(akt["schritte"]):
                    zeilen.append({"Schritte": s_, "Touren": s_ * akt["touren_je_schritt"], "Aktive Suche, Lernrate 0,001": BD.mittel(np.array(akt["aktiv"]["0.001"])[:, si, :]),
                                   "Aktive Suche, Lernrate 0,01": BD.mittel(np.array(akt["aktiv"]["0.01"])[:, si, :]), "Sampling gleicher Tourzahl": BD.mittel(np.array(akt["sampling"])[:, si, :])})
                st.markdown(f"**Aktive Suche** ({akt['instanzen']} Instanzen, Mittel über {band['kopf']['modell_seeds']} Gewichts-Zufalls-Seeds), Lücke in %:")
                st.dataframe(zeilen, hide_index=True, width="stretch", column_config={c_: st.column_config.NumberColumn(format="%.2f %%") for c_ in ("Aktive Suche, Lernrate 0,001", "Aktive Suche, Lernrate 0,01", "Sampling gleicher Tourzahl")})
                sy = band["symm"]
                st.markdown(f"**Symmetrien ×8** (Zeigernetz, Mehrfachstart aus 20 Startstädten, {sy['instanzen']} Instanzen): Lücke ohne {V.de(BD.mittel(np.array(sy['eins'])))} %, mit acht Abbildungen {V.de(BD.mittel(np.array(sy['acht'])))} %; "
                            f"{V.zahl(sy['K_eins'])} gegen {V.zahl(sy['K_acht'])} K, {V.zahl(sy['O_eins'])} gegen {V.zahl(sy['O_acht'])} O. Das Merkmalsnetz ist gegen alle acht Abbildungen invariant (Entfernungen): "
                            f"seine Lücke {V.de(BD.mittel(np.array(sy['merkmal'])))} % ändert sich nicht.")

st.markdown("---")
with st.expander("🧮 Handbeispiel", expanded=hand):
    handbeispiel()

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Grenzen und Formeln
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.GRENZEN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATHE)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html)."
)
