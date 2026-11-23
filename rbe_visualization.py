"""Plotly-Figuren: Karten der Touren, Punktdiagramme Lücke gegen K und O, Verlaufskurven gegen das Budget, Kurven der aktiven Suche, Wechselkurs.

Alle Achsen sind fest (`fixedrange`), damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt; Beschriftungen von Kanten sind Annotationen, keine Text-Spuren."""

import numpy as np
import plotly.graph_objects as go

FARBE_ZEIGER = "#2F6B65"
FARBE_MERKMAL = "#8E5EA2"
FARBE_NN = "#e45756"
FARBE_FI = "#b279a2"
FARBE_SUCHE = "#f58518"
FARBE_ZUFALL = "#8c8c8c"
FARBE_OPT = "#4c78a8"
FARBE_NETZ = {"zeiger": FARBE_ZEIGER, "merkmal": FARBE_MERKMAL}


def _fest(fig: go.Figure, hoehe: int = 380, titel: str | None = None) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    zeilen_legende = min(len(fig.data), 5) if len(fig.data) > 1 else 0          # die Legende steht unter der Achsenbeschriftung, je Spur eine Zeile (schmale Breite)
    fig.update_layout(height=hoehe + 20 * zeilen_legende, margin=dict(l=50, r=20, t=50 if titel else 20, b=70 + 20 * zeilen_legende), title=dict(text=titel, font=dict(size=14)) if titel else None,
                      legend=dict(orientation="h", yanchor="top", y=-0.2 - 0.02 * zeilen_legende, x=0), hovermode="closest")
    return fig


def karte(punkte: np.ndarray, touren: list, titel: str | None = None, hoehe: int = 420) -> go.Figure:
    """Orte und Touren. `touren`: Liste (Name, Tour, Farbe, gestrichelt)."""
    p = np.asarray(punkte, dtype=float)
    fig = go.Figure()
    for name, tour, farbe, strich in touren:
        t = list(tour) + [tour[0]]
        fig.add_trace(go.Scatter(x=p[t, 0], y=p[t, 1], mode="lines", line=dict(color=farbe, width=2.5, dash="dash" if strich else "solid"), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="markers", marker=dict(size=9, color="#333333", line=dict(color="white", width=1)), name="Orte",
                             customdata=np.arange(len(p)), hovertemplate="Ort %{customdata}<extra></extra>"))
    for i, (x, y) in enumerate(p):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, yshift=12, font=dict(size=10, color="#555555"))
    fig.update_xaxes(range=[-0.04, 1.04], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-0.04, 1.04], showgrid=False, zeroline=False, visible=False)
    return _fest(fig, hoehe, titel)


def punkte_luecke(zeilen: list, achse: str, titel: str | None = None) -> go.Figure:
    """Lücke gegen K oder O (logarithmische x-Achse). `zeilen`: Liste (Name, x, Lücke, Farbe, Symbol)."""
    fig = go.Figure()
    for name, x, gap, farbe, symbol in zeilen:
        fig.add_trace(go.Scatter(x=[x], y=[gap], mode="markers", marker=dict(size=13, color=farbe, symbol=symbol, line=dict(color="white", width=1)), name=name,
                                 hovertemplate=f"{name}<br>{achse} = %{{x:,.0f}}<br>Lücke %{{y:.2f}} %<extra></extra>"))
    fig.update_xaxes(type="log", title=dict(text=("Kandidatenbewertungen K" if achse == "K" else "Grobe Operationen O"), standoff=6), showgrid=True)
    fig.update_yaxes(title="Lücke zum Optimum in %", rangemode="tozero")
    return _fest(fig, 330, titel)


def kurve_budget(serien: list, einheit: str, titel: str | None = None, ymax: float = 20.0, senkrecht: float | None = None) -> go.Figure:
    """Lücke gegen das Budget. `serien`: Liste (Name, Budgets, Lücken, Farbe, Linie, Symbol); y-Achse fest von 0 bis ymax (Werte darüber sind abgeschnitten)."""
    fig = go.Figure()
    for name, xs, ys, farbe, linie, symbol in serien:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers" if linie else "markers", line=dict(color=farbe, width=2), marker=dict(size=8, color=farbe, symbol=symbol), name=name,
                                 hovertemplate=f"{name}<br>{einheit} = %{{x:,.0f}}<br>Lücke %{{y:.2f}} %<extra></extra>"))
    if senkrecht is not None:
        fig.add_vline(x=senkrecht, line=dict(color="#999999", dash="dot"))
    fig.update_xaxes(type="log", title=dict(text=("Kandidatenbewertungen K" if einheit == "K" else "Grobe Operationen O"), standoff=6), showgrid=True)
    fig.update_yaxes(title="Lücke zum Optimum in %", range=[0, ymax])
    return _fest(fig, 380, titel)


def kurve_schritte(serien: list, titel: str | None = None, ymax: float | None = None) -> go.Figure:
    """Beste Lücke gegen die Schritte (aktive Suche gegen Sampling). `serien`: Liste (Name, Schritte, Lücken, Farbe, Linie)."""
    fig = go.Figure()
    for name, xs, ys, farbe, strich in serien:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=farbe, width=2, dash="dash" if strich else "solid"), marker=dict(size=7), name=name,
                                 hovertemplate=f"{name}<br>Schritt %{{x}}<br>beste Lücke %{{y:.2f}} %<extra></extra>"))
    fig.update_xaxes(title="Schritt (Touren je Schritt wie angegeben)", type="log")
    fig.update_yaxes(title="Beste Lücke bisher in %", rangemode="tozero", range=None if ymax is None else [0, ymax])
    return _fest(fig, 340, titel)


def kurve_wechselkurs(cs: np.ndarray, luecken: np.ndarray, netz_luecke: float, c0: float, titel: str | None = None) -> go.Figure:
    """Lücke der Suche bei gleichem O (O des Netz-Verfahrens geteilt durch c) gegen c; waagerecht die Lücke des Netzes, senkrecht die Annahme c0."""
    fig = go.Figure()
    ok = np.isfinite(luecken)
    fig.add_trace(go.Scatter(x=cs[ok], y=luecken[ok], mode="lines", line=dict(color=FARBE_SUCHE, width=2), name="Suche mit gleichem O", hovertemplate="c = %{x:,.0f}<br>Lücke %{y:.2f} %<extra></extra>"))
    fig.add_hline(y=netz_luecke, line=dict(color=FARBE_ZEIGER, dash="dash"), annotation_text="Netz", annotation_position="top left")
    fig.add_vline(x=c0, line=dict(color="#999999", dash="dot"), annotation_text="Annahme", annotation_position="top")
    fig.update_xaxes(type="log", title="Operationen je Delta c")
    fig.update_yaxes(title="Lücke in %", rangemode="tozero")
    return _fest(fig, 330, titel)


def balken_kosten(namen: list, werte: list, farben: list, titel: str, achsentitel: str) -> go.Figure:
    """Waagerechte Balken mit logarithmischer Achse (Kosten verschiedener Verfahren)."""
    fig = go.Figure(go.Bar(y=namen, x=werte, orientation="h", marker=dict(color=farben), text=[f"{w:,.0f}".replace(",", " ") for w in werte], textposition="outside", cliponaxis=False))
    fig.update_xaxes(type="log", title=achsentitel)
    fig.update_yaxes(autorange="reversed")
    return _fest(fig, 60 + 38 * len(namen), titel)
