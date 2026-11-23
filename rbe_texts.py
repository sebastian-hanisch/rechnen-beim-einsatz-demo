"""Texte der App (Titel, Einleitung, Erklärungen, Grenzen, Formeln). Zahlen der Zählregel kommen aus `rbe_budget`, nicht aus dem Gedächtnis.

Regeln: echte Umlaute, „Zufalls-Seed“, kein Gedankenstrich im Fließtext, Zahl immer mit Zähleinheit, benannte Verfahren mit deutschem Wort zuerst und englischem in Klammern.
Befunde mit Zahlen stehen nicht hier, sondern werden in der App aus der eingefrorenen Messung gelesen und in der README mit Voraussetzungen genannt."""

import rbe_budget as B
import rbe_evaluation as V

TITEL = "🧮 Mehr Rechnen beim Einsatz: Wann holt die Suche das Netz ein?"

N = 20
KK = B.k_konstruktion(N)
M20 = B.K0.m_verschiedene_zuege(N)
NACHBARTOUREN_20 = 1570        # verschiedene Nachbartouren der M(20) = 1 850 Züge (Kantenmengen verglichen; `tests/test_texts.py` zählt sie nach)
ZUG_HINWEIS = f"Zugbeschreibungen, einige ergeben dieselbe Nachbartour; bei n = {N} sind es {V.zahl(NACHBARTOUREN_20)} verschiedene Nachbartouren"

INTRO = f"""
Ein **Netz**, das Rundtouren baut, wird einmal trainiert und kann beim Einsatz mehr rechnen: **mehrere Touren bauen** (Mehrfachstart, Sampling, Strahlsuche) oder sich **auf der Instanz
nachtrainieren** (aktive Suche). Diese Demo fragt, ob sich das gegen eine **einfache klassische Suche** lohnt, wenn beide **in derselben Einheit** gezählt werden: **Kandidatenbewertungen**
(K) und **grobe Operationen** (O), immer nebeneinander. Gerechnet wird auf erzeugten Rundtouren mit {N} Orten (gleichverteilt) mit zwei von Hand in numpy gebauten Netzen, dem **Zeigernetz**
und dem **Merkmalsnetz**; die Instanzen stammen aus einem festen Zufalls-Seed.

Weiter unten: sechs **Kernfragen** mit Verdikt, der Expander **🔧 Alle Verfahren im Vergleich (eingefroren)**, das **🧮 Handbeispiel**, **🚧 Wo die Annahmen enden** und die
**📐 Mathematische Formulierung**.
"""

EINFACHE_WORTE = """
Stell dir vor, du lässt ein Navigationsgerät die kürzeste Rundreise raten. Beim ersten Mal rät es einmal. Du kannst es aber auch mehrmals raten lassen und die beste Reise nehmen, oder du lässt
es auf dieser einen Aufgabe noch ein bisschen üben. Das kostet Rechenarbeit. Eine einfache Suche, die eine Reise Schritt für Schritt verbessert, kostet auch Rechenarbeit. Diese Demo stellt
beides gegenüber und zählt die Arbeit auf zwei Arten: nach der Zahl der Entscheidungen, die jemand prüfen muss, und nach den einzelnen Rechenschritten. Dann sieht man, bei wie viel Arbeit die
einfache Suche das Netz einholt und was das Netz als Vorarbeit für die Suche wert ist. Alles gilt nur für die hier gebauten Beispiele.
"""

SO_FUNKTIONIERT = f"""
1. **Das Netz baut eine Tour** Ort für Ort: in jedem Schritt bewertet es alle noch offenen Orte und wählt (Greedy) den besten oder zieht (Sampling) nach seinen Wahrscheinlichkeiten.
   Jede Bewertung eines offenen Ortes zählt als **eine Kandidatenbewertung**; eine ganze Tour kostet bei {N} Orten {V.zahl(KK)} (19 + 18 + ... + 1).
2. **Mehr Rechnen beim Einsatz** heißt: mehrere Touren bauen und die kürzeste nehmen (**Mehrfachstart** aus mehreren Startstädten, **Sampling** gezogener Touren), alle Hypothesen mit
   der größten Wahrscheinlichkeit weiterverfolgen (**Strahlsuche**) oder das Netz auf der einen Instanz weitertrainieren (**aktive Suche**).
3. **Die Gegenseite** ist eine einfache Suche: Zufallstour, dann Schritt für Schritt jeden Zug (2-opt, Or-opt) prüfen, den ersten verbessernden annehmen, im Lokalminimum neu starten
   (**Neustart-Suche**). Jedes bewertete Delta zählt wie eine Kandidatenbewertung.
4. **Verglichen wird bei gleichem Budget**, in K und in O. Das Budget in K ist die Zahl der Bewertungen; das Budget in O die Zahl grober Rechenoperationen.
"""

ZAEHLREGEL = f"""
**K: Kandidatenbewertungen.** Ein bewerteter Kandidat zählt 1: ein Delta der Suche, eine Merkmalszeile oder ein Zeigerwert des Netzes, ein Vergleich beim Nächsten Nachbarn. Eine
Konstruktion kostet nie 1, sondern n (n − 1) / 2 = {V.zahl(KK)} bei n = {N} (die Summe der offenen Orte über alle Schritte); eine Zufallstour zählt 1. Eine volle Prüfrunde der Suche
(Beweis des Lokalminimums) bewertet {V.zahl(M20)} Züge ({ZUG_HINWEIS}).

**O: grobe Operationen.** Multiplikationen und Additionen der Matrixprodukte des Netzes, aus den Formen einzeln gezählt; ein Delta der Suche und eine Nächster-Nachbar-Bewertung zählen
je {B.O_JE_DELTA} (Annahme, nur auf Zehnerpotenzen belastbar). Eine Tour des Zeigernetzes kostet {V.zahl(B.o_zeiger(N, 1))} (Kodierer {V.zahl(B.o_kodierer(N))} einmal je Instanz, Dekodierer
{V.zahl(B.o_dekodierer(N))} je Tour), eine des Merkmalsnetzes {V.zahl(B.o_merkmal(N, 1))}, eine Nächster-Nachbar-Tour {V.zahl(B.o_nn(N))}.

**Leserichtung.** K begünstigt das Netz, weil eine Netz-Bewertung viel mehr Rechenarbeit ist als ein Delta (der Wechselkurs steht in Frage 1). O benachteiligt es, weil die {B.O_JE_DELTA}
Operationen je Delta eine Schätzung sind. „Das Netz verliert“ gilt hier nur, wenn es schon in K verliert; „das Netz gewinnt“ nur, wenn es auch in O gewinnt.

**Training** ist ein dritter Posten und steht nicht in K und O: beide Netze wurden einmal mit 960 000 Touren trainiert (Zeigernetz 8 000 Schritte × 120 Touren, Merkmalsnetz 3 750 Schritte
× 256 Touren). Das Netz verteilt diese Kosten auf alle späteren Instanzen. **Zeit** (Sekunden) ist kein Befund dieser Demo.
"""

ERKLAERTABELLE = f"""
| Verfahren | Was es tut | Kosten bei n = 20 in K und O | Code |
|---|---|---|---|
| **Zeigernetz** (Pointer Network) | Ein Kodierer (Einbettung, Attention, Vorwärtsnetz) macht aus den Koordinaten Ortsvektoren; ein Dekodierer zeigt in jedem Schritt auf den nächsten Ort. 12 480 Parameter, eine Schicht. | Tour: {V.zahl(KK)} K, {V.zahl(B.o_zeiger(N, 1))} O | `rbe_net.py` |
| **Merkmalsnetz** (feature network) | Ein kleines Netz (7 Merkmale, 16 versteckte Einheiten, 145 Parameter) bewertet jeden offenen Ort aus Entfernungen und Rängen, alle durch den mittleren Nachbarabstand geteilt. Kein Kodierer. | Tour: {V.zahl(KK)} K, {V.zahl(B.o_merkmal(N, 1))} O | `rbek_featnet.py` |
| **Mehrfachstart** (Multi-Start, POMO-Inferenz) | Dasselbe Netz baut je eine Greedy-Tour ab jeder der k ersten Startstädte, die kürzeste zählt. | k Konstruktionen: k × {V.zahl(KK)} K | `rbe_construct.py` |
| **Sampling** (Ziehen von Touren) | Statt des besten Ortes wird nach den Wahrscheinlichkeiten des Netzes gezogen (Temperatur 1), k unabhängige Touren, die kürzeste zählt. | k × {V.zahl(KK)} K | `rbe_construct.py` |
| **Strahlsuche** (Beam Search) | Je Schritt wachsen alle lebenden Hypothesen um jeden offenen Ort, behalten werden die B mit der größten Wahrscheinlichkeit; am Ende zählt die kürzeste Hypothese. | B = 8: {V.zahl(B.k_strahl(N, 8))} K (nicht 8 × {V.zahl(KK)}, der erste Schritt hat nur eine Hypothese) | `rbe_construct.py` |
| **Aktive Suche** (Active Search) | Das Zeigernetz wird auf einer Instanz weitertrainiert: je Schritt k Touren ziehen, Referenzwert (Baseline) abziehen, alle Gewichte verändern. | je Schritt k × {V.zahl(KK)} K (nur Vorwärtsrechnung), {V.zahl(B.o_aktiv(N, 1, 16))} O bei k = 16 | `rbe_active.py` |
| **Nächster Nachbar** | Immer zum nächsten unbesuchten Ort. | {V.zahl(KK)} K, {V.zahl(B.o_nn(N))} O | `rbek_construct.py` |
| **Farthest Insertion** | Den Ort mit dem größten Abstand zur Tour dort einfügen, wo die Tour am wenigsten länger wird. | {V.zahl(KK)} K, {V.zahl(B.o_farthest(N))} O (Annahme 4 n²) | `rbek_construct.py` |
| **2-opt, Or-opt** | Zwei Kanten tauschen bzw. ein Teilstück aus 1 bis 3 Orten umsetzen; der erste verbessernde Zug wird angenommen. Die volle Zugmenge hat M(n) = n (11 n − 35) / 2 Züge (Zugbeschreibungen; einige ergeben dieselbe Nachbartour). | Prüfrunde: {V.zahl(M20)} K bei n = {N} | `rbek_moves.py`, `rbek_search.py` |
| **Neustart-Suche** | Zufallstour, Abstieg im Lokalminimum, neue Zufallstour, bis das Budget verbraucht ist. Nicht abgestimmt. | beliebig: 1 K je Zufallstour, 1 K je Delta | `rbe_suche.py` |
| **Held-Karp** | Exakte Rundtour per dynamischer Programmierung über Teilmengen; liefert die Referenz der Lücken. | n = {N}: rund 0,5 s, nicht im Budget | `rbe_exact.py` |
| **Lokalminimum** | Eine Tour, bei der kein Zug der Zugmenge mehr verbessert; bewiesen nur durch eine volle Prüfrunde. | M(n) Zugauswertungen | `rbek_search.py` |
| **Referenzwert** (Baseline) | Abzugsgröße im REINFORCE-Gradienten (hier das Mittel der gezogenen Touren); das Thema von `basislinie-lernen-demo`. | - | `rbe_train.py` |
"""

KERNFRAGEN = [
    "1. Wer baut mit einer Tour die beste Rundreise?",
    "2. Was bringt es, mehr Touren zu bauen?",
    "3. Und bei gleicher Gesamtarbeit gegen eine Suche?",
    "4. Hilft das Netz als Startlösung für die Suche?",
    "5. Lohnt es, das Netz auf der Instanz weiter zu trainieren?",
    "6. Wie teuer müsste ein Rechenschritt der Suche sein, damit sich die Antwort dreht?",
]

GRENZEN = f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eigene Umsetzungen, kleine Netze:** Zeigernetz mit 12 480, Merkmalsnetz mit 145 Parametern, je einmal bei n = {N} gleichverteilt trainiert. | Kein Urteil über Kool- oder POMO-Modelle, GPU-Zeiten oder größere Netze. | nicht gemessen |
| **Nur n = {N} gleichverteilt** (Ausblick n = 50 nur mit dem Merkmalsnetz und 10 Instanzen). | Für andere Familien (Gruppen, Gitter) und größere n gibt es hier keine Aussage; das Zeigernetz bei n = 50 wurde nicht neu gemessen. | `netz-fremde-instanzen-demo` (in dieser Reihe, noch ohne Live-Link) |
| **Einfachste lokale Suche:** 2-opt + Or-opt mit erster Verbesserung, ohne Nachbarlisten und Don't-look-Bits, Neustart-Suche nicht abgestimmt. | Mit Nachbarlisten wird die Prüfrunde billig, die Bilanz des Netzes in O wohl noch schlechter: nicht gemessen. Aussagen gelten nur für diese Suche, nicht für „die klassische Suche“. | `netz-gegen-gegner-demo` (in dieser Reihe, noch ohne Live-Link) |
| **O ist eine Schätzung:** {B.O_JE_DELTA} Operationen je Delta, 144 je Merkmalszeile. | Belastbar auf Zehnerpotenzen, nicht auf Prozent; der Regler „Operationen je Delta“ und Frage 6 zeigen, wie weit die Antwort an der Annahme hängt. | - |
| **K begünstigt das Netz, O benachteiligt es.** | „Das Netz verliert“ gilt hier nur, wenn es schon in K verliert; „das Netz gewinnt“ nur, wenn es auch in O gewinnt. | - |
| **Sampling nur mit Temperatur 1, Strahlsuche nicht abgestimmt.** | Andere Temperaturen und eine abgestimmte Strahlsuche sind nicht gemessen. | nicht gemessen |
| **Symmetrien (×8) und aktive Suche nur beim Zeigernetz.** Das Merkmalsnetz ist gegen die acht Abbildungen invariant (Entfernungen), eine aktive Suche gibt es dort nicht. | Der Regler „Symmetrien“ wirkt beim Merkmalsnetz nicht (die Spalte wäre lauter Nullen, darum steht hier ein Satz). | - |
| **Zeit ist kein Befund.** Die Streamlit-Cloud ist nicht gemessen. | Sekunden einer Maschine sagen nichts über das Verfahren. | - |
| **Ursachen sind nicht gemessen.** | Warum das Merkmalsnetz als Mehrfachstart ohne weiteres Training so gut ist, bleibt ungeklärt: hier keine Deutung. | offen |
| **Keine Aussage** über Lin-Kernighan, LKH, GPU-Netze, Praxis in der Logistik. | - | - |
"""

MATHE = r"""
**Gegeben:** n Orte mit Entfernungen $D_{ij}$ und ein trainiertes Netz. **Gesucht:** eine kurze Rundtour und die Kosten ihrer Berechnung in K und O.

**Tour und Maß.**
$$L(t)=\sum_{k=0}^{n-1} D_{t_k,\,t_{(k+1) \bmod n}}$$
$$g(t)=100\Bigl(\frac{L(t)}{L^\star}-1\Bigr)\ \%$$
$L^\star$ ist das Optimum (Held-Karp).

**Richtlinie (Politik) eines Netzes.** Die Tour entsteht in $n-1$ Schritten ab einer Startstadt $s$; $V_t$ sind die noch offenen Städte, $c$ die aktuelle.
$$\pi_\theta(j\mid c,V_t,s)=\frac{\exp(u_j)}{\sum_{k\in V_t}\exp(u_k)},\qquad j\in V_t$$
$$P_\theta(t)=\prod_{k=1}^{n-1}\pi_\theta(t_k\mid t_{k-1},V_k,s)$$
**Zeigernetz** (eine Schicht, ein Kopf, $d=32$, 12 480 Parameter): Einbettung $h^{(0)}_i=x_iW_0+b_0$; Attention mit Rückgriff, $Q=h^{(0)}W_q$, $K=h^{(0)}W_k$, $V=h^{(0)}W_v$:
$$h^{(1)}=h^{(0)}+\operatorname{softmax}\bigl(QK^\top/\sqrt d\bigr)V\,W_o,\qquad H=h^{(1)}+\operatorname{ReLU}(h^{(1)}W_1+b_1)W_2+b_2$$
Dekodierer: Kontext $q_c=[\bar H,\,H_s,\,H_c]\,W_c$, Zeigerwert
$$u_j=10\tanh\Bigl(\frac{q_c\cdot (H_jW_k)}{\sqrt d}\Bigr),\qquad u_j=-\infty\ \text{für besuchte Städte}$$
**Merkmalsnetz** (145 Parameter, kein Kodierer): sieben Merkmale $x_{cj}\in\mathbb R^7$, alle Entfernungen durch den mittleren Nachbarabstand $\bar s$ geteilt und bei 6 gekappt: Entfernung zur aktuellen
Stadt, Rang unter den offenen Nachbarn (bei 8 gekappt), Entfernung zur Startstadt, Entfernung zum nächsten anderen offenen Ort, Zahl offener Orte im Abstand $2\bar s$ (bei 8 gekappt), Anteil
besuchter Orte, relative Entfernung zum nächsten offenen Ort. Logit
$$u_j=w_2^\top\tanh(W_1x_{cj}+b_1)+b_2,\qquad W_1\in\mathbb R^{16\times 7}\ (7\cdot 16+16+16+1=145)$$
Das Netz ist gegen Verschieben, Drehen, Spiegeln und Skalieren der Koordinaten invariant (Entwurf, nicht gelernt).

**Verfahren des Einsatzes.** Greedy: $a_t=\arg\max_j\pi_\theta$. Sampling: $a_t\sim\pi_\theta$ (Temperatur 1), $k$ unabhängige Touren, die kürzeste zählt. Mehrfachstart: Greedy ab jeder Startstadt
$s\in S$, die kürzeste zählt. Strahlsuche der Breite $B$:
$$\mathcal H_{t+1}=\operatorname{top}_B\Bigl\{(h\!\cdot\! j,\ \ell_h+\log\pi_\theta(j\mid c_h,V_h,s))\ :\ h\in\mathcal H_t,\ j\in V_h\Bigr\},\qquad \mathcal H_1=\{(s,0)\}$$
Ergebnis ist die kürzeste der $B$ Endhypothesen (nicht die wahrscheinlichste). Aktive Suche (Bello u. a. 2016): je Schritt $k$ Touren $t^{(i)}\sim P_\theta$ auf **einer** Instanz,
Referenzwert $\bar L=\frac1k\sum_i L(t^{(i)})$:
$$\hat g=\frac1k\sum_{i=1}^{k}\bigl(L(t^{(i)})-\bar L\bigr)\,\nabla_\theta\log P_\theta(t^{(i)}),\qquad \theta\leftarrow\operatorname{Adam}(\theta,\hat g)$$
Ergebnis ist die kürzeste Tour aller bisherigen Schritte; dass sie besser ist als Sampling mit derselben Tourzahl, ist nicht zugesichert, sondern gemessen. Erwartungswert:
$\nabla_\theta\mathbb E[L]=\mathbb E\bigl[(L-b)\nabla_\theta\log P_\theta\bigr]$ für jeden festen Referenzwert $b$.

**Verbesserung.** 2-opt: $\Delta=D_{ac}+D_{bd}-D_{ab}-D_{cd}$ für die Kanten $(a,b)$, $(c,d)$; Or-opt: Segment der Länge 1 bis 3 an eine andere Stelle (Länge 1 nur in einer Richtung). Angenommen wird
der erste Zug mit $\Delta<0$; Ende im Lokalminimum. Eine volle Prüfrunde ohne Verbesserung bewertet
$$M(n)=\underbrace{\tfrac{n(n-3)}{2}}_{\text{2-opt}}+\underbrace{n(n-2)}_{\text{Or-opt }1}+\underbrace{2n(n-3)}_{\text{Or-opt }2}+\underbrace{2n(n-4)}_{\text{Or-opt }3}=\tfrac{n(11n-35)}{2}$$
Züge (1 850 bei $n=20$, 12 875 bei $n=50$). Das sind Zugbeschreibungen; einige ergeben dieselbe Nachbartour, bei $n=20$ sind es 1 570 verschiedene Nachbartouren.

**Zähleinheit K.** Ein bewerteter Kandidat zählt 1; Zufallstour 1.
$$K_{\text{Konstruktion}}=\sum_{t=1}^{n-1}(n-t)=\frac{n(n-1)}{2}$$
$$K_{\text{Strahl}}(n,B)=\sum_{t=1}^{n-1}h_t\,(n-t),\qquad h_1=1,\quad h_{t+1}=\min\bigl(B,\,h_t\,(n-t)\bigr)$$
Mehrfachstart mit $m$ Starts: $m\,n(n-1)/2$; Sampling mit $k$ Touren: $k\,n(n-1)/2$; aktive Suche (nur Vorwärtsrechnung, begünstigt das Netz zusätzlich): Schritte $\times\,k\times n(n-1)/2$.
Gesamt-K einer Startlösung mit Verbesserung: $m\,n(n-1)/2+Z$, $Z$ die Zugauswertungen.

**Zähleinheit O** (grobe Operationen; Zehnerpotenzen, nicht Prozent), Zeigernetz mit Kodierer und Dekodierer aus den einzelnen Matrixprodukten:
$$O_{\text{Zeiger}}=O_{\text{Kodierer}}(n,d)+m\,O_{\text{Dekodierer}}(n,d)$$
$$O_{\text{Strahl}}=O_{\text{Kodierer}}+2d^2+\Bigl(\sum_{t=1}^{n-1}h_t\Bigr)\bigl(d^2+nd+4n\bigr)$$
$$O_{\text{aktiv}}=3\,(\text{Schritte})\bigl(O_{\text{Kodierer}}+k\,O_{\text{Dekodierer}}\bigr)$$
Merkmalsnetz: $n^2+m\bigl[(n-1)(2n^2+n(8+\log_2 n))+\tfrac{n(n-1)}{2}\cdot 144\bigr]$. Nächster Nachbar 8 je Kandidat, Farthest Insertion $4n^2$, Suche $c_\Delta=8$ je Delta (Regler).

**Wechselkurs und Schwelle.**
$$w=\frac{O_{\text{Netz}}}{K_{\text{Netz}}\,c_\Delta},\qquad c^\star=\frac{O_{\text{Netz}}}{B^\star(g_{\text{Netz}})}$$
$w$ ist die Zahl Deltas, die eine Netz-Bewertung kostet; $B^\star(g)$ das kleinste Suchbudget (in Deltas), das die Lücke $g$ erreicht (Interpolation im Messbereich).

**Vergleichsmaß.** Gepaarter Unterschied $d=\frac1m\sum_i\bigl(g_i(A)-g_i(B)\bigr)$ über die Instanzen (je Instanz Mittel über die Gewichts-Zufalls-Seeds), 95-%-Bootstrap-Intervall über die Instanzen
(SplitMix64, 2 000 Ziehungen); A schlägt B bei $d\le -0{,}25$ Prozentpunkten und einem Intervall unter 0.

**Training (dritter Posten):** Touren $=\text{Schritte}\times\text{Batch}$, in K $\times\,n(n-1)/2$; nicht in K und O der Einsatzvergleiche.

| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Städte |
| $D$ | Entfernungsmatrix |
| $t$, $L$, $g$ | Tour, Länge, Lücke in Prozent |
| $\pi_\theta$, $P_\theta$ | Richtlinie, Wahrscheinlichkeit einer Tour |
| $s$, $V_t$ | Startstadt, offene Städte |
| $d$ | Einbettungsdimension (32) |
| $B$, $k$, $m$ | Breite der Strahlsuche, Zahl gezogener Touren, Zahl Konstruktionen |
| $h_t$ | lebende Hypothesen im Schritt $t$ |
| $K$, $O$ | Kandidatenbewertungen, grobe Operationen |
| $Z$, $M$ | Zugauswertungen, Prüfrunde |
| $c_\Delta$, $w$, $c^\star$ | Operationen je Delta, Wechselkurs, Schwellenwert |

**Quellen (nur gelesene Stufen):** Vinyals u. a. 2015: Zeigernetz, Strahlsuche über gültige Touren. Bello u. a. 2016: Zeigernetz mit Verstärkungslernen, Active Search, Budget in Lösungen je
Instanz. Kool u. a. 2019: Aufmerksamkeitsmodell, Sampling von 1 280 Touren. Kwon u. a. 2020 (POMO): Mehrfachstart aus allen Startstädten und ×8-Augmentierung beim Einsatz. Joshi u. a. 2021: ein
nicht gelerntes Einfügeverfahren schlägt gelernte Modelle bei TSP200 im Budget. Dass Mehrfachstart, Sampling und Strahlsuche mit mehr Rechnen besser werden, ist Stand der Literatur; neu ist
nur die Zählung in K und O nebeneinander am selben Netz.
"""
