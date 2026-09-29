#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Déformation p_D -> q_beta : visualisation interactive 3D (Plotly)
=================================================================

Objet
-----
Comprendre l'effet géométrique / statistique du reweighting

    w_beta(n)   = (1 - beta^n) / (1 - beta)              (compte effectif)
    f_beta(n)   = w_beta(n) / n                          (poids par occurrence)
    p_D(x)      = n_x / N
    q_beta(x)   = p_D(x) f_beta(n_x) / E_{p_D}[f_beta]   = w_beta(n_x) / sum_x' w_beta(n_x')
    R_beta(x)   = q_beta(x) / p_D(x)

sur un jeu de données synthétique de paires (s,a) = x in R^3 réparties en 4 clusters
séparés (A: très fréquent, B: moyen, C: rare, D: extrêmement rare).

Les COORDONNÉES DES POINTS NE CHANGENT JAMAIS : seule la masse de probabilité
(taille / couleur des points) est redistribuée.

Usage
-----
    pip install numpy plotly
    python ppo_reweighting_viz.py --open
    python ppo_reweighting_viz.py --seed 3 --out ma_figure.html

Sortie : un fichier HTML autonome (Plotly embarqué, fonctionne hors ligne) + un
rapport de vérifications numériques dans le terminal.

Contrôles dans la figure
------------------------
* curseur : beta, gradué en -log10(1 - beta) (0 -> 0, 1 -> 0.9, 2 -> 0.99, 3 -> 0.999,
  4 -> 0.9999) car tout se passe près de beta = 1 ;
* boutons : presets beta = 0, 0.5, 0.9, 0.99, 0.999, 0.9999 + lecture / pause ;
* rotation / zoom à la souris ; les 3 vues 3D partagent la même caméra ;
* survol : x, n_x, p_D(x), q_beta(x), q_beta(x)/p_D(x).
"""

import argparse
import math
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. Mathématiques
# =============================================================================


def w_beta(n, beta):
    """Compte effectif w_beta(n) = (1 - beta^n)/(1 - beta) = sum_{k<n} beta^k.

    Implémentation numériquement stable (expm1) valable jusqu'à beta -> 1.
    Cas limites : beta = 0 -> 1 (pour n >= 1) ; beta = 1 -> n.
    """
    n = np.asarray(n, dtype=float)
    if beta <= 0.0:
        return np.ones_like(n)
    if beta >= 1.0:
        return n.copy()
    lb = math.log(beta)
    return np.expm1(n * lb) / math.expm1(lb)


def f_beta(n, beta):
    """Poids par occurrence f_beta(n) = w_beta(n) / n."""
    n = np.asarray(n, dtype=float)
    return w_beta(n, beta) / n


def compute_state(n, cl, beta):
    """Calcule p_D, q_beta, R_beta et tous les indicateurs pour un beta donné."""
    n = np.asarray(n, dtype=float)
    N = n.sum()
    K = n.size
    p = n / N
    w = w_beta(n, beta)
    W = w.sum()
    q = w / W
    R = q / p
    logR = np.log(R)

    order = np.argsort(n, kind="stable")
    Rs = R[order]
    monotone = bool(np.all(np.diff(Rs) <= 1e-12 * np.abs(Rs[:-1])))

    n_clusters = int(cl.max()) + 1
    return dict(
        beta=float(beta), N=N, K=K, p=p, w=w, W=W, q=q, R=R,
        Hp=float(-np.sum(p * np.log(p))),
        Hq=float(-np.sum(q * np.log(q))),
        kl_qp=float(np.sum(q * logR)),          # KL(q || p_D) = E_q[log R]
        kl_pq=float(-np.sum(p * logR)),         # KL(p_D || q) = -E_p[log R]
        tv=float(0.5 * np.sum(np.abs(q - p))),
        inf_unif=float(np.max(np.abs(q - 1.0 / K))),
        Rmax=float(R.max()), Rmin=float(R.min()),
        n_at_Rmax=int(n[np.argmax(R)]), n_at_Rmin=int(n[np.argmin(R)]),
        inv_simpson_p=float(1.0 / np.sum(p ** 2)),
        inv_simpson_q=float(1.0 / np.sum(q ** 2)),
        monotone=monotone,
        mass_p=np.array([p[cl == k].sum() for k in range(n_clusters)]),
        mass_q=np.array([q[cl == k].sum() for k in range(n_clusters)]),
    )


# =============================================================================
# 2. Jeu de données synthétique
# =============================================================================

CLUSTERS = [
    dict(name="A", label="très fréquent", center=(3.0, 3.0, 3.0), spread=0.9, npts=45,
         color="#2ca02c",
         count_fn=lambda rng, m: np.clip(rng.lognormal(np.log(500), 0.45, m), 150, 1500)),
    dict(name="B", label="fréquence moyenne", center=(3.0, -3.0, -3.0), spread=0.9, npts=70,
         color="#ff7f0e",
         count_fn=lambda rng, m: np.clip(rng.lognormal(np.log(50), 0.5, m), 15, 200)),
    dict(name="C", label="rare", center=(-3.0, 3.0, -3.0), spread=0.9, npts=90,
         color="#9467bd",
         count_fn=lambda rng, m: np.clip(rng.lognormal(np.log(7), 0.4, m), 3, 15)),
    dict(name="D", label="extrêmement rare", center=(-3.0, -3.0, 3.0), spread=0.9, npts=45,
         color="#8c564b",
         count_fn=lambda rng, m: rng.choice([1, 2], size=m, p=[0.7, 0.3])),
]


def make_dataset(seed):
    """Retourne X (K,3) coordonnées, n (K,) comptes entiers >= 1, cl (K,) indice de cluster."""
    rng = np.random.default_rng(seed)
    X, counts, cl = [], [], []
    for k, c in enumerate(CLUSTERS):
        pts = rng.normal(loc=c["center"], scale=c["spread"], size=(c["npts"], 3))
        cnt = np.maximum(1, np.rint(c["count_fn"](rng, c["npts"]))).astype(int)
        X.append(pts)
        counts.append(cnt)
        cl.append(np.full(c["npts"], k, dtype=int))
    return np.vstack(X), np.concatenate(counts), np.concatenate(cl)


# =============================================================================
# 3. Vérifications numériques (terminal)
# =============================================================================


def run_sanity_checks(n, cl):
    print("=" * 78)
    print("VÉRIFICATIONS NUMÉRIQUES")
    print("=" * 78)
    K, N = len(n), int(n.sum())
    print(f"|X| = {K} paires distinctes, N = {N} occurrences, n_min = {n.min()}, n_max = {n.max()}\n")

    all_ok = True

    def report(name, ok, detail=""):
        nonlocal all_ok
        all_ok = all_ok and bool(ok)
        print(f"  [{'OK' if ok else 'ÉCHEC'}] {name} {detail}")

    # (1) beta = 0 -> uniforme
    st0 = compute_state(n, cl, 0.0)
    report("beta = 0 : q = 1/|X|", st0["inf_unif"] < 1e-15,
           f"(max|q - 1/|X|| = {st0['inf_unif']:.2e})")

    # (2) beta -> 1 : q -> p_D
    print("\n  beta -> 1 : q_beta -> p_D")
    print(f"    {'beta':>16} {'TV(q,p_D)':>12} {'max|R-1|':>12}")
    betas_lim = [0.9, 0.99, 0.999, 0.9999, 1 - 1e-6, 1 - 1e-9, 1.0]
    tvs = []
    for b in betas_lim:
        st = compute_state(n, cl, b)
        tvs.append(st["tv"])
        print(f"    {b:16.10f} {st['tv']:12.3e} {np.max(np.abs(st['R'] - 1)):12.3e}")
    report("TV(q_beta, p_D) décroît quand beta -> 1", all(np.diff(tvs) <= 0))
    report("beta = 1 - 1e-9 : TV < 1e-5", tvs[-2] < 1e-5, f"(TV = {tvs[-2]:.2e})")
    report("beta = 1 (limite exacte) : q = p_D", tvs[-1] < 1e-15, f"(TV = {tvs[-1]:.2e})")

    # (3) équivalence des deux écritures de q_beta + normalisation
    print()
    grid_t = np.linspace(0, 4, 60)
    betas_grid = np.round(1 - 10.0 ** (-grid_t), 12)
    max_diff, max_norm, max_meanR = 0.0, 0.0, 0.0
    for b in betas_grid:
        p = n / N
        f = f_beta(n, b)
        q1 = p * f / np.sum(p * f)                  # p_D f / E_{p_D}[f]
        q2 = w_beta(n, b) / w_beta(n, b).sum()      # w / sum w
        max_diff = max(max_diff, np.max(np.abs(q1 - q2)))
        max_norm = max(max_norm, abs(q2.sum() - 1))
        max_meanR = max(max_meanR, abs(np.sum(p * (q2 / p)) - 1))
    report("q = p_D f / E[f]  <=>  q = w / sum w", max_diff < 1e-13, f"(écart max = {max_diff:.2e})")
    report("sum q = 1 et E_{p_D}[R] = 1", max(max_norm, max_meanR) < 1e-12,
           f"(écarts max = {max_norm:.2e}, {max_meanR:.2e})")

    # (4) monotonie de R en n
    order = np.argsort(n, kind="stable")
    ns = n[order]
    strict_ok, weak_ok = True, True
    for b in betas_grid:
        R = compute_state(n, cl, b)["R"][order]
        dR = np.diff(R)
        dn = np.diff(ns)
        weak_ok &= bool(np.all(dR <= 1e-12 * np.abs(R[:-1])))
        strict_ok &= bool(np.all(dR[dn > 0] < 0))
    report("R_beta(x) décroît (au sens large) avec n_x", weak_ok,
           f"({len(betas_grid)} valeurs de beta dans [0, 0.9999])")
    report("R_beta(x) décroît strictement entre comptes distincts", strict_ok)

    print("\n" + ("Toutes les vérifications passent." if all_ok else "ATTENTION : au moins un test a échoué."))
    print("=" * 78 + "\n")
    return all_ok


# =============================================================================
# 4. Figure Plotly
# =============================================================================

CS = [[0.0, "#2166ac"], [0.5, "#c8c8c8"], [1.0, "#b2182b"]]   # bleu = perd, gris = inchangé, rouge = gagne
S_MAX, S_MIN = 34.0, 2.0        # taille (px) max / min des marqueurs de masse
N_HIST_BINS = 40

HOVER3D = ("<b>cluster %{text}</b><br>"
           "x = (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
           "n_x = %{customdata[0]:.0f}<br>"
           "p_D(x) = %{customdata[1]:.3e}<br>"
           "q_β(x) = %{customdata[2]:.3e}<br>"
           "q_β/p_D = %{customdata[3]:.4g}<extra></extra>")

POST_JS = """
var gd = document.getElementById('{plot_id}');
var syncing = false;
var scenes = ['scene', 'scene2', 'scene3'];
gd.on('plotly_relayout', function (ev) {
  if (syncing) return;
  var src = null, cam = null;
  scenes.forEach(function (s) {
    if (ev[s + '.camera'] !== undefined) { src = s; cam = ev[s + '.camera']; }
  });
  if (!src) return;
  var upd = {};
  scenes.forEach(function (s) { if (s !== src) upd[s + '.camera'] = cam; });
  syncing = true;
  Plotly.relayout(gd, upd).then(function () { syncing = false; },
                                function () { syncing = false; });
});
"""


def sig(a, digits=5):
    """Liste de floats arrondis à `digits` chiffres significatifs (HTML plus léger)."""
    return [float(f"{v:.{digits}g}") for v in np.ravel(a)]


def build_figure(X, n, cl):
    K, N = len(n), float(n.sum())
    nmax = float(n.max())
    n_plot_max = 1.3 * nmax
    n_grid = np.logspace(0, np.log10(n_plot_max), 200)

    # --- grille de beta : régulière en -log10(1-beta), presets inclus exactement ---
    t_grid = np.linspace(0.0, 4.0, 61)
    betas = np.round(1.0 - 10.0 ** (-t_grid), 10)
    betas = np.unique(np.concatenate([betas, [0.5]]))
    names = [f"{b:.6f}" for b in betas]
    presets = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999]
    i0 = int(np.argmin(np.abs(betas - 0.99)))

    # --- états pour tous les beta + constantes globales (échelles fixes) ---
    states = [compute_state(n, cl, b) for b in betas]
    p = states[0]["p"]
    ref_mass = max(p.max(), max(s["q"].max() for s in states))
    M = 1.02 * max(np.max(np.abs(np.log2(s["R"]))) for s in states) + 1e-9
    edges = np.linspace(-M, M, N_HIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = edges[1] - edges[0]
    hist_ymax = max(np.histogram(np.log2(s["R"]), bins=edges)[0].max() for s in states)
    R_all_min, R_all_max = np.inf, 0.0
    for s in states:
        Rc = f_beta(n_grid, s["beta"]) * N / s["W"]
        R_all_min = min(R_all_min, s["R"].min(), Rc.min())
        R_all_max = max(R_all_max, s["R"].max(), Rc.max())
    ks = sorted(set(range(0, int(M) + 1, 2)) | set(range(0, -int(M) - 1, -2)))
    k_txt = [f"{2.0 ** k:g}" for k in ks]

    pt_colors = [CLUSTERS[k]["color"] for k in cl]
    pt_text = [f"{CLUSTERS[k]['name']} ({CLUSTERS[k]['label']})" for k in cl]

    def mass_sizes(m):
        return sig(np.clip(S_MAX * np.sqrt(m / ref_mass), S_MIN, None), 3)

    # --- bornes des axes des graphiques 2D (échelles log) ---
    f_lo, f_hi = math.log10(0.5 / n_plot_max), math.log10(1.6)
    w_lo, w_hi = -0.1, math.log10(n_plot_max) + 0.05
    R_lo, R_hi = math.log10(R_all_min / 1.5), math.log10(R_all_max * 1.5)
    x_lo, x_hi = -0.05, math.log10(n_plot_max)

    # -------------------------------------------------------------------------
    # Parties dynamiques (dépendent de beta) : {indice de trace: (classe, props)}
    # -------------------------------------------------------------------------
    def metrics_text(st):
        b = st["beta"]
        lines = [
            f"<b>β = {b:.4f}</b>    n_β = 1/(1−β) = {1.0 / (1.0 - b):.4g}",
            f"|X| = {st['K']}   N = {int(st['N'])}   Σw = {st['W']:.1f}",
            "",
            f"H(p_D) = {st['Hp']:.4f}    exp(H) = {math.exp(st['Hp']):6.1f}",
            f"H(q_β) = {st['Hq']:.4f}    exp(H) = {math.exp(st['Hq']):6.1f}",
            f"log|X| = {math.log(st['K']):.4f}    (entropie max)",
            f"1/Σp² = {st['inv_simpson_p']:6.1f}   1/Σq² = {st['inv_simpson_q']:6.1f}",
            "",
            f"KL(q‖p_D) = {st['kl_qp']:.4f}",
            f"KL(p_D‖q) = {st['kl_pq']:.4f}",
            f"TV(q,p_D) = {st['tv']:.4f}",
            f"max q/p_D = {st['Rmax']:.4g}   (n = {st['n_at_Rmax']})",
            f"min q/p_D = {st['Rmin']:.4g}   (n = {st['n_at_Rmin']})",
            "",
            f"max|q − 1/|X|| = {st['inf_unif']:.2e}",
            f"R décroît avec n : {'✔' if st['monotone'] else '✘'}",
        ]
        return "<br>".join(lines)

    def dyn_props(st):
        b = st["beta"]
        q, R = st["q"], st["R"]
        lr2 = np.log2(R)
        cd = [[int(a), float(f"{pp:.5g}"), float(f"{qq:.5g}"), float(f"{rr:.5g}")]
              for a, pp, qq, rr in zip(n, p, q, R)]
        f_curve = f_beta(n_grid, b)
        w_curve = w_beta(n_grid, b)
        R_curve = f_curve * N / st["W"]
        asym = 1.0 / (1.0 - b)
        nb = 1.0 / (1.0 - b)
        hist = np.histogram(lr2, bins=edges)[0]
        marker_R = dict(
            size=sig(3.0 + 21.0 * np.abs(lr2) / M, 3), color=sig(lr2, 4),
            colorscale=CS, cmin=-M, cmax=M, opacity=0.9, showscale=True,
            colorbar=dict(title=dict(text="R = q/p_D"), tickvals=ks, ticktext=k_txt,
                          len=0.36, y=0.80, yanchor="middle", x=1.005, xanchor="left",
                          thickness=14),
        )
        return {
            0: (go.Scatter3d, dict(customdata=cd)),
            2: (go.Scatter3d, dict(marker=dict(size=mass_sizes(q), color=pt_colors, opacity=0.9),
                                   customdata=cd)),
            3: (go.Scatter3d, dict(marker=marker_R, customdata=cd)),
            6: (go.Scatter, dict(y=sig(f_curve))),
            7: (go.Scatter, dict(y=sig(st["w"] / n))),
            8: (go.Scatter, dict(x=[nb, nb])),
            11: (go.Scatter, dict(y=sig(w_curve))),
            12: (go.Scatter, dict(y=sig(st["w"]))),
            13: (go.Scatter, dict(y=[asym, asym])),
            14: (go.Scatter, dict(x=[nb, nb])),
            15: (go.Scatter, dict(y=sig(R_curve))),
            16: (go.Scatter, dict(y=sig(R))),
            17: (go.Scatter, dict(x=[nb, nb])),
            18: (go.Bar, dict(y=[int(h) for h in hist])),
            20: (go.Bar, dict(y=sig(st["mass_q"]),
                              text=[f"{100 * m:.1f}%" for m in st["mass_q"]])),
            21: (go.Scatter, dict(text=[metrics_text(st)])),
        }

    all_dyn = [dyn_props(s) for s in states]
    dyn0 = all_dyn[i0]
    dyn_idx = sorted(dyn0.keys())

    # -------------------------------------------------------------------------
    # Figure + traces (l'ordre = indices ci-dessus)
    # -------------------------------------------------------------------------
    sq = lambda color: f"<span style='color:{color}'>■</span>"   # noqa: E731
    titles = [
        "<b>p<sub>D</sub>(x)</b> — aire du point ∝ masse",
        "<b>q<sub>β</sub>(x)</b> — même échelle ; ○ gris = contour de p<sub>D</sub>",
        "<b>R<sub>β</sub>(x) = q<sub>β</sub>/p<sub>D</sub></b> — couleur = log₂R, taille ∝ |log₂R|",
        "<b>f<sub>β</sub>(n) = w<sub>β</sub>(n)/n</b> (poids par occurrence)",
        "<b>w<sub>β</sub>(n) = (1−β<sup>n</sup>)/(1−β)</b> (compte effectif)",
        "<b>R<sub>β</sub> = q<sub>β</sub>/p<sub>D</sub> en fonction de n<sub>x</sub></b>",
        "Histogramme de log₂ R<sub>β</sub>(x) (une paire = 1)",
        f"Masse par cluster : {sq('#334155')} p<sub>D</sub>   {sq('#0ea5e9')} q<sub>β</sub>",
        "Indicateurs (β courant)",
    ]
    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{"type": "scene"}] * 3, [{"type": "xy"}] * 3, [{"type": "xy"}] * 3],
        subplot_titles=titles, row_heights=[0.42, 0.29, 0.29],
        vertical_spacing=0.085, horizontal_spacing=0.06,
    )

    def add(idx, cls, row, col, **kw):
        assert len(fig.data) == idx, (len(fig.data), idx)
        if idx in dyn0:
            kw.update(dyn0[idx][1])
        fig.add_trace(cls(**kw), row=row, col=col)

    size_p = mass_sizes(p)
    xyz = dict(x=X[:, 0], y=X[:, 1], z=X[:, 2])

    # --- Row 1 : trois vues 3D (mêmes coordonnées) ---
    add(0, go.Scatter3d, 1, 1, **xyz, mode="markers", text=pt_text, hovertemplate=HOVER3D,
        marker=dict(size=size_p, color=pt_colors, opacity=0.9), name="p_D")
    add(1, go.Scatter3d, 1, 2, **xyz, mode="markers", hoverinfo="skip",
        marker=dict(size=size_p, symbol="circle-open", color="rgba(60,60,60,0.9)",
                    line=dict(width=2, color="rgba(60,60,60,0.9)")), name="contour p_D")
    add(2, go.Scatter3d, 1, 2, **xyz, mode="markers", text=pt_text, hovertemplate=HOVER3D,
        name="q_beta")
    add(3, go.Scatter3d, 1, 3, **xyz, mode="markers", text=pt_text, hovertemplate=HOVER3D,
        name="R_beta")

    # --- Row 2, col 1 : f_beta(n) ---
    ref_line = dict(color="gray", width=1.5)
    lab = dict(size=11, color="gray")
    txt_at = lambda length, i, s: [""] * i + [s] + [""] * (length - i - 1)   # noqa: E731
    add(4, go.Scatter, 2, 1, x=n_grid, y=1.0 / n_grid, mode="lines+text",
        line=dict(**ref_line, dash="dash"), text=txt_at(200, 150, "1/n  (β=0)"),
        textposition="top right", textfont=lab, hoverinfo="skip")
    add(5, go.Scatter, 2, 1, x=[1, n_plot_max], y=[1, 1], mode="lines+text",
        line=dict(**ref_line, dash="dot"), text=["", "1  (β→1)"],
        textposition="top left", textfont=lab, hoverinfo="skip")
    add(6, go.Scatter, 2, 1, x=n_grid, mode="lines", line=dict(color="#111", width=3),
        hovertemplate="n = %{x:.1f}<br>f_β = %{y:.4g}<extra></extra>")
    add(7, go.Scatter, 2, 1, x=n, mode="markers", text=pt_text,
        marker=dict(size=6, color=pt_colors, opacity=0.65),
        hovertemplate="cluster %{text}<br>n = %{x}<br>f_β = %{y:.4g}<extra></extra>")
    add(8, go.Scatter, 2, 1, x=[1, 1], y=[10 ** f_lo, 10 ** f_hi], mode="lines",
        line=dict(color="#e67e22", width=1.5, dash="dot"), hoverinfo="skip")

    # --- Row 2, col 2 : w_beta(n) ---
    add(9, go.Scatter, 2, 2, x=n_grid, y=n_grid, mode="lines+text",
        line=dict(**ref_line, dash="dot"), text=txt_at(200, 150, "n  (β→1)"),
        textposition="bottom right", textfont=lab, hoverinfo="skip")
    add(10, go.Scatter, 2, 2, x=[1, n_plot_max], y=[1, 1], mode="lines+text",
        line=dict(**ref_line, dash="dash"), text=["", "1  (β=0)"],
        textposition="top left", textfont=lab, hoverinfo="skip")
    add(11, go.Scatter, 2, 2, x=n_grid, mode="lines", line=dict(color="#111", width=3),
        hovertemplate="n = %{x:.1f}<br>w_β = %{y:.4g}<extra></extra>")
    add(12, go.Scatter, 2, 2, x=n, mode="markers", text=pt_text,
        marker=dict(size=6, color=pt_colors, opacity=0.65),
        hovertemplate="cluster %{text}<br>n = %{x}<br>w_β = %{y:.4g}<extra></extra>")
    add(13, go.Scatter, 2, 2, x=[1, n_plot_max], mode="lines+text",
        line=dict(color="gray", width=1.5, dash="dashdot"), text=["", "1/(1−β)"],
        textposition="top left", textfont=lab, hoverinfo="skip")
    add(14, go.Scatter, 2, 2, x=[1, 1], y=[10 ** w_lo, 10 ** w_hi], mode="lines",
        line=dict(color="#e67e22", width=1.5, dash="dot"), hoverinfo="skip")

    # --- Row 2, col 3 : R_beta vs n ---
    add(15, go.Scatter, 2, 3, x=n_grid, mode="lines", line=dict(color="#111", width=3),
        hovertemplate="n = %{x:.1f}<br>R_β = %{y:.4g}<extra></extra>")
    add(16, go.Scatter, 2, 3, x=n, mode="markers", text=pt_text,
        marker=dict(size=6, color=pt_colors, opacity=0.65),
        hovertemplate="cluster %{text}<br>n = %{x}<br>R_β = %{y:.4g}<extra></extra>")
    add(17, go.Scatter, 2, 3, x=[1, 1], y=[10 ** R_lo, 10 ** R_hi], mode="lines",
        line=dict(color="#e67e22", width=1.5, dash="dot"), hoverinfo="skip")

    # --- Row 3 : histogramme de log2 R, masses par cluster, indicateurs ---
    add(18, go.Bar, 3, 1, x=centers, width=bin_w * 0.92,
        marker=dict(color=centers, colorscale=CS, cmin=-M, cmax=M, showscale=False),
        hovertemplate="log₂R ≈ %{x:.2f}<br>%{y} paires<extra></extra>")
    cl_labels = [f"{c['name']}<br>n∈[{int(n[cl == k].min())}, {int(n[cl == k].max())}]"
                 for k, c in enumerate(CLUSTERS)]
    add(19, go.Bar, 3, 2, x=cl_labels, y=sig(states[0]["mass_p"]),
        text=[f"{100 * m:.1f}%" for m in states[0]["mass_p"]], textposition="outside",
        cliponaxis=False, marker_color="#334155", name="p_D",
        hovertemplate="p_D(cluster) = %{y:.4f}<extra></extra>")
    add(20, go.Bar, 3, 2, x=cl_labels, textposition="outside", cliponaxis=False,
        marker_color="#0ea5e9", name="q_beta",
        hovertemplate="q_β(cluster) = %{y:.4f}<extra></extra>")
    add(21, go.Scatter, 3, 3, x=[0.0], y=[0.5], mode="text", textposition="middle right",
        textfont=dict(family="Courier New, monospace", size=12), hoverinfo="skip")

    # Ligne de référence R_beta = 1 dans le graphe R_beta(n)
    fig.add_trace(
        go.Scatter(
            x=n_grid,
            y=np.ones_like(n_grid),
            mode="lines",
            line=dict(color="gray", dash="dot", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Ligne de référence log2(R) = 0 dans l'histogramme
    fig.add_trace(
        go.Scatter(
            x=[0.0, 0.0],
            y=[0.0, hist_ymax * 1.1],
            mode="lines",
            line=dict(color="black", dash="dot", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # -------------------------------------------------------------------------
    # Axes / scènes
    # -------------------------------------------------------------------------
    lo, hi = float(X.min()) - 1.0, float(X.max()) + 1.0
    ax3 = dict(range=[lo, hi], showbackground=True, backgroundcolor="rgb(244,245,250)",
               gridcolor="white", zerolinecolor="lightgray")
    fig.update_scenes(
        xaxis=dict(title="x₁", **ax3), yaxis=dict(title="x₂", **ax3), zaxis=dict(title="x₃", **ax3),
        aspectmode="cube", camera=dict(eye=dict(x=1.55, y=1.55, z=1.0)),
    )

    fig.update_xaxes(type="log", range=[x_lo, x_hi], title_text="n<sub>x</sub> (log)", row=2, col=1)
    fig.update_yaxes(type="log", range=[f_lo, f_hi], title_text="f<sub>β</sub>(n) (log)", row=2, col=1)
    fig.update_xaxes(type="log", range=[x_lo, x_hi], title_text="n<sub>x</sub> (log)", row=2, col=2)
    fig.update_yaxes(type="log", range=[w_lo, w_hi], title_text="w<sub>β</sub>(n) (log)", row=2, col=2)
    fig.update_xaxes(type="log", range=[x_lo, x_hi], title_text="n<sub>x</sub> (log)", row=2, col=3)
    fig.update_yaxes(type="log", range=[R_lo, R_hi], title_text="q<sub>β</sub>/p<sub>D</sub> (log)", row=2, col=3)
    fig.update_xaxes(title_text="log₂ R<sub>β</sub>(x)", range=[-M, M], row=3, col=1)
    fig.update_yaxes(title_text="nombre de paires distinctes", range=[0, hist_ymax * 1.1], row=3, col=1)
    fig.update_yaxes(title_text="masse totale", range=[0, 1.18], tickformat=".0%", row=3, col=2)
    fig.update_xaxes(visible=False, range=[0, 1], row=3, col=3)
    fig.update_yaxes(visible=False, range=[0, 1], row=3, col=3)

    # -------------------------------------------------------------------------
    # Curseur + boutons
    # -------------------------------------------------------------------------
    def anim_args(name):
        return [[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))]

    def frame_of(beta):
        return names[int(np.argmin(np.abs(betas - beta)))]

    steps = [dict(method="animate", label=f"{b:.4f}", args=anim_args(names[i]))
             for i, b in enumerate(betas)]
    sliders = [dict(
        active=i0, steps=steps, x=0.02, len=0.96, y=1.095, yanchor="top",
        pad=dict(t=0, b=0), ticklen=3,
        font=dict(color="rgba(0,0,0,0)"),            # cache les 60+ étiquettes (illisibles)
        currentvalue=dict(prefix="β = ", font=dict(size=20, color="#111"), xanchor="left"),
    )]
    preset_buttons = [dict(label=f"β = {b:g}", method="animate", args=anim_args(frame_of(b)))
                      for b in presets]
    play_buttons = [
        dict(label="▶ balayage", method="animate",
             args=[None, dict(mode="immediate", fromcurrent=True,
                              frame=dict(duration=150, redraw=True), transition=dict(duration=0))]),
        dict(label="⏸ pause", method="animate",
             args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0))]),
    ]
    updatemenus = [
        dict(type="buttons", direction="right", buttons=preset_buttons, showactive=False,
             x=0.02, xanchor="left", y=1.125, yanchor="bottom", pad=dict(r=4, t=2, b=2)),
        dict(type="buttons", direction="right", buttons=play_buttons, showactive=False,
             x=0.98, xanchor="right", y=1.125, yanchor="bottom", pad=dict(r=4, t=2, b=2)),
    ]

    fig.update_layout(
        template="plotly_white", height=1500, margin=dict(t=270, b=70, l=70, r=110),
        showlegend=False, uirevision="keep", barmode="group", bargap=0.15,
        title=dict(
            text=("<b>Déformation de la distribution empirique p<sub>D</sub> → q<sub>β</sub></b>"
                  " — mêmes points, mêmes coordonnées : seule la masse est redistribuée"
                  "<br><sup>Curseur gradué en −log₁₀(1−β) : 0 → β=0 · 1 → 0,9 · 2 → 0,99 · "
                  "3 → 0,999 · 4 → 0,9999. Trait orange pointillé = échelle de comptes "
                  "n<sub>β</sub> = 1/(1−β).</sup>"),
            x=0.5, xanchor="center", y=0.99, yanchor="top"),
        sliders=sliders, updatemenus=updatemenus,
    )

    # -------------------------------------------------------------------------
    # Frames (une par beta) : seules les traces dynamiques sont mises à jour
    # -------------------------------------------------------------------------
    frames = []
    for name, dyn in zip(names, all_dyn):
        data = [dyn[i][0](**dyn[i][1]) for i in dyn_idx]
        frames.append(go.Frame(name=name, data=data, traces=dyn_idx))
    fig.frames = frames
    return fig


# =============================================================================
# 5. Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description="Visualisation interactive p_D -> q_beta")
    ap.add_argument("--seed", type=int, default=7, help="graine du jeu de données synthétique")
    ap.add_argument("--out", default="ppo_reweighting_viz.html", help="fichier HTML de sortie")
    ap.add_argument("--open", action="store_true", help="ouvrir le HTML dans le navigateur")
    args = ap.parse_args()

    X, n, cl = make_dataset(args.seed)
    run_sanity_checks(n, cl)

    fig = build_figure(X, n, cl)
    out = Path(args.out)
    fig.write_html(
        str(out), include_plotlyjs=True, full_html=True, auto_play=False, post_script=POST_JS,
        config=dict(displaylogo=False, scrollZoom=True,
                    toImageButtonOptions=dict(format="png", scale=2)),
    )
    print(f"Figure écrite dans : {out.resolve()}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
