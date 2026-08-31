"""
esgclust.py
===========
Shared helpers for AI-integrated Sustainable Finance, Lecture 7c:
*Clustering as peer-group construction*.

Everything here is deterministic given a seed, and nothing touches the network.
The datasets are synthetic on purpose: the point of this lab is that you can
only evaluate an imputation method if you know the answer you were supposed to
recover, and with real ESG data you never do.

Design notes
------------
* Plotly only. No matplotlib, no seaborn.
* British spelling in prose. American spellings appear only where they are
  library API keywords (``color``, ``center``, ``normalize``).
* Every generator returns ground truth alongside the observable data, so the
  masked-cell benchmark in Section 10 has something to score against.

Author: Vitali Alexeev, UTS Business School
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ARCHETYPES",
    "SECTORS",
    "make_esg_panel",
    "ampute",
    "impute_by_group",
    "masked_cell_benchmark",
    "make_facilities",
    "make_name_variants",
    "normalised_name",
    "token_set_distance",
    "scatter_clusters",
    "line_curve",
    "silhouette_figure",
    "facility_map",
    "missingness_heatmap",
]

# --------------------------------------------------------------------------
# 1. The synthetic ESG panel
# --------------------------------------------------------------------------

# Five latent *production technologies*. These are the real peer groups.
# Students never see this column until Section 10.
#   name: (log_size_mu, capex_int_mu, energy_int_mu, emis_int_mu, water_int_mu)
ARCHETYPES = {
    "Heavy extractive":   (9.4, 0.22, 4.10, 5.60, 3.10),
    "Process industrial": (8.8, 0.17, 3.40, 4.30, 2.60),
    "Utilities / grid":   (9.1, 0.25, 4.60, 5.10, 1.40),
    "Asset-light service": (8.0, 0.04, 0.90, 0.70, 0.35),
    "Consumer / retail":  (8.4, 0.07, 1.60, 1.30, 1.90),
}

# GICS-style labels. Deliberately a *poor* proxy for the archetypes: each
# sector draws from several technologies, which is exactly the real problem.
SECTORS = ["Materials", "Energy", "Utilities", "Industrials",
           "Info Tech", "Cons. Staples", "Cons. Discret.", "Health Care"]

# P(archetype | sector). Rows sum to 1. Note the overlap.
_SECTOR_MIX = {
    "Materials":      [0.55, 0.35, 0.02, 0.03, 0.05],
    "Energy":         [0.60, 0.20, 0.15, 0.03, 0.02],
    "Utilities":      [0.05, 0.15, 0.75, 0.03, 0.02],
    "Industrials":    [0.20, 0.45, 0.10, 0.15, 0.10],
    "Info Tech":      [0.01, 0.04, 0.02, 0.88, 0.05],
    "Cons. Staples":  [0.03, 0.25, 0.02, 0.20, 0.50],
    "Cons. Discret.": [0.02, 0.15, 0.02, 0.31, 0.50],
    "Health Care":    [0.02, 0.20, 0.02, 0.56, 0.20],
}

REGIONS = ["Australia", "North America", "Europe", "Asia"]
# Grid carbon intensity multiplier by region -- a real driver that has nothing
# to do with what the firm makes.
_REGION_CARBON = {"Australia": 1.18, "North America": 1.02,
                  "Europe": 0.82, "Asia": 1.12}


def make_esg_panel(n: int = 260, seed: int = 7, n_orphans: int = 6) -> pd.DataFrame:
    """Build a synthetic firm-level ESG panel with known peer-group structure.

    Parameters
    ----------
    n : int
        Number of firms.
    seed : int
        Random seed. Everything downstream is reproducible from this.
    n_orphans : int
        Number of firms given a genuinely unique technology profile. These are
        the firms a density-based method should label as noise, and the firms
        you should refuse to impute.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``firm_id``. Columns:

        Observable, always present (safe to cluster on):
            ``sector``, ``region``, ``log_size``, ``capex_intensity``,
            ``energy_intensity``, ``revenue_per_employee``
        Target variables (these are what you will later hide and impute):
            ``emissions_intensity``, ``water_intensity``, ``board_independence``
        Ground truth (do not use as a feature):
            ``archetype``, ``is_orphan``
    """
    rng = np.random.default_rng(seed)
    arch_names = list(ARCHETYPES)

    sectors = rng.choice(SECTORS, size=n)
    archetypes, rows = [], []

    for i in range(n):
        sec = sectors[i]
        a_idx = rng.choice(len(arch_names), p=_SECTOR_MIX[sec])
        a_name = arch_names[a_idx]
        archetypes.append(a_name)

        size_mu, capex_mu, energy_mu, emis_mu, water_mu = ARCHETYPES[a_name]
        region = rng.choice(REGIONS)

        log_size = rng.normal(size_mu, 0.55)
        capex = np.clip(rng.normal(capex_mu, 0.035), 0.01, None)
        energy = np.clip(rng.normal(energy_mu, 0.55), 0.05, None)

        # Asset-light firms have far higher revenue per head. This is an
        # observable feature that carries archetype information.
        rev_per_emp = np.exp(rng.normal(12.9 - 0.42 * energy_mu, 0.30))

        # Emissions intensity depends on technology, on how much energy the
        # firm buys, and on where it operates. Only the first is "the peer
        # group"; the third is why region belongs in the feature set.
        emis = np.clip(
            (emis_mu + 0.35 * (energy - energy_mu)) * _REGION_CARBON[region]
            + rng.normal(0, 0.45),
            0.02, None)
        water = np.clip(water_mu + rng.normal(0, 0.40), 0.02, None)

        # Governance is deliberately NOT driven by production technology.
        # Clustering on operating features will not help you impute it, and
        # discovering that is part of the exercise.
        board_ind = np.clip(rng.normal(0.62, 0.14), 0.15, 0.98)

        rows.append((sec, region, log_size, capex, energy, rev_per_emp,
                     emis, water, board_ind))

    df = pd.DataFrame(
        rows,
        columns=["sector", "region", "log_size", "capex_intensity",
                 "energy_intensity", "revenue_per_employee",
                 "emissions_intensity", "water_intensity",
                 "board_independence"],
    )
    df.insert(0, "archetype", archetypes)
    df["is_orphan"] = False

    # A handful of firms with no peers: unusual technology, unusual scale.
    if n_orphans:
        idx = rng.choice(df.index, size=n_orphans, replace=False)
        df.loc[idx, "log_size"] = rng.normal(11.2, 0.30, size=n_orphans)
        df.loc[idx, "capex_intensity"] = rng.uniform(0.42, 0.60, size=n_orphans)
        df.loc[idx, "energy_intensity"] = rng.uniform(7.5, 9.5, size=n_orphans)
        df.loc[idx, "emissions_intensity"] = rng.uniform(9.0, 12.0, size=n_orphans)
        df.loc[idx, "archetype"] = "Orphan"
        df.loc[idx, "is_orphan"] = True

    df.index = [f"F{i:03d}" for i in range(len(df))]
    df.index.name = "firm_id"
    return df.round(4)


FEATURES_OBSERVABLE = ["log_size", "capex_intensity",
                       "energy_intensity", "revenue_per_employee"]
TARGETS = ["emissions_intensity", "water_intensity", "board_independence"]


# --------------------------------------------------------------------------
# 2. Amputation: making holes on purpose
# --------------------------------------------------------------------------

def ampute(df: pd.DataFrame,
           column: str,
           rate: float = 0.30,
           mechanism: str = "MCAR",
           seed: int = 11,
           driver: str = "log_size") -> pd.Series:
    """Return a boolean mask marking cells to hide, under a stated mechanism.

    ``mechanism`` is one of:

    ``"MCAR"``
        Missing completely at random. Every firm equally likely.
    ``"MAR"``
        Missing at random, conditional on an *observed* variable. Small firms
        are more likely to be missing, and ``driver`` is observed, so a method
        that conditions on it can in principle recover the truth.
    ``"MNAR"``
        Missing not at random. Firms with a *high value of the target itself*
        are more likely to be missing. This is the disclosure-is-strategic
        case, and no imputer can undo it from the observed data alone.

    Returns
    -------
    pandas.Series
        Boolean, ``True`` where the cell should be hidden.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    k = int(round(rate * n))

    if mechanism == "MCAR":
        score = rng.random(n)
    elif mechanism == "MAR":
        z = -(df[driver].to_numpy() - df[driver].mean()) / df[driver].std()
        score = 1.0 / (1.0 + np.exp(-1.6 * z)) + rng.normal(0, 0.18, n)
    elif mechanism == "MNAR":
        z = (df[column].to_numpy() - df[column].mean()) / df[column].std()
        score = 1.0 / (1.0 + np.exp(-1.6 * z)) + rng.normal(0, 0.18, n)
    else:
        raise ValueError("mechanism must be one of 'MCAR', 'MAR', 'MNAR'")

    cut = np.sort(score)[::-1][k - 1] if k > 0 else np.inf
    return pd.Series(score >= cut, index=df.index, name=f"{column}_missing")


# --------------------------------------------------------------------------
# 3. Group-median imputation and the masked-cell benchmark
# --------------------------------------------------------------------------

def impute_by_group(values: pd.Series,
                    groups: pd.Series,
                    fallback: str = "median") -> pd.Series:
    """Fill gaps in ``values`` with the median of the firm's group.

    Groups with no observed member at all fall back to the global median.
    This is the workhorse behind every "peer group" comparison in the lab:
    change ``groups`` and you change the imputation, with the method held
    fixed.
    """
    out = values.copy()
    grp_med = values.groupby(groups).median()
    global_fill = values.median() if fallback == "median" else values.mean()
    fill = groups.map(grp_med)
    fill = fill.fillna(global_fill)
    return out.fillna(fill)


def masked_cell_benchmark(truth: pd.Series,
                          observed: pd.Series,
                          filled: pd.Series,
                          mask: pd.Series) -> dict:
    """Score a fill against the values that were hidden.

    Returns MAE, RMSE, the ratio of filled-column standard deviation to the
    truth's (distributional fidelity), and the Spearman rank correlation over
    the *whole* column (rank fidelity).

    Report all three. Mean imputation scores respectably on RMSE while
    flattening the distribution, and rank fidelity is what actually matters
    when the data is used for screening rather than for point estimates.
    """
    from scipy.stats import spearmanr

    t = truth[mask].to_numpy(dtype=float)
    f = filled[mask].to_numpy(dtype=float)
    err = f - t

    full_truth = truth.to_numpy(dtype=float)
    full_filled = filled.to_numpy(dtype=float)

    return {
        "n_filled": int(mask.sum()),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "sd_ratio": float(np.std(full_filled) / np.std(full_truth)),
        "spearman": float(spearmanr(full_truth, full_filled).statistic),
    }


# --------------------------------------------------------------------------
# 4. Facility coordinates, for spatial clustering
# --------------------------------------------------------------------------

# Approximate centres of real Australian industrial regions, used only so the
# map looks like somewhere. The facilities themselves are invented.
_COMPLEX_CENTRES = [
    ("Pilbara",       -22.60, 118.60, 14, 0.42),
    ("Hunter Valley", -32.55, 151.15, 11, 0.22),
    ("Bowen Basin",   -22.30, 148.20, 12, 0.36),
    ("Gladstone",     -23.85, 151.25,  8, 0.14),
    ("Latrobe Valley", -38.20, 146.45, 7, 0.16),
    ("Kwinana",       -32.24, 115.77,  6, 0.10),
]


def make_facilities(seed: int = 3, n_scattered: int = 26) -> pd.DataFrame:
    """Synthetic asset register: clustered industrial complexes plus strays.

    The clusters are irregularly sized and irregularly dense, and the
    scattered assets genuinely belong to no complex. That combination is why
    ``KMeans`` is the wrong tool here and ``DBSCAN``/``HDBSCAN`` is the right
    one: you do not know the number of complexes in advance, and "this asset
    is on its own" must remain an available answer.

    Returns
    -------
    pandas.DataFrame
        Columns ``lat``, ``lon``, ``emissions_ktco2e``, ``true_complex``.
    """
    rng = np.random.default_rng(seed)
    lat, lon, emis, true = [], [], [], []

    for name, clat, clon, count, spread in _COMPLEX_CENTRES:
        lat.extend(rng.normal(clat, spread, count))
        lon.extend(rng.normal(clon, spread, count))
        emis.extend(np.exp(rng.normal(5.4, 0.85, count)))
        true.extend([name] * count)

    lat.extend(rng.uniform(-38.5, -12.0, n_scattered))
    lon.extend(rng.uniform(114.0, 153.0, n_scattered))
    emis.extend(np.exp(rng.normal(4.2, 0.9, n_scattered)))
    true.extend(["(none)"] * n_scattered)

    df = pd.DataFrame({"lat": lat, "lon": lon,
                       "emissions_ktco2e": emis, "true_complex": true})
    df.index = [f"A{i:03d}" for i in range(len(df))]
    df.index.name = "asset_id"
    return df.round(4)


# --------------------------------------------------------------------------
# 5. Company name variants, for entity resolution
# --------------------------------------------------------------------------

_BASE_NAMES = [
    "BHP Group", "Rio Tinto", "Fortescue Metals Group", "Woodside Energy",
    "Origin Energy", "AGL Energy", "Wesfarmers", "Woolworths Group",
    "Telstra Group", "Transurban Group", "Orica", "Incitec Pivot",
]
_SUFFIXES = ["Limited", "Ltd", "Ltd.", "", "Pty Ltd", "Group Ltd"]


def make_name_variants(seed: int = 5) -> pd.DataFrame:
    """Generate messy name variants of a small set of real-looking entities.

    Each base entity appears three to five times with realistic corruptions:
    case changes, punctuation, suffix churn, abbreviation and the occasional
    typographical error. ``true_entity`` is the ground truth your clustering
    should recover.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for base in _BASE_NAMES:
        for _ in range(int(rng.integers(3, 6))):
            s = base
            if rng.random() < 0.35:
                s = s.upper()
            if rng.random() < 0.25:
                s = s.replace(" ", ".").replace("..", ".")
            suffix = rng.choice(_SUFFIXES)
            if suffix:
                s = f"{s} {suffix}"
            if rng.random() < 0.20 and len(s) > 6:
                # single-character typo
                p = int(rng.integers(1, len(s) - 1))
                s = s[:p] + s[p + 1:]
            if rng.random() < 0.15:
                s = f"  {s} "
            rows.append((s, base))

    df = pd.DataFrame(rows, columns=["raw_name", "true_entity"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


_LEGAL_TOKENS = {"limited", "ltd", "pty", "plc", "inc", "incorporated",
                 "corporation", "corp", "nv", "sa", "ag", "group",
                 "holdings", "company", "co"}


def normalised_name(s: str, drop_legal: bool = True) -> str:
    """Case-fold, strip punctuation and optionally remove legal-form tokens.

    This is step one of the standard record-linkage pipeline. Run it before
    computing any string distance; without it you are measuring punctuation.
    """
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t]
    if drop_legal:
        tokens = [t for t in tokens if t not in _LEGAL_TOKENS]
    return " ".join(tokens)


def token_set_distance(a: str, b: str) -> float:
    """Jaccard distance on character trigrams, in [0, 1].

    Pure Python and dependency-free, so the notebook runs behind a firewall.
    ``rapidfuzz`` is faster and offers better-behaved scorers if you can
    install it; the pipeline around it is identical.
    """
    def grams(x: str) -> set:
        x = f"  {x} "
        return {x[i:i + 3] for i in range(len(x) - 2)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 1.0
    return 1.0 - len(ga & gb) / len(ga | gb)


# --------------------------------------------------------------------------
# 6. Plotly helpers
# --------------------------------------------------------------------------

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
_NOISE_COLOUR = "#c8c8c8"


def _colour_for(label) -> str:
    """Grey for density-method noise (label -1), palette colours otherwise."""
    if label == -1 or label == "(none)" or label == "Orphan":
        return _NOISE_COLOUR
    if isinstance(label, (int, np.integer)):
        return _PALETTE[int(label) % len(_PALETTE)]
    return _PALETTE[abs(hash(str(label))) % len(_PALETTE)]


def scatter_clusters(df: pd.DataFrame, x: str, y: str, labels,
                     title: str = "", hover: list | None = None,
                     size: str | None = None):
    """2-D scatter coloured by cluster label. Noise points are grey."""
    import plotly.graph_objects as go

    labels = pd.Series(labels, index=df.index, name="cluster")
    fig = go.Figure()
    for lab in sorted(labels.unique(), key=lambda v: (str(type(v)), str(v))):
        sub = df[labels == lab]
        name = "noise / no peers" if lab == -1 else f"cluster {lab}"
        text = None
        if hover:
            text = sub[hover].astype(str).agg(" | ".join, axis=1)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode="markers", name=str(name),
            text=text, hovertemplate="%{text}<extra></extra>" if hover else None,
            marker=dict(size=(9 if size is None else sub[size]),
                        color=_colour_for(lab),
                        line=dict(width=0.6, color="white"),
                        opacity=0.55 if lab == -1 else 0.85),
        ))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y,
                      template="plotly_white", height=460,
                      legend=dict(itemsizing="constant"))
    return fig


def line_curve(x, y, title: str = "", xaxis: str = "", yaxis: str = "",
               mark_best: str | None = None):
    """Single line with markers, optionally flagging the best point.

    ``mark_best`` is ``"max"``, ``"min"`` or ``None``.
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers",
                               line=dict(color="#1f77b4", width=2),
                               marker=dict(size=8), showlegend=False))
    if mark_best in {"max", "min"}:
        j = int(np.argmax(y) if mark_best == "max" else np.argmin(y))
        fig.add_trace(go.Scatter(x=[list(x)[j]], y=[list(y)[j]], mode="markers",
                                 marker=dict(size=15, color="#d62728",
                                             symbol="circle-open",
                                             line=dict(width=3)),
                                 name="selected", showlegend=False))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis,
                      template="plotly_white", height=380)
    return fig


def silhouette_figure(X, labels, title: str = "Silhouette by point"):
    """Per-point silhouette, sorted within cluster.

    The mean is one number and hides everything. The shape of this plot tells
    you whether a cluster is real or whether the algorithm simply partitioned
    a continuum because you asked it to.
    """
    import plotly.graph_objects as go
    from sklearn.metrics import silhouette_samples, silhouette_score

    labels = np.asarray(labels)
    keep = labels != -1
    if keep.sum() < 2 or len(np.unique(labels[keep])) < 2:
        raise ValueError("need at least two non-noise clusters")

    vals = silhouette_samples(X[keep], labels[keep])
    mean = silhouette_score(X[keep], labels[keep])

    fig = go.Figure()
    pos = 0
    for lab in np.unique(labels[keep]):
        v = np.sort(vals[labels[keep] == lab])
        fig.add_trace(go.Bar(x=v, y=np.arange(pos, pos + len(v)),
                             orientation="h", name=f"cluster {lab}",
                             marker=dict(color=_colour_for(lab)),
                             hovertemplate="s=%{x:.3f}<extra></extra>"))
        pos += len(v) + 4

    fig.add_vline(x=mean, line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"mean = {mean:.3f}")
    fig.update_layout(title=title, xaxis_title="silhouette coefficient",
                      yaxis=dict(showticklabels=False), bargap=0.0,
                      template="plotly_white", height=460)
    return fig


def facility_map(df: pd.DataFrame, labels, title: str = "",
                 size_col: str = "emissions_ktco2e"):
    """Geographic scatter of assets, coloured by cluster label."""
    import plotly.graph_objects as go

    labels = pd.Series(labels, index=df.index, name="cluster")
    sizes = 6 + 22 * (df[size_col] / df[size_col].max()) ** 0.5

    fig = go.Figure()
    for lab in sorted(labels.unique(), key=str):
        sub = df[labels == lab]
        name = "unclustered / isolated" if lab == -1 else f"complex {lab}"
        fig.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], mode="markers", name=str(name),
            text=[f"{i}<br>{e:,.0f} ktCO2e"
                  for i, e in zip(sub.index, sub[size_col])],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sizes.loc[sub.index], color=_colour_for(lab),
                        opacity=0.5 if lab == -1 else 0.85,
                        line=dict(width=0.5, color="white")),
        ))
    fig.update_layout(title=title, template="plotly_white", height=560,
                      geo=dict(scope="world", showland=True,
                               landcolor="#f4f4f0", showcountries=True,
                               countrycolor="#d9d9d9",
                               lataxis=dict(range=[-45, -8]),
                               lonaxis=dict(range=[110, 156])))
    return fig


def missingness_heatmap(df: pd.DataFrame, columns: list,
                        title: str = "Missingness pattern"):
    """Firms on the vertical axis, variables on the horizontal, gaps in red.

    Always plot this before choosing a method. The pattern is the finding.
    """
    import plotly.graph_objects as go

    m = df[columns].isna().astype(int)
    fig = go.Figure(go.Heatmap(
        z=m.to_numpy(), x=columns, y=list(df.index),
        colorscale=[[0, "#e8eef6"], [1, "#d62728"]],
        showscale=False, hovertemplate="%{y} | %{x}<extra></extra>"))
    fig.update_layout(title=title, template="plotly_white",
                      height=560, yaxis=dict(showticklabels=False,
                                             title="firms"))
    return fig
