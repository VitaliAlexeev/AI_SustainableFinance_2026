"""
esglabel.py
===========
Shared helpers for AI-integrated Sustainable Finance (25881), Lecture 9:
*Sustainable investment products: auditing the label*. Used by Labs 9a and 9b,
and by the scripts that produce every number quoted in the lecture deck, so the
slides and the notebooks cannot disagree.

    https://github.com/VitaliAlexeev/AI_SustainableFinance_2026

Data resolution
---------------
``resolve_data(name)`` looks in three places, in order, and never anywhere else:

1. ``data/<name>`` in a local clone of the repository;
2. ``_cache/<name>``, written on a previous run;
3. the repository over HTTPS, pinned by ``REF``, saving into ``_cache/``.

So the first run may touch the network; every run after that is offline.

``REF`` is ``"main"`` for convenience. For anything you intend to cite or
submit, set it to a commit SHA before loading anything. A branch name is a
moving target, which is precisely the restatement problem this lecture
complains about in labelled products and vendor data.

    esglabel.REF = "a1b2c3d..."

The two datasets
----------------
**The superannuation panel** (Lab 9a). Thirty-one investment options from
fifteen providers: for each provider, its core MySuper default plus one or more
ethical options. Six categories of concern, a de-duplicated total, and the share
of each option's value that was disclosed and could be analysed. Source:
Mindful Money (2026), *Inside Australia's Super Funds: An Ethical Review of
Investment Portfolios*, CC BY-NC 4.0; holdings as at 30 June 2025. The panel is
small enough that a verified copy is embedded in this module, so Lab 9a runs
even if the repository copy cannot be reached.

**The equity universe** (Lab 9b). S&P 500 constituents that carry both an ESG
risk rating and a market capitalisation: 368 names. The ratings and the
capitalisations come from **different vintages** (the capitalisations are an
older snapshot). The weights are therefore illustrative. Nothing in Lab 9b
depends on them being current; everything depends on them being the same for
every fund we build.

Known limitations, documented rather than patched
-------------------------------------------------
1. Disclosure coverage in the super panel runs from about 50% to 98%. Measured
   exposure is a lower bound, and the bound is looser for some options than
   others. Lab 9a, Section 11 tests whether that matters.
2. The six categories overlap: one company can sit in several. ``total`` is
   de-duplicated, which is why it is never used as a clustering feature.
3. In the equity universe, "fossil fuels" is proxied from industry labels.
   A proxy is a definition, and Lab 9b, Section 6 shows what changes when the
   definition does.

Design notes
------------
* Plotly only. No matplotlib, no seaborn.
* British spelling in prose. American spellings appear only where they are
  library API keywords (``color``, ``center``).
* Every function is deterministic given its arguments.

Author: Vitali Alexeev, UTS Business School
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "REPO", "REF", "CACHE", "resolve_data",
    "DIMS", "DIM_LABELS", "DIM_KIND", "DIM_DEFINITION", "PREFERENCE",
    "load_super_panel", "is_labelled", "short_name",
    "provider_pairs", "auc_low_means_labelled", "best_threshold_rule",
    "dimension_separation", "scale", "partition", "agreement",
    "sensitivity_grid", "default_cluster_ethicals", "outlier_frequency",
    "dbscan_sweep", "first_bisection", "disclosure_check",
    "load_universe", "SCREENS", "build_fund", "build_all_funds",
    "jaccard", "active_share", "distance_matrix", "fund_profile",
    "fossil_mask", "claim_check", "screen_features",
    "COLOURS", "fig_pairs", "fig_separation", "fig_pca_map", "fig_grid",
    "fig_dendrogram", "fig_disclosure", "fig_matrix",
]

REPO = "VitaliAlexeev/AI_SustainableFinance_2026"

# Change to a commit SHA for reproducible work. See the module docstring.
REF = "main"

CACHE = Path("_cache")


# ==========================================================================
# 0. Data resolution
# ==========================================================================

def resolve_data(name: str, verbose: bool = True) -> Path:
    """Return a local path to ``data/<name>``: local clone, then cache, then HTTPS."""
    for candidate in (Path("data") / name, Path("..") / "data" / name):
        if candidate.exists():
            if verbose:
                print(f"{name}: local clone ({candidate})")
            return candidate

    CACHE.mkdir(exist_ok=True)
    cached = CACHE / name
    if cached.exists():
        if verbose:
            print(f"{name}: cache ({cached})")
        return cached

    url = f"https://raw.githubusercontent.com/{REPO}/{REF}/data/{name}"
    if verbose:
        print(f"{name}: downloading from the repository ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            cached.write_bytes(r.read())
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(
            f"could not fetch {url}\n"
            f"  Check REF={REF!r} is a valid branch or commit and that the file "
            f"exists in data/.\n  Underlying error: {exc}") from exc
    if verbose:
        print(f"{cached.stat().st_size / 1e3:.0f} kB, cached")
    return cached


# ==========================================================================
# 1. The superannuation panel
# ==========================================================================

DIMS = ["fossil", "human_rights", "env_harm", "social_harm", "animal", "weapons"]

DIM_LABELS = {
    "fossil": "Fossil fuels", "human_rights": "Human rights",
    "env_harm": "Environmental harm", "social_harm": "Social harm",
    "animal": "Animal cruelty", "weapons": "Weapons",
}

# The source distinguishes two kinds of concern. SECTOR issues are binary: a
# company either earns revenue above a threshold in a harmful sector or it does
# not. CONDUCT issues are judgements about a company's behaviour, assessed on a
# continuum from controversy research. Two categories mix both.
DIM_KIND = {
    "fossil": "sector", "social_harm": "sector", "weapons": "sector",
    "human_rights": "conduct", "env_harm": "mixed", "animal": "mixed",
}

# Paraphrased from the Mindful Investing methodology page (May 2026).
DIM_DEFINITION = {
    "fossil": "extracting, producing, refining or distributing coal, oil or gas, "
              "including services to production and fossil-fired power generation",
    "human_rights": "operating in occupied territories, or records of significant "
                    "abuses such as risks to labour rights or public safety",
    "env_harm": "deforestation, highly hazardous pesticides, palm oil, GMOs, and "
                "ocean harm through plastic pollution, overfishing or dumping",
    "social_harm": "tobacco, gambling, alcohol, adult entertainment and "
                   "predatory lending",
    "animal": "animal testing, factory farming, fur and speciality leather, and "
              "other animal welfare issues",
    "weapons": "production and distribution of weapons",
}

# Share of Australians who say they want each issue avoided. Lonergan Research,
# January 2026, commissioned by Australian Ethical; nationally representative.
# The other three categories were not asked in comparable form.
PREFERENCE = {"human_rights": 87, "env_harm": 85, "fossil": 69}

# A verified copy of the panel. Fund type and disclosure coverage are as shown
# on mindfulmoney.nz/australia/funds (retrieved 30 September 2026); category
# exposures are from the report's per-option table.
_SUPER_PANEL_CSV = """fund,option,fund_type,fossil,human_rights,env_harm,social_harm,animal,weapons,total,disclosure_pct
AMP,Future Directions Balanced,MySuper,7.51,3.26,2.51,1.15,1.29,1.08,13.48,73.51
Australian Ethical,Growth,Ethical,0.01,1.12,0.0,0.0,0.0,0.0,1.13,79.25
Australian Ethical,MySuper Balanced,MySuper,0.01,0.95,0.0,0.0,0.0,0.0,0.96,76.23
Australian Retirement Trust,Socially Conscious Balanced,Ethical,0.46,0.43,0.22,0.0,0.67,0.0,1.56,55.82
Australian Retirement Trust,MySuper Lifecycle (under 50),MySuper,6.06,1.95,2.9,1.16,1.1,0.8,10.19,59.97
AustralianSuper,Socially Aware,Ethical,2.25,2.43,0.84,0.47,1.62,1.08,7.49,64.44
AustralianSuper,MySuper Balanced,MySuper,7.04,3.14,2.58,1.66,1.37,1.02,12.81,68.07
Aware Super,Balanced Socially Conscious,Ethical,0.17,1.39,0.09,0.0,0.96,0.06,2.65,61.79
Aware Super,High Growth Socially Conscious,Ethical,0.21,1.65,0.11,0.0,1.13,0.07,3.15,71.36
Aware Super,MySuper Lifecycle (under 55),MySuper,4.99,2.81,2.39,1.15,1.14,1.26,10.86,70.22
CareSuper,Sustainable Balanced,Ethical,2.62,1.27,1.96,0.05,0.37,0.02,3.91,57.1
CareSuper,MySuper Balanced,MySuper,4.69,1.91,2.01,1.16,1.37,0.42,8.9,56.97
Cbus,MySuper Growth,MySuper,4.11,2.17,1.99,1.13,1.03,0.73,8.87,59.41
Colonial First State,Thrive+ Sustainable Growth,Ethical,1.4,3.64,0.0,0.0,0.49,0.0,4.12,79.24
Colonial First State,MySuper Lifestage 2005-2009,MySuper,8.45,2.18,3.77,1.1,1.57,1.25,13.53,93.01
HESTA,MySuper Balanced Growth,MySuper,4.58,2.45,2.11,0.86,1.16,0.61,9.11,62.69
HESTA,Sustainable Growth,Ethical,0.38,2.08,0.39,0.2,1.35,0.19,4.44,67.59
Hostplus,MySuper Balanced,MySuper,4.03,0.96,1.44,1.28,1.09,0.14,6.83,49.73
Hostplus,Socially Responsible Balanced,Ethical,0.02,1.26,0.02,0.0,0.46,0.0,1.74,55.86
Hostplus,Socially Responsible High Growth,Ethical,0.03,2.11,0.04,0.0,0.77,0.0,2.91,86.58
Mercer Super,MySuper SmartPath 1964-68,MySuper,5.69,2.36,2.64,1.06,1.5,0.54,10.54,62.73
Mercer Super,Sustainable High Growth,Ethical,2.6,4.62,0.75,0.0,3.5,0.52,11.39,84.84
MLC,MySuper Growth (under 55),MySuper,6.22,2.68,3.35,1.0,1.62,0.92,11.62,72.65
MLC,Socially Responsible Growth,Ethical,7.25,3.41,3.67,0.0,1.87,0.78,12.7,87.79
Rest,MySuper Growth,MySuper,5.92,2.19,2.47,1.17,1.34,0.82,10.88,65.48
Rest,Sustainable Growth,Ethical,0.19,0.69,0.33,0.25,1.39,0.49,3.06,78.35
UniSuper,MySuper Balanced,MySuper,2.35,2.23,0.61,1.38,0.69,0.88,7.01,80.47
UniSuper,Sustainable Balanced,Ethical,0.18,2.29,0.43,0.0,1.81,0.17,4.53,89.97
UniSuper,Sustainable High Growth,Ethical,0.14,3.12,0.62,0.0,2.55,0.22,6.17,95.39
Vanguard Super,MySuper Lifecycle (High Growth),MySuper,5.4,3.73,1.18,1.35,2.03,0.86,12.26,92.62
Vanguard Super,Ethically Conscious Growth,Ethical,0.15,2.76,0.77,0.0,1.13,0.02,4.54,98.32
"""


def load_super_panel(verbose: bool = True) -> pd.DataFrame:
    """Load the 31-option superannuation panel.

    Tries the repository copy first (via :func:`resolve_data`) and checks it
    against the embedded copy. Falls back to the embedded copy if the file
    cannot be reached, and says so.
    """
    embedded = pd.read_csv(io.StringIO(_SUPER_PANEL_CSV))
    try:
        df = pd.read_csv(resolve_data("mindful_super_2026.csv", verbose=verbose))
        same = (df.shape == embedded.shape
                and np.allclose(df[DIMS + ["total", "disclosure_pct"]].values,
                                embedded[DIMS + ["total", "disclosure_pct"]].values)
                and (df["fund_type"] == embedded["fund_type"]).all())
        if not same and verbose:
            print("  WARNING: the repository copy differs from the copy this module "
                  "was verified against. Using the repository copy.")
    except RuntimeError:
        if verbose:
            print("mindful_super_2026.csv: not available from the repository "
                  "(missing or unreachable); using the embedded verified copy")
        df = embedded
    df["name"] = df["fund"] + " \u2014 " + df["option"]
    return df


def is_labelled(df: pd.DataFrame) -> np.ndarray:
    """1 for an ethical option, 0 for a MySuper default."""
    return (df["fund_type"] == "Ethical").astype(int).to_numpy()


_SHORT = {"Australian Retirement Trust": "ART", "Colonial First State": "CFS",
          "Vanguard Super": "Vanguard", "Mercer Super": "Mercer",
          "Australian Ethical": "Aust. Ethical", "AustralianSuper": "AustSuper"}


def short_name(fund: str, option: str | None = None) -> str:
    s = _SHORT.get(fund, fund)
    return s if option is None else f"{s} \u2014 {option}"


# ==========================================================================
# 2. What the label buys: pairs and dimensions
# ==========================================================================

def provider_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Every ethical option against its own provider's MySuper default.

    ``gap`` is the ethical option's total exposure minus the default's, in
    percentage points. Negative means the label bought a cleaner portfolio.
    """
    rows = []
    for fund, g in df.groupby("fund"):
        default = g.loc[g["fund_type"] == "MySuper", "total"]
        if len(default) != 1:
            continue
        d = float(default.iloc[0])
        for r in g[g["fund_type"] == "Ethical"].itertuples():
            rows.append({"fund": fund, "option": r.option, "ethical": r.total,
                         "default": d, "gap": r.total - d})
    return pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)


def auc_low_means_labelled(y: np.ndarray, x: np.ndarray) -> float:
    """AUC of the rule 'lower exposure means labelled'.

    Computed from ranks (the Mann--Whitney form), so it needs no scikit-learn
    and handles ties by mid-ranks. 1 means every labelled option is cleaner
    than every default on this dimension; 0.5 means the label is no guide.
    """
    y = np.asarray(y).astype(bool)
    r = pd.Series(-np.asarray(x, dtype=float)).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_threshold_rule(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Best single rule 'labelled if x < t'. Returns (t, number of errors)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(bool)
    v = np.unique(x)
    cands = np.r_[v[0] - 0.01, (v[:-1] + v[1:]) / 2, v[-1] + 0.01]
    errs = np.array([((x < t) != y).sum() for t in cands])
    i = int(errs.argmin())
    return float(cands[i]), int(errs[i])


def dimension_separation(df: pd.DataFrame) -> pd.DataFrame:
    """For each category: how well does the label separate it?"""
    y = is_labelled(df)
    rows = []
    for d in DIMS:
        t, e = best_threshold_rule(df[d], y)
        mys, eth = df.loc[y == 0, d].mean(), df.loc[y == 1, d].mean()
        rows.append({"dimension": DIM_LABELS[d], "key": d, "kind": DIM_KIND[d],
                     "auc": auc_low_means_labelled(y, df[d]),
                     "default_mean": mys, "labelled_mean": eth,
                     "ratio": mys / eth if eth > 0 else np.inf,
                     "best_threshold": t, "rule_errors": e,
                     "want_avoided_pct": PREFERENCE.get(d, np.nan)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# ==========================================================================
# 3. Clustering the products without the label
# ==========================================================================

def scale(df: pd.DataFrame, how: str = "standard", cols: list | None = None) -> np.ndarray:
    """Feature matrix under one of four scalings: standard, minmax, robust, none."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    X = df[cols or DIMS].to_numpy(dtype=float)
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
               "robust": RobustScaler(), "none": None}
    if how not in scalers:
        raise ValueError(f"how must be one of {list(scalers)}")
    return X if scalers[how] is None else scalers[how].fit_transform(X)


def partition(Z: np.ndarray, method: str = "kmeans", k: int = 2, seed: int = 42) -> np.ndarray:
    """Cluster labels from k-means or one of three agglomerative linkages."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=100, random_state=seed).fit(Z).labels_
    if method in ("ward", "complete", "average", "single"):
        return AgglomerativeClustering(n_clusters=k, linkage=method).fit(Z).labels_
    raise ValueError("method must be 'kmeans', 'ward', 'complete', 'average' or 'single'")


def agreement(y: np.ndarray, labels: np.ndarray, df: pd.DataFrame | None = None) -> dict:
    """Score a partition against the label. The label never entered the model."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    out = {"ari": float(adjusted_rand_score(y, labels)),
           "nmi": float(normalized_mutual_info_score(y, labels))}
    if df is not None:
        tot = df["total"].to_numpy()
        order = np.argsort([tot[labels == c].mean() for c in np.unique(labels)])
        names = {c: f"cluster {i + 1}" for i, c in enumerate(np.unique(labels)[order])}
        if len(order) == 2:
            names = {np.unique(labels)[order[0]]: "low-harm cluster",
                     np.unique(labels)[order[1]]: "high-harm cluster"}
        lab_named = pd.Series(labels).map(names)
        out["table"] = pd.crosstab(
            pd.Series(np.where(y == 1, "Ethical option", "MySuper default"), name=""),
            lab_named.rename(""))
        if len(order) == 2:
            clean = lab_named.to_numpy() == "low-harm cluster"
            out["agreement_rate"] = float(((y == 1) == clean).mean())
            out["labelled_in_high"] = df.loc[(y == 1) & ~clean, "name"].tolist()
            out["default_in_low"] = df.loc[(y == 0) & clean, "name"].tolist()
    return out


METHODS = ["kmeans", "ward", "complete", "average"]
SCALINGS = ["standard", "minmax", "robust", "none"]


def sensitivity_grid(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """ARI against the label for every method-by-scaling pipeline (4 x 4 = 16)."""
    from sklearn.metrics import adjusted_rand_score
    y = is_labelled(df)
    g = pd.DataFrame(index=METHODS, columns=SCALINGS, dtype=float)
    for s in SCALINGS:
        Z = scale(df, s)
        for m in METHODS:
            g.loc[m, s] = adjusted_rand_score(y, partition(Z, m, k))
    return g


def default_cluster_ethicals(df: pd.DataFrame, labels: np.ndarray) -> list:
    """Ethical options that land in the cluster holding most of the defaults."""
    y = is_labelled(df)
    dc = np.bincount(labels[y == 0], minlength=labels.max() + 1).argmax()
    return df.loc[(labels == dc) & (y == 1), "name"].tolist()


def outlier_frequency(df: pd.DataFrame, k: int = 2) -> pd.Series:
    """In how many of the 16 pipelines does each ethical option sit with the defaults?"""
    counts = pd.Series(0, index=df.loc[is_labelled(df) == 1, "name"])
    for s in SCALINGS:
        Z = scale(df, s)
        for m in METHODS:
            for n in default_cluster_ethicals(df, partition(Z, m, k)):
                counts[n] += 1
    return counts.sort_values(ascending=False)


def dbscan_sweep(df: pd.DataFrame, eps_values=(1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
                 min_samples: int = 3) -> pd.DataFrame:
    """Density clustering across neighbourhood radii. Noise = no comparable product."""
    from sklearn.cluster import DBSCAN
    Z, y = scale(df, "standard"), is_labelled(df)
    rows = []
    for eps in eps_values:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit(Z).labels_
        noise = lab == -1
        rows.append({"eps": eps,
                     "clusters": int(len(set(lab)) - (1 if noise.any() else 0)),
                     "noise": int(noise.sum()),
                     "noise_labelled": int((noise & (y == 1)).sum()),
                     "noise_members": df.loc[noise, "name"].tolist()})
    return pd.DataFrame(rows)


def first_bisection(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The divisive reading: split the universe once, then ask what divides it."""
    from sklearn.cluster import BisectingKMeans
    lab = BisectingKMeans(n_clusters=2, n_init=10, random_state=seed).fit(
        scale(df, "standard")).labels_
    rows = []
    for d in DIMS:
        a = auc_low_means_labelled(lab, df[d])
        rows.append({"dimension": DIM_LABELS[d], "kind": DIM_KIND[d],
                     "separates_the_halves": max(a, 1 - a)})
    out = pd.DataFrame(rows).sort_values("separates_the_halves", ascending=False)
    out.attrs["sizes"] = np.bincount(lab).tolist()
    return out.reset_index(drop=True)


def disclosure_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does measured exposure rise with disclosure coverage?

    If it does, options that disclose less look cleaner partly *because* they
    disclose less. A positive correlation is consistent with that, but it is not
    proof of it: growth options hold more listed equity, which is both easier to
    disclose and more likely to contain companies of concern.
    """
    from scipy.stats import spearmanr
    rows = []
    for grp, g in [("all options", df),
                   ("ethical options", df[df["fund_type"] == "Ethical"]),
                   ("MySuper defaults", df[df["fund_type"] == "MySuper"])]:
        rho, p = spearmanr(g["disclosure_pct"], g["total"])
        rows.append({"group": grp, "n": len(g), "spearman": float(rho), "p_value": float(p),
                     "median_disclosure": float(g["disclosure_pct"].median())})
    return pd.DataFrame(rows)


# ==========================================================================
# 4. The equity universe and the screens (Lab 9b)
# ==========================================================================

_ESG_COLS = ["Symbol", "Company", "Sector", "Industry", "Total ESG Risk",
             "Environment Risk", "Social Risk", "Governance Risk",
             "Controversy Level", "Controversy Score", "ESG Risk Level"]


def load_universe(verbose: bool = True) -> pd.DataFrame:
    """S&P 500 constituents with both an ESG risk rating and a market cap (368)."""
    esg = pd.read_csv(resolve_data("sp500_esg.csv", verbose), usecols=_ESG_COLS)
    cap = pd.read_csv(resolve_data("constituents-financials.csv", verbose),
                      usecols=["Symbol", "Market Cap"])
    u = (esg.merge(cap, on="Symbol", how="inner")
            .dropna(subset=["Total ESG Risk", "Market Cap", "Sector", "Industry"])
            .rename(columns={"Total ESG Risk": "esg_risk", "Environment Risk": "e_risk",
                             "Social Risk": "s_risk", "Governance Risk": "g_risk",
                             "Controversy Level": "controversy", "Market Cap": "mcap",
                             "ESG Risk Level": "risk_level"})
            .drop_duplicates("Symbol").set_index("Symbol").sort_index())
    u["controversy"] = u["controversy"].str.replace(" Controversy Level", "", regex=False)
    return u


VICE_INDUSTRIES = ("Tobacco", "Resorts & Casinos", "Beverages - Brewers",
                   "Beverages - Wineries & Distilleries")
WEAPONS_INDUSTRIES = ("Aerospace & Defense",)
FOSSIL_UTILITIES = ("Utilities - Regulated Electric", "Utilities - Regulated Gas",
                    "Utilities - Diversified", "Utilities - Independent Power Producers")


def fossil_mask(u: pd.DataFrame, definition: str = "narrow") -> pd.Series:
    """Which holdings count as fossil fuels? That depends on the definition.

    ``narrow``: the Energy sector (oil and gas producers, refiners, pipelines,
    services). ``broad``: that, plus utilities that generate or distribute fossil
    energy -- which is how the Mindful Investing methodology defines the category
    ("power companies generating electricity from fossil fuels"). Water and
    renewable utilities are excluded from both. The broad set is a *proxy*: an
    industry label cannot tell you a utility's generation mix.
    """
    narrow = u["Sector"].eq("Energy")
    if definition == "narrow":
        return narrow
    if definition == "broad":
        return narrow | u["Industry"].isin(FOSSIL_UTILITIES)
    raise ValueError("definition must be 'narrow' or 'broad'")


def _cap_weights(u: pd.DataFrame, keep: pd.Series) -> pd.Series:
    w = u["mcap"].where(keep, 0.0)
    return w / w.sum()


def _screen_benchmark(u):
    return _cap_weights(u, pd.Series(True, index=u.index))


def _screen_exclusion(u):
    """Values screen: no tobacco, gambling, alcohol, weapons or (narrow) fossil fuels."""
    bad = (u["Industry"].isin(VICE_INDUSTRIES + WEAPONS_INDUSTRIES)
           | fossil_mask(u, "narrow"))
    return _cap_weights(u, ~bad)


def _screen_best_in_class(u, keep_share: float = 0.5):
    """Keep the lower-risk half of every sector. Nothing is excluded by sector."""
    rk = u.groupby("Sector")["esg_risk"].rank(pct=True, method="first")
    return _cap_weights(u, rk <= keep_share)


def _screen_controversy(u):
    """Norms-based: drop Significant, High and Severe controversy levels."""
    return _cap_weights(u, ~u["controversy"].isin(["Significant", "High", "Severe"]))


def _screen_tilt(u, strength: float = 1.0):
    """Integration: hold everything, tilt weights away from ESG risk."""
    z = (u["esg_risk"] - u["esg_risk"].mean()) / u["esg_risk"].std()
    w = u["mcap"] * np.exp(-strength * z)
    return w / w.sum()


def _screen_leaders(u):
    """Leaders only: Negligible or Low ESG risk level."""
    return _cap_weights(u, u["risk_level"].isin(["Negligible", "Low"]))


SCREENS = {
    "Benchmark": _screen_benchmark,
    "Exclusion": _screen_exclusion,
    "Best-in-class": _screen_best_in_class,
    "Controversy screen": _screen_controversy,
    "ESG tilt": _screen_tilt,
    "Leaders only": _screen_leaders,
}


def build_fund(u: pd.DataFrame, screen: str, **kwargs) -> pd.Series:
    """Portfolio weights for one named screen. Weights sum to one."""
    return SCREENS[screen](u, **kwargs)


def build_all_funds(u: pd.DataFrame) -> pd.DataFrame:
    """One column of weights per screen."""
    return pd.DataFrame({name: fn(u) for name, fn in SCREENS.items()})


def jaccard(wa: pd.Series, wb: pd.Series) -> float:
    """Overlap of holding *sets*: |A and B| / |A or B|. Blind to weights."""
    a, b = set(wa[wa > 0].index), set(wb[wb > 0].index)
    return len(a & b) / len(a | b)


def active_share(wa: pd.Series, wb: pd.Series) -> float:
    """Half the sum of absolute weight differences. 0 = identical, 1 = disjoint."""
    idx = wa.index.union(wb.index)
    return 0.5 * float((wa.reindex(idx, fill_value=0) - wb.reindex(idx, fill_value=0)).abs().sum())


def distance_matrix(funds: pd.DataFrame, metric: str = "active_share") -> pd.DataFrame:
    """Pairwise distances between funds: active share, or 1 - Jaccard."""
    names = list(funds.columns)
    M = pd.DataFrame(0.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (active_share(funds[a], funds[b]) if metric == "active_share"
                 else 1 - jaccard(funds[a], funds[b]))
            M.loc[a, b] = M.loc[b, a] = d
    return M


def fund_profile(funds: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """What each fund actually holds, as weighted characteristics."""
    rows = {}
    for name in funds:
        w = funds[name]
        rows[name] = {
            "holdings": int((w > 0).sum()),
            "weighted ESG risk": float((w * u["esg_risk"]).sum()),
            "weighted E risk": float((w * u["e_risk"]).sum()),
            "fossil, narrow (%)": 100 * float(w[fossil_mask(u, "narrow")].sum()),
            "fossil, broad (%)": 100 * float(w[fossil_mask(u, "broad")].sum()),
            "weapons (%)": 100 * float(w[u["Industry"].isin(WEAPONS_INDUSTRIES)].sum()),
            "vice (%)": 100 * float(w[u["Industry"].isin(VICE_INDUSTRIES)].sum()),
            "high controversy (%)": 100 * float(
                w[u["controversy"].isin(["Significant", "High", "Severe"])].sum()),
        }
    return pd.DataFrame(rows).T


def claim_check(w: pd.Series, u: pd.DataFrame, claim: str = "excludes fossil fuels",
                definition: str = "broad") -> pd.DataFrame:
    """Reconcile a claim against holdings, as a ranked shortlist.

    Returns the holdings that contradict the claim under ``definition``, largest
    weight first, with the one innocent explanation this data can test: whether
    the holding would be clean under the narrower definition the manager may
    have used.
    """
    if claim != "excludes fossil fuels":
        raise NotImplementedError("only the fossil-fuel claim is implemented")
    hit = (w > 0) & fossil_mask(u, definition)
    out = u.loc[hit, ["Company", "Sector", "Industry"]].copy()
    out["weight_pct"] = 100 * w[hit]
    out["clean_under_narrow_definition"] = ~fossil_mask(u, "narrow")[hit]
    return out.sort_values("weight_pct", ascending=False)


def screen_features(u: pd.DataFrame) -> pd.DataFrame:
    """Stock-level features a regulator could observe, for reading a rule back.

    ``sector_risk_rank`` is the stock's ESG-risk percentile *within its own
    sector*. Leave it out and a best-in-class rule becomes very hard to recover:
    you can only audit a rule you have the features for.
    """
    ctl = {"None": 0, "Low": 1, "Moderate": 2, "Significant": 3, "High": 4, "Severe": 5}
    return pd.DataFrame({
        "esg_risk": u["esg_risk"], "e_risk": u["e_risk"], "s_risk": u["s_risk"],
        "g_risk": u["g_risk"], "controversy": u["controversy"].map(ctl),
        "energy_sector": fossil_mask(u, "narrow").astype(int),
        "vice_industry": u["Industry"].isin(VICE_INDUSTRIES).astype(int),
        "weapons_industry": u["Industry"].isin(WEAPONS_INDUSTRIES).astype(int),
        "sector_risk_rank": u.groupby("Sector")["esg_risk"].rank(pct=True, method="first"),
    }, index=u.index)


# ==========================================================================
# 5. Plotly figures
# ==========================================================================

COLOURS = {"labelled": "#6AAF23", "default": "#7A8794", "worse": "#C0392B",
           "navy": "#123F69", "orange": "#E07B00", "grid": "#D8DEE6"}


def _layout(fig, title, height=460, **kw):
    fig.update_layout(title=title, template="plotly_white", height=height,
                      margin=dict(l=10, r=20, t=60, b=40), **kw)
    return fig


def fig_pairs(pairs: pd.DataFrame):
    """Dumbbell: each ethical option against its own provider's default."""
    import plotly.graph_objects as go
    p = pairs.sort_values("gap", ascending=False).reset_index(drop=True)
    lab = [short_name(f, o) for f, o in zip(p["fund"], p["option"])]
    fig = go.Figure()
    for i, r in p.iterrows():
        c = COLOURS["worse"] if r.gap > 0 else COLOURS["labelled"]
        fig.add_trace(go.Scatter(x=[r["default"], r["ethical"]], y=[lab[i]] * 2,
                                 mode="lines", line=dict(color=c, width=3),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=p["default"], y=lab, mode="markers", name="provider's own default",
                             marker=dict(color=COLOURS["default"], size=11),
                             hovertemplate="default %{x:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=p["ethical"], y=lab, mode="markers", showlegend=False,
        marker=dict(color=[COLOURS["worse"] if g > 0 else COLOURS["labelled"] for g in p["gap"]],
                    size=11),
        customdata=p["gap"], hovertemplate="ethical %{x:.2f}% (gap %{customdata:+.2f} pp)<extra></extra>"))
    for nm, col in [("ethical option, cleaner", COLOURS["labelled"]),
                    ("ethical option, dirtier", COLOURS["worse"])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                 marker=dict(color=col, size=11)))
    fig.update_xaxes(title="portfolio in companies of concern (% of analysed FUM)")
    return _layout(fig, "What does the label buy you? Same provider, same year", height=560)


def fig_separation(sep: pd.DataFrame):
    """AUC by dimension, coloured by whether the category is sector- or conduct-based."""
    import plotly.graph_objects as go
    s = sep.sort_values("auc")
    col = {"sector": COLOURS["navy"], "mixed": COLOURS["orange"], "conduct": COLOURS["worse"]}
    fig = go.Figure(go.Bar(
        x=s["auc"], y=s["dimension"], orientation="h",
        marker_color=[col[k] for k in s["kind"]],
        text=[f"{a:.3f}  ({k})" for a, k in zip(s["auc"], s["kind"])], textposition="outside",
        hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLOURS["default"],
                  annotation_text="coin flip", annotation_position="bottom right")
    fig.update_xaxes(range=[0, 1.18], title="AUC of 'lower exposure means labelled'")
    fig.update_yaxes(ticksuffix="   ", automargin=True)
    return _layout(fig, "Which categories does the label actually separate?", height=400)


def fig_pca_map(df: pd.DataFrame, labels: np.ndarray, title: str = ""):
    """Options on the first two principal components.

    Colour = cluster, ordered from lowest to highest mean total exposure, so the
    cleanest cluster is always green. Symbol = the label (circle = ethical
    option, cross = default), which the clustering never saw.
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    Zs = scale(df, "standard")
    pca = PCA(n_components=2, random_state=42).fit(Zs)
    Z = pca.transform(Zs)
    y = is_labelled(df)
    labels = np.asarray(labels)
    order = sorted(np.unique(labels), key=lambda c: df["total"].to_numpy()[labels == c].mean())
    palette = ["#6AAF23", "#E07B00", "#7A8794", "#2E7EBB", "#9467bd", "#8c564b"]
    fig = go.Figure()
    for rank, c in enumerate(order):
        m = labels == c
        mean_tot = df["total"].to_numpy()[m].mean()
        fig.add_trace(go.Scatter(
            x=Z[m, 0], y=Z[m, 1], mode="markers",
            name=f"cluster {rank + 1}: {m.sum()} options, mean {mean_tot:.1f}%",
            marker=dict(color=palette[rank % len(palette)], size=12,
                        symbol=np.where(y[m] == 1, "circle", "x"),
                        line=dict(color="white", width=1)),
            text=df.loc[m, "name"], hovertemplate="%{text}<extra></extra>"))
    for nm, sym in [("\u25cf ethical option", "circle"), ("\u2715 MySuper default", "x")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                 marker=dict(color="#4A5568", size=11, symbol=sym)))
    ev = pca.explained_variance_ratio_
    fig.update_xaxes(title=f"PC1 \u2014 general harm exposure ({ev[0]:.0%})")
    fig.update_yaxes(title=f"PC2 \u2014 human rights and animal welfare ({ev[1]:.0%})")
    return _layout(fig, title or "The products, without their labels", height=520)


def fig_grid(grid: pd.DataFrame):
    """Heatmap of ARI across the sixteen pipelines."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=grid.values.astype(float), x=[s.replace("minmax", "min\u2013max") for s in grid.columns],
        y=[m.replace("kmeans", "k-means") for m in grid.index],
        colorscale="RdYlGn", zmin=-0.1, zmax=1.0, colorbar=dict(title="ARI"),
        texttemplate="%{z:.2f}",
        hovertemplate="%{y}, %{x}: ARI %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Adjusted Rand index against the label, sixteen pipelines", height=420)


def fig_dendrogram(df: pd.DataFrame, method: str = "ward"):
    """Hierarchical tree of the options, leaves coloured by label."""
    import plotly.figure_factory as ff
    from scipy.cluster.hierarchy import linkage
    y = is_labelled(df)
    names = [short_name(f, o) for f, o in zip(df["fund"], df["option"])]
    fig = ff.create_dendrogram(scale(df, "standard"), labels=names, orientation="left",
                               linkagefun=lambda x: linkage(x, method=method),
                               color_threshold=4.2)
    lookup = dict(zip(names, y))
    ticks = fig.layout.yaxis.ticktext
    fig.update_yaxes(ticktext=[
        f"<span style='color:{COLOURS['labelled'] if lookup[t] else COLOURS['navy']}'>"
        f"{'<b>' + t + '</b>' if lookup[t] else t}</span>" for t in ticks])
    fig.update_xaxes(title=f"{method} linkage distance")
    return _layout(fig, "The super universe as a tree (bold green = ethical option)",
                   height=760, showlegend=False)


def fig_disclosure(df: pd.DataFrame):
    """Measured exposure against disclosure coverage."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for ft, col, sym in [("Ethical", COLOURS["labelled"], "circle"),
                         ("MySuper", COLOURS["default"], "x")]:
        g = df[df["fund_type"] == ft]
        fig.add_trace(go.Scatter(x=g["disclosure_pct"], y=g["total"], mode="markers",
                                 name="ethical option" if ft == "Ethical" else "MySuper default",
                                 marker=dict(color=col, size=12, symbol=sym),
                                 text=g["name"], hovertemplate="%{text}<br>disclosed %{x:.1f}%"
                                 "<br>exposure %{y:.2f}%<extra></extra>"))
    fig.update_xaxes(title="share of the option's value disclosed and analysed (%)")
    fig.update_yaxes(title="measured exposure to companies of concern (%)")
    return _layout(fig, "Does looking harder find more?", height=460)


def fig_matrix(M: pd.DataFrame, title: str, fmt: str = ".2f", zmax: float | None = None):
    """Annotated heatmap for a fund-by-fund matrix."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=M.values, x=list(M.columns), y=list(M.index), colorscale="Blues",
        zmin=0, zmax=zmax if zmax is not None else float(np.nanmax(M.values)),
        text=M.values, texttemplate="%{text:" + fmt + "}",
        hovertemplate="%{y} vs %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=460)
