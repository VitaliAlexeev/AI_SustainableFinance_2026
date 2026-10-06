"""
esgprice.py -- helper module for Lecture 10 of AI-integrated Sustainable Finance (25881)
"Portfolio Optimisation with AI and ESG Criteria: the price of a principle"

Everything the Lecture 10 live demo (and the Lecture 10 labs) needs:

* data loading through a local-clone -> cache -> repository chain, so notebooks
  never depend on a third-party server;
* the 349-stock universe used on the slides (Lab 9b's 368 rated names that also
  have complete 2013--2018 prices);
* the six Lab 9b screens;
* the tracking-error optimiser, its frontier, the floor/penalty pair and the
  placebo test;
* Plotly figures in the deck's palette.

Every number on the Lecture 10 slides can be reproduced with these functions.

Conventions
-----------
* ESG *risk* (Sustainalytics-style) is LOWER-is-better, so an ESG target is a
  cap: ``r @ w <= t``. Scores that are higher-is-better need a floor instead.
* Tracking error (TE) is ex ante and annualised, against the cap-weighted
  benchmark.
* Expected returns are implied by the benchmark, so a portfolio's Sharpe ratio
  divided by the benchmark's equals its correlation with the benchmark.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "VitaliAlexeev/AI_SustainableFinance_2026"
REF = "main"
CACHE = Path("_cache")

NAVY, GREEN, ORANGE, RED, GREY, BAND = "#123F69", "#4E8F1D", "#E38B29", "#B22222", "#8A8A8A", "#E2EBF4"
FUND_ORDER = ["Benchmark", "Exclusion", "Controversy screen", "Best-in-class", "ESG tilt", "Leaders only"]

VICE = ("Tobacco", "Resorts & Casinos", "Beverages - Brewers", "Beverages - Wineries & Distilleries")
WEAPONS = ("Aerospace & Defense",)


# ==========================================================================
# 1. Data
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
        with urllib.request.urlopen(url, timeout=180) as r:
            cached.write_bytes(r.read())
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(f"could not fetch {url}\n  Underlying error: {exc}") from exc
    if verbose:
        print(f"{cached.stat().st_size / 1e6:.1f} MB, cached")
    return cached


@dataclass
class Universe:
    U: pd.DataFrame          # one row per stock: sector, industry, ESG risk, controversy, market cap
    R: pd.DataFrame          # daily log returns
    Sigma: np.ndarray        # annualised covariance (Ledoit-Wolf)
    L: np.ndarray            # Cholesky factor of Sigma
    b: np.ndarray            # cap-weighted benchmark
    r: np.ndarray            # ESG risk (lower is better)
    shrinkage: float

    @property
    def N(self) -> int:
        return len(self.U)


def load_universe(verbose: bool = True) -> Universe:
    """The 349 S&P 500 stocks used on the Lecture 10 slides."""
    from sklearn.covariance import LedoitWolf
    cols = ["Symbol", "Company", "Sector", "Industry", "Total ESG Risk", "Environment Risk",
            "Social Risk", "Governance Risk", "Controversy Level", "ESG Risk Level"]
    esg = pd.read_csv(resolve_data("sp500_esg.csv", verbose), usecols=cols)
    cap = pd.read_csv(resolve_data("constituents-financials.csv", verbose), usecols=["Symbol", "Market Cap"])
    u = (esg.merge(cap, on="Symbol", how="inner")
            .dropna(subset=["Total ESG Risk", "Market Cap", "Sector", "Industry"])
            .rename(columns={"Total ESG Risk": "esg_risk", "Environment Risk": "e_risk",
                             "Social Risk": "s_risk", "Governance Risk": "g_risk",
                             "Controversy Level": "controversy", "Market Cap": "mcap",
                             "ESG Risk Level": "risk_level"})
            .drop_duplicates("Symbol").set_index("Symbol").sort_index())
    u["controversy"] = u["controversy"].str.replace(" Controversy Level", "", regex=False)
    raw = pd.read_csv(resolve_data("all_stocks_5yr.csv", verbose), parse_dates=["date"])
    px = raw.pivot_table(index="date", columns="Name", values="close")
    px = px.loc[:, px.notna().all()]                          # complete histories only (survivorship!)
    keep = [s for s in u.index if s.replace("-", ".") in px.columns]
    U = u.loc[keep].copy()
    R = np.log(px[[s.replace("-", ".") for s in keep]]).diff().dropna()
    R.columns = keep
    lw = LedoitWolf().fit(R.values)
    Sigma = lw.covariance_ * 252
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(len(keep)))
    b = (U["mcap"] / U["mcap"].sum()).values
    if verbose:
        print(f"\n{len(keep)} stocks with ESG ratings, market caps and complete prices "
              f"({R.index[0].date()} to {R.index[-1].date()}, {len(R)} daily returns); "
              f"{len(u) - len(keep)} rated names dropped for missing prices.")
    return Universe(U=U, R=R, Sigma=Sigma, L=L, b=b, r=U["esg_risk"].values.astype(float),
                    shrinkage=float(lw.shrinkage_))


def load_msci(uni: Universe, as_of: int = 20180201, verbose: bool = True) -> pd.Series:
    """MSCI industry-adjusted ESG score (0-10, higher is better), aligned to the universe."""
    m = pd.read_csv(resolve_data("MSCI_ESG_US_2018.csv", verbose), low_memory=False,
                    usecols=["ISSUER_TICKER", "AS_OF_DATE", "INDUSTRY_ADJUSTED_SCORE"])
    m = m[m["AS_OF_DATE"] == as_of].drop_duplicates("ISSUER_TICKER")
    m["sym"] = m["ISSUER_TICKER"].astype(str).str.replace(".", "-", regex=False)
    return m.set_index("sym")["INDUSTRY_ADJUSTED_SCORE"].reindex(uni.U.index)


# ==========================================================================
# 2. Screens (identical to Lab 9b) and portfolio statistics
# ==========================================================================

def _cap_weights(uni: Universe, keep) -> np.ndarray:
    w = np.where(np.asarray(keep), uni.U["mcap"].values, 0.0)
    return w / w.sum()


def screens(uni: Universe, keep_share: float = 0.5, strength: float = 1.0) -> pd.DataFrame:
    """The six Lab 9b funds, rebuilt on the priced universe. One column of weights per fund."""
    U = uni.U
    rk = U.groupby("Sector")["esg_risk"].rank(pct=True, method="first")
    z = (U["esg_risk"] - U["esg_risk"].mean()) / U["esg_risk"].std()
    tilt = U["mcap"].values * np.exp(-strength * z.values)
    funds = {
        "Benchmark": uni.b,
        "Exclusion": _cap_weights(uni, ~(U["Industry"].isin(VICE + WEAPONS) | U["Sector"].eq("Energy"))),
        "Controversy screen": _cap_weights(uni, ~U["controversy"].isin(["Significant", "High", "Severe"])),
        "Best-in-class": _cap_weights(uni, rk <= keep_share),
        "ESG tilt": tilt / tilt.sum(),
        "Leaders only": _cap_weights(uni, U["risk_level"].isin(["Negligible", "Low"])),
    }
    return pd.DataFrame(funds, index=U.index)


def tracking_error(uni: Universe, w) -> float:
    d = np.asarray(w) - uni.b
    return float(np.sqrt(d @ uni.Sigma @ d))


def profile(uni: Universe, w) -> dict:
    """What a portfolio costs and what it holds."""
    w = np.asarray(w, dtype=float)
    vol = np.sqrt(w @ uni.Sigma @ w)
    vb = np.sqrt(uni.b @ uni.Sigma @ uni.b)
    tech = uni.U["Sector"].eq("Technology").values
    return {"holdings": int((w > 1e-6).sum()),
            "ESG risk": float(w @ uni.r),
            "TE (%)": 100 * tracking_error(uni, w),
            "Sharpe / benchmark": float(w @ uni.Sigma @ uni.b / (vol * vb)),
            "effective holdings": float(1 / np.sum(w ** 2)),
            "technology (%)": 100 * float(w[tech].sum())}


def profiles(uni: Universe, funds: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: profile(uni, funds[k].values) for k in funds}).T


def sector_weights(uni: Universe, w) -> pd.Series:
    return pd.Series(np.asarray(w), index=uni.U.index).groupby(uni.U["Sector"]).sum()


# ==========================================================================
# 3. The optimiser
# ==========================================================================

def min_te(uni: Universe, target: float, score=None, floor: bool = False, extra=None):
    """Lowest-tracking-error long-only portfolio meeting an ESG target.

    ``score`` defaults to ESG risk with a cap (``score @ w <= target``); set
    ``floor=True`` for higher-is-better scores (``score @ w >= target``).
    Returns ``(weights, shadow_price)``; ``(None, None)`` if infeasible.
    """
    import cvxpy as cp
    s = uni.r if score is None else np.asarray(score, dtype=float)
    w = cp.Variable(uni.N)
    esg = (s @ w >= target) if floor else (s @ w <= target)
    cons = [cp.sum(w) == 1, w >= 0, esg] + (extra(w) if callable(extra) else (extra or []))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(uni.L.T @ (w - uni.b))), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None, None
    x = np.clip(w.value, 0, None)
    return x / x.sum(), float(esg.dual_value)


def penalty_portfolio(uni: Universe, lam: float, score=None):
    """The soft version: minimise TE^2 + lam * (score @ w)."""
    import cvxpy as cp
    s = uni.r if score is None else np.asarray(score, dtype=float)
    w = cp.Variable(uni.N)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(uni.L.T @ (w - uni.b)) + lam * (s @ w)),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.CLARABEL)
    x = np.clip(w.value, 0, None)
    return x / x.sum()


def cut_target(uni: Universe, cut: float, score=None) -> float:
    """Target equal to the benchmark's ESG risk reduced by ``cut`` (0.20 = 20%)."""
    s = uni.r if score is None else np.asarray(score, dtype=float)
    return (1 - cut) * float(uni.b @ s)


def frontier(uni: Universe, n: int = 40) -> pd.DataFrame:
    """Minimum TE at n ESG targets, from the benchmark's level to near the lowest attainable."""
    top, rmin = float(uni.b @ uni.r), float(uni.r.min())
    rows = []
    for t in np.linspace(top, rmin + 0.15 * (top - rmin), n):
        w, dual = min_te(uni, float(t))
        p = profile(uni, w)
        te = p["TE (%)"] / 100
        p.update({"target": float(t), "shadow price": dual,
                  "marginal TE per point (bp)": 1e4 * dual / (2 * te) if te > 1e-6 else 0.0})
        rows.append(p)
    return pd.DataFrame(rows)


def placebo(uni: Universe, cut: float = 0.20, n: int = 30, within_sector: bool = False,
            seed: int = 20261006) -> np.ndarray:
    """TE (%) needed for the same cut when ESG scores are shuffled between firms."""
    rng = np.random.default_rng(seed)
    sect = uni.U["Sector"].values
    out = []
    for _ in range(n):
        s = uni.r.copy()
        if within_sector:
            for g in np.unique(sect):
                idx = np.where(sect == g)[0]
                s[idx] = rng.permutation(uni.r[idx])
        else:
            s = rng.permutation(uni.r)
        w, _ = min_te(uni, cut_target(uni, cut, s), score=s)
        if w is not None:
            out.append(100 * tracking_error(uni, w))
    return np.array(out)


# ==========================================================================
# 4. Figures (Plotly, deck palette)
# ==========================================================================

def _layout(fig, title, height=440, **kw):
    fig.update_layout(template="plotly_white", height=height, title=dict(text=f"<b>{title}</b>", x=0.01),
                      font=dict(size=12), margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.18), **kw)
    return fig


def fig_frontier(uni: Universe, fr: pd.DataFrame, funds: pd.DataFrame, mark=None):
    """TE against ESG risk: the optimiser's frontier and the rule-based funds."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr["TE (%)"], y=fr["ESG risk"], mode="lines", line=dict(color=NAVY, width=3),
                             name="optimiser: lowest TE for each ESG level"))
    P = profiles(uni, funds)
    for k in [f for f in FUND_ORDER if f != "Benchmark"]:
        p = P.loc[k]
        fig.add_trace(go.Scatter(x=[p["TE (%)"]], y=[p["ESG risk"]], mode="markers+text", text=[k],
                                 textposition="middle right", marker=dict(symbol="square", size=11, color=ORANGE),
                                 showlegend=False, hovertemplate=f"{k}<br>TE %{{x:.2f}}%<br>ESG %{{y:.2f}}<extra></extra>"))
    if mark is not None:
        p = profile(uni, mark)
        fig.add_trace(go.Scatter(x=[p["TE (%)"]], y=[p["ESG risk"]], mode="markers", name="your portfolio",
                                 marker=dict(size=15, color=GREEN, line=dict(color="white", width=2))))
    fig.update_yaxes(autorange="reversed", title="portfolio ESG risk (lower = better, up)")
    fig.update_xaxes(title="tracking error vs benchmark (% a year)")
    return _layout(fig, "Every screen sits inside the frontier")


def fig_price_curve(fr: pd.DataFrame, bench_esg: float):
    import plotly.graph_objects as go
    cut = 100 * (1 - fr["ESG risk"] / bench_esg)
    fig = go.Figure(go.Scatter(x=cut[1:], y=fr["marginal TE per point (bp)"][1:], mode="lines+markers",
                               line=dict(color=NAVY, width=3), marker=dict(size=5)))
    fig.update_xaxes(title="cut in portfolio ESG risk (%)"); fig.update_yaxes(title="bp of TE per extra point")
    return _layout(fig, "Each extra point of ESG costs more than the last", height=380)


def fig_placebo(real: float, full: np.ndarray, within: np.ndarray | None = None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=full, name="shuffled across all firms", marker_color=GREY, opacity=0.8, nbinsx=20))
    if within is not None:
        fig.add_trace(go.Histogram(x=within, name="shuffled within sectors", marker_color=NAVY, opacity=0.7, nbinsx=20))
    fig.add_vline(x=real, line=dict(color=ORANGE, width=4), annotation_text=f"real scores {real:.2f}%",
                  annotation_position="top left")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="minimum TE for a 20% cut in ESG risk (% a year)"); fig.update_yaxes(title="count")
    return _layout(fig, "Placebo: is ESG special, or is any constraint this dear?", height=400)


def fig_sectors(uni: Universe, w, title="Sector weights against the benchmark"):
    import plotly.graph_objects as go
    sw, sb = 100 * sector_weights(uni, w), 100 * sector_weights(uni, uni.b)
    fig = go.Figure([go.Bar(y=sb.index, x=sb.values, orientation="h", name="benchmark", marker_color=GREY),
                     go.Bar(y=sw.index, x=sw.values, orientation="h", name="portfolio", marker_color=ORANGE)])
    fig.update_layout(barmode="group"); fig.update_xaxes(title="weight (%)")
    return _layout(fig, title, height=460)
