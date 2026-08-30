"""
esgts6.py -- shared helpers for AI-integrated Sustainable Finance (25881)
Lecture 6 labs: Time-Series Analysis and Predictive Models with ESG Data.

Loading, plotting (Plotly only), spectral tools, benchmark forecasts and
backtesting. Import once at the top of Lecture06a and Lecture06b.

    import esgts6 as E
    E.use_house_theme()
    co2 = E.load_mauna_loa()

Data resolution order for every loader:
    1. ./data/<file>            (already downloaded, or the teaching repo clone)
    2. ~/.cache/esgts6/<file>   (cached from a previous run in this environment)
    3. the teaching repo on GitHub, which is then cached to (2)

So the first run needs a network; every run after that does not.

British spelling in prose. American spellings appear only where they are
library or API keywords (``color``, ``deseasonalized``, ``seasonal_decompose``).
"""
from __future__ import annotations

import os
import pathlib
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

__all__ = [
    "REPO", "load", "load_mauna_loa", "load_owid", "load_owid_vintage",
    "load_market", "VINTAGE_DATES", "use_house_theme", "COLOURS",
    "periodogram", "top_peaks", "fourier_terms", "annual_to_monthly",
    "naive", "drift", "seasonal_naive", "seasonal_naive_drift",
    "design_row", "build_training", "recursive_forecast",
    "mase", "rmse", "mae", "rolling_origin",
    "plot_series", "plot_periodogram", "plot_stl", "plot_forecast",
]

# --------------------------------------------------------------------------- #
# Where the data lives
# --------------------------------------------------------------------------- #
REPO = ("https://raw.githubusercontent.com/VitaliAlexeev/"
        "AI_SustainableFinance_2026/main/data")

LOCAL = pathlib.Path("data")
CACHE = pathlib.Path.home() / ".cache" / "esgts6"

VINTAGE_DATES = {"2023": "2023-11-08", "2024": "2024-11-22",
                 "2025": "2025-11-13", "2026": "2026-06-02"}

FILES = {
    "mlo": "co2_mm_mlo_noaa.csv",
    "mlo_mirror": "co2_mm_mlo.csv",
    "owid": "owid_co2_current.csv",
    "owid_2023": "owid_v2023.csv",
    "owid_2024": "owid_v2024.csv",
    "owid_2025": "owid_v2025.csv",
    "sp500": "all_stocks_5yr.csv",
}


def load(name, **read_csv_kwargs) -> pd.DataFrame:
    """Resolve a data file to a DataFrame, trying local, cache, then GitHub.

    ``name`` is either a short key from ``FILES`` or a bare filename.
    Any extra keyword arguments are passed straight to ``pd.read_csv`` --
    use ``usecols`` on the large panels to stay inside a 2 GB Binder.
    """
    fname = FILES.get(name, name)

    for candidate in (LOCAL / fname, CACHE / fname):
        if candidate.exists():
            return pd.read_csv(candidate, **read_csv_kwargs)

    url = f"{REPO}/{fname}"
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        target = CACHE / fname
        print(f"downloading {fname} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, target)
        print(f"cached to {target}")
        return pd.read_csv(target, **read_csv_kwargs)
    except (urllib.error.URLError, OSError) as exc:
        raise FileNotFoundError(
            f"Could not find '{fname}' locally and could not download it.\n"
            f"  Looked in: {LOCAL / fname}, {CACHE / fname}\n"
            f"  Tried:     {url}\n"
            f"  Error:     {exc}\n\n"
            "If you are offline, clone the teaching repository and run this "
            "notebook from its root so that ./data/ is visible."
        ) from exc


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_mauna_loa(mirror=False) -> pd.DataFrame:
    """Monthly mean atmospheric CO2 at Mauna Loa (ppm), 1958-present.

    Returns a monthly-indexed frame with columns:
        co2             monthly mean, the series we model
        deseasonalised  NOAA's own trend estimate, useful as a sanity check
        measured        False where NOAA has no station metadata

    NOAA flags the Scripps era (March 1958 to April 1974) with ndays = -1
    because the metadata was never recorded -- those months are *not*
    interpolated, and treating them as missing is a common mistake.
    """
    if mirror:
        raw = load("mlo_mirror", skiprows=1, header=None,
                   names=["date", "decimal_date", "average", "deseasonalized",
                          "ndays", "sdev", "unc"])
        idx = pd.PeriodIndex(raw["date"].astype(str), freq="M").to_timestamp()
    else:
        raw = load("mlo")
        idx = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))

    df = pd.DataFrame({
        "co2": raw["average"].astype(float),
        "deseasonalised": raw["deseasonalized"].astype(float),
        "measured": raw["ndays"].astype(float) >= 0,
    })
    df.index = idx
    df.index.name = "date"
    return df.asfreq("MS")


OWID_SMALL = ["country", "year", "co2", "cement_co2", "co2_per_capita",
              "population", "gdp"]


def load_owid(columns=OWID_SMALL) -> pd.DataFrame:
    """Our World in Data annual CO2 panel, current release."""
    return load("owid", usecols=lambda c: c in set(columns), low_memory=False)


def load_owid_vintage(tag, columns=("country", "year", "co2")) -> pd.DataFrame:
    """A dated snapshot of the OWID panel, as published on that date.

    ``tag`` in {'2023', '2024', '2025', '2026'}. These are genuine historical
    commits of owid-co2-data.csv, not simulated revisions. '2026' is the
    current release.
    """
    key = "owid" if str(tag) == "2026" else f"owid_{tag}"
    return load(key, usecols=lambda c: c in set(columns), low_memory=False)


def load_market(ticker="NEE") -> pd.Series:
    """Daily close for one ticker from the standard S&P 500 five-year panel."""
    d = load("sp500", usecols=["date", "close", "Name"], parse_dates=["date"])
    s = d.loc[d["Name"] == ticker].set_index("date")["close"].sort_index()
    if s.empty:
        raise KeyError(f"'{ticker}' is not in the panel. Try NEE, XOM, DUK, SO.")
    s.name = ticker
    return s


# --------------------------------------------------------------------------- #
# Plotly house theme
# --------------------------------------------------------------------------- #
COLOURS = {
    "blue": "#123F69", "green": "#8CC63E", "dark_green": "#1F5E20",
    "orange": "#E07B24", "red": "#C0392B", "purple": "#6C3D8C",
    "grey": "#7A8B99", "light": "#E2EBF4",
}
SEQUENCE = [COLOURS["blue"], COLOURS["orange"], COLOURS["dark_green"],
            COLOURS["red"], COLOURS["purple"], COLOURS["grey"]]


def use_house_theme(default=True):
    """Register and optionally activate the lecture's Plotly template."""
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["esgts6"] = go.layout.Template(
        layout=dict(
            colorway=SEQUENCE,
            font=dict(family="Helvetica, Arial, sans-serif", size=12,
                      color="#22313F"),
            title=dict(font=dict(size=15), x=0.0, xanchor="left"),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(gridcolor="#D8E1E8", zerolinecolor="#B7C4CE",
                       linecolor="#4A5C6A", ticks="outside", ticklen=4),
            yaxis=dict(gridcolor="#D8E1E8", zerolinecolor="#B7C4CE",
                       linecolor="#4A5C6A", ticks="outside", ticklen=4),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            margin=dict(l=60, r=25, t=55, b=50),
            hovermode="x unified",
        )
    )
    if default:
        pio.templates.default = "plotly_white+esgts6"
    return pio.templates["esgts6"]


# --------------------------------------------------------------------------- #
# Spectral tools
# --------------------------------------------------------------------------- #
_ORDERS = {"none": None, "mean": 0, "linear": 1, "quadratic": 2, "cubic": 3}


def periodogram(x, detrend="linear"):
    """One-sided periodogram I(f) = |X(f)|^2 / T for unit sampling.

    Returns ``(freq, power)`` with the zero frequency dropped.

    ``detrend`` removes a polynomial trend first. This is not cosmetic: a
    trend is a very low-frequency component, and leaving it in floods the
    low-frequency bins and buries any seasonal peak. Mauna Loa needs
    ``'quadratic'`` because its growth rate is itself rising.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 4:
        raise ValueError("need at least four observations")
    order = _ORDERS.get(detrend, None)
    if order is not None:
        t = np.arange(T)
        x = x - np.polyval(np.polyfit(t, x, order), t)
    power = np.abs(np.fft.rfft(x)) ** 2 / T
    freq = np.fft.rfftfreq(T, d=1.0)
    return freq[1:], power[1:]


def top_peaks(freq, power, k=3, min_period=2.0) -> pd.DataFrame:
    """The k strongest periodogram peaks, reported as periods.

    ``share_of_var`` is the fraction of the detrended series' variance sitting
    in that single frequency bin -- the number that separates a real cycle
    from a flat spectrum.
    """
    period = 1.0 / freq
    keep = period >= min_period
    f, p, per = freq[keep], power[keep], period[keep]
    idx = np.argsort(p)[::-1][:k]
    return pd.DataFrame({
        "period": per[idx], "freq": f[idx], "power": p[idx],
        "share_of_var": p[idx] / p.sum(),
    }).reset_index(drop=True)


def fourier_terms(n, period, K, start=0) -> pd.DataFrame:
    """K sine/cosine pairs at a given seasonal period.

    These are the deterministic-seasonality features the periodogram tells you
    to build: find the dominant period, then hand those harmonics to a model.
    """
    t = np.arange(start, start + n, dtype=float)
    out = {}
    for k in range(1, K + 1):
        out[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(out)


def annual_to_monthly(annual: pd.Series, how="ffill") -> pd.Series:
    """Expand an annual series to monthly, the way a data pipeline would.

    ``how='ffill'`` repeats each year's value across its twelve months.
    ``how='interpolate'`` draws a straight line between mid-year points and
    is the one that introduces look-ahead: January 2023 is filled using the
    2024 figure.
    """
    years = annual.dropna().index.astype(int)
    idx = pd.date_range(f"{years.min()}-01-01", f"{years.max()}-12-01",
                        freq="MS")
    if how == "ffill":
        out = pd.Series(index=idx, dtype=float)
        for yr, v in annual.dropna().items():
            out.loc[f"{int(yr)}-01-01":f"{int(yr)}-12-01"] = v
        return out
    if how == "interpolate":
        mid = annual.dropna().copy()
        mid.index = pd.to_datetime([f"{int(y)}-07-01" for y in mid.index])
        return (mid.reindex(idx.union(mid.index))
                   .interpolate("time").reindex(idx))
    raise ValueError("how must be 'ffill' or 'interpolate'")


# --------------------------------------------------------------------------- #
# Benchmark forecasts
# --------------------------------------------------------------------------- #
def naive(history, h):
    """Repeat the last observation."""
    return np.repeat(np.asarray(history, float)[-1], h)


def drift(history, h):
    """Last value plus the average historical slope."""
    v = np.asarray(history, float)
    slope = (v[-1] - v[0]) / (len(v) - 1)
    return v[-1] + slope * np.arange(1, h + 1)


def seasonal_naive(history, h, season=12):
    """Repeat the value from one season ago."""
    v = np.asarray(history, float)
    return np.array([v[-season + (j % season)] for j in range(h)])


def seasonal_naive_drift(history, h, season=12):
    """Seasonal naive plus the average year-on-year change.

    On a series with both a trend and a seasonal cycle this is the benchmark
    that actually deserves to be beaten. Plain seasonal naive ignores the
    trend entirely and is too easy a target.
    """
    v = np.asarray(history, float)
    if len(v) <= 2 * season:
        return seasonal_naive(v, h, season)
    yoy = np.mean(np.diff(v[::season]))
    return seasonal_naive(v, h, season) + yoy


# --------------------------------------------------------------------------- #
# Reduction to supervised learning
# --------------------------------------------------------------------------- #
def design_row(values, t_index, lags=(1, 2, 3, 12), K=2, season=12,
               with_trend=True) -> dict:
    """One row of features, built from a history buffer at absolute time t."""
    row = {f"lag_{L}": values[-L] for L in lags}
    for k in range(1, K + 1):
        row[f"sin_{k}"] = np.sin(2 * np.pi * k * t_index / season)
        row[f"cos_{k}"] = np.cos(2 * np.pi * k * t_index / season)
    if with_trend:
        row["t"] = float(t_index)
    return row


def build_training(series, t0=0, lags=(1, 2, 3, 12), K=2, season=12,
                   with_trend=True):
    """Reduce a series to (X, y) for any scikit-learn regressor.

    Row i uses only values strictly before i, so no row can see its own
    target. That is the whole of the look-ahead defence at this stage.
    """
    v = np.asarray(series, float)
    maxlag = max(lags)
    X, Y = [], []
    for i in range(maxlag, len(v)):
        X.append(design_row(v[:i], t0 + i, lags, K, season, with_trend))
        Y.append(v[i])
    return pd.DataFrame(X), np.asarray(Y)


def recursive_forecast(model, history, t_start, h, lags=(1, 2, 3, 12), K=2,
                       season=12, with_trend=True):
    """Multi-step forecast by feeding each prediction back as an input.

    Simple and coherent across horizons, but errors compound: a poor step-one
    forecast contaminates every step after it.
    """
    v = list(np.asarray(history, float))
    out = []
    for j in range(h):
        row = design_row(np.array(v), t_start + j, lags, K, season, with_trend)
        yhat = float(model.predict(pd.DataFrame([row]))[0])
        out.append(yhat)
        v.append(yhat)
    return np.array(out)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) -
                                np.asarray(y_pred, float))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) -
                                  np.asarray(y_pred, float)) ** 2)))


def mase(y_true, y_pred, y_train, season=12):
    """Mean absolute scaled error.

    The denominator is the *in-sample, one-step* seasonal naive MAE computed
    from ``y_train`` alone. Two consequences worth remembering:

    * MASE = 1 means "as accurate as a one-step seasonal naive forecast",
      which is a demanding bar for a 12-step forecast, not a trivial one.
      A 12-step seasonal naive forecast will usually score well above 1.
    * It is scale-free, so ppm, tonnes and dollars are comparable. Unlike
      MAPE it does not explode when the series passes through zero, which
      matters whenever the target is an emissions *change*.
    """
    y_train = np.asarray(y_train, float)
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return mae(y_true, y_pred) / denom


def rolling_origin(y, forecasters, h=12, season=12, first_origin=None,
                   step=12, min_train=120, verbose=False) -> pd.DataFrame:
    """Refit and score at a sequence of expanding-window origins.

    ``forecasters`` maps a name to a callable ``f(history, h) -> array``.
    ``history`` is a pandas Series ending strictly before the origin, so a
    forecaster physically cannot see its own test period.

    Returns one row per (origin, model) with MASE, MAE and RMSE. The MASE
    denominator is recomputed from each origin's own history -- using the
    full sample would leak the test period into the scale.
    """
    y = y.dropna()
    first_origin = first_origin or y.index[min_train]
    origins = pd.date_range(first_origin, y.index[-1], freq=f"{step}MS")
    rows = []
    for origin in origins:
        hist = y.loc[:origin - pd.offsets.MonthBegin(1)]
        fut = y.loc[origin:origin + pd.offsets.MonthBegin(h - 1)]
        if len(fut) < h or len(hist) < min_train:
            continue
        scale = np.mean(np.abs(hist.values[season:] - hist.values[:-season]))
        for name, fn in forecasters.items():
            try:
                pred = np.asarray(fn(hist, h), float)
            except Exception as exc:               # noqa: BLE001
                if verbose:
                    print(f"  {name} failed at {origin.date()}: {exc}")
                continue
            rows.append({
                "origin": origin, "model": name,
                "MASE": mae(fut.values, pred) / scale,
                "MAE": mae(fut.values, pred),
                "RMSE": rmse(fut.values, pred),
            })
        if verbose:
            print(f"  origin {origin.date()}: {len(hist)} train obs")
    if not rows:
        raise RuntimeError("no valid origins -- check min_train and h")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting (Plotly only)
# --------------------------------------------------------------------------- #
def plot_series(series_dict, title="", ylab="", height=380, mode="lines"):
    """Overlay one or more series on a shared time axis."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, s in series_dict.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode=mode))
    fig.update_layout(title=title, yaxis_title=ylab, height=height)
    return fig


def plot_periodogram(freq, power, title="", xlab="period (observations)",
                     mark=(), xmax=None, height=380, normalise=True):
    """Power against period on a log axis, as a share of variance."""
    import plotly.graph_objects as go
    p = power / power.sum() if normalise else power
    period = 1.0 / freq
    fig = go.Figure(go.Scatter(x=period, y=p, mode="lines",
                               name="share of variance"))
    for m in mark:
        fig.add_vline(x=m, line_dash="dot", line_color=COLOURS["grey"],
                      annotation_text=str(m), annotation_position="top")
    fig.update_xaxes(type="log", title=xlab,
                     range=[np.log10(2), np.log10(xmax or period.max())])
    fig.update_layout(title=title, height=height,
                      yaxis_title="share of variance" if normalise else "power",
                      hovermode="closest")
    return fig


def plot_stl(result, title="STL decomposition", height=680):
    """Four stacked panels from a fitted statsmodels STL result."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels = [("observed", result.observed, COLOURS["blue"]),
              ("trend", result.trend, COLOURS["dark_green"]),
              ("seasonal", result.seasonal, COLOURS["orange"]),
              ("remainder", result.resid, COLOURS["red"])]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.035,
                        subplot_titles=[p[0] for p in panels])
    for i, (name, s, colour) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name,
                                 line=dict(color=colour, width=1.2)),
                      row=i, col=1)
    fig.update_layout(title=title, height=height, showlegend=False)
    return fig


def plot_forecast(history, actual=None, forecasts=None, interval=None,
                  title="", ylab="", n_history=48, height=420):
    """History, actuals and one or more forecasts, with an optional band.

    ``interval`` is ``(lower, upper)`` as arrays aligned to the forecast index.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    h = history.iloc[-n_history:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, name="history",
                             line=dict(color=COLOURS["grey"], width=1.2)))
    if actual is not None:
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values,
                                 name="actual",
                                 line=dict(color="black", width=2)))
    if interval is not None and actual is not None:
        lo, hi = interval
        fig.add_trace(go.Scatter(
            x=list(actual.index) + list(actual.index[::-1]),
            y=list(hi) + list(lo[::-1]), fill="toself",
            fillcolor="rgba(18,63,105,0.15)", line=dict(width=0),
            name="interval", hoverinfo="skip"))
    for name, values in (forecasts or {}).items():
        idx = actual.index if actual is not None else history.index[-len(values):]
        fig.add_trace(go.Scatter(x=idx, y=values, name=name,
                                 line=dict(dash="dash", width=2)))
    fig.update_layout(title=title, yaxis_title=ylab, height=height)
    return fig
