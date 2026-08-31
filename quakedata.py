"""
quakedata.py
============
Data plumbing for the AI-integrated Sustainable Finance (25881) warm-up lab,
*Where does the Earth shake?*

The notebook is meant to run in a classroom on classroom wifi, so this module
degrades in four steps rather than failing:

    1. live      USGS real-time GeoJSON feed (no API key, no auth)
    2. cache     a copy from an earlier run, in ./_cache/
    3. mirror    a pinned snapshot in the teaching repository
    4. synthetic events generated along real plate-boundary paths

Whichever step succeeds, ``fetch_quakes()`` tells you which one it used. That
is deliberate: knowing the provenance of the data in front of you is the point
of the whole lecture this warm-up leads into.

Licence note: USGS earthquake data is US Government work and in the public
domain. Please identify your client in the User-Agent, which this module does.

Author: Vitali Alexeev, UTS Business School
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "FEEDS", "USGS_URL", "CACHE", "REPO", "REF",
    "fetch_quakes", "synthetic_quakes", "PLATE_BOUNDARIES",
    "load_capitals", "haversine_km",
]

USGS_URL = ("https://earthquake.usgs.gov/earthquakes/feed/v1.0/"
            "summary/{feed}.geojson")

# Valid magnitude/period combinations published by USGS.
FEEDS = {
    "2.5_month": "M2.5+ over the last 30 days (~3-4k events; the default)",
    "all_month": "every recorded event over 30 days (~10k; heavy US bias)",
    "4.5_month": "M4.5+ over 30 days (~600 events; sparse but global)",
    "2.5_week": "M2.5+ over 7 days (~800 events)",
    "significant_month": "only events flagged significant (~10 events)",
}

CACHE = Path("_cache")
REPO = "VitaliAlexeev/AI_SustainableFinance_2026"
REF = "main"

_UA = "UTS-25881-teaching-notebook (vitali.alexeev@uts.edu.au)"


# --------------------------------------------------------------------------
# Live fetch
# --------------------------------------------------------------------------

def _parse_geojson(payload: dict) -> pd.DataFrame:
    """Flatten a USGS FeatureCollection into a tidy DataFrame.

    Coordinates arrive as ``[longitude, latitude, depth]`` -- note the order,
    which is the GeoJSON convention and the reverse of how most people say it.
    Getting this backwards puts every earthquake in the wrong hemisphere, and
    the map will look plausible enough that you might not notice.
    """
    feats = payload.get("features", [])
    rows = []
    for f in feats:
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        coords = g.get("coordinates") or [None, None, None]
        rows.append({
            "id": f.get("id"),
            "time": p.get("time"),
            "place": p.get("place"),
            "mag": p.get("mag"),
            "magType": p.get("magType"),
            "lon": coords[0],
            "lat": coords[1],
            "depth_km": coords[2],
            "net": p.get("net"),
            "type": p.get("type"),
            "tsunami": p.get("tsunami"),
            "sig": p.get("sig"),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df = df.dropna(subset=["lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    return df.reset_index(drop=True)


def _try_live(feed: str, timeout: int = 30) -> pd.DataFrame | None:
    url = USGS_URL.format(feed=feed)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return _parse_geojson(payload)


def _try_mirror(feed: str, timeout: int = 30) -> pd.DataFrame | None:
    url = (f"https://raw.githubusercontent.com/{REPO}/{REF}"
           f"/data/processed/usgs_{feed}.csv")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        from io import BytesIO
        return pd.read_csv(BytesIO(r.read()), parse_dates=["time"])


def fetch_quakes(feed: str = "2.5_month",
                 allow_live: bool = True,
                 refresh: bool = False,
                 quiet: bool = False) -> tuple[pd.DataFrame, dict]:
    """Get earthquakes, falling back gracefully.

    Returns ``(DataFrame, provenance)`` where ``provenance`` records which of
    the four sources actually supplied the data, when, and from where.

    Always read the provenance. A notebook that silently substitutes synthetic
    data for real data has told you something false about the world.
    """
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"usgs_{feed}.csv"
    prov = {"feed": feed, "requested_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")}

    def say(msg):
        if not quiet:
            print(msg)

    # 1. cache, unless a refresh was asked for
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["time"])
        prov.update(source="cache", path=str(cache_path), n=len(df))
        say(f"[cache]  {len(df):,} events from {cache_path}")
        return df, prov

    # 2. live USGS
    if allow_live:
        try:
            df = _try_live(feed)
            if df is not None and len(df):
                df.to_csv(cache_path, index=False)
                prov.update(source="live", url=USGS_URL.format(feed=feed),
                            n=len(df))
                say(f"[live]   {len(df):,} events from USGS, cached to "
                    f"{cache_path}")
                return df, prov
        except Exception as exc:                       # noqa: BLE001
            say(f"[live]   unavailable ({type(exc).__name__}), trying mirror")

    # 3. pinned mirror in the teaching repository
    try:
        df = _try_mirror(feed)
        if df is not None and len(df):
            df.to_csv(cache_path, index=False)
            prov.update(source="mirror", repo=REPO, ref=REF, n=len(df))
            say(f"[mirror] {len(df):,} events from the teaching repo")
            return df, prov
    except Exception as exc:                           # noqa: BLE001
        say(f"[mirror] unavailable ({type(exc).__name__}), using synthetic data")

    # 4. synthetic, clearly labelled
    df = synthetic_quakes()
    prov.update(source="synthetic", n=len(df),
                warning="NOT REAL DATA. Generated along plate-boundary paths "
                        "so the lab still runs offline.")
    say(f"[SYNTHETIC] {len(df):,} generated events -- NOT REAL. "
        f"The clustering lessons still work, but do not quote any number.")
    return df, prov


# --------------------------------------------------------------------------
# Synthetic fallback: events along real plate-boundary paths
# --------------------------------------------------------------------------

# Approximate waypoints along the world's major seismic belts, in
# (latitude, longitude). These are rough by design -- enough to reproduce the
# shape of the problem, not a tectonic model.
PLATE_BOUNDARIES = {
    "Ring of Fire (west)": [
        (56, 162), (50, 157), (46, 152), (42, 145), (38, 142), (34, 140),
        (30, 140), (24, 142), (18, 146), (14, 124), (8, 126), (2, 127),
        (-2, 122), (-8, 110), (-9, 120), (-7, 130), (-6, 148), (-9, 160),
        (-17, 168), (-20, -175), (-25, -176), (-32, 179), (-38, 178),
    ],
    "Ring of Fire (east)": [
        (52, -175), (54, -165), (58, -152), (60, -145), (56, -135),
        (52, -130), (45, -125), (40, -124), (37, -122), (32, -117),
        (24, -110), (18, -103), (14, -91), (10, -85), (6, -78), (0, -80),
        (-8, -79), (-12, -77), (-20, -70), (-30, -72), (-38, -74), (-45, -75),
    ],
    "Mid-Atlantic Ridge": [
        (66, -18), (60, -28), (52, -32), (44, -28), (38, -30), (30, -42),
        (22, -45), (14, -45), (6, -33), (0, -25), (-8, -14), (-16, -13),
        (-24, -13), (-32, -14), (-40, -16), (-50, -10),
    ],
    "Alpine-Himalayan belt": [
        (36, -8), (37, 2), (40, 15), (38, 21), (37, 27), (39, 35), (38, 44),
        (33, 50), (30, 58), (30, 68), (34, 74), (28, 85), (26, 92), (23, 96),
        (16, 94), (10, 93),
    ],
    "East African Rift": [
        (14, 41), (12, 42), (8, 39), (4, 37), (0, 36), (-3, 36), (-6, 35),
        (-9, 34), (-13, 34), (-16, 35),
    ],
    "Southeast Indian Ridge": [
        (-38, 78), (-42, 88), (-45, 100), (-48, 112), (-50, 124), (-52, 140),
    ],
}


def _interpolate(path, n):
    """Sample ``n`` points along a polyline, handling the antimeridian."""
    pts = np.array(path, dtype=float)
    lat, lon = pts[:, 0], pts[:, 1]

    # Unwrap longitude so a jump from +179 to -175 is treated as 6 degrees,
    # not 354. Wrap back at the end.
    lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))

    seg = np.sqrt(np.diff(lat) ** 2 + np.diff(lon_unwrapped) ** 2)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)

    out_lat = np.interp(t, cum, lat)
    out_lon = np.interp(t, cum, lon_unwrapped)
    out_lon = ((out_lon + 180) % 360) - 180
    return out_lat, out_lon


def synthetic_quakes(n_per_belt: int = 420, n_intraplate: int = 130,
                     seed: int = 11, jitter_deg: float = 1.5) -> pd.DataFrame:
    """Generate plausible-looking events along the belts above.

    This exists so the lab still runs with no internet. The *shapes* are
    realistic -- long, thin, curved arcs -- which is all the clustering
    lessons depend on. The magnitudes and times are invented.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for belt, path in PLATE_BOUNDARIES.items():
        lat, lon = _interpolate(path, n_per_belt)
        lat = lat + rng.normal(0, jitter_deg, n_per_belt)
        lon = lon + rng.normal(0, jitter_deg, n_per_belt)
        lon = ((lon + 180) % 360) - 180
        mag = np.clip(rng.gamma(3.2, 0.42, n_per_belt) + 2.5, 2.5, 8.4)
        depth = np.clip(rng.gamma(1.6, 22, n_per_belt), 1, 680)
        for la, lo, m, d in zip(lat, lon, mag, depth):
            rows.append((la, lo, m, d, belt))

    # Intraplate events: real, scattered, and genuinely belonging to no belt.
    for _ in range(n_intraplate):
        la = rng.uniform(-60, 70)
        lo = rng.uniform(-180, 180)
        rows.append((la, lo, float(np.clip(rng.gamma(2.4, 0.4) + 2.5, 2.5, 6.5)),
                     float(np.clip(rng.gamma(1.4, 14), 1, 120)), "(intraplate)"))

    df = pd.DataFrame(rows, columns=["lat", "lon", "mag", "depth_km",
                                     "true_belt"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"syn{i:05d}" for i in range(len(df))])
    df["time"] = pd.Timestamp.utcnow() - pd.to_timedelta(
        rng.uniform(0, 30, len(df)), unit="D")
    df["place"] = df["true_belt"]
    df["magType"] = "synthetic"
    df["net"] = "syn"
    df["type"] = "earthquake"
    return df.round({"lat": 4, "lon": 4, "mag": 2, "depth_km": 1})


# --------------------------------------------------------------------------
# Coda: World Bank capital cities
# --------------------------------------------------------------------------

def load_capitals() -> pd.DataFrame:
    """Capital-city coordinates plus World Bank region and income group.

    Read from the teaching repository, cached locally. Used in the coda to ask
    whether the World Bank's "regions" are geographic groupings or political
    ones.
    """
    CACHE.mkdir(exist_ok=True)
    local = CACHE / "worldbank_countries.csv"
    if not local.exists():
        url = (f"https://raw.githubusercontent.com/{REPO}/{REF}"
               f"/data/processed/worldbank_countries.csv")
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            local.write_bytes(r.read())

    df = pd.read_csv(local)
    df = df[df["income_level"].notna()]        # drop aggregates like "World"
    df = df.dropna(subset=["lat", "lon"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Small numeric helper
# --------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2, R: float = 6371.0):
    """Great-circle distance in kilometres between two points in degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
