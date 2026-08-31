"""
esgdata.py
==========
Loaders for the pinned ESG data snapshot used in AI-integrated Sustainable
Finance (25881), Labs 7a and 7b.

Everything is read from the teaching repository over HTTPS and cached to a
local ``_cache/`` folder on first use, so the second run of any notebook is
offline.

    https://github.com/VitaliAlexeev/AI_SustainableFinance_2026

Pinning
-------
``REF`` below controls which version of the data you get. It is set to
``"main"`` for convenience, but **for anything you intend to cite or submit,
replace it with a commit SHA**. A branch name is a moving target, which is
precisely the restatement problem Lecture 7 complains about in commercial ESG
vendors. Do not exempt yourself from your own critique.

    esgdata.REF = "a1b2c3d..."      # before loading anything

Known defects in this snapshot
------------------------------
Documented rather than silently patched, because noticing them is part of the
lesson. See ``snapshot_health()`` for a machine-readable version.

1. ``climatetrace_sources.csv`` has an ``iso3`` column that says ``"AUS"`` for
   every row. It is **wrong**. It records the country that was *requested*,
   not the country each source is *in*. Use the ``country`` column, which
   comes from the API and covers 211 countries.
2. The same file is a *truncated global* extract, not an Australian one. The
   API ignored the country filter, and pagination stopped at 10,000 rows for
   five of the nine sectors. So it is "the largest emitters per sector,
   worldwide", which is useful but is not a census.
3. ``owner`` is empty for every row. The v7 ``/sources`` endpoint does not
   return ownership. Emissions can be located but not attributed to a company
   from this file alone.
4. Climate TRACE mixes **point facilities** (power stations, mines) with
   **administrative aggregates** (shires, cities) whose coordinates are
   polygon centroids. Use :func:`is_point_source` before mapping.

Author: Vitali Alexeev, UTS Business School
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = [
    "REF", "REPO", "data_url", "CACHE",
    "load_climatetrace", "load_worldbank_panel", "load_worldbank_countries",
    "load_worldbank_indicators", "load_edgar", "load_geocoded",
    "load_geocode_cache", "load_manifest",
    "POINT_SECTORS", "AGGREGATE_SECTORS", "is_point_source",
    "SEC_COUNTRY_CODES", "clean_sec_address",
    "country_rollup", "snapshot_health",
]

REPO = "VitaliAlexeev/AI_SustainableFinance_2026"

# Change to a commit SHA for reproducible work. See the module docstring.
REF = "main"

CACHE = Path("_cache")


def data_url(name: str) -> str:
    """Raw GitHub URL for one file in ``data/processed/``."""
    return (f"https://raw.githubusercontent.com/{REPO}/{REF}"
            f"/data/processed/{name}")


def _fetch(name: str, subdir: str = "processed") -> Path:
    """Download once, then read from ``_cache/`` forever after."""
    CACHE.mkdir(exist_ok=True)
    local = CACHE / name
    if local.exists():
        return local

    url = (f"https://raw.githubusercontent.com/{REPO}/{REF}/data/{subdir}/{name}"
           if subdir else
           f"https://raw.githubusercontent.com/{REPO}/{REF}/data/{name}")
    print(f"downloading {name} ...", end=" ", flush=True)
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=180) as r:
            local.write_bytes(r.read())
        print(f"{local.stat().st_size / 1e6:.1f} MB, cached")
    except Exception as exc:                           # noqa: BLE001
        raise RuntimeError(
            f"could not fetch {url}\n"
            f"  Check REF={REF!r} is a valid branch or commit, and that the "
            f"file exists in data/processed/.\n  Underlying error: {exc}"
        ) from exc
    return local


# --------------------------------------------------------------------------
# Climate TRACE
# --------------------------------------------------------------------------

# Sectors whose rows are individual physical facilities with real coordinates.
POINT_SECTORS = {"power", "manufacturing", "mineral-extraction",
                 "fossil-fuel-operations"}

# Sectors whose rows are administrative areas (shires, cities, regions).
# The coordinates are polygon centroids, so a marker on a map is a
# cartographic fiction: it says "somewhere in this shire", not "here".
AGGREGATE_SECTORS = {"forestry-and-land-use", "agriculture", "buildings",
                     "transportation", "waste"}


def is_point_source(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: rows that are genuine point facilities.

    Filter with this before plotting facility markers. Mapping a shire
    centroid next to a power station implies a precision that is not there.
    """
    return df["sector"].isin(POINT_SECTORS)


def load_climatetrace(top_only: bool = False,
                      country: str | None = None,
                      point_sources_only: bool = False) -> pd.DataFrame:
    """Climate TRACE source-level emissions.

    Parameters
    ----------
    top_only : bool
        Load the 2,000-row top-emitters extract instead of all 78,924 rows.
    country : str, optional
        ISO3 filter applied to the **``country``** column, not the broken
        ``iso3`` column.
    point_sources_only : bool
        Keep only physical facilities (see :data:`POINT_SECTORS`).

    Notes
    -----
    The broken ``iso3`` column is dropped and replaced by a corrected copy
    taken from ``country``, so downstream code cannot accidentally use it.
    """
    name = ("climatetrace_top_sources.csv" if top_only
            else "climatetrace_sources.csv")
    df = pd.read_csv(_fetch(name), low_memory=False)

    # Defect 1: repair the mislabelled country column.
    if "country" in df.columns:
        df = df.drop(columns=["iso3"], errors="ignore")
        df = df.rename(columns={"country": "iso3"})

    if country:
        df = df[df["iso3"] == country.upper()]
    if point_sources_only:
        df = df[is_point_source(df)]

    return df.reset_index(drop=True)


def country_rollup(ct: pd.DataFrame, countries: pd.DataFrame | None = None
                   ) -> pd.DataFrame:
    """Aggregate Climate TRACE sources to one row per country.

    Optionally joins World Bank region and income group, which is what makes
    the country choropleth and the income comparisons possible.
    """
    out = (ct.groupby("iso3", as_index=False)
           .agg(n_sources=("source_id", "size"),
                total_co2e=("emissions_quantity", "sum"),
                median_co2e=("emissions_quantity", "median"),
                n_sectors=("sector", "nunique")))
    if countries is not None:
        out = out.merge(
            countries[["iso3", "country", "region", "income_level",
                       "lat", "lon"]],
            on="iso3", how="left")
    return out


# --------------------------------------------------------------------------
# World Bank
# --------------------------------------------------------------------------

def load_worldbank_panel() -> pd.DataFrame:
    """Sovereign ESG panel: one row per economy-year, 80 indicator columns."""
    return pd.read_csv(_fetch("worldbank_esg_panel.csv"))


def load_worldbank_countries(drop_aggregates: bool = True) -> pd.DataFrame:
    """Economy metadata: region, income group, capital city coordinates.

    ``drop_aggregates=True`` removes rows like "World" and "OECD members",
    which are not countries and will distort any cross-country statistic.
    They are identified by having no income group.
    """
    df = pd.read_csv(_fetch("worldbank_countries.csv"))
    if drop_aggregates:
        df = df[df["income_level"].notna()]
    return df.reset_index(drop=True)


def load_worldbank_indicators() -> pd.DataFrame:
    """Indicator code to human-readable name."""
    return pd.read_csv(_fetch("worldbank_esg_indicators.csv"))


INDICATOR_COLS_EXCLUDE = {"iso3", "year"}


def indicator_columns(panel: pd.DataFrame) -> list:
    """The 80 indicator columns, excluding the keys."""
    return [c for c in panel.columns if c not in INDICATOR_COLS_EXCLUDE]


# --------------------------------------------------------------------------
# SEC EDGAR and geocoding
# --------------------------------------------------------------------------

# EDGAR uses its own state/country codes, which are NOT ISO codes. Feeding
# them to a geocoder produces silent, systematic failure on foreign filers.
# This is the identifier-mismatch problem from the lecture, in the wild.
#
# IMPORTANT: this table is PARTIAL and was built by checking codes against the
# actual filers in this snapshot -- an earlier version of it, written from
# memory, had four entries wrong (I0 is France, not Israel; P7 is the
# Netherlands, not Singapore; U0 is Singapore, not Sweden). The codes run
# roughly alphabetically by country name, but do not extend this table by
# guessing the pattern. Check each addition against a filer whose city you
# recognise, or against the official EDGAR code list. A crosswalk is a
# first-class artefact and deserves the same validation as any other data.
SEC_COUNTRY_CODES = {
    # --- Canadian provinces (EDGAR treats these like US states) ---
    "A0": "Alberta, Canada", "A1": "British Columbia, Canada",
    "A6": "Ontario, Canada", "A8": "Quebec, Canada",
    # --- Countries, validated against filers in this snapshot ---
    "C3": "Australia",       # Rio Tinto Ltd, Victoria
    "D5": "Brazil",          # Itau Unibanco, Sao Paulo
    "F4": "China",           # Midea Group, Foshan
    "G7": "Denmark",         # Novo Nordisk, Bagsvaerd
    "I0": "France",          # TotalEnergies, Courbevoie
    "L2": "Ireland",         # Eaton Corp plc, Dublin
    "L6": "Italy",           # Ferrari NV, Maranello
    "M0": "Japan",           # Mitsubishi UFJ, Tokyo
    "P7": "Netherlands",     # ASML Holding NV, Veldhoven
    "Q8": "Norway",          # Equinor ASA, Stavanger
    "U0": "Singapore",       # Seagate Technology, Singapore
    "U3": "Spain",           # Banco Santander, Madrid
    "V8": "Switzerland",     # Novartis AG, Basel
    "X0": "United Kingdom",  # HSBC Holdings, London
    "X3": "Uruguay",         # MercadoLibre, Montevideo
    "2M": "Germany",         # SAP SE, Walldorf
}


def clean_sec_address(address: str, state_or_country: str | None = None
                      ) -> str:
    """Make an EDGAR address geocodable.

    Two fixes, both of which materially change the success rate:

    1. Translate EDGAR's proprietary country codes into country names.
       ``"1 Churchill Place, London, X0, E14 5HP"`` is unresolvable;
       ``"1 Churchill Place, London, United Kingdom, E14 5HP"`` is not.
    2. Strip suite, floor and PO-box fragments, which geocoders cannot use
       and which frequently cause an otherwise-good address to fail.
    """
    import re

    if not isinstance(address, str):
        return ""

    s = address
    for code, name in SEC_COUNTRY_CODES.items():
        s = re.sub(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])",
                   name, s, flags=re.IGNORECASE)

    s = re.sub(r"\b(suite|ste\.?|floor|fl\.?|unit|level|room|rm\.?|"
               r"p\.?\s?o\.?\s?box|mailstop|ms)\b[^,]*", "", s,
               flags=re.IGNORECASE)
    s = re.sub(r"\s*,\s*,+", ",", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,")
    return s


def load_edgar(with_clean_address: bool = True) -> pd.DataFrame:
    """SEC filers with business addresses and SIC codes."""
    df = pd.read_csv(_fetch("edgar_companies.csv"))
    if with_clean_address:
        df["hq_address_clean"] = [
            clean_sec_address(a, s) for a, s in
            zip(df["hq_address"], df["hq_state_or_country"])]
        df["is_foreign_code"] = (df["hq_state_or_country"]
                                 .isin(SEC_COUNTRY_CODES))
    return df


def load_geocoded() -> pd.DataFrame:
    """Successfully geocoded addresses, with Nominatim's match quality.

    ``type`` and ``class`` record *how precise* the match was. A ``house``
    match is a building; a ``highway`` match is the street, not the address;
    a ``place`` match may be the whole suburb. Treating them as equivalent is
    an undisclosed imputation.
    """
    return pd.read_csv(_fetch("geocoded_addresses.csv"))


def load_geocode_cache() -> pd.DataFrame:
    """Every geocode attempt, including the failures.

    Returns one row per forward-geocode attempt with ``resolved`` False where
    Nominatim returned nothing. The failures are the interesting part: they
    are not randomly distributed.
    """
    raw = json.loads(Path(_fetch("geocode_cache.json")).read_text())
    rows = []
    for key, val in raw.items():
        if not key.startswith("fwd:"):
            continue
        query = key[4:]
        if val is None:
            rows.append({"query": query, "resolved": False, "lat": None,
                         "lon": None, "type": None, "class": None})
        else:
            rows.append({"query": query, "resolved": True,
                         "lat": val.get("lat"), "lon": val.get("lon"),
                         "type": val.get("type"), "class": val.get("class"),
                         "display_name": val.get("display_name")})
    return pd.DataFrame(rows)


def load_manifest() -> dict:
    """The provenance record: URLs, timestamps, row counts, SHA-256 digests."""
    return json.loads(Path(_fetch("MANIFEST.json", subdir="")).read_text())


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------

def snapshot_health() -> pd.DataFrame:
    """Check the snapshot against what it claims to be.

    Run this at the top of any analysis. The point of the exercise is that
    a dataset can load cleanly, have no missing values in its key columns,
    and still not be the dataset you asked for.
    """
    ct = load_climatetrace()
    wb = load_worldbank_panel()
    co = load_worldbank_countries(drop_aggregates=False)
    gc = load_geocode_cache()
    ind = indicator_columns(wb)

    checks = [
        ("Climate TRACE rows", f"{len(ct):,}", ""),
        ("Climate TRACE countries", f"{ct['iso3'].nunique()}",
         "requested 1 (AUS); the API filter did not apply"),
        ("... of which Australian", f"{(ct['iso3'] == 'AUS').sum():,}", ""),
        ("... point facilities", f"{is_point_source(ct).sum():,}",
         "the rest are administrative aggregates"),
        ("Climate TRACE owner populated", f"{ct['owner'].notna().mean():.0%}",
         "v7 /sources does not return ownership"),
        ("Sectors truncated at 10,000",
         f"{(ct['sector'].value_counts() >= 9999).sum()} of "
         f"{ct['sector'].nunique()}", "pagination cap, not a census"),
        ("World Bank rows", f"{len(wb):,}", ""),
        ("World Bank fill rate", f"{wb[ind].notna().mean().mean():.1%}", ""),
        ("World Bank 2025 fill",
         f"{wb.loc[wb.year == 2025, ind].notna().mean().mean():.1%}",
         "publication lag, not absence of the world"),
        ("Economies incl. aggregates", f"{len(co)}",
         f"{co['income_level'].isna().sum()} are aggregates, not countries"),
        ("Geocode success rate",
         f"{gc['resolved'].mean():.0%}",
         f"{(~gc['resolved']).sum()} failures, and they are not random"),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "note"])
