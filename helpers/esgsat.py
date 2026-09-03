"""
esgsat.py
=========
Shared helper module for AI-integrated Sustainable Finance (25881), Lecture 8:
AI Applications in Carbon Footprint Analytics.

Design notes
------------
* Works with **no credentials**. Scene search uses the Earth Search STAC API
  (https://earth-search.aws.element84.com/v1), which is free and unauthenticated.
* Works with **no network**. Every notebook can fall back to `demo_scene()`,
  which builds a synthetic but physically plausible Sentinel-2 style scene, so a
  firewalled lab machine still runs the whole lesson.
* Reflectance scaling is handled explicitly, because getting it wrong silently
  breaks SAVI and EVI while leaving NDVI looking fine. See `to_reflectance()`
  and `savi_scaling_demo()`.
* Band ordering is never inferred from a file listing. See `BANDS`.

Heavy optional dependencies (pystac_client, odc.stac, rioxarray, geopy,
leafmap) are imported lazily inside the functions that need them, so importing
this module never fails on a minimal environment.

Vitali Alexeev, UTS Business School. Spring 2026.
"""

from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BANDS",
    "BAND_ASSET",
    "SCL_CLASSES",
    "PRESET_AOIS",
    "INDEX_REGISTRY",
    "to_reflectance",
    "normalized_difference",
    "ndvi", "evi", "evi2", "savi", "ndre", "nirv", "kndvi", "vari",
    "ndwi", "ndmi", "mndwi", "nbr", "dnbr", "ndbi", "bsi", "ndsi", "ndci",
    "compute_index",
    "scl_mask",
    "bbox_from_point",
    "bbox_from_place",
    "search_scenes",
    "load_bands",
    "demo_scene",
    "savi_scaling_demo",
    "dnbr_severity",
    "pixel_area_ha",
    "class_areas",
    "fire_emissions",
    "propagate_quadrature",
    "monte_carlo_emissions",
    "show_index",
    "FIRE_PARAMETERS_INDICATIVE",
    "aef_dequantise", "aef_quantise", "aef_check_unit_norm", "aef_load_index",
    "aef_find_tiles", "aef_tile_url", "aef_read_window", "demo_embedding",
    "AEF_HTTPS", "AEF_ROOT", "AEF_INDEX_URL", "AEF_YEARS", "AEF_BANDS",
    "AEF_SCALE", "AEF_POWER", "AEF_NODATA", "AEF_ATTRIBUTION",
]

STAC_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-c1-l2a"   # Collection 1 reprocessed L2A
STAC_COLLECTION_LEGACY = "sentinel-2-l2a"


# ---------------------------------------------------------------------------
# 1. Band metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Band:
    """One Sentinel-2 MSI band."""
    sid: str            # Sentinel-2 band identifier, e.g. "B08"
    asset: str          # Earth Search STAC asset key, e.g. "nir"
    wavelength_nm: int  # nominal S2A central wavelength
    resolution_m: int
    role: str


#: Sentinel-2 MSI bands, in *instrument* order.
#:
#: Never derive band order from ``sorted(glob("*.tif"))``: an alphabetical sort
#: places ``B8A`` after ``B12``, which silently shifts every index in a stacked
#: array. Level-2A also omits B10 (cirrus), so an L2A stack has twelve bands,
#: not thirteen.
BANDS: Tuple[Band, ...] = (
    Band("B01", "coastal",  443, 60, "Coastal aerosol"),
    Band("B02", "blue",     490, 10, "Blue"),
    Band("B03", "green",    560, 10, "Green"),
    Band("B04", "red",      665, 10, "Red"),
    Band("B05", "rededge1", 705, 20, "Red edge 1"),
    Band("B06", "rededge2", 740, 20, "Red edge 2"),
    Band("B07", "rededge3", 783, 20, "Red edge 3"),
    Band("B08", "nir",      842, 10, "NIR broad"),
    Band("B8A", "nir08",    865, 20, "NIR narrow"),
    Band("B09", "nir09",    945, 60, "Water vapour"),
    # B10 (cirrus, 1375 nm) is present in L1C only and absent from L2A.
    Band("B11", "swir16",  1610, 20, "SWIR 1"),
    Band("B12", "swir22",  2190, 20, "SWIR 2"),
)

#: Map a Sentinel-2 band id ("B08") to its Earth Search asset key ("nir").
BAND_ASSET: Dict[str, str] = {b.sid: b.asset for b in BANDS}

#: Sentinel-2 Level-2A Scene Classification Layer.
SCL_CLASSES: Dict[int, str] = {
    0: "No data",
    1: "Saturated or defective",
    2: "Dark area / cast shadow",
    3: "Cloud shadow",
    4: "Vegetation",
    5: "Not vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud, medium probability",
    9: "Cloud, high probability",
    10: "Thin cirrus",
    11: "Snow or ice",
}

#: Classes normally kept for a land-surface analysis.
SCL_KEEP_DEFAULT: Tuple[int, ...] = (4, 5, 6, 7, 11)


# ---------------------------------------------------------------------------
# 2. Areas of interest
# ---------------------------------------------------------------------------

#: Fallback areas of interest, so the labs run without a geocoding service.
#: Each entry is (longitude, latitude, half-width in km, note).
PRESET_AOIS: Dict[str, Tuple[float, float, float, str]] = {
    "sydney":        (151.21, -33.87, 25.0, "Sydney metropolitan area"),
    "warragamba":    (150.60, -33.89, 20.0, "Warragamba Dam and Lake Burragorang"),
    "blue-mountains": (150.30, -33.70, 30.0, "Blue Mountains, 2019-20 fire ground"),
    "port-kembla":   (150.90, -34.47, 10.0, "Port Kembla steelworks and port"),
    "hunter-valley": (150.90, -32.55, 30.0, "Hunter Valley open-cut coal"),
    "sundarbans":    (88.95, 22.02, 30.0, "Sundarbans mangroves, blue carbon"),
}


def bbox_from_point(lon: float, lat: float, half_width_km: float = 20.0
                    ) -> Tuple[float, float, float, float]:
    """Return a (west, south, east, north) bbox around a point.

    A simple equirectangular approximation, adequate for a teaching AOI away
    from the poles. One degree of latitude is taken as 111 km; longitude is
    scaled by cos(latitude).
    """
    if not -90.0 < lat < 90.0:
        raise ValueError("latitude must be strictly between -90 and 90")
    dlat = half_width_km / 111.0
    dlon = half_width_km / (111.0 * max(np.cos(np.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def bbox_from_place(place: str, half_width_km: float = 20.0,
                    use_network: bool = True
                    ) -> Tuple[float, float, float, float]:
    """Geocode a place name to a bbox, falling back to `PRESET_AOIS`.

    Tries `geopy` Nominatim first when `use_network` is True. If geocoding is
    unavailable, blocked, or returns nothing, falls back to a preset if the
    name matches one, and otherwise raises.
    """
    key = place.strip().lower().replace(" ", "-")
    if use_network:
        try:
            from geopy.geocoders import Nominatim  # noqa: PLC0415

            geo = Nominatim(user_agent="uts-25881-lecture08", timeout=10)
            hit = geo.geocode(place)
            if hit is not None:
                return bbox_from_point(hit.longitude, hit.latitude, half_width_km)
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(f"Geocoding unavailable ({type(exc).__name__}); "
                          f"trying the preset list.", stacklevel=2)
    if key in PRESET_AOIS:
        lon, lat, default_km, _ = PRESET_AOIS[key]
        return bbox_from_point(lon, lat, half_width_km or default_km)
    raise LookupError(
        f"Could not resolve {place!r}. Available presets: "
        f"{', '.join(sorted(PRESET_AOIS))}. You can also pass explicit "
        f"coordinates to bbox_from_point()."
    )


# ---------------------------------------------------------------------------
# 3. Reflectance scaling
# ---------------------------------------------------------------------------

def to_reflectance(dn, scale: float = 10000.0, offset: float = -1000.0,
                   clip: bool = True):
    """Convert Sentinel-2 L2A digital numbers to surface reflectance.

    Parameters
    ----------
    dn
        Array of stored integers (numpy array or xarray DataArray).
    scale
        Quantification value. 10000 for all current Sentinel-2 products.
    offset
        BOA_ADD_OFFSET. Products generated under **processing baseline 04.00
        and later** (from 25 January 2022) carry an offset of -1000; products
        generated before that carry no offset. Pass ``offset=0`` for older
        scenes. Getting this wrong shifts reflectance by 0.1, which is enough
        to change an EVI or SAVI map without changing an NDVI map.
    clip
        Clip the result to [0, 1]. Reflectance outside that range is physically
        impossible and usually indicates cloud, saturation, or a scaling error.

    Returns
    -------
    Same type as the input, as float.
    """
    out = (np.asarray(dn, dtype="float32") + offset) / scale if not hasattr(dn, "dims") \
        else (dn.astype("float32") + offset) / scale
    if clip:
        out = out.clip(0.0, 1.0) if hasattr(out, "clip") else np.clip(out, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 4. Index algebra
# ---------------------------------------------------------------------------

def _safe_div(num, den):
    """Divide, returning NaN where the denominator is zero."""
    if hasattr(num, "dims"):          # xarray
        return num / den.where(den != 0)
    num = np.asarray(num, dtype="float32")
    den = np.asarray(den, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)
    return out


def normalized_difference(a, b):
    """(a - b) / (a + b). Invariant to any positive rescaling of both inputs."""
    return _safe_div(a - b, a + b)


# -- vegetation --------------------------------------------------------------

def ndvi(nir, red):
    """Normalized Difference Vegetation Index. Scale-invariant."""
    return normalized_difference(nir, red)


def savi(nir, red, L: float = 0.5):
    """Soil-Adjusted Vegetation Index.

    **Not** scale-invariant: `L` is an additive constant in reflectance units,
    so this must be given true reflectance in [0, 1]. On raw digital numbers
    `L` is negligible and SAVI collapses to (1 + L) x NDVI.
    """
    return (1.0 + L) * _safe_div(nir - red, nir + red + L)


def evi(nir, red, blue, G: float = 2.5, C1: float = 6.0,
        C2: float = 7.5, Lc: float = 1.0):
    """Enhanced Vegetation Index. Requires true reflectance."""
    return G * _safe_div(nir - red, nir + C1 * red - C2 * blue + Lc)


def evi2(nir, red, G: float = 2.5, C: float = 2.4, Lc: float = 1.0):
    """Two-band EVI, for sensors where the blue band is unreliable."""
    return G * _safe_div(nir - red, nir + C * red + Lc)


def ndre(nir, rededge):
    """Normalized Difference Red Edge. Sentinel-2 only (no Landsat red edge)."""
    return normalized_difference(nir, rededge)


def nirv(nir, red):
    """Near-infrared reflectance of vegetation, NDVI x NIR (Badgley et al. 2017).

    Carries reflectance units, so it is not dimensionless and not bounded.
    """
    return ndvi(nir, red) * nir


def kndvi(nir, red, sigma=None):
    """Kernel NDVI (Camps-Valls et al. 2021), bounded to [0, 1].

    With the recommended lengthscale sigma = 0.5 (nir + red) this reduces to
    tanh(NDVI**2). Mask water and snow before use.
    """
    if sigma is None:
        return np.tanh(ndvi(nir, red) ** 2)
    return np.tanh(_safe_div(nir - red, 2.0 * sigma) ** 2)


def vari(green, red, blue):
    """Visible Atmospherically Resistant Index. Denominator can approach zero."""
    return _safe_div(green - red, green + red - blue)


# -- water and moisture ------------------------------------------------------

def ndwi(green, nir):
    """NDWI of McFeeters (1996). Open water. Green/NIR."""
    return normalized_difference(green, nir)


def ndmi(nir, swir16):
    """NDMI, also published as NDWI by Gao (1996) and as NDII.

    Vegetation liquid water content. NIR/SWIR1 (B08/B11), *not* B12.
    """
    return normalized_difference(nir, swir16)


def mndwi(green, swir16):
    """Modified NDWI of Xu (2006). Water in built-up and turbid settings.

    Note this is algebraically identical to `ndsi`; only the threshold and the
    assumed context differ.
    """
    return normalized_difference(green, swir16)


# -- fire --------------------------------------------------------------------

def nbr(nir, swir22):
    """Normalized Burn Ratio. NIR/SWIR2 (B08/B12)."""
    return normalized_difference(nir, swir22)


def dnbr(nbr_pre, nbr_post):
    """Differenced NBR. Positive values indicate loss of vegetation."""
    return nbr_pre - nbr_post


#: Indicative dNBR severity breaks, after the FIREMON Landscape Assessment
#: convention. Thresholds are ecosystem-dependent: calibrate before quoting.
DNBR_BREAKS: Tuple[Tuple[float, float, str], ...] = (
    (-np.inf, 0.10, "Unburnt"),
    (0.10, 0.27, "Low severity"),
    (0.27, 0.44, "Moderate-low severity"),
    (0.44, 0.66, "Moderate-high severity"),
    (0.66, np.inf, "High severity"),
)


def dnbr_severity(d):
    """Classify a dNBR array into severity codes 0..4 using `DNBR_BREAKS`."""
    arr = np.asarray(d, dtype="float32")
    out = np.full(arr.shape, np.nan, dtype="float32")
    for code, (lo, hi, _label) in enumerate(DNBR_BREAKS):
        out = np.where((arr >= lo) & (arr < hi), float(code), out)
    return out


# -- surfaces ----------------------------------------------------------------

def ndbi(swir16, nir):
    """Normalized Difference Built-Up Index. Confused by bare soil."""
    return normalized_difference(swir16, nir)


def bsi(swir16, red, nir, blue):
    """Bare Soil Index."""
    return _safe_div((swir16 + red) - (nir + blue), (swir16 + red) + (nir + blue))


def ndsi(green, swir16):
    """Normalized Difference Snow Index. Same formula as `mndwi`."""
    return normalized_difference(green, swir16)


def ndci(rededge1, red):
    """Normalized Difference Chlorophyll Index, for chlorophyll-a in water."""
    return normalized_difference(rededge1, red)


# -- registry ----------------------------------------------------------------

@dataclass(frozen=True)
class IndexSpec:
    func: object
    bands: Tuple[str, ...]      # Sentinel-2 band ids, in call order
    needs_reflectance: bool
    valid_range: Tuple[float, float]   # typical range, not a hard bound
    note: str


#: Every index in one place, so a notebook can loop over them and so the
#: reflectance requirement is machine-checkable rather than folklore.
INDEX_REGISTRY: Dict[str, IndexSpec] = {
    "NDVI":  IndexSpec(ndvi,  ("B08", "B04"), False, (-1, 1), "Saturates above LAI ~3"),
    "SAVI":  IndexSpec(savi,  ("B08", "B04"), True,  (-1.5, 1.5), "L is in reflectance units"),
    "EVI":   IndexSpec(evi,   ("B08", "B04", "B02"), True, (-1, 1), "Blue band is noisy"),
    "EVI2":  IndexSpec(evi2,  ("B08", "B04"), True,  (-1, 1), "EVI without blue"),
    "NDRE":  IndexSpec(ndre,  ("B08", "B05"), False, (-1, 1), "Sentinel-2 only"),
    "NIRv":  IndexSpec(nirv,  ("B08", "B04"), True,  (-1, 1), "Carries reflectance units"),
    "kNDVI": IndexSpec(kndvi, ("B08", "B04"), False, (0, 1), "Mask water and snow first"),
    "VARI":  IndexSpec(vari,  ("B03", "B04", "B02"), False, (-2, 2), "Denominator near zero"),
    "NDWI":  IndexSpec(ndwi,  ("B03", "B08"), False, (-1, 1), "McFeeters: open water"),
    "NDMI":  IndexSpec(ndmi,  ("B08", "B11"), False, (-1, 1), "Gao: vegetation water"),
    "MNDWI": IndexSpec(mndwi, ("B03", "B11"), False, (-1, 1), "Same formula as NDSI"),
    "NBR":   IndexSpec(nbr,   ("B08", "B12"), False, (-1, 1), "Needs a matched pre-fire scene"),
    "NDBI":  IndexSpec(ndbi,  ("B11", "B08"), False, (-1, 1), "Confused by bare soil"),
    "BSI":   IndexSpec(bsi,   ("B11", "B04", "B08", "B02"), False, (-1, 1), "Seasonal in cropping"),
    "NDSI":  IndexSpec(ndsi,  ("B03", "B11"), False, (-1, 1), "Same formula as MNDWI"),
    "NDCI":  IndexSpec(ndci,  ("B05", "B04"), False, (-1, 1), "Shallow, turbid water only"),
}


def compute_index(name: str, scene, reflectance: bool = True):
    """Compute a registered index from a dict or Dataset keyed by band id.

    `scene` maps Sentinel-2 band ids ("B08") to arrays. Set `reflectance=False`
    if you are deliberately passing raw digital numbers; indices that require
    true reflectance will then warn rather than fail silently.
    """
    try:
        spec = INDEX_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown index {name!r}. "
                       f"Available: {', '.join(sorted(INDEX_REGISTRY))}") from None
    missing = [b for b in spec.bands if b not in scene]
    if missing:
        raise KeyError(f"{name} needs bands {missing}, which are not in the scene.")
    if spec.needs_reflectance and not reflectance:
        warnings.warn(
            f"{name} is not scale-invariant and you have passed raw digital "
            f"numbers. The result will be wrong in a way that still looks "
            f"plausible. Convert with to_reflectance() first.",
            stacklevel=2,
        )
    return spec.func(*(scene[b] for b in spec.bands))


# ---------------------------------------------------------------------------
# 5. Masking
# ---------------------------------------------------------------------------

def scl_mask(scl, keep: Iterable[int] = SCL_KEEP_DEFAULT):
    """Boolean mask that is True for pixels to keep, from the L2A SCL band."""
    keep = tuple(keep)
    arr = scl if hasattr(scl, "dims") else np.asarray(scl)
    out = None
    for k in keep:
        hit = arr == k
        out = hit if out is None else (out | hit)
    return out


# ---------------------------------------------------------------------------
# 6. Scene search and loading (network)
# ---------------------------------------------------------------------------

def search_scenes(bbox: Sequence[float], start: str, end: str,
                  max_cloud: float = 20.0, limit: int = 50,
                  collection: str = STAC_COLLECTION, stac_url: str = STAC_URL):
    """Search Earth Search for Sentinel-2 L2A scenes. No credentials needed.

    Returns a list of pystac Items sorted by cloud cover, least cloudy first.
    Raises a clear message if the network or the API is unavailable, so a
    notebook can fall back to `demo_scene()`.
    """
    try:
        from pystac_client import Client       # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "search_scenes needs pystac-client:  pip install pystac-client"
        ) from exc

    client = Client.open(stac_url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
    return items


def load_bands(items, bbox: Sequence[float], bands: Sequence[str] = ("B04", "B03", "B02", "B08"),
               resolution: int = 20, chunks: Optional[dict] = None):
    """Load selected bands for one or more STAC items into an xarray Dataset.

    Bands are given as Sentinel-2 ids ("B08"); they are translated to Earth
    Search asset keys internally and renamed back, so downstream code never
    depends on the provider's naming.
    """
    try:
        from odc.stac import load as odc_load   # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "load_bands needs odc-stac:  pip install odc-stac"
        ) from exc

    assets = [BAND_ASSET[b] for b in bands]
    ds = odc_load(
        items if isinstance(items, (list, tuple)) else [items],
        bands=assets + ["scl"],
        bbox=list(bbox),
        resolution=resolution,
        crs="utm",
        chunks=chunks or {},
        groupby="solar_day",
    )
    rename = {BAND_ASSET[b]: b for b in bands}
    return ds.rename(rename)


# ---------------------------------------------------------------------------
# 7. Offline demonstration data
# ---------------------------------------------------------------------------


def _box_blur(a, k: int):
    """Separable box blur via an integral image. No scipy dependency.

    Used to give synthetic scenes spatial autocorrelation, so that the
    cross-validation leakage lesson in Lecture 08c can be demonstrated offline.
    """
    a = np.asarray(a, dtype="float32")
    if k is None or k <= 1:
        return a
    pad = k // 2
    ap = np.pad(a, pad, mode="reflect")
    c = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    out = (c[k:k + h, k:k + w] - c[0:h, k:k + w]
           - c[k:k + h, 0:w] + c[0:h, 0:w]) / float(k * k)
    return out.astype("float32")


def demo_scene(size: int = 256, seed: int = 25881, as_dn: bool = True,
               burnt: bool = False, cloud: bool = False,
               noise_sd: float = 0.010, smooth_px: int = 0):
    """Build a synthetic Sentinel-2 style scene with four land-cover types.

    Returns a dict keyed by Sentinel-2 band id, plus "SCL". Values are stored
    integers if `as_dn` is True (matching what you get from the archive) or
    float reflectance if False.

    The scene is deliberately simple: water in the south-west, dense vegetation
    in the north, bare soil in the south-east, and built-up in the centre.
    With `burnt=True`, a patch of the vegetation is replaced by a burn
    signature, which lets `nbr` and `dnbr` be demonstrated without a download.
    With `cloud=True`, an opaque cloud and its offset shadow are added, and the
    scene classification layer is set accordingly, so that masking can be
    demonstrated offline.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / float(size)

    # Endmember surface reflectance, roughly realistic for Sentinel-2 bands.
    #                 B02   B03   B04   B05   B08   B8A   B11   B12
    endmembers = {
        "water":     [0.05, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01],
        "vegetation":[0.03, 0.06, 0.03, 0.12, 0.42, 0.45, 0.20, 0.09],
        "soil":      [0.12, 0.16, 0.22, 0.26, 0.31, 0.32, 0.36, 0.31],
        "builtup":   [0.14, 0.16, 0.18, 0.19, 0.22, 0.22, 0.27, 0.24],
        "burnt":     [0.06, 0.08, 0.11, 0.13, 0.16, 0.16, 0.30, 0.34],
        "cloud":     [0.72, 0.74, 0.75, 0.75, 0.76, 0.76, 0.55, 0.40],
        "shadow":    [0.02, 0.03, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02],
    }
    order = ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12")

    label = np.full((size, size), 1, dtype=np.uint8)          # vegetation
    label[(xx < 0.35) & (yy > 0.60)] = 0                      # water
    label[(xx > 0.65) & (yy > 0.55)] = 2                      # soil
    label[(np.abs(xx - 0.5) < 0.12) & (np.abs(yy - 0.4) < 0.12)] = 3   # built-up
    if burnt:
        label[(yy < 0.30) & (xx > 0.30) & (xx < 0.75)] = 4    # burn scar
    if cloud:
        # An elliptical cloud, with its shadow offset down and to the right.
        cl = (((xx - 0.72) / 0.16) ** 2 + ((yy - 0.22) / 0.11) ** 2) < 1.0
        sh = (((xx - 0.79) / 0.16) ** 2 + ((yy - 0.33) / 0.11) ** 2) < 1.0
        label[sh & ~cl] = 6                                   # shadow first
        label[cl] = 5                                         # cloud on top

    names = ("water", "vegetation", "soil", "builtup", "burnt", "cloud", "shadow")
    scene = {}
    for j, band in enumerate(order):
        base = np.zeros((size, size), dtype="float32")
        for code, nm in enumerate(names):
            base = np.where(label == code, endmembers[nm][j], base)
        noise = rng.normal(0.0, noise_sd, size=(size, size)).astype("float32")
        if smooth_px > 1:
            # Spatially correlated noise makes neighbouring pixels near-duplicates,
            # which is what causes random cross-validation to leak.
            noise = _box_blur(noise, smooth_px) * float(smooth_px)
        scene[band] = np.clip(base + noise, 0.0, 1.0)

    # Scene classification: 6 water, 4 vegetation, 5 not vegetated, 5 built-up.
    scl = np.select(
        [label == 0, label == 1, label == 2, label == 3,
         label == 4, label == 5, label == 6],
        [6, 4, 5, 5, 5, 9, 3],
    ).astype("uint8")
    scene["SCL"] = scl

    if as_dn:
        # Inverse of to_reflectance(): reflectance = (DN + offset) / scale with
        # offset = -1000, so DN = reflectance * 10000 + 1000. Getting this sign
        # backwards shifts every band by 0.2 in reflectance, which drives red
        # to zero and pins NDVI at 1.0. Ask how we know.
        for band in order:
            scene[band] = np.round(scene[band] * 10000.0 + 1000.0).astype("int16")
    return scene


def savi_scaling_demo(L: float = 0.5, verbose: bool = True):
    """Show, numerically, two ways that reflectance scaling breaks an index.

    Lesson 1. SAVI computed on digital numbers is not SAVI. The additive term
    `L` is in reflectance units, so against values in the thousands it is
    negligible and SAVI collapses to exactly (1 + L) x NDVI. The map still
    looks like a vegetation map, which is why nobody notices.

    Lesson 2. Since processing baseline 04.00, L2A products carry an additive
    offset of -1000. An additive offset is *not* removed by a normalised
    difference, so even NDVI differs depending on whether you converted to
    reflectance first. Scale invariance protects you from a multiplicative
    error, not from an additive one.

    Returns a dict of summary numbers so a notebook can assert on them.
    """
    scene = demo_scene(size=128, as_dn=True)
    nir_dn = scene["B08"].astype("float32")
    red_dn = scene["B04"].astype("float32")
    nir_rf = to_reflectance(nir_dn)
    red_rf = to_reflectance(red_dn)

    ndvi_dn = ndvi(nir_dn, red_dn)      # on stored integers
    ndvi_rf = ndvi(nir_rf, red_rf)      # on reflectance
    savi_rf = savi(nir_rf, red_rf, L)   # correct
    savi_dn = savi(nir_dn, red_dn, L)   # the mistake

    ok = np.isfinite(savi_dn) & np.isfinite(ndvi_dn) & (np.abs(ndvi_dn) > 1e-3)
    out = {
        "median_ratio_saviDN_to_ndviDN": float(np.nanmedian(savi_dn[ok] / ndvi_dn[ok])),
        "expected_ratio_if_L_ignored": 1.0 + L,
        "max_abs_deviation_from_scaled_ndvi": float(np.nanmax(np.abs(savi_dn - (1.0 + L) * ndvi_dn))),
        "mean_abs_difference_correct_vs_wrong_savi": float(np.nanmean(np.abs(savi_rf - savi_dn))),
        "mean_abs_ndvi_offset_error": float(np.nanmean(np.abs(ndvi_rf - ndvi_dn))),
        "max_abs_ndvi_offset_error": float(np.nanmax(np.abs(ndvi_rf - ndvi_dn))),
    }
    if verbose:
        print("Lesson 1: SAVI on digital numbers is NDVI in disguise")
        print("-" * 58)
        print(f"  median( SAVI_DN / NDVI_DN )       = {out['median_ratio_saviDN_to_ndviDN']:.6f}")
        print(f"  (1 + L)                           = {out['expected_ratio_if_L_ignored']:.6f}")
        print(f"  max | SAVI_DN - (1+L)*NDVI_DN |   = {out['max_abs_deviation_from_scaled_ndvi']:.2e}")
        print(f"  mean | SAVI_right - SAVI_wrong |  = {out['mean_abs_difference_correct_vs_wrong_savi']:.4f}")
        print()
        print("Lesson 2: an additive offset survives a normalised difference")
        print("-" * 58)
        print(f"  mean | NDVI_reflectance - NDVI_DN | = {out['mean_abs_ndvi_offset_error']:.4f}")
        print(f"  max  | NDVI_reflectance - NDVI_DN | = {out['max_abs_ndvi_offset_error']:.4f}")
        print()
        print("  NDVI is invariant to a multiplicative rescaling, so it survives")
        print("  the 1/10000. It is NOT invariant to the baseline 04.00 offset of")
        print("  -1000, so 'NDVI is scale-free, I can skip the conversion' is wrong.")
    return out



# ---------------------------------------------------------------------------
# 8. Area accounting, fire emissions and uncertainty (Lecture 08b)
# ---------------------------------------------------------------------------

def pixel_area_ha(resolution_m: float) -> float:
    """Ground area of one square pixel, in hectares."""
    return (float(resolution_m) ** 2) / 10_000.0


def class_areas(classified, resolution_m: float, labels: Optional[Dict[int, str]] = None):
    """Count pixels per class and convert to hectares.

    Returns a list of (code, label, n_pixels, hectares) tuples, sorted by code.
    NaN pixels are excluded and reported separately as code -1.
    """
    arr = np.asarray(classified, dtype="float32")
    finite = np.isfinite(arr)
    per_px = pixel_area_ha(resolution_m)
    rows = []
    codes = np.unique(arr[finite]).astype(int)
    for c in codes:
        n = int(np.count_nonzero(arr[finite] == c))
        lab = (labels or {}).get(int(c), str(int(c)))
        rows.append((int(c), lab, n, n * per_px))
    n_nan = int(np.count_nonzero(~finite))
    if n_nan:
        rows.append((-1, "No data / masked", n_nan, n_nan * per_px))
    return sorted(rows, key=lambda r: r[0])


#: Indicative parameter ranges for the IPCC Tier 1 fire equation.
#:
#: **These are placeholders, not authorities.** Sourcing and justifying each
#: parameter for the specific vegetation type is part of the exercise, and the
#: uncertainty in these three numbers dominates anything the satellite
#: contributes. Start from IPCC 2006 Guidelines Volume 4, Chapter 2, Tables 2.4
#: (mass of fuel available and combustion factor) and 2.5 (emission factors),
#: then look for Australian vegetation-specific literature.
FIRE_PARAMETERS_INDICATIVE = {
    "fuel_load_t_per_ha": (10.0, 60.0, "M_B, tonnes dry matter per hectare"),
    "combustion_factor": (0.20, 0.80, "C_f, dimensionless fraction consumed"),
    "ef_co2_g_per_kg": (1500.0, 1700.0, "G_ef for CO2, grams per kg dry matter"),
    "ef_ch4_g_per_kg": (2.0, 7.0, "G_ef for CH4, grams per kg dry matter"),
    "gwp100_ch4": (27.0, 30.0, "AR6 GWP-100 for fossil/non-fossil CH4"),
}


def fire_emissions(area_ha: float, fuel_load_t_per_ha: float,
                   combustion_factor: float, ef_co2_g_per_kg: float,
                   ef_ch4_g_per_kg: float = 0.0, gwp100_ch4: float = 0.0) -> float:
    """IPCC Tier 1 fire emissions, in tonnes CO2-equivalent.

    Implements the standard form

        L_fire = A * M_B * C_f * G_ef * 1e-3

    where A is burnt area (ha), M_B is the mass of fuel available for
    combustion (t dry matter per ha), C_f is the combustion factor, and G_ef is
    an emission factor (g of gas per kg of dry matter burnt).

    No parameter has a default except the optional methane terms, deliberately:
    every one of them must be sourced and justified for the vegetation type
    concerned. See `FIRE_PARAMETERS_INDICATIVE` for starting ranges and the
    tables to check them against.
    """
    dm_burnt_t = float(area_ha) * float(fuel_load_t_per_ha) * float(combustion_factor)
    # g gas per kg dm  ==  kg gas per tonne dm  ==  1e-3 t gas per tonne dm
    t_co2 = dm_burnt_t * ef_co2_g_per_kg * 1e-3
    t_ch4 = dm_burnt_t * ef_ch4_g_per_kg * 1e-3
    return t_co2 + t_ch4 * gwp100_ch4


def propagate_quadrature(relative_errors: Dict[str, float]) -> float:
    """Combine independent relative standard errors in quadrature.

    Valid for a product of independent terms. It is an approximation, and it
    assumes independence, which for a carbon estimate is generous: area error
    and biomass error usually share a driver.
    """
    return float(np.sqrt(sum(float(v) ** 2 for v in relative_errors.values())))


def monte_carlo_emissions(area_ha_range: Tuple[float, float],
                          fuel_load_range: Tuple[float, float],
                          combustion_range: Tuple[float, float],
                          ef_co2_range: Tuple[float, float],
                          n: int = 20_000, seed: int = 25881,
                          quantiles: Sequence[float] = (0.05, 0.5, 0.95)):
    """Uniform-prior Monte Carlo over the fire emissions parameters.

    Every input is a (low, high) pair. Uniform priors are a deliberate choice:
    they say "somewhere in this range and I do not know where", which is an
    honest description of the state of knowledge for most of these terms.

    Returns (samples, dict of requested quantiles).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(*area_ha_range, n)
    m = rng.uniform(*fuel_load_range, n)
    c = rng.uniform(*combustion_range, n)
    e = rng.uniform(*ef_co2_range, n)
    samples = a * m * c * e * 1e-3
    qs = {float(q): float(np.quantile(samples, q)) for q in quantiles}
    return samples, qs


def show_index(values, title: str = "", cmap: str = "RdYlGn",
               vmin: Optional[float] = None, vmax: Optional[float] = None,
               ax=None, colorbar: bool = True):
    """Display a 2-D index array with a sensible symmetric scale.

    Defaults to the 2nd and 98th percentiles rather than the full range, so a
    handful of extreme pixels cannot flatten the whole map. Pass explicit
    vmin/vmax when comparing two dates.
    """
    import matplotlib.pyplot as plt          # noqa: PLC0415

    arr = np.asarray(values, dtype="float32")
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            lo, hi = np.percentile(finite, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
    created = ax is None
    if created:
        _fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


# ---------------------------------------------------------------------------
# 9. AlphaEarth Foundations satellite embeddings (Lecture 08c)
# ---------------------------------------------------------------------------
#
# Access verified anonymously on 1 September 2026: the bucket is NOT
# requester-pays, listing and ranged GET both work without credentials, and
# rasterio opens a tile over /vsicurl/ in under two seconds.
#
# Layout:  satellite_embedding/v1/annual/{YEAR}/{UTM_ZONE}/{image_id}-{dy}-{dx}.tiff
# Tiles:   8192 x 8192 pixels, 64 bands, int8, tiled 1024, zstd, 10 m pixels,
#          300 MB to 3.2 GB each. Never download one; always read a window.
# Years:   2017 to 2025 inclusive.

AEF_HTTPS = "https://storage.googleapis.com/alphaearth_foundations"
AEF_ROOT = "satellite_embedding/v1/annual"
AEF_INDEX_URL = f"{AEF_HTTPS}/{AEF_ROOT}/aef_index.parquet"
AEF_YEARS = tuple(range(2017, 2026))
AEF_BANDS = 64

#: De-quantisation constants. Confirmed against Google's own engineering post
#: and an independent implementation: divide by 127.5, square, restore the sign.
AEF_SCALE = 127.5
AEF_POWER = 2.0
AEF_NODATA = -128           # -127..127 are data; -128 means masked

#: Required by the CC-BY 4.0 licence. Put this on any figure you publish.
AEF_ATTRIBUTION = ("The AlphaEarth Foundations Satellite Embedding dataset is "
                   "produced by Google and Google DeepMind.")


def aef_dequantise(q, nodata_to_nan: bool = True):
    """Convert stored int8 embeddings to analysis-ready floats in [-1, 1].

        x = sign(q) * (|q| / 127.5) ** 2

    Note the exponent. Unlike the Sentinel-2 reflectance conversion in
    Lecture 08a, this is **not** a linear rescaling, so no ratio or normalised
    difference survives skipping it, and the unit-norm property of the
    embedding vector is destroyed. Use `aef_check_unit_norm()` to verify.
    """
    arr = np.asarray(q)
    out = np.sign(arr).astype("float32") * (np.abs(arr).astype("float32") / AEF_SCALE) ** AEF_POWER
    if nodata_to_nan:
        out = np.where(arr == AEF_NODATA, np.nan, out)
    return out


def aef_quantise(x):
    """Inverse of `aef_dequantise`, used to build offline test data.

        q = round( sign(x) * sqrt(|x|) * 127.5 )
    """
    arr = np.asarray(x, dtype="float32")
    q = np.sign(arr) * np.sqrt(np.abs(arr)) * AEF_SCALE
    return np.clip(np.rint(q), -127, 127).astype("int8")


def aef_check_unit_norm(cube, axis: int = 0, tol: float = 0.02):
    """Check that de-quantised embedding vectors have Euclidean length 1.

    Google normalises each 64-vector to unit length after de-quantisation, so
    this is a free end-to-end correctness test: if the norms are not near 1,
    something in the read or the de-quantisation is wrong. Returns
    (median_norm, fraction_within_tol).
    """
    arr = np.asarray(cube, dtype="float32")
    norms = np.sqrt(np.nansum(arr ** 2, axis=axis))
    good = np.isfinite(norms) & (norms > 0)
    if not good.any():
        return float("nan"), 0.0
    med = float(np.median(norms[good]))
    frac = float(np.mean(np.abs(norms[good] - 1.0) < tol))
    return med, frac


def aef_load_index(cache_path: str = "aef_index_slim.parquet",
                   year: Optional[int] = None, force: bool = False):
    """Load the AlphaEarth tile index, keeping only the columns we need.

    The published index is 66 MB and 302,466 rows across all years, with a
    geometry column that dominates the file size. We read only the seven useful
    columns, optionally filter to one year, and cache the result to disk so the
    notebook is offline-capable after the first run.
    """
    try:
        import pandas as pd                                  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("aef_load_index needs pandas") from exc

    cache = pathlib.Path(cache_path)
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
    else:
        try:
            import fsspec                                    # noqa: PLC0415
            import pyarrow.parquet as pq                     # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("aef_load_index needs fsspec and pyarrow "
                              "(pandas 3 already requires pyarrow)") from exc
        cols = ["path", "year", "utm_zone", "crs",
                "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
        with fsspec.open(AEF_INDEX_URL) as fh:
            df = pq.ParquetFile(fh).read(columns=cols).to_pandas()
        df.to_parquet(cache, index=False)
    return df[df["year"] == year] if year is not None else df


def aef_find_tiles(index_df, lon: float, lat: float, year: Optional[int] = None):
    """Rows of the index whose WGS84 bounding box contains a point.

    More than one tile can match, because tiles from adjacent UTM zones
    overlap. Prefer the tile whose centre is nearest the point.
    """
    df = index_df if year is None else index_df[index_df["year"] == year]
    hit = df[(df["wgs84_west"] <= lon) & (df["wgs84_east"] >= lon)
             & (df["wgs84_south"] <= lat) & (df["wgs84_north"] >= lat)]
    if hit.empty:
        return hit
    cx = (hit["wgs84_west"] + hit["wgs84_east"]) / 2
    cy = (hit["wgs84_south"] + hit["wgs84_north"]) / 2
    return hit.assign(_d=np.hypot(cx - lon, cy - lat)).sort_values("_d").drop(columns="_d")


def aef_tile_url(path: str) -> str:
    """Turn an index `path` (a gs:// URI or an object key) into an HTTPS URL."""
    key = path[len("gs://alphaearth_foundations/"):] if path.startswith("gs://") else path
    return f"{AEF_HTTPS}/{key.lstrip('/')}"


def aef_read_window(tile_url: str, lon: float, lat: float, size_px: int = 256,
                    bands: Optional[Sequence[int]] = None, dequantise: bool = True):
    """Read a square window of embeddings centred on a coordinate.

    Reads over /vsicurl/, so only the needed blocks cross the network: a few
    megabytes out of a tile that may be three gigabytes. Returns
    (cube, transform, crs) with cube shaped (bands, rows, cols).
    """
    try:
        import os                                            # noqa: PLC0415
        import rasterio                                      # noqa: PLC0415
        from rasterio.warp import transform as warp_transform  # noqa: PLC0415
        from rasterio.windows import Window                  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("aef_read_window needs rasterio") from exc

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    url = tile_url if tile_url.startswith("/vsicurl/") else f"/vsicurl/{tile_url}"
    with rasterio.open(url) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        half = size_px // 2
        win = Window(col - half, row - half, size_px, size_px)
        idx = list(bands) if bands is not None else list(range(1, src.count + 1))
        cube = src.read(idx, window=win, boundless=True, fill_value=AEF_NODATA)
        tr = src.window_transform(win)
        crs = src.crs
    return (aef_dequantise(cube) if dequantise else cube), tr, crs


def demo_embedding(labels, seed: int = 25881, n_bands: int = AEF_BANDS,
                   nodata_fraction: float = 0.03, noise: float = 0.9,
                   smooth_px: int = 0, as_int8: bool = True):
    """Synthetic AlphaEarth-like embeddings, for the offline path.

    Each land-cover class gets a random unit direction in 64-space; pixels are
    that direction plus noise, renormalised to unit length, then quantised with
    the real forward transform. The offline data therefore has the same dtype,
    the same nodata sentinel and the same non-linear quantisation as the real
    product, so the de-quantisation lesson works without a network.

    **These embeddings are generated from the labels.** Any classifier
    comparison run against them is circular and is not evidence about the real
    dataset. Raise `noise` to make the task harder, but do not report the
    numbers. The offline path exists to test the code, not the claim.
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    codes = np.unique(lab)
    directions = rng.normal(size=(len(codes), n_bands)).astype("float32")
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    cube = np.zeros((n_bands,) + lab.shape, dtype="float32")
    for i, c in enumerate(codes):
        sel = lab == c
        n = int(sel.sum())
        if not n:
            continue
        cube[:, sel] = directions[i][:, None]
    eps = rng.normal(0, noise, size=(n_bands,) + lab.shape).astype("float32")
    if smooth_px > 1:
        eps = np.stack([_box_blur(e, smooth_px) * float(smooth_px) for e in eps])
    cube = cube + eps
    cube /= np.linalg.norm(cube, axis=0, keepdims=True)

    q = aef_quantise(cube)
    if nodata_fraction > 0:
        h, w = lab.shape
        mask = rng.random((h, w)) < nodata_fraction
        q[:, mask] = AEF_NODATA
    return q if as_int8 else aef_dequantise(q)

if __name__ == "__main__":       # a smoke test you can run from the terminal
    savi_scaling_demo()
