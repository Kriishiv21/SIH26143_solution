"""Cases: the hard-coded DEMO scenario (4 Port Said / DARTIS cases) and the REAL GeoTIFF loader.

DEMO  -> everything is hard-coded and clearly labelled synthetic.  SAR scenes are rendered with
         realistic speckle, an incidence-angle gradient, a look-alike patch and a ship target, on a
         proper georeferenced grid.  Ground-truth masks are attached for detector benchmarking.
REAL  -> a georeferenced GeoTIFF is read with rasterio: its own CRS/transform, its own acquisition
         time (raster tag or sidecar JSON).  Nothing is ever assigned or invented.

Switching: ``cfg.mode = "REAL"`` and fill ``cfg.real_cases`` (see config.py / the notebook).
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import Config
from .geo import grid_from_bbox, GeoRef, lonlat_to_local_km, destination_point
from .observation import SarScene

# --------------------------------------------------------------------------- DEMO (hard-coded)
DEMO_REGION = dict(lat=(31.0, 34.7), lon=(30.0, 36.0))          # DARTIS region (as supplied)

# Reported centroid/area per case (as supplied). Pass times are ASSUMED (date-only was supplied);
# Sentinel-1 over the Nile delta/Port Said is typically ~03:5x UTC (descending) or ~15:5x (ascending).
DEMO_CASES = [
    dict(case_id="case_A_2019-04-05", label="Port Said A -- 5 Apr 2019",  date="2019-04-05", centroid=(31.295, 32.283), area_km2=4.7),
    dict(case_id="case_B_2019-04-06", label="Port Said B -- 6 Apr 2019",  date="2019-04-06", centroid=(31.372, 32.246), area_km2=4.4),
    dict(case_id="case_C_2019-04-11", label="Port Said C -- 11 Apr 2019", date="2019-04-11", centroid=(31.315, 32.390), area_km2=2.2),
    dict(case_id="case_D_2019-05-18", label="Port Said D -- 18 May 2019", date="2019-05-18", centroid=(31.347, 32.325), area_km2=2.0),
]
DEMO_ASSUMED_PASS_UTC = "03:52:00"


def _smooth_noise(shape, sigma, rng):
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (n.std() + 1e-9)


def make_demo_scene(case: dict, drift_bearing_deg: float, seed: int = 0, size: int = 512, pixel_deg: float = 0.0009,
                    oil_db: float = -7.5, lookalike_db: float = -3.0, looks: float = 4.4,
                    with_lookalike: bool = True) -> SarScene:
    """Render a georeferenced synthetic Sentinel-1-like scene containing the reported slick."""
    rng = np.random.default_rng(seed + zlib.crc32(case["case_id"].encode()) % 10_000)   # stable across runs
    lat0, lon0 = case["centroid"]
    georef = grid_from_bbox(lon0 - size * pixel_deg / 2 * 1.0, lat0 + size * pixel_deg / 2, pixel_deg, (size, size))
    rr, cc = np.mgrid[0:size, 0:size]
    lon, lat = georef.pixel_to_lonlat(rr, cc)
    x, y = lonlat_to_local_km(lon, lat, lon0, lat0)        # km east / north of the reported centroid

    # oil slick: elongated, mildly meandering ellipse along the drift direction; area = reported area
    th = np.deg2rad(drift_bearing_deg + rng.uniform(-15, 15))
    b = np.sqrt(case["area_km2"] / (2 * np.pi)); a = 2.0 * b
    along = x * np.sin(th) + y * np.cos(th); across = x * np.cos(th) - y * np.sin(th)
    across = across - 0.35 * b * np.sin(2 * np.pi * along / (3.0 * a))
    slick = (along / a) ** 2 + (across / b) ** 2 <= 1.0
    # rescale so pixel-count area matches the reported area
    dx_km, dy_km = georef.pixel_size_km(size / 2, size / 2)
    target_px = case["area_km2"] / (dx_km * dy_km)
    if slick.sum() > 0:
        s = np.sqrt(target_px / slick.sum())
        slick = (along / (a * s)) ** 2 + (across / (b * s)) ** 2 <= 1.0

    # look-alike (low-wind patch): larger, fuzzy, weaker contrast, away from the slick
    look = np.zeros_like(slick, dtype=float)
    if with_lookalike:
        lx, ly = -0.30 * size * dx_km, -0.22 * size * dy_km
        look = np.exp(-(((x - lx) / (1.8 * a)) ** 2 + ((y - ly) / (1.2 * a)) ** 2) ** 1.5)

    # sea backscatter: incidence-angle (range) gradient + slow wind-field modulation, in dB
    grad = np.linspace(+1.2, -1.2, size)[None, :] * np.ones((size, 1))
    mod = 0.6 * _smooth_noise((size, size), 40, rng)
    db = 0.0 + grad + mod
    db = db + oil_db * ndimage.gaussian_filter(slick.astype(float), 1.2) + lookalike_db * look
    lin = 10 ** (db / 10.0) * 0.02                          # sea sigma0 ~ -17 dB reference
    lin = lin * rng.gamma(looks, 1.0 / looks, lin.shape)    # multilook speckle
    # bright ship target + short wake
    sr, sc = int(size * 0.25), int(size * 0.72)
    lin[sr - 1:sr + 2, sc - 1:sc + 2] *= 40.0
    lin = lin.astype(np.float32)

    t = pd.Timestamp(f"{case['date']}T{DEMO_ASSUMED_PASS_UTC}")
    return SarScene(case["case_id"], lin, "linear", georef, t, "DEMO_SYNTHETIC",
                    time_source="ASSUMED (date supplied; pass time hard-coded for DEMO)", label=case["label"],
                    gt_mask=slick.astype(np.uint8),
                    meta=dict(reported_centroid=case["centroid"], reported_area_km2=case["area_km2"], synthetic=True,
                              lookalike=bool(with_lookalike), oil_db=oil_db, lookalike_db=lookalike_db))


def make_benchmark_scenes(n: int = 6, seed: int = 123) -> List[SarScene]:
    """Independent synthetic scenes (varied area / contrast / look-alikes) for detector benchmarking."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        case = dict(case_id=f"bench_{i}", label=f"bench {i}", date="2019-04-05",
                    centroid=(31.3 + rng.uniform(-0.1, 0.1), 32.3 + rng.uniform(-0.1, 0.1)), area_km2=float(rng.uniform(1.5, 6.0)))
        out.append(make_demo_scene(case, drift_bearing_deg=float(rng.uniform(0, 180)), seed=seed + i,
                                   oil_db=float(rng.uniform(-9.0, -5.5)), lookalike_db=float(rng.uniform(-4.0, -2.0)),
                                   with_lookalike=bool(i % 3 != 2)))
    return out


# --------------------------------------------------------------------------- REAL (GeoTIFF)
_TIME_TAGS = ("ACQUISITION_START_TIME", "ACQUISITION_TIME", "SENSING_TIME", "SENSING_START", "PRODUCT_START_TIME",
              "TIFFTAG_DATETIME", "start_time", "acquisition_time", "datetime")


def _find_acquisition_time(path: Path, tags: dict, override) -> Tuple[pd.Timestamp, str]:
    if override is not None:
        return pd.Timestamp(override), "user_supplied"
    low = {str(k).lower(): v for k, v in tags.items()}
    for k in _TIME_TAGS:
        if k.lower() in low:
            try:
                t = pd.Timestamp(str(low[k.lower()]).replace(":", "-", 2) if str(low[k.lower()])[4:5] == ":" else low[k.lower()])
                return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t), f"raster_tag:{k}"
            except Exception:
                continue
    for side in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json"), path.with_name(path.stem + "_metadata.json")):
        if side.exists():
            j = json.loads(side.read_text())
            for k in ("acquisition_time", "acquisition_start", "sensing_time", "datetime"):
                if k in j:
                    t = pd.Timestamp(j[k])
                    return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t), f"sidecar:{side.name}"
    raise ValueError(
        f"No acquisition time for {path}. Provide real_cases[..]['acquisition_time'], a raster tag "
        f"({', '.join(_TIME_TAGS[:4])}...) or a sidecar JSON with 'acquisition_time'. It is never invented.")


def load_real_scene(spec: dict) -> SarScene:
    import rasterio
    path = Path(spec["sar_path"])
    if not path.exists():
        raise FileNotFoundError(f"SAR product not found: {path}")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS: a raw un-georeferenced image cannot be used for backtracking.")
        band = int(spec.get("polarisation_band", 1))
        arr = src.read(band).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        tags = {**src.tags(), **src.tags(band)}
        georef = GeoRef(src.transform, src.crs.to_string(), arr.shape)
        dtype = src.dtypes[band - 1]
    t, tsrc = _find_acquisition_time(path, tags, spec.get("acquisition_time"))
    units = spec.get("units", "auto")
    if units == "auto":
        fin = arr[np.isfinite(arr)]
        if dtype == "uint8":
            units = "uint8"
        elif fin.size and np.nanpercentile(fin, 99) <= 5.0 and np.nanmin(fin) >= 0:
            units = "linear"
        elif fin.size and np.nanmax(fin) <= 10 and np.nanmin(fin) < 0:
            units = "db"
        else:
            units = "linear"
    return SarScene(spec.get("case_id", path.stem), arr, units, georef, t, "REAL_GEOTIFF", time_source=tsrc,
                    label=spec.get("label", path.stem), meta=dict(path=str(path), crs=georef.crs, units=units,
                                                                   ais_csv=spec.get("ais_csv")))


def list_case_specs(cfg: Config) -> List[dict]:
    if cfg.mode.upper() == "DEMO":
        return [dict(c) for c in DEMO_CASES]
    if cfg.mode.upper() == "REAL":
        if not cfg.real_cases:
            raise ValueError("mode='REAL' but cfg.real_cases is empty. Fill it (see config.py / notebook REAL block).")
        return [dict(c) for c in cfg.real_cases]
    raise ValueError(f"Unknown mode {cfg.mode!r}")


def load_scene(spec: dict, cfg: Config, drift_bearing_deg: Optional[float] = None) -> SarScene:
    if cfg.mode.upper() == "DEMO":
        return make_demo_scene(spec, drift_bearing_deg if drift_bearing_deg is not None else 70.0, seed=cfg.seed)
    return load_real_scene(spec)
