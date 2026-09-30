"""AIS ingestion (real CSV or synthetic) and preprocessing.

Canonical columns: MMSI, BaseDateTime (UTC, tz-naive), LAT, LON, SOG (kn), COG (deg), VesselName, VesselType.

Fixes relative to the original notebook:
  * COG is resampled with a CIRCULAR mean (sin/cos) -- a linear mean of 350 deg and 10 deg gives 180 deg.
  * Raw (irregular) tracks are kept for time-synchronous attribution; the resampled tracks feed the
    anomaly models only.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AISConfig
from .geo import haversine_km, advect_lonlat, destination_point, naive_utc

log = logging.getLogger(__name__)

CANON = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselName", "VesselType"]
_SYN = {
    "MMSI": ("mmsi", "userid", "user_id"),
    "BaseDateTime": ("basedatetime", "timestamp", "# timestamp", "time", "datetime", "time_utc", "date_time_utc", "position_timestamp"),
    "LAT": ("lat", "latitude"),
    "LON": ("lon", "lng", "long", "longitude"),
    "SOG": ("sog", "speed", "speed_over_ground", "speedoverground"),
    "COG": ("cog", "course", "course_over_ground", "courseoverground"),
    "VesselName": ("vesselname", "name", "shipname", "ship_name", "vessel_name"),
    "VesselType": ("vesseltype", "type", "shiptype", "ship_type", "vessel_type"),
}


def load_ais_csv(path: str, column_map: Optional[Dict[str, str]] = None, bbox=None, t_range=None) -> pd.DataFrame:
    """Load a real AIS CSV. Columns are auto-detected (override with column_map={'canonical': 'your_col'})."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"AIS CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    lower = {c.lower().strip(): c for c in df.columns}
    ren = {}
    for canon, syns in _SYN.items():
        if column_map and canon in column_map:
            ren[column_map[canon]] = canon; continue
        for s in syns:
            if s in lower:
                ren[lower[s]] = canon; break
    df = df.rename(columns=ren)
    missing = [c for c in ("MMSI", "BaseDateTime", "LAT", "LON") if c not in df.columns]
    if missing:
        raise KeyError(f"AIS CSV missing required columns {missing}; found {list(df.columns)[:15]}. Use ais.column_map.")
    for c in ("SOG", "COG"):
        if c not in df.columns:
            df[c] = np.nan
    if "VesselName" not in df.columns:
        df["VesselName"] = df["MMSI"].astype(str)
    if "VesselType" not in df.columns:
        df["VesselType"] = "unknown"
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ("MMSI", "LAT", "LON", "SOG", "COG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    df["MMSI"] = df["MMSI"].astype("int64")
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        df = df[df.LAT.between(lat_min, lat_max) & df.LON.between(lon_min, lon_max)]
    if t_range is not None:
        df = df[(df.BaseDateTime >= t_range[0]) & (df.BaseDateTime <= t_range[1])]
    if df.empty:
        raise ValueError("AIS CSV has no rows inside the requested region/time window.")
    return df[CANON].reset_index(drop=True)


# ----------------------------------------------------------------------------- synthetic
def _transit(mmsi, rng, start, end, t_start, speed_kn, report_min=6, name=None, vtype="cargo", jitter_deg=3.0):
    dist = haversine_km(start[0], start[1], end[0], end[1])
    bearing = np.rad2deg(np.arctan2((end[1] - start[1]) * np.cos(np.deg2rad(start[0])), end[0] - start[0])) % 360
    n = max(int(dist / (speed_kn * 1.852) * 60 / report_min), 8)
    lat, lon, rows = start[0], start[1], []
    for k in range(n):
        sog = max(1.0, speed_kn + rng.normal(0, 0.4)); cog = (bearing + rng.normal(0, jitter_deg)) % 360
        rows.append((mmsi, t_start + pd.Timedelta(minutes=report_min * k), lat, lon, sog, cog, name or f"VESSEL_{mmsi}", vtype))
        lon, lat = advect_lonlat(lon, lat, np.sin(np.deg2rad(cog)) * sog * 0.514444, np.cos(np.deg2rad(cog)) * sog * 0.514444, report_min * 60)
    return pd.DataFrame(rows, columns=CANON)


def synthetic_ais(center_lonlat, t_obs, window_hours, n_vessels=60, seed=0, radius_km=90.0) -> pd.DataFrame:
    """Background shipping traffic converging on the centre (a plausible lane), SYNTHETIC."""
    rng = np.random.default_rng(seed)
    lon0, lat0 = center_lonlat
    parts, used = [], set()
    t_lo = t_obs - pd.Timedelta(hours=window_hours + 12)
    for i in range(n_vessels):
        while True:
            mmsi = int(rng.integers(200_000_000, 799_999_999))
            if mmsi not in used:
                used.add(mmsi); break
        b1, b2 = rng.uniform(0, 360, 2)
        d1, d2 = rng.uniform(0.3, 1.0) * radius_km, rng.uniform(0.3, 1.0) * radius_km
        s_lon, s_lat = destination_point(lon0, lat0, b1, d1); e_lon, e_lat = destination_point(lon0, lat0, b2, d2)
        t0 = t_lo + pd.Timedelta(hours=float(rng.uniform(0, window_hours + 6)))
        parts.append(_transit(mmsi, rng, (s_lat, s_lon), (e_lat, e_lon), t0, float(rng.uniform(8, 17)),
                              vtype=str(rng.choice(["cargo", "tanker", "container", "fishing"], p=[.35, .3, .25, .1]))))
    return pd.concat(parts, ignore_index=True)


def inject_test_vessels(df, source_lonlat, release_time, seed=1):
    """SYNTHETIC SELF-CHECK ONLY. Adds (1) a 'culprit' that crosses the source point at release_time,
    (2) a right-place/wrong-time decoy, (3) a right-time/wrong-place decoy, and (4) a dark-gap vessel
    that switches AIS off around the release. Returns (df, truth dict)."""
    rng = np.random.default_rng(seed)
    lon0, lat0 = source_lonlat
    truth = {}
    def track(kind, mmsi, t_cross, offset_km, bearing, speed=9.0, dark=False, vtype="tanker"):
        # vessel passes (offset_km away from source) at t_cross, heading `bearing`
        cx, cy = destination_point(lon0, lat0, (bearing + 90) % 360, offset_km)
        hrs = 10.0
        s_lon, s_lat = destination_point(cx, cy, (bearing + 180) % 360, speed * 1.852 * hrs / 2)
        e_lon, e_lat = destination_point(cx, cy, bearing, speed * 1.852 * hrs / 2)
        tr = _transit(mmsi, rng, (s_lat, s_lon), (e_lat, e_lon), t_cross - pd.Timedelta(hours=hrs / 2), speed, name=f"{kind}_{mmsi}", vtype=vtype)
        if dark:
            m = (tr.BaseDateTime > t_cross - pd.Timedelta(hours=1.5)) & (tr.BaseDateTime < t_cross + pd.Timedelta(hours=1.5))
            tr = tr[~m]
        return tr
    parts = [df,
             track("CULPRIT", 999000001, release_time, 0.3, 70.0), 
             track("DECOY_WRONG_TIME", 999000002, release_time - pd.Timedelta(hours=9), 0.3, 250.0),
             track("DECOY_WRONG_PLACE", 999000003, release_time, 22.0, 70.0),
             track("DARK_GAP", 999000004, release_time + pd.Timedelta(minutes=20), 1.0, 160.0, dark=True)]
    truth = dict(culprit=999000001, decoys=[999000002, 999000003], dark_gap=999000004, release_time=release_time,
                 source=(lon0, lat0))
    return pd.concat(parts, ignore_index=True), truth


# ----------------------------------------------------------------------------- preprocessing
FEATURE_COLS = ["east_m", "north_m", "SOG", "cog_sin", "cog_cos", "hod_sin", "hod_cos"]


def clean_ais(raw: pd.DataFrame, cfg: AISConfig) -> pd.DataFrame:
    df = raw.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    for c in ("LAT", "LON", "SOG", "COG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.LAT.between(-90, 90) & df.LON.between(-180, 180)]
    df = df[~((df.LAT == 0) & (df.LON == 0))]
    df = df.drop_duplicates(["MMSI", "BaseDateTime"]).sort_values(["MMSI", "BaseDateTime"])
    df["SOG"] = df["SOG"].clip(0, 40)
    df["COG"] = df["COG"] % 360
    # drop physically impossible jumps (per vessel)
    keep = []
    for mmsi, g in df.groupby("MMSI"):
        dt_h = g.BaseDateTime.diff().dt.total_seconds().div(3600).replace(0, np.nan)
        d = haversine_km(g.LAT.shift(1), g.LON.shift(1), g.LAT, g.LON)
        spd = (d / dt_h / 1.852).fillna(0)
        keep.append(g[spd <= cfg.implausible_speed_kn * 4])   # only obvious teleports; segmenting handles the rest
    return pd.concat(keep, ignore_index=True) if keep else df


def preprocess_ais(raw: pd.DataFrame, cfg: AISConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Segment on gaps / implausible jumps, resample, engineer features. Returns (proc, segment_meta)."""
    df = clean_ais(raw, cfg)
    segs, meta_rows = [], []
    for mmsi, g in df.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").reset_index(drop=True)
        gap_min = g.BaseDateTime.diff().dt.total_seconds().fillna(0) / 60.0
        dt_h = (gap_min / 60.0).replace(0, np.nan)
        speed_kn = (haversine_km(g.LAT.shift(1), g.LON.shift(1), g.LAT, g.LON) / dt_h / 1.852).fillna(0)
        boundary = (gap_min > cfg.gap_segment_minutes) | (speed_kn > cfg.implausible_speed_kn)
        g["segment_id"] = boundary.cumsum()
        g["internal_gap_min"] = gap_min.where(~boundary, 0.0)
        segs.append(g)
        for sid, sg in g.groupby("segment_id"):
            meta_rows.append(dict(MMSI=mmsi, segment_id=sid, n_raw=len(sg)))
    raw2 = pd.concat(segs, ignore_index=True)
    need = cfg.history_steps + 3
    parts = []
    for (mmsi, sid), g in raw2.groupby(["MMSI", "segment_id"]):
        if len(g) < 4:
            continue
        gi = g.set_index("BaseDateTime")
        rule = f"{cfg.resample_minutes}min"
        sc = np.sin(np.deg2rad(gi.COG)).resample(rule).mean(); cc = np.cos(np.deg2rad(gi.COG)).resample(rule).mean()
        rs = gi[["LAT", "LON", "SOG"]].resample(rule).mean().interpolate(limit=3).dropna()
        if len(rs) < need:
            continue
        rs["COG"] = (np.rad2deg(np.arctan2(sc, cc)) % 360).reindex(rs.index).ffill().bfill()
        rs["internal_gap_min"] = gi["internal_gap_min"].resample(rule).max().reindex(rs.index).fillna(0)
        rs["MMSI"], rs["segment_id"] = mmsi, sid
        rs["VesselName"] = g.VesselName.iloc[0]; rs["VesselType"] = g.VesselType.iloc[0] if "VesselType" in g else "unknown"
        parts.append(rs.reset_index())
    if not parts:
        raise ValueError("No AIS segment is long enough after preprocessing.")
    proc = pd.concat(parts, ignore_index=True).sort_values(["MMSI", "segment_id", "BaseDateTime"]).reset_index(drop=True)
    out = []
    for _, g in proc.groupby(["MMSI", "segment_id"]):
        g = g.copy()
        lat0, lon0 = g.LAT.iloc[0], g.LON.iloc[0]
        g["east_m"] = (g.LON - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0)); g["north_m"] = (g.LAT - lat0) * 111_320.0
        g["cog_sin"], g["cog_cos"] = np.sin(np.deg2rad(g.COG)), np.cos(np.deg2rad(g.COG))
        hod = g.BaseDateTime.dt.hour + g.BaseDateTime.dt.minute / 60.0
        g["hod_sin"], g["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
        out.append(g)
    return pd.concat(out, ignore_index=True), pd.DataFrame(meta_rows)


def build_tracks(raw: pd.DataFrame, cfg: AISConfig) -> Dict[int, pd.DataFrame]:
    """Per-vessel raw tracks (irregular timestamps kept) for time-synchronous attribution."""
    df = clean_ais(raw, cfg)
    tracks = {}
    for mmsi, g in df.groupby("MMSI"):
        g = g.sort_values("BaseDateTime")
        if len(g) >= 2:
            tracks[int(mmsi)] = g.reset_index(drop=True)
    return tracks
