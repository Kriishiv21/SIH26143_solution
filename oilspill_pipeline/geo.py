"""Geodesy helpers and -- critically -- the direction/unit conventions used everywhere.

CONVENTIONS (the single most common source of silent bugs in drift modelling)
---------------------------------------------------------------------------
* u = eastward component (m/s), v = northward component (m/s).  Files (GRIB u10/v10, current
  CSV water_u/water_v, NetCDF) already store *components*; they are used AS-IS -- never converted
  through a direction.
* Bearings are degrees clockwise from NORTH.
* Wind, meteorological convention  : direction the wind blows FROM.
* Current / drift, oceanographic   : direction the water/oil moves TOWARD.
  toward = (from + 180) % 360.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

R_EARTH_M = 6_371_008.8
_EPOCH = pd.Timestamp("1970-01-01")


# ----------------------------------------------------------------------------- conventions
def met_wind_to_uv(speed, dir_from_deg):
    """Meteorological wind (speed, direction wind blows FROM) -> (u, v) components in m/s."""
    th = np.deg2rad(np.asarray(dir_from_deg, dtype=float))
    s = np.asarray(speed, dtype=float)
    return -s * np.sin(th), -s * np.cos(th)


def current_to_uv(speed, dir_to_deg):
    """Oceanographic current (speed, direction flowing TOWARD) -> (u, v) components in m/s."""
    th = np.deg2rad(np.asarray(dir_to_deg, dtype=float))
    s = np.asarray(speed, dtype=float)
    return s * np.sin(th), s * np.cos(th)


def uv_to_speed_dir_to(u, v):
    """(u, v) -> (speed, bearing the vector points TOWARD, degrees clockwise from north)."""
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    return np.hypot(u, v), np.rad2deg(np.arctan2(u, v)) % 360.0


def uv_to_speed_dir_from(u, v):
    """(u, v) -> (speed, meteorological direction the wind comes FROM)."""
    sp, to = uv_to_speed_dir_to(u, v)
    return sp, (to + 180.0) % 360.0


def rotate_uv(u, v, angle_deg):
    """Rotate vectors clockwise (compass sense) by angle_deg."""
    a = np.deg2rad(angle_deg)
    return u * np.cos(a) + v * np.sin(a), -u * np.sin(a) + v * np.cos(a)


# ----------------------------------------------------------------------------- time
def to_epoch_s(ts) -> float:
    """Timestamp-like -> seconds since 1970 (UTC, tz-naive after conversion)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return (t - _EPOCH).total_seconds()


def from_epoch_s(sec: float) -> pd.Timestamp:
    return _EPOCH + pd.Timedelta(seconds=float(sec))


def naive_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


# ----------------------------------------------------------------------------- distances
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(lambda x: np.deg2rad(np.asarray(x, dtype=float)), (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH_M / 1000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def advect_lonlat(lon, lat, u, v, dt_s):
    """Move points by (u, v) m/s for dt_s seconds (dt_s may be negative). Spherical, per-point cos(lat)."""
    lat_r = np.deg2rad(lat)
    dlat = np.rad2deg(v * dt_s / R_EARTH_M)
    dlon = np.rad2deg(u * dt_s / (R_EARTH_M * np.maximum(np.cos(lat_r), 1e-6)))
    return lon + dlon, lat + dlat


def offset_lonlat_m(lon, lat, east_m, north_m):
    return advect_lonlat(lon, lat, east_m, north_m, 1.0)


def lonlat_to_local_km(lon, lat, lon0, lat0):
    x = np.deg2rad(np.asarray(lon) - lon0) * np.cos(np.deg2rad(lat0)) * R_EARTH_M / 1000.0
    y = np.deg2rad(np.asarray(lat) - lat0) * R_EARTH_M / 1000.0
    return x, y


def destination_point(lon, lat, bearing_deg, dist_km):
    th = np.deg2rad(bearing_deg)
    return advect_lonlat(lon, lat, np.sin(th) * dist_km * 1000.0, np.cos(th) * dist_km * 1000.0, 1.0)


# ----------------------------------------------------------------------------- georeferencing
@dataclass
class GeoRef:
    """Pixel <-> lon/lat for a raster. Works for EPSG:4326 and for any projected CRS (via pyproj)."""
    transform: "object"                # affine.Affine
    crs: str = "EPSG:4326"
    shape: Tuple[int, int] = (0, 0)

    def _to_lonlat(self, x, y):
        if str(self.crs).upper() in ("EPSG:4326", "OGC:CRS84", "WGS84"):
            return x, y
        from pyproj import Transformer
        tr = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return tr.transform(x, y)

    def _from_lonlat(self, lon, lat):
        if str(self.crs).upper() in ("EPSG:4326", "OGC:CRS84", "WGS84"):
            return lon, lat
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        return tr.transform(lon, lat)

    def pixel_to_lonlat(self, rows, cols):
        """Pixel CENTRES -> (lon, lat)."""
        rows = np.asarray(rows, dtype=float); cols = np.asarray(cols, dtype=float)
        x, y = self.transform * (cols + 0.5, rows + 0.5)
        return self._to_lonlat(np.asarray(x), np.asarray(y))

    def lonlat_to_pixel(self, lon, lat):
        """(lon, lat) -> fractional (row, col) of pixel centres."""
        x, y = self._from_lonlat(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        inv = ~self.transform
        cols, rows = inv * (x, y)
        return np.asarray(rows) - 0.5, np.asarray(cols) - 0.5

    def pixel_size_km(self, row: float, col: float) -> Tuple[float, float]:
        """Local (dx_km, dy_km) of one pixel at (row, col)."""
        lon0, lat0 = self.pixel_to_lonlat(row, col)
        lon1, lat1 = self.pixel_to_lonlat(row, col + 1)
        lon2, lat2 = self.pixel_to_lonlat(row + 1, col)
        return float(haversine_km(lat0, lon0, lat1, lon1)), float(haversine_km(lat0, lon0, lat2, lon2))

    def bounds_lonlat(self):
        h, w = self.shape
        rr = np.array([0, 0, h - 1, h - 1]); cc = np.array([0, w - 1, 0, w - 1])
        lon, lat = self.pixel_to_lonlat(rr, cc)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def grid_from_bbox(lon_min, lat_max, pixel_deg, shape) -> GeoRef:
    from affine import Affine
    return GeoRef(Affine.translation(lon_min, lat_max) * Affine.scale(pixel_deg, -pixel_deg), "EPSG:4326", tuple(shape))
