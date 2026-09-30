"""Wind + surface-current forcing, with correct conventions and correct windage.

Oil velocity = surface current + windage_frac * wind10   (both as (u, v) vectors)

The ORIGINAL notebook multiplied the *current speed* by (1 + windage) -- a fudge that
(a) ignores the wind entirely and (b) scales the wrong vector.  Here wind enters as a vector,
scaled by the windage fraction (default 3 %, per-particle 1-4 %), and added to the current.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ForcingConfig
from .geo import met_wind_to_uv, current_to_uv, to_epoch_s

log = logging.getLogger(__name__)


class ForcingCoverageError(RuntimeError):
    """Raised when forcing data does not cover the requested space/time -- never silently extrapolated."""


# ----------------------------------------------------------------------------- fields
class VectorField:
    name = "field"

    def uv(self, lat, lon, t) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def check_coverage(self, bbox, t0, t1):  # pragma: no cover - overridden
        return {}


class UniformField(VectorField):
    def __init__(self, u: float, v: float, name: str = "uniform"):
        self.u, self.v, self.name = float(u), float(v), name

    def uv(self, lat, lon, t):
        lat = np.asarray(lat, dtype=float)
        return np.full(lat.shape, self.u), np.full(lat.shape, self.v)


def _nearest_fill(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs (land / gaps) in the last two axes with the nearest valid value, per time slice."""
    from scipy import ndimage
    out = arr.copy()
    for k in range(out.shape[0]):
        m = np.isnan(out[k])
        if m.all():
            out[k] = 0.0
        elif m.any():
            idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
            out[k] = out[k][tuple(idx)]
    return out


class GriddedField(VectorField):
    """Regular (time, lat, lon) grid, bilinear in space, linear in time, NaN-safe.

    Never extrapolates in time (raises ForcingCoverageError); clamps to the domain edge in space
    and counts how often that happens so it can be reported.
    """

    def __init__(self, times_s, lats, lons, u, v, name="gridded", allow_time_clamp=False):
        order_t = np.argsort(times_s); order_la = np.argsort(lats); order_lo = np.argsort(lons)
        self.t = np.asarray(times_s, dtype=float)[order_t]
        self.lat = np.asarray(lats, dtype=float)[order_la]
        self.lon = np.asarray(lons, dtype=float)[order_lo]
        u = np.asarray(u, dtype=np.float64)[order_t][:, order_la][:, :, order_lo]
        v = np.asarray(v, dtype=np.float64)[order_t][:, order_la][:, :, order_lo]
        self.nan_fraction = float(np.isnan(u).mean())
        self.u, self.v = _nearest_fill(u), _nearest_fill(v)
        self.name, self.allow_time_clamp = name, allow_time_clamp
        self.n_out_of_domain = 0
        self.n_queries = 0

    def bbox(self):
        return float(self.lon.min()), float(self.lat.min()), float(self.lon.max()), float(self.lat.max())

    def time_range(self):
        return pd.Timestamp(self.t[0], unit="s"), pd.Timestamp(self.t[-1], unit="s")

    def check_coverage(self, bbox, t0, t1):
        lon_min, lat_min, lon_max, lat_max = bbox
        g = self.bbox()
        problems = []
        if lon_min < g[0] or lat_min < g[1] or lon_max > g[2] or lat_max > g[3]:
            problems.append(f"space: need lon[{lon_min:.3f},{lon_max:.3f}] lat[{lat_min:.3f},{lat_max:.3f}] "
                            f"but {self.name} covers lon[{g[0]:.3f},{g[2]:.3f}] lat[{g[1]:.3f},{g[3]:.3f}]")
        ts, te = to_epoch_s(t0), to_epoch_s(t1)
        if ts < self.t[0] - 1 or te > self.t[-1] + 1:
            a, b = self.time_range()
            problems.append(f"time: need {pd.Timestamp(t0)} .. {pd.Timestamp(t1)} but {self.name} covers {a} .. {b}")
        return dict(ok=not problems, problems=problems)

    def uv(self, lat, lon, t):
        lat = np.atleast_1d(np.asarray(lat, dtype=float)); lon = np.atleast_1d(np.asarray(lon, dtype=float))
        ts = to_epoch_s(t)
        if not (self.t[0] - 1 <= ts <= self.t[-1] + 1):
            if not self.allow_time_clamp:
                a, b = self.time_range()
                raise ForcingCoverageError(f"{self.name}: t={pd.Timestamp(t)} outside forcing time range {a} .. {b}")
            ts = float(np.clip(ts, self.t[0], self.t[-1]))
        ts = float(np.clip(ts, self.t[0], self.t[-1]))
        self.n_queries += lat.size
        self.n_out_of_domain += int(((lat < self.lat[0]) | (lat > self.lat[-1]) | (lon < self.lon[0]) | (lon > self.lon[-1])).sum())
        la = np.clip(lat, self.lat[0], self.lat[-1]); lo = np.clip(lon, self.lon[0], self.lon[-1])

        def _frac(axis, x):
            if axis.size == 1:
                return np.zeros(x.shape, dtype=int), np.zeros(x.shape)
            i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, axis.size - 2)
            w = (x - axis[i]) / (axis[i + 1] - axis[i])
            return i, np.clip(w, 0.0, 1.0)

        it = np.searchsorted(self.t, ts, side="right") - 1
        it = int(np.clip(it, 0, max(self.t.size - 2, 0)))
        wt = 0.0 if self.t.size == 1 else float(np.clip((ts - self.t[it]) / (self.t[it + 1] - self.t[it]), 0, 1))
        it1 = min(it + 1, self.t.size - 1)
        ia, wa = _frac(self.lat, la); io, wo = _frac(self.lon, lo)
        ia1 = np.minimum(ia + 1, self.lat.size - 1); io1 = np.minimum(io + 1, self.lon.size - 1)

        def interp(F):
            def plane(k):
                return ((1 - wa) * (1 - wo) * F[k, ia, io] + (1 - wa) * wo * F[k, ia, io1]
                        + wa * (1 - wo) * F[k, ia1, io] + wa * wo * F[k, ia1, io1])
            return (1 - wt) * plane(it) + wt * plane(it1)

        return interp(self.u), interp(self.v)


@dataclass
class DriftForcing:
    """Combines a wind field and a current field into an oil-drift velocity."""
    wind: VectorField
    current: VectorField
    windage_frac: float = 0.03
    description: str = ""

    def wind_uv(self, lat, lon, t):
        return self.wind.uv(lat, lon, t)

    def current_uv(self, lat, lon, t):
        return self.current.uv(lat, lon, t)

    def oil_velocity(self, lat, lon, t, windage=None, wind_scale=1.0, wind_rot_deg=None,
                     cur_scale=1.0, cur_rot_deg=None):
        """Oil velocity (u, v) m/s = current + windage * wind. Optional per-particle perturbations."""
        from .geo import rotate_uv
        uc, vc = self.current.uv(lat, lon, t)
        uw, vw = self.wind.uv(lat, lon, t)
        if cur_rot_deg is not None:
            uc, vc = rotate_uv(uc, vc, cur_rot_deg)
        if wind_rot_deg is not None:
            uw, vw = rotate_uv(uw, vw, wind_rot_deg)
        w = self.windage_frac if windage is None else windage
        return cur_scale * uc + w * wind_scale * uw, cur_scale * vc + w * wind_scale * vw

    def check_coverage(self, bbox, t0, t1):
        rep = {}
        for nm, f in (("wind", self.wind), ("current", self.current)):
            rep[nm] = f.check_coverage(bbox, t0, t1) if hasattr(f, "check_coverage") and isinstance(f, GriddedField) else dict(ok=True, problems=[])
        rep["ok"] = all(r["ok"] for r in rep.values())
        return rep

    def assert_coverage(self, bbox, t0, t1):
        rep = self.check_coverage(bbox, t0, t1)
        if not rep["ok"]:
            msgs = [f"[{k}] {p}" for k, r in rep.items() if isinstance(r, dict) for p in r.get("problems", [])]
            raise ForcingCoverageError("Forcing does not cover this case:\n  " + "\n  ".join(msgs))
        return rep


# ----------------------------------------------------------------------------- loaders
_U_WIND = ("u10", "10u", "u10m", "eastward_wind", "air_u", "x_wind", "uas")
_V_WIND = ("v10", "10v", "v10m", "northward_wind", "air_v", "y_wind", "vas")
_U_CUR = ("water_u", "uo", "u", "eastward_sea_water_velocity", "x_sea_water_velocity", "water_u_mps", "ugos")
_V_CUR = ("water_v", "vo", "v", "northward_sea_water_velocity", "y_sea_water_velocity", "water_v_mps", "vgos")


def _pick(ds, names, what):
    for n in names:
        if n in ds.variables:
            return n
    raise KeyError(f"No {what} variable found. Looked for {names}; dataset has {list(ds.variables)}")


def _coord(ds, names, what):
    for n in names:
        if n in ds.coords or n in ds.variables:
            return n
    raise KeyError(f"No {what} coordinate. Looked for {names}; has {list(ds.coords)}")


def _norm_lon(lon):
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def gridded_from_dataset(ds, kind: str, name: str, allow_time_clamp=False) -> GriddedField:
    """xarray Dataset -> GriddedField (handles lat-descending, 0..360 lon, singleton dims)."""
    un, vn = (_U_WIND, _V_WIND) if kind == "wind" else (_U_CUR, _V_CUR)
    uvar, vvar = _pick(ds, un, f"{kind} U"), _pick(ds, vn, f"{kind} V")
    if kind == "current" and ("bottom" in uvar.lower() or "bottom" in vvar.lower()):
        raise ForcingCoverageError(f"Refusing bottom-current variables ({uvar}, {vvar}) for floating oil.")
    tn = _coord(ds, ("time", "valid_time", "t"), "time")
    lan = _coord(ds, ("latitude", "lat", "y"), "latitude")
    lon_n = _coord(ds, ("longitude", "lon", "x"), "longitude")
    U, V = ds[uvar], ds[vvar]
    # drop singleton dims like depth / step / number, keep (time, lat, lon)
    for d in list(U.dims):
        if d not in (tn, lan, lon_n) and U.sizes[d] == 1:
            U, V = U.isel({d: 0}), V.isel({d: 0})
    U, V = U.transpose(tn, lan, lon_n), V.transpose(tn, lan, lon_n)
    times = pd.to_datetime(ds[tn].values)
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    times_s = np.array([to_epoch_s(t) for t in times])
    return GriddedField(times_s, ds[lan].values, _norm_lon(ds[lon_n].values), U.values, V.values,
                        name=name, allow_time_clamp=allow_time_clamp)


def load_wind(path: str, allow_time_clamp=False, bbox=None) -> GriddedField:
    import xarray as xr
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wind file not found: {path}")
    if path.lower().endswith((".grib", ".grib2", ".grb", ".grb2")):
        idx = os.path.join(os.environ.get("TMPDIR", "/tmp"), os.path.basename(path) + ".idx")
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": idx})
    else:
        ds = xr.open_dataset(path)
    if bbox is not None:
        ds = _subset(ds, bbox)
    return gridded_from_dataset(ds, "wind", f"wind[{os.path.basename(path)}]", allow_time_clamp)


def _subset(ds, bbox, pad=0.5):
    lon_min, lat_min, lon_max, lat_max = bbox
    lan = _coord(ds, ("latitude", "lat", "y"), "latitude"); lon_n = _coord(ds, ("longitude", "lon", "x"), "longitude")
    lons = _norm_lon(ds[lon_n].values)
    m_lon = (lons >= lon_min - pad) & (lons <= lon_max + pad)
    la = ds[lan].values
    m_lat = (la >= lat_min - pad) & (la <= lat_max + pad)
    return ds.isel({lon_n: np.where(m_lon)[0], lan: np.where(m_lat)[0]})


def load_current_csv(path_or_dir: str, pattern="Day_*.csv", bbox=None, allow_time_clamp=False) -> GriddedField:
    """Long-format CSVs: date,time_utc,latitude,longitude,water_u_mps,water_v_mps (as in the original notebook)."""
    files = sorted(glob.glob(os.path.join(path_or_dir, pattern))) if os.path.isdir(path_or_dir) else [path_or_dir]
    if not files:
        raise FileNotFoundError(f"No current CSVs matching {pattern!r} in {path_or_dir}")
    frames = []
    for fp in files:
        head = pd.read_csv(fp, nrows=0).columns.tolist()
        if any("bottom" in c.lower() for c in head):
            raise ForcingCoverageError(f"{fp} contains bottom-current columns; refusing to use as surface forcing.")
        df = pd.read_csv(fp, usecols=["date", "time_utc", "latitude", "longitude", "water_u_mps", "water_v_mps"])
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time_utc"].astype(str), errors="coerce")
        for c in ("latitude", "longitude", "water_u_mps", "water_v_mps"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime", "latitude", "longitude"])
        df["longitude"] = _norm_lon(df["longitude"].values)
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            df = df[df.latitude.between(lat_min - 0.5, lat_max + 0.5) & df.longitude.between(lon_min - 0.5, lon_max + 0.5)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["datetime", "latitude", "longitude"])
    if df.empty:
        raise ForcingCoverageError("Current CSVs contain no rows inside the requested region.")
    times = np.sort(df["datetime"].unique()); lats = np.sort(df["latitude"].unique()); lons = np.sort(df["longitude"].unique())
    U = np.full((times.size, lats.size, lons.size), np.nan); V = U.copy()
    ti = np.searchsorted(times, df["datetime"].values); la = np.searchsorted(lats, df["latitude"].values); lo = np.searchsorted(lons, df["longitude"].values)
    U[ti, la, lo] = df["water_u_mps"].values; V[ti, la, lo] = df["water_v_mps"].values
    times_s = np.array([to_epoch_s(pd.Timestamp(t)) for t in times])
    return GriddedField(times_s, lats, lons, U, V, name="current[csv]", allow_time_clamp=allow_time_clamp)


def load_current(path: str, pattern="Day_*.csv", bbox=None, allow_time_clamp=False, verified_surface=False) -> GriddedField:
    if not verified_surface:
        log.warning("current_verified_surface=False: assuming the current product is SURFACE (0 m). "
                    "Set it True only after you have checked; bottom-current variables are refused regardless.")
    if os.path.isdir(path) or path.lower().endswith(".csv"):
        return load_current_csv(path, pattern, bbox, allow_time_clamp)
    import xarray as xr
    ds = xr.open_dataset(path)
    if bbox is not None:
        ds = _subset(ds, bbox)
    return gridded_from_dataset(ds, "current", f"current[{os.path.basename(path)}]", allow_time_clamp)


def build_forcing(cfg: ForcingConfig, bbox=None) -> DriftForcing:
    """Factory: hard-coded demo forcing or real files, behind one call."""
    if cfg.source == "hardcoded":
        uw, vw = met_wind_to_uv(cfg.hardcoded_wind_speed_ms, cfg.hardcoded_wind_dir_from_deg)
        uc, vc = current_to_uv(cfg.hardcoded_current_speed_ms, cfg.hardcoded_current_dir_to_deg)
        desc = (f"HARD-CODED (synthetic) wind {cfg.hardcoded_wind_speed_ms} m/s FROM {cfg.hardcoded_wind_dir_from_deg} deg, "
                f"current {cfg.hardcoded_current_speed_ms} m/s TOWARD {cfg.hardcoded_current_dir_to_deg} deg")
        return DriftForcing(UniformField(uw, vw, "wind[hardcoded]"), UniformField(uc, vc, "current[hardcoded]"), cfg.windage_frac, desc)
    if cfg.source == "files":
        if not cfg.wind_path or not cfg.current_path:
            raise ValueError("forcing.source='files' requires forcing.wind_path and forcing.current_path")
        w = load_wind(cfg.wind_path, cfg.allow_time_clamp, bbox)
        c = load_current(cfg.current_path, cfg.current_glob, bbox, cfg.allow_time_clamp, cfg.current_verified_surface)
        return DriftForcing(w, c, cfg.windage_frac, f"FILES wind={cfg.wind_path} current={cfg.current_path}")
    raise ValueError(f"Unknown forcing.source {cfg.source!r}")


# ----------------------------------------------------------------------------- export for engines
def export_forcing_netcdf(forcing: DriftForcing, bbox, t0, t1, wind_path: str, current_path: str,
                          step_hours: float = 1.0, grid_deg: float = 0.1):
    """Write CF-compliant NetCDFs (same variable names/attrs the original PyGNOME notebook used)
    so OpenDrift/OpenOil and PyGNOME read exactly the forcing the NumPy engine used."""
    import xarray as xr
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = np.arange(lat_min - grid_deg, lat_max + 1.5 * grid_deg, grid_deg)
    lons = np.arange(lon_min - grid_deg, lon_max + 1.5 * grid_deg, grid_deg)
    times = pd.date_range(pd.Timestamp(t0).floor("h"), pd.Timestamp(t1).ceil("h"), freq=f"{int(step_hours * 60)}min")
    LON, LAT = np.meshgrid(lons, lats)
    Uw = np.zeros((len(times), lats.size, lons.size), np.float32); Vw = Uw.copy(); Uc = Uw.copy(); Vc = Uw.copy()

    def _t_eval(field, t):
        # The caller has already asserted coverage of [t0, t1]; the floor/ceil to whole hours may
        # overshoot by < step_hours, so clamp ONLY that overshoot (never a real coverage gap).
        if isinstance(field, GriddedField):
            lo, hi = field.t[0], field.t[-1]
            ts = to_epoch_s(t)
            if ts < lo - step_hours * 3600 or ts > hi + step_hours * 3600:
                return t  # genuine gap -> let uv() raise ForcingCoverageError
            return pd.Timestamp(min(max(ts, lo), hi), unit="s")
        return t

    for k, t in enumerate(times):
        a, b = forcing.wind.uv(LAT.ravel(), LON.ravel(), _t_eval(forcing.wind, t))
        c, d = forcing.current.uv(LAT.ravel(), LON.ravel(), _t_eval(forcing.current, t))
        Uw[k], Vw[k] = a.reshape(LAT.shape), b.reshape(LAT.shape)
        Uc[k], Vc[k] = c.reshape(LAT.shape), d.reshape(LAT.shape)
    coords = dict(time=times, lat=lats, lon=lons)
    wds = xr.Dataset({"air_u": (("time", "lat", "lon"), Uw), "air_v": (("time", "lat", "lon"), Vw)}, coords=coords)
    wds["air_u"].attrs.update(standard_name="eastward_wind", units="m s-1")
    wds["air_v"].attrs.update(standard_name="northward_wind", units="m s-1")
    cds = xr.Dataset({"water_u": (("time", "lat", "lon"), Uc), "water_v": (("time", "lat", "lon"), Vc)}, coords=coords)
    cds["water_u"].attrs.update(standard_name="eastward_sea_water_velocity", units="m s-1")
    cds["water_v"].attrs.update(standard_name="northward_sea_water_velocity", units="m s-1")
    for ds in (wds, cds):
        ds["lat"].attrs.update(standard_name="latitude", units="degrees_north"); ds["lon"].attrs.update(standard_name="longitude", units="degrees_east")
        ds.attrs["Conventions"] = "CF-1.8"
    os.makedirs(os.path.dirname(wind_path) or ".", exist_ok=True)
    wds.to_netcdf(wind_path); cds.to_netcdf(current_path)
    return wind_path, current_path
