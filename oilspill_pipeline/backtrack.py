"""Lagrangian particle engine (NumPy): backward source reconstruction and forward drift.

* Heun (2nd-order) advection -- notably better than Euler for backward runs.
* Oil velocity = current + windage * wind, as vectors (see forcing.py / geo.py conventions).
* Random-walk horizontal diffusion with a physical diffusivity K (m^2/s).
* A per-member forcing-error ensemble (persistent wind/current speed & direction perturbations
  and per-particle windage) represents forcing uncertainty -- this is what widens the source
  region honestly instead of hiding uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import PhysicsConfig
from .forcing import DriftForcing
from .geo import advect_lonlat, haversine_km, lonlat_to_local_km


@dataclass
class Trajectory:
    times: List[pd.Timestamp]
    lon: np.ndarray                    # (n_times, n_particles)
    lat: np.ndarray
    direction: int                     # +1 forward in time, -1 backward
    engine: str = "numpy"
    meta: dict = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        return self.lon.shape[1]

    def mean_path(self) -> pd.DataFrame:
        rows = []
        for k, t in enumerate(self.times):
            x, y = lonlat_to_local_km(self.lon[k], self.lat[k], self.lon[k].mean(), self.lat[k].mean())
            rows.append(dict(t=t, lon=float(self.lon[k].mean()), lat=float(self.lat[k].mean()),
                             spread_km=float(np.sqrt(x.var() + y.var()))))
        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        return df

    def index_at(self, t) -> int:
        d = np.array([abs((tt - pd.Timestamp(t)).total_seconds()) for tt in self.times])
        return int(d.argmin())

    def cloud_at(self, t):
        k = self.index_at(t)
        return self.lon[k], self.lat[k]

    def bbox(self, pct=2.5, pad_deg=0.05):
        lo, la = self.lon.ravel(), self.lat.ravel()
        return (float(np.percentile(lo, pct) - pad_deg), float(np.percentile(la, pct) - pad_deg),
                float(np.percentile(lo, 100 - pct) + pad_deg), float(np.percentile(la, 100 - pct) + pad_deg))


class LagrangianEngine:
    name = "numpy"

    def __init__(self, forcing: DriftForcing, phys: PhysicsConfig, windage_range=(0.01, 0.04), seed: int = 0,
                 perturb: bool = True):
        self.f, self.p, self.windage_range, self.seed, self.perturb = forcing, phys, windage_range, seed, perturb

    def _members(self, n, rng):
        p = self.p
        if self.perturb:
            return dict(
                windage=rng.uniform(*self.windage_range, n),
                wind_scale=np.clip(rng.normal(1.0, p.wind_speed_rel_sigma, n), 0.2, None),
                wind_rot=rng.normal(0.0, p.wind_dir_sigma_deg, n),
                cur_scale=np.clip(rng.normal(1.0, p.current_speed_rel_sigma, n), 0.2, None),
                cur_rot=rng.normal(0.0, p.current_dir_sigma_deg, n))
        return dict(windage=np.full(n, self.f.windage_frac), wind_scale=np.ones(n), wind_rot=np.zeros(n),
                    cur_scale=np.ones(n), cur_rot=np.zeros(n))

    def run(self, lon0, lat0, t_start, hours: float, direction: int = -1, n: Optional[int] = None,
            init_sigma_m: float = 0.0) -> Trajectory:
        assert direction in (-1, +1)
        rng = np.random.default_rng(self.seed)
        lon0 = np.atleast_1d(np.asarray(lon0, dtype=float)); lat0 = np.atleast_1d(np.asarray(lat0, dtype=float))
        if n is not None and lon0.size == 1:
            lon0 = np.full(n, lon0[0]); lat0 = np.full(n, lat0[0])
        N = lon0.size
        lon, lat = lon0.copy(), lat0.copy()
        if init_sigma_m > 0:
            lon, lat = advect_lonlat(lon, lat, rng.normal(0, init_sigma_m, N), rng.normal(0, init_sigma_m, N), 1.0)
        mem = self._members(N, rng)
        dt_s = self.p.dt_min * 60.0
        n_steps = int(np.ceil(hours * 3600.0 / dt_s))
        sdt = direction * dt_s
        sig = np.sqrt(2.0 * max(self.p.horizontal_diffusivity_m2s, 0.0) * dt_s)
        t = pd.Timestamp(t_start)
        times = [t]; L = [lon.copy()]; A = [lat.copy()]

        def vel(lo, la, tt):
            return self.f.oil_velocity(la, lo, tt, windage=mem["windage"], wind_scale=mem["wind_scale"],
                                       wind_rot_deg=mem["wind_rot"], cur_scale=mem["cur_scale"], cur_rot_deg=mem["cur_rot"])

        for _ in range(n_steps):
            u1, v1 = vel(lon, lat, t)
            lo1, la1 = advect_lonlat(lon, lat, u1, v1, sdt)
            t_next = t + pd.Timedelta(seconds=sdt)
            u2, v2 = vel(lo1, la1, t_next)
            lon, lat = advect_lonlat(lon, lat, 0.5 * (u1 + u2), 0.5 * (v1 + v2), sdt)
            if sig > 0:
                lon, lat = advect_lonlat(lon, lat, rng.normal(0, sig, N), rng.normal(0, sig, N), 1.0)
            t = t_next
            times.append(t); L.append(lon.copy()); A.append(lat.copy())
        return Trajectory(times, np.array(L), np.array(A), direction, self.name,
                          meta=dict(dt_min=self.p.dt_min, K=self.p.horizontal_diffusivity_m2s, perturb=self.perturb,
                                    forcing=self.f.description))


def source_centroid_km_difference(a: Trajectory, b: Trajectory, age_hours: float) -> float:
    """Cross-engine agreement: distance (km) between the two source-cloud centroids after age_hours."""
    ta = a.times[0] - pd.Timedelta(hours=age_hours)
    la, lb = a.cloud_at(ta), b.cloud_at(b.times[0] - pd.Timedelta(hours=age_hours))
    return float(haversine_km(la[1].mean(), la[0].mean(), lb[1].mean(), lb[0].mean()))
