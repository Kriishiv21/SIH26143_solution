"""Pluggable drift engines behind one interface:  engine.run(lon0, lat0, t_start, hours, direction, n, init_sigma_m) -> Trajectory

  numpy    -- always available, forward + backward, fully tested (backtrack.LagrangianEngine)
  openoil  -- OpenDrift's OpenOil, FORWARD only (oil weathering physics); reads the exported NetCDFs
  pygnome  -- NOAA PyGNOME, BACKWARD (run_backwards) via a subprocess in its own conda env

OpenOil/PyGNOME consume NetCDF exports of the *same* forcing the NumPy engine uses, so cross-engine
differences reflect the transport/physics model, not different input data.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from .backtrack import LagrangianEngine, Trajectory
from .config import PhysicsConfig
from .forcing import DriftForcing, export_forcing_netcdf

log = logging.getLogger(__name__)


class ForcingFiles:
    """Lazily exports forcing to CF NetCDF once per case window; shared by OpenOil and PyGNOME."""

    def __init__(self, forcing: DriftForcing, bbox, t0, t1, workdir: str, tag: str = "case"):
        self.forcing, self.bbox, self.t0, self.t1 = forcing, bbox, pd.Timestamp(t0), pd.Timestamp(t1)
        self.dir, self.tag = workdir, tag
        self._paths = None

    def paths(self):
        if self._paths is None:
            os.makedirs(self.dir, exist_ok=True)
            self._paths = export_forcing_netcdf(self.forcing, self.bbox, self.t0, self.t1,
                                                os.path.join(self.dir, f"{self.tag}_wind.nc"),
                                                os.path.join(self.dir, f"{self.tag}_current.nc"))
        return self._paths


def openoil_available() -> bool:
    try:
        from opendrift.models.openoil import OpenOil  # noqa: F401
        return True
    except Exception:
        return False


def _configure_open_ocean(o):
    """No landmask download (open-water case); constant fallbacks for variables we have no reader for."""
    o.set_config("general:use_auto_landmask", False)
    o.set_config("environment:constant:land_binary_mask", 0)
    o.set_config("general:coastline_action", "none")
    o.set_config("drift:vertical_mixing", False)
    for k, v in (("sea_surface_wave_significant_height", 0.5), ("sea_surface_wave_stokes_drift_x_velocity", 0.0),
                 ("sea_surface_wave_stokes_drift_y_velocity", 0.0), ("sea_surface_wave_period_at_variance_spectral_density_maximum", 5.0),
                 ("sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment", 5.0),
                 ("sea_water_temperature", 20.0), ("sea_water_salinity", 38.0)):
        try:
            o.set_config(f"environment:fallback:{k}", v)
        except Exception:
            pass


class OpenOilEngine:
    """Forward drift + weathering with OpenDrift/OpenOil."""
    name = "openoil"

    def __init__(self, files: ForcingFiles, phys: PhysicsConfig, oil_type: Optional[str] = None, seed: int = 0):
        if not openoil_available():
            raise RuntimeError("opendrift is not installed (pip install opendrift)")
        self.files, self.p, self.seed = files, phys, seed
        self.oil_type = oil_type or phys.oil_types[0]

    def run(self, lon0, lat0, t_start, hours, direction=+1, n=None, init_sigma_m=0.0) -> Trajectory:
        if direction != +1:
            raise ValueError("OpenOil engine is forward-only; use numpy/pygnome for backward runs")
        from opendrift.models.openoil import OpenOil
        from opendrift.readers import reader_netCDF_CF_generic
        lon0 = np.atleast_1d(lon0).astype(float); lat0 = np.atleast_1d(lat0).astype(float)
        N = lon0.size if n is None else n
        if lon0.size == 1 and n:
            lon0, lat0 = np.full(N, lon0[0]), np.full(N, lat0[0])
        wpath, cpath = self.files.paths()
        o = OpenOil(loglevel=50, weathering_model="noaa")
        o.add_reader([reader_netCDF_CF_generic.Reader(wpath), reader_netCDF_CF_generic.Reader(cpath)])
        _configure_open_ocean(o)
        o.set_config("processes:evaporation", True)
        o.set_config("processes:emulsification", True)
        o.set_config("processes:dispersion", False)
        o.set_config("drift:current_uncertainty", 0.05)
        o.set_config("drift:wind_uncertainty", 1.0)
        o.set_config("seed:wind_drift_factor", float(0.03))
        o.set_config("environment:fallback:x_wind", 0.0); o.set_config("environment:fallback:y_wind", 0.0)
        o.set_config("environment:fallback:x_sea_water_velocity", 0.0); o.set_config("environment:fallback:y_sea_water_velocity", 0.0)
        o.set_config("environment:fallback:horizontal_diffusivity", float(self.p.horizontal_diffusivity_m2s))
        o.seed_elements(lon=lon0, lat=lat0, number=N, time=pd.Timestamp(t_start).to_pydatetime(), oil_type=self.oil_type,
                        m3_per_hour=1.0, radius=float(max(init_sigma_m, 1.0)))
        dt_s = self.p.dt_min * 60.0
        o.run(duration=pd.Timedelta(hours=hours).to_pytimedelta(), time_step=dt_s, time_step_output=dt_s)
        ds = o.result
        lon = ds["lon"].values.T if ds["lon"].dims[0] == "trajectory" else ds["lon"].values
        lat = ds["lat"].values.T if ds["lat"].dims[0] == "trajectory" else ds["lat"].values
        # result dims are (trajectory, time); we want (time, particle)
        lon = np.asarray(ds["lon"].transpose("time", "trajectory").values); lat = np.asarray(ds["lat"].transpose("time", "trajectory").values)
        times = list(pd.to_datetime(ds["time"].values))
        # particles that never activated / deactivated are NaN -> forward-fill in time, drop all-NaN members
        keep = ~np.isnan(lon).all(axis=0)
        lon, lat = pd.DataFrame(lon[:, keep]).ffill().bfill().values, pd.DataFrame(lat[:, keep]).ffill().bfill().values
        return Trajectory(times, lon, lat, +1, self.name, meta=dict(oil_type=self.oil_type))

    def weathering_table(self, lon, lat, t_start, ages_hours, n=20) -> pd.DataFrame:
        """OpenOil weathering state (evaporated fraction, water uptake, viscosity, density) vs oil age.
        Informational: screens which oil type/age is consistent with a later measurement or SAR appearance."""
        from opendrift.models.openoil import OpenOil
        from opendrift.readers import reader_netCDF_CF_generic
        wpath, cpath = self.files.paths()
        o = OpenOil(loglevel=50, weathering_model="noaa")
        o.add_reader([reader_netCDF_CF_generic.Reader(wpath), reader_netCDF_CF_generic.Reader(cpath)])
        _configure_open_ocean(o)
        o.set_config("processes:evaporation", True); o.set_config("processes:emulsification", True)
        o.seed_elements(lon=float(lon), lat=float(lat), number=n, time=pd.Timestamp(t_start).to_pydatetime(),
                        oil_type=self.oil_type, m3_per_hour=1.0, radius=100.0)
        o.run(duration=pd.Timedelta(hours=float(max(ages_hours))).to_pytimedelta(), time_step=self.p.dt_min * 60.0, time_step_output=3600.0)
        ds = o.result; t = pd.to_datetime(ds["time"].values); rows = []
        for a in ages_hours:
            k = int(np.argmin(np.abs((t - (pd.Timestamp(t_start) + pd.Timedelta(hours=float(a)))).total_seconds())))
            g = lambda v: float(ds[v].isel(time=k).mean(skipna=True).item()) if v in ds else float("nan")
            rows.append(dict(oil_type=self.oil_type, age_h=a, fraction_evaporated=g("fraction_evaporated"),
                             water_fraction=g("water_fraction"), viscosity_m2s=g("viscosity"), density_kgm3=g("density")))
        return pd.DataFrame(rows)


_PYGNOME_WORKER = r'''
import sys, json
import numpy as np, pandas as pd
import gnome.scripting as gs
cfg = json.load(open(sys.argv[1]))
t_obs = pd.Timestamp(cfg["acquisition_time"]).to_pydatetime()
pos = np.asarray(cfg["positions"], dtype=float)                       # lon, lat
wind = gs.GridWind.from_netCDF(filename=cfg["wind_nc"])
cur = gs.GridCurrent.from_netCDF(filename=cfg["current_nc"])
model = gs.Model(start_time=t_obs, duration=gs.hours(cfg["hours"]), time_step=cfg["time_step_s"],
                 uncertain=False, cache_enabled=False, run_backwards=True)
model.movers += gs.WindMover(wind); model.movers += gs.CurrentMover(cur)
start = np.column_stack([pos[:, 0], pos[:, 1], np.zeros(len(pos))])
model.spills += gs.spatial_release_spill(start_positions=start, release_time=t_obs, amount=0, units="kg",
                                          windage_range=(0.03, 0.03), windage_persist=-1)
traj_lon, traj_lat, times = [], [], []
for _ in model:
    p = np.asarray(model.get_spill_property("positions")).copy()
    traj_lon.append(p[:, 0].tolist()); traj_lat.append(p[:, 1].tolist()); times.append(str(model.model_time))
json.dump(dict(lon=traj_lon, lat=traj_lat, times=times), open(cfg["output_json"], "w"))
print("PYGNOME_SUCCESS", len(times), len(pos))
'''


class PyGnomeBackwardEngine:
    """NOAA PyGNOME backward run in a separate interpreter (conda env). Untested outside Kaggle -- see README."""
    name = "pygnome"

    def __init__(self, files: ForcingFiles, phys: PhysicsConfig, workdir: str):
        if not phys.pygnome_python or not os.path.exists(phys.pygnome_python):
            raise RuntimeError("PhysicsConfig.pygnome_python must point to the python of the PyGNOME env "
                               "(e.g. /kaggle/working/micromamba_root/envs/pygnome/bin/python)")
        self.files, self.p, self.workdir = files, phys, workdir
        os.makedirs(workdir, exist_ok=True)
        self.script = os.path.join(workdir, "pygnome_backward_worker.py")
        open(self.script, "w").write(_PYGNOME_WORKER)

    def run(self, lon0, lat0, t_start, hours, direction=-1, n=None, init_sigma_m=0.0) -> Trajectory:
        if direction != -1:
            raise ValueError("PyGNOME engine here is backward-only")
        lon0 = np.atleast_1d(lon0).astype(float); lat0 = np.atleast_1d(lat0).astype(float)
        wpath, cpath = self.files.paths()
        cfgp = os.path.join(self.workdir, "pygnome_cfg.json"); outp = os.path.join(self.workdir, "pygnome_out.json")
        json.dump(dict(acquisition_time=str(pd.Timestamp(t_start)), hours=float(hours), time_step_s=int(self.p.dt_min * 60),
                       positions=np.column_stack([lon0, lat0]).tolist(), wind_nc=wpath, current_nc=cpath, output_json=outp), open(cfgp, "w"))
        r = subprocess.run([self.p.pygnome_python, self.script, cfgp], capture_output=True, text=True)
        if r.returncode != 0 or "PYGNOME_SUCCESS" not in r.stdout:
            raise RuntimeError(f"PyGNOME worker failed:\n{r.stdout[-800:]}\n{r.stderr[-1500:]}")
        d = json.load(open(outp))
        t0 = pd.Timestamp(t_start)
        n_t = len(d["lon"]); dt = pd.Timedelta(seconds=self.p.dt_min * 60)
        times = [t0 - k * dt for k in range(n_t)]
        return Trajectory(times, np.array(d["lon"]), np.array(d["lat"]), -1, self.name)
