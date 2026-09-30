"""Time-synchronous AIS attribution against the backtracked slick.

Physical logic: if vessel V discharged oil at time t0 from position x0, then some of the observed
slick's particles were AT x0 at t0. Backtracking the observed slick gives, for every earlier time t,
the cloud where those particles were. So V is a candidate iff V's position at time t lies inside the
backtracked cloud at the SAME time t -- co-location in space AND time.

The original notebook's Hausdorff / closest-point / DTW metrics compared the whole vessel track with
the mean drift path *ignoring time* (only one variant weighted it), which rewards any vessel that
ever sailed through the region. Here the score is a Gaussian likelihood density evaluated per time
step, normalised so tight, well-timed matches beat loose ones, and it degrades honestly across AIS gaps.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .backtrack import Trajectory
from .config import AISConfig, AttributionConfig
from .geo import haversine_km, to_epoch_s, from_epoch_s


def vessel_position_at(track: pd.DataFrame, q_s: np.ndarray, ais_sigma_km: float):
    """Linear interpolation of (lat, lon) at query epochs. Returns lat, lon, sigma_km, gap_min, covered.

    sigma grows with the surrounding AIS gap (a vessel that was silent for hours could have deviated
    from the straight line); positions outside the track's time span are 'not covered' (NaN).
    """
    ts = track["BaseDateTime"].map(to_epoch_s).to_numpy(float)
    lat, lon = track["LAT"].to_numpy(float), track["LON"].to_numpy(float)
    sog = np.nan_to_num(track["SOG"].to_numpy(float), nan=8.0)
    covered = (q_s >= ts[0]) & (q_s <= ts[-1])
    qi = np.clip(q_s, ts[0], ts[-1])
    i1 = np.clip(np.searchsorted(ts, qi, side="right"), 1, len(ts) - 1); i0 = i1 - 1
    gap_s = ts[i1] - ts[i0]
    w = np.where(gap_s > 0, (qi - ts[i0]) / np.where(gap_s > 0, gap_s, 1), 0.0)
    la = lat[i0] * (1 - w) + lat[i1] * w
    lo = lon[i0] * (1 - w) + lon[i1] * w
    gap_min = gap_s / 60.0
    speed = 0.5 * (sog[i0] + sog[i1])
    sigma = ais_sigma_km + np.where(gap_min > 30.0, 0.25 * (gap_min / 60.0) * 1.852 * np.maximum(speed, 3.0), 0.0)
    la = np.where(covered, la, np.nan); lo = np.where(covered, lo, np.nan)
    return la, lo, sigma, gap_min, covered


def rank_vessels(tracks: Dict[int, pd.DataFrame], traj: Trajectory, acfg: AttributionConfig, aiscfg: AISConfig,
                 anomaly: Optional[pd.DataFrame] = None, time_stride: int = 1) -> pd.DataFrame:
    """Funnel + score every vessel against the backtracked trajectory. Returns one row per vessel."""
    ks = np.arange(1, len(traj.times), time_stride)             # skip k=0 (the observation time itself)
    t_k = [traj.times[k] for k in ks]
    q = np.array([to_epoch_s(t) for t in t_k])
    cen_lon = traj.lon[ks].mean(axis=1); cen_lat = traj.lat[ks].mean(axis=1)
    sp = np.array([np.sqrt(np.var((traj.lon[k] - traj.lon[k].mean()) * np.cos(np.deg2rad(traj.lat[k].mean())) * 111.195) +
                           np.var((traj.lat[k] - traj.lat[k].mean()) * 111.195)) for k in ks])
    sigma_cloud = np.sqrt(np.maximum(sp, 0) ** 2 + acfg.sigma_min_km ** 2)
    an = anomaly.set_index("MMSI") if anomaly is not None and len(anomaly) else None
    rows = []
    for mmsi, tr in tracks.items():
        la, lo, sig_pos, gap_min, cov = vessel_position_at(tr, q, acfg.ais_position_sigma_km)
        name = str(tr["VesselName"].iloc[0]); vtype = str(tr["VesselType"].iloc[0]) if "VesselType" in tr else "unknown"
        row = dict(MMSI=int(mmsi), VesselName=name, VesselType=vtype, n_fixes=len(tr))
        if not cov.any():
            rows.append({**row, "temporal": False, "spatial": False, "ais_quality": False, "passes_all": False,
                         "attribution_score": 0.0, "min_z": np.nan}); continue
        d = haversine_km(la, lo, cen_lat, cen_lon)
        s2 = sigma_cloud ** 2 + sig_pos ** 2
        z_all = np.where(cov, d / np.sqrt(s2), np.inf)
        # co-location likelihood in [0, 1]: 1 = vessel exactly at the centre of the backtracked cloud at that time.
        # (No 1/sigma^2 density factor: it rescales every vessel identically and would only shrink the absolute score.)
        score_k = np.where(cov, np.exp(-0.5 * z_all ** 2), 0.0)
        kbest = int(np.nanargmax(score_k)); z = z_all
        plausible = np.where((z <= 2.0) & cov)[0]
        win_lo = t_k[plausible.min()] if plausible.size else t_k[kbest]
        win_hi = t_k[plausible.max()] if plausible.size else t_k[kbest]
        near = cov & (np.abs(q - q[kbest]) <= 2 * 3600)
        max_gap = float(gap_min[near].max()) if near.any() else 0.0
        n_in_win = int(((tr["BaseDateTime"] >= min(t_k)) & (tr["BaseDateTime"] <= max(t_k))).sum())
        temporal = bool(cov.any())
        spatial = bool(np.nanmin(z) <= acfg.z_gate)
        quality = bool(n_in_win >= 3 or max_gap >= 60.0)         # dark vessels are KEPT, and flagged below
        rows.append({**row, "temporal": temporal, "spatial": spatial, "ais_quality": quality,
                     "passes_all": temporal and spatial and quality,
                     "attribution_score": float(score_k[kbest]), "min_z": float(np.nanmin(z)),
                     "min_dist_km": float(np.nanmin(np.where(cov, d, np.inf))),
                     "est_release_time": t_k[kbest], "est_release_lat": float(la[kbest]), "est_release_lon": float(lo[kbest]),
                     "release_window_start": win_lo, "release_window_end": win_hi,
                     "max_ais_gap_min_near_release": max_gap, "dark_gap_flag": bool(max_gap >= 60.0),
                     "anomaly_confidence": float(an.loc[mmsi, "anomaly_confidence"]) if an is not None and mmsi in an.index else 0.0})
    df = pd.DataFrame(rows)
    if "attribution_score" in df:
        df = df.sort_values(["passes_all", "attribution_score"], ascending=[False, False]).reset_index(drop=True)
    return df
