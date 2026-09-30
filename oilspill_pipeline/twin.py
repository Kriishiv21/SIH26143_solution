"""Digital twin: 'if candidate V had released here at that time, would the oil be where SAR saw it?'"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import AttributionConfig
from .geo import haversine_km
from .observation import SlickObservation


def footprint_mask(lon, lat, georef, shape, trim_pct: float = 90.0) -> np.ndarray:
    """Convex hull of the central `trim_pct` % of particles, rasterised on the SAR pixel grid."""
    r, c = georef.lonlat_to_pixel(np.asarray(lon), np.asarray(lat))
    pts = np.column_stack([c, r])
    med = np.median(pts, axis=0); d = np.linalg.norm(pts - med, axis=1)
    pts = pts[d <= np.percentile(d, trim_pct)]
    m = np.zeros(shape, np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 1)
    return m


def compare_to_observation(lon, lat, obs: SlickObservation, cfg: AttributionConfig, drift_km: float = 0.0) -> dict:
    """Two separable questions: (1) POSITION -- is the observed slick where the simulated cloud is, given the
    ensemble's own spread (z-score)?  (2) SHAPE -- after aligning centroids, does the simulated footprint have
    the observed size/extent (IoU)?  Position is weighted more: shape depends on unresolved release physics."""
    lon = np.asarray(lon); lat = np.asarray(lat)
    c_lon, c_lat = float(lon.mean()), float(lat.mean())
    c_km = float(haversine_km(c_lat, c_lon, obs.center_lat, obs.center_lon))
    ex = (lon - c_lon) * np.cos(np.deg2rad(c_lat)) * 111.195; ey = (lat - c_lat) * 111.195
    sig_ens = float(np.sqrt(ex.var() + ey.var()) / np.sqrt(2)) if len(lon) > 2 else 0.0        # per-axis sigma
    sig_obs = float(max(obs.minor_km, obs.major_km) / 4.0)                                      # slick half-length ~ 2 sigma
    # transport-model error grows with distance travelled (engine-independent floor, so a tight ensemble
    # from one engine cannot over-penalise small offsets)
    sig_model = cfg.twin_model_error_frac * max(drift_km, 0.0)
    z = c_km / max(np.sqrt(sig_ens ** 2 + sig_obs ** 2 + sig_model ** 2), cfg.sigma_min_km)
    position = float(np.exp(-0.5 * z ** 2))
    # shape: shift the simulated cloud so its centroid sits on the observed centroid, then IoU on the SAR grid
    shift_lon = obs.center_lon - c_lon; shift_lat = obs.center_lat - c_lat
    pred = footprint_mask(lon + shift_lon, lat + shift_lat, obs.scene.georef, obs.mask.shape, trim_pct=80.0)
    inter = float((pred & obs.mask).sum()); union = float((pred | obs.mask).sum())
    iou = inter / union if union > 0 else 0.0
    dx, dy = obs.pixel_km; pred_area = float(pred.sum() * dx * dy)
    w = cfg.twin_weights
    score = w["centroid"] * position + w["iou"] * iou
    return dict(iou_aligned=iou, centroid_km=c_km, position_z=float(z), position_score=position, ensemble_sigma_km=sig_ens, model_sigma_km=sig_model,
                pred_area_km2=pred_area, obs_area_km2=obs.area_km2, area_ratio=pred_area / max(obs.area_km2, 1e-9),
                digital_twin_score=float(score))


def run_digital_twin(candidates: pd.DataFrame, engine, obs: SlickObservation, t_obs, cfg: AttributionConfig,
                     n_particles: int = 300, init_sigma_m: float = 150.0) -> pd.DataFrame:
    rows = []
    for _, r in candidates.iterrows():
        hours = (pd.Timestamp(t_obs) - pd.Timestamp(r["est_release_time"])).total_seconds() / 3600.0
        if hours <= 0:
            rows.append(dict(MMSI=r["MMSI"], digital_twin_score=0.0, note="release not before observation")); continue
        tr = engine.run(np.full(n_particles, r["est_release_lon"]), np.full(n_particles, r["est_release_lat"]),
                        r["est_release_time"], hours, direction=+1, init_sigma_m=init_sigma_m)
        drift_km = float(haversine_km(r["est_release_lat"], r["est_release_lon"], tr.lat[-1].mean(), tr.lon[-1].mean()))
        cmp = compare_to_observation(tr.lon[-1], tr.lat[-1], obs, cfg, drift_km)
        rows.append(dict(MMSI=r["MMSI"], hours_drift=hours, **cmp))
    return pd.DataFrame(rows)
