"""SAR scene + detected-slick observation (georeferenced geometry, particle sampling)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .geo import GeoRef, lonlat_to_local_km


@dataclass
class SarScene:
    case_id: str
    image: np.ndarray                    # 2-D raw pixel values (see `units`)
    units: str                           # 'linear' | 'db' | 'uint8'
    georef: GeoRef
    acquisition_time: pd.Timestamp
    source: str                          # 'DEMO_SYNTHETIC' | 'REAL_GEOTIFF'
    time_source: str = ""
    label: str = ""
    gt_mask: Optional[np.ndarray] = None # ONLY for benchmarking; never used by detectors
    meta: dict = field(default_factory=dict)

    def db_image(self) -> np.ndarray:
        img = self.image.astype(np.float32)
        if self.units == "db":
            return img
        if self.units == "linear":
            return 10.0 * np.log10(np.clip(img, 1e-6, None))
        # 8-bit rendered: treat as already log-like display scale
        return img

    def dataset_scale(self) -> np.ndarray:
        """0..255 float image in the same *style* as the Deep-SAR training PNGs (dB-stretched)."""
        if self.units == "uint8":
            return self.image.astype(np.float32)
        db = self.db_image()
        fin = db[np.isfinite(db)]
        if fin.size == 0:
            return np.zeros_like(db)
        lo, hi = np.percentile(fin, [1.0, 99.5])
        return np.clip((np.nan_to_num(db, nan=lo) - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.float32)


@dataclass
class SlickObservation:
    scene: SarScene
    mask: np.ndarray                     # uint8, union of retained components
    prob: Optional[np.ndarray]
    detector: str
    components: pd.DataFrame
    area_km2: float
    center_lon: float
    center_lat: float
    major_km: float
    minor_km: float
    orientation_deg: float               # bearing (deg cw from N) of the major axis, 0..180
    polygon_lonlat: Optional[np.ndarray]
    pixel_km: tuple

    @property
    def n_pixels(self) -> int:
        return int(self.mask.sum())

    def summary(self) -> dict:
        return dict(detector=self.detector, area_km2=round(self.area_km2, 3), center_lon=round(self.center_lon, 5),
                    center_lat=round(self.center_lat, 5), major_km=round(self.major_km, 2), minor_km=round(self.minor_km, 2),
                    orientation_deg=round(self.orientation_deg, 1), n_components=int(len(self.components)))

    def sample_particles(self, n: int, rng: np.random.Generator):
        """Probability-weighted particle cloud inside the slick, as (lon, lat)."""
        rows, cols = np.nonzero(self.mask)
        w = self.prob[rows, cols].astype(float) if self.prob is not None else np.ones(rows.size)
        w = np.clip(w, 1e-6, None); w /= w.sum()
        idx = rng.choice(rows.size, size=n, p=w, replace=True)
        r = rows[idx] + rng.uniform(-0.5, 0.5, n); c = cols[idx] + rng.uniform(-0.5, 0.5, n)
        return self.scene.georef.pixel_to_lonlat(r, c)


def _label_components(mask: np.ndarray):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, lab, stats, cent


def build_observation(scene: SarScene, mask: np.ndarray, prob: Optional[np.ndarray], detector: str,
                      min_area_km2: float = 0.05, keep_ratio: float = 0.4) -> Optional[SlickObservation]:
    """Turn a binary mask into a georeferenced SlickObservation.

    Components are scored by 'oil evidence x area' (probability for DeepLab, contrast in dB for
    Otsu) so a large but faint look-alike does not outrank a compact dark slick. Components
    scoring >= keep_ratio * best are retained (fragments of one slick survive).
    """
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return None
    H, W = mask.shape
    dx, dy = scene.georef.pixel_size_km(H / 2, W / 2)
    pix_km2 = dx * dy
    db = scene.db_image()
    sea = np.nanmedian(db[mask == 0]) if (mask == 0).any() else np.nanmedian(db)
    n, lab, stats, _ = _label_components(mask)
    rows = []
    for k in range(1, n):
        m = lab == k
        area = float(stats[k, cv2.CC_STAT_AREA] * pix_km2)
        if area < min_area_km2:
            continue
        evidence = float(prob[m].mean()) if prob is not None else float(max(sea - np.nanmean(db[m]), 0.0) / 6.0)
        rows.append(dict(label=k, area_km2=area, evidence=evidence, oil_score=area * evidence))
    if not rows:
        return None
    comp = pd.DataFrame(rows)
    keep = comp[comp.oil_score >= keep_ratio * comp.oil_score.max()].copy()
    comp["kept"] = comp.label.isin(keep.label)
    final = np.isin(lab, keep.label.values).astype(np.uint8)
    rr, cc = np.nonzero(final)
    lon, lat = scene.georef.pixel_to_lonlat(rr, cc)
    lon0, lat0 = float(lon.mean()), float(lat.mean())
    x, y = lonlat_to_local_km(lon, lat, lon0, lat0)
    if len(x) > 3:
        ev, evec = np.linalg.eigh(np.cov(np.vstack([x, y])))
        major = evec[:, np.argmax(ev)]
        orient = float(np.rad2deg(np.arctan2(major[0], major[1])) % 180.0)     # bearing from north
        maj_km, min_km = float(4 * np.sqrt(ev.max())), float(4 * np.sqrt(ev.min()))   # ~ full axis length (±2 sigma)
    else:
        orient, maj_km, min_km = float("nan"), 0.0, 0.0
    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = None
    if cnts:
        big = max(cnts, key=cv2.contourArea).reshape(-1, 2)
        pl, pa = scene.georef.pixel_to_lonlat(big[:, 1], big[:, 0])
        poly = np.column_stack([pl, pa])
    return SlickObservation(scene, final, prob, detector, comp, float(final.sum() * pix_km2), lon0, lat0,
                            maj_km, min_km, orient, poly, (dx, dy))
