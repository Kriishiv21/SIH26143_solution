"""Slick detection: Otsu baseline vs. trained DeepLabV3+ (.pt bundle), benchmarked and auto-selected.

The DeepLab detector consumes the artifacts written by Phase 34 of the classifier notebook:
    model_weights.pt, deployment_config.json, model_defs.py  (+ dino_projection_weights.pt, model_traced.pt)
It does NOT use the exported ``inference_utils.py`` (that file references notebook globals such as
CONFIG / load_any_image / extract_full_geometry and is not standalone). Pre-processing (speckle filter
-> normalise -> overlap-tile -> merge) is re-implemented here to match training exactly.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from .config import DetectionConfig
from .observation import SarScene, SlickObservation, build_observation

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    detector: str
    prob: Optional[np.ndarray]
    mask: np.ndarray
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- metrics
def seg_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = float((p & g).sum()); fp = float((p & ~g).sum()); fn = float((~p & g).sum())
    n_p = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)[0] - 1
    # object-level false alarms: predicted components with no overlap with GT
    lab = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)[1]
    fa = sum(1 for k in range(1, lab.max() + 1) if not (g & (lab == k)).any())
    return dict(dice=2 * tp / (2 * tp + fp + fn + eps), iou=tp / (tp + fp + fn + eps),
                precision=tp / (tp + fp + eps), recall=tp / (tp + fn + eps), false_alarm_objects=int(fa))


# ----------------------------------------------------------------------------- Otsu baseline
class OtsuDetector:
    """Otsu-threshold dark-object detector on smoothed dB imagery.

    mode='global' : one Otsu threshold for the whole scene. This FAILS on typical scenes -- a small
                    slick in a large sea area gives a unimodal histogram and Otsu then cuts the sea
                    itself in half (kept only as a reference row in the benchmark).
    mode='tiled'  : (default, name 'otsu') Otsu inside sliding windows, accepted only where the
                    window is genuinely bimodal (class-mean gap >= gate_db and a plausible dark
                    fraction). This is the strongest classical baseline the DeepLab model must beat.

    Range-profile flattening (per-column median subtraction) removes the incidence-angle gradient;
    the assumption that columns run along range is configurable via flatten_axis.
    """
    available = True
    reason = ""

    def __init__(self, mode: str = "tiled", smooth_px: float = 1.5, flatten_axis: Optional[str] = "cols",
                 min_pixels: int = 30, window: int = 96, gate_db: float = 3.0):
        self.mode, self.smooth_px, self.flatten_axis = mode, smooth_px, flatten_axis
        self.min_pixels, self.window, self.gate_db = min_pixels, window, gate_db
        self.name = "otsu" if mode == "tiled" else "otsu_global"

    def _prep(self, scene):
        db = scene.db_image().astype(np.float64)
        valid = np.isfinite(db)
        if valid.sum() < 100:
            return None, valid
        db = np.where(valid, db, np.nanmedian(db))
        if self.flatten_axis == "cols":
            db = db - np.median(db, axis=0, keepdims=True)
        elif self.flatten_axis == "rows":
            db = db - np.median(db, axis=1, keepdims=True)
        return ndimage.gaussian_filter(db, self.smooth_px), valid

    def predict(self, scene: SarScene) -> DetectionResult:
        from skimage.filters import threshold_otsu
        sm, valid = self._prep(scene)
        if sm is None:
            return DetectionResult(self.name, None, np.zeros(valid.shape, np.uint8), dict(note="no valid pixels"))
        H, W = sm.shape
        dark = np.zeros((H, W), bool); n_acc = 0
        if self.mode == "global":
            thr = float(threshold_otsu(sm[valid])); dark = (sm < thr) & valid; meta = dict(threshold_db=thr)
        else:
            w = min(self.window, H, W); stride = max(w // 2, 1)
            ys = list(range(0, max(H - w, 0) + 1, stride)); xs = list(range(0, max(W - w, 0) + 1, stride))
            if ys[-1] + w < H: ys.append(H - w)
            if xs[-1] + w < W: xs.append(W - w)
            for y in ys:
                for x in xs:
                    blk = sm[y:y + w, x:x + w]
                    if np.ptp(blk) < 1e-6:
                        continue
                    thr = threshold_otsu(blk); lo = blk < thr
                    frac = lo.mean()
                    if not (0.01 <= frac <= 0.6):
                        continue
                    if blk[~lo].mean() - blk[lo].mean() < self.gate_db:
                        continue
                    dark[y:y + w, x:x + w] |= lo; n_acc += 1
            dark &= valid
            meta = dict(windows_accepted=n_acc, gate_db=self.gate_db)
        dark = ndimage.binary_opening(dark, np.ones((3, 3))); dark = ndimage.binary_closing(dark, np.ones((3, 3)))
        lab, n = ndimage.label(dark)
        if n:
            sizes = ndimage.sum(dark, lab, range(1, n + 1))
            dark = np.isin(lab, 1 + np.where(sizes >= self.min_pixels)[0])
        return DetectionResult(self.name, None, dark.astype(np.uint8), meta)


# ----------------------------------------------------------------------------- DeepLab bundle
_FILTERS = {}


def _speckle_filters():
    if _FILTERS:
        return _FILTERS

    def median_filter(img, window_size=5):
        return cv2.medianBlur(img.astype(np.float32), window_size)

    def frost(image, window_size=5, damping_factor=2.0):
        img = image.astype(np.float32)
        mean = cv2.blur(img, (window_size, window_size)); mean_sq = cv2.blur(img * img, (window_size, window_size))
        var = np.clip(mean_sq - mean ** 2, 0, None); cu2 = var / (mean ** 2 + 1e-8)
        alpha = damping_factor * cu2
        smoothed = cv2.GaussianBlur(img, (window_size, window_size), max(1.0, window_size / 2.0))
        blend = np.clip(alpha / (alpha.max() + 1e-8), 0, 1)
        return blend * img + (1 - blend) * smoothed

    def lee_sigma(image, window_size=5):
        img = image.astype(np.float32)
        mean = cv2.blur(img, (window_size, window_size)); mean_sq = cv2.blur(img * img, (window_size, window_size))
        var = np.clip(mean_sq - mean ** 2, 0, None)
        k = var / (var + img.var() + 1e-8)
        return mean + k * (img - mean)

    def nlm(image, h=10, template_window=7, search_window=21):
        img8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.fastNlMeansDenoising(img8, None, h, template_window, search_window).astype(np.float32)

    _FILTERS.update(median=median_filter, frost=frost, lee_sigma=lee_sigma, nlm=nlm)
    return _FILTERS


def _clean_nan_inf(img):
    fin = img[np.isfinite(img)]
    fill = float(fin.max()) if fin.size else 1.0
    return np.nan_to_num(img, nan=0.0, posinf=fill, neginf=0.0)


def _tile_coords(H, W, tile, overlap):
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, stride)); xs = list(range(0, max(W - tile, 0) + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    return [(y, x) for y in ys for x in xs]


class DeepLabDetector:
    """Loads the Phase-34 bundle and runs tiled, overlap-merged inference on arbitrary-size scenes."""
    name = "deeplab"

    def __init__(self, bundle_dir: str, device: Optional[str] = None, min_area_pixels: Optional[int] = None):
        self.bundle_dir, self.available, self.reason = bundle_dir, False, ""
        self.cfg, self.model, self.dino, self.traced = {}, None, None, None
        self._min_area = min_area_pixels
        try:
            self._load(device)
            self.available = True
        except Exception as e:  # noqa: BLE001 -- report reason, never crash the pipeline
            self.reason = f"{type(e).__name__}: {e}"
            log.warning("DeepLab bundle unavailable (%s) -- pipeline will use Otsu.", self.reason)

    # ---- loading
    def _resolve_dir(self) -> str:
        d = self.bundle_dir
        if os.path.isfile(d) and d.endswith(".zip"):
            out = os.path.join(os.path.dirname(d), "_bundle_unzipped")
            with zipfile.ZipFile(d) as z:
                z.extractall(out)
            return out
        return d

    def _load(self, device):
        import torch
        d = self._resolve_dir()
        cfg_path = os.path.join(d, "deployment_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"deployment_config.json not found in {d} (run Phase 34 of the classifier notebook)")
        self.cfg = json.load(open(cfg_path))
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        need = ("normalization_mean", "normalization_std", "speckle_filter", "threshold", "patch_size", "patch_overlap")
        miss = [k for k in need if k not in self.cfg]
        if miss:
            raise KeyError(f"deployment_config.json missing keys: {miss}")
        if self.cfg["speckle_filter"] not in _speckle_filters():
            raise KeyError(f"unknown speckle filter {self.cfg['speckle_filter']!r}")
        weights = os.path.join(d, self.cfg.get("weights_file", "model_weights.pt"))
        defs = os.path.join(d, "model_defs.py")
        traced = os.path.join(d, "model_traced.pt")
        if os.path.exists(defs) and os.path.exists(weights):
            spec = importlib.util.spec_from_file_location("sar_model_defs", defs)
            mod = importlib.util.module_from_spec(spec); sys.modules["sar_model_defs"] = mod; spec.loader.exec_module(mod)
            cls = getattr(mod, self.cfg["model_class"])
            model = cls(**self.cfg["model_init_kwargs"])
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
            self.model = model.to(self.device).eval()
            if self.cfg.get("uses_dino"):
                dino_bb = torch.hub.load("facebookresearch/dinov2", self.cfg["dino_model"])
                ex = mod.DINOFeatureExtractor(dino_bb, out_channels=self.cfg["dino_out_channels"])
                ex.project.load_state_dict(torch.load(os.path.join(d, self.cfg["dino_projection_weights_file"]), map_location="cpu"))
                self.dino = ex.to(self.device).eval()
        elif os.path.exists(traced) and not self.cfg.get("uses_dino"):
            self.traced = torch.jit.load(traced, map_location=self.device).eval()
        else:
            raise FileNotFoundError(f"Need model_weights.pt + model_defs.py (or model_traced.pt) in {d}")
        self.tile, self.overlap = int(self.cfg["patch_size"]), int(self.cfg["patch_overlap"])
        self.threshold = float(self.cfg["threshold"])
        if self._min_area is None:
            self._min_area = int(self.cfg.get("min_area_pixels", 400))

    # ---- inference
    def _predict_tile(self, tile_np):
        torch = self.torch
        x = torch.from_numpy(tile_np).float()[None, None].to(self.device)
        with torch.no_grad():
            if self.traced is not None:
                logits = self.traced(x)
            elif self.dino is not None:
                logits = self.model(x, extra_features=self.dino(x))
            else:
                logits = self.model(x)
            return torch.sigmoid(logits)[0, 0].cpu().numpy()

    def predict_array(self, img: np.ndarray) -> np.ndarray:
        """Dataset-scale float image (any size) -> probability map (same size)."""
        img = _clean_nan_inf(np.asarray(img, dtype=np.float32))
        if img.ndim == 3:
            img = img[..., 0]
        img = _speckle_filters()[self.cfg["speckle_filter"]](img)
        img = (_clean_nan_inf(img) - self.cfg["normalization_mean"]) / self.cfg["normalization_std"]
        H, W = img.shape
        T = self.tile
        pad_h, pad_w = max(0, T - H), max(0, T - W)
        if pad_h or pad_w:                                     # scenes smaller than one tile
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101 if min(H, W) > 1 else cv2.BORDER_REPLICATE)
        Hp, Wp = img.shape
        acc = np.zeros((Hp, Wp), np.float32); cnt = np.zeros((Hp, Wp), np.float32)
        for (y, x) in _tile_coords(Hp, Wp, T, self.overlap):
            p = self._predict_tile(img[y:y + T, x:x + T])
            acc[y:y + T, x:x + T] += p; cnt[y:y + T, x:x + T] += 1
        return (acc / np.maximum(cnt, 1))[:H, :W]

    def predict(self, scene: SarScene) -> DetectionResult:
        if not self.available:
            raise RuntimeError(f"DeepLab unavailable: {self.reason}")
        prob = self.predict_array(scene.dataset_scale())
        mask = (prob > self.threshold).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min(self._min_area, 50)]
        mask = np.isin(lab, keep).astype(np.uint8)
        return DetectionResult(self.name, prob, mask, dict(threshold=self.threshold, arch=self.cfg.get("architecture"),
                                                            test_metrics=self.cfg.get("test_metrics")))


def validate_bundle(det: DeepLabDetector) -> dict:
    """Smoke-test that the exported model behaves on the input variety a backend will send."""
    if not det.available:
        return dict(ok=False, reason=det.reason)
    rng = np.random.default_rng(0)
    checks, ok = [], True
    cases = {"smaller_than_tile": (96, 130), "non_square": (300, 517), "exact_tile": (det.tile, det.tile),
             "multi_tile": (700, 700)}
    for nm, (h, w) in cases.items():
        for dt in ("uint8-like", "float-with-nan"):
            img = rng.gamma(4, 20, (h, w)).astype(np.float32)
            if dt == "float-with-nan":
                img[5:9, 5:9] = np.nan
            try:
                p = det.predict_array(img)
                good = p.shape == (h, w) and np.isfinite(p).all() and 0.0 <= p.min() and p.max() <= 1.0
            except Exception as e:  # noqa: BLE001
                good, p = False, None; checks.append(dict(case=f"{nm}/{dt}", ok=False, error=repr(e)))
                ok = False; continue
            checks.append(dict(case=f"{nm}/{dt}", ok=bool(good), shape=(h, w)))
            ok &= bool(good)
    return dict(ok=bool(ok), architecture=det.cfg.get("architecture"), test_metrics=det.cfg.get("test_metrics"),
                device=str(det.device), checks=checks)


# ----------------------------------------------------------------------------- benchmark + selection
def gather_labeled_samples(dataset_root: str, max_n: int = 30) -> List[SarScene]:
    """Best-effort discovery of (image, mask) pairs in a Deep-SAR-style dataset (test/val split preferred)."""
    from .geo import grid_from_bbox
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [os.path.join(r, f) for r, _, fs in os.walk(dataset_root) for f in fs if os.path.splitext(f)[1].lower() in exts]
    is_mask = lambda p: any(k in p.lower() for k in ("mask", "label", "/gt", "\\gt", "annot", "ground"))
    masks = {os.path.splitext(os.path.basename(p))[0]: p for p in files if is_mask(p)}
    pairs = [(p, masks[os.path.splitext(os.path.basename(p))[0]]) for p in files
             if not is_mask(p) and os.path.splitext(os.path.basename(p))[0] in masks]
    pref = [pp for pp in pairs if any(k in pp[0].lower() for k in ("test", "val"))]
    pairs = (pref or pairs)[:max_n]
    out = []
    for i, (ip, mp) in enumerate(pairs):
        im = cv2.imread(ip, cv2.IMREAD_UNCHANGED); mk = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if im is None or mk is None:
            continue
        im = im if im.ndim == 2 else im[..., 0]; mk = mk if mk.ndim == 2 else mk[..., 0]
        if mk.shape != im.shape:
            mk = cv2.resize(mk, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        g = grid_from_bbox(0.0, 0.0, 1e-4, im.shape)          # dummy georef: metrics are pixel-level
        out.append(SarScene(f"labeled_{i}", im.astype(np.float32), "uint8", g, pd.Timestamp("2000-01-01"), "LABELED_DATASET",
                            gt_mask=(mk > 0).astype(np.uint8)))
    return out


def benchmark_detectors(detectors: Sequence, scenes: Sequence[SarScene]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run every available detector on every labelled scene. Returns (per-scene, summary)."""
    rows = []
    for sc in scenes:
        if sc.gt_mask is None:
            continue
        for d in detectors:
            if not getattr(d, "available", True):
                continue
            r = d.predict(sc)
            rows.append(dict(scene=sc.case_id, detector=d.name, **seg_metrics(r.mask, sc.gt_mask)))
    per = pd.DataFrame(rows)
    if per.empty:
        return per, pd.DataFrame()
    summ = per.groupby("detector")[["dice", "iou", "precision", "recall", "false_alarm_objects"]].mean().reset_index()
    return per, summ


def select_detector(summary: pd.DataFrame, cfg: DetectionConfig, available: Dict[str, bool]) -> Tuple[str, str]:
    """Returns (detector_name, human-readable reason). Explicit modes are honoured (or fall back)."""
    if cfg.mode in ("otsu", "deeplab"):
        if cfg.mode == "deeplab" and not available.get("deeplab", False):
            return "otsu", "mode='deeplab' requested but the bundle is unavailable -> falling back to Otsu"
        return cfg.mode, f"mode='{cfg.mode}' forced by config"
    if not available.get("deeplab", False):
        return "otsu", "DeepLab bundle not available -> Otsu (train Part A / point detection.bundle_dir at the export)"
    if summary is None or summary.empty or "deeplab" not in set(summary.detector):
        return "deeplab", "No labelled benchmark data -> trusting the trained model (its own test metrics are in the bundle config)"
    d = summary.set_index("detector")["dice"]
    if "otsu" not in d.index:
        return "deeplab", "Otsu not benchmarked"
    gap = d["deeplab"] - d["otsu"]
    if gap > cfg.selection_margin:
        return "deeplab", f"DeepLab Dice {d['deeplab']:.3f} beats Otsu {d['otsu']:.3f} by {gap:.3f} (> margin {cfg.selection_margin})"
    if -gap > cfg.selection_margin:
        return "otsu", f"Otsu Dice {d['otsu']:.3f} beats DeepLab {d['deeplab']:.3f} by {-gap:.3f} (> margin)"
    return cfg.prefer_when_tied, f"Within margin (DeepLab {d['deeplab']:.3f} vs Otsu {d['otsu']:.3f}) -> prefer '{cfg.prefer_when_tied}'"


def detect_slick(scene: SarScene, detectors: Dict[str, object], selected: str, cfg: DetectionConfig
                 ) -> Tuple[Optional[SlickObservation], dict, dict]:
    """Run the selected detector; if it finds nothing, try the other one. Also reports inter-detector agreement."""
    order = [selected] + [k for k in detectors if k != selected]
    results, obs, used = {}, None, None
    for name in order:
        d = detectors[name]
        if not getattr(d, "available", True):
            continue
        r = d.predict(scene); results[name] = r
        if obs is None:
            o = build_observation(scene, r.mask, r.prob, name, cfg.min_area_km2)
            if o is not None:
                obs, used = o, name
    agreement = None
    if len(results) >= 2:
        a, b = [results[k].mask.astype(bool) for k in list(results)[:2]]
        agreement = float((a & b).sum() / max((a | b).sum(), 1))
    info = dict(selected=selected, used=used, fell_back=(used is not None and used != selected), inter_detector_iou=agreement,
                per_detector={k: dict(n_pixels=int(v.mask.sum())) for k, v in results.items()},
                gt_metrics=({k: seg_metrics(v.mask, scene.gt_mask) for k, v in results.items()} if scene.gt_mask is not None else None))
    return obs, info, results
