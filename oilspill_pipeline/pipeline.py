"""End-to-end orchestration: SAR -> detection -> backtrack -> AIS attribution -> twin -> fusion -> report."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import ais as aismod, ais_models as aim, attribution, backtrack, detection, engines, forcing as fmod, fusion, report, scenario, twin
from .config import Config
from .geo import haversine_km, naive_utc

log = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    label: str
    scene: object
    observation: object
    detection_info: dict
    detection_results: dict
    trajectory: object
    source_summary: dict
    ranking: pd.DataFrame
    fused: pd.DataFrame
    conclusion: dict
    provenance: dict
    tracks: dict = field(default_factory=dict)
    truth: Optional[dict] = None
    engine_check: dict = field(default_factory=dict)
    weathering: Optional[pd.DataFrame] = None
    validation: dict = field(default_factory=dict)
    out_dir: str = ""


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.detectors: Dict[str, object] = {}
        self.selected_detector = None
        self.detector_choice_reason = ""
        self.benchmark_summary = pd.DataFrame(); self.benchmark_per_scene = pd.DataFrame()
        self.bundle_validation: dict = {}

    # ------------------------------------------------------------------ detection setup
    def setup_detection(self) -> None:
        dc = self.cfg.detection
        self.detectors = {"otsu": detection.OtsuDetector(dc.otsu_smooth_px)}
        dl = detection.DeepLabDetector(dc.bundle_dir)
        self.detectors["deeplab"] = dl
        if dl.available and dc.run_bundle_validation:
            self.bundle_validation = detection.validate_bundle(dl)
        # benchmark scenes: real labelled data if provided, else independent synthetic scenes (DEMO)
        scenes = []
        if dc.benchmark_dataset_root and os.path.isdir(dc.benchmark_dataset_root):
            scenes = detection.gather_labeled_samples(dc.benchmark_dataset_root, dc.benchmark_max_samples)
            src = f"labelled dataset ({len(scenes)} pairs)"
        elif self.cfg.mode.upper() == "DEMO":
            scenes = scenario.make_benchmark_scenes(6, seed=self.cfg.seed); src = "synthetic benchmark scenes (DEMO)"
        else:
            src = "none available"
        bench = [d for d in (self.detectors["otsu"], self.detectors["deeplab"]) if getattr(d, "available", True)]
        self.benchmark_per_scene, self.benchmark_summary = detection.benchmark_detectors(bench, scenes) if scenes else (pd.DataFrame(), pd.DataFrame())
        self.selected_detector, self.detector_choice_reason = detection.select_detector(
            self.benchmark_summary, dc, {k: getattr(v, "available", True) for k, v in self.detectors.items()})
        self.detector_choice_reason += f"  [benchmark data: {src}]"
        log.info("Detector selected: %s -- %s", self.selected_detector, self.detector_choice_reason)
        self.benchmark_summary.to_csv(os.path.join(self.cfg.output_dir, "detector_benchmark_summary.csv"), index=False)
        self.benchmark_per_scene.to_csv(os.path.join(self.cfg.output_dir, "detector_benchmark_per_scene.csv"), index=False)

    # ------------------------------------------------------------------ one case
    def run_case(self, spec: dict) -> CaseResult:
        cfg, ph = self.cfg, self.cfg.physics
        demo = cfg.mode.upper() == "DEMO"
        out = os.path.join(cfg.output_dir, spec["case_id"]); os.makedirs(out, exist_ok=True)
        prov = dict(mode=cfg.mode, forcing_source=cfg.forcing.source, ais_source=None, sar_source=None,
                    warnings=[])

        # ---- forcing (needed first in DEMO: slick orientation follows the drift)
        f0 = fmod.build_forcing(cfg.forcing) if cfg.forcing.source == "hardcoded" else None
        drift_bearing = None
        if f0 is not None:
            u, v = f0.oil_velocity(np.array([spec.get("centroid", (31.3, 32.3))[0]]), np.array([spec.get("centroid", (31.3, 32.3))[1]]), pd.Timestamp("2019-01-01"))
            drift_bearing = float(np.rad2deg(np.arctan2(u[0], v[0])) % 360)
        scene = scenario.load_scene(spec, cfg, drift_bearing)
        prov["sar_source"] = scene.source; prov["acquisition_time"] = str(scene.acquisition_time); prov["time_source"] = scene.time_source
        if demo:
            prov["warnings"].append("DEMO: synthetic SAR scene, hard-coded forcing, synthetic AIS -- validates plumbing, not real-world accuracy.")
        if "ASSUMED" in scene.time_source:
            prov["warnings"].append("Acquisition time is ASSUMED (date supplied, pass time hard-coded).")

        # ---- detection
        obs, dinfo, dres = detection.detect_slick(scene, self.detectors, self.selected_detector, cfg.detection)
        if obs is None:
            print("***NO OIL DETECTED***")
        else:
            print("***OIL DETECTED***")

        if obs is None:
            log.info(
                "%s: no slick detected by any detector. "
                "Skipping downstream attribution for this case.",
                scene.case_id,
            )

            print("\n" + "=" * 70)
            print("CASE RESULT")
            print("=" * 70)
            print(f"SAR: {scene.case_id}")
            print("Detection: NO SLICK DETECTED")
            print()
            print(
                "No oil-spill region was identified by the available detectors."
            )
            print(
                "Downstream source backtracking, AIS attribution and "
                "digital-twin analysis were not performed for this case."
            )
            print("Case status: NO_OIL_DETECTED")
            print("=" * 70)

            return CaseResult(
                case_id=scene.case_id,
                label=scene.label or spec.get("label", scene.case_id),
                scene=scene,
                observation=None,
                detection_info=dinfo,
                detection_results=dres,
                trajectory=None,
                source_summary={},
                ranking=pd.DataFrame(),
                fused=pd.DataFrame(),
                conclusion={
                    "status": "NO_OIL_DETECTED",
                    "message": "No oil-spill slick detected."
                },
                provenance=prov,
                tracks={},
                truth=None,
                engine_check={},
                weathering=None,
                validation={},
                out_dir=out,
            )
        if dinfo["fell_back"]:
            prov["warnings"].append(f"Selected detector '{dinfo['selected']}' found nothing; used '{dinfo['used']}'.")
        t_obs = scene.acquisition_time
        t_lo = t_obs - pd.Timedelta(hours=ph.backtrack_hours)

        # ---- forcing for this case (real files: subset around the scene)
        b = scene.georef.bounds_lonlat(); pad = 1.0
        bbox = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        forc = f0 if f0 is not None else fmod.build_forcing(cfg.forcing, bbox)
        cov = forc.assert_coverage(bbox, t_lo, t_obs) if cfg.forcing.source == "files" else dict(ok=True)
        prov["forcing"] = forc.description

        # ---- backward reconstruction (NumPy) + optional PyGNOME cross-check
        rng = np.random.default_rng(cfg.seed)
        plon, plat = obs.sample_particles(ph.n_particles, rng)
        eng = backtrack.LagrangianEngine(forc, ph, cfg.forcing.windage_range, seed=cfg.seed)
        traj = eng.run(plon, plat, t_obs, ph.backtrack_hours, direction=-1)
        src_lo, src_la = traj.lon[-1], traj.lat[-1]
        src = dict(lon=float(src_lo.mean()), lat=float(src_la.mean()), bbox_95=traj.bbox(2.5, 0.0),
                   spread_km=float(np.sqrt(np.var((src_lo - src_lo.mean()) * np.cos(np.deg2rad(src_la.mean())) * 111.195) + np.var((src_la - src_la.mean()) * 111.195))),
                   hours_back=ph.backtrack_hours)
        engine_check = {}
        wfiles = None
        need_files = ph.forward_engine == "openoil" or ph.backtrack_engine in ("pygnome", "both") or ph.run_oil_age_screening
        if need_files:
            tb = traj.bbox(0.5, 0.4)
            wfiles = engines.ForcingFiles(forc, (min(tb[0], b[0]) - 0.1, min(tb[1], b[1]) - 0.1, max(tb[2], b[2]) + 0.1, max(tb[3], b[3]) + 0.1),
                                          t_lo - pd.Timedelta(hours=2), t_obs + pd.Timedelta(hours=2), os.path.join(out, "forcing_nc"), spec["case_id"])
        if ph.backtrack_engine in ("pygnome", "both"):
            try:
                pg = engines.PyGnomeBackwardEngine(wfiles, ph, os.path.join(out, "pygnome"))
                pt = pg.run(plon, plat, t_obs, ph.backtrack_hours, -1)
                d = float(haversine_km(pt.lat[-1].mean(), pt.lon[-1].mean(), src["lat"], src["lon"]))
                engine_check["pygnome_vs_numpy_source_km"] = d
                if ph.backtrack_engine == "pygnome":
                    traj = pt
                    src.update(lon=float(pt.lon[-1].mean()), lat=float(pt.lat[-1].mean()))
            except Exception as e:  # noqa: BLE001
                prov["warnings"].append(f"PyGNOME backward run unavailable: {str(e)[:200]} -- used NumPy engine.")

        # ---- AIS
        release_true, truth = None, None
        acfg = cfg.ais
        if acfg.source == "csv" or (not demo and (spec.get("ais_csv") or acfg.csv_path)):
            path = spec.get("ais_csv") or acfg.csv_path
            raw = aismod.load_ais_csv(path, acfg.column_map, bbox=bbox, t_range=(t_lo - pd.Timedelta(hours=24), t_obs))
            prov["ais_source"] = f"REAL CSV: {path}"
        else:
            raw = aismod.synthetic_ais((obs.center_lon, obs.center_lat), t_obs, ph.backtrack_hours, acfg.n_synthetic_vessels, seed=cfg.seed)
            prov["ais_source"] = "SYNTHETIC background traffic"
            prov["warnings"].append("AIS is SYNTHETIC. Set cfg.ais.source='csv' + csv_path for the real feed.")
            if acfg.inject_demo_culprit:
                # self-consistent ground truth: release 20 h before the pass from the deterministic backtracked source
                det = backtrack.LagrangianEngine(forc, ph, cfg.forcing.windage_range, seed=cfg.seed, perturb=False)
                rel_h = min(20.0, ph.backtrack_hours * 0.6)
                d = det.run(obs.center_lon, obs.center_lat, t_obs, rel_h, -1, n=1)
                truth_src = (float(d.lon[-1, 0]), float(d.lat[-1, 0]))
                raw, truth = aismod.inject_test_vessels(raw, truth_src, t_obs - pd.Timedelta(hours=rel_h), seed=cfg.seed)
                prov["warnings"].append("Synthetic ground truth injected (culprit + decoys + dark-gap vessel) for self-validation. "
                                        "It is generated with the same forcing/model used to rank it (an 'inverse crime'): it tests the plumbing and "
                                        "ranking logic, not physical accuracy.")

        proc, _ = aismod.preprocess_ais(raw, acfg)
        X, Y, meta = aim.build_sequences(proc, acfg)
        pe, s1 = aim.fit_stage1_oof(X, Y, meta, acfg)
        anom = aim.aggregate_vessel_scores(aim.compute_anomaly_scores(proc, pe, acfg))
        tracks = aismod.build_tracks(raw, acfg)

        # ---- attribution (time-synchronous)
        ranking = attribution.rank_vessels(tracks, traj, cfg.attribution, acfg, anom)
        surv = ranking[ranking.passes_all].head(cfg.attribution.top_n_for_twin)

        # ---- digital twin
        twin_eng = eng
        if ph.forward_engine == "openoil":
            try:
                twin_eng = engines.OpenOilEngine(wfiles, ph)
            except Exception as e:  # noqa: BLE001
                prov["warnings"].append(f"OpenOil unavailable: {str(e)[:150]} -- twin used NumPy engine.")
        tw = twin.run_digital_twin(surv, twin_eng, obs, t_obs, cfg.attribution, n_particles=min(300, ph.n_particles)) if len(surv) else pd.DataFrame()
        fused = fusion.fuse(ranking, tw, cfg.attribution)
        concl = fusion.conclusion(fused, cfg.attribution)

        # ---- oil weathering screening (optional)
        weath = None
        if ph.run_oil_age_screening and engines.openoil_available() and len(fused):
            try:
                top = fused.iloc[0]; parts = []
                for oil in ph.oil_types:
                    oe = engines.OpenOilEngine(wfiles, ph, oil_type=oil)
                    parts.append(oe.weathering_table(top.est_release_lon, top.est_release_lat, top.est_release_time,
                                                     tuple(a for a in ph.age_grid_hours if a <= ph.backtrack_hours)))
                weath = pd.concat(parts, ignore_index=True)
            except Exception as e:  # noqa: BLE001
                prov["warnings"].append(f"Oil weathering screening failed: {str(e)[:150]}")

        # ---- self-validation against synthetic truth
        validation = {}
        if truth is not None and len(fused):
            ranks = {int(r.MMSI): int(r["rank"]) for _, r in fused.iterrows()}
            validation = dict(culprit_rank=ranks.get(truth["culprit"]), decoy_ranks={k: ranks.get(k) for k in truth["decoys"]},
                              dark_gap_rank=ranks.get(truth["dark_gap"]), culprit_is_top1=ranks.get(truth["culprit"]) == 1)
            if truth["culprit"] in set(fused.MMSI):
                r = fused[fused.MMSI == truth["culprit"]].iloc[0]
                validation["release_time_error_min"] = float(abs((r.est_release_time - truth["release_time"]).total_seconds()) / 60)
                validation["release_position_error_km"] = float(haversine_km(r.est_release_lat, r.est_release_lon, truth["source"][1], truth["source"][0]))
            cl_lon, cl_lat = traj.cloud_at(truth["release_time"])
            validation["source_cloud_error_km_at_true_release_time"] = float(haversine_km(cl_lat.mean(), cl_lon.mean(), truth["source"][1], truth["source"][0]))
            spread = np.sqrt(np.var((cl_lon - cl_lon.mean()) * 95.0) + np.var((cl_lat - cl_lat.mean()) * 111.0))
            dd = np.hypot((cl_lon - truth["source"][0]) * 95.0, (cl_lat - truth["source"][1]) * 111.0)
            validation["true_source_inside_cloud_central_90pct"] = bool(np.hypot((truth["source"][0]-cl_lon.mean())*95.0,(truth["source"][1]-cl_lat.mean())*111.0) <= np.percentile(np.hypot((cl_lon-cl_lon.mean())*95.0,(cl_lat-cl_lat.mean())*111.0), 90))

        res = CaseResult(scene.case_id, scene.label or spec.get("label", scene.case_id), scene, obs, dinfo, dres, traj, src, ranking, fused,
                         concl, prov, tracks, truth, engine_check, weath, validation, out)
        report.save_case(res, out, s1)
        return res

    # ------------------------------------------------------------------ all cases
    def run(self) -> List[CaseResult]:
        self.setup_detection()
        self.cfg.save(os.path.join(self.cfg.output_dir, "config_used.json"))
        results = []
        for spec in scenario.list_case_specs(self.cfg):
            log.info("=== %s ===", spec["case_id"])
            results.append(self.run_case(spec))
        report.save_summary(self, results)
        return results
