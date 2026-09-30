"""Single source of truth for every tunable in the pipeline.

Switching from the hard-coded DEMO scenario to REAL data is done here (or in the notebook's
config cell) by changing ``mode`` and filling in the ``real_*`` fields -- no other file needs
to be touched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ForcingConfig:
    # 'hardcoded' -> constant values below (explicitly synthetic, for plumbing tests / demos)
    # 'files'     -> real wind + current files (paths below)
    source: str = "hardcoded"

    # ---- hard-coded values (CONVENTIONS ARE EXPLICIT IN THE FIELD NAMES) ----
    # Wind: METEOROLOGICAL convention -> direction the wind blows FROM, degrees clockwise from north.
    hardcoded_wind_speed_ms: float = 5.0
    hardcoded_wind_dir_from_deg: float = 315.0        # north-westerly (typical E. Mediterranean)
    # Current: OCEANOGRAPHIC convention -> direction the water flows TOWARD.
    hardcoded_current_speed_ms: float = 0.15
    hardcoded_current_dir_to_deg: float = 70.0        # east-north-east

    # ---- real files ----
    wind_path: Optional[str] = None                   # .grib/.grib2/.nc with u10 & v10
    current_path: Optional[str] = None                # folder of Day_*.csv, a single .csv, or .nc
    current_glob: str = "Day_*.csv"
    # Hard gate copied from the original notebook: bottom currents must never drive floating oil.
    current_verified_surface: bool = False

    # ---- oil drift physics ----
    windage_frac: float = 0.03                        # oil moves with current + windage_frac * wind10
    windage_range: Tuple[float, float] = (0.01, 0.04) # per-particle spread (ensemble)
    allow_time_clamp: bool = False                    # never extrapolate forcing in time by default


@dataclass
class DetectionConfig:
    bundle_dir: str = "./bundle"   # output of Part A Phase 34
    mode: str = "auto"                                # 'auto' | 'deeplab' | 'otsu'
    selection_margin: float = 0.02                    # DeepLab must beat Otsu Dice by this to win
    min_area_km2: float = 0.05
    otsu_smooth_px: float = 1.5
    prefer_when_tied: str = "deeplab"                 # learned model rejects look-alikes better
    input_scaling: str = "auto"                       # 'auto' | 'linear' | 'db' | 'uint8'
    # optional labelled data (image/mask pairs) used to benchmark detectors in REAL mode
    benchmark_dataset_root: Optional[str] = None
    benchmark_max_samples: int = 30
    run_bundle_validation: bool = True


@dataclass
class AISConfig:
    source: str = "synthetic"                         # 'synthetic' | 'csv'
    csv_path: Optional[str] = None
    column_map: Dict[str, str] = field(default_factory=dict)   # override auto-detected columns
    n_synthetic_vessels: int = 60
    inject_demo_culprit: bool = True                  # synthetic self-check ONLY
    resample_minutes: int = 6
    gap_segment_minutes: int = 60
    implausible_speed_kn: float = 50.0
    history_steps: int = 12
    max_vessel_speed_kn: float = 25.0
    stage1_models: Tuple[str, ...] = ("MLP", "RNN", "MLP+RNN")
    stage1_max_sequences: int = 6000
    stage1_rnn_epochs: int = 25
    n_folds: int = 4


@dataclass
class PhysicsConfig:
    backtrack_hours: float = 72.0
    min_release_age_hours: float = 0.0
    n_particles: int = 600
    dt_min: float = 15.0
    horizontal_diffusivity_m2s: float = 10.0
    # forcing-error ensemble: persistent per-member perturbations
    wind_speed_rel_sigma: float = 0.15
    wind_dir_sigma_deg: float = 15.0
    current_speed_rel_sigma: float = 0.20
    current_dir_sigma_deg: float = 15.0
    backtrack_engine: str = "numpy"                   # 'numpy' | 'pygnome' | 'both'
    forward_engine: str = "numpy"                     # 'numpy' | 'openoil'
    pygnome_python: Optional[str] = None              # e.g. /kaggle/working/micromamba_root/envs/pygnome/bin/python
    oil_types: Tuple[str, ...] = ("GENERIC MEDIUM CRUDE", "GENERIC HEAVY FUEL OIL", "GENERIC DIESEL")
    age_grid_hours: Tuple[float, ...] = (6, 12, 24, 36, 48, 72)
    run_oil_age_screening: bool = False               # needs OpenOil; adds runtime


@dataclass
class AttributionConfig:
    top_n_for_twin: int = 8
    z_gate: float = 3.0                               # vessel must be within 3 sigma of the drifted cloud
    sigma_min_km: float = 1.0
    ais_position_sigma_km: float = 0.2
    fusion_weights: Dict[str, float] = field(default_factory=lambda: dict(anomaly=0.20, attribution=0.45, digital_twin=0.35))
    twin_weights: Dict[str, float] = field(default_factory=lambda: dict(iou=0.3, centroid=0.7))
    strong_score: float = 0.70
    strong_margin: float = 0.15
    moderate_score: float = 0.50
    twin_model_error_frac: float = 0.10               # transport error = 10 % of distance drifted (added to twin position sigma)
    ambiguity_margin: float = 0.05                    # candidates within this of the top are 'indistinguishable'


@dataclass
class Config:
    # ---- THE SWITCH ----
    mode: str = "DEMO"                                # 'DEMO' (hard-coded scenario) | 'REAL'
    seed: int = 20190405
    output_dir: str = "./outputs"

    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ais: AISConfig = field(default_factory=AISConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)

    # REAL mode: list of dicts, one per SAR product, e.g.
    #   dict(case_id="S1A_2019-04-05", sar_path="/kaggle/input/sar/S1A_xxx.tif",
    #        acquisition_time=None,            # None -> read from tags / sidecar JSON, never invented
    #        polarisation_band=1, units="auto",
    #        ais_csv="/kaggle/input/ais/port_said.csv")
    real_cases: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def demo_fast_config(output_dir: str = "./outputs") -> Config:
    """Small/fast settings used by the tests and the notebook smoke run."""
    cfg = Config(output_dir=output_dir)
    cfg.physics.n_particles = 300
    cfg.physics.backtrack_hours = 48.0
    cfg.physics.dt_min = 30.0
    cfg.ais.n_synthetic_vessels = 30
    cfg.ais.stage1_models = ("MLP",)
    cfg.ais.stage1_max_sequences = 2500
    cfg.ais.n_folds = 3
    cfg.attribution.top_n_for_twin = 5
    return cfg
