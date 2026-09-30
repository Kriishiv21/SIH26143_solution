# SIH 26143 -- SAR oil-spill -> backtracking -> AIS attribution (merged pipeline)

    oilspill_pipeline/   config, geo (conventions), forcing, detection, backtrack, engines (OpenOil/PyGNOME),
                         ais, ais_models, attribution, twin, fusion, report, pipeline, setup_env
    SIH26143_EndToEnd_Pipeline.ipynb   one-switch notebook (DEMO / REAL)
    run_pipeline.py                    CLI:  python run_pipeline.py --fast --out outputs
    tests/                             34 tests:  python -m pytest -q tests
    sample_outputs/                    a DEMO run (CSVs, JSON, PNGs)

## Two-step workflow
1. **Train once** (Kaggle GPU): run `oil_spill_pipeline_hardened.ipynb`. Phase 34 writes
   `/kaggle/working/robust_pipeline/deployment/{model_weights.pt, deployment_config.json, model_defs.py, ...}`.
   (Or set `TRAIN_CLASSIFIER_IF_MISSING=True` in the notebook to run it headlessly.)
2. **Run the pipeline**: `cfg.detection.bundle_dir` -> that folder (or `deployment_bundle.zip`). With `detection.mode="auto"`
   Otsu and DeepLabV3+ are benchmarked on labelled scenes and the better one is used (fallback: the other one).

## Conventions (geo.py, unit-tested)
u = east, v = north (m/s); bearings clockwise from north. **Wind = direction it blows FROM**, **current/drift = direction TOWARD**.
Files store components -> used as-is. Oil velocity = current + windage x wind10 (vector); windage 3 % (ensemble 1-4 %).

## DEMO -> REAL
Uncomment the REAL block in notebook cell 2 (or fill `cfg.real_cases` / `cfg.forcing.*` / `cfg.ais.*`).
SAR must be a georeferenced GeoTIFF with an acquisition time (tag, sidecar json, or explicit) -- never invented.
Forcing must cover the scene area and `backtrack_hours` before the pass, else `ForcingCoverageError`.

## Verification status
| Component | Status |
|---|---|
| Conventions, windage, Heun advection, backward = inverse of forward | unit-tested |
| Wind NetCDF/GRIB + current CSV loaders, coverage gate, bottom-current refusal | unit-tested (GRIB via cfgrib not exercised) |
| Georeferenced GeoTIFF -> REAL mode end-to-end from files | tested with generated files |
| OpenOil forward twin + weathering table | tested; transport agrees with NumPy within ~1 % |
| DeepLab loader: tiling/padding/merge/fallback logic | tested with a stub network; **the real .pt path is untested (no torch/GPU here)** |
| PyGNOME backward engine | **untested** (needs conda env; adapted from the original worker) |
| Detector benchmark | run on synthetic scenes only; DeepLab-vs-Otsu result **must be re-run with your bundle** |

## Known limits
* DEMO ground truth is generated with the same forcing model that ranks it ("inverse crime"): it validates plumbing/ranking, not accuracy.
* Domain shift: Part A trains on Deep-SAR 8-bit images; real Sentinel-1 sigma0 is converted to a matching dB-stretched 0-255 scale (`SarScene.dataset_scale`). Verify on real labelled scenes.
* Lead levels are investigative leads, not probabilities of guilt. When several vessels are statistically indistinguishable the output is `AMBIGUOUS_SHORTLIST`.
