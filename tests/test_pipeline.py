"""Run with:  python -m pytest -q tests   (torch/DeepLab tests are skipped when torch or a bundle is absent)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd, pytest
from oilspill_pipeline import geo, forcing as F, backtrack as B, detection as D, scenario as S, attribution as A, fusion
from oilspill_pipeline.config import *
from oilspill_pipeline.pipeline import Pipeline


# ---------------- conventions (points 6 & 7) ----------------
def test_wind_from_north_blows_south():
    u, v = geo.met_wind_to_uv(5, 0); assert abs(u) < 1e-9 and v == pytest.approx(-5)

def test_wind_from_west_blows_east():
    u, v = geo.met_wind_to_uv(5, 270); assert u == pytest.approx(5) and abs(v) < 1e-9

def test_current_toward_east():
    u, v = geo.current_to_uv(1, 90); assert u == pytest.approx(1) and abs(v) < 1e-9

def test_roundtrip_to_from():
    sp, to = geo.uv_to_speed_dir_to(*geo.current_to_uv(2, 123)); assert (sp, to) == (pytest.approx(2), pytest.approx(123))
    sp, fr = geo.uv_to_speed_dir_from(*geo.met_wind_to_uv(7, 45)); assert fr == pytest.approx(45)

def test_windage_is_vector_added_to_current():
    f = F.build_forcing(ForcingConfig()); t = pd.Timestamp("2019-04-05"); la = lo = np.array([31.0])
    u, v = f.oil_velocity(la, lo, t); uc, vc = f.current_uv(la, lo, t); uw, vw = f.wind_uv(la, lo, t)
    assert np.allclose(u, uc + 0.03 * uw) and np.allclose(v, vc + 0.03 * vw)

def test_zero_wind_gives_pure_current_regardless_of_windage():
    f = F.build_forcing(ForcingConfig(hardcoded_wind_speed_ms=0.0)); u, v = f.oil_velocity(np.array([31.]), np.array([32.]), pd.Timestamp("2019-04-05"), windage=0.04)
    uc, vc = f.current_uv(np.array([31.]), np.array([32.]), pd.Timestamp("2019-04-05")); assert np.allclose(u, uc)   # old code scaled the current by (1+windage)


# ---------------- physics ----------------
def _still_current_east(): return F.build_forcing(ForcingConfig(hardcoded_wind_speed_ms=0.0, hardcoded_current_speed_ms=0.2, hardcoded_current_dir_to_deg=90.0))

def test_forward_distance_and_direction():
    e = B.LagrangianEngine(_still_current_east(), PhysicsConfig(horizontal_diffusivity_m2s=0.0), perturb=False)
    tr = e.run(32.0, 31.5, "2019-04-05", 24, +1, n=5)
    assert geo.haversine_km(31.5, 32.0, tr.lat[-1].mean(), tr.lon[-1].mean()) == pytest.approx(17.28, abs=0.05) and tr.lon[-1].mean() > 32.0

def test_backward_is_exact_inverse_of_forward():
    e = B.LagrangianEngine(_still_current_east(), PhysicsConfig(horizontal_diffusivity_m2s=0.0), perturb=False)
    fw = e.run(32.0, 31.5, "2019-04-05", 24, +1, n=5); bw = e.run(fw.lon[-1], fw.lat[-1], fw.times[-1], 24, -1)
    assert geo.haversine_km(31.5, 32.0, bw.lat[-1].mean(), bw.lon[-1].mean()) < 0.01

def test_backward_source_is_upstream():
    e = B.LagrangianEngine(_still_current_east(), PhysicsConfig(horizontal_diffusivity_m2s=0.0), perturb=False)
    bw = e.run(32.0, 31.5, "2019-04-05", 24, -1, n=5); assert bw.lon[-1].mean() < 32.0 and bw.times[-1] == pd.Timestamp("2019-04-04")


# ---------------- forcing I/O ----------------
def test_forcing_never_extrapolates_in_time_and_refuses_bottom_current(tmp_path):
    rows = [dict(date="2019-04-05", time_utc=f"{h:02d}:00:00", latitude=la, longitude=lo, water_u_mps=0.1, water_v_mps=0.0) for h in (0, 6) for la in (31., 32.) for lo in (32., 33.)]
    (tmp_path / "Day_01.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    g = F.load_current_csv(str(tmp_path))
    with pytest.raises(F.ForcingCoverageError): g.uv(np.array([31.5]), np.array([32.5]), pd.Timestamp("2019-05-01"))
    pd.DataFrame(dict(date=["2019-04-05"], time_utc=["00:00:00"], latitude=[31], longitude=[32], water_u_bottom=[0.1], water_v_bottom=[0.1])).to_csv(tmp_path / "bot.csv", index=False)
    with pytest.raises(Exception): F.load_current_csv(str(tmp_path / "bot.csv"))

def test_coverage_report_flags_missing_region(tmp_path):
    rows = [dict(date="2019-04-05", time_utc="00:00:00", latitude=la, longitude=lo, water_u_mps=0.1, water_v_mps=0.0) for la in (31., 32.) for lo in (32., 33.)]
    (tmp_path / "Day_01.csv").write_text(pd.DataFrame(rows).to_csv(index=False))
    g = F.load_current_csv(str(tmp_path)); assert not g.check_coverage((20, 31, 33, 32), "2019-04-05", "2019-04-05")["ok"]


# ---------------- detection (point 4) ----------------
def test_tiled_otsu_beats_global_otsu():
    scenes = S.make_benchmark_scenes(4, seed=5)
    per, summ = D.benchmark_detectors([D.OtsuDetector(), D.OtsuDetector("global")], scenes)
    d = summ.set_index("detector").dice; assert d["otsu"] > 0.85 and d["otsu_global"] < 0.2

def test_selection_logic():
    dc = DetectionConfig(); summ = pd.DataFrame(dict(detector=["otsu", "deeplab"], dice=[.80, .90]))
    assert D.select_detector(summ, dc, dict(deeplab=True, otsu=True))[0] == "deeplab"
    assert D.select_detector(pd.DataFrame(dict(detector=["otsu", "deeplab"], dice=[.90, .80])), dc, dict(deeplab=True, otsu=True))[0] == "otsu"
    assert D.select_detector(summ, dc, dict(deeplab=False, otsu=True))[0] == "otsu"
    assert D.select_detector(summ, DetectionConfig(mode="deeplab"), dict(deeplab=False, otsu=True))[0] == "otsu"

def test_missing_bundle_degrades_gracefully():
    d = D.DeepLabDetector("/definitely/not/here"); assert not d.available and d.reason


# ---------------- georeferencing (point 1) ----------------
def test_real_geotiff_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio"); from rasterio.transform import from_origin
    sc = S.make_demo_scene(S.DEMO_CASES[0], 70.0); p = tmp_path / "s1.tif"
    with rasterio.open(p, "w", driver="GTiff", height=512, width=512, count=1, dtype="float32", crs="EPSG:4326",
                       transform=sc.georef.transform) as dst:
        dst.write(sc.image, 1); dst.update_tags(ACQUISITION_START_TIME="2019-04-05T03:52:10Z")
    r = S.load_real_scene(dict(case_id="x", sar_path=str(p)))
    assert r.acquisition_time == pd.Timestamp("2019-04-05 03:52:10") and r.time_source.startswith("raster_tag") and r.units == "linear"
    assert np.allclose(r.georef.pixel_to_lonlat(10, 20), sc.georef.pixel_to_lonlat(10, 20))

def test_real_scene_without_time_is_refused(tmp_path):
    rasterio = pytest.importorskip("rasterio"); sc = S.make_demo_scene(S.DEMO_CASES[0], 70.0); p = tmp_path / "s1.tif"
    with rasterio.open(p, "w", driver="GTiff", height=512, width=512, count=1, dtype="float32", crs="EPSG:4326", transform=sc.georef.transform) as dst:
        dst.write(sc.image, 1)
    with pytest.raises(ValueError, match="acquisition time"): S.load_real_scene(dict(sar_path=str(p)))


# ---------------- end-to-end (points 1,3,5,6,7) ----------------
@pytest.fixture(scope="module")
def demo_results(tmp_path_factory):
    cfg = demo_fast_config(str(tmp_path_factory.mktemp("out"))); cfg.mode = "DEMO"
    cfg.detection.bundle_dir = "/nonexistent"; cfg.detection.run_bundle_validation = False
    return Pipeline(cfg).run()

def test_demo_runs_all_four_cases(demo_results): assert len(demo_results) == 4

def test_demo_synthetic_culprit_is_ranked_first(demo_results):
    assert all(r.validation["culprit_is_top1"] for r in demo_results)

def test_demo_true_source_inside_backtracked_cloud(demo_results):
    assert all(r.validation["true_source_inside_cloud_central_90pct"] for r in demo_results)

def test_dark_gap_vessel_is_flagged_not_dropped(demo_results):
    r = demo_results[0]; row = r.fused[r.fused.MMSI == r.truth["dark_gap"]]
    assert len(row) == 1 and bool(row.iloc[0].dark_gap_flag)

def test_wrong_place_decoy_is_excluded(demo_results):
    r = demo_results[0]; assert r.truth["decoys"][1] not in set(r.fused.MMSI)

def test_provenance_labels_synthetic(demo_results):
    w = " ".join(demo_results[0].provenance["warnings"]); assert "SYNTHETIC" in w and "DEMO" in w

def test_conclusion_is_non_accusatory(demo_results):
    assert "not a probability of guilt" in demo_results[0].conclusion["text"]

def test_outputs_written(demo_results):
    for r in demo_results:
        for f in ("ranked_suspects.csv", "vessel_funnel.csv", "case_report.json", "1_detection.png", "2_overview.png", "3_evidence.png"):
            assert os.path.exists(os.path.join(r.out_dir, f)), f


# ---------------- OpenOil (point 5) ----------------
def test_openoil_matches_numpy_transport(tmp_path):
    pytest.importorskip("opendrift"); from oilspill_pipeline import engines
    f = _still_current_east(); ph = PhysicsConfig(dt_min=30, horizontal_diffusivity_m2s=0.0); t0 = pd.Timestamp("2019-04-05")
    ff = engines.ForcingFiles(f, (32.0, 31.0, 33.0, 32.0), t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=30), str(tmp_path))
    tr = engines.OpenOilEngine(ff, ph).run(32.3, 31.5, t0, 12, +1, n=30)
    assert geo.haversine_km(31.5, 32.3, tr.lat[-1].mean(), tr.lon[-1].mean()) == pytest.approx(8.64, rel=0.03)


# ---------------- DeepLab wrapper logic without torch (tiling / padding / merge) ----------------
def _stub_detector(tile=64, overlap=16):
    d = D.DeepLabDetector.__new__(D.DeepLabDetector)
    d.available, d.reason, d.tile, d.overlap, d.threshold, d._min_area = True, "", tile, overlap, 0.5, 10
    d.cfg = dict(speckle_filter="median", normalization_mean=0.0, normalization_std=1.0, architecture="stub", test_metrics=None)
    d.traced = d.dino = None; d.device = "cpu"
    d._predict_tile = lambda t: (t > 0.5).astype(np.float32)          # deterministic stand-in for the network
    return d

@pytest.mark.parametrize("shape", [(30, 40), (64, 64), (200, 131), (300, 517)])
def test_tiled_inference_preserves_shape_and_signal(shape):
    d = _stub_detector(); img = np.zeros(shape, np.float32); h, w = shape
    img[h // 3:h // 3 + 10, w // 3:w // 3 + 12] = 1.0
    p = d.predict_array(img * 255.0)
    assert p.shape == shape and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1.0
    assert p[h // 3 + 5, w // 3 + 6] > 0.5 and p[0, 0] < 0.5

def test_tiled_inference_tolerates_nan():
    d = _stub_detector(); img = np.random.default_rng(0).gamma(4, 20, (100, 100)).astype(np.float32); img[3:6, 3:6] = np.nan
    assert np.isfinite(d.predict_array(img)).all()

def test_detect_slick_falls_back_when_selected_finds_nothing():
    class Empty:
        name, available = "deeplab", True
        def predict(self, sc): return D.DetectionResult("deeplab", np.zeros(sc.image.shape, np.float32), np.zeros(sc.image.shape, np.uint8))
    sc = S.make_demo_scene(S.DEMO_CASES[0], 70.0)
    obs, info, _ = D.detect_slick(sc, {"deeplab": Empty(), "otsu": D.OtsuDetector()}, "deeplab", DetectionConfig())
    assert obs is not None and info["used"] == "otsu" and info["fell_back"]


# ---------------- REAL-mode switch, end to end, from files on disk (point 1) ----------------
def _write_real_inputs(d):
    import rasterio, xarray as xr
    case = S.DEMO_CASES[0]
    f = F.build_forcing(ForcingConfig())
    u, v = f.oil_velocity(np.array([31.3]), np.array([32.3]), pd.Timestamp("2019-04-05")); br = float(np.rad2deg(np.arctan2(u[0], v[0])) % 360)
    sc = S.make_demo_scene(case, br, seed=1)
    tif = os.path.join(d, "scene.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=512, width=512, count=1, dtype="float32", crs="EPSG:4326", transform=sc.georef.transform) as dst:
        dst.write(sc.image, 1); dst.update_tags(ACQUISITION_START_TIME=str(sc.acquisition_time))
    t = pd.date_range("2019-04-01", "2019-04-07", freq="3h"); lat = np.arange(30.0, 33.01, 0.25); lon = np.arange(31.0, 34.01, 0.25)
    shp = (len(t), len(lat), len(lon)); uw, vw = geo.met_wind_to_uv(5.0, 315.0)
    xr.Dataset({"u10": (("time", "latitude", "longitude"), np.full(shp, uw, np.float32)), "v10": (("time", "latitude", "longitude"), np.full(shp, vw, np.float32))},
               coords=dict(time=t, latitude=lat[::-1], longitude=lon)).isel(latitude=slice(None)).to_netcdf(os.path.join(d, "wind.nc"))
    cu, cv = geo.current_to_uv(0.15, 70.0); os.makedirs(os.path.join(d, "cur"))
    rows = [dict(date=str(tt.date()), time_utc=tt.strftime("%H:%M:%S"), latitude=a, longitude=o, water_u_mps=cu, water_v_mps=cv)
            for tt in pd.date_range("2019-04-01", "2019-04-07", freq="6h") for a in np.arange(30.0, 33.01, 0.5) for o in np.arange(31.0, 34.01, 0.5)]
    pd.DataFrame(rows).to_csv(os.path.join(d, "cur", "Day_01.csv"), index=False)
    raw = A_ais.synthetic_ais((32.28, 31.30), sc.acquisition_time, 48, 25, seed=3)
    raw, truth = A_ais.inject_test_vessels(raw, (32.10, 31.29), sc.acquisition_time - pd.Timedelta(hours=16))
    raw = raw.rename(columns={"BaseDateTime": "# Timestamp", "LAT": "Latitude", "LON": "Longitude", "MMSI": "MMSI", "SOG": "SOG", "COG": "COG"})
    raw.to_csv(os.path.join(d, "ais.csv"), index=False)
    return tif, truth

from oilspill_pipeline import ais as A_ais

def test_real_mode_end_to_end_from_files(tmp_path):
    pytest.importorskip("rasterio"); d = str(tmp_path); tif, truth = _write_real_inputs(d)
    cfg = demo_fast_config(os.path.join(d, "out")); cfg.mode = "REAL"; cfg.detection.bundle_dir = "/nonexistent"; cfg.detection.run_bundle_validation = False
    cfg.real_cases = [dict(case_id="real1", sar_path=tif, ais_csv=os.path.join(d, "ais.csv"))]
    cfg.forcing.source = "files"; cfg.forcing.wind_path = os.path.join(d, "wind.nc"); cfg.forcing.current_path = os.path.join(d, "cur"); cfg.forcing.current_verified_surface = True
    cfg.ais.source = "csv"
    res = Pipeline(cfg).run()[0]
    assert res.provenance["sar_source"] == "REAL_GEOTIFF" and res.provenance["time_source"].startswith("raster_tag")
    assert "REAL CSV" in res.provenance["ais_source"] and not any("SYNTHETIC" in w for w in res.provenance["warnings"])
    rank = {int(r.MMSI): int(r["rank"]) for _, r in res.fused.iterrows()}
    # the culprit here is hand-placed (not generated from the drift model), so demand 'shortlisted', not 'first'
    assert truth["culprit"] in rank and rank[truth["culprit"]] <= 3

def test_real_mode_fails_loudly_when_forcing_does_not_cover(tmp_path):
    pytest.importorskip("rasterio"); d = str(tmp_path); tif, _ = _write_real_inputs(d)
    cfg = demo_fast_config(os.path.join(d, "out")); cfg.mode = "REAL"; cfg.detection.bundle_dir = "/nonexistent"
    cfg.real_cases = [dict(case_id="real1", sar_path=tif, acquisition_time="2020-01-01T04:00:00", ais_csv=os.path.join(d, "ais.csv"))]
    cfg.forcing.source = "files"; cfg.forcing.wind_path = os.path.join(d, "wind.nc"); cfg.forcing.current_path = os.path.join(d, "cur")
    with pytest.raises(F.ForcingCoverageError): Pipeline(cfg).run()

def test_openoil_forward_engine_end_to_end(tmp_path):
    pytest.importorskip("opendrift")
    cfg = demo_fast_config(str(tmp_path)); cfg.detection.bundle_dir = "/nonexistent"; cfg.detection.run_bundle_validation = False
    cfg.physics.forward_engine = "openoil"; cfg.physics.run_oil_age_screening = True; cfg.physics.age_grid_hours = (6, 12, 24)
    from oilspill_pipeline import scenario as SS
    cfg.mode = "DEMO"; p = Pipeline(cfg); p.setup_detection(); r = p.run_case(SS.DEMO_CASES[0])
    assert r.validation["culprit_rank"] <= 3 and r.weathering is not None and len(r.weathering) >= 3
    assert not any("OpenOil unavailable" in w for w in r.provenance["warnings"])
