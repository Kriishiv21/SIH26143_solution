"""Outputs: CSV/JSON per case, figures, and a cross-case summary."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mask_overlay(ax, scene, mask, color, label):
    ext = _extent(scene)
    rgba = np.zeros(mask.shape + (4,)); rgba[mask > 0] = color
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="nearest")
    ax.plot([], [], color=color[:3], lw=6, label=label)


def _extent(scene):
    lo0, la0 = scene.georef.pixel_to_lonlat(-0.5, -0.5); lo1, la1 = scene.georef.pixel_to_lonlat(scene.image.shape[0] - 0.5, scene.image.shape[1] - 0.5)
    return [lo0, lo1, la1, la0]


def plot_detection(res, path):
    sc = res.scene; fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(sc.dataset_scale(), cmap="gray", extent=_extent(sc)); axs[0].set_title(f"SAR (dB-stretched)\n{sc.case_id} @ {sc.acquisition_time}")
    names = list(res.detection_results)
    for ax, nm in zip(axs[1:], names[:2]):
        ax.imshow(sc.dataset_scale(), cmap="gray", extent=_extent(sc))
        _mask_overlay(ax, sc, res.detection_results[nm].mask, (1, 0.1, 0.1, 0.55), f"{nm} mask")
        if sc.gt_mask is not None:
            ax.contour(sc.gt_mask, levels=[0.5], colors="cyan", linewidths=1, extent=_extent(sc), origin="upper")
        m = (res.detection_info.get("gt_metrics") or {}).get(nm)
        ax.set_title(f"{nm}" + (f"  Dice={m['dice']:.2f}" if m else "") + ("  [USED]" if nm == res.detection_info["used"] else ""))
    for ax in axs:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_overview(res, path):
    fig, ax = plt.subplots(figsize=(9, 8)); sc = res.scene; tr = res.trajectory
    mp = tr.mean_path()
    ax.scatter(tr.lon[0], tr.lat[0], s=2, c="crimson", alpha=.3, label="observed slick particles")
    ax.scatter(tr.lon[-1], tr.lat[-1], s=3, c="orange", alpha=.35, label=f"source cloud (t-{res.source_summary['hours_back']:.0f}h)")
    ax.plot(mp.lon, mp.lat, "k-", lw=2, label="mean backtrack path")
    fu = res.fused.head(5)
    cols = plt.cm.viridis(np.linspace(0, .9, max(len(fu), 1)))
    for (_, r), c in zip(fu.iterrows(), cols):
        t = res.tracks[int(r.MMSI)]
        ax.plot(t.LON, t.LAT, "-", color=c, lw=1.2, alpha=.9, label=f"#{int(r['rank'])} {int(r.MMSI)}  ({r.confidence:.2f})")
        ax.scatter([r.est_release_lon], [r.est_release_lat], marker="*", s=140, color=c, edgecolor="k", zorder=5)
    if res.truth:
        ax.scatter([res.truth["source"][0]], [res.truth["source"][1]], marker="X", s=160, c="lime", edgecolor="k", zorder=6, label="synthetic true source")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title(f"{res.label}\nbacktrack + AIS candidates"); ax.legend(fontsize=7, loc="best"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_confidence(res, path):
    fu = res.fused.head(8)
    if fu.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4)); y = np.arange(len(fu))[::-1]
    w = {k: v for k, v in dict(anomaly=.20, attribution=.45, digital_twin=.35).items()}
    parts = [("anomaly_confidence", "anomaly"), ("attribution_score", "attribution"), ("digital_twin_score", "digital_twin")]
    left = np.zeros(len(fu)); cfgw = res.provenance.get("_weights", w)
    for col, k in parts:
        val = fu[col].clip(0, 1).to_numpy() * cfgw[k]; ax.barh(y, val, left=left, label=k); left += val
    ax.set_yticks(y); ax.set_yticklabels([f"{int(m)}" for m in fu.MMSI]); ax.set_xlabel("weighted evidence"); ax.legend(); ax.set_title(f"{res.label}: ranking evidence")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_case(res, out, stage1):
    res.provenance["_weights"] = None
    res.provenance.pop("_weights")
    res.ranking.to_csv(os.path.join(out, "vessel_funnel.csv"), index=False)
    res.fused.to_csv(os.path.join(out, "ranked_suspects.csv"), index=False)
    res.trajectory.mean_path().to_csv(os.path.join(out, "backtrack_mean_path.csv"), index=False)
    if res.weathering is not None:
        res.weathering.to_csv(os.path.join(out, "oil_weathering_screening.csv"), index=False)
    plot_detection(res, os.path.join(out, "1_detection.png"))
    plot_overview(res, os.path.join(out, "2_overview.png"))
    plot_confidence(res, os.path.join(out, "3_evidence.png"))
    doc = dict(case_id=res.case_id, label=res.label, provenance=res.provenance, observation=res.observation.summary(),
               detection=dict(res.detection_info), source=res.source_summary, conclusion=res.conclusion,
               engine_check=res.engine_check, self_validation=res.validation, stage1_ais_model=stage1)
    json.dump(doc, open(os.path.join(out, "case_report.json"), "w"), indent=2, default=str)


def save_summary(pipe, results):
    rows = []
    for r in results:

        o = r.observation
        c = r.conclusion

        # Handle cases where no oil slick was detected.
        if o is None:
            rows.append(dict(
                case_id=r.case_id,
                sar_source=r.provenance["sar_source"],
                ais_source=r.provenance["ais_source"],
                detector_used=r.detection_info.get("used"),
                area_km2=None,
                source_lat=None,
                source_lon=None,
                source_spread_km=None,
                n_candidates=0,
                lead_level="NO_OIL_DETECTED",
                top_mmsi=None,
                top_confidence=None,
                selfcheck_culprit_rank=None,
                selfcheck_culprit_is_top1=None,
            ))
            continue

        # Normal case: oil slick was detected.
        rows.append(dict(
            case_id=r.case_id,
            sar_source=r.provenance["sar_source"],
            ais_source=r.provenance["ais_source"],
            detector_used=r.detection_info["used"],
            area_km2=round(o.area_km2, 2),
            source_lat=round(r.source_summary["lat"], 4),
            source_lon=round(r.source_summary["lon"], 4),
            source_spread_km=round(r.source_summary["spread_km"], 2),
            n_candidates=c.get("n_candidates"),
            lead_level=c["level"],
            top_mmsi=c.get("top_mmsi"),
            top_confidence=c.get("top_confidence"),
            **{
                f"selfcheck_{k}": v
                for k, v in r.validation.items()
                if k in ("culprit_rank", "culprit_is_top1")
            }
        ))

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(pipe.cfg.output_dir, "all_cases_summary.csv"),
        index=False
    )

    meta = dict(
        detector_selected=pipe.selected_detector,
        detector_reason=pipe.detector_choice_reason,
        deeplab_available=bool(pipe.detectors["deeplab"].available),
        deeplab_unavailable_reason=pipe.detectors["deeplab"].reason,
        bundle_validation=pipe.bundle_validation
    )

    json.dump(
        meta,
        open(os.path.join(pipe.cfg.output_dir, "run_metadata.json"), "w"),
        indent=2,
        default=str
    )

    return df
