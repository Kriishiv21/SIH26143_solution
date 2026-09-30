"""Evidence fusion and non-accusatory conclusion levels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AttributionConfig


def fuse(ranking: pd.DataFrame, twin: pd.DataFrame, cfg: AttributionConfig) -> pd.DataFrame:
    """Three separate evidence axes (behavioural anomaly, space-time attribution, physics twin)."""
    surv = ranking[ranking["passes_all"]].copy()
    if surv.empty:
        return surv
    df = surv.merge(twin, on="MMSI", how="left") if len(twin) else surv.assign(digital_twin_score=0.0)
    df["digital_twin_score"] = df.get("digital_twin_score", 0.0)
    df["digital_twin_score"] = df["digital_twin_score"].fillna(0.0)
    w = cfg.fusion_weights
    df["confidence"] = (w["anomaly"] * df["anomaly_confidence"] + w["attribution"] * df["attribution_score"].clip(0, 1)
                        + w["digital_twin"] * df["digital_twin_score"])
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    flags = []
    for _, r in df.iterrows():
        f = []
        if r.get("dark_gap_flag"): f.append(f"AIS_GAP_{int(r['max_ais_gap_min_near_release'])}min_NEAR_RELEASE")
        if r.get("anomaly_confidence", 0) > 0.7: f.append("BEHAVIOURAL_ANOMALY")
        if r.get("digital_twin_score", 0) >= 0.5: f.append("PHYSICS_TWIN_MATCH")
        flags.append(";".join(f))
    df["evidence_flags"] = flags
    return df


def conclusion(fused: pd.DataFrame, cfg: AttributionConfig) -> dict:
    """Investigative lead level. A relative ranking score -- NOT a probability of guilt."""
    if fused.empty:
        return dict(level="NO_CANDIDATE", text="No vessel was co-located with the backtracked slick in space and time.")
    top = fused.iloc[0]; margin = float(top["confidence"] - fused.iloc[1]["confidence"]) if len(fused) > 1 else float(top["confidence"])
    n_close = int((fused["confidence"] >= top["confidence"] - cfg.ambiguity_margin).sum())
    if top["confidence"] >= cfg.strong_score and margin >= cfg.strong_margin:
        lvl = "STRONG_LEAD"
    elif top["confidence"] >= cfg.moderate_score and n_close > 1:
        lvl = "AMBIGUOUS_SHORTLIST"          # several vessels are statistically indistinguishable -> report a shortlist
    elif top["confidence"] >= cfg.moderate_score:
        lvl = "MODERATE_LEAD"
    else:
        lvl = "WEAK_INCONCLUSIVE"
    return dict(level=lvl, top_mmsi=int(top["MMSI"]), top_name=str(top["VesselName"]), top_confidence=round(float(top["confidence"]), 3),
                margin_to_second=round(margin, 3), n_candidates=int(len(fused)), n_within_ambiguity_margin=n_close,
                shortlist=[int(m) for m in fused.head(max(n_close, 1))["MMSI"]],
                text=("Ranking score for further investigation (evidence weight), not a probability of guilt; "
                      "confirm with additional evidence (e.g. sampling, satellite re-tasking, port-state records)."))
