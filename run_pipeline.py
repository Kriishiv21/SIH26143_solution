#!/usr/bin/env python
"""CLI:  python run_pipeline.py [--mode DEMO|REAL] [--fast] [--out DIR] [--bundle DIR]"""
import argparse, logging, sys
from oilspill_pipeline.config import Config, demo_fast_config
from oilspill_pipeline.pipeline import Pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="DEMO"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="./outputs"); ap.add_argument("--bundle", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = demo_fast_config(a.out) if a.fast else Config(output_dir=a.out)
    cfg.mode = a.mode
    if a.bundle: cfg.detection.bundle_dir = a.bundle
    res = Pipeline(cfg).run()
    for r in res:
        print(r.case_id, r.conclusion.get("level"), r.conclusion.get("top_mmsi"), r.validation)
