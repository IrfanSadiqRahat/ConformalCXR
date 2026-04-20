"""
Full evaluation pipeline for ConformalCXR.

Loads the calibrated predictor, generates prediction sets on the
test split, and prints the full evaluation report.

Usage:
    python scripts/evaluate.py --config configs/chexpert.yaml
"""

import sys, argparse, yaml, pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import full_evaluation_report, print_report, CHEXPERT_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/chexpert.yaml")
    p.add_argument("--predictor", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    out  = Path(cfg.get("output_dir", "outputs/run_001"))
    pkl  = args.predictor or str(out / "calibrated_predictor.pkl")

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    predictor   = data["predictor"]
    probs_test  = data["probs_test"]
    labels_test = data["labels_test"]
    alpha       = cfg.get("alpha", 0.10)

    print(f"[Evaluate] {len(labels_test)} test examples | method={cfg.get('method','raps')}")

    pred_sets = predictor.predict(probs_test)
    report    = full_evaluation_report(
        pred_sets=pred_sets, probs=probs_test,
        primary_labels=labels_test, class_names=CHEXPERT_CLASSES)

    print_report(report, alpha=alpha)

    # Save results
    import json
    # Convert non-serializable values
    def serialize(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, dict):        return {str(k):serialize(v) for k,v in obj.items()}
        return obj

    results_path = out / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(serialize(report), f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__": main()
