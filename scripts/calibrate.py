"""
Run conformal calibration on the trained probe.

Loads the best probe checkpoint, extracts softmax scores on the
calibration split, calibrates RAPS / RAPS-CC threshold, and saves
the calibrated predictor to disk.

Usage:
    python scripts/calibrate.py --config configs/chexpert.yaml
"""

import sys, argparse, yaml, pickle
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.chexagent_wrapper import CheXClassifier
from src.data.chexpert_dataset import build_chexpert_loaders, NUM_CLASSES
from src.conformal.calibration import build_conformal_predictor, split_calibration, softmax


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/chexpert.yaml")
    p.add_argument("--ckpt",      default=None)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


@torch.no_grad()
def extract_logits(model, loader, device):
    model.eval()
    all_logits, all_primary = [], []
    for batch in tqdm(loader, desc="Extracting logits"):
        images = batch["image"].to(device)
        logits = model(images).cpu().numpy()
        all_logits.append(logits)
        all_primary.append(batch["primary_label"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_primary)


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out    = Path(cfg.get("output_dir", "outputs/run_001")); out.mkdir(parents=True, exist_ok=True)
    ckpt   = args.ckpt or str(out / "best_probe.pth")

    model = CheXClassifier(num_classes=NUM_CLASSES, device=str(device)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(f"[Calibrate] Loaded probe from {ckpt}")

    train_loader, _ = build_chexpert_loaders(
        cfg["chexpert_root"], cfg.get("batch_size", 64),
        cfg.get("image_size", 224), cfg.get("num_workers", 4))

    logits, labels = extract_logits(model, train_loader, device)
    probs_all      = softmax(logits)

    probs_calib, labels_calib, probs_test, labels_test = split_calibration(
        probs_all, labels, calib_frac=cfg.get("calib_frac", 0.20))

    print(f"[Calibrate] calib={len(labels_calib)} | test={len(labels_test)}")

    predictor = build_conformal_predictor(
        method     = cfg.get("method", "raps"),
        alpha      = cfg.get("alpha", 0.10),
        lambda_reg = cfg.get("raps_lambda", 0.01),
        k_reg      = cfg.get("raps_k_reg", 5),
        num_classes= NUM_CLASSES,
    )
    predictor.calibrate(probs_calib, labels_calib)

    # Save calibrated predictor and test data for evaluate.py
    save_path = out / "calibrated_predictor.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "predictor":    predictor,
            "probs_test":   probs_test,
            "labels_test":  labels_test,
            "config":       cfg,
        }, f)
    print(f"[Calibrate] Saved to {save_path}")


if __name__ == "__main__": main()
