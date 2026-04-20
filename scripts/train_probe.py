"""
Train a linear classification probe on frozen CheXagent features.

Usage:
    python scripts/train_probe.py --config configs/chexpert.yaml
"""

import sys, argparse, yaml, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.chexagent_wrapper import CheXClassifier
from src.data.chexpert_dataset import build_chexpert_loaders, NUM_CLASSES, CHEXPERT_CLASSES
from src.evaluation.metrics import compute_auc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/chexpert.yaml")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--resume",   default=None)
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train(); total = 0
    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        if i % 100 == 0:
            print(f"  [{epoch}][{i}/{len(loader)}] loss={loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs  = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc    = compute_auc(probs, labels)
    return total_loss / len(loader), auc


def main():
    args = parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out    = Path(cfg.get("output_dir", "outputs/run_001")); out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_chexpert_loaders(
        cfg["chexpert_root"], cfg.get("batch_size", 64),
        cfg.get("image_size", 224), cfg.get("num_workers", 4))

    model = CheXClassifier(
        num_classes=NUM_CLASSES,
        hidden_dim=cfg.get("probe_hidden", 512),
        device=str(device),
    ).to(device)

    # Only train the probe — backbone is frozen
    pos_w   = torch.ones(NUM_CLASSES).to(device) * cfg.get("pos_weight", 5.0)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt     = AdamW(model.get_trainable_params(),
                    lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    sched   = CosineAnnealingLR(opt, T_max=cfg.get("epochs", 30))

    best_auc = 0.0
    for epoch in range(1, cfg.get("epochs", 30) + 1):
        t0       = time.time()
        tr_loss  = train_epoch(model, train_loader, opt, crit, device, epoch)
        va_loss, va_auc = validate(model, val_loader, crit, device)
        sched.step()
        mean_auc = va_auc.get("mean_auc", 0.0)
        print(f"Epoch {epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} | "
              f"AUC={mean_auc:.4f} | {time.time()-t0:.1f}s")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "auc": best_auc}, out / "best_probe.pth")
            print(f"  ✅ New best AUC={best_auc:.4f} saved")

    print(f"
Done. Best AUC: {best_auc:.4f}")


if __name__ == "__main__": main()
