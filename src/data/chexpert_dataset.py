"""
CheXpert Dataset Loader.

CheXpert (Irvin et al., Stanford 2019) — 224,316 frontal/lateral CXRs
with weak labels for 14 pathologies from radiology reports.

Label encoding:
  -1 (uncertain) → treated as 0 (negative) by default — common practice
   0 (negative)  → 0
   1 (positive)  → 1

For multi-label classification we use binary cross-entropy per class.
For conformal prediction (single-label setting) we use the primary
positive class (argmax of label vector), or skip if all-negative.

Download: https://stanfordmlgroup.github.io/competitions/chexpert/
(free registration; ~439 GB full, ~11 GB small version)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


CHEXPERT_CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
NUM_CLASSES = len(CHEXPERT_CLASSES)


def get_transforms(split: str, image_size: int = 224) -> T.Compose:
    """Standard CheXpert preprocessing transforms."""
    if split == "train":
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


class CheXpertDataset(Dataset):
    """
    CheXpert multi-label chest X-ray dataset.

    Args:
        csv_path:   path to train.csv or valid.csv
        root:       root directory of CheXpert (parent of train/ and valid/)
        split:      "train", "valid", or "test"
        image_size: target image size
        frontal_only: use only frontal (PA/AP) views
        uncertain_policy: how to handle uncertain labels (-1):
                           "zero"  → treat as negative
                           "one"   → treat as positive
                           "skip"  → exclude from loss (using mask)
    """

    def __init__(self, csv_path: str, root: str, split: str = "train",
                 image_size: int = 224, frontal_only: bool = True,
                 uncertain_policy: str = "zero"):
        self.root     = Path(root)
        self.split    = split
        self.transforms = get_transforms(split, image_size)
        self.uncertain_policy = uncertain_policy

        df = pd.read_csv(csv_path)
        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        self.paths  = df["Path"].tolist()
        self.labels = self._process_labels(df)
        print(f"[CheXpert] {split}: {len(self.paths)} images | "
              f"frontal_only={frontal_only}")

    def _process_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and clean label matrix (n, 14)."""
        label_cols = [c for c in CHEXPERT_CLASSES if c in df.columns]
        labels = df[label_cols].fillna(0).values.astype(np.float32)
        if self.uncertain_policy == "zero":
            labels = np.clip(labels, 0, 1)
        elif self.uncertain_policy == "one":
            labels[labels == -1] = 1.0
        # "skip" keeps -1; the training loop uses a mask
        return labels

    def get_primary_label(self, idx: int) -> int:
        """
        Get the primary (single) class label for conformal calibration.
        Returns the index of the first positive class, or 0 ("No Finding").
        """
        row = self.labels[idx]
        positives = np.where(row > 0.5)[0]
        return int(positives[0]) if len(positives) > 0 else 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.root / self.paths[idx]
        try:
            img = Image.open(str(path)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)

        img = self.transforms(img)
        return {
            "image":         img,
            "labels":        torch.from_numpy(self.labels[idx]),
            "primary_label": self.get_primary_label(idx),
            "path":          str(path),
        }


def build_chexpert_loaders(
    root: str,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    frontal_only: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders for CheXpert."""
    train_csv = Path(root) / "train.csv"
    valid_csv = Path(root) / "valid.csv"

    train_ds = CheXpertDataset(str(train_csv), root, "train",
                                image_size, frontal_only)
    valid_ds = CheXpertDataset(str(valid_csv), root, "valid",
                                image_size, frontal_only)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, valid_loader
