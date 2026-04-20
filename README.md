# ConformalCXR

> **Uncertainty-Aware Chest X-Ray Diagnosis via Conformal Prediction on Vision-Language Models**
> *Irfan Sadiq Rahat · 2025–2026*

[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-b31b1b.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CheXpert](https://img.shields.io/badge/Dataset-CheXpert-green.svg)](https://stanfordmlgroup.github.io/competitions/chexpert/)

---

## Overview

Vision-language models (VLMs) for radiology routinely output confident predictions
that are clinically wrong — a 92% softmax score provides no formal safety guarantee
and cannot tell a clinician when to trust the model.

**ConformalCXR** adds a post-hoc conformal prediction head to CheXagent (a
state-of-the-art chest X-ray VLM) that:

1. Produces **prediction sets** with a user-specified marginal coverage guarantee
   (e.g. the true pathology label is in the set with ≥90% probability)
2. Uses **RAPS** (Regularized Adaptive Prediction Sets) so sets stay small on
   easy cases and expand only when the model is genuinely uncertain
3. Introduces **class-conditional calibration** so coverage holds separately for
   each of the 14 CheXpert pathology classes — not just on average
4. Requires **zero retraining** — pure post-hoc, works on any frozen VLM

We evaluate on CheXpert (224,316 images) and NIH ChestX-ray14 (112,120 images),
reporting coverage, set size, singleton rate, and clinical utility metrics.

---

## Key Results (Target)

| Method | Coverage@90% | Avg Set Size | Singleton Rate | ET Rate |
|---|---|---|---|---|
| Softmax threshold | 76.2% | 1.0 | 100% | 8.3% |
| Temperature scaling | 81.4% | 1.4 | 82% | 6.1% |
| APS (Angelopoulos et al.) | 90.1% | 3.8 | 61% | 1.2% |
| **RAPS (ours baseline)** | **90.0%** | **2.1** | **74%** | **1.8%** |
| **RAPS + class-conditional** | **90.0%** | **1.7** | **81%** | **0.4%** |

*ET Rate = empirical miscoverage on critical (high-risk) findings.*
*Results to be updated as experiments complete.*

---

## Architecture

```
CheXpert Image (224×224)
        │
  ┌─────▼────────────────┐
  │   CheXagent VLM      │  ← frozen, no retraining
  │  (ViT + Q-Former)    │
  └─────┬────────────────┘
        │  logits (B, 14)
  ┌─────▼────────────────┐
  │  Conformal Head      │
  │  ┌──────────────┐    │
  │  │ RAPS scores  │    │  ← calibrated on held-out set
  │  │ per class    │    │
  │  └──────┬───────┘    │
  │  ┌──────▼───────┐    │
  │  │ Threshold τ  │    │  ← τ chosen to guarantee α-coverage
  │  └──────┬───────┘    │
  └─────────┼────────────┘
            │
   Prediction Set S(x) ⊆ {1,...,14}
   P(Y ∈ S(X)) ≥ 1-α  [guaranteed]
```

---

## Repository Structure

```
ConformalCXR/
├── src/
│   ├── models/
│   │   ├── chexagent_wrapper.py    # Frozen CheXagent feature extractor
│   │   └── linear_probe.py        # Linear probe head for 14-class CXR
│   ├── conformal/
│   │   ├── raps.py                 # RAPS algorithm (core contribution)
│   │   ├── aps.py                  # APS baseline
│   │   ├── naive.py                # Naive softmax threshold baseline
│   │   ├── calibration.py          # Calibration set management
│   │   └── class_conditional.py   # Class-conditional coverage
│   ├── data/
│   │   ├── chexpert_dataset.py    # CheXpert dataloader
│   │   └── nihcxr_dataset.py      # NIH ChestX-ray14 dataloader
│   ├── training/
│   │   └── train_probe.py         # Train the linear classification probe
│   └── evaluation/
│       ├── metrics.py             # Coverage, set size, clinical metrics
│       └── evaluator.py           # Full evaluation pipeline
├── configs/
│   └── chexpert.yaml
├── scripts/
│   ├── train_probe.py             # Train linear probe on CheXagent features
│   ├── calibrate.py               # Run conformal calibration
│   └── evaluate.py                # Full evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_calibration_analysis.ipynb
├── tests/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/ConformalCXR.git
cd ConformalCXR
pip install -r requirements.txt

# Download CheXpert (free Stanford registration required)
# https://stanfordmlgroup.github.io/competitions/chexpert/

# Download NIH ChestX-ray14 (free)
# https://nihcc.app.box.com/v/ChestXray-NIHCC

python scripts/train_probe.py --config configs/chexpert.yaml
python scripts/calibrate.py   --config configs/chexpert.yaml
python scripts/evaluate.py    --config configs/chexpert.yaml
```

---

## Datasets

| Dataset | Images | Classes | Source |
|---|---|---|---|
| CheXpert | 224,316 | 14 pathologies | Stanford (free registration) |
| NIH ChestX-ray14 | 112,120 | 14 pathologies | NIH (free download) |

Both datasets use the same 14 CheXpert pathology labels for cross-dataset evaluation.

---

## Citation

```bibtex
@article{rahat2026conformalcxr,
  title={Uncertainty-Aware Chest X-Ray Diagnosis via Conformal Prediction on Vision-Language Models},
  author={Rahat, Irfan Sadiq},
  journal={IEEE Transactions on Medical Imaging / NeurIPS Workshop},
  year={2026}
}
```

---

## Acknowledgements

- [CheXagent](https://stanford-aimi.github.io/chexagent.html) — Chen et al., Stanford AIMI
- [RAPS](https://github.com/aangelopoulos/conformal-classification) — Angelopoulos et al., UC Berkeley
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) — Irvin et al., Stanford
- [BioViL-T](https://github.com/microsoft/hi-ml) — Bannur et al., Microsoft Research

---

*First-author research paper · Target: IEEE TMI 2026 / NeurIPS 2026 ML4H Workshop*
