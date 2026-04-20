"""
Evaluation metrics for conformal CXR prediction.

Clinical metrics beyond standard coverage/set-size:

  - Coverage@alpha:         P(Y in S(X)) — the guarantee we want >= 1-alpha
  - Average set size:       smaller = more efficient/useful
  - Singleton rate:         fraction of examples where |S(x)| = 1 (decisive)
  - Empty set rate:         fraction of examples where S(x) = {} (abstains)
  - ET rate (empirical miscoverage on critical findings):
      For high-stakes pathologies (Pneumothorax, Consolidation, Pneumonia),
      what fraction of TRUE positives are MISSED by the prediction set?
      This is the key clinical safety metric.
  - AUC (per class):        standard AUROC for the underlying classifier
  - ECE:                    expected calibration error of softmax scores
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score


CHEXPERT_CLASSES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

# High-risk pathologies where missing is dangerous
CRITICAL_CLASSES = ["Consolidation", "Pneumonia", "Pneumothorax",
                    "Pleural Effusion", "Atelectasis"]


def coverage(pred_sets: List[List[int]], labels: np.ndarray) -> float:
    """Marginal coverage: fraction of examples where true label is in set."""
    n = len(labels)
    return sum(int(labels[i]) in pred_sets[i] for i in range(n)) / n


def avg_set_size(pred_sets: List[List[int]]) -> float:
    return float(np.mean([len(s) for s in pred_sets]))


def singleton_rate(pred_sets: List[List[int]]) -> float:
    return sum(1 for s in pred_sets if len(s) == 1) / len(pred_sets)


def empty_set_rate(pred_sets: List[List[int]]) -> float:
    return sum(1 for s in pred_sets if len(s) == 0) / len(pred_sets)


def per_class_coverage(
    pred_sets: List[List[int]],
    labels: np.ndarray,
    num_classes: int = 14,
) -> Dict[int, float]:
    """Coverage conditioned on Y=c for each class c."""
    result = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        cov_c   = sum(c in pred_sets[i] for i in indices) / len(indices)
        result[c] = float(cov_c)
    return result


def critical_miscoverage_rate(
    pred_sets: List[List[int]],
    labels: np.ndarray,
    class_names: List[str] = CHEXPERT_CLASSES,
    critical: List[str] = CRITICAL_CLASSES,
) -> Dict[str, float]:
    """
    For each critical pathology c:
        Fraction of TRUE POSITIVES (Y=c) where c NOT in S(X).
    Lower is better (0 = never miss a critical case).
    """
    result = {}
    for c_name in critical:
        if c_name not in class_names:
            continue
        c = class_names.index(c_name)
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        missed  = sum(c not in pred_sets[i] for i in indices)
        result[c_name] = float(missed / len(indices))
    return result


def compute_auc(logits: np.ndarray, multilabels: np.ndarray,
                class_names: List[str] = CHEXPERT_CLASSES) -> Dict[str, float]:
    """
    Per-class AUROC for multi-label classification.
    logits: (n, num_classes), multilabels: (n, num_classes) binary
    """
    result = {}
    for c, name in enumerate(class_names):
        if c >= logits.shape[1] or c >= multilabels.shape[1]:
            break
        y_true = multilabels[:, c]
        y_score = logits[:, c]
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            continue  # skip if only one class present
        try:
            result[name] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    result["mean_auc"] = float(np.mean(list(result.values()))) if result else 0.0
    return result


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """
    ECE: measures how well softmax confidences align with actual accuracy.
    Lower is better (0 = perfect calibration).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def full_evaluation_report(
    pred_sets:    List[List[int]],
    probs:        np.ndarray,
    primary_labels: np.ndarray,
    multilabels:  Optional[np.ndarray] = None,
    class_names:  List[str] = CHEXPERT_CLASSES,
) -> Dict:
    """Compute all metrics and return a single summary dict."""
    report = {
        "coverage":             coverage(pred_sets, primary_labels),
        "avg_set_size":         avg_set_size(pred_sets),
        "singleton_rate":       singleton_rate(pred_sets),
        "empty_set_rate":       empty_set_rate(pred_sets),
        "ece":                  expected_calibration_error(probs, primary_labels),
        "per_class_coverage":   per_class_coverage(pred_sets, primary_labels),
        "critical_miscoverage": critical_miscoverage_rate(pred_sets, primary_labels),
    }
    if multilabels is not None:
        report["auc"] = compute_auc(probs, multilabels, class_names)
    return report


def print_report(report: Dict, alpha: float = 0.10):
    """Pretty-print the evaluation report."""
    print(f"
{'='*60}")
    print(f"  Conformal Evaluation Report  (target coverage = {1-alpha:.0%})")
    print(f"{'='*60}")
    print(f"  Coverage:       {report['coverage']:.4f}  "
          f"({'✅ MET' if report['coverage'] >= 1-alpha else '❌ MISSED'})")
    print(f"  Avg set size:   {report['avg_set_size']:.2f}")
    print(f"  Singleton rate: {report['singleton_rate']:.3f}")
    print(f"  Empty set rate: {report['empty_set_rate']:.4f}")
    print(f"  ECE:            {report['ece']:.4f}")
    if "auc" in report:
        print(f"  Mean AUC:       {report['auc'].get('mean_auc',0):.4f}")
    print(f"
  Critical pathology miscoverage (lower = safer):")
    for c_name, rate in report.get("critical_miscoverage", {}).items():
        flag = "✅" if rate < alpha else "⚠️ "
        print(f"    {flag} {c_name:25s}: {rate:.4f}")
    print(f"{'='*60}
")
