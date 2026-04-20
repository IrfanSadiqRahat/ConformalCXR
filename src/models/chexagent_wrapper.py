"""
Frozen CheXagent feature extractor wrapper.

CheXagent (Chen et al., Stanford AIMI 2024) is a vision-language foundation
model trained on 6M CXR-text pairs. We use only its vision encoder as a
frozen feature extractor — no retraining, pure post-hoc approach.

If CheXagent is unavailable, falls back to torchvision ResNet-50 pretrained
on CheXpert (TorchXRayVision) as a drop-in replacement for experiments.

Reference:
  CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation
  Chen et al., Stanford AIMI, 2024
  https://arxiv.org/abs/2401.12208
"""

import torch
import torch.nn as nn
from typing import Optional


class CheXagentEncoder(nn.Module):
    """
    Frozen CheXagent vision encoder.
    Outputs (B, feature_dim) embeddings for downstream classification.

    Args:
        model_name: HuggingFace model id or local path
        feature_dim: output embedding dimension (1408 for CheXagent ViT-L)
        device: torch device
    """

    FEATURE_DIM = 1408  # CheXagent ViT-L/14 output dim

    def __init__(self, model_name: str = "StanfordAIMI/CheXagent",
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = self.FEATURE_DIM
        self._load_model(model_name)
        self._freeze()

    def _load_model(self, model_name: str):
        try:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True)
            # Use only the vision encoder (ignore language decoder)
            self.encoder = model.vision_model
            print(f"[CheXagent] Loaded vision encoder from {model_name}")
        except Exception as e:
            print(f"[CheXagent] Could not load {model_name}: {e}")
            print("[CheXagent] Falling back to ResNet-50 (TorchXRayVision)")
            self._load_fallback()

    def _load_fallback(self):
        """Use DenseNet121 pretrained on CheXpert as fallback."""
        try:
            import torchxrayvision as xrv
            self.encoder = xrv.models.DenseNet(weights="densenet121-res224-chex")
            self.encoder.classifier = nn.Identity()
            self.feature_dim = 1024  # DenseNet121 feature dim
            self.processor = None
            print("[Fallback] TorchXRayVision DenseNet121 loaded")
        except ImportError:
            # Final fallback: torchvision ResNet-50
            import torchvision.models as tvm
            m = tvm.resnet50(pretrained=True)
            m.fc = nn.Identity()
            self.encoder = m
            self.feature_dim = 2048
            self.processor = None
            print("[Fallback] ResNet-50 (ImageNet pretrained) loaded")

    def _freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"[CheXagentEncoder] Frozen {total:,} params | "
              f"feature_dim={self.feature_dim}")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) normalized tensor
        Returns:
            features: (B, feature_dim)
        """
        self.encoder.eval()
        out = self.encoder(images)
        # Handle both (B, D) and (B, D, 1, 1) output shapes
        if out.dim() == 4:
            out = out.flatten(1)
        elif hasattr(out, "pooler_output"):
            out = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state[:, 0]  # CLS token
        return out


class LinearProbe(nn.Module):
    """
    Lightweight linear probe on top of frozen backbone features.
    Maps (B, feature_dim) → (B, num_classes) logits.

    For multi-label CXR classification (14 CheXpert classes), each class
    is treated as an independent binary classification problem.

    Args:
        feature_dim: input feature dimension from backbone
        num_classes: number of output classes (14 for CheXpert)
        hidden_dim:  if > 0, adds one hidden layer with GELU activation
        dropout:     dropout before classifier
    """

    def __init__(self, feature_dim: int, num_classes: int = 14,
                 hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, feature_dim) → (B, num_classes) logits"""
        return self.head(features)


class CheXClassifier(nn.Module):
    """
    Full pipeline: frozen CheXagent encoder + trainable linear probe.
    This is what gets trained in scripts/train_probe.py.
    """

    def __init__(self, model_name: str = "StanfordAIMI/CheXagent",
                 num_classes: int = 14, hidden_dim: int = 512,
                 dropout: float = 0.2, device: str = "cuda"):
        super().__init__()
        self.encoder = CheXagentEncoder(model_name=model_name, device=device)
        self.probe   = LinearProbe(
            feature_dim=self.encoder.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_classes) logits"""
        features = self.encoder(images)
        return self.probe(features)

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features only (for caching)."""
        return self.encoder(images)

    def get_trainable_params(self):
        return list(self.probe.parameters())
