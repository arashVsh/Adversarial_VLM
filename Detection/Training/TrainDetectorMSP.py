"""
MSPDetector.py
--------------

Maximum Softmax Probability (MSP) detector.

The adversarial anomaly score is 1 - MSP, so larger values consistently
indicate a greater likelihood of an adversarial input.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from Utils import amp_context


def orient_msp_scores(labels, low_confidence_scores):
    """Choose MSP polarity using validation labels only."""
    base_auc = float(roc_auc_score(labels, low_confidence_scores))
    if base_auc < 0.5:
        return 1.0 - low_confidence_scores, "high_confidence", base_auc
    return low_confidence_scores, "low_confidence", base_auc


# =========================================================
# Config
# =========================================================

@dataclass
class TrainConfig:
    device: torch.device
    temperature: float = 1.0


# =========================================================
# Detector
# =========================================================

class MSPDetector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return 1.0 - probs.max(dim=1).values


# =========================================================
# Trainer
# =========================================================

class MSPTrainDetector:

    def __init__(
        self,
        feature_extractor,
        model,
        tokenizer,
        class_names,
        device: torch.device,
        cfg: TrainConfig,
    ):
        self.fe = feature_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.cfg = cfg

        self.detector = MSPDetector().to(self.device)

        self.clean_mean = None
        self.clean_cov_inv_sqrt = None

        self.history = {
            "train": {"loss": [], "acc": [], "f1": [], "roc_auc": []},
            "val": {"loss": [], "acc": [], "f1": [], "roc_auc": []},
        }

        self._text_features = None
        self.score_mode = "low_confidence"
        self.validation_base_auc = None

    # =========================================================
    # Build text features
    # =========================================================

    def _build_text_features(self):
        if self._text_features is not None:
            return self._text_features

        prompts = [f"a photo of a {name}." for name in self.class_names]
        tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            with amp_context(self.device):
                text_features = self.model.encode_text(tokens)

        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        self._text_features = text_features
        return self._text_features

    # =========================================================
    # Compute probabilities
    # =========================================================

    def _compute_probs(self, images: torch.Tensor) -> torch.Tensor:
        text_features = self._build_text_features()

        with amp_context(self.device):
            image_features = self.model.encode_image(images)

        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        logits = image_features @ text_features.t()
        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp()
        probs = torch.softmax(logits / self.cfg.temperature, dim=1)

        return probs

    # =========================================================
    # Step (NO training — only stats)
    # =========================================================

    def _step(self, images, adv_images):

        images = images.to(self.device)
        adv_images = adv_images.to(self.device)

        with torch.no_grad():
            probs_clean = self._compute_probs(images)
            probs_adv = self._compute_probs(adv_images)

            score_clean = self.detector(probs_clean)
            score_adv = self.detector(probs_adv)

        adv_probs = torch.cat([score_clean, score_adv], dim=0)

        # Labels
        y = torch.cat(
            [
                torch.zeros(images.size(0), device=self.device),
                torch.ones(images.size(0), device=self.device),
            ],
            dim=0,
        )

        preds = (adv_probs > 0.5).float()

        y_cpu = y.cpu()
        adv_probs_cpu = adv_probs.cpu()

        # Dummy loss (for logging only)
        eps = 1e-12
        loss = -(
            y * torch.log(adv_probs + eps) +
            (1 - y) * torch.log(1 - adv_probs + eps)
        ).mean()

        acc = (preds.cpu() == y_cpu).float().mean().item()
        f1 = f1_score(y_cpu.numpy(), preds.cpu().numpy())

        try:
            roc = roc_auc_score(y_cpu.numpy(), adv_probs_cpu.numpy())
        except Exception:
            roc = 0.5

        return loss.item(), acc, f1, roc

    # =========================================================
    # Fit (threshold search)
    # =========================================================

    def fit(
        self,
        train_clean_loader,
        val_clean_loader,
        train_adv_loader,
        val_adv_loader,
        save_dir: str,
    ):

        print("📐 MSP detector (parameter-free)")

        # MSP has no learned parameters.  Training-split adversarial examples do
        # not affect its score direction or threshold, so generating them only
        # for diagnostic logging wastes most of the baseline runtime.
        print("[MSP] Skipping adversarial training split (parameter-free baseline)")

        # =========================
        # VALIDATION (threshold)
        # =========================
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for (images, _), (adv_images, _) in zip(val_clean_loader, val_adv_loader):

                images = images.to(self.device)
                adv_images = adv_images.to(self.device)

                probs_clean = self._compute_probs(images)
                probs_adv = self._compute_probs(adv_images)

                score_clean = self.detector(probs_clean)
                score_adv = self.detector(probs_adv)

                adv_probs = torch.cat([score_clean, score_adv], dim=0)

                y = torch.cat(
                    [
                        torch.zeros(images.size(0), device=self.device),
                        torch.ones(images.size(0), device=self.device),
                    ],
                    dim=0,
                )

                all_probs.append(adv_probs.cpu())
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Some attacks reduce confidence, while others push the model toward a
        # confidently wrong class.  Select that one-dimensional orientation on
        # validation data and freeze it before evaluating the test set.
        all_probs, self.score_mode, base_auc = orient_msp_scores(all_labels, all_probs)
        self.validation_base_auc = base_auc

        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        val_preds = (all_probs >= optimal_threshold).astype(int)
        val_acc = (val_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, val_preds)

        val_roc = roc_auc_score(all_labels, all_probs)

        print(
            f"[MSP] VAL acc={val_acc:.4f} f1={val_f1:.4f} "
            f"roc_auc={val_roc:.4f} thr={optimal_threshold:.4f} "
            f"score_mode={self.score_mode} base_auc={base_auc:.4f}"
        )

        return float(optimal_threshold)
