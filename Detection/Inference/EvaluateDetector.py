from __future__ import annotations
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

from Utils import amp_context
import os
import json


class EvaluateEnergyDetector:
    def __init__(
        self,
        detector,
        feature_extractor,
        threshold: float,
        clean_mean: torch.Tensor,
        clean_cov_inv_sqrt: torch.Tensor,
        device: torch.device,
        temperature: float = 0.7,
    ):
        self.detector = detector
        self.fe = feature_extractor
        self.threshold = threshold
        self.device = device
        self.temperature = temperature

        self.clean_mean = clean_mean
        self.clean_cov_inv_sqrt = clean_cov_inv_sqrt

        self.detector.eval()

    # -----------------------------------------------------
    # Whitening (same as training)
    # -----------------------------------------------------
    def _whiten(self, x: torch.Tensor):
        if self.clean_mean is None or self.clean_cov_inv_sqrt is None:
            return x
        return (x - self.clean_mean) * self.clean_cov_inv_sqrt

    # -----------------------------------------------------
    # Collect predictions
    # -----------------------------------------------------
    def _collect_predictions(self, clean_loader, adv_loader):

        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        if len(clean_loader) != len(adv_loader):
            raise ValueError(
                f"loader and adv_loader must have same length.\nlen(loader)={len(clean_loader)}, len(adv_loader)={len(adv_loader)}"
            )

        with torch.no_grad():
            iterator = zip(clean_loader, adv_loader)
            total = min(len(clean_loader), len(adv_loader))

            pbar = tqdm(
                iterator,
                total=total,
                desc="Evaluating",
                leave=False,
                dynamic_ncols=True,
            )

            for batch in pbar:

                # -------------------------------------------------
                # CLEAN ONLY MODE
                # -------------------------------------------------
                if adv_loader is None:

                    images, _ = batch
                    images = images.to(self.device)

                    with amp_context(self.device):

                        phi = self.fe.phi(images)
                        phi = self._whiten(phi)

                        logits = self.detector(phi)
                        probs = logits.float()

                    all_probs.append(probs.cpu())
                    all_labels.append(torch.zeros_like(probs).cpu())

                # -------------------------------------------------
                # CLEAN + ADV MODE
                # -------------------------------------------------
                else:

                    (images, _), (adv_images, _) = batch

                    images = images.to(self.device)
                    adv_images = adv_images.to(self.device)

                    with amp_context(self.device):

                        # -------- CLEAN --------
                        phi_clean = self.fe.phi(images)
                        phi_clean = self._whiten(phi_clean)

                        logits_clean = self.detector(phi_clean)
                        probs_clean = logits_clean.float()

                        # -------- ADV --------
                        phi_adv = self.fe.phi(adv_images)
                        phi_adv = self._whiten(phi_adv)

                        logits_adv = self.detector(phi_adv)
                        probs_adv = logits_adv.float()

                    all_probs.append(probs_clean.cpu())
                    all_labels.append(torch.zeros_like(probs_clean).cpu())

                    all_probs.append(probs_adv.cpu())
                    all_labels.append(torch.ones_like(probs_adv).cpu())

        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()

        return y_true, y_prob

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    def _compute_metrics(self, y_true, y_prob) -> Dict[str, float]:

        if not np.isfinite(y_prob).all():
            raise ValueError("Non-finite detector scores; evaluation cannot be reported")
        y_pred = (y_prob >= self.threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)

        try:
            f1 = f1_score(y_true, y_pred)
        except Exception:
            f1 = 0.0

        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = 0.5

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)

            valid = tpr[fpr <= 0.05]
            tpr_at_5fpr = valid.max() if valid.size else 0.0

        except Exception:
            tpr_at_5fpr = 0.0

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "tpr@5%fpr": float(tpr_at_5fpr),
        }

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def evaluate(self, clean_loader, adv_loader, save_dir):
        import result_store
        cached = result_store.get(save_dir)
        if cached is not None:
            return cached
        y_true, y_prob = self._collect_predictions(clean_loader, adv_loader)

        results = self._compute_metrics(y_true, y_prob)

        print("\n📊 Evaluation Results")

        lines = []
        for k, v in results.items():
            line = f"{k:12s}: {v:.4f}"
            print(line)
            lines.append(line)

        # ---------------------------------------
        # Ensure directory exists
        # ---------------------------------------
        os.makedirs(save_dir, exist_ok=True)

        log_path = os.path.join(save_dir, "evaluation.txt")

        # ---------------------------------------
        # Write logs (creates file if missing)
        # ---------------------------------------
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n📊 Evaluation Results\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

        if result_store.enabled():
            result_store.put(save_dir, results)
        else:
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
        np.savez_compressed(os.path.join(save_dir, "scores.npz"), labels=y_true, scores=y_prob, threshold=self.threshold)
        return results
