from __future__ import annotations
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm
import os
import json

from Utils import amp_context


class EvaluateMSPDetector:
    def __init__(
        self,
        detector,
        model,
        tokenizer,
        class_names,
        threshold: float,
        device: torch.device,
        temperature: float = 1.0,
        score_mode: str = "low_confidence",
    ):
        self.detector = detector
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.threshold = threshold
        self.device = device
        self.temperature = temperature
        if score_mode not in {"low_confidence", "high_confidence"}:
            raise ValueError(f"Unsupported MSP score mode: {score_mode}")
        self.score_mode = score_mode

        self.detector.eval()
        self._text_features = None

    # -----------------------------------------------------
    # Build text features once
    # -----------------------------------------------------
    def _build_text_features(self):

        if self._text_features is not None:
            return self._text_features

        prompts = [f"a photo of a {name}." for name in self.class_names]
        tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            with amp_context(self.device):
                text_features = self.model.encode_text(tokens)

        text_features = text_features.float()
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        self._text_features = text_features
        return self._text_features

    # -----------------------------------------------------
    # Compute MSP scores
    # -----------------------------------------------------
    def _compute_msp(self, images):

        text_features = self._build_text_features()

        with amp_context(self.device):
            image_features = self.model.encode_image(images)

        image_features = image_features.float()
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        logits = image_features @ text_features.t()
        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp()
        probs = torch.softmax(logits / self.temperature, dim=1)

        max_probability = probs.max(dim=1).values
        msp = 1.0 - max_probability
        if self.score_mode == "high_confidence":
            msp = max_probability

        return msp

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
                desc="Evaluating MSP",
                leave=False,
                dynamic_ncols=True,
            )

            for (images, _), (adv_images, _) in pbar:
                images = images.to(self.device)
                adv_images = adv_images.to(self.device)
                # -------- CLEAN --------
                msp_clean = self._compute_msp(images)
                # -------- ADV --------
                msp_adv = self._compute_msp(adv_images)
                all_probs.append(msp_clean.cpu())
                all_labels.append(torch.zeros_like(msp_clean).cpu())
                all_probs.append(msp_adv.cpu())
                all_labels.append(torch.ones_like(msp_adv).cpu())

        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()

        return y_true, y_prob

    # -----------------------------------------------------
    # Metrics (same as your original)
    # -----------------------------------------------------
    def _compute_metrics(self, y_true, y_prob) -> Dict[str, float]:

        if not np.isfinite(y_prob).all():
            raise ValueError("Non-finite MSP scores; evaluation cannot be reported")
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
        results["score_mode"] = self.score_mode

        print("\n📊 MSP Evaluation Results")

        lines = []
        for k, v in results.items():
            line = f"{k:12s}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:12s}: {v}"
            print(line)
            lines.append(line)

        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "evaluation_msp.txt")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n📊 MSP Evaluation Results\n")
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
