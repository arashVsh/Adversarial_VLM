"""
MahalanobisDetector.py
----------------------
Mahalanobis detector based on:

"A Simple Unified Framework for Detecting Out-of-Distribution Samples
and Adversarial Attacks"

Adapted to the current VLM pipeline and keeps a fit(...) interface
compatible with TrainDetector.

Core ideas:
  ✔ class-conditional Gaussian means in feature space
  ✔ tied covariance across classes
  ✔ Mahalanobis confidence score
  ✔ optional feature ensemble across multiple layers
  ✔ optional logistic-regression calibration
  ✔ optional input preprocessing
  ✔ threshold optimization using validation metrics
  ✔ compatibility with image input OR pre-extracted feature input
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from Utils import amp_context


# =========================================================
# ⚙️ Config
# =========================================================


@dataclass
class MahalanobisConfig:
    device: torch.device

    # numerical stability
    covariance_eps: float = 1e-4

    # score settings
    use_feature_ensemble: bool = True
    use_input_preprocessing: bool = False
    noise_eps: float = 0.001

    # feature handling
    normalize_features: bool = False

    # validation / thresholding
    optimize_threshold: bool = True
    monitor: str = "f1"  # "f1" or "acc"

    # saving
    save_stats: bool = True


# =========================================================
# 🧠 Mahalanobis Detector
# =========================================================


class MahalanobisDetector:
    def __init__(
        self,
        feature_extractor,
        model,
        tokenizer,
        class_names,
        device: torch.device,
        cfg: Optional[MahalanobisConfig] = None,
    ):
        self.fe = feature_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.cfg = cfg if cfg is not None else MahalanobisConfig(device=device)

        self.num_classes = len(class_names)

        # layer_name -> {"means": [C, D], "precision": [D, D]}
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}

        # Optional logistic regression calibrator over per-layer scores
        self.lr_calibrator: Optional[LogisticRegression] = None

        # Selected decision threshold on adversarial probability
        self.threshold: float = 0.5

        self.history = {
            "train": {
                "acc": [],
                "f1": [],
                "roc_auc": [],
            },
            "val": {
                "acc": [],
                "f1": [],
                "roc_auc": [],
            },
        }

    # =========================================================
    # Compatibility helpers
    # =========================================================

    def eval(self):
        return self

    def train(self):
        return self

    def parameters(self):
        return []

    # =========================================================
    # Feature extraction helpers
    # =========================================================

    def _is_image_tensor(self, x: torch.Tensor) -> bool:
        return x.ndim == 4

    def _is_feature_tensor(self, x: torch.Tensor) -> bool:
        return x.ndim == 2

    def _reduce_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert arbitrary feature tensor into [B, D].
        """
        if feat.ndim == 2:
            out = feat
        elif feat.ndim == 4:
            out = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        elif feat.ndim == 3:
            out = feat.mean(dim=1)
        else:
            out = feat.flatten(1)

        if self.cfg.normalize_features:
            out = F.normalize(out, dim=1)

        return out.float()

    def _extract_layer_features_from_images(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Expected priority:
          1) feature_extractor.layer_features(images) -> dict[name] = tensor
          2) feature_extractor.phi(images) -> final feature only
        """
        with amp_context(self.device):
            if hasattr(self.fe, "layer_features") and callable(self.fe.layer_features):
                raw = self.fe.layer_features(images)
                if not isinstance(raw, dict):
                    raise TypeError(
                        "feature_extractor.layer_features(images) must return a dict."
                    )
                return {k: self._reduce_feature(v) for k, v in raw.items()}

            phi = self.fe.phi(images)
            return {"final": self._reduce_feature(phi)}

    def _extract_layer_features(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Supports both:
          - images [B, C, H, W]
          - precomputed features [B, D]

        This is necessary because EvaluateDetector may pass phi(x)
        instead of raw images.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input to MahalanobisDetector must be a torch.Tensor.")

        x = x.to(self.device)

        if self._is_image_tensor(x):
            return self._extract_layer_features_from_images(x)

        if self._is_feature_tensor(x):
            return {"final": self._reduce_feature(x)}

        raise ValueError(
            f"Unsupported input shape {tuple(x.shape)}. "
            f"Expected image tensor [B, C, H, W] or feature tensor [B, D]."
        )

    # =========================================================
    # Mahalanobis statistics
    # =========================================================

    def _fit_gaussian_stats(self, train_loader):
        from tqdm import tqdm

        print("📐 Fitting class-conditional Gaussian statistics...")

        layer_feats_all: Dict[str, List[torch.Tensor]] = {}
        labels_all: List[torch.Tensor] = []

        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(
                train_loader,
                desc="Extracting clean train features",
                total=len(train_loader),
                leave=True,
            ):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                feats = self._extract_layer_features(images)

                for layer_name, feat in feats.items():
                    layer_feats_all.setdefault(layer_name, []).append(
                        feat.detach().cpu()
                    )

                labels_all.append(labels.detach().cpu())

        if len(labels_all) == 0:
            raise ValueError(
                "train_clean_loader is empty. Cannot fit Mahalanobis stats."
            )

        labels_all = torch.cat(labels_all, dim=0).long()
        unique_labels = torch.unique(labels_all)

        if unique_labels.numel() != self.num_classes:
            print(
                f"⚠️ Warning: expected {self.num_classes} classes, "
                f"but found {unique_labels.numel()} labels in train_clean_loader."
            )

        for layer_name, feat_list in layer_feats_all.items():
            X = torch.cat(feat_list, dim=0)
            _, D = X.shape

            valid_classes = torch.unique(labels_all).tolist()
            means = []
            centered_chunks = []

            for c in valid_classes:
                mask = labels_all == c
                Xc = X[mask]
                mu_c = Xc.mean(dim=0)
                means.append(mu_c)
                centered_chunks.append(Xc - mu_c)

            means = torch.stack(means, dim=0)
            centered = torch.cat(centered_chunks, dim=0)

            cov = centered.T @ centered / float(centered.shape[0])
            cov = cov + self.cfg.covariance_eps * torch.eye(D, dtype=cov.dtype)

            precision = torch.linalg.inv(cov)

            self.stats[layer_name] = {
                "means": means.to(self.device).float(),
                "precision": precision.to(self.device).float(),
            }

            print(
                f"✅ Layer '{layer_name}': fitted means {means.shape}, "
                f"precision {precision.shape}"
            )

    # =========================================================
    # Mahalanobis score computation
    # =========================================================

    def _mahalanobis_score_single_layer(
        self,
        feat: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Returns score [B], where larger = more normal / more ID.
        score = - min_c (f - mu_c)^T P (f - mu_c)
        """
        stats = self.stats[layer_name]
        means = stats["means"]
        precision = stats["precision"]

        delta = feat.unsqueeze(1) - means.unsqueeze(0)
        m = torch.einsum("bcd,df,bcf->bc", delta, precision, delta)
        score = -m.min(dim=1).values
        return score

    def _compute_layer_scores(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats = self._extract_layer_features(x)
        scores = {}

        for layer_name, feat in feats.items():
            if layer_name not in self.stats:
                raise KeyError(
                    f"Layer '{layer_name}' was extracted at inference time but has no fitted stats."
                )
            scores[layer_name] = self._mahalanobis_score_single_layer(feat, layer_name)

        return scores

    def _stack_scores(self, score_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ordered_layers = list(self.stats.keys())
        return torch.stack([score_dict[layer] for layer in ordered_layers], dim=1)

    # =========================================================
    # Optional input preprocessing
    # =========================================================

    def _preprocess_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """
        Paper-style preprocessing only makes sense for raw images.
        If features are already passed in, we bypass preprocessing.
        """
        if not self.cfg.use_input_preprocessing:
            return images

        if not self._is_image_tensor(images):
            return images

        x = images.detach().clone().to(self.device)
        x.requires_grad_(True)

        feats = self._extract_layer_features_from_images(x)
        last_layer = list(feats.keys())[-1]
        feat = feats[last_layer]

        stats = self.stats[last_layer]
        means = stats["means"]
        precision = stats["precision"]

        delta = feat.unsqueeze(1) - means.unsqueeze(0)
        m = torch.einsum("bcd,df,bcf->bc", delta, precision, delta)
        nearest = m.argmin(dim=1)

        chosen_means = means[nearest]
        chosen_delta = feat - chosen_means
        score = -torch.einsum(
            "bd,df,bf->b", chosen_delta, precision, chosen_delta
        ).sum()

        grad = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        x_hat = x + self.cfg.noise_eps * torch.sign(grad)
        return x_hat.detach()

    # =========================================================
    # Score collection helpers
    # =========================================================

    def _collect_loader_scores(
        self,
        loader,
        adv_label: int,
        desc: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from tqdm import tqdm

        self.model.eval()

        feature_list: List[np.ndarray] = []
        label_list: List[np.ndarray] = []

        if loader is None:
            raise ValueError(f"{desc} loader is None.")

        with torch.no_grad():
            for images, _ in tqdm(loader, desc=desc, total=len(loader), leave=False):
                images = images.to(self.device, non_blocking=True)

                if self.cfg.use_input_preprocessing:
                    images = self._preprocess_inputs(images)

                score_dict = self._compute_layer_scores(images)
                X = self._stack_scores(score_dict).detach().cpu().numpy()
                y = np.full((X.shape[0],), adv_label, dtype=np.int64)

                feature_list.append(X)
                label_list.append(y)

        if len(feature_list) == 0:
            raise ValueError(f"{desc} is empty.")

        X_all = np.concatenate(feature_list, axis=0)
        y_all = np.concatenate(label_list, axis=0)
        return X_all, y_all

    def _collect_dataset_scores(
        self,
        clean_loader,
        adv_loader,
        clean_desc: str,
        adv_desc: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        clean_X, clean_y = self._collect_loader_scores(
            clean_loader,
            adv_label=0,
            desc=clean_desc,
        )
        adv_X, adv_y = self._collect_loader_scores(
            adv_loader,
            adv_label=1,
            desc=adv_desc,
        )

        X_all = np.concatenate([clean_X, adv_X], axis=0)
        y_all = np.concatenate([clean_y, adv_y], axis=0)
        return X_all, y_all

    # =========================================================
    # Probability / prediction helpers
    # =========================================================

    @staticmethod
    def _single_layer_probs_from_scores(X: np.ndarray) -> np.ndarray:
        raw = X[:, -1].astype(np.float64)
        raw = (raw - raw.mean()) / (raw.std() + 1e-12)
        probs_adv = 1.0 / (1.0 + np.exp(raw))
        return probs_adv

    def _predict_proba_from_score_matrix(self, X: np.ndarray) -> np.ndarray:
        if (
            self.cfg.use_feature_ensemble
            and X.shape[1] > 1
            and self.lr_calibrator is not None
        ):
            probs_adv = self.lr_calibrator.predict_proba(X)[:, 1]
        else:
            probs_adv = self._single_layer_probs_from_scores(X)

        return probs_adv.astype(np.float64)

    def _evaluate_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        probs = self._predict_proba_from_score_matrix(X)
        preds = (probs > threshold).astype(np.int64)

        acc = float((preds == y).mean())
        f1 = float(f1_score(y, preds))

        try:
            roc = float(roc_auc_score(y, probs))
        except Exception:
            roc = 0.5

        return acc, f1, roc

    def _select_threshold(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        if not self.cfg.optimize_threshold:
            return 0.5

        precision, recall, thresholds = precision_recall_curve(y_true, probs)

        if thresholds.size == 0:
            return 0.5

        if self.cfg.monitor.lower() == "acc":
            best_thr = 0.5
            best_metric = -1.0

            for thr in thresholds:
                preds = (probs > thr).astype(np.int64)
                acc = float((preds == y_true).mean())
                if acc > best_metric:
                    best_metric = acc
                    best_thr = float(thr)

            return best_thr

        f1_scores = (
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx])

    # =========================================================
    # Public prediction API
    # =========================================================

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns probability of being adversarial (class 1).

        Supports:
          - images [B, C, H, W]
          - features [B, D]
        """
        self.model.eval()

        x = x.to(self.device)

        if self.cfg.use_input_preprocessing:
            x = self._preprocess_inputs(x)

        score_dict = self._compute_layer_scores(x)
        X = self._stack_scores(score_dict).detach().cpu().numpy()
        probs_adv = self._predict_proba_from_score_matrix(X)

        return torch.tensor(probs_adv, dtype=torch.float32, device=self.device)

    def predict(
        self, x: torch.Tensor, threshold: Optional[float] = None
    ) -> torch.Tensor:
        thr = self.threshold if threshold is None else float(threshold)
        probs = self.predict_proba(x)
        return (probs > thr).long()

    # =========================================================
    # Fit
    # =========================================================

    def fit(
        self,
        train_clean_loader,
        val_clean_loader,
        train_adv_loader,
        val_adv_loader,
        save_dir: str,
    ):
        if train_clean_loader is None:
            raise ValueError("train_clean_loader is required.")
        if val_clean_loader is None:
            raise ValueError("val_clean_loader is required.")
        if train_adv_loader is None:
            raise ValueError("train_adv_loader is required.")
        if val_adv_loader is None:
            raise ValueError("val_adv_loader is required.")

        os.makedirs(save_dir, exist_ok=True)

        self._fit_gaussian_stats(train_clean_loader)

        print("📊 Collecting TRAIN Mahalanobis scores...")
        X_train, y_train = self._collect_dataset_scores(
            clean_loader=train_clean_loader,
            adv_loader=train_adv_loader,
            clean_desc="Train clean scores",
            adv_desc="Train adv scores",
        )

        print("📊 Collecting VAL Mahalanobis scores...")
        X_val, y_val = self._collect_dataset_scores(
            clean_loader=val_clean_loader,
            adv_loader=val_adv_loader,
            clean_desc="Val clean scores",
            adv_desc="Val adv scores",
        )

        if self.cfg.use_feature_ensemble and X_train.shape[1] > 1:
            print("🧮 Training logistic-regression calibrator on TRAIN layer scores...")
            self.lr_calibrator = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
            )
            self.lr_calibrator.fit(X_train, y_train)
        else:
            self.lr_calibrator = None

        train_probs = self._predict_proba_from_score_matrix(X_train)
        train_threshold = self._select_threshold(train_probs, y_train)
        train_acc, train_f1, train_roc = self._evaluate_split(
            X_train, y_train, threshold=train_threshold
        )

        print(
            f"✅ Mahalanobis TRAIN "
            f"acc={train_acc:.4f} "
            f"f1={train_f1:.4f} "
            f"roc_auc={train_roc:.4f} "
            f"thr={train_threshold:.4f}"
        )

        self.history["train"]["acc"].append(train_acc)
        self.history["train"]["f1"].append(train_f1)
        self.history["train"]["roc_auc"].append(train_roc)

        val_probs = self._predict_proba_from_score_matrix(X_val)
        best_threshold = self._select_threshold(val_probs, y_val)
        self.threshold = float(best_threshold)

        val_acc, val_f1, val_roc = self._evaluate_split(
            X_val, y_val, threshold=self.threshold
        )

        print(
            f"✅ Mahalanobis VAL "
            f"acc={val_acc:.4f} "
            f"f1={val_f1:.4f} "
            f"roc_auc={val_roc:.4f} "
            f"thr={self.threshold:.4f}"
        )

        self.history["val"]["acc"].append(val_acc)
        self.history["val"]["f1"].append(val_f1)
        self.history["val"]["roc_auc"].append(val_roc)

        if self.cfg.save_stats:
            self._save_detector(save_dir)

        self.plot_validation_summary(save_dir)

        return float(self.threshold)

    # =========================================================
    # Save / load
    # =========================================================

    def _save_detector(self, save_dir: str):
        stats_to_save = {}

        for layer_name, layer_stats in self.stats.items():
            stats_to_save[layer_name] = {
                "means": layer_stats["means"].detach().cpu(),
                "precision": layer_stats["precision"].detach().cpu(),
            }

        torch.save(stats_to_save, os.path.join(save_dir, "mahalanobis_stats.pt"))

        if self.lr_calibrator is not None:
            with open(os.path.join(save_dir, "mahalanobis_lr.pkl"), "wb") as f:
                pickle.dump(self.lr_calibrator, f)

        with open(os.path.join(save_dir, "mahalanobis_meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "threshold": self.threshold,
                    "history": self.history,
                    "use_feature_ensemble": self.cfg.use_feature_ensemble,
                    "use_input_preprocessing": self.cfg.use_input_preprocessing,
                    "normalize_features": self.cfg.normalize_features,
                },
                f,
            )

    # =========================================================
    # Plot
    # =========================================================

    def plot_validation_summary(self, save_dir: str):
        if len(self.history["val"]["f1"]) == 0:
            return

        metrics = [
            ("acc", "Validation Accuracy"),
            ("f1", "Validation F1"),
            ("roc_auc", "Validation ROC AUC"),
        ]

        for key, title in metrics:
            plt.figure()
            plt.bar([0], [self.history["val"][key][-1]])
            plt.xticks([0], ["Mahalanobis"])
            plt.title(title)
            plt.grid(True)

            plt.savefig(os.path.join(save_dir, f"val_{key}.png"))
            plt.close()

    # =========================================================
    # PyTorch compatibility
    # =========================================================

    def state_dict(self):
        state = {}

        for layer, layer_stats in self.stats.items():
            state[f"{layer}_means"] = layer_stats["means"]
            state[f"{layer}_precision"] = layer_stats["precision"]

        return state

    def load_state_dict(self, state):
        self.stats = {}

        for key in state.keys():
            if key.endswith("_means"):
                layer = key.replace("_means", "")
                if layer not in self.stats:
                    self.stats[layer] = {}
                self.stats[layer]["means"] = state[key].to(self.device)

            elif key.endswith("_precision"):
                layer = key.replace("_precision", "")
                if layer not in self.stats:
                    self.stats[layer] = {}
                self.stats[layer]["precision"] = state[key].to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 2-class output expected by EvaluateDetector:
            column 0 = clean
            column 1 = adversarial

        Supports image tensors or feature tensors.
        """
        probs_adv = self.predict_proba(x)
        probs_clean = 1.0 - probs_adv
        logits = torch.stack([probs_clean, probs_adv], dim=1)
        return logits

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def to(self, device):
        self.device = device

        for layer in self.stats:
            self.stats[layer]["means"] = self.stats[layer]["means"].to(device)
            self.stats[layer]["precision"] = self.stats[layer]["precision"].to(device)

        return self
