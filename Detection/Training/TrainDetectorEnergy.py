"""
TrainDetector.py
----------------

Revised for better stability on Tiny-ImageNet:
  - removed BatchNorm1d from detector head
  - whitening made less aggressive
  - slightly larger, stable MLP
  - longer default training
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from Utils import amp_context


@dataclass
class EnergyTrainConfig:
    device: torch.device

    epochs: int = 30
    lr: float = 2e-4
    monitor: str = "roc_auc"

    weight_decay: float = 1e-4

    early_stopping: bool = True
    patience: int = 6
    min_delta: float = 1e-4

    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    temperature: float = 0.7

    grad_clip_norm: Optional[float] = 5.0
    label_smoothing: float = 0.0

    use_scheduler: bool = False
    sched_type: str = "cosine"
    sched_step_size: int = 10
    sched_gamma: float = 0.5
    sched_tmax: int = 50

    optimize_threshold: bool = True
    margin: float = 1.0
    margin_weight: float = 1e-3


class EnergyDetector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        # BatchNorm removed: it was hurting already-collapsed feature distributions
        self.feature_norm = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.feature_norm(x)
        energy = self.net(x).squeeze(1)
        return energy


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, temperature=1.0):
        super().__init__()
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits / self.temperature)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        return loss.mean()


class EnergyTrainDetector:
    def __init__(
        self,
        feature_extractor,
        model,
        tokenizer,
        class_names,
        device: torch.device,
        cfg: EnergyTrainConfig,
    ):
        self.fe = feature_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.cfg = cfg

        self.detector = EnergyDetector(self.fe.feature_dim).to(self.device)

        self.opt = torch.optim.Adam(
            self.detector.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.sched = None
        if self.cfg.use_scheduler:
            if self.cfg.sched_type == "cosine":
                self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt,
                    T_max=self.cfg.sched_tmax,
                )
            elif self.cfg.sched_type == "step":
                self.sched = torch.optim.lr_scheduler.StepLR(
                    self.opt,
                    step_size=self.cfg.sched_step_size,
                    gamma=self.cfg.sched_gamma,
                )

        self.criterion = (
            FocalLoss(self.cfg.focal_gamma, self.cfg.temperature)
            if self.cfg.use_focal_loss
            else nn.BCEWithLogitsLoss()
        )

        self.clean_mean = None
        self.clean_cov_inv_sqrt = None

        self.history = {
            "train": {"loss": [], "acc": [], "f1": [], "roc_auc": []},
            "val": {"loss": [], "acc": [], "f1": [], "roc_auc": []},
        }

    @torch.no_grad()
    def _feature_loader(self, loader):
        from torch.utils.data import DataLoader, TensorDataset
        features, labels = [], []
        for images, target in loader:
            with amp_context(self.device):
                features.append(self.fe.phi(images.to(self.device)).float().cpu())
            labels.append(target.cpu())
        return DataLoader(TensorDataset(torch.cat(features), torch.cat(labels)),
                          batch_size=loader.batch_size, shuffle=False)

    def _fit_whitening(self, train_loader):
        print("📐 Estimating clean feature whitening statistics...")

        X = train_loader.dataset.tensors[0].to(self.device)

        self.clean_mean = X.mean(dim=0, keepdim=True)
        var = X.var(dim=0, unbiased=False)

        # Less aggressive whitening for collapsed / low-variance dimensions
        self.clean_cov_inv_sqrt = 1.0 / torch.sqrt(var + 1e-3)

        self.clean_mean = self.clean_mean.to(self.device).float()
        self.clean_cov_inv_sqrt = self.clean_cov_inv_sqrt.to(self.device).float()

        if torch.isnan(self.clean_mean).any():
            raise RuntimeError("NaN detected in clean_mean during whitening fit.")
        if torch.isnan(self.clean_cov_inv_sqrt).any():
            raise RuntimeError("NaN detected in clean_cov_inv_sqrt during whitening fit.")

        print("✅ Whitening stats computed (diagonal)")

    def _whiten(self, x):
        x = x.float()

        if self.clean_mean is None or self.clean_cov_inv_sqrt is None:
            return x

        out = (x - self.clean_mean) * self.clean_cov_inv_sqrt

        if torch.isnan(out).any():
            print("⚠️ NaN after whitening")
            print("x min/max:", x.min().item(), x.max().item())
            print("mean min/max:", self.clean_mean.min().item(), self.clean_mean.max().item())
            print(
                "inv_sqrt min/max:",
                self.clean_cov_inv_sqrt.min().item(),
                self.clean_cov_inv_sqrt.max().item(),
            )
            raise RuntimeError("NaN inside whitened features.")

        return out

    def _step(self, images, adv_images, train=True):
        images = images.to(self.device)
        adv_images = adv_images.to(self.device)

        if torch.isnan(images).any():
            raise RuntimeError("NaN inside clean images")

        if torch.isnan(adv_images).any():
            raise RuntimeError("NaN inside adversarial images")

        phi_clean_raw = images
        phi_adv_raw = adv_images

        if torch.isnan(phi_clean_raw).any():
            raise RuntimeError("NaN in clean features")

        if torch.isnan(phi_adv_raw).any():
            raise RuntimeError("NaN in adversarial features")

        phi_clean = self._whiten(phi_clean_raw)
        phi_adv = self._whiten(phi_adv_raw)

        X = torch.cat([phi_clean, phi_adv], dim=0)

        if torch.isnan(X).any():
            raise RuntimeError("NaN detected in detector input.")

        logits = self.detector(X)

        if torch.isnan(logits).any():
            raise RuntimeError("NaN detected in detector logits.")

        y = torch.cat(
            [
                torch.zeros(images.size(0), device=self.device),
                torch.ones(images.size(0), device=self.device),
            ],
            dim=0,
        )

        base_loss = self.criterion(logits, y)

        N = images.size(0)
        energy_clean = logits[:N]
        energy_adv = logits[N:]

        margin_loss = torch.relu(
            self.cfg.margin - (energy_adv - energy_clean)
        ).mean()

        loss = base_loss + self.cfg.margin_weight * margin_loss

        if torch.isnan(loss):
            raise RuntimeError("NaN detected in training loss.")

        if train:
            self.opt.zero_grad()
            loss.backward()

            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.detector.parameters(),
                    self.cfg.grad_clip_norm,
                )

            self.opt.step()

        probs = torch.sigmoid(logits / self.cfg.temperature).detach().cpu()
        preds = (probs > 0.5).float()
        y_cpu = y.detach().cpu()

        acc = (preds == y_cpu).float().mean().item()
        f1 = f1_score(y_cpu, preds)

        try:
            roc = roc_auc_score(y_cpu, probs)
        except Exception:
            roc = 0.5

        return loss.item(), acc, f1, roc

    def fit(
        self,
        train_clean_loader,
        val_clean_loader,
        train_adv_loader,
        val_adv_loader,
        save_dir: str,
    ):
        if train_adv_loader is None:
            raise ValueError("train_adv_loader is required.")
        if val_clean_loader is None:
            raise ValueError("val_loader is required.")
        if val_adv_loader is None:
            raise ValueError("val_adv_loader is required.")

        # The encoder and all transformations are fixed: extract once, train only the MLP.
        train_clean_loader = self._feature_loader(train_clean_loader)
        train_adv_loader = self._feature_loader(train_adv_loader)
        val_clean_loader = self._feature_loader(val_clean_loader)
        val_adv_loader = self._feature_loader(val_adv_loader)
        for clean, adv in [(train_clean_loader, train_adv_loader),
                           (val_clean_loader, val_adv_loader)]:
            if not torch.equal(clean.dataset.tensors[1], adv.dataset.tensors[1]):
                raise ValueError("Clean/adversarial labels are not paired in sample order")
        self._fit_whitening(train_clean_loader)

        # Keep every clean/adversarial pair together while changing pair order
        # each epoch.  This preserves the paired margin loss and improves SGD
        # compared with replaying one fixed sample order for every epoch.
        from Utils import SEED
        from torch.utils.data import DataLoader, TensorDataset
        train_pairs = TensorDataset(
            train_clean_loader.dataset.tensors[0],
            train_adv_loader.dataset.tensors[0],
        )
        shuffle_generator = torch.Generator().manual_seed(SEED)
        paired_train_loader = DataLoader(
            train_pairs,
            batch_size=train_clean_loader.batch_size,
            shuffle=True,
            generator=shuffle_generator,
        )

        best_score = -float("inf")
        best_state = None
        best_threshold = 0.0
        epochs_no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.detector.train()

            train_loss = 0.0
            train_acc = 0.0
            train_f1 = 0.0
            train_roc = 0.0
            n = 0

            for images, adv_images in paired_train_loader:
                loss, acc, f1, roc = self._step(images, adv_images, train=True)

                train_loss += loss
                train_acc += acc
                train_f1 += f1
                train_roc += roc
                n += 1

            train_loss /= max(n, 1)
            train_acc /= max(n, 1)
            train_f1 /= max(n, 1)
            train_roc /= max(n, 1)

            print(
                f"\n[Epoch {epoch}] TRAIN "
                f"loss={train_loss:.4f} "
                f"acc={train_acc:.4f} "
                f"f1={train_f1:.4f} "
                f"roc_auc={train_roc:.4f}"
            )

            self.history["train"]["loss"].append(train_loss)
            self.history["train"]["acc"].append(train_acc)
            self.history["train"]["f1"].append(train_f1)
            self.history["train"]["roc_auc"].append(train_roc)

            self.detector.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for (images, _), (adv_images, _) in zip(val_clean_loader, val_adv_loader):
                    images = images.to(self.device)
                    adv_images = adv_images.to(self.device)

                    if torch.isnan(adv_images).any():
                        raise RuntimeError("NaN detected in validation adversarial images.")

                    phi_clean_raw = images
                    phi_adv_raw = adv_images

                    if torch.isnan(phi_clean_raw).any():
                        raise RuntimeError("NaN in validation clean features.")
                    if torch.isnan(phi_adv_raw).any():
                        raise RuntimeError("NaN in validation adversarial features.")

                    phi_clean = self._whiten(phi_clean_raw)
                    phi_adv = self._whiten(phi_adv_raw)

                    X = torch.cat([phi_clean, phi_adv], dim=0)

                    if torch.isnan(X).any():
                        raise RuntimeError("NaN in validation detector input.")

                    y = torch.cat(
                        [
                            torch.zeros(images.size(0), device=self.device),
                            torch.ones(images.size(0), device=self.device),
                        ],
                        dim=0,
                    )

                    logits = self.detector(X)

                    if torch.isnan(logits).any():
                        raise RuntimeError("NaN in validation detector logits.")

                    base_loss = self.criterion(logits, y)

                    N = images.size(0)
                    energy_clean = logits[:N]
                    energy_adv = logits[N:]

                    margin_loss = torch.relu(
                        self.cfg.margin - (energy_adv - energy_clean)
                    ).mean()

                    loss = base_loss + self.cfg.margin_weight * margin_loss
                    val_loss += loss.item()

                    probs = logits.float()

                    all_probs.append(probs.cpu())
                    all_labels.append(y.cpu())

            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()

            if np.isnan(all_probs).any():
                raise RuntimeError("NaN detected in validation probabilities.")

            val_loss /= max(len(val_clean_loader), 1)

            precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
            best_idx = np.argmax(f1_scores)

            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0

            val_preds = (all_probs >= optimal_threshold).astype(int)

            val_acc = (val_preds == all_labels).mean()
            val_f1 = f1_score(all_labels, val_preds)

            try:
                val_roc = roc_auc_score(all_labels, all_probs)
            except Exception:
                val_roc = 0.5

            print(
                f"[Epoch {epoch}] VAL   "
                f"loss={val_loss:.4f} "
                f"acc={val_acc:.4f} "
                f"f1={val_f1:.4f} "
                f"roc_auc={val_roc:.4f} "
                f"thr={optimal_threshold:.4f}"
            )

            score = (
                val_f1
                if self.cfg.monitor == "f1"
                else val_roc if self.cfg.monitor == "roc_auc" else val_acc
            )

            self.history["val"]["loss"].append(val_loss)
            self.history["val"]["acc"].append(val_acc)
            self.history["val"]["f1"].append(val_f1)
            self.history["val"]["roc_auc"].append(val_roc)

            if score > best_score + self.cfg.min_delta:
                best_score = score
                best_state = copy.deepcopy(self.detector.state_dict())
                best_threshold = float(optimal_threshold)
                epochs_no_improve = 0
                print(f"✅ New best model saved (score={score:.4f})")
            else:
                epochs_no_improve += 1
                print(f"⏳ No improvement ({epochs_no_improve}/{self.cfg.patience})")

            if self.sched is not None:
                self.sched.step()

            if self.cfg.early_stopping and epochs_no_improve >= self.cfg.patience:
                print(f"⛔ Early stopping triggered at epoch {epoch}")
                break

        if best_state is not None:
            self.detector.load_state_dict(best_state)
            print("🔒 Loaded best validation model")

        print("📊 Plotting training curves...")
        self.plot_training_curves(save_dir=save_dir)

        return best_threshold

    def plot_training_curves(self, save_dir: str):
        if len(self.history["train"]["loss"]) == 0:
            return

        metrics = [
            ("loss", "Loss"),
            ("acc", "Accuracy"),
            ("f1", "F1"),
            ("roc_auc", "ROC AUC"),
        ]

        epochs = range(1, len(self.history["train"]["loss"]) + 1)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for key, title in metrics:
            plt.figure()
            plt.plot(epochs, self.history["train"][key], label="Train")
            plt.plot(epochs, self.history["val"][key], label="Val")
            plt.title(title)
            plt.legend()
            plt.grid(True)

            if save_dir:
                plt.savefig(f"{save_dir}/{key}.png")

            plt.close()
