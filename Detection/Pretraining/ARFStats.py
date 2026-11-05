"""Clean-reference statistics for category-wise residual features."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================================================
# Artifacts container
# =========================================================

@dataclass
class ARFArtifacts:
    mu: Dict[str, torch.Tensor]
    cov: Dict[str, torch.Tensor]
    inv_cov: Dict[str, torch.Tensor]
    counts: Dict[str, int]
    lambda_reg: float
    shrinkage_alpha: float


# =========================================================
# ARFStats
# =========================================================

class ARFStats:
    def __init__(
        self,
        model: Any,
        anchor_bank: Any,
        device: torch.device,
        lambda_reg: float = 1e-3,
        shrinkage_alpha: float = 0.20,
        eps: float = 1e-12,
    ):
        self.model = model
        self.anchors = anchor_bank
        self.device = device
        self.lambda_reg = float(lambda_reg)
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.eps = float(eps)

        self.category_names: List[str] = list(self.anchors.category_names)
        self.category_to_indices = self.anchors.category_to_indices

        zT = self.anchors.get_prompt_embeddings()
        self.zT = zT.to(self.device, dtype=torch.float32)

        self.mu: Dict[str, torch.Tensor] = {}
        self.cov: Dict[str, torch.Tensor] = {}
        self.inv_cov: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # =========================================================
    # Fit ARF statistics
    # =========================================================

    @torch.no_grad()
    def fit(
        self,
        train_loader: DataLoader,
        max_batches: Optional[int] = None,
        verbose: bool = True,
    ) -> ARFArtifacts:

        sum_vec: Dict[str, torch.Tensor] = {}
        sum_outer: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}

        for cat in self.category_names:
            d_k = len(self.category_to_indices[cat])
            sum_vec[cat] = torch.zeros((d_k,), device=self.device, dtype=torch.float64)
            sum_outer[cat] = torch.zeros((d_k, d_k), device=self.device, dtype=torch.float64)
            counts[cat] = 0

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            images = batch[0].to(self.device)

            zI = self._encode_image(images)
            # Equation (9) uses the raw cosine-similarity subvector s_k(x).
            s = zI @ self.zT.T

            for cat in self.category_names:
                idx = self.category_to_indices[cat]
                v = s[:, idx].double()

                sum_vec[cat] += v.sum(dim=0)
                sum_outer[cat] += v.T @ v
                counts[cat] += v.shape[0]

            if verbose and (b_idx + 1) % 50 == 0:
                print(f"[ARFStats] processed batches: {b_idx + 1}")

        # ===== Compute μ, Σ =====
        for cat in self.category_names:
            n = max(counts[cat], 1)
            mu = sum_vec[cat] / float(n)
            exx = sum_outer[cat] / float(n)
            cov = exx - torch.outer(mu, mu)
            cov = 0.5 * (cov + cov.T)
            diagonal = torch.diag(cov).clamp_min(0.0)
            cov = cov - torch.diag(torch.diag(cov)) + torch.diag(diagonal)

            # ===== Shrinkage =====
            alpha = self.shrinkage_alpha
            if alpha > 0:
                diag = torch.diag(torch.diag(cov))
                cov = (1 - alpha) * cov + alpha * diag

            # ===== Regularization =====
            d_k = cov.shape[0]
            cov = cov + self.lambda_reg * torch.eye(d_k, device=self.device)

            inv_cov = torch.linalg.pinv(cov)

            self.mu[cat] = mu.float().detach()
            self.cov[cat] = cov.float().detach()
            self.inv_cov[cat] = inv_cov.float().detach()
            self.counts[cat] = counts[cat]

        return ARFArtifacts(
            mu=self.mu,
            cov=self.cov,
            inv_cov=self.inv_cov,
            counts=self.counts,
            lambda_reg=self.lambda_reg,
            shrinkage_alpha=self.shrinkage_alpha,
        )

    # =========================================================
    # Graph-Conditioned Mahalanobis
    # =========================================================

    def mahalanobis(self, category: str, vec: torch.Tensor) -> torch.Tensor:
        if category not in self.mu:
            return torch.tensor(0.0, device=self.device)

        mu = self.mu[category]
        inv_cov = self.inv_cov[category]
        v = vec.to(self.device, dtype=torch.float32)

        diff = v - mu
        base = (diff.unsqueeze(0) @ inv_cov @ diff.unsqueeze(1)).squeeze()

        return base

    # =========================================================
    # Utilities
    # =========================================================

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    # =========================================================
    # Save / Load
    # =========================================================
    def state_dict(self):

        return {
            "mu": {k: v.detach().cpu() for k, v in self.mu.items()},
            "cov": {k: v.detach().cpu() for k, v in self.cov.items()},
            "inv_cov": {k: v.detach().cpu() for k, v in self.inv_cov.items()},
            "counts": self.counts,

            "lambda_reg": self.lambda_reg,
            "shrinkage_alpha": self.shrinkage_alpha,
        }


    def load_state_dict(self, state):

        self.mu = {k: v.to(self.device) for k, v in state["mu"].items()}
        self.cov = {k: v.to(self.device) for k, v in state["cov"].items()}
        self.inv_cov = {k: v.to(self.device) for k, v in state["inv_cov"].items()}
        self.counts = state.get("counts", {})

        self.lambda_reg = state["lambda_reg"]
        self.shrinkage_alpha = state["shrinkage_alpha"]

        
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        meta = {
            "counts": self.counts,
            "lambda_reg": self.lambda_reg,
            "shrinkage_alpha": self.shrinkage_alpha,
        }

        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        torch.save({k: v.cpu() for k, v in self.mu.items()}, os.path.join(folder, "mu.pt"))
        torch.save({k: v.cpu() for k, v in self.cov.items()}, os.path.join(folder, "cov.pt"))
        torch.save({k: v.cpu() for k, v in self.inv_cov.items()}, os.path.join(folder, "inv_cov.pt"))
