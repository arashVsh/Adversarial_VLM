"""Prompt graph used by the response-profile detector.

The graph follows the paper: each prompt is connected to its top-r
neighbours in frozen text-embedding space and the resulting adjacency is
symmetrized. No dataset labels or category-dependent edge weights are used.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn.functional as F


LaplacianType = Literal["unnormalized", "sym_norm", "rw_norm"]
WeightType = Literal["binary", "cosine", "temperature"]


@dataclass
class PromptGraphArtifacts:
    adjacency: torch.Tensor
    degree: torch.Tensor
    laplacian: torch.Tensor
    knn_indices: torch.Tensor
    laplacian_type: str
    weight_type: str
    k: int
    random_walk: torch.Tensor  # NEW: D^{-1} A


class PromptGraph:
    def __init__(
        self,
        anchor_bank,
        device: torch.device,
        k: int = 8,
        laplacian_type: LaplacianType = "unnormalized",
        weight_type: WeightType = "cosine",
        include_self: bool = False,
        tau: float = 0.07,
    ):
        self.anchor_bank = anchor_bank
        self.k = int(k)
        self.laplacian_type = laplacian_type
        self.weight_type = weight_type
        self.include_self = include_self
        self.device = device

        self.tau = float(tau)

        self._artifacts: Optional[PromptGraphArtifacts] = None

    # =========================================================
    # Public API
    # =========================================================

    def build(self) -> PromptGraphArtifacts:
        zT = self.anchor_bank.get_prompt_embeddings().to(self.device)
        zT = F.normalize(zT, dim=-1)

        M = zT.shape[0]
        sim = zT @ zT.T

        if not self.include_self:
            sim.fill_diagonal_(-1e9)

        knn_vals, knn_idx = torch.topk(sim, k=self.k, dim=1)

        A = torch.zeros((M, M), device=self.device)

        # ================= weight computation =================
        if self.weight_type == "binary":
            A.scatter_(1, knn_idx, 1.0)

        elif self.weight_type == "cosine":
            weights = torch.clamp(knn_vals, min=0.0)
            A.scatter_(1, knn_idx, weights)

        elif self.weight_type == "temperature":
            # temperature-scaled cosine
            weights = torch.exp((knn_vals - 1.0) / self.tau)
            A.scatter_(1, knn_idx, weights)

        else:
            raise ValueError(f"Unsupported weight_type: {self.weight_type}")

        # Symmetric union of directed top-r neighbourhoods.
        A = torch.maximum(A, A.T)

        if self.include_self:
            A.fill_diagonal_(1.0)

        # degree & random walk
        deg = torch.sum(A, dim=1)
        D = torch.diag(deg)
        P = A / (deg.unsqueeze(1) + 1e-12)

        L = self._compute_laplacian(A, deg)

        self._artifacts = PromptGraphArtifacts(
            adjacency=A.detach(),
            degree=D.detach(),
            laplacian=L.detach(),
            knn_indices=knn_idx.detach(),
            laplacian_type=self.laplacian_type,
            weight_type=self.weight_type,
            k=self.k,
            random_walk=P.detach(),
        )
        return self._artifacts

    def artifacts(self) -> PromptGraphArtifacts:
        if self._artifacts is None:
            return self.build()
        return self._artifacts

    def laplacian(self) -> torch.Tensor:
        return self.artifacts().laplacian

    # =========================================================
    # Laplacian
    # =========================================================

    def _compute_laplacian(self, A: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        M = A.shape[0]

        if self.laplacian_type == "unnormalized":
            return torch.diag(deg) - A

        if self.laplacian_type == "sym_norm":
            inv_sqrt = torch.rsqrt(torch.clamp(deg, min=1e-12))
            D_inv_sqrt = torch.diag(inv_sqrt)
            I = torch.eye(M, device=A.device)
            return I - (D_inv_sqrt @ A @ D_inv_sqrt)

        if self.laplacian_type == "rw_norm":
            inv = 1.0 / torch.clamp(deg, min=1e-12)
            D_inv = torch.diag(inv)
            I = torch.eye(M, device=A.device)
            return I - (D_inv @ A)

        raise ValueError(f"Unsupported laplacian_type: {self.laplacian_type}")
