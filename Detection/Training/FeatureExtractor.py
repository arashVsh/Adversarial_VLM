"""Construct the 43-dimensional response profile defined in the paper."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, List
import torch
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(
        self,
        model: Any,
        anchor_bank: Any,
        prompt_graph: Any,
        device: torch.device,
        arf_stats: Optional[Any] = None,
        msa_computer: Optional[Any] = None,
        eps: float = 1e-12,
        diffusion_alpha: float = 0.15,
        diffusion_steps: int = 3,
        lga_topk: int = 8,
    ):
        self.model = model
        self.anchors = anchor_bank
        self.graph = prompt_graph
        self.arf_stats = arf_stats
        self.msa = msa_computer

        self.device = device
        self.eps = float(eps)

        self.diffusion_alpha = float(diffusion_alpha)
        self.diffusion_steps = int(diffusion_steps)
        self.lga_topk = int(lga_topk)

        self.category_names: List[str] = list(self.anchors.category_names)
        self.K = len(self.category_names)

        zT = self.anchors.get_prompt_embeddings()
        self.zT = zT.to(self.device, dtype=torch.float32)
        self.M = self.zT.shape[0]

        if hasattr(self.graph, "artifacts") and callable(getattr(self.graph, "artifacts")):
            art = self.graph.artifacts()
            self.A = art.adjacency.to(self.device, dtype=torch.float32)

            if art.degree.ndim == 2:
                deg_vec = torch.diag(art.degree)
            else:
                deg_vec = art.degree

            self.deg = deg_vec.to(self.device, dtype=torch.float32)
            self.L = art.laplacian.to(self.device, dtype=torch.float32)
        else:
            L = self.graph.laplacian()
            self.L = L.to(self.device, dtype=torch.float32)
            self.A = None
            self.deg = None

        self.category_to_indices = self.anchors.category_to_indices

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def phi(
        self,
        images: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = images.to(self.device)

        # 1) Image embedding
        raw_embedding = self.model.encode_image(images).float()
        zI = F.normalize(raw_embedding, dim=-1)

        # 2) Prompt similarities
        s_raw = self._prompt_scores(zI)  # [B, M]

        # Category statistics use the globally normalized response s-hat.
        s_hat = F.normalize(s_raw, p=2, dim=1)

        # 4) Category stats
        masses, means, stds, maxs = self._category_stats(s_hat)
        entropy, margin, masses_norm = self._coherence(masses)

        # 5) Global graph signal
        gse = self._gse_energy(s_raw)

        # 6) Localized graph anomaly
        lga_max, lga_mean, lga_var, lga_topkmean, a = self._lga_features(s_raw)

        # 7) Diffusion + graph entropy
        sT, diff_delta_l2, diff_cos = self._diffusion_features(s_raw)
        graph_entropy = self._graph_entropy_from_scores(sT)

        # 8) ARF
        R1, R2, top2_idx = self._arf_residuals(s_raw, masses)

        # 9) MSA
        S, D = self._msa_metrics(images, zI)

        # 10) Extra robust scalar feature
        f_norm = torch.norm(raw_embedding, p=2, dim=1)

        phi_vec = torch.cat(
            [
                masses,
                means,
                stds,
                maxs,
                entropy.unsqueeze(1),
                margin.unsqueeze(1),
                gse.unsqueeze(1),
                lga_max.unsqueeze(1),
                lga_mean.unsqueeze(1),
                lga_var.unsqueeze(1),
                lga_topkmean.unsqueeze(1),
                diff_delta_l2.unsqueeze(1),
                diff_cos.unsqueeze(1),
                graph_entropy.unsqueeze(1),
                R1.unsqueeze(1),
                R2.unsqueeze(1),
                S.unsqueeze(1),
                D.unsqueeze(1),
                f_norm.unsqueeze(1),
            ],
            dim=1,
        )

        if not return_debug:
            return phi_vec

        debug = {
            "zI": zI,
            "s_raw": s_raw,
            "s": s_hat,
            "masses": masses,
            "masses_norm": masses_norm,
            "entropy": entropy,
            "margin": margin,
            "gse": gse,
            "a_lga": a,
            "lga_max": lga_max,
            "lga_mean": lga_mean,
            "lga_var": lga_var,
            "lga_topkmean": lga_topkmean,
            "sT": sT,
            "diff_delta_l2": diff_delta_l2,
            "diff_cos": diff_cos,
            "graph_entropy": graph_entropy,
            "R1": R1,
            "R2": R2,
            "top2_idx": top2_idx,
            "S": S,
            "D": D,
            "f_norm": f_norm,
        }
        return phi_vec, debug

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    def _prompt_scores(self, zI: torch.Tensor) -> torch.Tensor:
        return zI @ self.zT.T

    def _category_stats(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = s.shape[0]
        masses = torch.zeros((B, self.K), device=self.device)
        means = torch.zeros((B, self.K), device=self.device)
        stds = torch.zeros((B, self.K), device=self.device)
        maxs = torch.zeros((B, self.K), device=self.device)

        for k_idx, cat in enumerate(self.category_names):
            idx = self.category_to_indices[cat]
            sub = s[:, idx]

            masses[:, k_idx] = F.relu(sub).sum(dim=1)
            means[:, k_idx] = sub.mean(dim=1)
            stds[:, k_idx] = sub.std(dim=1, unbiased=False)
            maxs[:, k_idx] = sub.max(dim=1).values

        return masses, means, stds, maxs

    def _coherence(
        self, masses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = masses.sum(dim=1, keepdim=True) + self.eps
        masses_norm = masses / denom
        entropy = -(masses_norm * torch.log(masses_norm + self.eps)).sum(dim=1)

        top2 = torch.topk(
            masses_norm,
            k=2,
            dim=1,
            largest=True,
            sorted=True,
        ).values
        margin = top2[:, 0] - top2[:, 1]

        return entropy, margin, masses_norm

    def _gse_energy(self, s: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", s, self.L, s)

    def _lga_features(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, M = s.shape

        if self.A is None or self.deg is None:
            zeros = torch.zeros((B,), device=self.device, dtype=torch.float32)
            return zeros, zeros, zeros, zeros, None

        A = self.A
        deg = torch.clamp(self.deg, min=self.eps)

        s2 = s * s
        As = s @ A
        As2 = s2 @ A

        a = s2 * deg.unsqueeze(0) - 2.0 * s * As + As2
        a = torch.clamp(a, min=0.0)

        lga_max = a.max(dim=1).values
        lga_mean = a.mean(dim=1)
        lga_var = a.var(dim=1, unbiased=False)

        k = max(1, min(self.lga_topk, M))
        topk_vals = torch.topk(a, k=k, dim=1, largest=True, sorted=False).values
        lga_topkmean = topk_vals.mean(dim=1)

        return lga_max, lga_mean, lga_var, lga_topkmean, a

    def _diffusion_features(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M = s.shape

        if self.A is None or self.deg is None or self.diffusion_steps <= 0:
            sT = s
            zeros = torch.zeros((B,), device=self.device, dtype=torch.float32)
            ones = torch.ones((B,), device=self.device, dtype=torch.float32)
            return sT, zeros, ones

        A = self.A
        deg = torch.clamp(self.deg, min=self.eps)
        alpha = float(self.diffusion_alpha)

        s_t = s
        for _ in range(self.diffusion_steps):
            neighbor_sum = s_t @ A
            neighbor_avg = neighbor_sum / deg.unsqueeze(0)
            s_t = (1.0 - alpha) * s_t + alpha * neighbor_avg

        sT = s_t

        diff = sT - s
        diff_delta_l2 = torch.norm(diff, p=2, dim=1)

        s_norm = torch.norm(s, p=2, dim=1)
        sT_norm = torch.norm(sT, p=2, dim=1)
        diff_cos = (s * sT).sum(dim=1) / (s_norm * sT_norm + self.eps)

        return sT, diff_delta_l2, diff_cos

    def _graph_entropy_from_scores(self, sT: torch.Tensor) -> torch.Tensor:
        massesT, _, _, _ = self._category_stats(F.normalize(sT, p=2, dim=1))
        denom = massesT.sum(dim=1, keepdim=True) + self.eps
        p = massesT / denom
        H = -(p * torch.log(p + self.eps)).sum(dim=1)
        return H

    def _arf_residuals(
        self,
        s: torch.Tensor,
        masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = s.shape[0]

        if self.arf_stats is None:
            zeros = torch.zeros((B,), device=self.device, dtype=torch.float32)
            top2_idx = torch.topk(masses, k=2, dim=1).indices
            return zeros, zeros, top2_idx

        top2_idx = torch.topk(masses, k=2, dim=1).indices
        # Evaluate samples category-wise.  The former sample loop called
        # Tensor.item() twice per image, forcing repeated GPU synchronizations.
        residuals = torch.zeros((B, 2), device=self.device, dtype=torch.float32)
        can_batch = hasattr(self.arf_stats, "mu") and hasattr(self.arf_stats, "inv_cov")
        if can_batch:
            for k, category in enumerate(self.category_names):
                rows, ranks = torch.where(top2_idx == k)
                if rows.numel() == 0:
                    continue
                indices = self.category_to_indices[category]
                vectors = s.index_select(0, rows)[:, indices].float()
                mu = self.arf_stats.mu[category].to(self.device, dtype=torch.float32)
                inverse = self.arf_stats.inv_cov[category].to(self.device, dtype=torch.float32)
                # phi() may be called inside CUDA autocast during frozen feature
                # extraction.  Mahalanobis products must remain float32 both for
                # numerical stability and for assignment into `residuals`.
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    delta = vectors.float() - mu.unsqueeze(0)
                    values = torch.einsum(
                        "bi,ij,bj->b", delta, inverse.float(), delta
                    ).float()
                residuals[rows, ranks] = values.to(
                    device=residuals.device, dtype=residuals.dtype
                )
        else:
            # Compatibility path for custom statistics providers.
            for b in range(B):
                for rank in range(2):
                    category = self.category_names[int(top2_idx[b, rank].item())]
                    vector = s[b, self.category_to_indices[category]]
                    residuals[b, rank] = self._arf_score(category, vector)

        return residuals[:, 0], residuals[:, 1], top2_idx

    def _arf_score(self, category: str, vec: torch.Tensor) -> torch.Tensor:
        if hasattr(self.arf_stats, "mahalanobis") and callable(getattr(self.arf_stats, "mahalanobis")):
            out = self.arf_stats.mahalanobis(category, vec)
            return out.to(self.device, dtype=torch.float32)

        if hasattr(self.arf_stats, "score") and callable(getattr(self.arf_stats, "score")):
            out = self.arf_stats.score(category, vec)
            return out.to(self.device, dtype=torch.float32)

        mu = None
        inv_cov = None

        if hasattr(self.arf_stats, "mu"):
            mu = self.arf_stats.mu[category]
        elif isinstance(self.arf_stats, dict) and "mu" in self.arf_stats:
            mu = self.arf_stats["mu"][category]

        if hasattr(self.arf_stats, "inv_cov"):
            inv_cov = self.arf_stats.inv_cov[category]
        elif isinstance(self.arf_stats, dict) and "inv_cov" in self.arf_stats:
            inv_cov = self.arf_stats["inv_cov"][category]

        if mu is None or inv_cov is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        mu = mu.to(self.device, dtype=torch.float32)
        inv_cov = inv_cov.to(self.device, dtype=torch.float32)
        diff = (vec.to(self.device, dtype=torch.float32) - mu).unsqueeze(0)
        val = diff @ inv_cov @ diff.transpose(0, 1)
        return val.squeeze()

    def _msa_metrics(
        self, images: torch.Tensor, zI: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]

        if self.msa is None:
            zeros = torch.zeros((B,), device=self.device, dtype=torch.float32)
            return zeros, zeros

        if hasattr(self.msa, "compute") and callable(getattr(self.msa, "compute")):
            out = self.msa.compute(images=images, zI=zI, model=self.model)
            if isinstance(out, tuple) and len(out) == 2:
                S, D = out
                return (
                    S.to(self.device, dtype=torch.float32),
                    D.to(self.device, dtype=torch.float32),
                )

        zeros = torch.zeros((B,), device=self.device, dtype=torch.float32)
        return zeros, zeros

    @property
    def feature_dim(self) -> int:
        """
        4K category stats -> 4K
        Scalars:
          entropy, margin, gse                         -> 3
          lga_max, lga_mean, lga_var, lga_topkmean     -> 4
          diff_delta_l2, diff_cos                      -> 2
          graph_entropy                                -> 1
          R1, R2, S, D                                 -> 4
          f_norm                                      -> 1

        Total scalars = 15
        => feature_dim = 4K + 15
        """
        return 4 * self.K + 15

    def _class_energy(self, s):
        B = s.shape[0]
        energies = torch.zeros((B, self.K), device=self.device)

        for k_idx, cat in enumerate(self.category_names):
            idx = self.category_to_indices[cat]
            sub_s = s[:, idx]
            sub_L = self.L[idx][:, idx]
            energies[:, k_idx] = torch.einsum("bi,ij,bj->b", sub_s, sub_L, sub_s)

        return energies
