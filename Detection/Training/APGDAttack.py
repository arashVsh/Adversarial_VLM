"""Auto-PGD (cross-entropy loss) for CLIP-style zero-shot classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from Detection.Training.AutoAttackCLIP import (
    CLIPClassifier,
    CLIP_MEAN,
    CLIP_STD,
    denormalize_to_pixel,
    normalize_to_clip,
)


@dataclass
class APGDConfig:
    eps: float = 8 / 255
    norm: str = "Linf"
    steps: int = 100
    restarts: int = 1
    seed: int = 42
    loss: str = "ce"
    eot_iter: int = 1
    rho: float = 0.75


class APGDAttack:
    """Untargeted APGD-CE operating in pixel space."""

    cache_semantics_version = 1

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        class_names: List[str],
        device: torch.device,
        cfg: Optional[APGDConfig] = None,
    ):
        from autoattack.autopgd_base import APGDAttack as AutoAttackAPGD

        self.device = device
        self.cfg = cfg or APGDConfig()
        self.classifier = CLIPClassifier(
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
        ).to(device)
        self.adversary = AutoAttackAPGD(
            predict=self.classifier,
            n_iter=self.cfg.steps,
            norm=self.cfg.norm,
            n_restarts=self.cfg.restarts,
            eps=self.cfg.eps,
            seed=self.cfg.seed,
            loss=self.cfg.loss,
            eot_iter=self.cfg.eot_iter,
            rho=self.cfg.rho,
            verbose=False,
            device=device,
        )
        self.cache_identity = {
            "attack": "apgd-ce",
            "norm": self.cfg.norm,
            "eps": self.cfg.eps,
            "steps": self.cfg.steps,
            "restarts": self.cfg.restarts,
            "loss": self.cfg.loss,
        }

    def generate(self, images_norm, labels):
        images_norm = images_norm.to(self.device)
        labels = labels.to(self.device).long()
        pixels = denormalize_to_pixel(
            images_norm, CLIP_MEAN, CLIP_STD).detach()
        adversarial = self.adversary.perturb(pixels, labels)
        return normalize_to_clip(
            adversarial.detach(), CLIP_MEAN, CLIP_STD).detach()
