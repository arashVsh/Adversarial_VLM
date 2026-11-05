from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import amp_context
from Constants import Datasets


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _to_3x1x1(x: torch.Tensor):
    return x.view(3, 1, 1) if x.ndim == 1 else x


def denormalize_to_pixel(x_norm: torch.Tensor, mean, std):
    mean = _to_3x1x1(torch.tensor(mean, device=x_norm.device, dtype=x_norm.dtype))
    std = _to_3x1x1(torch.tensor(std, device=x_norm.device, dtype=x_norm.dtype))
    x = x_norm * std + mean
    return torch.clamp(x, 0.0, 1.0)


def normalize_to_clip(x_pixel: torch.Tensor, mean, std):
    mean = _to_3x1x1(torch.tensor(mean, device=x_pixel.device, dtype=x_pixel.dtype))
    std = _to_3x1x1(torch.tensor(std, device=x_pixel.device, dtype=x_pixel.dtype))
    return (x_pixel - mean) / std


# =========================================================
# ✅ FIXED CLIP CLASSIFIER (clean nn.Module)
# =========================================================
class CLIPClassifier(nn.Module):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        class_names: List[str],
        device: torch.device,
        prompt_template: str = "a photo of a {}.",
    ):
        super().__init__()

        self.clip = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.prompt_template = prompt_template

        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self.register_buffer(
            "text_features",
            self._build_text_features().to(next(self.clip.parameters()).dtype),
        )

    @torch.no_grad()
    def _build_text_features(self):
        prompts = [
            self.prompt_template.format(c.replace("_", " "))
            for c in self.class_names
        ]
        tokens = self.tokenizer(prompts)

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)
        else:
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

        txt = self.clip.encode_text(tokens)
        txt = F.normalize(txt, dim=-1)
        return txt

    def forward(self, x_pixel: torch.Tensor):
        x_norm = normalize_to_clip(x_pixel, CLIP_MEAN, CLIP_STD)

        with amp_context(self.device):
            img_feat = self.clip.encode_image(x_norm)
            img_feat = F.normalize(img_feat, dim=-1)

            logits = img_feat @ self.text_features.t()

            if hasattr(self.clip, "logit_scale"):
                logits = logits * self.clip.logit_scale.exp()

        return logits


# =========================================================
# Config
# =========================================================
@dataclass
class AutoAttackConfig:
    eps: float = 8 / 255
    norm: str = "Linf"
    version: str = "standard"
    seed: int = 42
    batch_size: int = 32


# =========================================================
# AutoAttack wrapper
# =========================================================
class AutoAttackCLIP:

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        class_names: List[str],
        device: torch.device,
        dataset_name: str,
        cfg: Optional[AutoAttackConfig] = None,
    ):
        self.device = device
        self.cfg = cfg or AutoAttackConfig()
        self.dataset_name = dataset_name

        from autoattack import AutoAttack

        self.classifier = CLIPClassifier(
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
        ).to(device)

        self.adversary = AutoAttack(
            self.classifier,
            norm=self.cfg.norm,
            eps=self.cfg.eps,
            version=self.cfg.version,
            seed=self.cfg.seed,
            device=device,
            verbose=False,
        )

        self._tiny_mode = dataset_name == Datasets.TINY_IMAGENET

    def _iterative_attack(self, x, y):
        eps = self.cfg.eps
        alpha = 2 / 255
        steps = 20

        x_orig = x.detach()
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)

        for _ in range(steps):
            x_adv.requires_grad_(True)

            logits = self.classifier(x_adv)
            loss = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                delta = torch.clamp(x_adv - x_orig, -eps, eps)
                x_adv = (x_orig + delta).clamp(0, 1)

        return x_adv.detach()

    def generate(self, images_norm, labels):

        x_norm = images_norm.to(self.device)
        y = labels.to(self.device).long()

        x_pixel = denormalize_to_pixel(x_norm, CLIP_MEAN, CLIP_STD).detach()

        if self._tiny_mode:
            x_adv_pixel = self._iterative_attack(x_pixel, y)
        else:
            x_adv_pixel = self.adversary.run_standard_evaluation(
                x_pixel,
                y,
                bs=self.cfg.batch_size,
            )

        x_adv_norm = normalize_to_clip(x_adv_pixel, CLIP_MEAN, CLIP_STD)
        return x_adv_norm.detach()
