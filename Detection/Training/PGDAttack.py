from dataclasses import dataclass
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
from Utils import amp_context, denormalize_clip, normalize_clip


@dataclass
class PGDConfig:
    eps: float = 0.03
    alpha: float = 0.0075
    steps: int = 10
    random_start: bool = True
    targeted: bool = False
    prompt_template: str = "a photo of a {}."


class PGDAttack:
    """
    PGD attack for CLIP-style models.
    Unified API: generate(images, labels)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        class_names: List[str],
        device: torch.device,
        cfg: Optional[PGDConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.cfg = cfg or PGDConfig()

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.text_features = self._build_text_features().to(device)

    @torch.no_grad()
    def _build_text_features(self):
        prompts = [
            self.cfg.prompt_template.format(c.replace("_", " "))
            for c in self.class_names
        ]
        tokens = self.tokenizer(prompts).to(self.device)

        txt = self.model.encode_text(tokens)
        txt = F.normalize(txt, dim=-1)

        return txt

    def _clip_logits(self, img_feat):
        logits = img_feat @ self.text_features.T

        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp()

        return logits

    def generate(self, images, labels):

        cfg = self.cfg

        x_orig = denormalize_clip(images.to(self.device))
        y = labels.to(self.device)

        if cfg.random_start:
            x_adv = (
                x_orig + torch.empty_like(x_orig).uniform_(-cfg.eps, cfg.eps)
            ).clamp(0.0, 1.0)
        else:
            x_adv = x_orig.clone()

        x_adv = x_adv.detach()

        for _ in range(cfg.steps):

            x_adv.requires_grad_(True)

            with amp_context(self.device):
                img_feat = self.model.encode_image(normalize_clip(x_adv))

            img_feat = F.normalize(img_feat, dim=-1).float()
            logits = self._clip_logits(img_feat)

            loss = F.cross_entropy(logits, y)

            if cfg.targeted:
                loss = -loss

            grad = torch.autograd.grad(loss, x_adv)[0]

            with torch.no_grad():

                x_adv = x_adv + cfg.alpha * grad.sign()

                delta = torch.clamp(x_adv - x_orig, -cfg.eps, cfg.eps)

                x_adv = x_orig + delta

                x_adv.clamp_(0.0, 1.0)

            x_adv = x_adv.detach()

        return normalize_clip(x_adv).detach()
