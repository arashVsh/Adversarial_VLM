from dataclasses import dataclass
from typing import Any, List, Optional
from Utils import amp_context, denormalize_clip, normalize_clip

import torch
import torch.nn.functional as F



@dataclass
class FlowAttackConfig:
    eps: float = 0.03
    alpha: float = 0.0075
    steps: int = 10
    random_start: bool = True
    targeted: bool = False
    prompt_template: str = "a photo of a {}."
    smooth_weight: float = 0.05
    padding_mode: str = "border"


class FlowAttack:
    """
    Flow-based adversarial attack for CLIP-style models.

    Instead of adding pixel noise directly, this attack learns a small
    spatial flow field and warps the image using grid_sample.

    Unified API: generate(images, labels)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        class_names: List[str],
        device: torch.device,
        cfg: Optional[FlowAttackConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.cfg = cfg or FlowAttackConfig()

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

        tokens = self.tokenizer(prompts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)
        else:
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

        txt = self.model.encode_text(tokens)
        txt = F.normalize(txt, dim=-1).float()
        return txt

    def _clip_logits(self, img_feat: torch.Tensor) -> torch.Tensor:
        text_feat = self.text_features.to(img_feat.dtype)
        logits = img_feat @ text_feat.T

        if hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale.exp().to(logits.dtype)

        return logits

    def _make_base_grid(self, batch_size: int, h: int, w: int, dtype: torch.dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=self.device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=self.device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack((xs, ys), dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]
        return base_grid

    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, C, H, W]
        flow: [B, H, W, 2] in normalized grid coordinates
        """
        b, _, h, w = x.shape
        base_grid = self._make_base_grid(b, h, w, x.dtype)
        grid = base_grid + flow
        grid = torch.clamp(grid, -1.0, 1.0)

        x_warp = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.cfg.padding_mode,
            align_corners=True,
        )
        return x_warp

    def _flow_smoothness_loss(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Total variation-style smoothness penalty on the flow field.
        flow: [B, H, W, 2]
        """
        dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        dy = flow[:, 1:, :, :] - flow[:, :-1, :, :]
        return dx.abs().mean() + dy.abs().mean()

    def generate(self, images, labels):
        cfg = self.cfg

        x_orig = denormalize_clip(images.to(self.device))
        y = labels.to(self.device)

        b, _, h, w = x_orig.shape

        if cfg.random_start:
            flow = torch.empty(
                b, h, w, 2, device=self.device, dtype=x_orig.dtype
            ).uniform_(-cfg.eps, cfg.eps)
        else:
            flow = torch.zeros(
                b, h, w, 2, device=self.device, dtype=x_orig.dtype
            )

        flow = flow.detach()

        for _ in range(cfg.steps):
            flow.requires_grad_(True)

            x_adv = self._warp(x_orig, flow)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            with amp_context(self.device):
                img_feat = self.model.encode_image(normalize_clip(x_adv))

            img_feat = F.normalize(img_feat, dim=-1).float()
            logits = self._clip_logits(img_feat)

            ce_loss = F.cross_entropy(logits, y)
            smooth_loss = self._flow_smoothness_loss(flow)

            if cfg.targeted:
                loss = -ce_loss + cfg.smooth_weight * smooth_loss
            else:
                loss = ce_loss - cfg.smooth_weight * smooth_loss

            grad = torch.autograd.grad(loss, flow)[0]

            with torch.no_grad():
                flow = flow + cfg.alpha * grad.sign()
                flow = torch.clamp(flow, -cfg.eps, cfg.eps)

            flow = flow.detach()

        x_adv = self._warp(x_orig, flow)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return normalize_clip(x_adv).detach()
