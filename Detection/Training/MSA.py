"""Transformation-stability features from Equations (10) and (11)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MSAComputer:
    """Compute embedding stability S(x) and semantic drift Delta_A(x).

    Dataset inputs are CLIP-normalized. Transformations are applied in pixel
    space and converted back before encoding.
    """

    def __init__(
        self,
        device: torch.device,
        anchor_bank: Optional[Any] = None,
        prompt_graph: Optional[Any] = None,
        shift_px: Tuple[int, int] = (1, 1),
        blur_kernel_size: int = 3,
        blur_sigma: float = 0.7,
        contrast_scale: float = 0.99,
        eps: float = 1e-12,
    ):
        del prompt_graph  # Compatibility with older experiment configurations.
        self.device = device
        self.anchors = anchor_bank
        self.shift_px = shift_px
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.contrast_scale = contrast_scale
        self.eps = eps

        self._mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        self._gauss_kernel_2d = self._build_gaussian_kernel_2d(
            blur_kernel_size, blur_sigma, device
        )
        self.zT = None
        if anchor_bank is not None:
            self.zT = anchor_bank.get_prompt_embeddings().to(
                device=device, dtype=torch.float32
            )

    def compute(self, images: torch.Tensor, zI: torch.Tensor, model: Any):
        if self.zT is None:
            raise RuntimeError("MSAComputer requires an anchor_bank.")

        images = images.to(self.device)
        zI = zI.to(self.device, dtype=torch.float32)
        transformed = (
            self._perturb_shift(images),
            self._perturb_blur(images),
            self._perturb_contrast(images),
        )
        # One encoder call is considerably faster than three sequential calls
        # and is mathematically equivalent for an evaluation-mode encoder.
        batch_size = images.shape[0]
        encoded = self._encode_image(model, torch.cat(transformed, dim=0))
        embeddings = encoded.split(batch_size, dim=0)

        cosine_values = [(zI * z_a).sum(dim=1) for z_a in embeddings]
        stability = torch.stack(cosine_values, dim=0).mean(dim=0)

        s = zI @ self.zT.T
        semantic_drifts = [
            torch.norm(s - (z_a @ self.zT.T), p=2, dim=1) for z_a in embeddings
        ]
        drift = torch.stack(semantic_drifts, dim=0).mean(dim=0)
        return stability, drift

    def _to_pixel(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(dtype=x.dtype)
        std = self._std.to(dtype=x.dtype)
        return (x * std + mean).clamp(0.0, 1.0)

    def _to_normalized(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(dtype=x.dtype)
        std = self._std.to(dtype=x.dtype)
        return (x - mean) / std

    def _perturb_shift(self, x: torch.Tensor) -> torch.Tensor:
        pixel = self._to_pixel(x)
        dy, dx = self.shift_px
        shifted = torch.roll(pixel, shifts=(dy, dx), dims=(2, 3))
        return self._to_normalized(shifted)

    def _perturb_blur(self, x: torch.Tensor) -> torch.Tensor:
        pixel = self._to_pixel(x)
        channels = pixel.shape[1]
        kernel = self._gauss_kernel_2d.to(pixel.dtype).repeat(channels, 1, 1, 1)
        blurred = F.conv2d(
            pixel,
            kernel,
            padding=self.blur_kernel_size // 2,
            groups=channels,
        ).clamp(0.0, 1.0)
        return self._to_normalized(blurred)

    def _perturb_contrast(self, x: torch.Tensor) -> torch.Tensor:
        pixel = self._to_pixel(x)
        channel_mean = pixel.mean(dim=(2, 3), keepdim=True)
        adjusted = (
            channel_mean + self.contrast_scale * (pixel - channel_mean)
        ).clamp(0.0, 1.0)
        return self._to_normalized(adjusted)

    @staticmethod
    def _encode_image(model: Any, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(model.encode_image(images).float(), dim=-1)

    @staticmethod
    def _build_gaussian_kernel_2d(k: int, sigma: float, device: torch.device):
        ax = torch.arange(k, device=device) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, k, k)
