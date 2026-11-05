#!/usr/bin/env python
"""
Create a real VLM similarity figure from one actual dataset image.

This script:
1. Loads a real image from CIFAR-10, STL-10, or Tiny-ImageNet.
2. Loads a CLIP-style VLM using OpenCLIP.
3. Builds your real semantic prompt bank from AnchorBank.
4. Computes real image-text similarities s(x) = Z_T z_I(x).
5. Creates a paper-style figure showing:
   - input image,
   - frozen image and text encoders,
   - semantic prompt set,
   - shared embedding-space similarity diagram,
   - top prompt similarities.

Run example:
    python make_real_vlm_similarity_figure.py ^
        --dataset cifar_10 ^
        --model ViT-B-16 ^
        --sample_index 0 ^
        --top_k 6 ^
        --out Images/real_vlm_similarity

Outputs:
    Images/real_vlm_similarity.png
    Images/real_vlm_similarity.pdf
"""

import os
import argparse
import math
from textwrap import shorten

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Rectangle,
)
from matplotlib.lines import Line2D

import open_clip

from Constants import Datasets, Models
from Utils import (get_device, set_global_seed, load_dataset_deterministic,
                   create_model_and_transforms_cached)
from Detection.Pretraining.Anchors import AnchorBank


# =========================================================
# Display helpers
# =========================================================

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


CATEGORY_COLORS = {
    "animal": "#5aa832",
    "vehicle": "#e5b300",
    "person": "#4b83c4",
    "food": "#d95550",
    "object": "#e67e22",
    "scene": "#2aa198",
    "plant": "#76b852",
}

CATEGORY_ICON_TEXT = {
    "animal": "🐾",
    "vehicle": "🚗",
    "person": "●",
    "food": "◧",
    "object": "□",
    "scene": "△",
    "plant": "♣",
}


def tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert normalized CLIP tensor [3,H,W] into RGB image [H,W,3].
    """
    x = x.detach().cpu().float()
    x = x * CLIP_STD + CLIP_MEAN
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def add_round_box(
    ax,
    xy,
    wh,
    edgecolor="#cccccc",
    facecolor="#ffffff",
    linewidth=1.4,
    radius=0.03,
    alpha=1.0,
):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(box)
    return box


def add_arrow(ax, start, end, color="#4f7cc9", linewidth=2.5, mutation_scale=18):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=0.9,
    )
    ax.add_patch(arr)
    return arr


def draw_encoder_box(ax, x, y, w, h, title, subtitle, edgecolor, facecolor):
    add_round_box(ax, (x, y), (w, h), edgecolor=edgecolor, facecolor=facecolor, linewidth=1.6)
    ax.text(x + w / 2, y + h * 0.76, "❄", ha="center", va="center", fontsize=28, color="#2673c9")
    ax.text(
        x + w / 2,
        y + h * 0.50,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#102060",
    )
    ax.text(
        x + w / 2,
        y + h * 0.23,
        subtitle,
        ha="center",
        va="center",
        fontsize=15,
        color="#202040",
    )


def draw_prompt_list(ax, prompts, cats, sims, x, y, w, h, title="Text prompt set"):
    add_round_box(ax, (x, y), (w, h), edgecolor="#b8d5a7", facecolor="#fbfffa", linewidth=1.5)
    ax.text(
        x + w / 2,
        y + h - 0.055,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1d3b1d",
    )

    n = len(prompts)
    item_h = (h - 0.11) / n
    for i, (p, c, s) in enumerate(zip(prompts, cats, sims)):
        yy = y + h - 0.12 - (i + 1) * item_h + 0.01
        add_round_box(
            ax,
            (x + 0.02, yy),
            (w - 0.04, item_h - 0.012),
            edgecolor="#d4e6ca",
            facecolor="#ffffff",
            linewidth=1.0,
            radius=0.018,
        )
        color = CATEGORY_COLORS.get(c, "#999999")
        icon = CATEGORY_ICON_TEXT.get(c, "•")

        circ = Circle((x + 0.052, yy + item_h * 0.47), 0.022, facecolor=color, edgecolor=color, alpha=0.18)
        ax.add_patch(circ)
        ax.text(x + 0.052, yy + item_h * 0.47, icon, ha="center", va="center", fontsize=11, color=color)

        clean_prompt = shorten(p, width=34, placeholder="...")
        ax.text(
            x + 0.088,
            yy + item_h * 0.47,
            clean_prompt,
            ha="left",
            va="center",
            fontsize=11,
            color="#111111",
        )


def draw_similarity_formula_box(ax, x, y, w, h):
    add_round_box(ax, (x, y), (w, h), edgecolor="#b7cbea", facecolor="#fbfdff", linewidth=1.4)
    ax.text(x + w / 2, y + h - 0.055, "Similarity computation", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#162b74")

    add_round_box(ax, (x + 0.025, y + h - 0.135), (w - 0.05, 0.06),
                  edgecolor="#c8d5ed", facecolor="#ffffff", linewidth=1.0, radius=0.012)

    ax.text(
        x + w / 2,
        y + h - 0.105,
        r"$s(x)=Z_T z_I(x)$",
        ha="center",
        va="center",
        fontsize=13,
        color="#162b74",
    )

    ax.text(
        x + 0.035,
        y + h - 0.205,
        r"$z_I(x)\in\mathbb{R}^{d}$ is the image embedding.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#1e2e5f",
    )
    ax.text(
        x + 0.035,
        y + h - 0.255,
        r"$Z_T\in\mathbb{R}^{M\times d}$ stacks prompt embeddings.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#1e2e5f",
    )

    ax.plot([x + 0.03, x + w - 0.03], [y + 0.095, y + 0.095], color="#7fb5ff", linestyle=":", linewidth=1.2)

    ax.text(x + 0.052, y + 0.052, "◎", ha="center", va="center", fontsize=22, color="#162b74")
    ax.text(
        x + 0.095,
        y + 0.052,
        "Higher cosine similarity\nindicates stronger alignment.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#162b74",
    )


def draw_embedding_space(ax, x, y, w, h, prompts, cats, sims):
    add_round_box(ax, (x, y), (w, h), edgecolor="#c8bde8", facecolor="#fdfcff", linewidth=1.5)

    ax.text(
        x + w / 2,
        y + h - 0.06,
        "Shared vision-language embedding space",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#162b74",
    )
    ax.text(
        x + w / 2,
        y + h - 0.105,
        r"$\mathbb{R}^{d}$",
        ha="center",
        va="center",
        fontsize=15,
        color="#162b74",
    )

    # Subtle background grid
    center = np.array([x + 0.28 * w, y + 0.43 * h])
    for r in np.linspace(0.05, 0.30, 5):
        circ = Circle(center, r * w, facecolor="none", edgecolor="#d9d4ee", linewidth=0.6, alpha=0.35)
        ax.add_patch(circ)

    for ang in np.linspace(-0.4, 0.55, 8):
        end = center + np.array([0.58 * w * math.cos(ang), 0.32 * h * math.sin(ang)])
        ax.plot([center[0], end[0]], [center[1], end[1]], color="#e2def1", linewidth=0.5, alpha=0.55)

    # Image embedding node
    ax.scatter([center[0]], [center[1]], s=180, c="#0b70d5", edgecolors="#073f89", linewidths=1.0, zorder=5)
    ax.text(center[0] - 0.06, center[1] + 0.07, "image\nembedding", ha="center", va="center", fontsize=10, color="#162b74")

    # Prompt nodes
    n = len(prompts)
    y_offsets = np.linspace(0.26, -0.23, n)
    x_prompt = x + 0.70 * w

    for i, (p, c, s) in enumerate(zip(prompts, cats, sims)):
        yy = y + 0.53 * h + y_offsets[i] * h
        color = CATEGORY_COLORS.get(c, "#777777")

        # dotted similarity line
        ax.plot(
            [center[0], x_prompt - 0.055],
            [center[1], yy],
            linestyle=":",
            linewidth=1.5,
            color="#9178d0",
            alpha=0.75,
        )

        # prompt cluster glow
        glow = Circle((x_prompt, yy), 0.045, facecolor=color, edgecolor="none", alpha=0.16)
        ax.add_patch(glow)

        # three small prompt tokens
        rng = np.random.default_rng(100 + i)
        for _ in range(3):
            dx, dy = rng.normal(scale=0.010, size=2)
            ax.scatter(
                [x_prompt + dx],
                [yy + dy],
                s=60,
                c=color,
                edgecolors="#555555",
                linewidths=0.4,
                zorder=4,
            )

        # similarity value
        mx = (center[0] + x_prompt) / 2
        my = (center[1] + yy) / 2 + 0.018
        ax.text(
            mx,
            my,
            rf"$\cos={s:.2f}$",
            ha="center",
            va="center",
            fontsize=9,
            color="#162b74",
            fontweight="bold",
        )

        # prompt text
        prompt_short = shorten(p, width=24, placeholder="...")
        ax.text(
            x_prompt + 0.07,
            yy,
            f"“{prompt_short}”",
            ha="left",
            va="center",
            fontsize=9.5,
            color="#111111",
        )


def draw_top_similarity_bars(ax, prompts, cats, sims, x, y, w, h):
    add_round_box(ax, (x, y), (w, h), edgecolor="#b8d5a7", facecolor="#fbfffa", linewidth=1.4)

    ax.text(
        x + w / 2,
        y + h - 0.055,
        r"Top prompt similarities $s(x)$",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1d3b1d",
    )

    max_sim = max(max(sims), 1e-8)
    min_sim = min(min(sims), 0.0)

    n = len(prompts)
    row_h = (h - 0.12) / n

    # Axis labels
    ax.text(x + w - 0.09, y + h - 0.095, "cosine similarity", ha="center", va="center", fontsize=7)
    ax.text(x + w - 0.165, y + h - 0.118, "0", ha="center", va="center", fontsize=7)
    ax.text(x + w - 0.095, y + h - 0.118, "0.5", ha="center", va="center", fontsize=7)
    ax.text(x + w - 0.025, y + h - 0.118, "1.0", ha="center", va="center", fontsize=7)

    for i, (p, c, s) in enumerate(zip(prompts, cats, sims)):
        yy = y + h - 0.155 - i * row_h
        color = CATEGORY_COLORS.get(c, "#777777")
        icon = CATEGORY_ICON_TEXT.get(c, "•")

        rank_x = x + 0.025
        ax.add_patch(Circle((rank_x, yy), 0.014, facecolor="#6aa84f", edgecolor="none"))
        ax.text(rank_x, yy, str(i + 1), ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        ax.add_patch(Circle((x + 0.055, yy), 0.018, facecolor=color, edgecolor=color, alpha=0.18))
        ax.text(x + 0.055, yy, icon, ha="center", va="center", fontsize=8, color=color)

        prompt_short = shorten(p, width=25, placeholder="...")
        ax.text(x + 0.085, yy, prompt_short, ha="left", va="center", fontsize=7.8)

        ax.text(x + w - 0.205, yy, f"{s:.2f}", ha="right", va="center", fontsize=8.2)

        bar_x = x + w - 0.19
        bar_w = 0.15
        ax.add_patch(Rectangle((bar_x, yy - 0.009), bar_w, 0.018, facecolor="#eeeeee", edgecolor="none"))
        scaled = np.clip((s - min_sim) / (max_sim - min_sim + 1e-12), 0.0, 1.0)
        ax.add_patch(Rectangle((bar_x, yy - 0.009), bar_w * scaled, 0.018, facecolor=color, edgecolor="none", alpha=0.9))


def draw_legend(ax, x, y, w, h):
    add_round_box(ax, (x, y), (w, h), edgecolor="#cccccc", facecolor="#ffffff", linewidth=1.1)
    items = [
        ("#0b70d5", "image embedding"),
        (CATEGORY_COLORS["animal"], "animal prompts"),
        (CATEGORY_COLORS["vehicle"], "vehicle prompts"),
        (CATEGORY_COLORS["scene"], "scene prompt"),
        (CATEGORY_COLORS["object"], "object prompt"),
        ("#9178d0", "cosine similarity"),
    ]

    xpos = x + 0.035
    for color, label in items:
        if label == "cosine similarity":
            ax.plot([xpos - 0.01, xpos + 0.025], [y + h / 2, y + h / 2], linestyle=":", color=color, linewidth=2.0)
        else:
            ax.scatter([xpos], [y + h / 2], s=80, c=color, edgecolors="#444444", linewidths=0.4)

        ax.text(xpos + 0.025, y + h / 2, label, ha="left", va="center", fontsize=9)
        xpos += 0.15


# =========================================================
# Real VLM computation
# =========================================================

@torch.no_grad()
def compute_real_prompt_similarities(model, anchors, image_tensor, device):
    image_tensor = image_tensor.to(device)

    zI = model.encode_image(image_tensor)
    zI = F.normalize(zI.float(), dim=-1)

    zT = anchors.get_prompt_embeddings().to(device).float()
    zT = F.normalize(zT, dim=-1)

    sims = (zI @ zT.T).squeeze(0)

    return zI.squeeze(0), zT, sims


def select_balanced_top_prompts(prompts, cats, sims, top_k):
    """
    Choose top prompts while avoiding a figure where all top prompts come
    from only one category. This is only for visualization readability.
    The values still come from the real similarity vector.
    """
    sims_np = sims.detach().cpu().numpy()
    order = np.argsort(-sims_np)

    chosen = []
    used_cats = set()

    # First pass: choose high scoring prompts from distinct categories
    for idx in order:
        cat = cats[idx]
        if cat not in used_cats:
            chosen.append(idx)
            used_cats.add(cat)
        if len(chosen) >= top_k:
            break

    # Second pass: fill remaining with highest scoring prompts
    for idx in order:
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= top_k:
            break

    # Sort chosen by similarity descending for display
    chosen = sorted(chosen, key=lambda i: -sims_np[i])

    selected_prompts = [prompts[i] for i in chosen]
    selected_cats = [cats[i] for i in chosen]
    selected_sims = [float(sims_np[i]) for i in chosen]

    return chosen, selected_prompts, selected_cats, selected_sims


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=Datasets.CIFAR_10,
        choices=[Datasets.CIFAR_10, Datasets.STL_10, Datasets.TINY_IMAGENET],
    )
    parser.add_argument(
        "--model",
        default=Models.VIT_B16,
        choices=[Models.VIT_B16, Models.VIT_B32, Models.CONVNEXT_BASE_W],
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out", default="Images/real_vlm_similarity")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_global_seed()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = get_device(show_device_info=False)

    model_conf = Models.get_config(args.model)
    pretrained = model_conf["pretrained"]

    if args.batch_size is None:
        batch_size = Datasets.get_config(args.dataset)["batch_size"]
    else:
        batch_size = args.batch_size

    print(f"Loading model: {args.model} / {pretrained}")
    model, _, preprocess = create_model_and_transforms_cached(
        args.model,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    print("Building semantic prompt bank")
    anchors = AnchorBank(model, tokenizer, device=device)
    anchors.compute_embeddings()

    print("Loading dataset")
    train_loader, val_loader, test_loader, class_names = load_dataset_deterministic(
        dataset_name=args.dataset,
        preprocess=preprocess,
        batch_size=batch_size,
    )

    print("Selecting one real test image")
    selected_image = None
    selected_label = None
    seen = 0

    for images, labels in test_loader:
        b = images.shape[0]
        if seen + b > args.sample_index:
            local_idx = args.sample_index - seen
            selected_image = images[local_idx:local_idx + 1].to(device)
            selected_label = int(labels[local_idx].item())
            break
        seen += b

    if selected_image is None:
        raise IndexError(f"sample_index {args.sample_index} is outside the test loader.")

    label_name = class_names[selected_label] if selected_label < len(class_names) else str(selected_label)

    print("Computing real image-text similarities")
    zI, zT, sims = compute_real_prompt_similarities(model, anchors, selected_image, device)

    prompts = list(anchors.prompts)
    prompt_categories = list(anchors.prompt_categories)

    chosen_idx, top_prompts, top_cats, top_sims = select_balanced_top_prompts(
        prompts=prompts,
        cats=prompt_categories,
        sims=sims,
        top_k=args.top_k,
    )

    image_np = tensor_to_display_image(selected_image[0])

    # =====================================================
    # Figure layout
    # =====================================================

    fig = plt.figure(figsize=(15.5, 10.5), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Input image panel
    add_round_box(ax, (0.015, 0.49), (0.17, 0.47), edgecolor="#b7cbea", facecolor="#fbfdff", linewidth=1.4)
    ax.text(
        0.10,
        0.925,
        f"Input image: {label_name}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#162b74",
    )
    img_ax = fig.add_axes([0.032, 0.545, 0.136, 0.33])
    img_ax.imshow(image_np)
    img_ax.axis("off")

    # Encoder boxes
    draw_encoder_box(
        ax,
        x=0.255,
        y=0.66,
        w=0.15,
        h=0.24,
        title="Frozen\nimage\nencoder",
        subtitle=r"$f_I(\cdot)$",
        edgecolor="#7facdf",
        facecolor="#f3f8ff",
    )
    draw_encoder_box(
        ax,
        x=0.455,
        y=0.66,
        w=0.15,
        h=0.24,
        title="Frozen\ntext\nencoder",
        subtitle=r"$f_T(\cdot)$",
        edgecolor="#9cc48a",
        facecolor="#f8fff4",
    )

    # Prompt list
    draw_prompt_list(
        ax,
        prompts=top_prompts,
        cats=top_cats,
        sims=top_sims,
        x=0.705,
        y=0.48,
        w=0.275,
        h=0.48,
        title="Text prompt set",
    )

    # Similarity formula
    draw_similarity_formula_box(ax, x=0.015, y=0.10, w=0.17, h=0.30)

    # Embedding space
    draw_embedding_space(
        ax,
        x=0.215,
        y=0.13,
        w=0.45,
        h=0.42,
        prompts=top_prompts,
        cats=top_cats,
        sims=top_sims,
    )

    # Top similarity bars
    draw_top_similarity_bars(
        ax,
        prompts=top_prompts,
        cats=top_cats,
        sims=top_sims,
        x=0.705,
        y=0.10,
        w=0.275,
        h=0.32,
    )

    # Legend
    draw_legend(ax, x=0.18, y=0.025, w=0.64, h=0.055)

    # Arrows
    add_arrow(ax, (0.188, 0.73), (0.245, 0.78), color="#4f83cc")
    add_arrow(ax, (0.335, 0.66), (0.335, 0.56), color="#4f83cc")
    add_arrow(ax, (0.535, 0.66), (0.535, 0.56), color="#5c963f")
    add_arrow(ax, (0.705, 0.72), (0.615, 0.76), color="#5c963f")

    ax.text(
        0.455,
        0.635,
        "OpenCLIP / CLIP-style encoders",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#162b74",
    )

    # Dashed visual link from image to embedding
    ax.plot([0.185, 0.31], [0.49, 0.37], linestyle=":", color="#4f83cc", linewidth=1.3, alpha=0.75)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    png_path = args.out + ".png"
    pdf_path = args.out + ".pdf"

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    print("\nSelected image")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"label: {label_name}")
    print(f"sample_index: {args.sample_index}")

    print("\nDisplayed prompt similarities")
    for rank, (idx, p, c, s) in enumerate(zip(chosen_idx, top_prompts, top_cats, top_sims), start=1):
        print(f"{rank:02d}. idx={idx:02d}, category={c:8s}, sim={s:.6f}, prompt={p}")


if __name__ == "__main__":
    main()
