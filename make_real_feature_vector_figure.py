#!/usr/bin/env python
"""
Create a real feature-vector construction figure from one actual image.

This script:
1. Loads a real image from CIFAR-10, STL-10, or Tiny-ImageNet.
2. Builds the same CLIP model, anchor prompts, prompt graph, ARF stats, and feature extractor.
3. Computes the actual feature values for one image.
4. Creates a paper-ready PNG and PDF figure.

Run example:
    python make_real_feature_vector_figure.py \
        --dataset cifar_10 \
        --model ViT-B-16 \
        --sample_index 0 \
        --out Images/real_feature_vector_construction

For portable PDF fonts, it also saves a PDF with Type 42 fonts.
"""

import os
import argparse
import math
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
from PIL import Image

import open_clip

from Constants import Datasets, Models
from Utils import (get_device, set_global_seed, load_dataset_deterministic,
                   create_model_and_transforms_cached)
from Detection.Pretraining.Anchors import AnchorBank
from Detection.Pretraining.PromptGraph import PromptGraph
from Detection.Pretraining.ARFStats import ARFStats
from Detection.Training.MSA import MSAComputer
from Detection.Training.FeatureExtractor import FeatureExtractor


# =========================================================
# Helpers
# =========================================================

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def add_arrow_between(
    fig,
    ax_from,
    ax_to,
    color="#555555",
    from_anchor="top",
    to_anchor="bottom",
    mutation_scale=18,
    linewidth=2.4,
):
    """
    Draw a directional arrow between two axes in figure coordinates.

    Use this instead of add_arrow when you need explicit control over
    arrow direction.
    """
    p1 = ax_from.get_position()
    p2 = ax_to.get_position()

    def anchor_point(p, anchor):
        cx = (p.x0 + p.x1) / 2
        cy = (p.y0 + p.y1) / 2

        if anchor == "top":
            return (cx, p.y1)
        if anchor == "bottom":
            return (cx, p.y0)
        if anchor == "left":
            return (p.x0, cy)
        if anchor == "right":
            return (p.x1, cy)
        if anchor == "center":
            return (cx, cy)

        raise ValueError(f"Unknown anchor: {anchor}")

    start = anchor_point(p1, from_anchor)
    end = anchor_point(p2, to_anchor)

    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=0.82,
        shrinkA=6,
        shrinkB=6,
        connectionstyle="arc3,rad=0.0",
    )
    fig.patches.append(arrow)

def tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized CLIP tensor [3,H,W] into displayable RGB [H,W,3].
    """
    x = x.detach().cpu().float()
    x = x * CLIP_STD + CLIP_MEAN
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def minmax01(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + eps)


def add_panel(ax, title, edgecolor="#c8c8c8", facecolor="#fbfbfb"):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        0.5, 0.91, title,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
    )


def add_arrow(fig, ax_from, ax_to, color="#555555"):
    """
    Draw an arrow between two axes in figure coordinates.
    """
    p1 = ax_from.get_position()
    p2 = ax_to.get_position()

    start = ((p1.x0 + p1.x1) / 2, p1.y0)
    end = ((p2.x0 + p2.x1) / 2, p2.y1)

    arrow = FancyArrowPatch(
        start, end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        alpha=0.75,
    )
    fig.patches.append(arrow)


def draw_category_panel(ax, category_names, masses, means, stds, maxs, entropy, margin):
    add_panel(ax, "Category statistics", edgecolor="#b7cbea", facecolor="#fbfdff")

    inner = ax.inset_axes([0.08, 0.18, 0.86, 0.58])
    x = np.arange(len(category_names))
    width = 0.18

    vals = [
        np.asarray(masses),
        np.asarray(means),
        np.asarray(stds),
        np.asarray(maxs),
    ]
    labels = ["mass", "mean", "std", "max"]

    for i, v in enumerate(vals):
        inner.bar(x + (i - 1.5) * width, v, width=width, label=labels[i], alpha=0.8)

    inner.set_xticks(x)
    inner.set_xticklabels(category_names, rotation=25, ha="right", fontsize=8)
    inner.tick_params(axis="y", labelsize=8)
    inner.set_ylabel("value", fontsize=9)
    inner.grid(axis="y", linestyle=":", alpha=0.4)
    inner.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    ax.text(
        0.5, 0.08,
        rf"$H(x)={entropy:.3f}$,  $\gamma(x)={margin:.3f}$",
        ha="center", va="center",
        fontsize=11,
    )


def draw_graph_panel(ax, A, s, delta, prompt_categories, category_names, gse, lga_mean, lga_max):
    add_panel(ax, "Prompt graph features", edgecolor="#c7d8b8", facecolor="#fcfffb")

    # Use a deterministic circular-category layout.
    M = A.shape[0]
    K = len(category_names)

    cat_to_angle = {
        cat: 2 * math.pi * i / K
        for i, cat in enumerate(category_names)
    }

    coords = np.zeros((M, 2), dtype=float)
    rng = np.random.default_rng(42)

    for i, cat in enumerate(prompt_categories):
        angle = cat_to_angle[cat]
        radius = 0.33 + 0.08 * rng.random()
        jitter = rng.normal(scale=0.035, size=2)
        coords[i, 0] = 0.50 + radius * math.cos(angle) + jitter[0]
        coords[i, 1] = 0.50 + radius * math.sin(angle) + jitter[1]

    # Edges
    edges = []
    A_np = A.detach().cpu().numpy()
    for i in range(M):
        for j in range(i + 1, M):
            if A_np[i, j] > 0:
                edges.append([coords[i], coords[j]])

    graph_ax = ax.inset_axes([0.06, 0.18, 0.58, 0.62])
    graph_ax.set_xticks([])
    graph_ax.set_yticks([])
    graph_ax.axis("off")

    if edges:
        lc = LineCollection(edges, colors="0.72", linewidths=0.7, alpha=0.55)
        graph_ax.add_collection(lc)

    s01 = minmax01(s)
    graph_ax.scatter(
        coords[:, 0], coords[:, 1],
        c=s01,
        s=70,
        cmap="viridis",
        edgecolors="0.25",
        linewidths=0.5,
        zorder=3,
    )
    graph_ax.set_xlim(0, 1)
    graph_ax.set_ylim(0, 1)
    graph_ax.set_title("signal on prompt graph", fontsize=9)

    # Local deviation mini graph
    local_ax = ax.inset_axes([0.68, 0.25, 0.24, 0.45])
    local_ax.set_xticks([])
    local_ax.set_yticks([])
    local_ax.axis("off")

    top_idx = np.argsort(delta)[-12:]
    sub_coords = coords[top_idx]
    sub_delta = minmax01(delta[top_idx])

    for a in range(len(top_idx)):
        for b in range(a + 1, len(top_idx)):
            if A_np[top_idx[a], top_idx[b]] > 0:
                local_ax.plot(
                    [sub_coords[a, 0], sub_coords[b, 0]],
                    [sub_coords[a, 1], sub_coords[b, 1]],
                    color="0.75",
                    linewidth=0.8,
                    alpha=0.7,
                )

    local_ax.scatter(
        sub_coords[:, 0], sub_coords[:, 1],
        c=sub_delta,
        s=65,
        cmap="coolwarm",
        edgecolors="0.25",
        linewidths=0.5,
        zorder=3,
    )
    local_ax.set_xlim(0, 1)
    local_ax.set_ylim(0, 1)
    local_ax.set_title(r"local $\delta$", fontsize=9)

    ax.text(
        0.5, 0.08,
        rf"$E_G(x)={gse:.4f}$,  mean$(\delta)={lga_mean:.4f}$,  max$(\delta)={lga_max:.4f}$",
        ha="center", va="center",
        fontsize=10,
    )


def draw_residual_panel(ax, category_names, top2_idx, R1, R2):
    add_panel(ax, "Clean residual features", edgecolor="#d7c7ee", facecolor="#fefdff")

    top1 = category_names[int(top2_idx[0])]
    top2 = category_names[int(top2_idx[1])]

    inner = ax.inset_axes([0.20, 0.25, 0.60, 0.48])
    values = [float(R1), float(R2)]
    names = [rf"$R_{{top1}}$" + f"\n{top1}", rf"$R_{{top2}}$" + f"\n{top2}"]

    inner.bar(np.arange(2), values, width=0.45, alpha=0.85)
    inner.set_xticks(np.arange(2))
    inner.set_xticklabels(names, fontsize=9)
    inner.tick_params(axis="y", labelsize=8)
    inner.set_ylabel("Mahalanobis residual", fontsize=9)
    inner.grid(axis="y", linestyle=":", alpha=0.4)

    ax.text(
        0.5, 0.10,
        "computed from clean training statistics",
        ha="center", va="center",
        fontsize=10,
    )


def draw_stability_panel(ax, image_np, transformed_nps, S, D, diff_l2, diff_cos):
    add_panel(ax, "Stability features", edgecolor="#efd0aa", facecolor="#fffdf9")

    ax_img = ax.inset_axes([0.07, 0.33, 0.22, 0.42])
    ax_img.imshow(image_np)
    ax_img.axis("off")
    ax_img.set_title("source image", fontsize=9)

    labels = ["shift", "blur", "contrast"]
    for i, im in enumerate(transformed_nps):
        a = ax.inset_axes([0.36 + 0.17 * i, 0.36, 0.14, 0.34])
        a.imshow(im)
        a.axis("off")
        a.set_title(labels[i], fontsize=8)

    gauge = ax.inset_axes([0.78, 0.35, 0.15, 0.33])
    gauge.axis("off")
    theta = np.linspace(math.pi, 0, 100)
    gauge.plot(np.cos(theta), np.sin(theta), linewidth=8, alpha=0.25)
    needle_angle = math.pi * (1 - float(np.clip(S, 0, 1)))
    gauge.plot([0, 0.85 * math.cos(needle_angle)], [0, 0.85 * math.sin(needle_angle)], linewidth=2.5)
    gauge.scatter([0], [0], s=30)
    gauge.set_xlim(-1.15, 1.15)
    gauge.set_ylim(-0.15, 1.15)
    gauge.set_title("stability", fontsize=9)

    ax.text(
        0.5, 0.12,
        rf"$S={S:.4f}$, $D={D:.4f}$, $\Delta_{{diff}}={diff_l2:.4f}$, cos$_{{diff}}={diff_cos:.4f}$",
        ha="center", va="center",
        fontsize=10,
    )


def draw_feature_vector_panel(ax, phi, feature_dim):
    add_panel(ax, "Feature vector construction", edgecolor="#cec3ee", facecolor="#fdfcff")

    phi_np = phi.detach().cpu().numpy().reshape(-1)
    n = len(phi_np)

    inner = ax.inset_axes([0.09, 0.36, 0.82, 0.25])
    inner.imshow(phi_np.reshape(1, -1), aspect="auto", cmap="viridis")
    inner.set_yticks([])
    inner.set_xticks([])
    inner.set_title(r"actual $\phi(x)$", fontsize=11)

    ax.text(
        0.5, 0.25,
        f"feature dimension = {feature_dim}",
        ha="center", va="center",
        fontsize=11,
    )
    ax.text(
        0.5, 0.13,
        "category + graph + residual + stability features",
        ha="center", va="center",
        fontsize=10,
    )


def draw_detector_panel(ax):
    add_panel(ax, "MLP detector", edgecolor="#d4c7ef", facecolor="#fefdff")

    # Simple schematic of the actual MLP head.
    layers = [4, 6, 5, 3, 1]
    xs = np.linspace(0.22, 0.78, len(layers))

    for li, count in enumerate(layers):
        ys = np.linspace(0.25, 0.75, count)
        for y in ys:
            ax.scatter(xs[li], y, s=70, facecolor="#f1ecff", edgecolor="#4b3b8f", linewidth=1.0, zorder=3)

    for li in range(len(layers) - 1):
        ys1 = np.linspace(0.25, 0.75, layers[li])
        ys2 = np.linspace(0.25, 0.75, layers[li + 1])
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([xs[li], xs[li + 1]], [y1, y2], color="#8f85bf", linewidth=0.5, alpha=0.45)

    ax.text(0.5, 0.12, "score for adversarial detection", ha="center", fontsize=10)


def make_transformed_images(msa, image_tensor):
    """
    Use the same transformation functions used by MSAComputer.
    """
    with torch.no_grad():
        x = image_tensor
        x_shift = msa._perturb_shift(x)
        x_blur = msa._perturb_blur(x)
        x_contrast = msa._perturb_contrast(x)

    return [
        tensor_to_display_image(x_shift[0]),
        tensor_to_display_image(x_blur[0]),
        tensor_to_display_image(x_contrast[0]),
    ]


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=Datasets.CIFAR_10, choices=[
        Datasets.CIFAR_10, Datasets.STL_10, Datasets.TINY_IMAGENET
    ])
    parser.add_argument("--model", default=Models.VIT_B16, choices=[
        Models.VIT_B16, Models.VIT_B32, Models.CONVNEXT_BASE_W
    ])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--arf_batches", type=int, default=None)
    parser.add_argument("--out", default="Images/real_feature_vector_construction")
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

    print("Building anchors and prompt graph")
    anchors = AnchorBank(model, tokenizer, device=device)
    anchors.compute_embeddings()

    graph = PromptGraph(
        anchors,
        device=device,
        k=8,
        laplacian_type="unnormalized",
        weight_type="cosine",
    )
    graph_artifacts = graph.artifacts()

    print("Loading dataset")
    train_loader, val_loader, test_loader, class_names = load_dataset_deterministic(
        dataset_name=args.dataset,
        preprocess=preprocess,
        batch_size=batch_size,
    )

    print("Fitting clean residual statistics")
    arf = ARFStats(
        model=model,
        anchor_bank=anchors,
        device=device,
    )
    arf.fit(train_loader, max_batches=args.arf_batches)

    # Transformation stability uses the same frozen prompt embeddings.
    msa = MSAComputer(
        device=device,
        prompt_graph=graph,
        anchor_bank=anchors,
    )

    fe = FeatureExtractor(
        model=model,
        anchor_bank=anchors,
        prompt_graph=graph,
        arf_stats=arf,
        msa_computer=msa,
        device=device,
    )

    print("Selecting one real test image")
    all_seen = 0
    selected_image = None
    selected_label = None

    for images, labels in test_loader:
        b = images.shape[0]
        if all_seen + b > args.sample_index:
            local_idx = args.sample_index - all_seen
            selected_image = images[local_idx:local_idx + 1].to(device)
            selected_label = int(labels[local_idx].item())
            break
        all_seen += b

    if selected_image is None:
        raise IndexError(f"sample_index {args.sample_index} is outside the test loader.")

    with torch.no_grad():
        phi, debug = fe.phi(selected_image, return_debug=True)

    category_names = fe.category_names
    A = graph_artifacts.adjacency.detach().cpu()
    prompt_categories = anchors.prompt_categories

    masses = debug["masses"][0].detach().cpu().numpy()

    # Actual category-level values from FeatureExtractor.
    # Recompute means, stds, and maxs from s using the same category indices for clean access.
    s = debug["s"][0]
    cat_means, cat_stds, cat_maxs = [], [], []
    for cat in category_names:
        idx = fe.category_to_indices[cat]
        sub = s[idx]
        cat_means.append(float(sub.mean().detach().cpu()))
        cat_stds.append(float(sub.std(unbiased=False).detach().cpu()))
        cat_maxs.append(float(sub.max().detach().cpu()))

    cat_means = np.asarray(cat_means)
    cat_stds = np.asarray(cat_stds)
    cat_maxs = np.asarray(cat_maxs)

    entropy = float(debug["entropy"][0].detach().cpu())
    margin = float(debug["margin"][0].detach().cpu())
    gse = float(debug["gse"][0].detach().cpu())
    delta = debug["a_lga"][0].detach().cpu().numpy()
    lga_mean = float(debug["lga_mean"][0].detach().cpu())
    lga_max = float(debug["lga_max"][0].detach().cpu())

    R1 = float(debug["R1"][0].detach().cpu())
    R2 = float(debug["R2"][0].detach().cpu())
    top2_idx = debug["top2_idx"][0].detach().cpu().numpy()

    S = float(debug["S"][0].detach().cpu())
    D = float(debug["D"][0].detach().cpu())
    diff_l2 = float(debug["diff_delta_l2"][0].detach().cpu())
    diff_cos = float(debug["diff_cos"][0].detach().cpu())

    image_np = tensor_to_display_image(selected_image[0])
    transformed_nps = make_transformed_images(msa, selected_image)

    print("Creating figure")
    fig = plt.figure(figsize=(14, 9), dpi=220)
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.35, 1.35, 1.35, 0.95],
        height_ratios=[1.05, 0.95, 1.05],
        wspace=0.28,
        hspace=0.28,
    )

    ax_cat = fig.add_subplot(gs[0, 0:2])
    ax_graph = fig.add_subplot(gs[0, 2:4])
    ax_feat = fig.add_subplot(gs[1, 1:3])
    ax_det = fig.add_subplot(gs[1, 3])
    ax_res = fig.add_subplot(gs[2, 0:2])
    ax_stab = fig.add_subplot(gs[2, 2:4])

    draw_category_panel(
        ax_cat,
        category_names,
        masses,
        cat_means,
        cat_stds,
        cat_maxs,
        entropy,
        margin,
    )

    draw_graph_panel(
        ax_graph,
        A,
        debug["s"][0].detach().cpu().numpy(),
        delta,
        prompt_categories,
        category_names,
        gse,
        lga_mean,
        lga_max,
    )

    draw_residual_panel(
        ax_res,
        category_names,
        top2_idx,
        R1,
        R2,
    )

    draw_stability_panel(
        ax_stab,
        image_np,
        transformed_nps,
        S,
        D,
        diff_l2,
        diff_cos,
    )

    draw_feature_vector_panel(
        ax_feat,
        phi[0],
        fe.feature_dim,
    )

    draw_detector_panel(ax_det)

        # All feature groups are inputs to phi(x), so arrows point INTO the feature vector.
    add_arrow_between(
        fig, ax_cat, ax_feat,
        color="#2d6ca2",
        from_anchor="bottom",
        to_anchor="top",
    )

    add_arrow_between(
        fig, ax_graph, ax_feat,
        color="#5b8c3c",
        from_anchor="bottom",
        to_anchor="top",
    )

    add_arrow_between(
        fig, ax_res, ax_feat,
        color="#6b4fa0",
        from_anchor="top",
        to_anchor="bottom",
    )

    add_arrow_between(
        fig, ax_stab, ax_feat,
        color="#c8781d",
        from_anchor="top",
        to_anchor="bottom",
    )

    p_feat = ax_feat.get_position()
    p_det = ax_det.get_position()
    arrow = FancyArrowPatch(
        (p_feat.x1, (p_feat.y0 + p_feat.y1) / 2),
        (p_det.x0, (p_det.y0 + p_det.y1) / 2),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.5,
        color="#7b61b8",
        alpha=0.8,
    )
    fig.patches.append(arrow)

    label_name = class_names[selected_label] if selected_label < len(class_names) else str(selected_label)
    fig.suptitle(
        f"Real feature-vector construction for one {args.dataset} test image "
        f"(label: {label_name}, model: {args.model})",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    png_path = args.out + ".png"
    pdf_path = args.out + ".pdf"

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")

    print("\nReal values used in the figure:")
    print(f"category masses: {dict(zip(category_names, np.round(masses, 4)))}")
    print(f"entropy: {entropy:.6f}")
    print(f"margin: {margin:.6f}")
    print(f"graph energy: {gse:.6f}")
    print(f"R_top1: {R1:.6f}")
    print(f"R_top2: {R2:.6f}")
    print(f"S stability: {S:.6f}")
    print(f"D drift: {D:.6f}")
    print(f"feature dimension: {fe.feature_dim}")


if __name__ == "__main__":
    main()
