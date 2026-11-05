#!/usr/bin/env python
"""
run_ablation_study.py

Standalone ablation study runner for the VLM adversarial detector.

It does not modify the original codebase. It:
  1. Loads the existing model, dataset, anchors, prompt graph, ARF stats, and MSA.
  2. Builds the full 43-dimensional FeatureExtractor.
  3. Wraps the feature extractor to select feature subsets.
  4. Trains one detector per ablation variant.
  5. Evaluates each detector on clean + adversarial test samples.
  6. Saves ablation_results.csv and ablation_results.json.

Recommended first run:
  CIFAR-10 + ViT-B/16 + PGD

Example:
  python ablation.py
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import open_clip

from Constants import Datasets, Models, Attacks
from Utils import (
    build_adv_loader,
    load_dataset_deterministic,
    create_attack,
    get_device,
    release_memory,
    set_global_seed,
    build_text_features,
    create_model_and_transforms_cached,
)

from Detection.Pretraining.Anchors import AnchorBank
from Detection.Pretraining.PromptGraph import PromptGraph
from Detection.Pretraining.ARFStats import ARFStats
from Detection.Training.MSA import MSAComputer
from Detection.Training.FeatureExtractor import FeatureExtractor
from Detection.Training.TrainDetectorEnergy import EnergyTrainDetector, EnergyTrainConfig
from Detection.Inference.EvaluateDetector import EvaluateEnergyDetector


# =========================================================
# Configuration
# =========================================================

@dataclass
class AblationRunConfig:
    dataset_name: str = Datasets.CIFAR_10
    model_name: str = Models.VIT_B16
    attack_name: str = Attacks.PGD

    base_dir: str = "outputs/ablation_study"

    force_attack: bool = False
    force_train: bool = True

    graph_k: int = 8
    arf_max_batches: int | None = None

    # Keep this False for a compact paper ablation.
    evaluate_transfer: bool = False


# =========================================================
# Feature wrapper
# =========================================================

class AblationFeatureExtractor:
    """
    Wraps the original FeatureExtractor and selects only some dimensions.

    This lets us run ablations without editing FeatureExtractor.py.
    """

    def __init__(self, base_feature_extractor, feature_indices: Iterable[int], name: str):
        self.base = base_feature_extractor
        self.indices: List[int] = list(feature_indices)
        self.name = name

        if len(self.indices) == 0:
            raise ValueError("AblationFeatureExtractor received no feature indices.")

    def phi(self, images: torch.Tensor, return_debug: bool = False):
        if return_debug:
            phi_full, debug = self.base.phi(images, return_debug=True)
            return phi_full[:, self.indices], debug

        phi_full = self.base.phi(images, return_debug=False)
        return phi_full[:, self.indices]

    @property
    def feature_dim(self) -> int:
        return len(self.indices)

    def __getattr__(self, attr):
        return getattr(self.base, attr)


# =========================================================
# Feature groups
# =========================================================

def make_feature_groups(K: int = 7) -> Dict[str, List[int]]:
    """
    Feature order from FeatureExtractor.phi():

      0-6      masses
      7-13     means
      14-20    stds
      21-27    maxs
      28       entropy
      29       margin
      30       gse
      31       lga_max
      32       lga_mean
      33       lga_var
      34       lga_topkmean
      35       diff_delta_l2
      36       diff_cos
      37       graph_entropy
      38       R1
      39       R2
      40       S
      41       D
      42       f_norm

    For K=7, total dimension = 43.
    """

    category = list(range(0, 4 * K + 2))          # 0..29: category stats + entropy + margin
    graph = list(range(4 * K + 2, 4 * K + 7))     # 30..34: gse + lga stats
    diffusion = list(range(4 * K + 7, 4 * K + 10))# 35..37: diffusion + graph entropy
    residual = list(range(4 * K + 10, 4 * K + 12))# 38..39: R1, R2
    stability = list(range(4 * K + 12, 4 * K + 14))# 40..41: S, D
    norm = [4 * K + 14]                           # 42: raw encoder norm

    return {
        "category_only": category,
        "category_graph": category + graph + diffusion,
        "category_residual": category + residual,
        "category_stability": category + stability,
        "category_graph_residual": category + graph + diffusion + residual,
        "category_graph_stability": category + graph + diffusion + stability,
        "full": category + graph + diffusion + residual + stability + norm,
    }


def pretty_variant_name(name: str) -> str:
    return {
        "category_only": "Category only",
        "category_graph": "Category + graph",
        "category_residual": "Category + residual",
        "category_stability": "Category + stability",
        "category_graph_residual": "Category + graph + residual",
        "category_graph_stability": "Category + graph + stability",
        "full": "Full model",
    }.get(name, name)


# =========================================================
# Main ablation logic
# =========================================================

def run_ablation_study(cfg: AblationRunConfig):
    set_global_seed()
    device = get_device(show_device_info=True)

    model_conf = Models.get_config(cfg.model_name)
    pretrained_weights = model_conf["pretrained"]

    dataset_conf = Datasets.get_config(cfg.dataset_name)
    batch_size = dataset_conf["batch_size"]

    run_root = os.path.join(
        cfg.base_dir,
        cfg.dataset_name,
        cfg.model_name,
        cfg.attack_name,
    )
    os.makedirs(run_root, exist_ok=True)

    cache_dir = os.path.join(run_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    results_csv = os.path.join(run_root, "ablation_results.csv")
    results_json = os.path.join(run_root, "ablation_results.json")

    print("\n===================================================")
    print("Ablation study")
    print(f"Dataset : {cfg.dataset_name}")
    print(f"Model   : {cfg.model_name}")
    print(f"Weights : {pretrained_weights}")
    print(f"Attack  : {cfg.attack_name}")
    print(f"Output  : {run_root}")
    print("===================================================\n")

    # -----------------------------------------------------
    # CLIP model
    # -----------------------------------------------------
    model, _, preprocess = create_model_and_transforms_cached(
        cfg.model_name,
        pretrained=pretrained_weights,
    )
    tokenizer = open_clip.get_tokenizer(cfg.model_name)
    model = model.to(device).eval()

    # -----------------------------------------------------
    # Anchors and prompt graph
    # -----------------------------------------------------
    anchors = AnchorBank(model, tokenizer, device=device)
    anchors.compute_embeddings()

    graph = PromptGraph(
        anchors,
        device=device,
        k=cfg.graph_k,
        laplacian_type="unnormalized",
        weight_type="cosine",
    )
    graph.laplacian()

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    train_clean_loader, val_clean_loader, test_clean_loader, class_names = (
        load_dataset_deterministic(
            dataset_name=cfg.dataset_name,
            preprocess=preprocess,
            batch_size=batch_size,
        )
    )

    # -----------------------------------------------------
    # ARF statistics
    # -----------------------------------------------------
    arf = ARFStats(model=model, anchor_bank=anchors, device=device)
    arf.fit(train_clean_loader, max_batches=cfg.arf_max_batches)

    # Stability uses embedding cosine and raw prompt-response drift.
    msa = MSAComputer(
        device=device,
        prompt_graph=graph,
        anchor_bank=anchors,
    )

    full_fe = FeatureExtractor(
        model=model,
        anchor_bank=anchors,
        prompt_graph=graph,
        arf_stats=arf,
        msa_computer=msa,
        device=device,
    )

    print(f"Full feature dimension from extractor: {full_fe.feature_dim}")

    # -----------------------------------------------------
    # Attack and cached adversarial sets
    # -----------------------------------------------------
    attacker = create_attack(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        device=device,
        attack_name=cfg.attack_name,
        dataset_name=cfg.dataset_name,
    )

    train_adv_loader = build_adv_loader(
        train_clean_loader,
        attacker,
        cache_path=os.path.join(cache_dir, "train_adv.pt"),
        device=device,
        use_cached_attack=not cfg.force_attack,
    )

    val_adv_loader = build_adv_loader(
        val_clean_loader,
        attacker,
        cache_path=os.path.join(cache_dir, "val_adv.pt"),
        device=device,
        use_cached_attack=not cfg.force_attack,
    )

    # Table 5 changes only the detector feature groups.  Every variant uses
    # the same classifier-only PGD test examples, matching the paper protocol.
    test_adv_loader = build_adv_loader(
        test_clean_loader,
        attacker,
        cache_path=os.path.join(cache_dir, "test_adv.pt"),
        device=device,
        use_cached_attack=not cfg.force_attack,
    )

    text_features = build_text_features(model, tokenizer, class_names, device)

    @torch.no_grad()
    def attack_success_metrics(clean_loader, adversarial_loader):
        clean_correct = adversarial_correct = successful = initially_correct = total = 0
        for (clean, labels), (adversarial, adversarial_labels) in zip(clean_loader, adversarial_loader):
            if not torch.equal(labels, adversarial_labels):
                raise ValueError("Clean/adversarial labels are not aligned")
            labels = labels.to(device)
            def predict(images):
                features = torch.nn.functional.normalize(
                    model.encode_image(images.to(device)).float(), dim=-1
                )
                logits = features @ text_features.float().T
                if hasattr(model, "logit_scale"):
                    logits = logits * model.logit_scale.exp().float()
                return logits.argmax(1)
            clean_prediction = predict(clean)
            adversarial_prediction = predict(adversarial)
            correct = clean_prediction.eq(labels)
            clean_correct += int(correct.sum())
            adversarial_correct += int(adversarial_prediction.eq(labels).sum())
            initially_correct += int(correct.sum())
            successful += int((correct & adversarial_prediction.ne(labels)).sum())
            total += len(labels)
        return {
            "clean_classification_accuracy": clean_correct / max(total, 1),
            "adversarial_classification_accuracy": adversarial_correct / max(total, 1),
            "attack_success_rate": successful / max(initially_correct, 1),
        }

    # -----------------------------------------------------
    # Ablation variants
    # -----------------------------------------------------
    groups = make_feature_groups(K=full_fe.K)

    # You can shorten this list for a faster paper table.
    variants_to_run = [
        "category_only",
        "category_graph",
        "category_residual",
        "category_stability",
        "category_graph_residual",
        "category_graph_stability",
        "full",
    ]

    all_results = []

    import result_store
    for variant in variants_to_run:
        result_key = os.path.join(run_root, variant, 'completed')
        cached = result_store.get(result_key)
        if cached is not None:
            all_results.append(cached)
            continue
        print("\n---------------------------------------------------")
        print(f"Running ablation: {variant}")
        print("---------------------------------------------------")

        set_global_seed()
        feature_indices = groups[variant]
        fe = AblationFeatureExtractor(
            base_feature_extractor=full_fe,
            feature_indices=feature_indices,
            name=variant,
        )

        variant_dir = os.path.join(run_root, variant)
        plots_dir = os.path.join(variant_dir, "detector_plots")
        os.makedirs(plots_dir, exist_ok=True)

        trainer = EnergyTrainDetector(
            feature_extractor=fe,
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            device=device,
            cfg=EnergyTrainConfig(device=device),
        )

        best_threshold = trainer.fit(
            train_clean_loader=train_clean_loader,
            val_clean_loader=val_clean_loader,
            train_adv_loader=train_adv_loader,
            val_adv_loader=val_adv_loader,
            save_dir=plots_dir,
        )

        evaluator = EvaluateEnergyDetector(
            detector=trainer.detector,
            feature_extractor=fe,
            threshold=best_threshold,
            clean_mean=trainer.clean_mean,
            clean_cov_inv_sqrt=trainer.clean_cov_inv_sqrt,
            device=device,
        )

        metrics = evaluator.evaluate(
            clean_loader=test_clean_loader,
            adv_loader=test_adv_loader,
            save_dir=variant_dir,
        )
        metrics.update(attack_success_metrics(test_clean_loader, test_adv_loader))
        result_store.put(variant_dir, metrics)

        row = {
            "variant": variant,
            "variant_name": pretty_variant_name(variant),
            "feature_dim": fe.feature_dim,
            "dataset": cfg.dataset_name,
            "model": cfg.model_name,
            "attack": cfg.attack_name,
            "threshold": float(best_threshold),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "roc_auc": float(metrics.get("roc_auc", 0.0)),
            "tpr@5%fpr": float(metrics.get("tpr@5%fpr", 0.0)),
            "clean_classification_accuracy": float(metrics["clean_classification_accuracy"]),
            "adversarial_classification_accuracy": float(metrics["adversarial_classification_accuracy"]),
            "attack_success_rate": float(metrics["attack_success_rate"]),
        }

        all_results.append(row)

        # Save after each variant, so progress is not lost.
        if result_store.enabled():
            result_store.put(result_key, row)
        else:
            save_results(all_results, results_csv, results_json)

        print(f"Result for {variant}: {row}")

        release_memory()

    print("\n===================================================")
    print("Ablation study complete")
    print(f"CSV : {results_csv}")
    print(f"JSON: {results_json}")
    print("===================================================")

    print_latex_table(all_results)
    return all_results


# =========================================================
# Saving and table output
# =========================================================

def save_results(rows: List[Dict], csv_path: str, json_path: str):
    if len(rows) == 0:
        return

    keys = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def print_latex_table(rows: List[Dict]):
    print("\nLaTeX table rows:")
    print("---------------------------------------------------")
    for r in rows:
        acc = 100.0 * r["accuracy"]
        auc = 100.0 * r["roc_auc"]
        tpr = 100.0 * r["tpr@5%fpr"]
        print(
            f"{r['variant_name']} & {r['feature_dim']} & "
            f"{acc:.2f} & {auc:.2f} & {tpr:.2f} \\\\"
        )
    print("---------------------------------------------------\n")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    cfg = AblationRunConfig(
        dataset_name=Datasets.CIFAR_10,
        model_name=Models.VIT_B16,
        attack_name=Attacks.PGD,
        base_dir="outputs/ablation_study",
        force_attack=False,
        force_train=True,
        evaluate_transfer=False,
    )

    run_ablation_study(cfg)
