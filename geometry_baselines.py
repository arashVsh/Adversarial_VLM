"""Ordinary PGD-calibrated embedding baselines with unseen-attack evaluation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Constants import Attacks, DATA_PROTOCOL_VERSION, Datasets, Detectors, Models
from Utils import (build_adv_loader, build_text_features,
                   create_attack, create_model_and_transforms_cached,
                   load_dataset_deterministic, release_memory, set_global_seed)


BASELINE_NAMES = ("KNN", "Mahalanobis", "KDE", "LID")
GEOMETRY_BASELINE_VERSION = 1


@dataclass(frozen=True)
class GeometryBaselineConfig:
    version: int = GEOMETRY_BASELINE_VERSION
    reference_samples: int = 5000
    score_chunk_size: int = 256
    covariance_shrinkage: float = 0.20
    covariance_ridge: float = 1e-3
    knn_candidates: tuple = (5, 10, 20, 50)
    lid_candidates: tuple = (5, 10, 20, 50)
    kde_bandwidths: tuple = (0.05, 0.1, 0.2, 0.5)


@torch.no_grad()
def collect_embeddings(model, loader, device, description):
    embeddings, labels = [], []
    for images, batch_labels in tqdm(loader, desc=description, leave=False):
        with torch.amp.autocast(device_type=device.type,
                                enabled=device.type == "cuda"):
            encoded = model.encode_image(images.to(device, non_blocking=True))
        embeddings.append(F.normalize(encoded.float(), dim=-1).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(embeddings), torch.cat(labels)


def fit_geometry_state(reference, cfg, device):
    reference = reference.to(device)
    mean = reference.mean(0)
    centered = reference - mean
    covariance = centered.T @ centered / len(reference)
    covariance = (
        (1 - cfg.covariance_shrinkage) * covariance
        + cfg.covariance_shrinkage * torch.diag(covariance.diag())
        + cfg.covariance_ridge
        * torch.eye(covariance.shape[0], device=device, dtype=covariance.dtype)
    )
    inverse = torch.linalg.inv(covariance)
    return {"reference": reference.cpu(), "mean": mean.cpu(),
            "inverse": inverse.cpu()}


def _key(name, value=None):
    return name if value is None else f"{name}:{value:g}"


@torch.no_grad()
def calculate_candidate_scores(embeddings, state, cfg, device, selection=None):
    """Calculate all validation candidates or only the frozen selected candidates."""
    reference = state["reference"].to(device)
    mean = state["mean"].to(device)
    inverse = state["inverse"].to(device)
    if selection is None:
        knn_values = cfg.knn_candidates
        lid_values = cfg.lid_candidates
        kde_values = cfg.kde_bandwidths
    else:
        knn_values = (selection["KNN"]["k"],)
        lid_values = (selection["LID"]["k"],)
        kde_values = (selection["KDE"]["bandwidth"],)
    knn_values = tuple(min(int(k), len(reference)) for k in knn_values)
    lid_values = tuple(min(int(k), len(reference)) for k in lid_values)
    maximum_k = max((*knn_values, *lid_values))
    keys = (["Mahalanobis"]
            + [_key("KNN", k) for k in knn_values]
            + [_key("LID", k) for k in lid_values]
            + [_key("KDE", bandwidth) for bandwidth in kde_values])
    values = {key: [] for key in keys}
    for start in range(0, len(embeddings), cfg.score_chunk_size):
        z = embeddings[start:start + cfg.score_chunk_size].to(device)
        delta = z - mean
        values["Mahalanobis"].append(
            torch.einsum("bi,ij,bj->b", delta, inverse, delta).cpu())
        distances_squared = (
            z.square().sum(1, keepdim=True)
            + reference.square().sum(1).unsqueeze(0)
            - 2 * z @ reference.T
        ).clamp_min(1e-12)
        nearest = distances_squared.topk(maximum_k, largest=False).values.sqrt()
        for k in knn_values:
            values[_key("KNN", k)].append(nearest[:, :k].mean(1).cpu())
        for k in lid_values:
            distances = nearest[:, :k]
            denominator = torch.log(
                (distances / distances[:, -1:].clamp_min(1e-12)).clamp_min(1e-12)
            ).sum(1).clamp_max(-1e-6)
            values[_key("LID", k)].append((-k / denominator).cpu())
        for bandwidth in kde_values:
            score = (-torch.logsumexp(
                -distances_squared / (2 * float(bandwidth) ** 2), dim=1)
                     + math.log(len(reference)))
            values[_key("KDE", bandwidth)].append(score.cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def detection_metrics(clean_scores, adversarial_scores, threshold):
    scores = torch.cat([clean_scores, adversarial_scores]).numpy()
    labels = np.concatenate([
        np.zeros(len(clean_scores), dtype=np.int64),
        np.ones(len(adversarial_scores), dtype=np.int64),
    ])
    predictions = scores >= threshold
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tpr@5%fpr": float(tpr[fpr <= 0.05].max()),
    }


def select_threshold(clean_scores, adversarial_scores):
    scores = torch.cat([clean_scores, adversarial_scores]).numpy()
    labels = np.concatenate([
        np.zeros(len(clean_scores), dtype=np.int64),
        np.ones(len(adversarial_scores), dtype=np.int64),
    ])
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(f1.argmax())])


def choose_baselines(clean_candidates, pgd_candidates, cfg):
    families = {
        "KNN": [_key("KNN", k) for k in cfg.knn_candidates],
        "Mahalanobis": ["Mahalanobis"],
        "KDE": [_key("KDE", bandwidth) for bandwidth in cfg.kde_bandwidths],
        "LID": [_key("LID", k) for k in cfg.lid_candidates],
    }
    selected = {}
    for name, candidate_keys in families.items():
        best_key = max(
            candidate_keys,
            key=lambda key: detection_metrics(
                clean_candidates[key], pgd_candidates[key], 0.0)["roc_auc"],
        )
        threshold = select_threshold(clean_candidates[best_key],
                                     pgd_candidates[best_key])
        item = {"score_key": best_key, "threshold": threshold}
        if name in {"KNN", "LID"}:
            item["k"] = int(best_key.split(":", 1)[1])
        if name == "KDE":
            item["bandwidth"] = float(best_key.split(":", 1)[1])
        selected[name] = item
    return selected


def classification_metrics(clean_embeddings, adversarial_embeddings, labels,
                           text_features):
    text_features = text_features.cpu().float()
    clean_predictions = (clean_embeddings.float() @ text_features.T).argmax(1)
    adversarial_predictions = (
        adversarial_embeddings.float() @ text_features.T).argmax(1)
    correct = clean_predictions.eq(labels)
    return {
        "clean_classification_accuracy": float(correct.float().mean()),
        "adversarial_classification_accuracy": float(
            adversarial_predictions.eq(labels).float().mean()),
        "attack_success_rate": float(
            (correct & adversarial_predictions.ne(labels)).sum()
            / max(int(correct.sum()), 1)),
    }


def _reference_loader(train_loader, samples):
    subset = Subset(train_loader.dataset,
                    range(min(int(samples), len(train_loader.dataset))))
    return DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def run_geometry_baselines(dataset, model_name, output_root, device,
                           cfg=None):
    """Fit on clean/PGD train-validation data and test on ordinary unseen attacks."""
    import open_clip
    import result_store

    cfg = cfg or GeometryBaselineConfig()
    set_global_seed()
    model_cfg = Models.get_config(model_name)
    dataset_cfg = Datasets.get_config(dataset)
    batch_size = int(os.environ.get(
        "GEOMETRY_BASELINE_BATCH_SIZE", dataset_cfg["batch_size"]))
    if batch_size < 1:
        raise ValueError("GEOMETRY_BASELINE_BATCH_SIZE must be positive")
    root = (Path(output_root) / "geometry_baselines" / "pgd"
            / dataset / model_name)
    root.mkdir(parents=True, exist_ok=True)
    (root / "configuration.json").write_text(json.dumps({
        **asdict(cfg), "dataset": dataset, "model": model_name,
        "pretrained": model_cfg["pretrained"], "training_attack": Attacks.PGD,
        "evaluation_attacks": list(Attacks.ALL), "batch_size": batch_size,
        "data_protocol_version": DATA_PROTOCOL_VERSION,
    }, indent=2), encoding="utf-8")

    model, _, preprocess = create_model_and_transforms_cached(
        model_name, pretrained=model_cfg["pretrained"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(model_name)
    train, val, test, class_names = load_dataset_deterministic(
        dataset, preprocess, batch_size)
    text_features = build_text_features(model, tokenizer, class_names, device)

    bundle_path = root / "baseline_bundle.pt"
    bundle = None
    if bundle_path.exists():
        candidate = torch.load(bundle_path, map_location="cpu", weights_only=True)
        if (candidate.get("geometry_baseline_version") == cfg.version
                and candidate.get("dataset") == dataset
                and candidate.get("model") == model_name
                and candidate.get("data_protocol_version")
                == DATA_PROTOCOL_VERSION):
            bundle = candidate
    if bundle is None:
        reference, _ = collect_embeddings(
            model, _reference_loader(train, cfg.reference_samples), device,
            "Embedding clean reference")
        state = fit_geometry_state(reference, cfg, device)
        clean_val, _ = collect_embeddings(model, val, device,
                                          "Embedding clean validation")
        pgd = create_attack(model, tokenizer, class_names, device, Attacks.PGD,
                            dataset_name=dataset)
        pgd_val_path = (Path(output_root) / Detectors.ENERGY_BASED / Attacks.PGD
                        / dataset / model_name / "cache" / "val_adv.pt")
        pgd_val = build_adv_loader(val, pgd, str(pgd_val_path), device, True)
        adversarial_val, _ = collect_embeddings(
            model, pgd_val, device, "Embedding PGD validation")
        clean_candidates = calculate_candidate_scores(
            clean_val, state, cfg, device)
        pgd_candidates = calculate_candidate_scores(
            adversarial_val, state, cfg, device)
        selection = choose_baselines(clean_candidates, pgd_candidates, cfg)
        bundle = {
            "geometry_baseline_version": cfg.version,
            "dataset": dataset, "model": model_name,
            "data_protocol_version": DATA_PROTOCOL_VERSION,
            "state": state, "selection": selection,
        }
        temporary = bundle_path.with_suffix(".tmp")
        torch.save(bundle, temporary)
        temporary.replace(bundle_path)
    state, selection = bundle["state"], bundle["selection"]

    clean_test, test_labels = collect_embeddings(
        model, test, device, "Embedding clean test")
    clean_scores = calculate_candidate_scores(
        clean_test, state, cfg, device, selection=selection)
    results = {name: {"standard": None, "cross_attack": {}}
               for name in BASELINE_NAMES}

    for attack_name in Attacks.ALL:
        cached = {}
        for baseline in BASELINE_NAMES:
            key = str(root / "results" / attack_name / baseline)
            value = result_store.get(key)
            if (value is not None
                    and value.get("geometry_baseline_version") == cfg.version
                    and value.get("data_protocol_version")
                    == DATA_PROTOCOL_VERSION):
                cached[baseline] = value
        if len(cached) == len(BASELINE_NAMES):
            attack_results = cached
        else:
            set_global_seed()
            attacker = create_attack(model, tokenizer, class_names, device,
                                     attack_name, dataset_name=dataset)
            cache_path = (Path(output_root) / Detectors.ENERGY_BASED / attack_name
                          / dataset / model_name / "cache" / "test_adv.pt")
            adversarial_loader = build_adv_loader(
                test, attacker, str(cache_path), device, True)
            adversarial_test, adversarial_labels = collect_embeddings(
                model, adversarial_loader, device,
                f"Embedding {attack_name} test")
            if not torch.equal(test_labels, adversarial_labels):
                raise ValueError("Clean and adversarial test labels are not aligned")
            adversarial_scores = calculate_candidate_scores(
                adversarial_test, state, cfg, device, selection=selection)
            classification = classification_metrics(
                clean_test, adversarial_test, test_labels, text_features)
            attack_results = {}
            for baseline in BASELINE_NAMES:
                selected = selection[baseline]
                score_key = selected["score_key"]
                result = detection_metrics(
                    clean_scores[score_key], adversarial_scores[score_key],
                    selected["threshold"])
                result.update(classification)
                result.update({
                    "geometry_baseline_version": cfg.version,
                    "training_attack": Attacks.PGD,
                    "data_protocol_version": DATA_PROTOCOL_VERSION,
                    "threshold": selected["threshold"],
                    "selected": {key: value for key, value in selected.items()
                                 if key not in {"score_key", "threshold"}},
                })
                attack_results[baseline] = result
                torch.save({
                    "clean": clean_scores[score_key],
                    "adversarial": adversarial_scores[score_key],
                    "threshold": selected["threshold"],
                }, root / f"{attack_name}_{baseline}_scores.pt")
                result_store.put(
                    str(root / "results" / attack_name / baseline), result)
            release_memory()
        for baseline, result in attack_results.items():
            if attack_name == Attacks.PGD:
                results[baseline]["standard"] = result
            else:
                results[baseline]["cross_attack"][attack_name] = result
    return {"baselines": results, "config": {
        **asdict(cfg), "training_attack": Attacks.PGD,
        "evaluation_attacks": list(Attacks.ALL),
        "data_protocol_version": DATA_PROTOCOL_VERSION,
    }}

