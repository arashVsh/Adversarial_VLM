import gc
import os
import inspect
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from Loader import load_dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext
from Constants import Attacks, Datasets


SEED = 42
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_MODEL_CACHE_RESTORED = False


def _copy_completed_cache_tree(source, destination):
    """Copy complete cache files atomically, ignoring partial downloads."""
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return 0
    copied = 0
    for path in source.rglob('*'):
        if not path.is_file() or path.name.endswith(
                ('.lock', '.incomplete', '.tmp', '.copying')):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            actual_source = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if target.exists() and target.stat().st_size == actual_source.stat().st_size:
            continue
        temporary = target.with_name(target.name + '.copying')
        shutil.copyfile(actual_source, temporary)
        temporary.replace(target)
        copied += 1
    return copied


def restore_persistent_model_cache():
    """Restore validated pretrained weights from durable storage to local disk."""
    global _MODEL_CACHE_RESTORED
    if _MODEL_CACHE_RESTORED:
        return 0
    _MODEL_CACHE_RESTORED = True
    persistent = os.environ.get('REPRODUCTION_MODEL_CACHE_DIR')
    if not persistent:
        return 0
    copied = 0
    for name, local in (
        ('huggingface', os.environ.get('HF_HOME')),
        ('torch', os.environ.get('TORCH_HOME')),
    ):
        if local:
            copied += _copy_completed_cache_tree(Path(persistent) / name, local)
    if copied:
        print(f'Restored {copied} validated pretrained-cache file(s).', flush=True)
    return copied


def persist_validated_model_cache():
    """Atomically persist weights only after OpenCLIP loaded them successfully."""
    persistent = os.environ.get('REPRODUCTION_MODEL_CACHE_DIR')
    if not persistent:
        return 0
    copied = 0
    for name, local in (
        ('huggingface', os.environ.get('HF_HOME')),
        ('torch', os.environ.get('TORCH_HOME')),
    ):
        if local:
            copied += _copy_completed_cache_tree(local, Path(persistent) / name)
    if copied:
        print(f'Persisted {copied} validated pretrained-cache file(s).', flush=True)
    return copied


def create_model_and_transforms_cached(model_name, pretrained):
    """Load OpenCLIP locally and retain its validated cache for later sessions."""
    import open_clip
    restore_persistent_model_cache()
    result = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    # Reaching this point proves that OpenCLIP/safetensors successfully loaded
    # the checkpoint. Only validated cache files are copied to durable storage.
    persist_validated_model_cache()
    return result


def denormalize_clip(x_norm: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x_norm.device, dtype=x_norm.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(CLIP_STD, device=x_norm.device, dtype=x_norm.dtype).view(
        1, 3, 1, 1
    )
    return (x_norm * std + mean).clamp(0.0, 1.0)


def normalize_clip(x_pixel: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, device=x_pixel.device, dtype=x_pixel.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(CLIP_STD, device=x_pixel.device, dtype=x_pixel.dtype).view(
        1, 3, 1, 1
    )
    return (x_pixel - mean) / std


def amp_context(device: torch.device, enabled=True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def get_device(show_device_info=False) -> torch.device:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        configured = os.environ.get("CUDA_DEVICE_INDEX")
        gpu_index = int(configured) if configured is not None else (1 if gpu_count > 1 else 0)
        if not 0 <= gpu_index < gpu_count:
            raise ValueError(
                f"CUDA_DEVICE_INDEX={gpu_index} is invalid; detected {gpu_count} GPU(s)"
            )
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"🧠 Using device: {device}")

    if show_device_info:
        print(f"🧠 Available CPU cores: {os.cpu_count()}")

        num_gpus = torch.cuda.device_count()
        print(f"🖥️ Number of GPUs available: {num_gpus}")

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    return device


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def collect_scores(
    detector,
    feature_extractor,
    loader,
    device,
    use_amp: bool = True,
):
    scores = []
    for images, _ in loader:
        images = images.to(device)
        with amp_context(device, enabled=use_amp):
            feats = feature_extractor(images)  # shape: [B, D]
            out = detector(feats)  # could be logits or probs

        if out.ndim == 2 and out.shape[1] == 2:
            p_adv = torch.softmax(out, dim=1)[:, 1]
        else:
            p_adv = out.squeeze()
            # if these are logits, uncomment:
            # p_adv = torch.sigmoid(p_adv)

        scores.append(p_adv.detach().float().cpu())
    return torch.cat(scores)


def find_best_threshold_for_accuracy(clean_scores, adv_scores, num_steps=2001):
    scores = torch.cat([clean_scores, adv_scores])
    labels = torch.cat([torch.zeros_like(clean_scores), torch.ones_like(adv_scores)])

    t_min = float(scores.min())
    t_max = float(scores.max())
    thresholds = torch.linspace(t_min, t_max, steps=num_steps)

    best_t, best_acc = None, -1.0
    for t in thresholds:
        preds = (scores >= t).float()
        acc = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def set_global_seed(seed=None):
    global SEED
    if seed is not None:
        SEED = int(seed)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# CLIP zero-shot classifier helpers (correct logits)
# =========================================================
@torch.no_grad()
def build_text_features(
    model, tokenizer, class_names, device, prompt_template="a photo of a {}."
):
    prompts = [prompt_template.format(c.replace("_", " ")) for c in class_names]
    tokens = tokenizer(prompts)

    if isinstance(tokens, torch.Tensor):
        tokens = tokens.to(device)
    else:
        tokens = {k: v.to(device) for k, v in tokens.items()}

    with amp_context(device):
        txt = model.encode_text(tokens)

    txt = F.normalize(txt.float(), dim=-1)
    return txt  # [C, D]


@torch.no_grad()
def clip_logits_from_images(model, images_norm, text_features, use_amp=True):
    """
    images_norm: CLIP-preprocessed tensor [B,3,H,W]
    returns logits [B,C]
    """

    device = images_norm.device

    with amp_context(device, enabled=use_amp):

        img_feat = model.encode_image(images_norm)
        img_feat = F.normalize(img_feat, dim=-1)

        logits = img_feat @ text_features.t()

        if hasattr(model, "logit_scale"):
            try:
                logits = logits * model.logit_scale.exp()
            except Exception:
                pass

    return logits


# =========================================================
# Attack Factory (PGD / CW / AutoAttack)
# =========================================================
def create_attack(model, tokenizer, class_names, device, attack_name, dataset_name):

    # Stronger settings for Tiny-ImageNet
    if dataset_name in [Datasets.TINY_IMAGENET, Datasets.CIFAR_10]:
        eps = 0.04
        alpha = 0.01
        steps = 20
    else:
        eps = 0.03
        alpha = 0.0075
        steps = 10

    match attack_name:

        case Attacks.PGD:
            from Detection.Training.PGDAttack import PGDAttack, PGDConfig

            return PGDAttack(
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=PGDConfig(
                    eps=eps,
                    alpha=alpha,
                    steps=steps,
                    random_start=True,
                ),
            )

        case Attacks.FLOW:
            from Detection.Training.FlowAttack import FlowAttack, FlowAttackConfig

            return FlowAttack(
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=FlowAttackConfig(
                    eps=eps,
                    alpha=alpha,
                    steps=steps,
                    random_start=True,
                    targeted=False,
                    smooth_weight=0.05,
                ),
            )

        case Attacks.AUTOATTACK:
            from Detection.Training.AutoAttackCLIP import (
                AutoAttackCLIP,
                AutoAttackConfig,
            )

            return AutoAttackCLIP(
                dataset_name=dataset_name,
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=AutoAttackConfig(
                    eps=eps,
                    batch_size=32,
                    seed=SEED,
                ),
            )

        case Attacks.APGD:
            from Detection.Training.APGDAttack import APGDAttack, APGDConfig

            return APGDAttack(
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=APGDConfig(
                    eps=eps,
                    steps=100,
                    restarts=1,
                    seed=SEED,
                    loss="ce",
                ),
            )

        case Attacks.MIFGSM:
            from Detection.Training.MIFGSMAttack import MIFGSMAttack, MIFGSMConfig

            return MIFGSMAttack(
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=MIFGSMConfig(
                    eps=eps,
                    alpha=alpha,
                    steps=steps,
                    mu=1.0,
                    targeted=False,
                ),
            )

        case _:
            raise ValueError(f"Unsupported attack: {attack_name}")


# =========================================================
# Attack Success Rate (ASR)
# =========================================================


def compute_attack_success_rate(
    clip_model,
    text_features,
    attacker,
    loader,
    device,
    max_batches: int = 10,
    use_amp: bool = True,
):
    """
    ASR = fraction of samples where zero-shot prediction changes after attack.

    Uses CLIP similarity logits.
    Supports tqdm progress and faster tensor accumulation.
    """

    total = 0
    success = 0

    iterator = loader
    if max_batches is not None:
        iterator = enumerate(loader)
    else:
        iterator = enumerate(loader)

    pbar = tqdm(iterator, total=max_batches, desc="⚔️ Computing ASR", leave=False)

    for b, (images, labels) in pbar:

        if max_batches is not None and b >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---------------- CLEAN PREDICTIONS ----------------
        with torch.no_grad():

            logits_clean = clip_logits_from_images(
                clip_model, images, text_features, use_amp=use_amp
            )

            pred_clean = logits_clean.argmax(dim=1)

        # ---------------- ADVERSARIAL GENERATION ----------------
        with torch.enable_grad():
            x_adv = attacker.generate(images, labels)

        # ---------------- ADVERSARIAL PREDICTIONS ----------------
        with torch.no_grad():

            logits_adv = clip_logits_from_images(
                clip_model, x_adv, text_features, use_amp=use_amp
            )

            pred_adv = logits_adv.argmax(dim=1)

        # ---------------- UPDATE COUNTERS ----------------
        batch_success = (pred_clean != pred_adv).sum()

        success += batch_success.item()
        total += labels.size(0)

        pbar.set_postfix(ASR=f"{(success / total):.4f}" if total > 0 else "0.0")

    return success / total if total > 0 else 0.0


# =========================
# Adversarial cache builder
# =========================
def build_adv_loader(
    loader, attacker, cache_path: str, device: torch.device, use_cached_attack: bool
):

    from Detection.Utils.ShardCache import build_sharded_loader
    return build_sharded_loader(loader, attacker, cache_path, device, use_cached_attack)


# =========================
# Save / Load detector bundle
# =========================
def save_detector_bundle(
    save_dir: str,
    detector,
    threshold: float,
    class_names,
    model_name: str,
    pretrained: str,
    dataset_name,
    clean_mean=None,
    clean_cov_inv_sqrt=None,
    extra_cfg: dict | None = None,
    arf=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if arf is not None and (not hasattr(arf, "state_dict") or not callable(arf.state_dict)):
        raise TypeError("ARF object must implement state_dict() in order to be saved.")

    bundle_path = os.path.join(save_dir, "detector_bundle.pt")

    detector_state = {k: v.detach().cpu() for k, v in detector.state_dict().items()}

    whitening = None
    if clean_mean is not None and clean_cov_inv_sqrt is not None:
        whitening = {
            "clean_mean": clean_mean.detach().cpu(),
            "clean_cov_inv_sqrt": clean_cov_inv_sqrt.detach().cpu(),
        }

    attack_name = extra_cfg.get("attack_used") if isinstance(extra_cfg, dict) else None

    meta = {
        "method_version": 2,
        "threshold": float(threshold),
        "class_names": list(class_names),
        "clip_model_name": model_name,
        "clip_pretrained": pretrained,
        "dataset_name": str(dataset_name),
        "seed": SEED,
        "attack_name": attack_name,
        "extra_cfg": extra_cfg or {},
    }

    bundle = {
        "detector_state_dict": detector_state,
        "whitening": whitening,
        "arf_state_dict": arf.state_dict() if arf is not None else None,
        "meta": meta,
    }

    torch.save(bundle, bundle_path)
    print(f"✅ Detector bundle saved → {bundle_path}")


def load_detector_bundle(
    save_dir: str,
    detector,
    arf,
    device,
):
    """
    Loads the full detector bundle from detector_bundle.pt
    """

    bundle_path = os.path.join(save_dir, "detector_bundle.pt")

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Missing detector bundle: {bundle_path}")

    bundle = torch.load(bundle_path, map_location=device)

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    meta = bundle["meta"]
    if meta.get("method_version") != 2:
        raise ValueError("Incompatible detector bundle; regenerate with the current method")
    threshold = float(meta["threshold"])

    # -------------------------------------------------
    # Detector
    # -------------------------------------------------

    detector.load_state_dict(bundle["detector_state_dict"])
    detector.to(device)
    detector.eval()

    # -------------------------------------------------
    # Whitening
    # -------------------------------------------------

    clean_mean = None
    clean_cov_inv_sqrt = None

    whitening = bundle.get("whitening")

    if whitening is not None:
        clean_mean = whitening["clean_mean"].to(device)
        clean_cov_inv_sqrt = whitening["clean_cov_inv_sqrt"].to(device)

    # -------------------------------------------------
    # ARF
    # -------------------------------------------------

    saved_arf = bundle.get("arf_state_dict")
    if saved_arf is not None:
        if arf is None:
            raise ValueError(
                "An initialized ARFStats object must be passed for a bundle containing ARF statistics."
            )
        arf.load_state_dict(saved_arf)

    print("✅ Detector bundle loaded")
    print(f"Threshold: {threshold}")
    print(f"Whitening mean loaded: {clean_mean is not None}")
    print(f"Whitening matrix loaded: {clean_cov_inv_sqrt is not None}")
    print(f"ARF stats loaded: {saved_arf is not None}")
    print("Experiment metadata:", meta)

    return detector, threshold, clean_mean, clean_cov_inv_sqrt, arf, meta


# =========================
# Deterministic dataset loading (best effort)
# =========================
def load_dataset_deterministic(
    dataset_name,
    preprocess,
    batch_size,
):
    """
    Ensures train/val/test splits are reproducible across runs.
    """
    set_global_seed()

    sig = inspect.signature(load_dataset)
    kwargs = dict(
        dataset_name=dataset_name,
        preprocess=preprocess,
        batch_size=batch_size,
    )

    persistent_data_root = os.environ.get("REPRODUCTION_DATA_DIR")
    if persistent_data_root and "data_root" in sig.parameters:
        kwargs["data_root"] = persistent_data_root

    for k in ["seed", "subset_seed", "split_seed"]:
        if k in sig.parameters:
            kwargs[k] = SEED

    return load_dataset(**kwargs)


# =========================
# Detection helper (score -> threshold)
# =========================
@torch.no_grad()
def detect_is_adv(
    detector,
    feature_extractor,
    images,
    threshold: float,
    device: torch.device,
    use_amp: bool = True,
):
    images = images.to(device)

    with amp_context(device, enabled=use_amp):
        feats = feature_extractor(images)
        out = detector(feats)

    if out.ndim == 2 and out.shape[1] == 2:
        p_adv = torch.softmax(out, dim=1)[:, 1]
    else:
        p_adv = out.squeeze()
        # if your detector outputs logits, uncomment:
        # p_adv = torch.sigmoid(p_adv)

    is_adv = p_adv >= threshold
    return is_adv, p_adv


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def visualize_adversarial_examples(
    clip_model,
    text_features,
    clean_images,
    adv_images,
    labels,
    class_names,
    max_samples,
    device,
    save_path,
    use_amp: bool = True,
):
    """
    Shows clean vs adversarial images with:
      - True class
      - Clean prediction
      - Adversarial prediction
    """

    clean_images = clean_images.detach().cpu()
    adv_images = adv_images.detach().cpu()
    labels = labels.detach().cpu()

    n = min(max_samples, clean_images.size(0))

    logits_clean = clip_logits_from_images(
        clip_model,
        clean_images.to(device),
        text_features,
        use_amp=use_amp,
    ).cpu()

    logits_adv = clip_logits_from_images(
        clip_model,
        adv_images.to(device),
        text_features,
        use_amp=use_amp,
    ).cpu()

    pred_clean = logits_clean.argmax(dim=1)
    pred_adv = logits_adv.argmax(dim=1)

    plt.figure(figsize=(n * 3, 6))

    for i in range(n):
        true_cls = class_names[labels[i]]
        clean_cls = class_names[pred_clean[i]]
        adv_cls = class_names[pred_adv[i]]

        plt.subplot(2, n, i + 1)
        img = clean_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"True: {true_cls}\nClean: {clean_cls}", fontsize=9)

        plt.subplot(2, n, n + i + 1)
        img = adv_images[i].permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Adv: {adv_cls}", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close()
