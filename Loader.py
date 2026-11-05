"""
loaders.py
----------

Dataset loaders for the VLM adversarial detection pipeline.

Supports:
    • CIFAR-10
    • STL-10

All datasets return:
    train_loader, val_loader, test_loader, class_names
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from Constants import Datasets
import os
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset


# TorchVision still points STL-10 at an HTTP endpoint. Some compute servers
# reject outbound port 80 even though the dataset's official HTTPS host works.
# Keep TorchVision's integrity checks and extraction logic, but use HTTPS.
STL10_HTTPS_URL = (
    "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
)


def fix_tiny_imagenet_val(dataset_dir):
    val_dir = os.path.join(dataset_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotation_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(images_dir):
        return

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        img, cls, *_ = line.strip().split("\t")

        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        src = os.path.join(images_dir, img)
        dst = os.path.join(cls_dir, img)

        if os.path.exists(src):
            os.replace(src, dst)

    os.rmdir(images_dir)


def ensure_tiny_imagenet(data_root: str):
    """Download once; repair interrupted extraction and validation reorganization."""
    from pathlib import Path
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "tiny-imagenet-200"
    zip_path = root / "tiny-imagenet-200.zip"
    marker = dataset_dir / ".complete"
    if marker.exists():
        return str(dataset_dir)
    # Recognize a complete existing local copy before attempting network access.
    if (dataset_dir / "words.txt").exists():
        if (dataset_dir / "val" / "val_annotations.txt").exists():
            fix_tiny_imagenet_val(str(dataset_dir))
        train_count = sum(1 for _ in (dataset_dir / "train").rglob("*.JPEG"))
        val_count = sum(1 for _ in (dataset_dir / "val").rglob("*.JPEG"))
        if train_count == 100000 and val_count == 10000:
            marker.touch()
            return str(dataset_dir)
    if not zip_path.exists():
        temporary = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(
            "https://cs231n.stanford.edu/tiny-imagenet-200.zip", str(temporary))
        temporary.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt dataset archive member: {bad}; remove {zip_path} and retry")
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Dataset archive contains an unsafe path")
        archive.extractall(root)
    fix_tiny_imagenet_val(str(dataset_dir))
    if (sum(1 for _ in (dataset_dir / "train").rglob("*.JPEG")) != 100000 or
            sum(1 for _ in (dataset_dir / "val").rglob("*.JPEG")) != 10000):
        raise RuntimeError("Incomplete Tiny-ImageNet extraction")
    marker.touch()
    return str(dataset_dir)


def tiny_imagenet_class_names(dataset_dir: str, class_ids):
    """Map Tiny-ImageNet WordNet ids to readable zero-shot class prompts."""
    words_path = os.path.join(dataset_dir, "words.txt")
    names = {}
    with open(words_path, "r", encoding="utf-8") as handle:
        for line in handle:
            class_id, descriptions = line.rstrip("\n").split("\t", maxsplit=1)
            names[class_id] = descriptions.split(",", maxsplit=1)[0].strip()
    return [names.get(class_id, class_id) for class_id in class_ids]


def load_dataset(
    dataset_name: str,
    preprocess,
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    split_seed: int = 42,
):
    """
    Returns:
        train_loader, val_loader, test_loader, class_names
    """

    Datasets.validate(dataset_name=dataset_name)

    generator = torch.Generator().manual_seed(split_seed)

    # =========================================================
    # CIFAR-10
    # =========================================================
    if dataset_name == Datasets.CIFAR_10:

        train_part = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=preprocess,
        )

        test_part = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=preprocess,
        )

        full_ds = torch.utils.data.ConcatDataset([train_part, test_part])
        class_names = train_part.classes

        N = len(full_ds)

        train_size = int(0.70 * N)
        val_size = int(0.15 * N)
        test_size = N - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [train_size, val_size, test_size],
            generator=generator,
        )

    # =========================================================
    # STL-10
    # =========================================================
    elif dataset_name == Datasets.STL_10:

        torchvision.datasets.STL10.url = STL10_HTTPS_URL

        train_part = torchvision.datasets.STL10(
            root=data_root,
            split="train",
            download=True,
            transform=preprocess,
        )

        test_part = torchvision.datasets.STL10(
            root=data_root,
            split="test",
            download=True,
            transform=preprocess,
        )

        full_ds = torch.utils.data.ConcatDataset([train_part, test_part])

        class_names = [
            "airplane",
            "bird",
            "car",
            "cat",
            "deer",
            "dog",
            "horse",
            "monkey",
            "ship",
            "truck",
        ]

        N = len(full_ds)

        train_size = int(0.70 * N)
        val_size = int(0.15 * N)
        test_size = N - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [train_size, val_size, test_size],
            generator=generator,
        )

    # =========================================================
    # Tiny-ImageNet
    # =========================================================
    elif dataset_name == Datasets.TINY_IMAGENET:

        dataset_dir = ensure_tiny_imagenet(data_root)

        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")

        train_part = ImageFolder(train_dir, transform=preprocess)
        val_part = ImageFolder(val_dir, transform=preprocess)

        full_ds = ConcatDataset([train_part, val_part])
        class_names = tiny_imagenet_class_names(dataset_dir, train_part.classes)

        N = len(full_ds)

        train_size = int(0.70 * N)
        val_size = int(0.15 * N)
        test_size = N - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [train_size, val_size, test_size],
            generator=generator,
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # =========================================================
    # DataLoaders
    # =========================================================
    # Keep deterministic sample order so each clean image remains paired with
    # the corresponding cached adversarial image in the margin loss.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_names
