"""
Constants.py
------------

Centralized identifiers for datasets, models, and attacks
used in the VLM adversarial detection experiments.

This ensures reproducibility and prevents typos.
"""

# =========================================================
# DATASETS
# =========================================================
DATA_PROTOCOL_VERSION = 1

class Datasets:
    CIFAR_10 = "cifar_10"
    STL_10 = "stl_10"
    TINY_IMAGENET = "tiny_imagenet"

    @staticmethod
    def validate(dataset_name: str):
        valid = [
            Datasets.CIFAR_10,
            Datasets.STL_10,
            Datasets.TINY_IMAGENET,
        ]
        if dataset_name not in valid:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    @staticmethod
    def get_config(dataset_name: str):
        Datasets.validate(dataset_name=dataset_name)

        if dataset_name == Datasets.CIFAR_10:
            return dict(name=Datasets.CIFAR_10, batch_size=32)

        elif dataset_name == Datasets.STL_10:
            return dict(name=Datasets.STL_10, batch_size=32)

        elif dataset_name == Datasets.TINY_IMAGENET:
            return dict(name=Datasets.TINY_IMAGENET, batch_size=64)

# =========================================================
# MODELS (VLM / CLIP-based)
# =========================================================


class Models:
    # Core CLIP models
    VIT_B32 = "ViT-B-32"
    VIT_B16 = "ViT-B-16"
    CONVNEXT_BASE_W = "convnext_base_w"

    @staticmethod
    def validate(model_name: str):
        valid = [
            Models.VIT_B32,
            Models.VIT_B16,
            Models.CONVNEXT_BASE_W,
        ]
        if model_name not in valid:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def get_config(model_name: str):
        Models.validate(model_name=model_name)
        if model_name == Models.VIT_B32:
            return dict(name=Models.VIT_B32, pretrained="laion2b_s34b_b79k")
        elif model_name == Models.VIT_B16:
            return dict(name=Models.VIT_B16, pretrained="laion2b_s34b_b88k")
        elif model_name == Models.CONVNEXT_BASE_W:
            return dict(name=Models.CONVNEXT_BASE_W, pretrained="laion2b_s13b_b82k_augreg")
        return None

# =========================================================
# DETECTORS
# =========================================================

class Detectors:
    # Historical constant name retained for import compatibility.
    ENERGY_BASED = "response_profile"
    MSP = "msp"

    @staticmethod
    def validate(detector_name: str):
        valid = [
            Detectors.ENERGY_BASED,
            Detectors.MSP,
        ]
        if detector_name not in valid:
            raise ValueError(f"Unsupported detector: {detector_name}")

            
class Attacks:
    MIFGSM = "mifgsm"
    PGD = "pgd"
    AUTOATTACK = "autoattack"
    APGD = "apgd"
    FLOW = "flow"

    ALL = [PGD, MIFGSM, FLOW, AUTOATTACK, APGD]

    @staticmethod
    def validate(attack_name: str):
        if attack_name not in Attacks.ALL:
            raise ValueError(f"Unsupported attack: {attack_name}")
