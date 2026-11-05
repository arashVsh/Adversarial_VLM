#!/usr/bin/env python
"""Command-line runner for standard and cross-attack experiments."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any

from Constants import Attacks, Datasets, Detectors, Models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run response-profile adversarial-image detection experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=[Datasets.CIFAR_10, Datasets.STL_10, Datasets.TINY_IMAGENET],
        default=Datasets.CIFAR_10,
    )
    parser.add_argument(
        "--model",
        choices=[Models.VIT_B16, Models.VIT_B32, Models.CONVNEXT_BASE_W],
        default=Models.VIT_B16,
    )
    parser.add_argument("--attack", choices=Attacks.ALL, default=Attacks.PGD)
    parser.add_argument(
        "--detector",
        choices=[Detectors.ENERGY_BASED, Detectors.MSP],
        default=Detectors.ENERGY_BASED,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root for generated caches, checkpoints, plots, and metrics.",
    )
    parser.add_argument(
        "--all-standard",
        action="store_true",
        help="Run the shared-PGD-detector protocol across every dataset/backbone.",
    )
    parser.add_argument(
        "--cross-attack",
        action="store_true",
        help="Evaluate each PGD-trained detector on the four unseen scheduled attacks.",
    )
    parser.add_argument(
        "--reuse-attacks",
        action="store_true",
        help="Reuse adversarial-example caches when present.",
    )
    parser.add_argument(
        "--reuse-detector",
        action="store_true",
        help="Reuse a trained detector bundle when present.",
    )
    args = parser.parse_args()
    if args.cross_attack and args.detector != Detectors.ENERGY_BASED:
        parser.error("--cross-attack is available for the response-profile detector only")
    if args.cross_attack and not args.all_standard and args.attack != Attacks.PGD:
        parser.error("the paper cross-attack protocol requires --attack pgd")
    return args


def prepare_experiment(
    *,
    attack_name: str,
    model_name: str,
    dataset_name: str,
    detector_name: str,
    output_root: str,
    device: Any,
    force_attack: bool,
    force_train: bool,
    evaluate_transferability: bool,
    detector_source_dir: str | None = None,
) -> dict:
    from Experiment import run_experiment

    model_conf = Models.get_config(model_name)
    dataset_conf = Datasets.get_config(dataset_name)
    save_dir = os.path.join(
        output_root, detector_name, attack_name, dataset_name, model_name
    )
    adversarial_save_dir = os.path.join(
        output_root, Detectors.ENERGY_BASED, attack_name, dataset_name, model_name
    )
    exp_name = f"{detector_name} | {dataset_name} | {model_name} | {attack_name}"

    print("\n===================================================")
    print(f"Running experiment: {exp_name}")
    print(f"Output directory: {save_dir}")
    print("===================================================\n")
    start_time = datetime.now()

    result = run_experiment(
        save_dir=save_dir,
        adversarial_save_dir=adversarial_save_dir,
        dataset_name=dataset_name,
        batch_size=dataset_conf["batch_size"],
        attack_name=attack_name,
        model_name=model_name,
        pretrained_weights=model_conf["pretrained"],
        detector_name=detector_name,
        device=device,
        force_attack=force_attack,
        force_train=force_train,
        evaluate_transferability=evaluate_transferability,
        output_root=output_root,
        detector_source_dir=detector_source_dir,
    )
    print(f"Finished {exp_name} in {datetime.now() - start_time}")
    return result


def main() -> None:
    args = parse_args()
    from Utils import get_device, release_memory, set_global_seed

    set_global_seed()
    device = get_device(show_device_info=True)
    output_root = os.path.abspath(args.output_dir)

    if args.all_standard and args.detector == Detectors.ENERGY_BASED:
        attacks = [Attacks.PGD]
    else:
        attacks = Attacks.ALL if args.all_standard else [args.attack]
    models = (
        [Models.VIT_B16, Models.VIT_B32, Models.CONVNEXT_BASE_W]
        if args.all_standard
        else [args.model]
    )
    datasets = (
        [Datasets.CIFAR_10, Datasets.STL_10, Datasets.TINY_IMAGENET]
        if args.all_standard
        else [args.dataset]
    )

    for attack_name in attacks:
        for model_name in models:
            for dataset_name in datasets:
                prepare_experiment(
                    attack_name=attack_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    detector_name=args.detector,
                    output_root=output_root,
                    device=device,
                    force_attack=not args.reuse_attacks,
                    force_train=not args.reuse_detector,
                    evaluate_transferability=(
                        (args.cross_attack or args.all_standard)
                        and args.detector == Detectors.ENERGY_BASED
                        and attack_name == Attacks.PGD
                    ),
                )
                release_memory()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        from reproduce import main as reproduce_all
        reproduce_all()
    else:
        main()
