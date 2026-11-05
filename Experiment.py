import os
import json
import torch
import open_clip
from Detection.Pretraining.Anchors import AnchorBank
from Detection.Pretraining.PromptGraph import PromptGraph
from Detection.Pretraining.ARFStats import ARFStats
from Detection.Training.MSA import MSAComputer
from Detection.Training.FeatureExtractor import FeatureExtractor
from Utils import (
    build_adv_loader,
    load_dataset_deterministic,
    save_detector_bundle,
    create_attack,
    build_text_features,
    load_detector_bundle,
    release_memory,
    create_model_and_transforms_cached,
)
from Constants import Attacks, DATA_PROTOCOL_VERSION, Detectors


def run_experiment(
    *,
    save_dir,
    adversarial_save_dir,
    dataset_name,
    batch_size,
    attack_name,
    model_name,
    pretrained_weights,
    detector_name,
    device: torch.device,
    force_attack: bool,
    force_train: bool,
    evaluate_transferability: bool,
    output_root: str,
    detector_source_dir: str | None = None,
):
    from Utils import set_global_seed, SEED
    set_global_seed()
    Detectors.validate(detector_name=detector_name)

    figures_dir = os.path.join(save_dir, "figures")
    cache_dir = os.path.join(adversarial_save_dir, "cache")
    plots_dir = os.path.join(save_dir, "detector_plots")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    # =========================================================
    # CLIP
    # =========================================================

    model, _, preprocess = create_model_and_transforms_cached(
        model_name,
        pretrained=pretrained_weights,
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # =========================================================
    # Anchors + Graph
    # =========================================================

    anchors = None
    graph = None
    if detector_name == Detectors.ENERGY_BASED:
        anchors = AnchorBank(model, tokenizer, device=device)
        anchors.compute_embeddings()

        graph = PromptGraph(
            anchors,
            device=device,
            k=8,
            laplacian_type="unnormalized",
            weight_type="cosine",
        )

        graph.laplacian()

    # =========================================================
    # Dataset
    # =========================================================

    train_clean_loader, val_clean_loader, test_clean_loader, class_names = (
        load_dataset_deterministic(
            dataset_name=dataset_name,
            preprocess=preprocess,
            batch_size=batch_size,
        )
    )
    with open(os.path.join(save_dir, "splits.json"), "w", encoding="utf-8") as handle:
        json.dump({name: list(map(int, loader.dataset.indices)) for name, loader in
                   [("train", train_clean_loader), ("val", val_clean_loader),
                    ("test", test_clean_loader)]}, handle)
    with open(os.path.join(save_dir, "configuration.json"), "w", encoding="utf-8") as handle:
        json.dump(dict(dataset=dataset_name, model=model_name, pretrained=pretrained_weights,
                       attack=attack_name, detector=detector_name, batch_size=batch_size,
                       data_protocol_version=DATA_PROTOCOL_VERSION,
                       train_samples=len(train_clean_loader.dataset),
                       validation_samples=len(val_clean_loader.dataset),
                       evaluation_samples=len(test_clean_loader.dataset),
                       detector_source_dir=detector_source_dir,
                       seed=SEED, prompts=(anchors.categories if anchors is not None else
                                           [f"a photo of a {name}." for name in class_names]),
                       preprocessing=str(preprocess)), handle, indent=2)

    # Decide reuse before fitting ARF statistics. A valid detector bundle already
    # contains the fitted ARF state, so recomputing it on every resumed launch is
    # both expensive and unnecessary.
    bundle_dir = detector_source_dir or save_dir
    bundle_path = os.path.join(bundle_dir, "detector_bundle.pt")
    detector_exists = os.path.exists(bundle_path)
    use_saved_detector = detector_exists and not force_train
    expected_training_attack = (
        Attacks.PGD if detector_source_dir is not None else attack_name)
    if use_saved_detector:
        saved = torch.load(bundle_path, map_location="cpu")
        saved_meta = saved.get("meta", {})
        saved_cfg = saved.get("meta", {}).get("extra_cfg", {})
        identity_matches = (
            saved_meta.get("dataset_name") == dataset_name
            and saved_meta.get("clip_model_name") == model_name
            and int(saved_meta.get("seed", -1)) == int(SEED)
            and saved_meta.get("attack_name") == expected_training_attack
        )
        if detector_name == Detectors.MSP:
            use_saved_detector = (
                identity_matches
                and
                saved_cfg.get("msp_score_version") == 2
                and saved_cfg.get("data_protocol_version") == DATA_PROTOCOL_VERSION
            )
        else:
            use_saved_detector = (
                identity_matches
                and
                saved_cfg.get("response_method_version") == 3
                and saved_cfg.get("data_protocol_version") == DATA_PROTOCOL_VERSION
            )
    if detector_source_dir is not None and not use_saved_detector:
        raise ValueError(
            "The requested PGD detector bundle is missing or incompatible: "
            f"{bundle_path}")

    # =========================================================
    # ARF + Feature extractor
    # =========================================================

    arf = None
    fe = None
    if detector_name == Detectors.ENERGY_BASED:
        arf = ARFStats(model=model, anchor_bank=anchors, device=device)
        if not use_saved_detector:
            arf.fit(train_clean_loader)

        msa = MSAComputer(device=device, anchor_bank=anchors)

        fe = FeatureExtractor(
            model=model,
            anchor_bank=anchors,
            prompt_graph=graph,
            arf_stats=arf,
            msa_computer=msa,
            device=device,
        )

    # =========================================================
    # Attack
    # =========================================================

    attacker = create_attack(
        model,
        tokenizer,
        class_names,
        device,
        attack_name,
        dataset_name=dataset_name,
    )

    print(f"\n🛡 Using attack: {attack_name.upper()}")

    text_features = build_text_features(model, tokenizer, class_names, device)

    # =========================================================
    # Detector reuse
    # =========================================================

    if use_saved_detector:

        print("📦 Loading saved detector")

        if detector_name == Detectors.ENERGY_BASED:
            arf = ARFStats(model=model, anchor_bank=anchors, device=device)

        if detector_name == Detectors.ENERGY_BASED:
            from Detection.Training.TrainDetectorEnergy import EnergyDetector

            detector = EnergyDetector(fe.feature_dim).to(device)

        elif detector_name == Detectors.MSP:
            from Detection.Training.TrainDetectorMSP import MSPDetector  # add import

            detector = MSPDetector().to(device)

        detector, best_threshold, clean_mean, clean_cov_inv_sqrt, arf, meta = (
            load_detector_bundle(
                save_dir=bundle_dir,
                detector=detector,
                arf=arf,
                device=device,
            )
        )
        score_mode = meta.get("extra_cfg", {}).get("score_mode", "low_confidence")

    else:
        from Detection.Training.TrainDetectorEnergy import (
            EnergyTrainDetector,
            EnergyTrainConfig,
        )

        if detector_name == Detectors.ENERGY_BASED:

            trainer = EnergyTrainDetector(
                feature_extractor=fe,
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=EnergyTrainConfig(device=device),
            )

        elif detector_name == Detectors.MSP:
            from Detection.Training.TrainDetectorMSP import (
                MSPTrainDetector,
                TrainConfig as MSPTrainConfig,
            )

            trainer = MSPTrainDetector(
                feature_extractor=fe,
                model=model,
                tokenizer=tokenizer,
                class_names=class_names,
                device=device,
                cfg=MSPTrainConfig(device=device, temperature=1.0),
            )

        train_adv_loader = None
        if detector_name == Detectors.ENERGY_BASED:
            train_cache = os.path.join(cache_dir, "train_adv.pt")
            train_adv_loader = build_adv_loader(
                train_clean_loader,
                attacker,
                cache_path=train_cache,
                device=device,
                use_cached_attack=not force_attack,
            )

        val_cache = os.path.join(cache_dir, "val_adv.pt")
        val_adv_loader = build_adv_loader(
            val_clean_loader,
            attacker,
            cache_path=val_cache,
            device=device,
            use_cached_attack=not force_attack,
        )

        best_threshold = trainer.fit(
            train_clean_loader=train_clean_loader,
            val_clean_loader=val_clean_loader,
            train_adv_loader=train_adv_loader,
            val_adv_loader=val_adv_loader,
            save_dir=plots_dir,
        )

        detector = trainer.detector
        score_mode = getattr(trainer, "score_mode", "low_confidence")

        clean_mean = getattr(trainer, "clean_mean", None)
        clean_cov_inv_sqrt = getattr(trainer, "clean_cov_inv_sqrt", None)

        # =========================================================
        # Save trained detector
        # =========================================================

        save_detector_bundle(
            save_dir=save_dir,
            detector=detector,
            threshold=best_threshold,
            class_names=class_names,
            model_name=model_name,
            pretrained=pretrained_weights,
            dataset_name=dataset_name,
            clean_mean=clean_mean,
            clean_cov_inv_sqrt=clean_cov_inv_sqrt,
            arf=arf,
            extra_cfg={
                "attack_used": attack_name,
                "detector_name": detector_name,
                "data_protocol_version": DATA_PROTOCOL_VERSION,
                **({"response_method_version": 3}
                   if detector_name == Detectors.ENERGY_BASED else {}),
                **({
                    "msp_score_version": 2,
                    "score_mode": score_mode,
                    "validation_base_auc": trainer.validation_base_auc,
                } if detector_name == Detectors.MSP else {}),
            },
        )

    if fe is not None:
        fe.arf_stats = arf

    # Freeze the fitted detector for evaluation.  In the paper's standard and
    # cross-attack settings, adversarial examples optimize only the frozen
    # VLM's zero-shot classification logits.
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)

    @torch.no_grad()
    def attack_success_metrics(clean_loader, adversarial_loader):
        clean_correct = 0
        adversarial_correct = 0
        successful_from_correct = 0
        initially_correct = 0
        total = 0
        for (clean, labels), (adversarial, adversarial_labels) in zip(
                clean_loader, adversarial_loader):
            if not torch.equal(labels, adversarial_labels):
                raise ValueError("Clean/adversarial labels are not aligned")
            labels = labels.to(device)
            def predict(images):
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    encoded = model.encode_image(images.to(device))
                encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
                logits = encoded @ text_features.float().T
                if hasattr(model, "logit_scale"):
                    logits = logits * model.logit_scale.exp().float()
                return logits.argmax(1)
            clean_prediction = predict(clean)
            adversarial_prediction = predict(adversarial)
            correct = clean_prediction.eq(labels)
            clean_correct += int(correct.sum())
            adversarial_correct += int(adversarial_prediction.eq(labels).sum())
            initially_correct += int(correct.sum())
            successful_from_correct += int((correct & adversarial_prediction.ne(labels)).sum())
            total += len(labels)
        return {
            "clean_classification_accuracy": clean_correct / max(total, 1),
            "adversarial_classification_accuracy": adversarial_correct / max(total, 1),
            "attack_success_rate": successful_from_correct / max(initially_correct, 1),
        }

    # =========================================================
    # Evaluation (always runs)
    # =========================================================

    release_memory()

    test_cache = os.path.join(cache_dir, "test_adv.pt")
    test_adv_loader = build_adv_loader(
        test_clean_loader,
        attacker,
        cache_path=test_cache,
        device=device,
        use_cached_attack=not force_attack,
    )

    if detector_name == Detectors.MSP:
        from Detection.Inference.EvaluateMSPDetector import EvaluateMSPDetector

        evaluator = EvaluateMSPDetector(
            detector=detector,
            model=model,
            tokenizer=tokenizer,
            class_names=class_names,
            threshold=best_threshold,
            device=device,
            score_mode=score_mode,
        )
    else:
        from Detection.Inference.EvaluateDetector import EvaluateEnergyDetector

        evaluator = EvaluateEnergyDetector(
            detector=detector,
            feature_extractor=fe,
            threshold=best_threshold,
            clean_mean=clean_mean,
            clean_cov_inv_sqrt=clean_cov_inv_sqrt,
            device=device,
        )

    results = evaluator.evaluate(
        clean_loader=test_clean_loader, adv_loader=test_adv_loader, save_dir=save_dir
    )
    results.update(attack_success_metrics(test_clean_loader, test_adv_loader))
    results["data_protocol_version"] = DATA_PROTOCOL_VERSION
    results["evaluation_samples"] = len(test_clean_loader.dataset)
    results["detector_training_attack"] = expected_training_attack
    import result_store
    result_store.put(save_dir, results)

    print("\n📊 Final Test Results:", results)
    transfer_results = {}



    # =========================================================
    # 🔄 Transferability Evaluation (Energy ONLY)
    # =========================================================

    if evaluate_transferability and detector_name == Detectors.ENERGY_BASED:

        print("\n🔄 Transferability Evaluation Enabled")

        transfer_results = {}
        other_attacks = [item for item in Attacks.ALL if item != attack_name]


        for other_attack in other_attacks:
            print(f"\n➡️ Evaluating on transfer attack: {other_attack}")

            # Create attacker
            transfer_attacker = create_attack(
                model,
                tokenizer,
                class_names,
                device,
                other_attack,
                dataset_name=dataset_name,
            )

            # Share the ordinary attack cache with the corresponding standard
            # experiment.  The detector remains the PGD-trained detector.
            transfer_cache_dir = os.path.join(
                output_root, Detectors.ENERGY_BASED, other_attack,
                dataset_name, model_name, "cache"
            )
            os.makedirs(transfer_cache_dir, exist_ok=True)
            transfer_test_cache = os.path.join(transfer_cache_dir, "test_adv.pt")

            # Build adversarial loader
            transfer_adv_loader = build_adv_loader(
                test_clean_loader,
                transfer_attacker,
                cache_path=transfer_test_cache,
                device=device,
                use_cached_attack=not force_attack,
            )

            # Evaluate
            transfer_result = evaluator.evaluate(
                clean_loader=test_clean_loader,
                adv_loader=transfer_adv_loader,
                save_dir=os.path.join(save_dir, f"transfer_{other_attack}"),
            )
            transfer_result.update(
                attack_success_metrics(test_clean_loader, transfer_adv_loader)
            )
            transfer_result["data_protocol_version"] = DATA_PROTOCOL_VERSION
            transfer_result["evaluation_samples"] = len(test_clean_loader.dataset)
            transfer_result["detector_training_attack"] = expected_training_attack
            result_store.put(os.path.join(save_dir, f"transfer_{other_attack}"), transfer_result)

            transfer_results[other_attack] = transfer_result

            print(f"📊 Transfer Result ({other_attack}):", transfer_result)
            transfer_results_file = os.path.join(save_dir, "transfer_results.txt")

            # ✅ Save each transfer result
            with open(transfer_results_file, "a") as f:
                f.write("\n------------------------------\n")
                f.write(f"Transfer Attack: {other_attack}\n")
                f.write(f"Results: {transfer_result}\n")

            release_memory()

        print("\n🔥 Transferability Summary:")
        for k, v in transfer_results.items():
            print(f"{k}: {v}")

        with open(transfer_results_file, "a") as f:
            f.write("\n==============================\n")
            f.write("TRANSFERABILITY SUMMARY\n")
            f.write("==============================\n")

            for k, v in transfer_results.items():
                f.write(f"{k}: {v}\n")

    return {"standard": results, "cross_attack": transfer_results}
