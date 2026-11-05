"""Fast, offline checks for the public reproduction pipeline."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset


class ReproductionPlanTests(unittest.TestCase):
    def test_complete_plan_has_every_reported_setting(self):
        import reproduce

        with patch.dict(os.environ, {"REPRODUCTION_EXPERIMENT_MODE": "full"}):
            jobs = reproduce.experiment_plan()
        standard = [job for job in jobs if job["kind"] == "standard"]
        geometry = [job for job in jobs if job["kind"] == "geometry_baselines"]

        self.assertEqual(reproduce.SEEDS, (44, 43, 42))
        self.assertEqual(len(jobs), 197)
        self.assertEqual(len(standard), 162)
        self.assertEqual(len(geometry), 27)
        self.assertEqual({job["seed"] for job in standard}, {42, 43, 44})
        self.assertEqual(
            {job["attack"] for job in standard if job["detector"] == "msp"},
            {"pgd", "mifgsm", "flow", "autoattack", "apgd"},
        )
        self.assertTrue(
            all(job["attack"] == "pgd" and job["cross_attack"]
                for job in standard if job["detector"] == "response_profile")
        )
        self.assertTrue(
            all(job["data_protocol_version"] == reproduce.DATA_PROTOCOL_VERSION
                for job in standard + geometry)
        )

    def test_seed_shards_preserve_canonical_reverse_order(self):
        import reproduce

        with patch.dict(os.environ, {"REPRODUCTION_SEEDS": "44,42"}):
            self.assertEqual(reproduce.configured_seeds(), (44, 42))
        with patch.dict(os.environ, {"REPRODUCTION_SEEDS": "42,44"}):
            with self.assertRaises(ValueError):
                reproduce.configured_seeds()


class FullDatasetTests(unittest.TestCase):
    def test_all_loaders_use_complete_dataset_pools(self):
        import Loader

        class FakeDataset(TensorDataset):
            classes = ["a", "b"]

            def __init__(self, *args, **kwargs):
                super().__init__(
                    torch.zeros(50, 3, 2, 2),
                    torch.zeros(50, dtype=torch.long),
                )

        with patch.object(Loader.torchvision.datasets, "CIFAR10",
                          side_effect=FakeDataset), \
             patch.object(Loader.torchvision.datasets, "STL10",
                          side_effect=FakeDataset), \
             patch.object(Loader, "ImageFolder", side_effect=FakeDataset), \
             patch.object(Loader, "ensure_tiny_imagenet", return_value="unused"), \
             patch.object(Loader, "tiny_imagenet_class_names",
                          return_value=["a", "b"]):
            for dataset in ("cifar_10", "stl_10", "tiny_imagenet"):
                loaders = Loader.load_dataset(dataset, None, split_seed=44)[:3]
                self.assertEqual(
                    [len(loader.dataset) for loader in loaders], [70, 15, 15])
                indices = [set(loader.dataset.indices) for loader in loaders]
                self.assertEqual(len(set.union(*indices)), 100)
                self.assertFalse(indices[0] & indices[1])
                self.assertFalse(indices[0] & indices[2])
                self.assertFalse(indices[1] & indices[2])

    def test_fractional_dataset_arguments_are_not_supported(self):
        import inspect
        import Loader

        self.assertNotIn("sample_fraction", inspect.signature(
            Loader.load_dataset).parameters)
        self.assertNotIn("REPRODUCTION_DATA_FRACTION", Path(
            "Utils.py").read_text(encoding="utf-8"))


class ResultStoreTests(unittest.TestCase):
    def test_atomic_backup_recovers_last_valid_document(self):
        from result_store import ResultStore

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.json"
            store = ResultStore(path)
            store.data["status"] = "first"
            store.save()
            store.data["status"] = "second"
            store.save()
            path.write_text("{interrupted", encoding="utf-8")
            recovered = ResultStore(path)
            self.assertEqual(recovered.data["status"], "first")
            self.assertEqual(
                json.loads((Path(folder) / "results.backup.json").read_text())["status"],
                "first",
            )


if __name__ == "__main__":
    unittest.main()
