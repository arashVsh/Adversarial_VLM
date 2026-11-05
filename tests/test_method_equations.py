"""Numerical checks for the reported 43-dimensional response profile."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Detection.Pretraining.ARFStats import ARFStats
from Detection.Pretraining.PromptGraph import PromptGraph
from Detection.Training.APGDAttack import APGDAttack, APGDConfig
from Detection.Training.FeatureExtractor import FeatureExtractor
from Detection.Training.MSA import MSAComputer
from Detection.Utils.ShardCache import build_sharded_loader
from Utils import denormalize_clip, normalize_clip, set_global_seed


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        mean = images.mean((2, 3))
        energy = images.square().mean((1, 2, 3))[:, None] + 1
        return torch.cat((mean, energy), 1) * self.scale

    def encode_text(self, tokens):
        return F.one_hot(tokens % 4, 4).float()


def fixture():
    set_global_seed(42)
    model = TinyModel()
    prompts = F.normalize(torch.rand(70, 4) + 0.2, dim=1)
    categories = list("abcdefg")
    anchors = SimpleNamespace(
        category_names=categories,
        category_to_indices={
            category: list(range(10 * index, 10 * index + 10))
            for index, category in enumerate(categories)
        },
        get_prompt_embeddings=lambda: prompts,
    )
    images = normalize_clip(torch.rand(6, 3, 8, 8))
    loader = DataLoader(
        TensorDataset(images, torch.arange(6) % 2), batch_size=3)
    device = torch.device("cpu")
    arf = ARFStats(model, anchors, device)
    arf.fit(loader, verbose=False)
    graph = PromptGraph(anchors, device)
    msa = MSAComputer(device, anchors)
    extractor = FeatureExtractor(model, anchors, graph, device, arf, msa)
    return model, anchors, images, loader, arf, graph, msa, extractor


class MethodEquationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_feature_vector_has_reported_dimension_and_finite_values(self):
        *_, images, _, _, _, _, extractor = fixture()
        features = extractor.phi(images)
        self.assertEqual(features.shape, (6, 43))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all())

    def test_residual_statistics_match_shrunk_clean_covariance(self):
        model, anchors, images, _, arf, *_ = fixture()
        responses = (F.normalize(model.encode_image(images), dim=1)
                     @ anchors.get_prompt_embeddings().T)
        for category, indices in anchors.category_to_indices.items():
            values = responses[:, indices]
            mean = values.mean(0)
            covariance = (values - mean).T @ (values - mean) / len(values)
            expected = (0.8 * covariance
                        + 0.2 * torch.diag(covariance.diag())
                        + 0.001 * torch.eye(10))
            torch.testing.assert_close(arf.mu[category], mean)
            torch.testing.assert_close(
                arf.cov[category], expected, atol=1e-6, rtol=1e-4)

    def test_resumable_attack_cache_reuses_complete_shards(self):
        class Attack:
            cfg = "fixed"
            calls = 0

            def generate(self, images, labels):
                self.calls += 1
                return images + 1

        attack = Attack()
        loader = DataLoader(
            TensorDataset(torch.zeros(5, 1), torch.arange(5)), batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            cache = str(Path(folder) / "attack.pt")
            first = build_sharded_loader(
                loader, attack, cache, torch.device("cpu"), True)
            second = build_sharded_loader(
                loader, attack, cache, torch.device("cpu"), True)
            self.assertEqual(attack.calls, 3)
            torch.testing.assert_close(
                torch.cat([images for images, _ in first]),
                torch.cat([images for images, _ in second]),
            )

    def test_apgd_respects_pixel_space_budget(self):
        model = TinyModel()
        tokenizer = lambda prompts: torch.arange(len(prompts))
        attack = APGDAttack(
            model, tokenizer, ["zero", "one"], torch.device("cpu"),
            APGDConfig(eps=0.05, steps=2, restarts=1, seed=7),
        )
        pixels = torch.rand(2, 3, 8, 8) * 0.8 + 0.1
        images = normalize_clip(pixels)
        adversarial = attack.generate(images, torch.tensor([0, 1]))
        delta = (denormalize_clip(adversarial)
                 - denormalize_clip(images)).abs().max().item()
        self.assertLessEqual(delta, 0.05001)


if __name__ == "__main__":
    unittest.main()
