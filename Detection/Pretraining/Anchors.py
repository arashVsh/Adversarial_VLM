"""
Anchors.py
----------

Builds the Category Anchor Prompt Bank used in the detection pipeline.

Responsibilities:
    • Define semantic categories and prompts
    • Compute prompt embeddings using CLIP text encoder
    • Provide prompt → category mappings
    • Save / load anchor bank for reproducibility

Exports:
    AnchorBank
"""

import os
import json
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


class AnchorBank:
    """
    Category Anchor Prompt Bank.

    Provides:
        - Prompt list
        - Category mapping
        - Prompt embeddings (zT)
    """

    def __init__(
        self, model, tokenizer, device, categories: Dict[str, List[str]] = None
    ):
        """
        Args:
            model       : CLIP model
            tokenizer   : CLIP tokenizer
            device      : torch device
            categories  : optional custom category prompt dict
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # --------------------------------------------------
        # Default semantic categories (paper-ready)
        # --------------------------------------------------
        if categories is None:
            self.categories = self._default_categories()
        else:
            self.categories = categories

        # Flatten prompt list
        self.prompts = []
        self.prompt_categories = []
        for cat, prompts in self.categories.items():
            for p in prompts:
                self.prompts.append(p)
                self.prompt_categories.append(cat)
        print(
            f"len(self.prompts): {len(self.prompts)}, len(self.prompt_categories): {len(self.prompt_categories)}"
        )

        self.num_prompts = len(self.prompts)
        self.category_names = list(self.categories.keys())
        self.category_to_indices = self._build_category_indices()

        self.prompt_embeddings = None  # zT

    # =====================================================
    # Default categories
    # =====================================================

    def _default_categories(self) -> Dict[str, List[str]]:
        """
        Expanded semantic super-categories.

        7 categories × 10 prompts = 70 prompts.
        Adds explicit plant coverage for trees, flowers, and vegetation.
        Improves ARF covariance stability and GSE graph richness.
        """
        return {
            "animal": [
                "a photo of an animal",
                "a wild animal",
                "a domestic animal",
                "a close-up of an animal",
                "an animal in nature",
                "a mammal in the wild",
                "a bird in nature",
                "a reptile in the wild",
                "a farm animal",
                "a pet animal",
            ],
            "vehicle": [
                "a photo of a vehicle",
                "a car on the road",
                "a moving vehicle",
                "a parked vehicle",
                "a vehicle in traffic",
                "a truck on the highway",
                "a bus on the street",
                "a motorcycle on the road",
                "an airplane in the sky",
                "a ship on the water",
            ],
            "person": [
                "a photo of a person",
                "a person standing",
                "a close-up of a face",
                "a person in action",
                "a portrait of a person",
                "a person walking",
                "a group of people",
                "a person sitting",
                "a person working",
                "a person smiling",
            ],
            "food": [
                "a photo of food",
                "a prepared meal",
                "a dish on a plate",
                "a close-up of food",
                "a fresh meal",
                "a bowl of food",
                "a dessert on a plate",
                "a cooked dish",
                "a healthy meal",
                "a street food dish",
            ],
            "object": [
                "a household object",
                "a common object",
                "a small object",
                "an everyday item",
                "a product photo",
                "a tool on a table",
                "an electronic device",
                "a piece of furniture",
                "a kitchen utensil",
                "a decorative object",
            ],
            "scene": [
                "a natural landscape",
                "an outdoor scene",
                "a city scene",
                "a scenic view of nature",
                "a forest landscape",
                "a mountain view",
                "a beach scene",
                "a park with trees",
                "a countryside view",
                "a street view",
            ],
            "plant": [
                "a photo of a plant",
                "a tree in nature",
                "a flowering plant",
                "a green plant",
                "a garden plant",
                "a forest tree",
                "a potted plant",
                "leaves on a plant",
                "a tropical plant",
                "a field of crops",
            ],
        }

    # =====================================================
    # Category index mapping
    # =====================================================

    def _build_category_indices(self) -> Dict[str, List[int]]:
        """
        Map category → indices of prompts belonging to it.
        """
        mapping = {cat: [] for cat in self.category_names}
        for idx, cat in enumerate(self.prompt_categories):
            mapping[cat].append(idx)
        return mapping

    # =====================================================
    # Prompt embedding computation
    # =====================================================

    def compute_embeddings(self, batch_size: int = 16):
        """
        Compute normalized text embeddings for all prompts.

        zT[i] = f_T(prompt_i) / ||f_T(prompt_i)||

        Returns:
            Tensor [M, d]
        """
        if self.prompt_embeddings is not None:
            return self.prompt_embeddings

        embeddings = []

        with torch.no_grad():
            for i in range(0, self.num_prompts, batch_size):
                batch_prompts = self.prompts[i : i + batch_size]
                tokens = self.tokenizer(batch_prompts).to(self.device)

                text_features = self.model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)

                embeddings.append(text_features.cpu())

        self.prompt_embeddings = torch.cat(embeddings, dim=0)
        return self.prompt_embeddings

    # =====================================================
    # Utilities
    # =====================================================

    def get_category_indices(self, category: str) -> List[int]:
        """Return prompt indices for a category."""
        return self.category_to_indices[category]

    def get_prompt_embeddings(self) -> torch.Tensor:
        """Return prompt embeddings (compute if needed)."""
        if self.prompt_embeddings is None:
            self.compute_embeddings()
        return self.prompt_embeddings

    # =====================================================
    # Save / Load (for reproducibility)
    # =====================================================

    def save(self, path: str):
        """
        Save anchor bank to disk.
        """
        os.makedirs(path, exist_ok=True)

        # Save prompts & categories
        with open(os.path.join(path, "categories.json"), "w") as f:
            json.dump(self.categories, f, indent=2)

        # Save embeddings
        torch.save(
            self.get_prompt_embeddings(), os.path.join(path, "prompt_embeddings.pt")
        )

    @classmethod
    def load(cls, path: str, model, tokenizer, device: torch.device):
        """
        Load anchor bank from disk.
        """
        with open(os.path.join(path, "categories.json"), "r") as f:
            categories = json.load(f)

        anchor_bank = cls(model, tokenizer, device=device, categories=categories)
        anchor_bank.prompt_embeddings = torch.load(
            os.path.join(path, "prompt_embeddings.pt"), map_location=device
        )

        return anchor_bank
