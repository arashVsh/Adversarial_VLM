import os
import torch


class AdvCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.cache_dir, f"{name}.pt")

    def exists(self, name):
        return os.path.exists(self._path(name))

    def save(self, name, images_adv, labels):
        torch.save(
            {
                "images_adv": images_adv.cpu(),
                "labels": labels.cpu(),
            },
            self._path(name),
        )
        print(f"💾 Saved adversarial cache: {name}")

    def load(self, name, device):
        data = torch.load(self._path(name), map_location=device)
        print(f"📂 Loaded adversarial cache: {name}")
        return data["images_adv"].to(device), data["labels"].to(device)
