"""Bounded-memory, restartable adversarial cache with split validation."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm.auto import tqdm


def _load_manifest(path):
    """Return a validated manifest, or an empty value after an interrupted write."""
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('manifest root is not an object')
        signature = value.get('signature', {})
        sizes = value.get('sizes', [])
        if not isinstance(signature, dict) or not isinstance(sizes, list):
            raise ValueError('manifest fields have invalid types')
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f'Ignoring unreadable adversarial-cache manifest {path}: {error}',
              flush=True)
        return {}


def _save_manifest(path, value):
    """Atomically publish a manifest so interruption cannot truncate it."""
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


class ShardDataset(Dataset):
    def __init__(self, directory, sizes):
        self.directory = Path(directory)
        self.sizes = sizes
        self.ends = []
        for size in sizes:
            self.ends.append(size + (self.ends[-1] if self.ends else 0))
        self.current = None
        self.data = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        from bisect import bisect_right
        shard = bisect_right(self.ends, index)
        if shard != self.current:
            self.data = torch.load(self.directory / f'{shard:06d}.pt',
                                   map_location='cpu', weights_only=True)
            self.current = shard
        offset = index - (self.ends[shard - 1] if shard else 0)
        return self.data['images'][offset], self.data['labels'][offset]


def build_sharded_loader(loader, attacker, cache_path, device, reuse):
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError('Adversarial caching requires ordered, complete clean batches')
    from Utils import SEED
    directory = Path(str(cache_path) + '.shards')
    directory.mkdir(parents=True, exist_ok=True)
    dataset = loader.dataset
    indices = list(range(len(dataset)))
    while hasattr(dataset, 'indices'):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    config = getattr(attacker, 'cfg', None)
    text = getattr(attacker, 'text_features', None)
    if text is None and hasattr(attacker, 'classifier'):
        text = getattr(attacker.classifier, 'text_features', None)
    signature = {
        'version': 4, 'seed': SEED, 'samples': len(loader.dataset), 'batch_size': loader.batch_size,
        'indices': hashlib.sha256(json.dumps(list(map(int, indices))).encode()).hexdigest(),
        'attack': type(attacker).__name__, 'config': repr(config),
        'attack_semantics_version': getattr(attacker, 'cache_semantics_version', 1),
        'cache_identity': getattr(attacker, 'cache_identity', None),
        'text_embeddings': hashlib.sha256(text.detach().float().cpu().numpy().tobytes()).hexdigest() if text is not None else None,
        'parameters': {k: getattr(attacker, k) for k in ('steps', 'c', 'lr', 'kappa') if hasattr(attacker, k)},
        'source': hashlib.sha256(b''.join(p.read_bytes() for p in sorted(
            list(Path(__file__).resolve().parents[2].glob('*.py')) +
            list(Path(__file__).resolve().parents[1].rglob('*.py'))
        ))).hexdigest(),
    }
    manifest = directory / 'manifest.json'
    previous = _load_manifest(manifest)
    previous_signature = previous.get('signature', {})
    # Source-only changes such as logging/progress must not invalidate expensive
    # attacks.  Scientific compatibility is captured explicitly below; attack
    # implementations must bump cache_semantics_version when their output changes.
    scientific_keys = (
        'seed', 'samples', 'batch_size', 'indices', 'attack', 'config',
        'text_embeddings', 'parameters', 'cache_identity',
    )
    compatible = reuse and all(
        previous_signature.get(key) == signature.get(key) for key in scientific_keys
    ) and previous_signature.get('attack_semantics_version', 1) == signature['attack_semantics_version']
    sizes = []
    # A per-batch seed makes an interrupted run equivalent to an uninterrupted run.
    batches = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Generating {type(attacker).__name__}",
        unit="batch",
        dynamic_ncols=True,
    )
    for i, (images, labels) in batches:
        path = directory / f'{i:06d}.pt'
        reusable = False
        if compatible and path.exists():
            try:
                saved = torch.load(path, map_location='cpu', weights_only=True)
                reusable = (
                    len(saved['images']) == len(labels)
                    and torch.equal(saved['labels'], labels.cpu())
                )
            except (OSError, RuntimeError, KeyError, EOFError):
                reusable = False
        if not reusable:
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(SEED + i)
                with torch.enable_grad():
                    adv = attacker.generate(images.to(device), labels.to(device))
            temporary = path.with_suffix('.tmp')
            torch.save({'images': adv.detach().float().cpu(), 'labels': labels.cpu()}, temporary)
            temporary.replace(path)
            batches.set_postfix(generated=i + 1, refresh=False)
        else:
            batches.set_postfix(reused=i + 1, refresh=False)
        sizes.append(len(labels))
        # Only this many files may be reused after a partial overwrite.
        if not compatible:
            _save_manifest(manifest, {'signature': signature, 'sizes': sizes})
    _save_manifest(manifest, {'signature': signature, 'sizes': sizes})
    dataset = ShardDataset(directory, sizes)
    return DataLoader(dataset, batch_size=loader.batch_size, shuffle=False)
