"""Conservative cleanup of downloaded datasets and regenerable attack caches."""
from pathlib import Path
import shutil

from Constants import Datasets, Detectors


def _remove_under(path, allowed_root):
    """Remove one resolved target only when it is strictly below allowed_root."""
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve()
    if path == allowed_root or not path.is_relative_to(allowed_root):
        raise ValueError(f"Refusing cleanup outside the allowed directory: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return str(path)


def cleanup_attack_cache(seed_output, dataset, model, attack):
    """Delete one completed standard experiment's regenerable image shards."""
    seed_output = Path(seed_output).resolve()
    cache = (seed_output / Detectors.ENERGY_BASED / attack / dataset / model / "cache").resolve()
    removed = []
    if cache.exists():
        removed.append(_remove_under(cache, seed_output))
    return removed


def cleanup_job_caches(seed_output, job):
    """Clean caches only after the corresponding result job is durable."""
    if job.get("kind") == "standard" and job.get("detector") == Detectors.MSP:
        return cleanup_attack_cache(
            seed_output, job["dataset"], job["model"], job["attack"]
        )
    if job.get("kind") == "ablation":
        return cleanup_dataset_shards(seed_output, job.get("dataset", Datasets.CIFAR_10))
    return []


def cleanup_dataset_shards(seed_output, dataset):
    """Delete only shard directories whose path belongs to the named dataset."""
    seed_output = Path(seed_output).resolve()
    removed = []
    if not seed_output.exists():
        return removed
    for path in sorted(seed_output.rglob("*.pt.shards"), reverse=True):
        if dataset in path.parts and path.exists():
            removed.append(_remove_under(path, seed_output))
    return removed


def cleanup_dataset(project_root, reproduction_root, dataset):
    """Clean regenerable attack shards while retaining downloaded datasets."""
    reproduction_root = Path(reproduction_root).resolve()
    removed = []
    for seed_dir in sorted(reproduction_root.glob("seed_*")):
        if seed_dir.is_dir():
            removed.extend(cleanup_dataset_shards(seed_dir, dataset))
    return {"dataset": dataset, "removed": removed,
            "dataset_preserved": True}


def cleanup_transient_files(project_root, reproduction_root):
    """Remove only known temporary artifacts; preserve every scientific result."""
    project_root = Path(project_root).resolve()
    reproduction_root = Path(reproduction_root).resolve()
    removed = []
    for name in ("tmp", "temp", ".pytest_cache"):
        target = project_root / name
        if target.exists():
            removed.append(_remove_under(target, project_root))
    for target in sorted(project_root.rglob("__pycache__"), reverse=True):
        if ".venv" not in target.parts and target.exists():
            removed.append(_remove_under(target, project_root))
    for target in sorted(reproduction_root.rglob("*.tmp")):
        if target.exists():
            removed.append(_remove_under(target, reproduction_root))
    return {"removed": removed}
