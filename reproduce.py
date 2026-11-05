#!/usr/bin/env python
"""Run every experimental family and generate measured tables without arguments."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from Constants import (
    Attacks, DATA_PROTOCOL_VERSION, Datasets, Detectors, Models,
)

ROOT = Path(__file__).resolve().parent
# The complete study consists of these three replicates in reverse numeric
# order. REPRODUCTION_SEEDS lets machines execute disjoint shards without
# maintaining different scientific code or experiment planners.
CANONICAL_SEEDS = (44, 43, 42)


def configured_seeds():
    raw = os.environ.get('REPRODUCTION_SEEDS', '').strip()
    if not raw:
        return CANONICAL_SEEDS
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(',')
                      if value.strip())
    except ValueError as error:
        raise ValueError(
            'REPRODUCTION_SEEDS must be a comma-separated list of integers') from error
    if not seeds:
        raise ValueError('REPRODUCTION_SEEDS selected no seeds')
    if len(set(seeds)) != len(seeds):
        raise ValueError('REPRODUCTION_SEEDS contains a duplicate seed')
    unknown = set(seeds) - set(CANONICAL_SEEDS)
    if unknown:
        raise ValueError(
            f'Unsupported seed(s) {sorted(unknown)}; choose from {CANONICAL_SEEDS}')
    expected_order = tuple(seed for seed in CANONICAL_SEEDS if seed in seeds)
    if seeds != expected_order:
        raise ValueError(
            f'Seeds must retain reverse order; use {expected_order}')
    return seeds


SEEDS = configured_seeds()
MSP_SCORE_VERSION = 2
RESPONSE_METHOD_VERSION = 3
GEOMETRY_BASELINE_VERSION = 1
# Version 4 adds Auto-PGD to the same full-dataset evaluation matrix.
ATTACK_PROTOCOL_VERSION = 4
EXPERIMENT_MODES = (
    'full', 'full_without_geometry_baselines', 'geometry_baselines')


def configured_experiment_mode():
    """Select the complete study or the dedicated local baseline shard."""
    mode = os.environ.get('REPRODUCTION_EXPERIMENT_MODE', 'full').strip().lower()
    if mode not in EXPERIMENT_MODES:
        raise ValueError(
            f'REPRODUCTION_EXPERIMENT_MODE must be one of {EXPERIMENT_MODES}')
    return mode


def reproduction_output_dir():
    """Return the durable result root, optionally overridden by the environment."""
    configured = os.environ.get('REPRODUCTION_OUTPUT_DIR')
    if not configured:
        return (ROOT / 'outputs' / 'reproduction').resolve()
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def experiment_plan():
    if not SEEDS or len(set(SEEDS)) != len(SEEDS):
        raise ValueError("The configured seed shard must contain distinct seeds")
    jobs = []
    experiment_mode = configured_experiment_mode()
    if experiment_mode == 'geometry_baselines':
        for dataset in (Datasets.CIFAR_10, Datasets.STL_10,
                        Datasets.TINY_IMAGENET):
            for seed in SEEDS:
                for model in (Models.VIT_B16, Models.VIT_B32,
                              Models.CONVNEXT_BASE_W):
                    jobs.append(dict(
                        kind='geometry_baselines', dataset=dataset, model=model,
                        seed=seed,
                        geometry_baseline_version=GEOMETRY_BASELINE_VERSION,
                        training_attack=Attacks.PGD,
                        attack_protocol_version=ATTACK_PROTOCOL_VERSION,
                        data_protocol_version=DATA_PROTOCOL_VERSION,
                    ))
        return jobs
    for dataset in (Datasets.CIFAR_10, Datasets.STL_10, Datasets.TINY_IMAGENET):
        for seed in SEEDS:
            for model in (Models.VIT_B16, Models.VIT_B32, Models.CONVNEXT_BASE_W):
                jobs.append(dict(
                    kind='standard', dataset=dataset, model=model,
                    attack=Attacks.PGD, detector=Detectors.ENERGY_BASED,
                    seed=seed, cross_attack=True,
                    detector_training_attack=Attacks.PGD,
                    response_method_version=RESPONSE_METHOD_VERSION,
                    attack_protocol_version=ATTACK_PROTOCOL_VERSION,
                    data_protocol_version=DATA_PROTOCOL_VERSION,
                ))
                if experiment_mode == 'full':
                    jobs.append(dict(
                        kind='geometry_baselines', dataset=dataset, model=model,
                        seed=seed,
                        geometry_baseline_version=GEOMETRY_BASELINE_VERSION,
                        training_attack=Attacks.PGD,
                        attack_protocol_version=ATTACK_PROTOCOL_VERSION,
                        data_protocol_version=DATA_PROTOCOL_VERSION,
                    ))
                for attack in Attacks.ALL:
                    jobs.append(dict(
                        kind='standard', dataset=dataset, model=model,
                        attack=attack, detector=Detectors.MSP, seed=seed,
                        detector_training_attack=(
                            Attacks.PGD if attack == Attacks.APGD else attack),
                        cross_attack=False, msp_score_version=MSP_SCORE_VERSION,
                        attack_protocol_version=ATTACK_PROTOCOL_VERSION,
                        data_protocol_version=DATA_PROTOCOL_VERSION,
                    ))
            if dataset == Datasets.CIFAR_10:
                jobs.append(dict(kind='ablation', dataset=dataset, seed=seed,
                                 response_method_version=RESPONSE_METHOD_VERSION,
                                 attack_protocol_version=ATTACK_PROTOCOL_VERSION,
                                 data_protocol_version=DATA_PROTOCOL_VERSION))
        if dataset == Datasets.CIFAR_10 and 42 in SEEDS:
            jobs.append(dict(kind='figures', dataset=dataset, seed=42))
        jobs.append(dict(kind='cleanup_dataset', dataset=dataset, seed=SEEDS[-1]))
    jobs.append(dict(kind='cleanup_final', seed=SEEDS[-1]))
    return jobs


def source_fingerprint():
    files = sorted(list(ROOT.glob('*.py')) + list((ROOT / 'Detection').rglob('*.py')) + [ROOT / 'requirements.txt'])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def summarize(completed):
    """Keep raw per-seed results and aggregate numeric metrics without mixing protocols."""
    import statistics
    groups = {}
    for entry in completed:
        job, result = entry['job'], entry['result']
        rows = []
        if job['kind'] == 'standard':
            identity = {
                **{k: job[k] for k in ('dataset', 'model', 'attack', 'detector')},
            }
            rows.append((dict(identity, protocol='standard'), result['standard']))
            rows.extend((dict(identity, protocol='cross_attack', attack=attack), values)
                        for attack, values in result['cross_attack'].items())
        elif job['kind'] == 'ablation':
            rows.extend((dict(kind='ablation', variant=r['variant']), r) for r in result)
        elif job['kind'] == 'geometry_baselines':
            for detector, values in result['baselines'].items():
                identity = dict(dataset=job['dataset'], model=job['model'],
                                attack=Attacks.PGD, detector=detector)
                rows.append((dict(identity, protocol='standard'),
                             values['standard']))
                rows.extend((dict(identity, protocol='cross_attack', attack=attack), metrics)
                            for attack, metrics in values['cross_attack'].items())
        for identity, values in rows:
            key = json.dumps(identity, sort_keys=True)
            group = groups.setdefault(key, dict(experiment=identity, seeds=[], metrics={}))
            group['seeds'].append(job['seed'])
            for metric, value in values.items():
                if isinstance(value, (float, int)) and metric not in ('seed', 'threshold', 'feature_dim', 'batch_size', 'warmup', 'repeats'):
                    group['metrics'].setdefault(metric, []).append(value)
    for group in groups.values():
        group['metrics'] = {k: dict(mean=statistics.mean(v), std=statistics.stdev(v) if len(v)>1 else None,
                                    n=len(v)) for k,v in group['metrics'].items()}
    return list(groups.values())


def archive_incompatible_data_protocol(store):
    """Preserve old sampled-data records but never reuse or aggregate them."""
    excluded = store.data.setdefault('excluded_jobs', {})
    excluded_results = store.data.setdefault('excluded_results', {})
    stale = []
    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        if (job.get('kind') in {'standard', 'geometry_baselines', 'ablation'}
                and job.get('data_protocol_version') != DATA_PROTOCOL_VERSION):
            stale.append((key, entry))
    for key, entry in stale:
        excluded[key] = dict(
            entry,
            exclusion_reason='Superseded by the full-dataset protocol.',
        )
        del store.data['jobs'][key]
    if stale:
        for key, value in list(store.data.get('experiments', {}).items()):
            excluded_results[f'legacy-data-protocol:{key}'] = {
                'result': value,
                'exclusion_reason': 'Superseded by the full-dataset protocol.',
            }
            del store.data['experiments'][key]
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
        print(
            f'Archived {len(stale)} result(s) from an older data protocol; '
            'full-dataset jobs will run.',
            flush=True,
        )


def discard_outdated_msp_results(store, output):
    """Remove only MSP results produced before validation-oriented scoring."""
    stale_keys = []
    stale_directories = []
    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        if (job.get('kind') == 'standard' and job.get('detector') == Detectors.MSP
                and job.get('msp_score_version') != MSP_SCORE_VERSION):
            stale_keys.append(key)
            stale_directories.append(str(
                output / f"seed_{job['seed']}" / Detectors.MSP / job['attack']
                / job['dataset'] / job['model']))
    for key in stale_keys:
        del store.data['jobs'][key]
    for directory in stale_directories:
        store.data.get('experiments', {}).pop(directory, None)
    if stale_keys:
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
        print(f'Discarded {len(stale_keys)} outdated MSP result(s); they will be rerun.', flush=True)


def discard_outdated_response_results(store, output):
    """Invalidate only results affected by response-method implementation changes."""
    stale_keys = []
    prefixes = []
    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        affected = (
            (job.get('kind') == 'standard' and job.get('detector') == Detectors.ENERGY_BASED)
            or job.get('kind') == 'ablation'
        )
        if affected and job.get('response_method_version') != RESPONSE_METHOD_VERSION:
            stale_keys.append(key)
            seed_root = output / f"seed_{job['seed']}"
            if job['kind'] == 'standard':
                prefixes.append(seed_root / Detectors.ENERGY_BASED / job['attack']
                                / job['dataset'] / job['model'])
            elif job['kind'] == 'ablation':
                prefixes.append(seed_root / 'ablation')
    for key in stale_keys:
        del store.data['jobs'][key]
    prefix_strings = [str(path) for path in prefixes]
    for key in list(store.data.get('experiments', {})):
        if any(key == prefix or key.startswith(prefix + os.sep) for prefix in prefix_strings):
            del store.data['experiments'][key]
    if stale_keys:
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
        print(f'Discarded {len(stale_keys)} outdated response-method result(s); they will be rerun.', flush=True)


def discard_outdated_attack_protocol(store, output):
    """Invalidate non-paper evaluations while retaining fitted detectors and data."""
    stale_keys = []
    prefixes = []

    def affected(job):
        return (job.get('kind') in {'standard', 'ablation'}
                and job.get('attack_protocol_version') != ATTACK_PROTOCOL_VERSION)

    def add_prefix(job):
        seed_root = output / f"seed_{job['seed']}"
        if job['kind'] == 'standard':
            prefixes.append(seed_root / job['detector'] / job['attack']
                            / job['dataset'] / job['model'])
        else:
            prefixes.append(seed_root / 'ablation')

    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        if affected(job):
            stale_keys.append(key)
            add_prefix(job)

    # A long cross-attack job can save individual evaluation records before
    # the enclosing job is committed. Remove those partial records too when
    # switching protocols, so an old adaptive number never appears in a new
    # paper-protocol snapshot.
    current_job = store.data.get('current_job') or {}
    stale_partial = affected(current_job)
    if stale_partial:
        add_prefix(current_job)
    for key in stale_keys:
        del store.data['jobs'][key]
    prefix_strings = [str(path) for path in prefixes]
    for key in list(store.data.get('experiments', {})):
        if any(key == prefix or key.startswith(prefix + os.sep) for prefix in prefix_strings):
            del store.data['experiments'][key]
    if stale_keys:
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
    if stale_keys or stale_partial:
        count = len(stale_keys) + int(stale_partial)
        print(f'Discarded {count} completed or partial result(s) from a different attack protocol; paper evaluations will rerun.', flush=True)


def discard_outdated_geometry_baselines(store, output):
    """Invalidate only ordinary geometry baselines from a different protocol."""
    stale_keys = []
    prefixes = []
    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        if (job.get('kind') == 'geometry_baselines'
                and (job.get('geometry_baseline_version')
                     != GEOMETRY_BASELINE_VERSION
                     or job.get('attack_protocol_version')
                     != ATTACK_PROTOCOL_VERSION)):
            stale_keys.append(key)
            prefixes.append(
                output / f"seed_{job['seed']}" / 'geometry_baselines' / 'pgd'
                / job['dataset'] / job['model'])
    for key in stale_keys:
        del store.data['jobs'][key]
    prefix_strings = [str(path) for path in prefixes]
    for key in list(store.data.get('experiments', {})):
        if any(key == prefix or key.startswith(prefix + os.sep)
               for prefix in prefix_strings):
            del store.data['experiments'][key]
    if stale_keys:
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
        print(f'Discarded {len(stale_keys)} outdated geometry-baseline '
              'result(s); they will be rerun.', flush=True)


def archive_out_of_scope_jobs(store):
    """Preserve superseded jobs while excluding them from core aggregates."""
    excluded = store.data.setdefault('excluded_jobs', {})
    excluded_results = store.data.setdefault('excluded_results', {})
    moved = 0
    changed = False
    for key, entry in list(store.data.get('jobs', {}).items()):
        job = entry.get('job', {})
        out_of_scope = (
            job.get('kind') in {'controlled', 'adaptive_continual',
                                'adaptive_suite'}
            or (job.get('kind') == 'standard' and job.get('attack') == 'cw')
            or (job.get('kind') == 'standard'
                and job.get('detector') == Detectors.ENERGY_BASED
                and job.get('attack') != Attacks.PGD)
        )
        if out_of_scope:
            excluded[key] = dict(entry, exclusion_reason=(
                'Outside the ordinary and cross-attack reproduction protocol.'))
            del store.data['jobs'][key]
            moved += 1
            changed = True
            continue
        if (job.get('kind') == 'standard'
                and job.get('detector') == Detectors.ENERGY_BASED
                and job.get('attack') == Attacks.PGD):
            cross = entry.get('result', {}).get('cross_attack', {})
            if 'cw' in cross:
                excluded_results[f'{key}:cross_attack:cw'] = {
                    'job': job, 'result': cross.pop('cw'),
                    'exclusion_reason': 'CW was removed from the deadline protocol.',
                }
                changed = True
    # Low-level adaptive experiment records are not part of the scheduled job
    # map, but older runs may have written them before interruption. Preserve
    # them outside the active experiment namespace so summaries and exports
    # cannot accidentally treat them as current results.
    experiments = store.data.setdefault('experiments', {})
    for key in list(experiments):
        if '/adaptive_suite/' in key.replace('\\', '/'):
            excluded_results[f'experiment:{key}'] = {
                'result': experiments.pop(key),
                'exclusion_reason': (
                    'Outside the ordinary and cross-attack reproduction protocol.'),
            }
            changed = True
    if changed:
        store.data['completed_jobs'] = len(store.data['jobs'])
        store.data['summary'] = summarize(store.data['jobs'].values())
    if moved:
        print(f'Archived {moved} superseded job(s); raw records were preserved.',
              flush=True)


def run_job(job, output, device):
    if job['kind'] == 'standard':
        from main import prepare_experiment
        detector_source_dir = None
        if (job['detector'] == Detectors.MSP
                and job['attack'] == Attacks.APGD):
            detector_source_dir = str(
                output / Detectors.MSP / Attacks.PGD
                / job['dataset'] / job['model'])
        return prepare_experiment(attack_name=job['attack'], model_name=job['model'],
                                  dataset_name=job['dataset'], detector_name=job['detector'],
                                  output_root=str(output), device=device, force_attack=False,
                                  force_train=False, evaluate_transferability=job['cross_attack'],
                                  detector_source_dir=detector_source_dir)
    if job['kind'] == 'ablation':
        from ablation import AblationRunConfig, run_ablation_study
        return run_ablation_study(AblationRunConfig(base_dir=str(output / 'ablation')))
    if job['kind'] == 'geometry_baselines':
        from geometry_baselines import run_geometry_baselines
        return run_geometry_baselines(job['dataset'], job['model'], output, device)
    if job['kind'] == 'figures':
        figure_dir = output / 'figures'
        figure_dir.mkdir(exist_ok=True)
        for script, name in [('make_real_vlm_similarity_figure.py', 'similarity'),
                             ('make_real_feature_vector_figure.py', 'feature_vector')]:
            subprocess.run([sys.executable, str(ROOT / script), '--out', str(figure_dir / name)],
                           cwd=ROOT, check=True)
        from response_graph_figure import make_graph_figure
        make_graph_figure(figure_dir / 'prompt_graph.png', device)
        return {'directory': str(figure_dir)}
    if job['kind'] == 'cleanup_dataset':
        from storage_cleanup import cleanup_dataset
        return cleanup_dataset(ROOT, output.parent, job['dataset'])
    if job['kind'] == 'cleanup_final':
        from storage_cleanup import cleanup_transient_files
        return cleanup_transient_files(ROOT, output.parent)
    raise ValueError(job)


def main():
    if len(sys.argv) != 1:
        raise SystemExit('reproduce.py takes no command-line arguments')
    os.chdir(ROOT)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('PYTHONUTF8', '1')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    from Utils import get_device, release_memory, set_global_seed
    import torch
    device = get_device(show_device_info=True)
    fingerprint = source_fingerprint()
    versions = dict(sorted((distribution.metadata['Name'], distribution.version)
                           for distribution in importlib.metadata.distributions()
                           if distribution.metadata['Name']))
    environment = dict(python=platform.python_version(), packages=versions,
                       hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                       cuda=torch.version.cuda)
    import result_store
    output = reproduction_output_dir()
    store = result_store.ResultStore(output / 'results.json')
    store.data['seeds'] = list(SEEDS)
    result_store.ACTIVE = store
    archive_incompatible_data_protocol(store)
    archive_out_of_scope_jobs(store)
    discard_outdated_msp_results(store, output)
    discard_outdated_response_results(store, output)
    discard_outdated_attack_protocol(store, output)
    discard_outdated_geometry_baselines(store, output)
    plan = experiment_plan()
    store.data['sessions'].append(dict(
        source_sha256=fingerprint,
        environment=environment,
        experiment_mode=configured_experiment_mode(),
        data_protocol_version=DATA_PROTOCOL_VERSION,
        started_utc=datetime.now(timezone.utc).isoformat(),
    ))
    store.data['plan'] = plan
    store.data['status'] = 'running'
    store.data.pop('error', None)
    store.save()
    try:
        for index, job in enumerate(plan):
            key = json.dumps(job, sort_keys=True)
            if key in store.data['jobs']:
                print(f'[{index + 1}/{len(plan)}] Already completed: {job}', flush=True)
                from storage_cleanup import cleanup_job_caches
                seed_output = output / f"seed_{job['seed']}"
                cleanup_job_caches(seed_output, job)
                continue
            print(f'[{index + 1}/{len(plan)}] Running: {job}', flush=True)
            store.data['current_job'] = job
            store.save()
            set_global_seed(job['seed'])
            seed_output = output / f"seed_{job['seed']}"
            seed_output.mkdir(parents=True, exist_ok=True)
            result = run_job(job, seed_output, device)
            store.data['jobs'][key] = dict(job=job, result=result)
            store.data['summary'] = summarize(store.data['jobs'].values())
            store.data['completed_jobs'] = len(store.data['jobs'])
            store.save()
            from storage_cleanup import cleanup_job_caches
            removed = cleanup_job_caches(seed_output, job)
            if removed:
                print(f"Cleaned {len(removed)} regenerable cache path(s).", flush=True)
            release_memory()
        store.data['status'] = 'complete'
        store.data.pop('current_job', None)
        store.data.pop('error', None)
        print(f"All results saved to {store.path}")
    except KeyboardInterrupt:
        store.data['status'] = 'stopped'
        print('Stopped. Run the same command to resume unfinished experiments.')
    except Exception as error:
        store.data['status'] = 'failed'
        store.data['error'] = repr(error)
        raise
    finally:
        store.save()
        result_store.ACTIVE = None


if __name__ == '__main__':
    main()
