import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import physical_baseline_v2 as baseline
from evaluation.compression import ReducedReference, Selection, matched_random
from evaluation.dataset_schema import sha256
from evaluation.eco_calibration import EcologicalOOFCalibrator
from evaluation.metrics import compression_metrics


def raw_prediction(engine, keep, raw, ids, pre):
    scaled = engine.predict(keep, baseline.transform(raw, pre), ids)['scaled'].astype(float)
    return scaled*pre['y_scale']+pre['y_mean']


def load_calibrator(path):
    with np.load(path, allow_pickle=False) as saved:
        categories = saved['categories'].astype(str)
        coefficients = {category: saved['coefficients'][index] for index, category in enumerate(categories)}
        return EcologicalOOFCalibrator(coefficients, float(saved['center']), float(saved['scale']),
                                      int(saved['minimum_support']))


def deployment_masks(agent_case, bundle, cache_dir):
    cache_dir = Path(cache_dir)
    cache_path = cache_dir/'target_masks.npz'
    metadata_path = cache_dir/'metadata.json'
    if cache_path.exists() and metadata_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            return {int(key.split('-')[1]): saved[key].copy() for key in saved.files}, json.loads(metadata_path.read_text(encoding='utf-8'))
    n = len(bundle['source_ids'])
    targets = {int(n*(1-rate)) for rate in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)}
    minimum_k = min(targets)
    env = baseline.environment(bundle)
    saved = torch.load(agent_case/'agent/policy.pt', map_location='cpu', weights_only=False)
    model = baseline.agent.StopAgent(saved['seed'], anchored=True)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    candidates = {}
    original_evaluate = env.evaluate
    original_propose = env.propose

    def evaluate(keep):
        error = original_evaluate(keep)
        k = int(keep.sum())
        if k in targets:
            candidate = (env.utility(keep), -float(error), keep.copy())
            if k not in candidates or candidate[:2] > candidates[k][:2]:
                candidates[k] = candidate
        return error

    def propose(keep, rng, phase, valid):
        actions, counts = original_propose(keep, rng, phase, valid)
        current = int(keep.sum())
        if current <= minimum_k:
            return [((), (), 3)], counts
        next_targets = [k for k in targets if k < current]
        budget_floor = max(next_targets) if next_targets else minimum_k
        filtered = [action for action in actions
                    if action[2] != 3 and current-len(action[0])+len(action[1]) >= budget_floor]
        if not filtered:
            raise RuntimeError(f'No legal action before target K={budget_floor}; retained={current}')
        return filtered, counts

    seed = int(bundle['config']['seed'])+10000*baseline.r.REGIONS.index(bundle['config']['region'])+200000
    env.group_rng = np.random.default_rng(seed+460000)
    env.evaluate = evaluate
    env.propose = propose
    try:
        baseline.agent.episode(env, model, seed+900, (64, 4*n), env.full, env.base_mse)
    finally:
        env.evaluate = original_evaluate
        env.propose = original_propose
    cache_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(cache_path, **{f'mask-{k}': value[2] for k, value in candidates.items()})
    metadata = {'selection_source': 'frozen policy target-budget deployment',
                'policy_sha256': sha256(agent_case/'agent/policy.pt'), 'target_K': sorted(targets),
                'covered_K': sorted(candidates), 'minimum_K': minimum_k,
                'deployment_feedback': env.logical_queries, 'parent_unchanged': env.engine.unchanged_parameters(),
                'final_holdout_access': False}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    assert metadata['parent_unchanged']
    return {k: value[2] for k, value in candidates.items()}, metadata


def write_incomplete(args, n, k, reason):
    args.output.mkdir(parents=True)
    summary = {'status': 'INCOMPLETE', 'dataset': args.dataset, 'fold': args.fold, 'seed': args.seed,
               'method': args.method, 'rate': args.rate, 'K': k, 'original_references': n,
               'reason': reason, 'zero_reference': 'A5', 'final_holdout_access': False}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.output/'execution.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


def run(args):
    assert 0 < args.rate < 1
    assert not args.output.exists()
    source, config, data, _ = baseline.r.load_fold(args.dataset, args.fold)
    n = int(data['n_train'])
    k = int(np.floor(n*(1-args.rate)))
    assert 0 < k < n
    raw, ids = data['raw_features'], data['sample_ids'].astype(str)
    target, coords = data['y'].astype(float), data['coords'].astype(float)
    ecology_table = pd.read_csv(args.ecology)
    ecology_by_id = ecology_table.set_index('sample_id').species.astype(str).to_dict()
    ecology = np.array([ecology_by_id[value] for value in ids])
    bundle = torch.load(args.parent_case/'predictor/predictor.pt', map_location='cpu', weights_only=False)
    engine = baseline.engine(bundle)
    assert engine.unchanged_parameters()
    x = bundle['reference_x']
    source_ids = bundle['source_ids'].astype(str)
    calibrator = None
    if args.method == 'Full':
        masks, diagnostics = deployment_masks(args.full_case, bundle, args.cache_dir)
        if k not in masks:
            write_incomplete(args, n, k, 'Target K was not reached by the frozen target-budget deployment')
            return
        keep = masks[k].copy()
        calibrator = load_calibrator(args.full_case/'calibration.npz')
    else:
        selection = matched_random(n, k, args.seed)
        keep = np.zeros(n, dtype=bool)
        keep[selection.selected] = True
        diagnostics = {**selection.diagnostics, 'pure_random': True, 'feedback': 0, 'swaps': 0,
                       'best_of': False}
    assert int(keep.sum()) == k
    full = raw_prediction(engine, np.ones(n, dtype=bool), raw[n:], ids[n:], bundle['preprocessor'])
    prediction = raw_prediction(engine, keep, raw[n:], ids[n:], bundle['preprocessor'])
    correction = np.zeros(len(prediction), dtype=float)
    if calibrator is not None:
        prediction, correction = calibrator.transform(ecology[n:], prediction)
    assert engine.unchanged_parameters()
    metrics = compression_metrics(target[n:], full, prediction, n, k)
    args.output.mkdir(parents=True)
    selected = np.flatnonzero(keep)
    reduced = ReducedReference.create(x, source_ids, coords[:n], ecology[:n],
                                      Selection(args.method, selected, {}))
    reduced.save(args.output/'reduced_reference.npz')
    engine.export(keep, args.output/'reduced.pt')
    np.savez_compressed(args.output/'selection.npz', keep=keep, selected=selected,
                        source_ids=source_ids, retained_ids=source_ids[selected])
    pd.DataFrame({'sample_id': ids[n:], 'target': target[n:], 'full_prediction': full,
                  'ecological_offset': correction, 'prediction': prediction}).to_csv(
                      args.output/'predictions.csv', index=False)
    summary = {'status': 'COMPLETE', 'dataset': args.dataset, 'fold': args.fold, 'seed': args.seed,
               'method': args.method, 'rate': args.rate, 'K': k, 'metrics': metrics,
               'diagnostics': diagnostics, 'parent_unchanged': True, 'zero_reference': 'A5',
               'final_holdout_access': False}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    execution = dict(summary)
    execution.update({'original_references': n,
                      'data_sha256': sha256(Path(source)/'data/development.npz'),
                      'parent_sha256': sha256(args.parent_case/'predictor/predictor.pt'),
                      'actor_sha256': sha256(args.full_case/'agent/policy.pt') if args.method == 'Full' else None,
                      'calibration_sha256': sha256(args.full_case/'calibration.npz') if args.method == 'Full' else None,
                      'runner_sha256': sha256(__file__)})
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=('hunan', 'hubei', 'guizhou'))
    parser.add_argument('--fold', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--parent-case', type=Path, required=True)
    parser.add_argument('--full-case', type=Path)
    parser.add_argument('--method', required=True, choices=('Full', 'PURE_RANDOM'))
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1000, 2000))
    parser.add_argument('--ecology', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
