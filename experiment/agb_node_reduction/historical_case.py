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
import physical_factorial_v2 as factorial
from evaluation.dataset_schema import sha256
from evaluation.eco_calibration import EcologicalOOFCalibrator
from evaluation.metrics import compression_metrics


def fit_parent(region, fold, kind, seed, indices, output, small):
    original = baseline.source_data

    def selected_source(selected_region, selected_fold):
        assert selected_region == region and selected_fold == fold
        source, config, data = original(region, fold)
        config = {**config, 'seed': int(config['seed']+seed), 'relation_n_train': len(indices)}
        data = {key: value[indices].copy() for key, value in data.items()}
        return source, config, data

    baseline.source_data = selected_source
    try:
        return baseline.fit_predictor(region, fold, kind, output, device='cuda', small=small)
    finally:
        baseline.source_data = original


def raw_prediction(engine, keep, raw, ids, pre):
    x = baseline.transform(raw, pre)
    scaled = engine.predict(keep, x, ids)['scaled'].astype(float)
    return scaled*pre['y_scale']+pre['y_mean']


def run(args):
    assert not args.output.exists()
    assert args.arm in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
    use_rp = args.arm in ('A1', 'A4', 'A5', 'A7')
    use_agent = args.arm in ('A2', 'A4', 'A6', 'A7')
    use_eco = args.arm in ('A3', 'A5', 'A6', 'A7')
    kind = 'RP' if use_rp else 'G0'
    baseline.r.setup()
    source, config, data, _ = baseline.r.load_fold(args.dataset, args.fold)
    n = int(data['n_train'])
    raw, ids = data['raw_features'], data['sample_ids'].astype(str)
    target, coords = data['y'].astype(float), data['coords'].astype(float)
    ecology_table = pd.read_csv(args.ecology)
    ecology_by_id = ecology_table.set_index('sample_id').species.astype(str).to_dict()
    ecology = np.array([ecology_by_id[value] for value in ids])
    args.output.mkdir(parents=True)
    oof = None
    if use_eco:
        rng = np.random.default_rng(int(config['seed'])+args.seed+31337)
        permutation = rng.permutation(n)
        directions = ((permutation[n//2:], permutation[:n//2]),
                      (permutation[:n//2], permutation[n//2:]))
        oof = np.empty(n, dtype=float)
        coverage = np.zeros(n, dtype=np.int64)
        for direction, (held, retained) in enumerate(directions):
            folder = args.output/f'oof-{direction}'
            bundle = fit_parent(args.dataset, args.fold, kind, args.seed+direction+1,
                                retained, folder/'predictor', args.small)
            keep = np.ones(len(retained), dtype=bool)
            if use_agent:
                factorial.agent_select(bundle, folder/'agent', small=args.small)
                keep = np.load(folder/'agent/selection.npz', allow_pickle=False)['keep']
            engine = baseline.engine(bundle)
            oof[held] = raw_prediction(engine, keep, raw[held], ids[held], bundle['preprocessor'])
            coverage[held] += 1
            del engine, bundle
        assert np.array_equal(coverage, np.ones(n, dtype=np.int64)) and np.isfinite(oof).all()
        np.savez_compressed(args.output/'oof_predictions.npz', sample_ids=ids[:n], target=target[:n],
                            prediction=oof, held_coverage=coverage)
    bundle = fit_parent(args.dataset, args.fold, kind, args.seed, np.arange(n),
                        args.output/'predictor', args.small)
    engine = baseline.engine(bundle)
    full = raw_prediction(engine, np.ones(n, dtype=bool), raw[n:], ids[n:], bundle['preprocessor'])
    keep = np.ones(n, dtype=bool)
    agent_report = None
    if use_agent:
        agent_report = factorial.agent_select(bundle, args.output/'agent', small=args.small)
        keep = np.load(args.output/'agent/selection.npz', allow_pickle=False)['keep']
    compressed = raw_prediction(engine, keep, raw[n:], ids[n:], bundle['preprocessor'])
    correction = np.zeros(len(compressed))
    if use_eco:
        calibrator = EcologicalOOFCalibrator.fit(ecology[:n], target[:n]-oof, oof)
        compressed, correction = calibrator.transform(ecology[n:], compressed)
        categories = list(calibrator.coefficients)
        np.savez_compressed(args.output/'calibration.npz', categories=np.asarray(categories),
                            coefficients=np.asarray([calibrator.coefficients[value] for value in categories]),
                            center=np.asarray(calibrator.center), scale=np.asarray(calibrator.scale),
                            minimum_support=np.asarray(calibrator.minimum_support))
    np.savez_compressed(args.output/'selection.npz', keep=keep, source_ids=ids[:n],
                        retained=np.flatnonzero(keep))
    metrics = compression_metrics(target[n:], full, compressed, n, int(keep.sum()))
    pd.DataFrame({'sample_id': ids[n:], 'target': target[n:], 'full_prediction': full,
                  'compressed_prediction': compressed-correction, 'ecological_offset': correction,
                  'prediction': compressed}).to_csv(args.output/'predictions.csv', index=False)
    assert engine.unchanged_parameters()
    summary = {'arm': args.arm, 'kind': kind, 'agent': use_agent, 'ecological_calibration': use_eco,
               'seed': args.seed, 'metrics': metrics, 'agent_report': agent_report,
               'oof_source_labels_only': use_eco, 'outer_labels_used_for_selection': False,
               'parent_unchanged': True, 'final_holdout_access': False,
               'adaptation': 'historical exact G0/RP and Agent with source-only OOF ECRC'}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    execution = {'status': 'COMPLETE', 'dataset': args.dataset, 'fold': args.fold, 'seed': args.seed,
                 'arm': args.arm, 'data_sha256': sha256(Path(source)/'data/development.npz'),
                 'runner_sha256': sha256(__file__), 'device': 'cuda', 'final_holdout_access': False}
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=('hunan', 'hubei', 'guizhou'))
    parser.add_argument('--fold', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1000, 2000))
    parser.add_argument('--arm', required=True)
    parser.add_argument('--ecology', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--small', action='store_true')
    run(parser.parse_args())
