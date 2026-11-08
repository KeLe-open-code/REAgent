import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from evaluation.compression import NearestNeighbors
from evaluation.dataset_schema import audit_frame, read_table
from evaluation.splitting import deterministic_group_folds, write_fold_manifest


def prepare(args):
    mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    assert args.dataset in ('mexico', 'west_kalimantan', 'belgium')
    frame = read_table(args.input)
    audit = audit_frame(frame, mapping['columns'], args.input)
    ids = frame[mapping['columns']['sample_id']].astype(str).to_numpy()
    coords = frame[[mapping['columns']['longitude'], mapping['columns']['latitude']]].to_numpy(float)
    target = pd.to_numeric(frame[mapping['columns']['agb']], errors='raise').to_numpy(float)
    ecology = frame[mapping['columns']['ecology']].astype(str).to_numpy()
    raw = frame[mapping['columns']['features']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    groups = frame[mapping['group']].astype(str).to_numpy()
    assignment = deterministic_group_folds(ids, groups)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    split = write_fold_manifest(output/'splits', args.dataset, ids, groups, assignment, {
        'group_source': mapping['group'], 'label_source': mapping['label_source'],
        'label_unit': mapping['label_unit'], 'ecology_source': mapping['ecology_source'],
        'feature_count': raw.shape[1], 'source_audit': audit,
        'spatial_scale': mapping.get('spatial_scale', 'verify from publication'),
        'coordinate_crs': mapping.get('coordinate_crs', 'EPSG:4326'),
        'measurement_year': mapping.get('measurement_year', 'see source fields')})
    for fold in (1, 2, 3):
        train = np.flatnonzero(assignment != fold)
        valid = np.flatnonzero(assignment == fold)
        order = np.concatenate((train, valid))
        median = np.nanmedian(raw[train], axis=0)
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, median[None, :], raw)
        mean = imputed[train].mean(axis=0)
        scale = np.where(imputed[train].std(axis=0) > 1e-8, imputed[train].std(axis=0), 1.)
        x = ((imputed-mean)/scale).astype(np.float32)
        y_mean, y_scale = float(target[train].mean()), max(float(target[train].std()), 1e-8)
        n = len(train)
        ref_neighbors = NearestNeighbors(n_neighbors=min(9, n)).fit(x[train]).kneighbors(x[train], return_distance=False)
        train_edges = np.array((np.repeat(np.arange(n), ref_neighbors.shape[1]), ref_neighbors.ravel()), dtype=np.int64)
        query_neighbors = NearestNeighbors(n_neighbors=min(8, n)).fit(x[train]).kneighbors(x[valid], return_distance=False)
        eval_edges = np.array((np.repeat(np.arange(n, len(order)), query_neighbors.shape[1]), query_neighbors.ravel()), dtype=np.int64)
        directory = output/f'fold-{fold:02d}'
        (directory/'data').mkdir(parents=True)
        np.savez_compressed(directory/'data/preprocessor.npz', median=median, mean=mean, scale=scale,
                            y_mean=np.asarray(y_mean), y_scale=np.asarray(y_scale))
        np.savez_compressed(directory/'data/development.npz', raw_features=raw[order], x=x[order], y=target[order],
                            coords=coords[order], ecology=ecology[order].astype(str), sample_ids=ids[order].astype(str), groups=groups[order].astype(str),
                            missing_mask=missing[order], feature_names=np.asarray(mapping['columns']['features'], dtype=str),
                            n_train=n, train_edges=train_edges, eval_edges=eval_edges)
        pd.DataFrame({'sample_id': ids[order], 'role': ['train']*n+['validation']*len(valid)}).to_csv(directory/'data/split.csv', index=False)
        (directory/'config.json').write_text(json.dumps({'dataset': args.dataset, 'fold': fold, 'seed': 20261027+fold-1,
            'input_dim': raw.shape[1], 'relation_n_train': n, 'knn_k': 8, 'hidden_dims': [32, 32], 'dropout': .1,
            'learning_rate': .001, 'weight_decay': 1e-4, 'max_epochs': 400, 'gradient_clip': 5., 'n_splits': 3}, indent=2), encoding='utf-8')
    (output/'prepared.json').write_text(json.dumps({'ready': True, 'samples': len(ids), 'features': raw.shape[1],
        'split_sha256': split['fold_sha256'], 'source_sha256': audit['source_sha256'], 'final_holdout_access': False}, indent=2), encoding='utf-8')
    print(json.dumps({'ready': True, 'dataset': args.dataset, 'samples': len(ids), 'features': raw.shape[1]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())
