import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (0, 1000, 2000)


def deterministic_group_folds(sample_ids, groups, n_splits=3, split_seed=20261027):
    sample_ids = np.asarray(sample_ids).astype(str)
    groups = np.asarray(groups).astype(str)
    assert len(sample_ids) == len(groups) and len(set(sample_ids)) == len(sample_ids)
    assert n_splits == 3 and np.isfinite(pd.to_numeric(pd.Series(groups), errors='coerce').fillna(0)).all()
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    assert len(unique) >= n_splits, 'At least three real spatial/source groups are required'
    rng = np.random.default_rng(split_seed)
    tie = rng.random(len(unique))
    order = np.lexsort((unique, tie, -counts))
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    group_assignment = {}
    for group_index in order:
        fold = int(np.flatnonzero(fold_sizes == fold_sizes.min())[0])
        group_assignment[unique[group_index]] = fold+1
        fold_sizes[fold] += counts[group_index]
    assignment = np.array([group_assignment[value] for value in groups], dtype=np.int64)
    assert set(assignment) == {1, 2, 3}
    return assignment


def write_fold_manifest(output, dataset, sample_ids, groups, assignment, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame({'sample_id': np.asarray(sample_ids).astype(str),
                          'group': np.asarray(groups).astype(str), 'fold': assignment})
    path = output/'fold_assignment.csv'
    frame.to_csv(path, index=False)
    group_overlaps = []
    for fold in (1, 2, 3):
        train_groups = set(frame.loc[frame.fold != fold, 'group'])
        valid_groups = set(frame.loc[frame.fold == fold, 'group'])
        assert not train_groups & valid_groups
        group_overlaps.append({'fold': fold, 'shared_groups': 0,
                               'train_n': int((frame.fold != fold).sum()),
                               'validation_n': int((frame.fold == fold).sum())})
    manifest = {'dataset': dataset, 'n_splits': 3, 'seeds': list(SEEDS),
                'fold_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'folds': group_overlaps, **metadata}
    (output/'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def agent_matched_k(original_references, agent_retained_references):
    original_references = int(original_references)
    agent_retained_references = int(agent_retained_references)
    assert 0 < agent_retained_references <= original_references
    return {'fixed_half': original_references//2,
            'agent_matched': agent_retained_references}
