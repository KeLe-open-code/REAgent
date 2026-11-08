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
from evaluation.dataset_schema import sha256
from evaluation.dynamic_agent import DynamicFunctionAgent
from evaluation.eco_calibration import EcologicalOOFCalibrator
from evaluation.metrics import compression_metrics
from evaluation.run_compression_case import restore_parent
from evaluation.training import transform


def run(args):
    assert not args.output.exists()
    with np.load(args.target_data, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    n = int(data['n_train'])
    assert n > 0 and len(data['y']) > n
    engine = restore_parent(args.source_case, data, args.device)
    parent_hash = engine.parent_state()
    source = json.loads((args.source_case/'case_summary.json').read_text(encoding='utf-8'))
    use_agent = bool(source['agent'])
    use_eco = bool(source['ecological_calibration'])
    keep = np.ones(n, dtype=bool)
    agent_report = None
    if use_agent:
        assert args.actor_case
        actor_path = args.actor_case/'actor.pt'
        agent = DynamicFunctionAgent.restore(actor_path, engine, engine.x, engine.ids, engine.coords)
        selected, agent_report = agent.deploy(n//2)
        keep[:] = False
        keep[selected] = True
    query_ids = data['sample_ids'][n:].astype(str)
    query_x = transform(data['raw_features'][n:], engine.pre)
    full = engine.predict(np.ones(n, dtype=bool), query_x, query_ids)['prediction']
    prediction = engine.predict(keep, query_x, query_ids)['prediction']
    correction = np.zeros(len(prediction))
    if use_eco:
        with np.load(args.source_case/'calibration.npz', allow_pickle=False) as saved:
            categories = saved['categories'].astype(str)
            coefficients = {category: saved['coefficients'][index] for index, category in enumerate(categories)}
            calibrator = EcologicalOOFCalibrator(coefficients, float(saved['center']), float(saved['scale']),
                                                int(saved['minimum_support']))
        prediction, correction = calibrator.transform(data['ecology'][n:].astype(str), prediction)
    assert engine.parent_state() == parent_hash
    metrics = compression_metrics(data['y'][n:], full, prediction, n, int(keep.sum()))
    args.output.mkdir(parents=True)
    pd.DataFrame({'sample_id': query_ids, 'target': data['y'][n:], 'full_prediction': full,
                  'ecological_offset': correction, 'prediction': prediction}).to_csv(args.output/'predictions.csv', index=False)
    np.savez_compressed(args.output/'selection.npz', keep=keep, source_ids=data['sample_ids'][:n].astype(str))
    summary = {'source_case': str(args.source_case), 'actor_case': str(args.actor_case) if args.actor_case else None,
               'zero_shot': True, 'target_fit': False, 'target_preprocessing_fit': False,
               'target_agent_fit': False, 'target_eco_fit': False, 'target_labels': 'evaluation only',
               'metrics': metrics, 'agent_report': agent_report, 'parent_unchanged': True,
               'final362_accessed': False}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    execution = {'status': 'COMPLETE', 'target_data_sha256': sha256(args.target_data),
                 'source_parent_sha256': sha256(args.source_case/'parent.pt'),
                 'actor_sha256': sha256(args.actor_case/'actor.pt') if args.actor_case else None,
                 'target_fit': False, 'runner_sha256': sha256(__file__)}
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-case', type=Path, required=True)
    parser.add_argument('--actor-case', type=Path)
    parser.add_argument('--target-data', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
