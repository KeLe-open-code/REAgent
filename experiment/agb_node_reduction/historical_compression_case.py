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
from evaluation.compression import ReducedReference, rebuild_reference_graph
from evaluation.dataset_schema import sha256
from evaluation.metrics import compression_metrics
from evaluation.selection_methods import select_real_nodes


def raw_prediction(engine, keep, raw, ids, pre):
    scaled = engine.predict(keep, baseline.transform(raw, pre), ids)['scaled'].astype(float)
    return scaled*pre['y_scale']+pre['y_mean']


def sagpool_scores(engine, bundle, target, edge_index, epochs):
    from torch_geometric.nn import GCNConv
    full = engine.predict(np.ones(engine.n, dtype=bool), bundle['feedback_x'], bundle['feedback_ids'])['scaled'].astype(float)
    residual = np.abs(target-full)
    residual = (residual-residual.mean())/max(residual.std(), 1e-12)
    layer = GCNConv(bundle['reference_x'].shape[1], 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=.001)
    x = torch.as_tensor(bundle['reference_x'], dtype=torch.float32)
    y = torch.as_tensor(residual, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        score = layer(x, edge_index).squeeze(-1)
        loss = torch.nn.functional.mse_loss(score, y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        scores = layer(x, edge_index).squeeze(-1).numpy()
    return scores, {'source_fit': 'training-only absolute parent residual', 'epochs': epochs,
                    'parent_frozen': True, 'official_score_mechanism': 'GCNConv', 'legacy_topk_api_used': False}


def run(args):
    assert not args.output.exists()
    _, config, data, _ = baseline.r.load_fold(args.dataset, args.fold)
    n = int(data['n_train'])
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
    reference = ReducedReference(x, source_ids, coords[:n], ecology[:n], bundle['arm'], n)
    adjacency = rebuild_reference_graph(reference, x[:0])
    edge_index = np.array(adjacency.nonzero())
    torch_edges = torch.as_tensor(edge_index, dtype=torch.long)
    k = n//2 if args.budget == 'half' else int(args.agent_endpoint)
    if args.method in ('Agent', 'RP_Agent', 'Full'):
        source = json.loads((args.agent_case/'case_summary.json').read_text(encoding='utf-8'))
        agent_k = int(source['metrics']['retained_references'])
        k = n//2 if args.budget == 'half' else agent_k
        assert args.budget == 'half' or k == agent_k
        keep = np.load(args.agent_case/'selection.npz', allow_pickle=False)['keep']
        selected = np.flatnonzero(keep)
        if len(selected) != k:
            raise ValueError('Requested half K differs from trained Agent endpoint')
        agent_predictions = pd.read_csv(args.agent_case/'predictions.csv')
        assert np.array_equal(agent_predictions.sample_id.astype(str), ids[n:])
        prediction = agent_predictions.prediction.to_numpy(float)
        full = agent_predictions.full_prediction.to_numpy(float)
        diagnostics = {'shared_agent_case': str(args.agent_case), 'shared_training': True}
    else:
        fixed = None
        if args.method == 'fixed_score':
            parent = raw_prediction(engine, np.ones(n, dtype=bool), raw[:n], ids[:n], bundle['preprocessor'])
            fixed = np.abs(target[:n]-parent)
        sag = None
        if args.method == 'SAGPool_real_node_adaptation':
            sag, sag_diagnostics = sagpool_scores(engine, bundle, target[:n], torch_edges, args.sagpool_epochs)
        selection = select_real_nodes(args.method, x, source_ids, coords[:n], k, args.seed,
                                      edge_index=edge_index, fixed_scores=fixed, sagpool_scores=sag)
        selected = selection.selected
        keep = np.zeros(n, dtype=bool)
        keep[selected] = True
        full = raw_prediction(engine, np.ones(n, dtype=bool), raw[n:], ids[n:], bundle['preprocessor'])
        prediction = raw_prediction(engine, keep, raw[n:], ids[n:], bundle['preprocessor'])
        diagnostics = selection.diagnostics
        if args.method == 'SAGPool_real_node_adaptation':
            diagnostics.update(sag_diagnostics)
    assert engine.unchanged_parameters()
    args.output.mkdir(parents=True)
    reduced = ReducedReference.create(x, source_ids, coords[:n], ecology[:n],
                                      select_real_nodes('fixed_score', x, source_ids, coords[:n], k, args.seed,
                                                        fixed_scores=np.flatnonzero(keep)))
    reduced.save(args.output/'reduced_reference.npz')
    engine.export(keep, args.output/'reduced.pt')
    np.savez_compressed(args.output/'selection.npz', selected=selected, keep=keep,
                        source_ids=source_ids, retained_ids=source_ids[selected])
    metrics = compression_metrics(target[n:], full, prediction, n, k)
    pd.DataFrame({'sample_id': ids[n:], 'target': target[n:], 'full_prediction': full,
                  'prediction': prediction}).to_csv(args.output/'predictions.csv', index=False)
    summary = {'method': args.method, 'budget': args.budget, 'K': k, 'metrics': metrics,
               'diagnostics': diagnostics, 'parent_unchanged': True,
               'adaptation': 'historical exact parent with real-node adaptation',
               'final_holdout_access': False}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    execution = {'status': 'COMPLETE', 'dataset': args.dataset, 'fold': args.fold, 'seed': args.seed,
                 'method': args.method, 'budget': args.budget,
                 'parent_sha256': sha256(args.parent_case/'predictor/predictor.pt'),
                 'runner_sha256': sha256(__file__), 'final_holdout_access': False}
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=('hunan', 'hubei', 'guizhou'))
    parser.add_argument('--fold', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('--parent-case', type=Path, required=True)
    parser.add_argument('--method', required=True, choices=('fixed_score', 'matched_random', 'feature_kcenter',
        'spatial_kcenter', 'kmeans_real_representative', 'SGGC_real_node_adaptation',
        'SAGPool_real_node_adaptation', 'GDEM_real_node_adaptation', 'Agent', 'RP_Agent', 'Full'))
    parser.add_argument('--budget', choices=('half', 'agent_endpoint'), default='half')
    parser.add_argument('--agent-case', type=Path)
    parser.add_argument('--agent-endpoint', type=int)
    parser.add_argument('--ecology', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sagpool-epochs', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
