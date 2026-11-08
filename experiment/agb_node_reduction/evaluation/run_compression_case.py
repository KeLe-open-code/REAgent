import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from evaluation.compression import ReducedReference, rebuild_reference_graph
from evaluation.dataset_schema import sha256
from evaluation.dynamic_rp import DynamicPrecisionSAGE
from evaluation.metrics import compression_metrics
from evaluation.public_models import construct_public_model
from evaluation.selection_methods import select_real_nodes
from evaluation.training import ParentEngine, transform
from evaluation.versioning import evaluation_code_fingerprint


def restore_parent(parent_case, data, device):
    saved = torch.load(parent_case/'parent.pt', map_location=device, weights_only=False)
    kind = saved['kind']
    n = int(data['n_train'])
    pre = saved['preprocessor']
    x = transform(data['raw_features'][:n], pre)
    config = saved['config']
    if kind == 'RP':
        model = DynamicPrecisionSAGE(config).to(device)
    else:
        model = construct_public_model('graphsage', x.shape[1], config['seed']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    engine = ParentEngine(kind, model, x, data['sample_ids'][:n].astype(str), data['coords'][:n],
                          data['ecology'][:n], pre, device)
    return engine


def source_graph(engine):
    adjacency = rebuild_reference_graph(ReducedReference(engine.x, engine.ids, engine.coords, engine.ecology,
                                                         engine.kind, engine.n), engine.x[:0])
    return np.array(adjacency.nonzero()), torch.as_tensor(np.array(adjacency.nonzero()), dtype=torch.long)


def sagpool_scores(engine, target, edge_index, epochs=200):
    from torch_geometric.nn import GCNConv

    path = ROOT/'02_模型代码/external/SAGPool-master/layers.py'
    full = engine.predict(np.ones(engine.n, dtype=bool), engine.x, engine.ids)['prediction']
    residual = np.abs(target-full)
    residual = (residual-residual.mean())/max(residual.std(), 1e-12)
    score_layer = GCNConv(engine.x.shape[1], 1)
    optimizer = torch.optim.Adam(score_layer.parameters(), lr=.001)
    x = torch.as_tensor(engine.x, dtype=torch.float32)
    y = torch.as_tensor(residual, dtype=torch.float32)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        score = score_layer(x, edge_index).squeeze(-1)
        loss = torch.nn.functional.mse_loss(score, y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        scores = score_layer(x, edge_index).squeeze(-1).numpy()
    return scores, {'source_fit': 'training-only absolute parent residual', 'epochs': epochs,
                    'parent_frozen': True, 'official_score_layer': str(path),
                    'legacy_topk_api_used': False}


def run(args):
    assert not args.output.exists()
    with np.load(args.data, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    n = int(data['n_train'])
    engine = restore_parent(args.parent_case, data, args.device)
    before = engine.parent_state()
    edge_index, torch_edges = source_graph(engine)
    source_ids = data['sample_ids'][:n].astype(str)
    query_ids = data['sample_ids'][n:].astype(str)
    query_x = transform(data['raw_features'][n:], engine.pre)
    if args.method in ('Agent', 'RP_Agent', 'Full'):
        source = json.loads((args.agent_case/'case_summary.json').read_text(encoding='utf-8'))
        agent_k = int(source['metrics']['retained_references'])
        k = n//2 if args.budget == 'half' else agent_k
        assert args.budget == 'half' or k == agent_k
        selected = np.flatnonzero(np.load(args.agent_case/'selection.npz', allow_pickle=False)['keep'])
        if len(selected) != k:
            raise ValueError('Requested half K differs from trained Agent endpoint; train a separate fixed-half Agent case')
        agent_predictions = pd.read_csv(args.agent_case/'predictions.csv')
        assert np.array_equal(agent_predictions.sample_id.astype(str), query_ids)
        prediction = agent_predictions.prediction.to_numpy(float)
        full = agent_predictions.full_prediction.to_numpy(float)
        diagnostics = {'shared_agent_case': str(args.agent_case), 'shared_training': True}
    else:
        k = n//2 if args.budget == 'half' else int(args.agent_endpoint)
        assert 0 < k <= n
        fixed = None
        if args.method == 'fixed_score':
            parent = engine.predict(np.ones(n, dtype=bool), engine.x, engine.ids)['prediction']
            fixed = np.abs(data['y'][:n]-parent)
        sag = None
        if args.method == 'SAGPool_real_node_adaptation':
            sag, sag_diagnostics = sagpool_scores(engine, data['y'][:n], torch_edges, args.sagpool_epochs)
        schedule = None
        if args.method == 'matched_random' and args.agent_case:
            agent_summary = json.loads((args.agent_case/'case_summary.json').read_text(encoding='utf-8'))
            schedule = agent_summary['agent_report']['deletion_schedule']
            assert sum(schedule) == n-k
        selection = select_real_nodes(args.method, engine.x, source_ids, engine.coords, k, args.seed,
                                      edge_index=edge_index, fixed_scores=fixed, sagpool_scores=sag,
                                      deletion_schedule=schedule)
        reduced = ReducedReference.create(engine.x, source_ids, engine.coords, engine.ecology, selection)
        full = engine.predict(np.ones(n, dtype=bool), query_x, query_ids)['prediction']
        keep = np.zeros(n, dtype=bool)
        keep[selection.selected] = True
        prediction = engine.predict(keep, query_x, query_ids)['prediction']
        diagnostics = selection.diagnostics
        if args.method == 'SAGPool_real_node_adaptation':
            diagnostics.update(sag_diagnostics)
        args.output.mkdir(parents=True)
        reduced.save(args.output/'reduced_reference.npz')
        np.savez_compressed(args.output/'selection.npz', selected=selection.selected, source_ids=source_ids,
                            retained_ids=source_ids[selection.selected])
    assert engine.parent_state() == before
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = compression_metrics(data['y'][n:], full, prediction, n, k)
    pd.DataFrame({'sample_id': query_ids, 'target': data['y'][n:], 'full_prediction': full,
                  'prediction': prediction}).to_csv(args.output/'predictions.csv', index=False)
    summary = {'method': args.method, 'budget': args.budget, 'K': k, 'metrics': metrics,
               'diagnostics': diagnostics, 'parent_unchanged': True,
               'adaptation': 'real-node adaptation' if args.method not in ('Agent', 'RP_Agent', 'Full') else 'project shared artifact',
               'final_holdout_access': False}
    (args.output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    execution = {'status': 'COMPLETE', 'data_sha256': sha256(args.data),
                 'parent_case': str(args.parent_case), 'parent_sha256': sha256(args.parent_case/'parent.pt'),
                 'runner_sha256': evaluation_code_fingerprint(), 'method': args.method, 'budget': args.budget,
                 'seed': args.seed, 'sagpool_epochs': args.sagpool_epochs,
                 'device': args.device, 'final_holdout_access': False}
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--parent-case', type=Path, required=True)
    parser.add_argument('--method', required=True, choices=('fixed_score', 'matched_random', 'feature_kcenter',
        'spatial_kcenter', 'kmeans_real_representative', 'SGGC_real_node_adaptation',
        'SAGPool_real_node_adaptation', 'GDEM_real_node_adaptation', 'Agent', 'RP_Agent', 'Full'))
    parser.add_argument('--budget', choices=('half', 'agent_endpoint'), default='half')
    parser.add_argument('--agent-case', type=Path)
    parser.add_argument('--agent-endpoint', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sagpool-epochs', type=int, default=200)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
