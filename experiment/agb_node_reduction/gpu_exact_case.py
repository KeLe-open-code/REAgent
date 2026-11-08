import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from evaluation import compression
from evaluation.dynamic_agent import DynamicFunctionAgent
from agent_speedup import fit_with_prediction_cache


def cached_agent_fit(self, k, episodes=4, candidates_per_step=4):
    return fit_with_prediction_cache(self, k, episodes, candidates_per_step, batch_rp=False)


DISTANCE_CACHE = {}


def exact_knn_graph(reduced, query_x, k=8):
    query_input = query_x
    query_x = np.asarray(query_x, dtype=np.float64)
    reference_x = np.asarray(reduced.reference_x, dtype=np.float64)
    if not torch.cuda.is_available() or len(reference_x)*max(len(query_x), 1) < 1_000_000:
        return original_graph(reduced, query_x, k)
    assert query_x.ndim == 2 and np.isfinite(query_x).all()
    assert reference_x.ndim == 2 and len(reference_x) >= 2 and np.isfinite(reference_x).all()
    assert reference_x.shape[1] == query_x.shape[1]
    device = torch.device('cuda')

    def topk(distance, count):
        output = []
        for start in range(0, len(distance), 512):
            output.append(torch.topk(distance[start:start+512], count, dim=1,
                                     largest=False, sorted=True).indices.cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, count), dtype=np.int64)

    cache_key = id(query_input)
    cached = DISTANCE_CACHE.get(cache_key)
    reference_indices = None
    if cached is not None and cached[0] is query_input:
        index = {value: position for position, value in enumerate(cached[1])}
        if all(value in index for value in reduced.reference_ids.astype(str)):
            reference_indices = np.asarray([index[value] for value in reduced.reference_ids.astype(str)])
    if reference_indices is not None:
        full_distance = cached[2]
        selected = torch.as_tensor(reference_indices, device=device)
        reference_neighbors = topk(full_distance[selected][:, selected], min(k, len(reference_x)-1))
        query_neighbors = topk(full_distance[:, selected], min(k, len(reference_x)))
    else:
        reference = torch.as_tensor(reference_x, device=device)
        reference_norm = reference.square().sum(dim=1)

        def neighbors(values, count):
            output = []
            for start in range(0, len(values), 512):
                query = torch.as_tensor(values[start:start+512], device=device)
                distance = (query.square().sum(dim=1)[:, None]+reference_norm[None, :]-2*query@reference.T).clamp_min(0)
                output.append(torch.topk(distance, count, dim=1, largest=False, sorted=True).indices.cpu().numpy())
            return np.concatenate(output) if output else np.empty((0, count), dtype=np.int64)

        reference_neighbors = neighbors(reference_x, min(k, len(reference_x)-1))
        query_neighbors = neighbors(query_x, min(k, len(reference_x)))
        if len(reference_x) == len(query_x) and np.array_equal(reference_x, query_x):
            blocks = []
            for start in range(0, len(reference_x), 512):
                query = torch.as_tensor(reference_x[start:start+512], device=device)
                blocks.append((query.square().sum(dim=1)[:, None]+reference_norm[None, :]-2*query@reference.T).clamp_min(0))
            DISTANCE_CACHE[cache_key] = (query_input, tuple(reduced.reference_ids.astype(str)), torch.cat(blocks))

    row = np.repeat(np.arange(len(reference_x)), reference_neighbors.shape[1])
    column = reference_neighbors.ravel()
    mask = row != column
    reference_rows = np.column_stack((row[mask], column[mask])).ravel()
    reference_columns = np.column_stack((column[mask], row[mask])).ravel()
    query_rows = len(reference_x)+np.repeat(np.arange(len(query_x)), query_neighbors.shape[1])
    rows = np.concatenate((reference_rows, query_rows))
    columns = np.concatenate((reference_columns, query_neighbors.ravel()))
    return sparse.coo_matrix((np.ones(len(rows)), (rows, columns)),
                             shape=(len(reference_x)+len(query_x),)*2).tocsr()


original_graph = compression.rebuild_reference_graph


def main():
    compression.rebuild_reference_graph = exact_knn_graph
    DynamicFunctionAgent.fit = cached_agent_fit
    output = Path(sys.argv[sys.argv.index('--output')+1])
    runpy.run_path('experiment/agb_node_reduction/evaluation/run_case.py', run_name='__main__')
    (output/'graph_backend.json').write_text(json.dumps({
        'backend': 'torch_cuda_blocked_exact_knn', 'distance_dtype': 'float64', 'query_block': 512,
        'agent_prediction_cache': True, 'agent_batched_predictions': False,
        'definition_unchanged': True
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
