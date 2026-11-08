import numpy as np

from evaluation import compression


def select_real_nodes(method, reference_x, reference_ids, coords, k, seed, edge_index=None,
                      frozen_embeddings=None, fixed_scores=None, sagpool_scores=None,
                      deletion_schedule=None, agent_callback=None):
    reference_count = len(reference_ids)
    assert reference_x.shape[0] == coords.shape[0] == reference_count
    assert 0 < k <= reference_count
    if method == 'fixed_score':
        assert fixed_scores is not None
        result = compression.topk_scores(fixed_scores, k)
    elif method == 'matched_random':
        result = compression.matched_random(reference_count, k, seed, deletion_schedule)
    elif method == 'feature_kcenter':
        values = reference_x if frozen_embeddings is None else frozen_embeddings
        result = compression.kcenter(values, k, seed)
    elif method == 'spatial_kcenter':
        result = compression.kcenter(coords, k, seed, method='spatial_kcenter')
    elif method == 'kmeans_real_representative':
        values = reference_x if frozen_embeddings is None else frozen_embeddings
        result = compression.kmeans_real_representatives(values, k, seed)
    elif method == 'SGGC_real_node_adaptation':
        values = reference_x if frozen_embeddings is None else frozen_embeddings
        result = compression.sggc_real_nodes(values, edge_index, k)
    elif method == 'SAGPool_real_node_adaptation':
        assert sagpool_scores is not None, 'SAGPool requires a source-trained frozen pooling scorer'
        result = compression.topk_scores(sagpool_scores, k, method)
        result.diagnostics.update(source='https://github.com/inyeoplee77/SAGPool',
                                  synthetic_nodes_used_for_inference=False)
    elif method == 'GDEM_real_node_adaptation':
        values = reference_x if frozen_embeddings is None else frozen_embeddings
        result = compression.gdem_real_nodes(values, edge_index, k, seed)
    elif method in ('Agent', 'RP_Agent', 'Full'):
        assert agent_callback is not None, f'{method} requires its trained deployment selector'
        selected, diagnostics = agent_callback(method=method, k=k, seed=seed)
        result = compression.Selection(method, np.asarray(selected, dtype=np.int64), diagnostics)
    else:
        raise KeyError(method)
    return result.validated(reference_count, k)
