from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Selection:
    method: str
    selected: np.ndarray
    diagnostics: dict

    def validated(self, reference_count, k):
        selected = np.asarray(self.selected, dtype=np.int64)
        assert selected.shape == (k,)
        assert len(np.unique(selected)) == k
        assert np.all((selected >= 0)&(selected < reference_count))
        return Selection(self.method, selected, dict(self.diagnostics))


def topk_scores(scores, k, method='fixed_score'):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and len(scores) >= k and np.isfinite(scores).all()
    indices = np.arange(len(scores))
    selected = np.lexsort((indices, -scores))[:k]
    return Selection(method, selected, {'score_source': 'frozen training-side scores'}).validated(len(scores), k)


def matched_random(reference_count, k, seed, deletion_schedule=None):
    rng = np.random.default_rng(seed)
    if deletion_schedule is None:
        selected = np.sort(rng.choice(reference_count, k, replace=False))
        schedule = []
    else:
        schedule = [int(value) for value in deletion_schedule]
        assert all(value > 0 for value in schedule) and sum(schedule) == reference_count-k
        keep = np.ones(reference_count, dtype=bool)
        for count in schedule:
            candidates = np.flatnonzero(keep)
            keep[rng.choice(candidates, count, replace=False)] = False
        selected = np.flatnonzero(keep)
    return Selection('matched_random', selected, {'seed': int(seed), 'deletion_schedule': schedule,
        'matched_agent_endpoint': deletion_schedule is not None}).validated(reference_count, k)


def _finite_matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), name
    return values


def _unit(values):
    norm = np.linalg.norm(values)
    return values/norm if norm > 1e-15 else np.zeros_like(values)


def kcenter(values, k, seed=0, method='feature_kcenter'):
    values = _finite_matrix(values, method)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(values)))
    selected = [first]
    nearest = np.sum((values-values[first])**2, axis=1)
    nearest[first] = -1
    while len(selected) < k:
        nxt = int(np.argmax(nearest))
        selected.append(nxt)
        distance = np.sum((values-values[nxt])**2, axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[selected] = -1
    return Selection(method, np.array(selected, dtype=np.int64),
                     {'initialization_seed': int(seed), 'representation': method}).validated(len(values), k)


def kmeans_real_representatives(values, k, seed=0):
    values = _finite_matrix(values, 'kmeans')
    cluster_class = MiniBatchKMeans if len(values) > 10000 else KMeans
    kwargs = {'n_clusters': k, 'random_state': seed, 'n_init': 10}
    if cluster_class is MiniBatchKMeans:
        kwargs.update(batch_size=min(4096, len(values)), max_no_improvement=50)
    model = cluster_class(**kwargs)
    labels = model.fit_predict(values)
    selected = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        assert len(members), f'Empty cluster {cluster}'
        distance = np.sum((values[members]-model.cluster_centers_[cluster])**2, axis=1)
        selected.append(int(members[np.argmin(distance)]))
    return Selection('kmeans_real_representative', np.array(selected, dtype=np.int64),
        {'synthetic_centroids_used_for_inference': False, 'mapping': 'nearest real T node per cluster'}
    ).validated(len(values), k)


def _adjacency(edge_index, reference_count):
    if sparse.issparse(edge_index):
        adjacency = edge_index.tocsr().astype(np.float64)
        assert adjacency.shape == (reference_count, reference_count)
    else:
        edges = np.asarray(edge_index, dtype=np.int64)
        assert edges.ndim == 2 and edges.shape[0] == 2
        assert np.all((edges >= 0)&(edges < reference_count))
        adjacency = sparse.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                                      shape=(reference_count, reference_count)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def _diffusion(edge_index, reference_count):
    adjacency = _adjacency(edge_index, reference_count)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    assert np.all(degree > 0), 'SGGC/GDEM require every T node to have a graph neighbor'
    diffusion = .5*sparse.eye(reference_count, format='csr')+.5*sparse.diags(1/degree)@adjacency
    return adjacency, diffusion.tocsr()


def sggc_real_nodes(values, edge_index, k, kappa=.5):
    values = _finite_matrix(values, 'sggc_features')
    n = len(values)
    adjacency, diffusion = _diffusion(edge_index, n)
    norms = np.sqrt(np.asarray(diffusion.power(2).sum(axis=0)).ravel())
    assert np.all(norms > 0)
    columns = diffusion@sparse.diags(1/norms)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbor_degree = np.asarray(adjacency@degree).ravel()/degree
    signatures = np.column_stack((np.log1p(degree), np.log1p(neighbor_degree)))
    signatures = (signatures-signatures.mean(0))/np.where(signatures.std(0) > 0, signatures.std(0), 1)
    target = np.ones(n)/np.sqrt(n)
    current = np.zeros(n)
    selected = []
    for _ in range(k):
        if not selected:
            alignment = np.asarray(target@columns).ravel()
            candidates = np.flatnonzero(alignment >= kappa*alignment.max())
        else:
            residual = _unit(target-float(target@current)*current)
            projection = np.asarray(current@columns).ravel()
            residual_norm = np.sqrt(np.maximum(1-projection**2, 1e-15))
            alignment = np.asarray(residual@columns).ravel()/residual_norm
            alignment[selected] = -np.inf
            candidates = np.flatnonzero(alignment >= kappa*np.max(alignment))
            candidates = candidates[np.isfinite(alignment[candidates])]
        assert len(candidates)
        if selected:
            diversity = cdist(signatures[candidates], signatures[selected]).min(axis=1)
            nxt = int(candidates[np.argmax(diversity)])
        else:
            nxt = int(candidates[0])
        column = columns[:, nxt].toarray().ravel()
        if not selected:
            current = column
        else:
            zeta0 = float(target@column)
            zeta1 = float(target@current)
            zeta2 = float(column@current)
            numerator = zeta0-zeta1*zeta2
            denominator = numerator+zeta1-zeta0*zeta2
            eta = numerator/denominator if abs(denominator) > 1e-12 else .5
            eta = float(np.clip(eta, 0, 1))
            current = _unit((1-eta)*current+eta*column)
        selected.append(nxt)
    return Selection('SGGC_real_node_adaptation', np.array(selected, dtype=np.int64), {
        'kappa': float(kappa), 'source': 'https://arxiv.org/abs/2405.17404',
        'mechanism': 'diffusion spectral GIGA alignment with real-node topology-diversity refinement',
        'not_exact_original': 'classification CRAIG loss is not used; no AGB label enters selection'
    }).validated(n, k)


def _graph_eigenvectors(adjacency, rank):
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.eye(n, format='csr')-sparse.diags(1/np.sqrt(degree))@adjacency@sparse.diags(1/np.sqrt(degree))
    if n <= rank+2:
        vectors = np.linalg.eigh(laplacian.toarray())[1]
    else:
        vectors = eigsh(laplacian, k=rank, which='SM', return_eigenvectors=True, tol=1e-5)[1]
    return np.asarray(vectors[:, :rank], dtype=np.float64)


def gdem_real_nodes(values, edge_index, k, seed=0, steps=100, eigen_rank=8):
    import torch

    values = _finite_matrix(values, 'gdem_features')
    n, width = values.shape
    assert eigen_rank >= 1 and steps >= 1
    adjacency, _ = _diffusion(edge_index, n)
    eigenvectors = _graph_eigenvectors(adjacency, min(eigen_rank, n))
    scale = np.where(values.std(0) > 0, values.std(0), 1)
    x_real = (values-values.mean(0))/scale
    projection = eigenvectors.T@x_real
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-12)
    target_covariance = np.einsum('kd,ke->kde', projection, projection)
    rng = np.random.default_rng(seed)
    initial = rng.choice(n, k, replace=False)
    x_syn = torch.nn.Parameter(torch.as_tensor(x_real[initial], dtype=torch.float64))
    u_syn = torch.nn.Parameter(torch.as_tensor(eigenvectors[initial], dtype=torch.float64))
    target = torch.as_tensor(target_covariance, dtype=torch.float64)
    optimizer = torch.optim.Adam([x_syn, u_syn], lr=.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        projected = u_syn.T@x_syn
        projected = projected/torch.clamp(projected.norm(dim=0, keepdim=True), min=1e-12)
        covariance = torch.einsum('kd,ke->kde', projected, projected)
        eigenbasis_loss = torch.nn.functional.mse_loss(covariance, target)
        orthogonal_loss = torch.nn.functional.mse_loss(u_syn.T@u_syn, torch.eye(u_syn.shape[1], dtype=torch.float64))
        loss = eigenbasis_loss+.1*orthogonal_loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    real = np.column_stack((x_real, eigenvectors))
    synthetic = np.column_stack((x_syn.detach().numpy(), u_syn.detach().numpy()))
    joint_scale = np.where(real.std(0) > 0, real.std(0), 1)
    costs = cdist(synthetic/joint_scale, real/joint_scale, metric='sqeuclidean')
    rows, columns = linear_sum_assignment(costs)
    assert len(rows) == k and len(np.unique(columns)) == k
    return Selection('GDEM_real_node_adaptation', np.sort(columns.astype(np.int64)), {
        'seed': int(seed), 'optimization_steps': int(steps), 'eigen_rank': int(eigenvectors.shape[1]),
        'source': 'https://github.com/liuyang-tian/GDEM',
        'mechanism': 'feature covariance matching in graph eigenbasis; Hungarian mapping to unique real T nodes',
        'not_exact_original': 'regression adaptation omits class term; synthetic tensors never enter inference'
    }).validated(n, k)


@dataclass(frozen=True)
class ReducedReference:
    reference_x: np.ndarray
    reference_ids: np.ndarray
    coords: np.ndarray
    ecology: np.ndarray
    method: str
    original_reference_count: int

    @classmethod
    def create(cls, reference_x, reference_ids, coords, ecology, selection, original_reference_count=None):
        selected = np.asarray(selection.selected, dtype=np.int64)
        reference_x = _finite_matrix(reference_x, 'reference_x')
        reference_ids = np.asarray(reference_ids).astype(str)
        coords = _finite_matrix(coords, 'coords')
        ecology = np.asarray(ecology).astype(str)
        assert len(reference_x) == len(reference_ids) == len(coords) == len(ecology)
        selection.validated(len(reference_x), len(selected))
        original = len(reference_x) if original_reference_count is None else int(original_reference_count)
        return cls(reference_x[selected].copy(), reference_ids[selected].copy(), coords[selected].copy(),
                   ecology[selected].copy(), selection.method, original)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, reference_x=self.reference_x, reference_ids=self.reference_ids,
                            coords=self.coords, ecology=self.ecology,
                            method=np.asarray(self.method), original_reference_count=np.asarray(self.original_reference_count))
        return path

    @classmethod
    def restore(cls, path):
        allowed = {'reference_x', 'reference_ids', 'coords', 'ecology', 'method', 'original_reference_count'}
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == allowed, f'Unexpected hidden arrays: {set(saved.files)-allowed}'
            values = {key: saved[key].copy() for key in saved.files}
        result = cls(values['reference_x'], values['reference_ids'], values['coords'], values['ecology'],
                     str(values['method']), int(values['original_reference_count']))
        assert 0 < len(result.reference_ids) <= result.original_reference_count
        return result


def rebuild_reference_graph(reduced, query_x, k=8):
    query_x = np.asarray(query_x, dtype=np.float64)
    assert query_x.ndim == 2 and np.isfinite(query_x).all()
    reference_x = _finite_matrix(reduced.reference_x, 'reduced_reference_x')
    assert reference_x.shape[1] == query_x.shape[1] and len(reference_x) >= 2
    reference_count = len(reference_x)
    packed = np.concatenate((reference_x, query_x), axis=0)
    rows, columns = [], []
    reference_neighbors = NearestNeighbors(n_neighbors=min(k, reference_count-1)).fit(reference_x)
    indices = reference_neighbors.kneighbors(reference_x, return_distance=False)
    for row, neighbors in enumerate(indices):
        for column in neighbors:
            if row != column:
                rows.extend((row, column))
                columns.extend((int(column), row))
    query_neighbors = NearestNeighbors(n_neighbors=min(k, reference_count)).fit(reference_x)
    indices = query_neighbors.kneighbors(query_x, return_distance=False) if len(query_x) else []
    for offset, neighbors in enumerate(indices):
        row = reference_count+offset
        for column in neighbors:
            rows.append(row)
            columns.append(int(column))
    adjacency = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(packed), len(packed))).tocsr()
    assert adjacency.shape[0] == reference_count+len(query_x)
    return adjacency


def predict_all_queries(reduced, query_x, query_ids, frozen_predictor):
    query_ids = np.asarray(query_ids).astype(str)
    assert len(query_x) == len(query_ids) and len(set(query_ids)) == len(query_ids)
    graph = rebuild_reference_graph(reduced, query_x)
    prediction = np.asarray(frozen_predictor(reduced, query_x, query_ids, graph), dtype=np.float64)
    assert prediction.shape == (len(query_ids),) and np.isfinite(prediction).all()
    return prediction
