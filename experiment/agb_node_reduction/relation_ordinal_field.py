"""Mean and ordinal-target Jacobians for coupled local SPD relationship fields."""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import relation_coupled as coupled

r, a, layer = coupled.r, coupled.a, coupled.layer
TASK = 'CARBON-RELATION-ORDINAL-FIELD-001'
PREVIOUS = r.HERE/'runs/CARBON-RELATION-BLOCK-SPD-001/attempt-01'
CACHED = {arm: arm for arm in ('G0', 'GB', 'GS', 'GU', 'GP', 'GR')}
NEW = ('GE', 'GT', 'GL')
ARMS = (*CACHED, *NEW)
HEADS = 5


def target_channels(target):
    y = target.double()
    quantiles = torch.quantile(y, y.new_tensor([.2, .4, .6, .8]))
    binary = (y[:, None] <= quantiles).double()
    frequency = binary.mean(0)
    variance = frequency*(1-frequency)
    ordinal = (binary-frequency)/variance.sqrt().clamp_min(1e-12)
    return torch.cat((y[:, None], ordinal), 1), quantiles, frequency


def kernel_multi_gradient(support, target, query, tau):
    eye = torch.eye(support.shape[1], device=support.device, dtype=support.dtype)
    kernel = torch.exp(-a.distance(support, support, eye)/tau)
    system = kernel+.01*len(support)*torch.eye(len(support), device=support.device, dtype=support.dtype)
    center = target.mean(0)
    alpha = torch.linalg.solve(system, target-center)
    cross = torch.exp(-a.distance(query, support, eye)/tau)
    coefficient = cross[:, :, None]*alpha[None, :, :]
    gradient = (2/tau)*(torch.einsum('qnh,nd->qhd', coefficient, support)
                       -coefficient.sum(1)[:, :, None]*query[:, None, :])
    return center+cross@alpha, gradient


class OrdinalFieldSAGE(coupled.CoupledLayerSAGE):
    def __init__(self, arm, config):
        super().__init__(arm, config)
        n = len(self.field_support)
        self.field_directions = torch.zeros((n*HEADS, 185), dtype=torch.float64)
        self.field_log_norm = torch.zeros(n*HEADS, dtype=torch.float64)
        self.register_buffer('target_quantiles', torch.zeros(4, dtype=torch.float64))
        self.register_buffer('target_frequencies', torch.zeros(4, dtype=torch.float64))
        self.permutation_seed = 20261017+15800+100*r.REGIONS.index(config['region'])+config['fold']

    def field_weights(self, u):
        if self.arm == 'GE':
            return torch.softmax(self.field_log_norm, 0).expand(len(u), -1)
        eye = torch.eye(185, device=u.device, dtype=u.dtype)
        scores = (-a.distance(u, self.field_support, eye)/self.field_temperature).repeat_interleave(HEADS, 1)
        return torch.softmax(scores+self.field_log_norm, 1)

    def endpoint_weights(self, u, log_metric):
        if self.arm == 'GE':
            return self.field_weights(u), self.field_weights(self.field_support)
        root = (.5*log_metric).exp()
        query, support = u*root, self.field_support*root
        distance = torch.cdist(support, support, compute_mode='donot_use_mm_for_euclid_dist').square()
        mask = torch.eye(len(support), device=u.device, dtype=torch.bool)
        tau = distance.masked_fill(mask, torch.inf).topk(8, largest=False).values.mean().clamp_min(1e-8)
        scores = (-torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist').square()/tau).repeat_interleave(HEADS, 1)
        reference = (-distance/tau).repeat_interleave(HEADS, 1)
        return torch.softmax(scores+self.field_log_norm, 1), torch.softmax(reference+self.field_log_norm, 1)

    def corrected_distance(self, u, log_metric):
        w, wt = self.endpoint_weights(u, log_metric)
        root = (.5*log_metric).exp(); q = u*root; s = self.field_support*root
        scores = q@self.field_directions.T; reference = s@self.field_directions.T
        positive_left = (w*scores.square()).sum(1)[:, None]+w@reference.square().T
        positive_right = scores.square()@wt.T+(wt*reference.square()).sum(1)[None, :]
        left = positive_left-2*(w*scores)@reference.T
        right = positive_right-2*scores@(wt*reference).T
        base = torch.cdist(q, s, compute_mode='donot_use_mm_for_euclid_dist').square()
        distance = (185*(left+right)*.5+.01*base)/1.01
        # Bound the scale of cancellation, including the length of the dot products.
        positive_scale = (185*(positive_left+positive_right)*.5
                          +.01*(q.square().sum(1)[:, None]+s.square().sum(1)[None, :]))/1.01
        eps = torch.finfo(u.dtype).eps; terms = len(self.field_directions)+u.shape[1]
        gamma = terms*eps/(1-terms*eps)
        pairs = torch.nonzero(distance.detach() <= 8*gamma*positive_scale.detach()/1e-8)
        if len(pairs):
            direct = []
            for block in pairs.split(256):
                i, j = block.unbind(1)
                delta = (u[i]-self.field_support[j])*root
                projected = delta@self.field_directions.T
                direct.append((185*(projected.square()*(w[i]+wt[j])*.5).sum(1)
                               +.01*delta.square().sum(1))/1.01)
            distance = distance.index_put(tuple(pairs.unbind(1)), torch.cat(direct))
        diagonal = (185*(w@self.field_directions.square())+.01)/1.01*log_metric.exp()
        return distance.clamp_min(0), diagonal

    @torch.no_grad()
    def set_field(self, u, targets, tau):
        prediction, gradient = kernel_multi_gradient(u, targets, u, tau)
        g = gradient.reshape(-1, 185)
        norm = g.square().sum(1)
        positive = norm > 0
        assert positive.any() and torch.isfinite(g).all()
        self.field_support.copy_(u)
        self.field_directions.copy_(g/norm.sqrt().clamp_min(1e-150)[:, None])
        self.field_log_norm.copy_(torch.where(positive, norm.log(), -torch.inf))
        self.field_temperature.copy_(tau)
        self.field_ready.fill_(True)
        distance, diagonal = self.field_distance(u)
        distance.fill_diagonal_(torch.inf)
        self.temperature.copy_(distance.topk(8, largest=False).values.mean().clamp_min(1e-8))
        self._relation_cache.clear()
        energy = norm.reshape(len(u), HEADS).sum(0)
        return {'kernel_solves': 1, 'kernel_right_hand_sides': HEADS, 'zero_gradient_directions': int((~positive).sum()),
                'training_target_proxy_rmse': float((prediction-targets).square().mean().sqrt()),
                'field_min_diagonal': float(diagonal.min()), 'graph_temperature': float(self.temperature),
                'reference_temperature': float(tau), 'localized': self.arm != 'GE',
                **{f'head_{i}_gradient_energy_fraction': float(energy[i]/energy.sum()) for i in range(HEADS)}}

    @torch.no_grad()
    def fit_relations(self, x, target, edges):
        assert len(x) == len(target) == len(self.field_support) and int(edges.max()) < len(x)
        u = self.coordinates(x)
        targets, quantiles, frequencies = target_channels(target)
        self.target_quantiles.copy_(quantiles)
        self.target_frequencies.copy_(frequencies)
        if self.arm == 'GT':
            order = np.random.default_rng(self.permutation_seed).permutation(len(x))
            targets = targets[torch.as_tensor(order, device=x.device)]
        tau = a.bandwidth(u, torch.eye(185, device=x.device, dtype=torch.float64))
        record = self.set_field(u, targets, tau)
        record.update(permuted_targets=self.arm == 'GT', permutation_seed=self.permutation_seed,
                      **{f'target_quantile_{i}': float(value) for i, value in enumerate(quantiles)},
                      **{f'target_frequency_{i}': float(value) for i, value in enumerate(frequencies)})
        return [record]


def construct(arm, config, device):
    r.fix_seed(config['seed'], 1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train'] = int(data['n_train'])
    return OrdinalFieldSAGE(arm, config).to(device)


def algebra_checks():
    generator = torch.Generator().manual_seed(58)
    support = torch.randn((13, 7), generator=generator, dtype=torch.float64)
    query = torch.randn((3, 7), generator=generator, dtype=torch.float64, requires_grad=True)
    targets = torch.randn((13, HEADS), generator=generator, dtype=torch.float64)
    tau = torch.tensor(5., dtype=torch.float64)
    pred, gradient = kernel_multi_gradient(support, targets, query, tau)
    errors = []
    for h in range(HEADS):
        autodiff, = torch.autograd.grad(pred[:, h].sum(), query, retain_graph=True)
        single, analytic, _ = a.kernel_gradient(support, targets[:, h], query, torch.eye(7, dtype=torch.float64), tau)
        errors.extend([float((autodiff-gradient[:, h]).abs().max().detach()), float((single-pred[:, h]).abs().max().detach()),
                       float((analytic-gradient[:, h]).abs().max().detach())])
    assert max(errors) < 1e-12
    channels, _, frequency = target_channels(torch.ones(10))
    assert torch.equal(channels[:, 1:], torch.zeros((10, 4), dtype=torch.float64)) and (frequency == 1).all()
    return {'multioutput_gradient_and_scalar_solve_max_error': max(errors), 'constant_ordinal_columns_zero': True}


def check_graph(near, weight, diagonal, n):
    assert near.shape[1] == 8 and (near < n).all()
    assert (near[:n] != torch.arange(n, device=near.device)[:, None]).all()
    assert all(len(torch.unique(row)) == 8 for row in near)
    assert torch.isfinite(weight).all() and (weight >= 0).all()
    assert torch.allclose(weight.sum(1), torch.ones(len(weight), device=weight.device), atol=1e-6)
    assert torch.isfinite(diagonal).all() and (diagonal > 0).all()


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = algebra_checks()
    cases = []
    for region in r.REGIONS:
        _, config, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, y = r.tensors(data, pre, 'cuda')
        first = None
        for arm in NEW:
            model = construct(arm, config, 'cuda').eval(); model.fit_reference(x[:n], te)
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            with torch.no_grad(): initial = model(x, n, ee)[0]
            if first is None: first = (r.cpu_state(model.backbone), rng, initial)
            else:
                assert all(torch.equal(v, first[0][k]) for k, v in r.cpu_state(model.backbone).items())
                assert all(torch.equal(p, q) for p, q in zip(rng, first[1])) and torch.equal(initial, first[2])
            parent = coupled.construct('GE' if arm == 'GE' else 'GL', config, 'cuda').eval()
            parent.fit_reference(x[:n], te); parent.fit_relations(x[:n], y, te)
            u = model.coordinates(x)
            model.set_field(u[:n], y.double()[:, None].expand(-1, HEADS), parent.field_temperature)
            log_metric = torch.linspace(-.15, .2, 185, device=x.device, dtype=torch.float64)
            with torch.no_grad():
                expanded, scalar = model.corrected_distance(u, log_metric), parent.corrected_distance(u, log_metric)
                scalar_error = max(float((p-q).abs().max()) for p, q in zip(expanded, scalar))
                scalar_relative = max(float(((p-q).abs()/(1+q.abs())).max()) for p, q in zip(expanded, scalar))
                assert all(torch.allclose(p, q, atol=1e-7, rtol=1e-6) for p, q in zip(expanded, scalar))
                for logarithm in (log_metric, -log_metric):
                    new_graph = model.graph(u, logarithm, torch.float32)
                    old_graph = parent.graph(u, logarithm, torch.float32)
                    assert torch.equal(new_graph[0], old_graph[0])
                    assert torch.allclose(new_graph[1], old_graph[1], atol=1e-6, rtol=1e-6)
                assert torch.allclose(model(x, n, ee)[0], parent(x, n, ee)[0], atol=3e-6, rtol=3e-6)
                assert torch.equal(model.filtered(x), parent.filtered(x))
            history = model.fit_relations(x[:n], y, te)
            frozen = {k: v.clone() for k, v in model.named_buffers()}
            optimizer = model.make_optimizer(config)
            for _ in range(8):
                model.train(); optimizer.zero_grad(); F.mse_loss(model(x[:n], n, te)[0], y).backward()
                assert torch.isfinite(model.layer_log_metric.grad).all() and (model.layer_log_metric.grad.norm(dim=1) > 0).all()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
            assert (model.layer_log_metric.detach().norm(dim=1) > 0).all()
            assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in frozen.items())
            model.eval()
            with torch.no_grad():
                full = model(x, n, ee); alone = model(x[:n], n, te); single = model(torch.cat((x[:n], x[n:n+1])), n, ee)
                assert torch.allclose(full[0][:n], alone[0], atol=3e-6, rtol=3e-6)
                assert torch.allclose(full[0][n], single[0][n], atol=3e-6, rtol=3e-6)
                graphs = model.layer_graphs(x, n)
                for graph in graphs: check_graph(*graph, n)
                for logarithm in model.layer_log_metric:
                    weights, _ = model.endpoint_weights(u[[n, 0]], logarithm)
                    root = torch.diag((.5*logarithm).exp()); directions = model.field_directions
                    matrices = [root@(185*directions.T@(w[:, None]*directions)+.01*torch.eye(185, device=x.device))/1.01@root for w in weights]
                    assert all(torch.linalg.eigvalsh(a.symmetric(m)).min() > 0 for m in matrices)
                    distance, diagonal = model.corrected_distance(u, logarithm); delta = u[n]-u[0]
                    assert torch.allclose(delta@((matrices[0]+matrices[1])*.5)@delta, distance[n, 0], atol=1e-8, rtol=1e-8)
                    assert torch.allclose(diagonal[n], matrices[0].diag(), atol=1e-8, rtol=1e-8)
            theta = model.layer_log_metric[0].detach().clone().requires_grad_()
            direction = torch.linspace(-1, 1, 185, device=x.device, dtype=torch.float64); direction /= direction.norm()
            def objective(t): return model.corrected_distance(u[n:n+3], t)[0].mean()
            gradient, = torch.autograd.grad(objective(theta), theta)
            numeric = (objective(theta.detach()+1e-6*direction)-objective(theta.detach()-1e-6*direction))/2e-6
            error = float((numeric-(gradient*direction).sum()).abs().detach()); assert error < 1e-5
            path = output/f'{region}_{arm}.pt'
            torch.save({'state_dict': r.cpu_state(model), 'config': config}, path)
            restored = construct(arm, config, 'cuda').eval()
            restored.load_state_dict(torch.load(path, weights_only=False)['state_dict'])
            with torch.no_grad(): assert torch.equal(full[0], restored(x, n, ee)[0])
            assert sum(p.numel() for p in model.parameters()) == 28275
            cases.append({'region': region, 'arm': arm, 'scalar_reduction_error': scalar_error,
                          'scalar_reduction_relative_error': scalar_relative, 'scalar_neighbors_exact_two_signs': True, 'distance_FD_error': error,
                          'eight_real_updates_and_inductive_restore': True, 'paired_initialization_rng': True, 'history': history})
            print(f'PREFLIGHT {region}/{arm}', flush=True)
    r.write_json(output/'preflight.json', {'source_sha256': r.digest(Path(__file__)), 'checks': checks, 'cases': cases,
                 'wall_seconds': time.perf_counter()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


@torch.no_grad()
def export(output):
    started = time.perf_counter(); rows, points, hashes = [], [], {}
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold); x, _, _, _ = r.tensors(data, pre, 'cuda'); n = int(data['n_train'])
            for arm in NEW:
                directory = output/region/f'fold-{fold:02d}'/arm
                saved = torch.load(directory/'best.pt', map_location='cuda', weights_only=False)
                model = construct(arm, saved['config'], 'cuda').eval(); model.load_state_dict(saved['state_dict'])
                with np.load(output/region/f'fold-{fold:02d}'/'GP/layer_graphs.npz') as z:
                    parent = [z[f'layer{i}_neighbors'].copy() for i in (1, 2)]
                arrays = {'sample_ids': data['sample_ids'], 'n_train': n, 'layer_log_metric': model.layer_log_metric.cpu().numpy()}
                changes = []
                for index, (near, weight, diagonal) in enumerate(model.layer_graphs(x, n), 1):
                    check_graph(near, weight, diagonal, n)
                    nn, ww, dd = [v.cpu().numpy() for v in (near, weight, diagonal)]
                    change = np.array([1-len(set(p)&set(q))/8 for p, q in zip(nn[n:], parent[index-1][n:])]); changes.append(change)
                    edges = np.stack((nn.ravel(), np.repeat(np.arange(len(x)), 8)))
                    arrays.update({f'layer{index}_neighbors': nn, f'layer{index}_weights': ww, f'layer{index}_local_metric_diagonal': dd,
                                   f'layer{index}_eval_edges': edges, f'layer{index}_train_edges': edges[:, :n*8]})
                    lognorm = model.field_log_norm.reshape(n, HEADS)
                    fractions = torch.softmax(lognorm.flatten(), 0).reshape(n, HEADS).sum(0)
                    rows.append({'region': region, 'fold': fold, 'arm': arm, 'layer': index,
                                 'log_metric_norm': float(model.layer_log_metric[index-1].norm()),
                                 'validation_neighbor_change_from_GP': float(change.mean()),
                                 **{f'head_{h}_energy_fraction': float(fractions[h]) for h in range(HEADS)}})
                with np.load(directory/'responses.npz') as z:
                    assert np.array_equal(z['neighbors'], arrays['layer1_neighbors']) and np.array_equal(z['weights'], arrays['layer1_weights'])
                    assert np.array_equal(z['local_metric'], arrays['layer1_local_metric_diagonal'])
                path = directory/'layer_graphs.npz'; np.savez_compressed(path, **arrays); hashes[str(path.relative_to(output))] = r.digest(path)
                for i, sid in enumerate(data['sample_ids'][n:]):
                    points.append({'region': region, 'fold': fold, 'arm': arm, 'sample_id': str(sid),
                                   'layer1_neighbor_change_from_GP': float(changes[0][i]), 'layer2_neighbor_change_from_GP': float(changes[1][i])})
    assert len(rows) == 90 and len(points) == 4263
    r.write_csv(output/'ordinal_geometry_diagnostics.csv', rows); r.write_csv(output/'all_ids_ordinal_geometry.csv', points)
    r.write_json(output/'graph_export.json', {'layers': 90, 'method_IDs': 4263, 'graph_hashes': hashes,
                 'wall_seconds': time.perf_counter()-started, 'first_layer_response_exact': True, 'no_GNN_forward_replay': True})


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text())
    assert len(check['cases']) == 9 and check['source_sha256'] == r.digest(Path(__file__))
    old = json.loads((PREVIOUS/'manifest.json').read_text())
    inputs = {p: v for p, v in old['source_hashes'].items() if '/data/' in p.replace('\\', '/') or p.endswith('config.json')}
    assert len(inputs) == 60
    for path, value in inputs.items(): assert r.digest(r.ROOT/path) == value
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), Path(coupled.__file__), Path(layer.__file__), Path(layer.local.__file__),
               Path(layer.unroll.__file__), Path(a.__file__), Path(a.b.__file__), Path(r.__file__), r.BASE/'baseline.py']
    hashes = {str(p.relative_to(r.ROOT)): r.digest(p) for p in sources}; hashes.update(inputs)
    cached, references = {}, {}
    for region in r.REGIONS:
        for fold in range(1, 6):
            for alias, original in CACHED.items():
                src = PREVIOUS/region/f'fold-{fold:02d}'/original
                for p in src.iterdir():
                    if p.is_file(): cached[str(p.relative_to(r.ROOT))] = r.digest(p)
                shutil.copytree(src, output/region/f'fold-{fold:02d}'/alias)
            shutil.copyfile(PREVIOUS/region/f'fold-{fold:02d}'/'internal_split.csv', output/region/f'fold-{fold:02d}'/'internal_split.csv')
            p = layer.local.SOURCE/region/f'fold-{fold:02d}'/'GE/best.pt'; references[str(p.relative_to(r.ROOT))] = r.digest(p)
            assert r.digest(p) == r.digest(output/region/f'fold-{fold:02d}'/'GS/best.pt')
    for path in sources: (output/path.name).write_bytes(path.read_bytes())
    r.write_json(output/'manifest.json', {'task_id': TASK, 'branch': 'RELATION/v0.58', 'started_at': datetime.now().astimezone().isoformat(),
        'argv': sys.argv, 'device': 'cuda', 'source_hashes': hashes, 'cached_hashes': cached, 'reference_geometry_hashes': references,
        'cache_source': str(PREVIOUS.relative_to(r.ROOT)), 'cache_aliases': CACHED, 'new_fits': 45, 'cached_fits': 90,
        'final_holdout_accessed': False, 'preflight_sha256': r.digest(preflight_dir/'preflight.json'), 'maximum_GNN_updates': 400,
        'model_parameters': 28275, 'geometry_parameters': 370, 'kernel_target_channels': HEADS, 'original_MSE_and_messages': True})
    started = time.perf_counter()
    for fold in range(1, 6):
        for region in r.REGIONS:
            for arm in NEW: r.fit(arm, region, fold, output, 'cuda', model_factory=construct, task_id=TASK, max_gnn_epochs=400)
    for path, value in {**hashes, **cached, **references}.items(): assert r.digest(r.ROOT/path) == value
    r.write_json(output/'execution.json', {'finished_at': datetime.now().astimezone().isoformat(), 'wall_seconds': time.perf_counter()-started,
                 'new_fits': 45, 'cached_fits': 90, 'all_inputs_unchanged': True})
    export(output); layer.summarize(output, arms=ARMS, new_fits=45, cached_fits=90)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('preflight', 'run'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight': preflight(args.output)
    else: run(args.output, args.preflight_dir)
