"""Train-only relation learning for the existing AGB GraphSAGE predictor."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'experiment/10.14'
sys.path.insert(0, str(BASE))
from baseline import Regressor, digest, fix_seed, metrics, write_csv, write_json

SOURCE = BASE / 'runs/CARBON-THREE-REGION-001/attempt-01'
REGIONS = ('hunan', 'hubei', 'guizhou')
ARMS = ('G0', 'GE', 'GL')


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_fold(region, fold):
    source = SOURCE / region / f'fold-{fold:02d}'
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    with np.load(source / 'data/development.npz', allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    with np.load(source / 'data/preprocessor.npz', allow_pickle=False) as z:
        pre = {key: z[key] for key in z.files}
    n = int(data['n_train'])
    assert data['x'].shape[1] == 185 and np.isfinite(data['x']).all()
    assert len(set(data['sample_ids'])) == len(data['y'])
    assert np.all(data['eval_edges'][0] < n)
    assert np.all(data['train_edges'] < n)
    assert config['seed'] == 20261014 + fold - 1
    return source, config, data, pre


class RelationSAGE(nn.Module):
    def __init__(self, arm, config):
        super().__init__()
        self.arm = arm
        self.k = config['knn_k']
        # Construct the unchanged backbone before any new random draws.
        self.backbone = Regressor('graphsage', config)
        if arm != 'G0':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config['seed'] + 100000)
                self.projection = nn.Linear(185, 16)
                self.local = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
                nn.init.zeros_(self.local[-1].weight)
                nn.init.zeros_(self.local[-1].bias)

    def relations(self, x, n_train, edges):
        if self.arm == 'G0':
            neighbors = edges[0].reshape(len(x), self.k)
            weights = torch.full((len(x), self.k), 1 / self.k, device=x.device)
            return neighbors, weights, torch.ones((len(x), 16), device=x.device)
        z = F.normalize(self.projection(x), dim=1)
        if self.arm == 'GE':
            z = F.normalize(z + self.local(z), dim=1)
            g = torch.ones_like(z)
        else:
            a = 0.5 * torch.tanh(self.local(z))
            g = torch.exp(a - a.mean(dim=1, keepdim=True))
        # Symmetric diagonal quadratic distance, without an N x N x 16 tensor.
        zt, gt = z[:n_train], g[:n_train]
        distance = 0.5 * (
            (g * z.square()).sum(1)[:, None] + (gt * zt.square()).sum(1)[None, :]
            + g @ zt.square().T + z.square() @ gt.T
            - 2 * (g * z) @ zt.T - 2 * z @ (gt * zt).T
        )
        distance = distance.clamp_min(0)
        self_mask = torch.arange(len(x), device=x.device)[:, None] == torch.arange(n_train, device=x.device)
        distance = distance.masked_fill(self_mask, torch.inf)
        values, neighbors = torch.topk(distance, self.k, dim=1, largest=False, sorted=True)
        weights = torch.softmax(-values, dim=1)
        return neighbors, weights, g

    @staticmethod
    def weighted_layer(layer, x, neighbors, weights):
        aggregate = (x[neighbors] * weights[:, :, None]).sum(dim=1)
        return layer.lin_l(aggregate) + layer.lin_r(x)

    def forward(self, x, n_train, edges):
        neighbors, weights, g = self.relations(x, n_train, edges)
        if self.arm == 'G0':
            prediction, embedding = self.backbone(x, edges)
        else:
            h = self.weighted_layer(self.backbone.first, x, neighbors, weights)
            h = self.backbone.dropout(torch.relu(h))
            h = self.weighted_layer(self.backbone.second, h, neighbors, weights)
            embedding = torch.relu(h)
            prediction = self.backbone.head(embedding).squeeze(-1)
        return prediction, embedding, neighbors, weights, g


def construct(arm, config, device):
    fix_seed(config['seed'], 1)
    return RelationSAGE(arm, config).to(device)


def tensors(data, pre, device):
    n = int(data['n_train'])
    return (torch.as_tensor(data['x'], device=device),
            torch.as_tensor(data['train_edges'], device=device),
            torch.as_tensor(data['eval_edges'], device=device),
            torch.tensor((data['y'][:n] - pre['y_mean']) / pre['y_scale'],
                         dtype=torch.float32, device=device))


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    source, config, data, pre = load_fold('hunan', 1)
    n = int(data['n_train'])
    x, train_edges, eval_edges, target = tensors(data, pre, 'cpu')
    models, rng_states = {}, {}
    for arm in ARMS:
        models[arm] = construct(arm, config, 'cpu')
        rng_states[arm] = (torch.get_rng_state().clone(),
                           torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None)
    for arm in ('GE', 'GL'):
        assert torch.equal(rng_states['G0'][0], rng_states[arm][0])
        if torch.cuda.is_available():
            assert torch.equal(rng_states['G0'][1], rng_states[arm][1])
    for arm in ('GE', 'GL'):
        for key, value in models['G0'].backbone.state_dict().items():
            assert torch.equal(value, models[arm].backbone.state_dict()[key]), key
    for key, value in models['GE'].projection.state_dict().items():
        assert torch.equal(value, models['GL'].projection.state_dict()[key])
    g0 = models['G0'].eval()
    with torch.no_grad():
        neighbors, weights, _ = g0.relations(x, n, eval_edges)
        h = g0.weighted_layer(g0.backbone.first, x, neighbors, weights)
        h = torch.relu(h)
        h = torch.relu(g0.weighted_layer(g0.backbone.second, h, neighbors, weights))
        weighted = g0.backbone.head(h).squeeze(-1)
        original = g0(x, n, eval_edges)[0]
    equal_diff = float((weighted - original).abs().max())
    assert torch.allclose(weighted, original, rtol=2e-6, atol=2e-6)
    checks = {'equal_weight_max_scaled_difference': equal_diff, 'paired_backbone': True,
              'paired_projection': True, 'paired_cpu_rng': True,
              'paired_cuda_rng': True if torch.cuda.is_available() else None,
              'source': str(source.relative_to(ROOT)), 'arms': {}}
    for arm in ('GE', 'GL'):
        model = models[arm]
        before = cpu_state(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
        for step in range(3):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, _, near, weight, g = model(x[:n], n, train_edges)
            loss = F.mse_loss(pred, target)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            gradients = {key: float(p.grad.norm()) for key, p in model.named_parameters()
                         if key.startswith(('projection', 'local'))}
            assert gradients['projection.weight'] > 0 and gradients['local.2.weight'] > 0
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full = model(x, n, eval_edges)
            train = model(x[:n], n, train_edges)
            isolated = model(torch.cat((x[:n], x[n:n+1])), n, eval_edges[:, :n*8+8])
        assert torch.allclose(full[0][:n], train[0], atol=2e-6, rtol=2e-6)
        assert torch.allclose(full[0][n:n+1], isolated[0][n:], atol=2e-6, rtol=2e-6)
        assert torch.all(full[2] < n) and torch.all(full[2][:n] != torch.arange(n)[:, None])
        assert torch.allclose(full[3].sum(1), torch.ones(len(x)), atol=1e-6)
        assert torch.isfinite(full[4]).all() and torch.all(full[4] > 0)
        assert torch.allclose(torch.log(full[4]).sum(1), torch.zeros(len(x)), atol=2e-6)
        checkpoint = output / (arm + '.pt')
        torch.save(cpu_state(model), checkpoint)
        restored = construct(arm, config, 'cpu')
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        restored.eval()
        with torch.no_grad():
            assert torch.equal(restored(x, n, eval_edges)[0], full[0])
        changes = {key: float((value.detach().cpu() - before[key]).abs().max())
                   for key, value in model.named_parameters() if key.startswith(('projection', 'local'))}
        checks['arms'][arm] = {'parameters': sum(p.numel() for p in model.parameters()),
                              'gradient_norms': gradients, 'parameter_changes': changes,
                              'query_addition_max_difference': float((full[0][:n] - train[0]).abs().max()),
                              'weights_std': float(full[3].std()), 'metric_min': float(full[4].min()),
                              'metric_max': float(full[4].max()), 'checkpoint_exact': True}
    benchmark = {}
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        model = construct('GL', config, device)
        xd, ted, eed, yd = tensors(data, pre, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for step in range(40):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(xd[:n], n, ted)[0]
            F.mse_loss(pred, yd).backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            model.eval()
            with torch.no_grad():
                prediction = model(xd, n, eed)[0].cpu().numpy()
                metrics(data['y'][n:], prediction[n:].astype(float) * pre['y_scale'] + pre['y_mean'])
            if device == 'cuda':
                torch.cuda.synchronize()
            if step >= 10:
                samples.append(time.perf_counter() - start)
        benchmark[device] = {'median_epoch_seconds': float(np.median(samples)),
                             'mean_epoch_seconds': float(np.mean(samples)), 'measured_epochs': 30,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
        del model, optimizer, xd, ted, eed, yd
        if device == 'cuda':
            torch.cuda.empty_cache()
    # CPU four-worker throughput is measured separately, not assumed to be 4x.
    chosen = min(benchmark, key=lambda dev: benchmark[dev]['median_epoch_seconds'])
    checks.update(benchmark=benchmark, selected_device=chosen, recorded_at=datetime.now().astimezone().isoformat(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    write_json(output / 'preflight.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


def fit(arm, region, fold, output, device, model_factory=None, task_id='CARBON-RELATION-001',
        warmup_steps=0, auxiliary_weight=0, max_gnn_epochs=None, seed_offset=0):
    source, config, data, pre = load_fold(region, fold)
    if seed_offset:
        config = {**config, 'seed': config['seed'] + seed_offset}
    out = output / region / f'fold-{fold:02d}' / arm
    if (out / 'metrics.json').exists():
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    model = (model_factory or construct)(arm, config, device)
    n = int(data['n_train'])
    x, train_edges, eval_edges, target = tensors(data, pre, device)
    if hasattr(model, 'prepare_inputs'):
        x = model.prepare_inputs(x, data['missing_mask'])
    if hasattr(model, 'fit_reference'):
        model.fit_reference(x[:n], train_edges)
    optimizer = (model.make_optimizer(config) if hasattr(model, 'make_optimizer') else
                 torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']))
    initial = cpu_state(model)
    model.eval()
    with torch.no_grad():
        initial_neighbors = model(x, n, eval_edges)[2].cpu().numpy()
    best, stale, best_epoch, best_state = float('inf'), 0, 0, None
    history = []
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if hasattr(model, 'fit_relations'):
        relation_history = model.fit_relations(x[:n], target, train_edges)
        write_csv(out / 'relation_fit.csv', relation_history)
    warmup_history = []
    for step in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        auxiliary = model.metric_loss(x[:n], target, train_edges)
        assert torch.isfinite(auxiliary)
        auxiliary.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()
        warmup_history.append({'step': step + 1, getattr(model, 'warmup_loss_name', 'training_leave_one_out_mse'): float(auxiliary.detach())})
    if warmup_history:
        write_csv(out / 'warmup.csv', warmup_history)
    gnn_budget = config['max_epochs'] - warmup_steps if max_gnn_epochs is None else max_gnn_epochs
    for epoch in range(1, gnn_budget + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, 'training_loss'):
            task_loss = model.training_loss(x[:n], target, train_edges)
        else:
            prediction = model(x[:n], n, train_edges)[0]
            task_loss = F.mse_loss(prediction, target)
        auxiliary = model.metric_loss(x[:n], target, train_edges) if auxiliary_weight else None
        loss = task_loss + auxiliary_weight * auxiliary if auxiliary is not None else task_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss: {region}/{fold}/{arm}/{epoch}')
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        if not torch.isfinite(gradient):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        model.eval()
        if hasattr(model, 'fit_prediction_head'):
            model.fit_prediction_head(x[:n], target, train_edges)
        with torch.no_grad():
            scaled = model(x, n, eval_edges)[0].cpu().numpy().astype(float)
        pred = scaled * pre['y_scale'] + pre['y_mean']
        mt, mv = metrics(data['y'][:n], pred[:n]), metrics(data['y'][n:], pred[n:])
        history.append({'epoch': epoch, 'train_loss_standardized': float(loss.detach()),
                        'task_mse_standardized': float(task_loss.detach()),
                        'auxiliary_loo_mse': float(auxiliary.detach()) if auxiliary is not None else 0.0,
                        'train_rmse': mt['rmse'], 'validation_rmse': mv['rmse'],
                        'validation_r2': mv['r2'], 'gradient_norm': float(gradient)})
        if mv['rmse'] < best - config['min_delta']:
            best, stale, best_epoch, best_state = mv['rmse'], 0, epoch, cpu_state(model)
        else:
            stale += 1
        if epoch % 50 == 0:
            write_csv(out / 'history.csv', history)
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scaled, embedding, near, weight, g = model(x, n, eval_edges)
        train_alone = model(x[:n], n, train_edges)[0]
        if not torch.allclose(scaled[:n], train_alone, atol=3e-6, rtol=3e-6):
            train_graph = model(x[:n], n, train_edges)
            details = {'max_scaled_difference': float((scaled[:n]-train_alone).abs().max()),
                       'changed_neighbor_entries': int((near[:n]!=train_graph[2]).sum())}
            torch.save({'state_dict': best_state, 'config': config, 'epoch': best_epoch}, out/'query_failure.pt')
            write_json(out/'query_failure.json', details)
            raise AssertionError(f'Inductive query contract failed: {details}')
    pred = scaled.cpu().numpy().astype(float) * pre['y_scale'] + pre['y_mean']
    neighbors, weights, metric = near.cpu().numpy(), weight.cpu().numpy(), g.cpu().numpy()
    assert np.isfinite(pred).all() and np.isfinite(metric).all() and (metric > 0).all()
    variable_graph = bool(getattr(model, 'variable_graph', False))
    assert neighbors.shape == (len(pred), n if variable_graph else 8) and (neighbors < n).all()
    active = weights > 0 if variable_graph else np.ones_like(weights, dtype=bool)
    targets = np.broadcast_to(np.arange(len(pred))[:, None], neighbors.shape)
    final_edges = np.stack([neighbors[active], targets[active]])
    if variable_graph:
        assert not np.any(final_edges[0] == final_edges[1])
        assert np.isfinite(weights).all() and (weights >= 0).all() and np.allclose(weights.sum(1), 1, atol=1e-6)
    checkpoint = {'arm': arm, 'state_dict': best_state, 'config': config, 'epoch': best_epoch,
                  'task_id': task_id, 'device': device, 'max_gnn_epochs': gnn_budget,
                  'split_sha256': digest(source / 'data/split.csv'), 'source_sha256': digest(Path(__file__))}
    torch.save(checkpoint, out / 'best.pt')
    np.savez_compressed(out / 'responses.npz', prediction=pred, embedding=embedding.cpu().numpy(),
                        absolute_residual=np.abs(pred - data['y']), sample_ids=data['sample_ids'],
                        train_edges=final_edges[:, final_edges[1] < n], eval_edges=final_edges,
                        neighbors=neighbors, weights=weights, local_metric=metric)
    write_csv(out / 'history.csv', history)
    write_csv(out / 'predictions.csv', [
        {'sample_id': str(sid), 'role': 'train' if i < n else 'validation',
         'observed_agb': float(data['y'][i]), 'predicted_agb': float(pred[i]),
         'residual': float(pred[i] - data['y'][i])} for i, sid in enumerate(data['sample_ids'])])
    old_neighbors = data['eval_edges'][0].reshape(len(pred), 8)
    active_neighbors = [row[mask] for row, mask in zip(neighbors, active)]
    overlap_original = np.array([len(set(a) & set(b))/8 for a, b in zip(active_neighbors, old_neighbors)])
    overlap_initial = np.array([len(set(a) & set(b))/8 for a, b in zip(active_neighbors, initial_neighbors)])
    relation_change = {key: float((value - initial[key]).abs().max()) for key, value in best_state.items()
                       if key.startswith(('projection', 'local', 'conditioner'))}
    probability_weights = bool((weights >= 0).all() and np.allclose(weights.sum(1), 1, atol=1e-5))
    absolute_mass = np.abs(weights) / np.abs(weights).sum(1, keepdims=True).clip(1e-30)
    result = {'region': region, 'fold': fold, 'arm': arm, 'seed': config['seed'], 'device': device,
              'train': metrics(data['y'][:n], pred[:n]), 'validation': metrics(data['y'][n:], pred[n:]),
              'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
              'epochs_run': epoch, 'warmup_steps': warmup_steps, 'total_updates': epoch + warmup_steps,
              'max_gnn_epochs': gnn_budget,
              'training_objective': getattr(model, 'training_objective', 'training_mse'),
              'auxiliary_weight': auxiliary_weight, 'elapsed_seconds': time.perf_counter() - start,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'max_train_query_addition_difference': float((scaled[:n] - train_alone).abs().max()),
              'edge_replacement_from_original': float(1 - overlap_original.mean()),
              'edge_replacement_from_initial': float(1 - overlap_initial.mean()),
              'mean_weight_entropy': float(-(weights * np.log(weights.clip(1e-30))).sum(1).mean()) if probability_weights else None,
              'mean_abs_weight_entropy': float(-(absolute_mass*np.log(absolute_mass.clip(1e-30))).sum(1).mean()),
              'weight_semantics': 'probabilities' if probability_weights else 'signed_or_unnormalized_coefficients',
              'negative_weight_fraction': float((weights < 0).mean()), 'mean_weight_sum': float(weights.sum(1).mean()),
              'metric_min': float(metric.min()), 'metric_max': float(metric.max()),
              'relation_parameter_changes': relation_change,
              'checkpoint_sha256': digest(out / 'best.pt'), 'prediction_sha256': digest(out / 'predictions.csv')}
    if hasattr(optimizer, 'diagnostics'):
        result['optimizer_diagnostics'] = optimizer.diagnostics()
        p = model.projection.weight.detach()
        result['projection_orthogonality_error'] = float((p@p.T-torch.eye(len(p),device=p.device)).abs().max())
    if variable_graph:
        result['neighbor_table_semantics'] = 'all_training_candidates; only positive weights are edges; first layer only'
        result['active_neighbor_count_mean'] = float(active.sum(1).mean())
        result['active_neighbor_count_min'] = int(active.sum(1).min())
        result['active_neighbor_count_max'] = int(active.sum(1).max())
    write_json(out / 'metrics.json', result)
    print(f"DONE {region}/fold-{fold:02d}/{arm} RMSE={result['validation']['rmse']:.6f} "
          f"R2={result['validation']['r2']:.6f} epoch={best_epoch}/{epoch} "
          f"seconds={result['elapsed_seconds']:.2f}", flush=True)
    return result


def fold_job(args):
    region, fold, output, device = args
    return [fit(arm, region, fold, Path(output), device) for arm in ARMS]


def run(output, preflight_dir):
    check = json.loads((preflight_dir / 'preflight.json').read_text(encoding='utf-8'))
    device = check['selected_device']
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
              'experiment/10.14/baseline.py': digest(BASE / 'baseline.py'),
              'experiment/agb_node_reduction/distance_shared_agent.py': digest(HERE / 'distance_shared_agent.py'),
              'experiment/agb_node_reduction/distance_agent.py': digest(HERE / 'distance_agent.py')}
    for region in REGIONS:
        for fold in range(1, 6):
            source = SOURCE / region / f'fold-{fold:02d}'
            for name in ('config.json', 'data/development.npz', 'data/preprocessor.npz', 'data/split.csv'):
                path = source / name
                hashes[str(path.relative_to(ROOT))] = digest(path)
    manifest = {'task_id': 'CARBON-RELATION-001', 'branch': 'RELATION/v0.1',
                'started_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv,
                'device': device, 'workers': 4 if device == 'cpu' else 1,
                'source_hashes': hashes, 'python': sys.version, 'torch': torch.__version__,
                'preflight_sha256': digest(preflight_dir / 'preflight.json'),
                'final_holdout_accessed': False}
    write_json(output / 'manifest.json', manifest)
    (output / 'relation_graphsage.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    jobs = [(region, fold, str(output), device) for fold in range(1, 6) for region in REGIONS]
    if device == 'cpu':
        with ProcessPoolExecutor(max_workers=4, initializer=setup) as executor:
            pending = [executor.submit(fold_job, job) for job in jobs]
            for future in as_completed(pending):
                future.result()
    else:
        for job in jobs:
            for arm in ARMS:
                fit(arm, job[0], job[1], output, device)
    for path, value in hashes.items():
        assert digest(ROOT / path) == value, path
    write_json(output / 'execution.json', {'finished_at': datetime.now().astimezone().isoformat(),
                                         'wall_seconds': time.perf_counter() - started, 'all_inputs_unchanged': True})
    summarize(output)


def summarize(output):
    rows, id_rows = [], []
    for region in REGIONS:
        for fold in range(1, 6):
            _, _, data, _ = load_fold(region, fold)
            n = int(data['n_train'])
            results, predictions = {}, {}
            for arm in ARMS:
                out = output / region / f'fold-{fold:02d}' / arm
                result = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
                results[arm] = result
                with np.load(out / 'responses.npz', allow_pickle=False) as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    predictions[arm] = z['prediction'].copy()
                calculated = metrics(data['y'][n:], predictions[arm][n:])
                for key in calculated:
                    assert np.isclose(calculated[key], result['validation'][key], atol=1e-10)
                rows.append({'region': region, 'fold': fold, 'arm': arm,
                             **result['validation'], 'best_epoch': result['best_epoch'],
                             'elapsed_seconds': result['elapsed_seconds'],
                             'edge_replacement': result['edge_replacement_from_original']})
            for i in range(n, len(data['y'])):
                observed = float(data['y'][i])
                item = {'region': region, 'fold': fold, 'sample_id': str(data['sample_ids'][i]),
                        'observed_agb': observed, 'missing_features': int(data['missing_mask'][i].sum()),
                        'high_agb_train_q80': bool(observed >= np.quantile(data['y'][:n], .8))}
                for arm in ARMS:
                    item[arm + '_prediction'] = float(predictions[arm][i])
                    item[arm + '_absolute_error'] = abs(item[arm + '_prediction'] - observed)
                item['GL_absolute_error_gain_vs_G0'] = item['G0_absolute_error'] - item['GL_absolute_error']
                item['GL_absolute_error_gain_vs_GE'] = item['GE_absolute_error'] - item['GL_absolute_error']
                id_rows.append(item)
    assert len(id_rows) == 1421 and len({(x['region'], x['sample_id']) for x in id_rows}) == 1421
    write_csv(output / 'fold_metrics.csv', rows)
    write_csv(output / 'all_1421_development_ids.csv', id_rows)
    grouped, pairs = {}, []
    for region in REGIONS:
        grouped[region] = {}
        for arm in ARMS:
            subset = [r for r in rows if r['region'] == region and r['arm'] == arm]
            grouped[region][arm] = {key: float(np.mean([r[key] for r in subset]))
                                    for key in ('rmse', 'mae', 'r2', 'best_epoch', 'elapsed_seconds', 'edge_replacement')}
        for fold in range(1, 6):
            chosen = {r['arm']: r for r in rows if r['region'] == region and r['fold'] == fold}
            pairs.append({'region': region, 'fold': fold,
                          'GL_vs_G0_rmse_gain_percent': 100 * (1-chosen['GL']['rmse']/chosen['G0']['rmse']),
                          'GL_vs_GE_rmse_gain_percent': 100 * (1-chosen['GL']['rmse']/chosen['GE']['rmse']),
                          'GE_vs_G0_rmse_gain_percent': 100 * (1-chosen['GE']['rmse']/chosen['G0']['rmse'])})
    gain_g0 = float(np.mean([p['GL_vs_G0_rmse_gain_percent'] for p in pairs]))
    gain_ge = float(np.mean([p['GL_vs_GE_rmse_gain_percent'] for p in pairs]))
    by_region_gain = {r: float(np.mean([p['GL_vs_G0_rmse_gain_percent'] for p in pairs if p['region'] == r])) for r in REGIONS}
    by_region_r2 = {r: grouped[r]['GL']['r2']-grouped[r]['G0']['r2'] for r in REGIONS}
    full = gain_g0 >= 3 and gain_ge >= .5 and min(by_region_gain.values()) >= -.0001 and min(by_region_r2.values()) >= -1e-6
    pareto = gain_g0 >= 1 and gain_ge > 0 and min(by_region_gain.values()) >= -1 and min(by_region_r2.values()) >= -.01
    summary = {'province_means': grouped, 'GL_vs_G0_gain_percent': gain_g0, 'GL_vs_GE_gain_percent': gain_ge,
               'GE_vs_G0_gain_percent': float(np.mean([p['GE_vs_G0_rmse_gain_percent'] for p in pairs])),
               'province_GL_vs_G0_gain_percent': by_region_gain, 'province_GL_vs_G0_r2_change': by_region_r2,
               'classification': 'FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
               'folds': 15, 'fits': 45, 'development_ids': len(id_rows), 'final_test_used': False,
               'GL_better_IDs_vs_G0': sum(r['GL_absolute_error_gain_vs_G0'] > 0 for r in id_rows),
               'GL_better_IDs_vs_GE': sum(r['GL_absolute_error_gain_vs_GE'] > 0 for r in id_rows)}
    write_csv(output / 'paired_fold_gains.csv', pairs)
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'run', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args()
    setup()
    if args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output, args.preflight_dir)
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
