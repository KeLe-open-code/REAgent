"""MSE-preserving spherical trace supervision of learned GraphSAGE relations."""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import relation_membership as member

r, layer = member.r, member.layer
TASK = 'CARBON-RELATION-SPHERICAL-TRACE-001'
NEW = ('CE', 'GE', 'GL')
BASE = r.HERE / 'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01'
MATCH = r.HERE / 'runs/CARBON-RELATION-MEMBERSHIP-001/attempt-02'
CACHED = {'G0': (BASE, 'G0'), 'GR': (MATCH, 'GE'), 'L0': (MATCH, 'GL'),
          'GP': (BASE, 'GP'), 'GN': (BASE, 'GN')}


def spherical_terms(feature, bins, label_values, sphere):
    normalized = F.normalize(feature, dim=1)
    centers = F.normalize(torch.stack([normalized[bins == k].mean(0)
                                      for k in range(len(label_values))]), dim=1)
    envelope = 1 - (centers @ sphere.T).max(dim=0).values.mean()
    spacing = (label_values[1:] - label_values[:-1]).to(feature.dtype)
    assert (spacing > 0).all()
    homogeneous = ((centers[1:] - centers[:-1]).square().sum(1) / spacing).sum()
    contrast = F.cross_entropy(normalized @ centers.detach().T / .1, bins)
    return envelope, homogeneous, contrast, centers


class Optimizer(layer.JointOptimizer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.model = model

    def diagnostics(self):
        return {**super().diagnostics(), 'geometry_history_at_training_end_not_best_checkpoint': self.model.geometry_history}


class SphericalSAGE(member.MembershipSAGE):
    training_objective = 'MSE+prototype_contrast+0.01*spherical_envelope+0.01*ordered_homogeneity;CE_control_has_contrast_only'

    def __init__(self, variant, config):
        super().__init__('GL' if variant == 'GL' else 'GE', config)
        self.variant = variant
        self.register_buffer('label_bins', torch.zeros(config['relation_n_train'], dtype=torch.long))
        self.register_buffer('label_values', torch.zeros(10, dtype=torch.float64))
        self.register_buffer('label_edges', torch.zeros(9, dtype=torch.float32))
        generator = torch.Generator(device='cpu').manual_seed(config['seed'] + 87000)
        self.register_buffer('sphere_points', F.normalize(torch.randn(1000, 32, generator=generator), dim=1))
        self.geometry_history = []
        self.geometry_feature = None

    def fit_relations(self, x, target, edges):
        history = super().fit_relations(x, target, edges)
        with torch.no_grad():
            quantiles = torch.linspace(.1, .9, 9, device=target.device)
            self.label_edges.copy_(torch.quantile(target, quantiles))
            bins = torch.bucketize(target.contiguous(), self.label_edges)
            counts = torch.bincount(bins, minlength=10)
            assert (counts > 0).all()
            self.label_bins.copy_(bins)
            self.label_values.copy_(torch.stack([target[bins == k].double().mean() for k in range(10)]))
            assert (self.label_values[1:] > self.label_values[:-1]).all()
            history[0].update(label_bin_counts=counts.cpu().tolist(), label_bin_rule='training_quantiles10',
                              geometry_representation='signed_second_layer_before_ReLU', sphere_samples=1000)
        return history

    def forward(self, x, n_train, edges):
        if not bool(self.field_ready):
            return super().forward(x, n_train, edges)
        assert n_train == len(self.field_support)
        if self.training and torch.is_grad_enabled():
            u = self.coordinates(x)
            first = self.membership_graph(u, self.layer_log_metric[0])
            second = self.membership_graph(u, self.layer_log_metric[1])

            def apply(module, values, graph):
                near, hard, _, membership, coefficient = graph
                aggregate = member.membership_aggregate(values, near, hard, membership, coefficient)
                return module.lin_l(aggregate) + module.lin_r(values)
        else:
            first, second = self.layer_graphs(x, n_train)

            def apply(module, values, graph):
                return self.weighted_layer(module, values, graph[0], graph[1])
        features = ((self.filtered(x) - self.input_mean) / self.input_scale).float()
        hidden = self.backbone.dropout(torch.relu(apply(self.backbone.first, features, first)))
        raw = apply(self.backbone.second, hidden, second)
        self.geometry_feature = raw
        embedding = torch.relu(raw)
        return self.backbone.head(embedding).squeeze(-1), embedding, *first[:3]

    def training_loss(self, x, target, edges):
        prediction = self(x, len(target), edges)[0]
        mse = F.mse_loss(prediction, target)
        envelope, homogeneous, contrast, centers = spherical_terms(
            self.geometry_feature, self.label_bins, self.label_values, self.sphere_points)
        geometry = .01 * (envelope + homogeneous) if self.variant != 'CE' else envelope * 0
        warmup = not self.geometry_history
        loss = mse if warmup else mse + contrast + geometry
        self.geometry_history.append({'step': len(self.geometry_history)+1, 'mse': float(mse.detach()),
            'contrast': float(contrast.detach()), 'envelope': float(envelope.detach()),
            'homogeneous': float(homogeneous.detach()),
            'negative_preReLU_fraction': float((self.geometry_feature.detach() < 0).float().mean()),
            'centroid_norm_error': float((centers.detach().norm(dim=1)-1).abs().max()),
            'first_update_MSE_only': warmup})
        return loss

    def make_optimizer(self, config):
        return Optimizer(self, config)


def construct(arm, config, device):
    r.fix_seed(config['seed'], 1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE / config['region'] / f"fold-{config['fold']:02d}" / 'data/development.npz') as data:
            config = {**config, 'relation_n_train': int(data['n_train'])}
    return SphericalSAGE(arm, config).to(device)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    feature = torch.tensor([[1., .2], [.8, .3], [.1, 1.], [.2, .8]], dtype=torch.float64, requires_grad=True)
    bins = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0., 2.], dtype=torch.float64)
    sphere = torch.eye(2, dtype=torch.float64)
    terms = spherical_terms(feature, bins, labels, sphere)
    scaled = spherical_terms(feature * torch.tensor([1., 2., 3., 4.])[:, None], bins, labels, sphere)
    assert all(torch.allclose(a, b, atol=1e-12, rtol=0) for a, b in zip(terms, scaled))
    literal_envelope = 1 - torch.stack([max((c*p).sum() for c in terms[-1]) for p in sphere]).mean()
    literal_homogeneous = (terms[-1][1]-terms[-1][0]).square().sum() / 2
    assert torch.allclose(terms[0], literal_envelope, atol=1e-12, rtol=0)
    assert torch.allclose(terms[1], literal_homogeneous, atol=1e-12, rtol=0)
    assert torch.isfinite(torch.autograd.grad(sum(terms[:3]), feature)[0]).all()
    cases = []
    for region in r.REGIONS:
        _, cfg, data, pre = r.load_fold(region, 1)
        n = int(data['n_train'])
        x, te, ee, y = r.tensors(data, pre, 'cuda')
        for arm in NEW:
            model = construct(arm, cfg, 'cuda').eval()
            reference = member.construct('GL' if arm == 'GL' else 'GE', cfg, 'cuda').eval()
            assert all(torch.equal(value, reference.backbone.state_dict()[key]) for key, value in model.backbone.state_dict().items())
            model.fit_reference(x[:n], te); model.fit_relations(x[:n], y, te)
            reference.fit_reference(x[:n], te); reference.fit_relations(x[:n], y, te)
            with torch.no_grad():
                assert torch.equal(model(x, n, ee)[0], reference(x, n, ee)[0])
            before = r.cpu_state(model)
            optimizer = model.make_optimizer(cfg)
            for _ in range(3):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(x[:n], y, te)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert model.layer_log_metric.grad.norm() > 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); optimizer.step()
            assert not torch.equal(before['backbone.head.weight'], model.backbone.head.weight.cpu())
            model.eval()
            with torch.no_grad():
                full = model(x, n, ee)[0]
                alone = model(x[:n], n, te)[0]
                single = model(torch.cat([x[:n], x[n:n+1]]), n, ee)[0]
                assert torch.allclose(full[:n], alone, atol=3e-6, rtol=3e-6)
                assert torch.allclose(full[n], single[n], atol=3e-6, rtol=3e-6)
            cases.append({'region': region, 'arm': arm, 'fresh_paired_backbone': True,
                          'initial_prediction_exact': True, 'query_independence': True,
                          'three_updates': True, 'last_losses': model.geometry_history[-1]})
    r.write_json(output / 'preflight.json', {'cases': cases, 'algebra_and_normalization': True,
                 'source_sha256': r.digest(Path(__file__)), 'sphere_seed_offset': 87000})
    print('PREFLIGHT', len(cases), 'cases', flush=True)


def directory(output, region, fold, arm):
    root, name = CACHED[arm] if arm in CACHED else (output, arm)
    return root / region / f'fold-{fold:02d}' / name


def analyze(output):
    methods = (*CACHED, *NEW)
    points, folds = [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold)
            n, y = int(data['n_train']), data['y']
            q80 = float(np.quantile(y[:n], .8))
            predictions = {}
            for arm in methods:
                path = directory(output, region, fold, arm)
                saved = json.loads((path / 'metrics.json').read_text())
                with np.load(path / 'responses.npz', allow_pickle=False) as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    assert (z['neighbors'] < n).all()
                    pred = z['prediction']; predictions[arm] = pred.copy()
                mt, mv = r.metrics(y[:n], pred[:n]), r.metrics(y[n:], pred[n:])
                assert abs(mv['rmse'] - saved['validation']['rmse']) < 1e-7
                assert abs(mt['rmse'] - saved['train']['rmse']) < 1e-7
                folds.append({'region': region, 'fold': fold, 'arm': arm, 'rmse': mv['rmse'],
                    'mae': mv['mae'], 'r2': mv['r2'], 'train_rmse': mt['rmse'],
                    'prediction_sd_ratio': float(pred[n:].std() / y[n:].std()),
                    'best_epoch': saved['best_epoch'], 'epochs_run': saved['epochs_run'],
                    'elapsed_seconds': saved['elapsed_seconds']})
            for i in range(n, len(y)):
                point = {'sample_id': str(data['sample_ids'][i]), 'region': region, 'fold': fold,
                    'observed_agb': float(y[i]), 'high_agb': int(y[i] >= q80),
                    'missing_features': int(data['missing_mask'][i].sum()),
                    'max_abs_training_z': float(np.abs(data['x'][i]).max())}
                for arm, pred in predictions.items():
                    point[arm+'_prediction'] = float(pred[i])
                    point[arm+'_squared_error'] = float((pred[i] - y[i]) ** 2)
                points.append(point)
    assert len(points) == len({v['sample_id'] for v in points}) == 1421
    assert len(folds) == 120
    lookup = {(v['region'], v['fold'], v['arm']): v for v in folds}
    comparisons, provinces = {}, {}
    for arm in NEW:
        comparisons[arm] = {other: float(np.mean([
            100*(1-lookup[(reg, f, arm)]['rmse']/lookup[(reg, f, other)]['rmse'])
            for reg in r.REGIONS for f in range(1, 6)])) for other in methods if other != arm}
        provinces[arm] = {reg: float(np.mean([
            100*(1-lookup[(reg, f, arm)]['rmse']/lookup[(reg, f, 'G0')]['rmse'])
            for f in range(1, 6)])) for reg in r.REGIONS}
    summary = {'comparisons': comparisons, 'province_gains': provinces,
               'new_fits': 45, 'cached_fits': 75, 'method_folds': 120, 'IDs': 1421}
    r.write_csv(output / 'all_1421_ids.csv', points)
    r.write_csv(output / 'all_fold_metrics.csv', folds)
    r.write_json(output / 'summary.json', summary)
    print('SUMMARY', json.dumps(summary), flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir / 'preflight.json').read_text())
    assert check['source_sha256'] == r.digest(Path(__file__)) and len(check['cases']) == 9
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(r.HERE / 'TASKS' / (TASK + '.md'), output / 'task_frozen.md')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    sources = [Path(__file__), r.HERE/'relation_graphsage.py', r.HERE/'relation_membership.py',
               r.HERE/'relation_coupled.py', r.HERE/'relation_layer.py', r.HERE/'relation_local_egop.py',
               r.HERE/'relation_agop.py', r.HERE/'relation_soft_topk.py', r.HERE/'relation_unroll.py']
    for region in r.REGIONS:
        for fold in range(1, 6):
            sources.extend(r.SOURCE/region/f'fold-{fold:02d}'/'data'/name
                           for name in ('development.npz', 'preprocessor.npz'))
    hashes = {str(path): r.digest(path) for path in sources}
    r.write_json(output/'manifest.json', {'task_id': TASK, 'started_at': datetime.now().astimezone().isoformat(),
        'source_hashes': hashes, 'new_fits': 45, 'cached_fits': 75, 'seed_offsets': [0], 'final_holdout_accessed': False})
    started = time.perf_counter()
    for region in r.REGIONS:
        for fold in range(1, 6):
            for arm in NEW:
                r.fit(arm, region, fold, output, 'cuda', model_factory=construct, task_id=TASK, max_gnn_epochs=400)
    analyze(output)
    assert all(r.digest(Path(path)) == value for path, value in hashes.items())
    r.write_json(output/'finished.json', {'finished_at': datetime.now().astimezone().isoformat(),
        'wall_seconds': time.perf_counter()-started, 'source_hashes_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'run', 'analyze'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args()
    r.setup()
    if args.mode == 'preflight': preflight(args.output)
    elif args.mode == 'run':
        assert args.preflight_dir
        run(args.output, args.preflight_dir)
    else: analyze(args.output)
