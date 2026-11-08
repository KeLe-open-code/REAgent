"""Lagged covariance coordinates for one GraphSAGE and its positive relations."""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import relation_spherical_trace as sphere
import relation_numeric_message as archive

r, layer = sphere.r, sphere.layer
TASK = 'CARBON-RELATION-BACKBONE-CURVATURE-001'
NEW = ('JS', 'JA', 'FS', 'FA')
CACHED = {key: archive.CACHED[key] for key in ('G0', 'GR', 'SG')}
REFERENCE = archive.CACHED['SG']


def transport_variance(value, old_left, old_right, new_left, new_right):
    left = new_left.T@old_left
    right = old_right.T@new_right
    return left.square()@value@right.square()


class BasisAdam:
    """Adam moments with a lagged, optional two-sided orthogonal frame."""
    def __init__(self, parameters, config, adaptive):
        self.parameters = list(parameters)
        self.adaptive = adaptive
        self.lr, self.decay = config['learning_rate'], config['weight_decay']
        self.steps, self.states = 0, []
        for name, parameter in self.parameters:
            state = {'name': name, 'mean': torch.zeros_like(parameter, dtype=torch.float64),
                     'variance': torch.zeros_like(parameter, dtype=torch.float64),
                     'shadow_variance': torch.zeros_like(parameter, dtype=torch.float64),
                     'max_direction_change': parameter.new_zeros((), dtype=torch.float64),
                     'max_transport_trace_error': parameter.new_zeros((), dtype=torch.float64), 'rotations': 0}
            if parameter.ndim == 2:
                rows, cols = parameter.shape
                state.update(left=parameter.new_zeros((rows, rows), dtype=torch.float64),
                             right=parameter.new_zeros((cols, cols), dtype=torch.float64),
                             ql=torch.eye(rows, device=parameter.device, dtype=torch.float64),
                             qr=torch.eye(cols, device=parameter.device, dtype=torch.float64))
            else:
                assert parameter.ndim == 1
            self.states.append(state)

    def zero_grad(self, set_to_none=True):
        for _, parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    @staticmethod
    def basis(covariance, previous, initial):
        jitter = 1e-12*(covariance.trace()/len(covariance)).clamp_min(1e-30)
        regularized = covariance+jitter*torch.eye(len(covariance), device=covariance.device, dtype=covariance.dtype)
        if initial:
            return torch.linalg.eigh(regularized)[1].flip(1)
        order = torch.argsort(torch.diag(previous.T@regularized@previous), descending=True, stable=True)
        q, triangular = torch.linalg.qr(regularized@previous[:, order])
        signs = torch.where(triangular.diag() < 0, -1., 1.)
        return q*signs[None, :]

    @torch.no_grad()
    def step(self):
        self.steps += 1
        correction1, correction2 = 1-.9**self.steps, 1-.999**self.steps
        for (_, parameter), state in zip(self.parameters, self.states):
            assert parameter.grad is not None
            gradient = parameter.grad.double()+self.decay*parameter.double()
            state['mean'].mul_(.9).add_(gradient, alpha=.1)
            matrix = parameter.ndim == 2
            projected = state['ql'].T@gradient@state['qr'] if matrix else gradient
            state['variance'].mul_(.999).add_(projected.square(), alpha=.001)
            state['shadow_variance'].mul_(.999).add_(gradient.square(), alpha=.001)
            first = state['ql'].T@state['mean']@state['qr'] if matrix else state['mean']
            update = (first/correction1)/((state['variance']/correction2).sqrt()+1e-8)
            if matrix:
                update = state['ql']@update@state['qr'].T
            shadow = (state['mean']/correction1)/((state['shadow_variance']/correction2).sqrt()+1e-8)
            difference = (update-shadow).norm()/shadow.norm().clamp_min(1e-30)
            state['max_direction_change'] = torch.maximum(state['max_direction_change'], difference)
            parameter.add_(update.to(parameter.dtype), alpha=-self.lr)
            if not matrix:
                continue
            state['left'].mul_(.95).add_(gradient@gradient.T, alpha=.05)
            state['right'].mul_(.95).add_(gradient.T@gradient, alpha=.05)
            if self.adaptive and (self.steps == 1 or self.steps % 10 == 0):
                left = self.basis(state['left'], state['ql'], self.steps == 1)
                right = self.basis(state['right'], state['qr'], self.steps == 1)
                variance = transport_variance(state['variance'], state['ql'], state['qr'], left, right)
                trace_error = (variance.sum()-state['variance'].sum()).abs()/state['variance'].sum().clamp_min(1e-30)
                state['max_transport_trace_error'] = torch.maximum(state['max_transport_trace_error'], trace_error)
                state['variance'], state['ql'], state['qr'] = variance, left, right
                state['rotations'] += 1

    def report(self):
        rows = []
        for (_, parameter), state in zip(self.parameters, self.states):
            item = {'name': state['name'], 'shape': list(parameter.shape), 'rotations': state['rotations'],
                    'maximum_relative_direction_change_from_identity': float(state['max_direction_change']),
                    'maximum_variance_trace_transport_error': float(state['max_transport_trace_error']),
                    'variance_nonnegative': bool((state['variance'] >= 0).all()),
                    'moment_state_scope': 'end_of_training_not_best_checkpoint'}
            if parameter.ndim == 2:
                errors, off = [], []
                for key, basis in (('left', 'ql'), ('right', 'qr')):
                    covariance, q = state[key], state[basis]
                    errors.append(float((q.T@q-torch.eye(len(q), device=q.device, dtype=q.dtype)).abs().max()))
                    off.append(float((covariance-torch.diag(covariance.diag())).norm()/covariance.norm().clamp_min(1e-30)))
                item.update(basis_orthogonality_error=max(errors), covariance_offdiagonal_fractions=off,
                            basis_distance_from_identity=[float((state[key]-torch.eye(len(state[key]), device=parameter.device)).norm()) for key in ('ql', 'qr')])
            rows.append(item)
        return rows


class JointOptimizer:
    def __init__(self, model, config):
        self.backbone = BasisAdam(model.backbone.named_parameters(), config, model.variant.endswith('S'))
        self.geometry = layer.RDoG(model.layer_log_metric) if model.variant.startswith('J') else None
        ids = [id(p) for _, p in self.backbone.parameters]+([] if self.geometry is None else [id(self.geometry.parameter)])
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
        model.executed_optimizer = self

    def zero_grad(self, set_to_none=True):
        self.backbone.zero_grad(set_to_none=set_to_none)
        if self.geometry is not None:
            self.geometry.parameter.grad = None

    def step(self):
        self.backbone.step()
        if self.geometry is not None:
            self.geometry.step()


class CurvatureSAGE(sphere.SphericalSAGE):
    def __init__(self, variant, config):
        super().__init__('GL', config)
        self.variant, self.config = variant, config
        self.layer_log_metric.requires_grad_(variant.startswith('J'))
        self.reference = REFERENCE[0]/config['region']/f"fold-{config['fold']:02d}"/REFERENCE[1]/'best.pt'

    @torch.no_grad()
    def fit_reference(self, x, edges):
        saved = torch.load(self.reference, map_location=x.device, weights_only=False)['state_dict']
        fresh = r.cpu_state(self)
        for name, value in self.named_buffers():
            value.copy_(saved[name])
        for name, parameter in self.named_parameters():
            assert torch.equal(parameter.cpu(), fresh[name])
        assert edges.max() < len(x) and torch.equal(self.coordinates(x), self.field_support)
        assert torch.count_nonzero(self.layer_log_metric) == 0 and bool(self.field_ready)
        self._relation_cache.clear()

    def fit_relations(self, x, target, edges):
        assert len(x) == len(target) == len(self.field_support) and edges.max() < len(x)
        assert torch.equal(self.label_bins, torch.bucketize(target.contiguous(), self.label_edges))
        assert torch.allclose(self.label_values, torch.stack([target[self.label_bins == k].double().mean() for k in range(10)]))
        return [{'new_kernel_fits': 0, 'reused_training_fitted_kernels': 26, 'original_SG_temperature_preserved': True}]

    def make_optimizer(self, config):
        return JointOptimizer(self, config)

    def load_state_dict(self, *args, **kwargs):
        self._relation_cache.clear()
        return super().load_state_dict(*args, **kwargs)


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as z:
            config['relation_n_train'] = int(z['n_train'])
    r.fix_seed(config['seed'], 1)
    return CurvatureSAGE(arm, config).to(device)


def math_checks():
    generator = torch.Generator().manual_seed(10000)
    p = torch.nn.Parameter(torch.randn(3, 5, dtype=torch.float64, generator=generator))
    q = torch.nn.Parameter(p.detach().clone())
    config = {'learning_rate': .001, 'weight_decay': 1e-4}
    own, native = BasisAdam([('weight', p)], config, False), torch.optim.Adam([q], lr=.001, weight_decay=1e-4)
    error = 0.
    for _ in range(12):
        p.grad = torch.randn(p.shape, generator=generator, dtype=p.dtype); q.grad = p.grad.clone()
        own.step(); native.step(); error = max(error, float((p-q).abs().max().detach()))
    assert error < 1e-12
    a = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))[0]
    b = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64))[0]
    variance = torch.rand(3, 5, generator=generator, dtype=torch.float64)
    identity3, identity5 = torch.eye(3, dtype=torch.float64), torch.eye(5, dtype=torch.float64)
    got = transport_variance(variance, identity3, identity5, a, b)
    mapping = torch.stack([(a.T@unit.reshape(3, 5)@b).flatten() for unit in torch.eye(15, dtype=torch.float64)], 1)
    expected = ((mapping*variance.flatten()[None])@mapping.T).diag().reshape(3, 5)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0) and (got >= 0).all()
    assert abs(float(got.sum()-variance.sum())) < 1e-12
    zero = torch.nn.Parameter(torch.zeros(3, 5, dtype=torch.float64))
    opt = BasisAdam([('zero', zero)], {'learning_rate': .001, 'weight_decay': 0}, True)
    for _ in range(12):
        zero.grad = torch.zeros_like(zero); opt.step()
    assert torch.count_nonzero(zero) == 0 and torch.isfinite(opt.states[0]['variance']).all()
    return {'identity_Adam_max_error': error, 'covariance_transport_literal': True, 'zero_gradient_stable': True}


@torch.no_grad()
def export(model, data, pre, output):
    model.eval(); n = int(data['n_train']); x, _, ee, _ = r.tensors(data, pre, 'cuda')
    prediction = model(x, n, ee)[0].cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    if (output/'responses.npz').exists():
        with np.load(output/'responses.npz') as z:
            assert np.array_equal(z['prediction'], prediction)
    graphs = model.layer_graphs(x, n)
    np.savez_compressed(output/'curvature_graphs.npz', sample_ids=data['sample_ids'],
        neighbors=np.stack([g[0].cpu().numpy() for g in graphs]), weights=np.stack([g[1].cpu().numpy() for g in graphs]))
    opt = model.executed_optimizer
    report = {'variant': model.variant, 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'geometry_log_norms': model.layer_log_metric.norm(dim=1).tolist(), 'backbone_steps': opt.backbone.steps,
        'backbone_moment_state': opt.backbone.report(), 'geometry_steps': opt.geometry.steps if opt.geometry is not None else 0,
        'scope': 'optimizer diagnostics are end_of_training,geometry norms selected_best_checkpoint'}
    r.write_json(output/'curvature_diagnostics.json', report)
    r.write_json(output/'spherical_loss_history.json', model.geometry_history)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    math, cases = math_checks(), []
    for region in r.REGIONS:
        _, cfg, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, target = r.tensors(data, pre, 'cuda')
        common, first_states = None, {}
        for arm in NEW:
            model = construct(arm, cfg, 'cuda').eval(); model.fit_reference(x[:n], te); model.fit_relations(x[:n], target, te)
            initial = r.cpu_state(model)
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            with torch.no_grad():
                pred = model(x, n, ee)[0]
            if common is None:
                common = (initial, rng, pred.clone())
            else:
                assert all(torch.equal(v, common[0][k]) for k, v in initial.items())
                assert all(torch.equal(a, b) for a, b in zip(rng, common[1])) and torch.equal(pred, common[2])
            buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
            opt = model.make_optimizer(cfg)
            for step in range(12):
                model.train(); opt.zero_grad()
                loss = model.training_loss(x[:n], target, te); loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
                if arm.startswith('J'):
                    assert model.layer_log_metric.grad.norm(dim=1).min() > 0
                else:
                    assert model.layer_log_metric.grad is None
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); opt.step()
                if step == 0:
                    state = r.cpu_state(model.backbone)
                    if arm[0] in first_states:
                        assert all(torch.equal(v, first_states[arm[0]][k]) for k, v in state.items())
                    else:
                        first_states[arm[0]] = state
            assert all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in buffers.items())
            assert bool(model.layer_log_metric.norm() > 0) == arm.startswith('J')
            report = opt.backbone.report(); matrices = [row for row in report if len(row['shape']) == 2]
            assert len(matrices) == 5
            assert all(row['rotations'] == (2 if arm.endswith('S') else 0) for row in matrices)
            assert all(row['basis_orthogonality_error'] < 1e-10 and row['variance_nonnegative'] for row in matrices)
            if arm.endswith('S'):
                assert min(row['maximum_relative_direction_change_from_identity'] for row in matrices) > 1e-8
            assert max(row['maximum_variance_trace_transport_error'] for row in matrices) < 1e-10
            assert all(not torch.equal(initial['backbone.'+k], v.cpu()) for k, v in model.backbone.state_dict().items())
            model.eval()
            with torch.no_grad():
                full = model(x, n, ee)[0]
                assert torch.allclose(full[:n], model(x[:n], n, te)[0], atol=3e-6, rtol=3e-6)
                assert torch.allclose(full[n], model(torch.cat([x[:n], x[n:n+1]]), n, ee)[0][-1], atol=3e-6, rtol=3e-6)
            path = output/f'{region}_{arm}'; path.mkdir()
            torch.save(r.cpu_state(model), path/'state.pt')
            restored = construct(arm, cfg, 'cuda').eval(); restored.load_state_dict(torch.load(path/'state.pt', map_location='cuda', weights_only=True))
            with torch.no_grad():
                assert torch.equal(full, restored(x, n, ee)[0])
            export(model, data, pre, path)
            cases.append({'region': region, 'arm': arm, 'initial_state_RNG_prediction_paired': True,
                'first_backbone_updates_identity_paired': True, 'all_five_matrices_executed': True,
                'fixed_or_learned_geometry_correct': True, 'query_independent': True, 'exact_restore': True})
    r.write_json(output/'preflight.json', {'source_sha256': r.digest(Path(__file__)), 'math': math, 'cases': cases})
    print('PREFLIGHT', len(cases), 'cases', flush=True)


def directory(output, region, fold, arm):
    root, name = CACHED.get(arm, (output, arm))
    return root/region/f'fold-{fold:02d}'/name


def analyze(output):
    methods = (*CACHED, *NEW)
    folds, points, diagnostics, histories, restores = [], [], [], [], []
    started = time.perf_counter()
    for region in r.REGIONS:
        for fold in range(1, 6):
            source, cfg, data, pre = r.load_fold(region, fold)
            n, predictions = int(data['n_train']), {}
            for arm in methods:
                path = directory(output, region, fold, arm)
                saved = json.loads((path/'metrics.json').read_text())
                assert saved['seed'] == cfg['seed'] and saved['epochs_run'] <= 400
                with np.load(path/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    pred = z['prediction'].copy()
                mt, mv = r.metrics(data['y'][:n], pred[:n]), r.metrics(data['y'][n:], pred[n:])
                for key in mt:
                    assert abs(mt[key]-saved['train'][key]) < 1e-7 and abs(mv[key]-saved['validation'][key]) < 1e-7
                predictions[arm] = pred
                folds.append({'region': region, 'fold': fold, 'arm': arm, **mv,
                    'train_rmse': mt['rmse'], 'train_r2': mt['r2'], 'parameters': saved['parameters'],
                    'epochs_run': saved['epochs_run'], 'best_epoch': saved['best_epoch'],
                    'elapsed_seconds': saved['elapsed_seconds'], 'peak_cuda_allocated_bytes': saved['peak_cuda_allocated_bytes'],
                    'prediction_sd_ratio': float(pred[n:].std()/data['y'][n:].std())})
                if arm not in NEW:
                    continue
                info = json.loads((path/'curvature_diagnostics.json').read_text())
                assert info['backbone_steps'] == saved['epochs_run'] and info['parameters'] == 28275
                assert info['trainable_parameters'] == (28275 if arm.startswith('J') else 27905)
                assert info['geometry_steps'] == (saved['epochs_run'] if arm.startswith('J') else 0)
                assert (min(info['geometry_log_norms']) > 0) if arm.startswith('J') else max(info['geometry_log_norms']) == 0
                matrices = [row for row in info['backbone_moment_state'] if len(row['shape']) == 2]
                assert len(matrices) == 5
                for row in matrices:
                    assert row['rotations'] == (1+saved['epochs_run']//10 if arm.endswith('S') else 0)
                    assert row['basis_orthogonality_error'] < 1e-10 and row['variance_nonnegative']
                    assert row['maximum_variance_trace_transport_error'] < 1e-10
                    if arm.endswith('S'):
                        assert row['maximum_relative_direction_change_from_identity'] > 0
                diagnostics.append({'region': region, 'fold': fold, 'arm': arm, **info})
                with np.load(path/'curvature_graphs.npz') as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    near, weight = z['neighbors'], z['weights']
                    assert near.shape == weight.shape == (2, len(pred), 8)
                    assert (near >= 0).all() and (near < n).all() and not (near[:, :n] == np.arange(n)[None, :, None]).any()
                    assert (np.diff(np.sort(near, axis=-1), axis=-1) > 0).all()
                    assert np.isfinite(weight).all() and (weight >= 0).all() and np.allclose(weight.sum(-1), 1, atol=1e-6, rtol=0)
                history = json.loads((path/'spherical_loss_history.json').read_text())
                parent_history = pd.read_csv(path/'history.csv')
                assert len(history) == len(parent_history) == saved['epochs_run'] and history[0]['first_update_MSE_only']
                expected = np.array([h['mse']+(0 if h['first_update_MSE_only'] else h['contrast']+.01*(h['envelope']+h['homogeneous'])) for h in history])
                actual = parent_history.train_loss_standardized.to_numpy()
                assert np.isfinite(expected).all() and np.allclose(expected, actual, atol=2e-6, rtol=1e-6)
                histories.append({'region': region, 'fold': fold, 'arm': arm, 'epochs': len(history),
                    'maximum_loss_identity_error': float(np.abs(expected-actual).max())})
                if fold == 1:
                    checkpoint = torch.load(path/'best.pt', map_location='cuda', weights_only=False)
                    assert checkpoint['max_gnn_epochs'] == 400 and checkpoint['split_sha256'] == r.digest(source/'data/split.csv')
                    model = construct(arm, checkpoint['config'], 'cuda').eval(); model.load_state_dict(checkpoint['state_dict'])
                    x, _, ee, _ = r.tensors(data, pre, 'cuda')
                    with torch.no_grad():
                        restored = model(x, n, ee)[0].cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
                    assert np.array_equal(pred, restored); restores.append({'region': region, 'arm': arm, 'exact': True})
            for i in range(n, len(data['y'])):
                row = {'region': region, 'fold': fold, 'sample_id': str(data['sample_ids'][i]),
                    'observed_agb': float(data['y'][i]), 'high_agb': bool(data['y'][i] >= np.quantile(data['y'][:n], .8))}
                for arm, pred in predictions.items():
                    row[arm+'_prediction'] = float(pred[i]); row[arm+'_squared_error'] = float((pred[i]-data['y'][i])**2)
                points.append(row)
    ff, pp = pd.DataFrame(folds), pd.DataFrame(points)
    assert len(ff) == 105 and len(pp) == pp.sample_id.nunique() == 1421 and len(histories) == 60 and len(restores) == 12
    pain = pd.read_csv(r.HERE/'runs/CARBON-GAIN10-PAIN-AUDIT-001/attempt-02/all_1421_ids.csv',
        usecols=['sample_id', 'region', 'fold', 'missing_features', 'max_abs_training_standardized_feature'], dtype={'sample_id': str})
    pp = pp.merge(pain, on=['sample_id', 'region', 'fold'], how='outer', validate='one_to_one', indicator=True)
    assert len(pp) == 1421 and pp['_merge'].eq('both').all(); pp.drop(columns='_merge', inplace=True)
    wide = ff.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    comparisons = {arm: {other: float((100*(1-wide[arm]/wide[other])).mean()) for other in methods if other != arm} for arm in NEW}
    province = ff.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'train_rmse', 'prediction_sd_ratio']].mean()
    noninferior = all(province.loc[(reg, 'JS'), 'rmse'] <= province.loc[(reg, 'G0'), 'rmse'] and
                     province.loc[(reg, 'JS'), 'r2'] >= province.loc[(reg, 'G0'), 'r2'] for reg in r.REGIONS)
    cap = all(province.loc[(reg, 'JS'), 'rmse'] <= 1.01*province.loc[(reg, 'G0'), 'rmse'] and
              province.loc[(reg, 'JS'), 'r2'] >= province.loc[(reg, 'G0'), 'r2']-.01 for reg in r.REGIONS)
    g = comparisons['JS']
    full = g['G0'] >= 10 and g['JA'] >= .5 and g['FS'] >= .5 and all(g[k] > 0 for k in ('FA', 'GR', 'SG')) and noninferior
    pareto = g['G0'] >= 3 and all(g[k] > 0 for k in ('JA', 'FA', 'FS', 'GR', 'SG')) and cap
    strata, examples = [], []
    for region, group in pp.groupby('region'):
        for name, flag in {'all': group.index == group.index, 'high': group.high_agb, 'ordinary': ~group.high_agb,
                           'missing': group.missing_features.gt(0), 'complete': group.missing_features.eq(0),
                           'input_outlier': group.max_abs_training_standardized_feature.gt(5)}.items():
            sub = group[flag]
            if len(sub):
                for arm in NEW:
                    for other in methods:
                        if arm != other:
                            strata.append({'region': region, 'stratum': name, 'arm': arm, 'control': other, 'count': len(sub),
                                'SSE_candidate_minus_control': float((sub[arm+'_squared_error']-sub[other+'_squared_error']).sum()),
                                'bias': float((sub[arm+'_prediction']-sub.observed_agb).mean())})
        selected = group.assign(JS_minus_SG_SSE=group.JS_squared_error-group.SG_squared_error)
        for category, sub in [('harmed', selected.nlargest(5, 'JS_minus_SG_SSE')), ('helped', selected.nsmallest(5, 'JS_minus_SG_SSE'))]:
            examples.extend(sub.assign(example_type=category).to_dict('records'))
    ff.to_csv(output/'all_fold_metrics.csv', index=False); pp.to_csv(output/'all_1421_failure_assessments.csv', index=False)
    province.reset_index().to_csv(output/'province_metrics.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_subgroups.csv', index=False); pd.DataFrame(examples).to_csv(output/'failure_examples.csv', index=False)
    r.write_json(output/'curvature_diagnostics.json', diagnostics); r.write_json(output/'loss_history_checks.json', histories)
    r.write_json(output/'restore_checks.json', restores)
    summary = {'task_id': TASK, 'comparisons': comparisons, 'FULL': bool(full), 'PARETO': bool(pareto),
        'integration_fidelity_valid': True, 'G0_province_noninferior': bool(noninferior),
        'new_fits': 60, 'cached_fits': 45, 'IDs': 1421, 'final_holdout_access': False, 'goal_achieved': False,
        'factorial_interaction_RMSE_diagnostic': float(((wide['JS']-wide['JA'])-(wide['FS']-wide['FA'])).mean()),
        'analysis_seconds': time.perf_counter()-started}
    r.write_json(output/'summary.json', summary); print('SUMMARY', json.dumps(summary), flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text())
    assert check['source_sha256'] == r.digest(Path(__file__)) and len(check['cases']) == 12
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md'); shutil.copy2(Path(__file__), output/Path(__file__).name)
    files = [Path(__file__), r.BASE/'baseline.py', output/'task_frozen.md', preflight_dir/'preflight.json',
             r.HERE/'runs/CARBON-GAIN10-PAIN-AUDIT-001/attempt-02/all_1421_ids.csv']
    files.extend(Path(mod.__file__) for mod in tuple(sys.modules.values()) if getattr(mod, '__file__', None)
        and Path(mod.__file__).is_file() and Path(mod.__file__).suffix == '.py' and Path(mod.__file__).parent == r.HERE)
    for region in r.REGIONS:
        for fold in range(1, 6):
            source, _, _, _ = r.load_fold(region, fold)
            files.extend([source/'config.json', *[source/'data'/name for name in ('development.npz', 'preprocessor.npz', 'split.csv')]])
            files.append(REFERENCE[0]/region/f'fold-{fold:02d}'/REFERENCE[1]/'best.pt')
            for root, name in CACHED.values():
                files.extend(root/region/f'fold-{fold:02d}'/name/f for f in ('metrics.json', 'responses.npz'))
    hashes = {str(path.resolve()): r.digest(path) for path in files}
    started_at, started = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', {'task_id': TASK, 'started_at': started_at, 'source_input_hashes': hashes,
        'new_fits': 60, 'cached_fits': 45, 'final_holdout_access': False, 'device': torch.cuda.get_device_name(),
        'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory})
    export_seconds = 0.
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                t = time.perf_counter(); export(holder[0], data, pre, directory(output, region, fold, arm))
                export_seconds += time.perf_counter()-t
    fit_finished_at, fit_seconds = datetime.now().astimezone().isoformat(), time.perf_counter()-started
    analyze(output)
    assert all(r.digest(Path(path)) == expected for path, expected in hashes.items())
    r.write_json(output/'finished.json', {'started_at': started_at, 'fitting_finished_at': fit_finished_at,
        'fitting_and_export_seconds': fit_seconds, 'graph_export_seconds': export_seconds,
        'finished_at': datetime.now().astimezone().isoformat(), 'wall_seconds': time.perf_counter()-started,
        'hashes_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output, args.preflight_dir)
    else:
        analyze(args.output)
