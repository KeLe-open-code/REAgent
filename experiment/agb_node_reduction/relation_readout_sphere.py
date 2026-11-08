"""Existing SG auxiliaries measured in a detached, readout-sensitive SPD geometry."""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import relation_angle_compensation as base

r, previous, sphere, layer = base.r, base.previous, base.sphere, base.layer
context, CACHED = base.context, base.CACHED
TASK, NEW = 'CARBON-RELATION-READOUT-SPHERE-001', ('FM', 'DM', 'PM')
directory = base.directory


@torch.no_grad()
def feature_metric(raw, weight, arm, permutation):
    d = raw.shape[1]; eye = torch.eye(d, device=raw.device, dtype=torch.float64)
    jacobian = (raw.detach() > 0)*weight.detach().double().reshape(1, -1)
    covariance = jacobian.T@jacobian/len(raw); trace = covariance.trace()
    if arm == 'N' or trace == 0:
        metric, root = eye, eye
    else:
        if arm == 'DM': covariance = covariance.diag().diag()
        elif arm == 'PM': covariance = covariance[permutation][:, permutation]
        metric = .5*(eye+d*covariance/trace)
        values, vectors = torch.linalg.eigh(metric)
        root = (vectors*values.sqrt()[None])@vectors.T
    values = torch.linalg.eigvalsh(metric)
    return metric, root, {'fisher_trace': float(trace), 'metric_trace': float(metric.trace()),
        'metric_min_eigenvalue': float(values.min()), 'metric_max_eigenvalue': float(values.max()),
        'metric_identity_norm': float((metric-eye).norm()),
        'metric_offdiagonal_norm': float((metric-metric.diag().diag()).norm()),
        'root_square_max_error': float((root@root-metric).abs().max())}


class ReadoutSAGE(base.AngleSAGE):
    training_objective = 'native_SG_loss_in_detached_readout_sensitive_signed_feature_metric;native_prediction_and_optimizer'

    def fit_relations(self, x, target, edges):
        history = previous.CurvatureSAGE.fit_relations(self, x, target, edges)
        generator = torch.Generator(device='cpu').manual_seed(self.config['seed']+130000)
        self.metric_permutation = torch.randperm(32, generator=generator).to(x.device)
        return history

    def training_loss(self, x, target, edges):
        if self.angle_arm == 'N': return sphere.SphericalSAGE.training_loss(self, x, target, edges)
        prediction = self(x, len(target), edges)[0]; self.last_training_prediction = prediction
        mse = F.mse_loss(prediction, target); raw = self.geometry_feature
        metric, root, details = feature_metric(raw, self.backbone.head.weight, self.angle_arm, self.metric_permutation)
        transformed = (raw.double()@root).to(raw.dtype)
        env, homo, contrast, centers = sphere.spherical_terms(transformed, self.label_bins, self.label_values, self.sphere_points)
        native_env, native_homo, native_contrast, _ = sphere.spherical_terms(raw, self.label_bins, self.label_values, self.sphere_points)
        auxiliary = contrast+.01*(env+homo)
        native_auxiliary = native_contrast+.01*(native_env+native_homo)
        warmup = not self.geometry_history; loss = mse if warmup else mse+contrast+.01*(env+homo)
        self.geometry_history.append({'step': len(self.geometry_history)+1, 'mse': float(mse.detach()),
            'contrast': float(contrast.detach()), 'native_prototype_contrast': float(native_contrast.detach()),
            'envelope': float(env.detach()), 'homogeneous': float(homo.detach()),
            'native_auxiliary_difference': float((auxiliary-native_auxiliary).detach()),
            'centroid_norm_error': float((centers.detach().norm(dim=1)-1).abs().max()),
            'first_update_MSE_only': warmup, 'metric_is_stop_gradient': not root.requires_grad, **details})
        return loss


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train'] = int(data['n_train'])
    r.fix_seed(config['seed'], 1)
    return ReadoutSAGE(arm, config).to(device)


def source_hashes():
    prior_path = r.HERE/'runs/CARBON-RELATION-ADJACENCY-FISHER-001/attempt-02/manifest.json'
    expected = json.loads(prior_path.read_text(encoding='utf-8'))['source_input_hashes']
    hashes = {**context.source_hashes(), str(Path(__file__).resolve()): r.digest(Path(__file__)),
        str(Path(base.__file__).resolve()): r.digest(Path(base.__file__)), str(prior_path.resolve()): r.digest(prior_path)}
    paths = [context.PAIN/'all_1421_ids.csv', context.PAIN/'feature_diagnostics.csv', context.NAMED, context.TAIL,
             r.HERE/'runs/CARBON-RELATION-POLAR-READOUT-AUDIT-001/attempt-01/summary.json']
    for region in r.REGIONS:
        for fold in range(1, 6):
            source = r.SOURCE/region/f'fold-{fold:02d}'
            paths.extend(source/name for name in ('config.json', 'data/development.npz', 'data/preprocessor.npz', 'data/split.csv'))
            for arm in CACHED: paths.extend(directory(None, region, fold, arm)/name for name in ('best.pt', 'metrics.json', 'responses.npz'))
    for path in paths:
        key = str(path.resolve())
        if path.name == 'summary.json' and path.parent.parent.name == 'CARBON-RELATION-POLAR-READOUT-AUDIT-001':
            value = r.digest(path)
        else:
            assert key in expected, key
            value = expected[key]
        assert r.digest(path) == value, path
        hashes[key] = value
    return hashes


def math_checks():
    generator = torch.Generator().manual_seed(130000)
    raw = torch.randn((8, 5), dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn((1, 5), dtype=torch.float64, generator=generator, requires_grad=True)
    permutation = torch.randperm(5, generator=generator)
    jacobian = torch.autograd.grad((torch.relu(raw)@weight.T).sum(), raw, retain_graph=True)[0]
    exact = (raw.detach() > 0)*weight.detach()
    assert torch.equal(jacobian, exact)
    covariance = exact.T@exact/len(raw)
    expected_score_outer = (exact.T@exact+(-exact).T@(-exact))/(2*len(raw))
    assert torch.equal(covariance, expected_score_outer)
    matrices, root_errors = {}, []
    for arm in NEW:
        metric, root, details = feature_metric(raw, weight, arm, permutation)
        assert not root.requires_grad and abs(float(metric.trace())-5) < 1e-12
        assert details['metric_min_eigenvalue'] >= .5-1e-12 and details['metric_max_eigenvalue'] <= 3+1e-12
        root_errors.append(details['root_square_max_error']); assert root_errors[-1] < 1e-12
        matrices[arm] = metric
    spectrum_error = float((torch.linalg.eigvalsh(matrices['FM'])-torch.linalg.eigvalsh(matrices['PM'])).abs().max())
    assert spectrum_error < 1e-12
    _, root, _ = feature_metric(raw, weight, 'FM', permutation)
    sphere_points = F.normalize(torch.randn((41, 5), dtype=torch.float64, generator=generator), dim=1)
    bins = torch.arange(4).repeat_interleave(2); labels = torch.arange(4, dtype=torch.float64)
    fixed_centers = sphere.spherical_terms(raw@root, bins, labels, sphere_points)[3].detach()
    def objective(value):
        transformed = value@root
        env, homo, _, _ = sphere.spherical_terms(transformed, bins, labels, sphere_points)
        contrast = F.cross_entropy(F.normalize(transformed, dim=1)@fixed_centers.T/.1, bins)
        return contrast+.01*(env+homo)
    loss = objective(raw); gradient = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    assert torch.autograd.grad(loss, weight, allow_unused=True)[0] is None
    plus, minus = raw.detach().clone(), raw.detach().clone(); plus[1, 2] += 1e-6; minus[1, 2] -= 1e-6
    fd_error = float(abs((objective(plus)-objective(minus))/2e-6-gradient[1, 2]))
    assert fd_error < 1e-6
    identity, identity_root, _ = feature_metric(raw, weight*0, 'FM', permutation)
    assert torch.equal(identity, torch.eye(5, dtype=torch.float64)) and torch.equal(identity, identity_root)
    return {'ReLU_head_Jacobian_exact': True, 'Gaussian_score_expectation_exact': True,
        'max_root_square_error': max(root_errors), 'same_state_permutation_spectrum_error': spectrum_error,
        'fixed_metric_FD_error': fd_error, 'auxiliary_head_gradient_absent': True,
        'native_detached_prototypes_also_fixed_in_FD': True,
        'SPD_trace_and_eigen_bounds': True, 'zero_Fisher_identity_exact': True}


@torch.no_grad()
def export(model, data, pre, output, restore):
    model.eval(); n = int(data['n_train']); x, _, ee, _ = r.tensors(data, pre, 'cuda')
    actual = model(x, n, ee)[0]; raw = model.geometry_feature[:n].clone(); graphs = model.layer_graphs(x, n)
    prediction = actual.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    with np.load(output/'responses.npz') as saved: assert np.array_equal(prediction, saved['prediction'])
    metric = json.loads((output/'metrics.json').read_text(encoding='utf-8')); best = int(metric['best_epoch'])
    theta = model.layer_log_metric.clone(); model.layer_log_metric.zero_(); model._relation_cache.clear()
    bypass = model(x, n, ee)[0]
    model.layer_log_metric.copy_(theta); model._relation_cache.clear()
    difference = float((actual-bypass).abs().max())
    with np.load(directory(output, model.config['region'], model.config['fold'], 'SG')/'responses.npz') as saved:
        sg_difference = float(abs(prediction-saved['prediction']).max())
    exposed = sum(gradient['geometry_gradient_norm'] > 1e-12 and row['metric_identity_norm'] > 1e-3
        and abs(row['native_auxiliary_difference']) > 1e-12
        for row, gradient in zip(model.geometry_history[1:best], model.angle_feedback[1:best]))
    matrix, root, details = feature_metric(raw, model.backbone.head.weight, model.angle_arm, model.metric_permutation)
    transformed = (raw.double()@root).to(raw.dtype)
    env, homo, contrast, _ = sphere.spherical_terms(transformed, model.label_bins, model.label_values, model.sphere_points)
    native_env, native_homo, native_contrast, _ = sphere.spherical_terms(raw, model.label_bins, model.label_values, model.sphere_points)
    report = {'selected_epoch': best, 'selected_objective_exposed_steps': exposed,
        'selected_deployment_active': bool(exposed > 0 and details['metric_identity_norm'] > 1e-3 and difference > 1e-6 and sg_difference > 1e-6 and theta.norm() > 1e-10),
        'zero_log_bypass_prediction_max_standardized_difference': difference,
        'SG_prediction_max_absolute_difference': sg_difference, 'selected_contrast': float(contrast),
        'selected_envelope': float(env), 'selected_homogeneous': float(homo),
        'selected_native_contrast': float(native_contrast), 'selected_native_envelope': float(native_env),
        'selected_native_homogeneous': float(native_homo), 'restore_checked': restore,
        'metric_and_head_surrogate_stop_gradient': True, **details}
    if restore:
        fresh = construct(model.angle_arm, model.config, 'cuda').eval(); fresh.load_state_dict(model.state_dict())
        assert torch.equal(fresh(x, n, ee)[0], actual); report['restore_exact'] = True
    arrays = {'sample_ids': data['sample_ids']}
    for i, graph in enumerate(graphs, 1):
        arrays[f'layer{i}_neighbors'], arrays[f'layer{i}_weights'] = graph[0].cpu().numpy(), graph[1].cpu().numpy()
    np.savez_compressed(output/'readout_graphs.npz', **arrays)
    np.savez_compressed(output/'readout_context.npz', sample_ids=data['sample_ids'][:n], raw=raw.cpu().numpy(),
        transformed=transformed.cpu().numpy(), matrix=matrix.cpu().numpy(), root=root.cpu().numpy(),
        permutation=model.metric_permutation.cpu().numpy())
    r.write_json(output/'readout_diagnostics.json', report)
    r.write_json(output/'readout_history.json', model.geometry_history)
    r.write_json(output/'readout_feedback.json', model.angle_feedback)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False); protected = source_hashes(); tick = time.perf_counter(); math_result = math_checks(); cases = []
    for region in r.REGIONS:
        _, config, data, pre = r.load_fold(region, 1); n = int(data['n_train'])
        x, te, ee, target = r.tensors(data, pre, 'cuda')
        native = previous.construct('JA', config, 'cuda')
        native.fit_reference(x[:n], te); native.fit_relations(x[:n], target, te)
        baseline_initial = r.cpu_state(native)
        native.eval()
        with torch.no_grad(): baseline_prediction = native(x, n, ee)[0].clone()
        control = construct('N', config, 'cuda'); control.fit_reference(x[:n], te); control.fit_relations(x[:n], target, te)
        opt_native, opt_control = layer.JointOptimizer(native, config), control.make_optimizer(config)
        for step in range(4):
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            control.train(); opt_control.zero_grad(); control.training_loss(x[:n], target, te).backward()
            torch.nn.utils.clip_grad_norm_(control.parameters(), config['gradient_clip']); opt_control.step()
            after = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
            native.train(); opt_native.zero_grad(); native.training_loss(x[:n], target, te).backward()
            torch.nn.utils.clip_grad_norm_(native.parameters(), config['gradient_clip']); opt_native.step()
            assert all(torch.equal(value.cpu(), native.state_dict()[key].cpu()) for key, value in control.state_dict().items())
            assert torch.equal(after[0], torch.get_rng_state()) and torch.equal(after[1], torch.cuda.get_rng_state())
            if step == 0: first_state = r.cpu_state(native)
        second_gradients = []
        for arm in NEW:
            model = construct(arm, config, 'cuda'); model.fit_reference(x[:n], te); model.fit_relations(x[:n], target, te)
            assert all(torch.equal(value.cpu(), baseline_initial[key]) for key, value in model.state_dict().items())
            assert sum(p.numel() for p in model.parameters()) == 28275 and len(list(model.named_buffers())) == 16
            model.eval()
            with torch.no_grad(): assert torch.equal(model(x, n, ee)[0], baseline_prediction)
            optimizer = model.make_optimizer(config); delta_norm = None
            for step in range(6):
                model.train(); optimizer.zero_grad(); loss = model.training_loss(x[:n], target, te)
                if step == 1:
                    env, homo, proto, _ = sphere.spherical_terms(model.geometry_feature, model.label_bins, model.label_values, model.sphere_points)
                    original = F.mse_loss(model.last_training_prediction, target)+proto+.01*(env+homo)
                    delta = torch.autograd.grad(loss-original, model.layer_log_metric, retain_graph=True)[0]
                    delta_norm = float(delta.norm()); assert torch.isfinite(delta).all() and delta_norm > 1e-10
                    head_delta = torch.autograd.grad(loss-original, tuple(model.backbone.head.parameters()), retain_graph=True)
                    assert all(torch.count_nonzero(value) == 0 for value in head_delta)
                    second_gradients.append(torch.autograd.grad(loss, model.layer_log_metric, retain_graph=True)[0].detach().clone())
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                assert torch.isfinite(loss) and torch.isfinite(norm); optimizer.step()
                if step == 0: assert all(torch.equal(value.cpu(), first_state[key]) for key, value in model.state_dict().items())
            model.eval()
            with torch.no_grad():
                full, alone = model(x, n, ee), model(x[:n], n, te)
                single = model(torch.cat([x[:n], x[n:n+1]]), n, ee)
            assert torch.equal(full[2][:n], alone[2]) and torch.allclose(full[0][:n], alone[0], atol=3e-6, rtol=3e-6)
            assert torch.equal(full[2][n], single[2][n]) and torch.allclose(full[0][n], single[0][n], atol=3e-6, rtol=3e-6)
            assert all(torch.equal(value.cpu(), baseline_initial[key]) for key, value in model.named_buffers())
            saved = output/f'{region}_{arm}.pt'; torch.save({'config': model.config, 'state_dict': r.cpu_state(model)}, saved)
            fresh = construct(arm, model.config, 'cuda').eval(); fresh.load_state_dict(model.state_dict())
            with torch.no_grad(): assert torch.equal(fresh(x, n, ee)[0], full[0])
            cases.append({'region': region, 'arm': arm, 'steps': 6, 'native_initial_and_first_step_exact': True,
                'step2_native_objective_relation_gradient_difference_norm': delta_norm, 'buffers_preserved': True,
                'query_isolated': True, 'restore_without_training_runtime_exact': True, 'head_auxiliary_gradient_zero': True})
            model.executed_optimizer.model = None; model.executed_optimizer = None
        assert all(not torch.equal(second_gradients[i], second_gradients[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(r.digest(Path(name)) == value for name, value in protected.items())
    r.write_json(output/'preflight.json', {'math': math_result, 'cases': cases, 'native_SG4steps_all_provinces_exact': True,
        'source_input_hashes': protected, 'source_sha256': r.digest(Path(__file__)), 'elapsed_seconds': time.perf_counter()-tick})

def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 9 and check['native_SG4steps_all_provinces_exact']
    assert check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(name)) == value for name, value in check['source_input_hashes'].items())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    hashes = source_hashes(); hashes[str((output/'task_frozen.md').resolve())] = r.digest(output/'task_frozen.md')
    hashes[str((preflight_dir/'preflight.json').resolve())] = r.digest(preflight_dir/'preflight.json')
    started, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', {'task_id': TASK, 'branch': 'RELATION/v1.30', 'parent': 'v1.29-audit',
        'change_type': 'pivot', 'started_at': started, 'source_input_hashes': hashes, 'new_fits': 45, 'cached_fits': 45,
        'training_runtime_feature_metric_not_in_checkpoint': True, 'final_holdout_access': False})
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                export(holder[0], data, pre, output/region/f'fold-{fold:02d}'/arm, arm == 'FM' and fold == 1)
                holder[0].executed_optimizer.model = None; holder[0].executed_optimizer = None; holder.clear()
    fit_seconds = time.perf_counter()-tick
    analyze(output)
    assert all(r.digest(Path(name)) == value for name, value in hashes.items())
    r.write_json(output/'finished.json', {'started_at': started, 'finished_at': datetime.now().astimezone().isoformat(),
        'fit_export_seconds': fit_seconds, 'wall_seconds': time.perf_counter()-tick, 'protected_hashes_unchanged': True})

def analyze(output):
    folds, points, training = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, config, data, _ = r.load_fold(region, fold)
            n, y, predictions = int(data['n_train']), data['y'], {}
            for arm in (*CACHED, *NEW):
                path = directory(output, region, fold, arm)
                metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                name = CACHED[arm][1] if arm in CACHED else arm
                assert (metric['region'], metric['fold'], metric['arm'], metric['seed']) == (region, fold, name, config['seed'])
                with np.load(path/'responses.npz') as responses:
                    assert np.array_equal(responses['sample_ids'], data['sample_ids'])
                    prediction = responses['prediction'].copy(); predictions[arm] = prediction
                mt, mv = r.metrics(y[:n], prediction[:n]), r.metrics(y[n:], prediction[n:])
                assert all(abs(value[key]-metric[role][key]) < 1e-7 for role, value in [('train', mt), ('validation', mv)] for key in value)
                row = {'region': region, 'fold': fold, 'arm': arm, **mv, 'train_rmse': mt['rmse'],
                    'best_epoch': metric['best_epoch'], 'epochs_run': metric['epochs_run'], 'elapsed_seconds': metric['elapsed_seconds']}
                if arm in NEW:
                    diagnostic = json.loads((path/'readout_diagnostics.json').read_text(encoding='utf-8'))
                    row.update(selected_deployment_active=diagnostic['selected_deployment_active'],
                               selected_objective_exposed_steps=diagnostic['selected_objective_exposed_steps'])
                    updates = json.loads((path/'readout_history.json').read_text(encoding='utf-8'))
                    feedback = json.loads((path/'readout_feedback.json').read_text(encoding='utf-8'))
                    row['training_changed_geometry_steps'] = sum(not update['first_update_MSE_only'] and abs(update['native_auxiliary_difference']) > 1e-12 and gradient['geometry_gradient_norm'] > 1e-12 for update, gradient in zip(updates, feedback))
                    row['training_nonzero_geometry_feedback_steps'] = sum(update['geometry_gradient_norm'] > 1e-12 for update in feedback)
                folds.append(row)
            q80 = np.quantile(y[:n], .8)
            for i, identity in enumerate(data['sample_ids']):
                row = {'sample_id': str(identity), 'region': region, 'fold': fold, 'observed_agb': float(y[i]),
                       'high_agb': int(y[i] >= q80), 'missing_features_current': int(data['missing_mask'][i].sum())}
                for arm, prediction in predictions.items():
                    row[arm+'_prediction'], row[arm+'_squared_error'] = float(prediction[i]), float((prediction[i]-y[i])**2)
                (training if i < n else points).append(row)
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    p = p.merge(pd.read_csv(context.PAIN/'all_1421_ids.csv', float_precision='round_trip'),
        on=['sample_id', 'region', 'fold', 'observed_agb'], how='left', validate='one_to_one', suffixes=('', '_archive'))
    p = p.merge(pd.read_csv(context.NAMED, float_precision='round_trip'),
        on=['sample_id', 'region', 'fold'], how='left', validate='one_to_one')
    p = p.merge(pd.read_csv(context.TAIL, float_precision='round_trip'),
        on=['sample_id', 'region'], how='left', validate='one_to_one', suffixes=('', '_tail'))
    p = p.join(t.groupby('sample_id').agg(FM_train_MSE=('FM_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean')),
               on='sample_id', validate='one_to_one')
    assert p.feature1.notna().all() and p.FM_train_MSE.notna().all()
    p['FM_minus_SG_SSE'] = p.FM_squared_error-p.SG_squared_error
    p['training_FM_minus_SG_MSE'] = p.FM_train_MSE-p.SG_train_MSE
    p['failure_phase'] = np.select([
        (p.FM_minus_SG_SSE > 0)&(p.training_FM_minus_SG_MSE > 0),
        (p.FM_minus_SG_SSE > 0)&(p.training_FM_minus_SG_MSE <= 0), p.FM_minus_SG_SSE < 0],
        ['training_and_OOF_worse', 'training_better_or_tied_OOF_worse', 'OOF_better'], default='OOF_tied')
    strata = []
    for region, group in p.groupby('region'):
        masks = {'all': np.ones(len(group), dtype=bool), 'high': group.high_agb == 1, 'ordinary': group.high_agb == 0,
            'missing': group.missing_features_current > 0, 'complete': group.missing_features_current == 0,
            'complete_typical': (group.missing_features_current == 0)&(group.max_abs_training_standardized_feature <= 3),
            'persistent_tail': group.tail_appearances >= 3}
        for label, mask in masks.items():
            selected = group[mask]
            if len(selected):
                strata.append({'region': region, 'stratum': label, 'n': len(selected),
                    'FM_minus_SG_SSE': float(selected.FM_minus_SG_SSE.sum()),
                    'FM_bias': float((selected.FM_prediction-selected.observed_agb).mean()),
                    'SG_bias': float((selected.SG_prediction-selected.observed_agb).mean())})
    features = []
    for region in r.REGIONS:
        group = p[p.region == region]; raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold); n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]): data['raw_features'][i] for i in range(n, len(data['y']))})
        matrix = np.stack([raw[identity] for identity in group.sample_id]); change = group.FM_minus_SG_SSE.to_numpy()
        for index, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, index]; valid = np.isfinite(values)&(values != -9999)
            correlation = pd.Series(values[valid]).corr(pd.Series(change[valid]), method='spearman')
            features.append({'region': region, 'feature': name, 'valid_n': int(valid.sum()),
                'spearman_feature_vs_FM_SSE_change': float(correlation) if np.isfinite(correlation) else None,
                'missing_SSE_change': float(change[~valid].sum()), 'observed_SSE_change': float(change[valid].sum())})
    assert len(features) == 555
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False); t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False)
    pd.DataFrame(features).to_csv(output/'all_555_feature_error_context.csv', index=False)
    pd.concat([p.nsmallest(20, 'FM_minus_SG_SSE'), p.nlargest(20, 'FM_minus_SG_SSE')]).to_csv(output/'forty_explanatory_cases.csv', index=False)
    pivot = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    gains = {arm: float((100*(1-pivot[arm]/pivot.G0)).mean()) for arm in (*CACHED, *NEW) if arm != 'G0'}
    comparisons = {arm: float((100*(1-pivot.FM/pivot[arm])).mean()) for arm in ('DM', 'PM', 'SG', 'GR')}
    province = f.groupby(['region', 'arm'])[['rmse', 'r2']].mean()
    noninferior = all(province.loc[(region, 'FM'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*(1+1e-6)
        and province.loc[(region, 'FM'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-1e-6 for region in r.REGIONS)
    cap = all(province.loc[(region, 'FM'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*1.01
        and province.loc[(region, 'FM'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-.01 for region in r.REGIONS)
    main = f[f.arm == 'FM']; active = main.set_index(['region', 'fold']).selected_deployment_active.astype(bool)
    deployment = int(active.sum()) >= 9 and all(int(active.loc[region].sum()) >= 2 for region in r.REGIONS)
    valid = bool((main.training_changed_geometry_steps > 0).all() and (main.training_nonzero_geometry_feedback_steps > 0).all())
    full = gains['FM'] >= 10 and comparisons['DM'] >= .5 and comparisons['PM'] >= .5 and comparisons['SG'] > 0 and comparisons['GR'] > 0 and noninferior and deployment
    pareto = gains['FM'] >= 3 and comparisons['SG'] >= .5 and all(value > 0 for value in comparisons.values()) and cap and deployment
    r.write_json(output/'summary.json', {'candidate_G0_RMSE_gains_percent': gains, 'FM_paired_RMSE_reductions_percent': comparisons,
        'province_G0_noninferiority': bool(noninferior), 'active_selected_FM_folds': int(active.sum()), 'deployment_gate': bool(deployment),
        'intervention_valid': valid, 'FULL': bool(full and valid), 'PARETO': bool(pareto and valid),
        'classification': 'INVALID' if not valid else 'FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        'FM_helped_SG': int((p.FM_minus_SG_SSE < 0).sum()), 'FM_harmed_SG': int((p.FM_minus_SG_SSE > 0).sum()),
        'FM_minus_SG_SSE': float(p.FM_minus_SG_SSE.sum()), 'all_1421_IDs': True,
        'training_appearances_per_new_method': 5684, 'feature_context_rows': 555, 'final_holdout_access': False})
    print((output/'summary.json').read_text(encoding='utf-8'), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight': preflight(args.output)
    else:
        assert args.preflight_dir is not None
        run(args.output, args.preflight_dir)

if __name__ == '__main__': main()
