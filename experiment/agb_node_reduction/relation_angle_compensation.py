"""Training-label angle compensation in the original SG predictor."""
import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import relation_functional_conformal as archive

r, previous, sphere, layer = archive.r, archive.previous, archive.sphere, archive.layer
context, CACHED = archive.context, archive.CACHED
TASK, NEW = 'CARBON-RELATION-ANGLE-COMPENSATION-001', ('AC', 'SC', 'PC')


def pair_context(target, bins, seed):
    positive = (bins[:, None] == bins[None, :]) & ~torch.eye(len(bins), device=bins.device, dtype=torch.bool)
    negative = bins[:, None] != bins[None, :]
    target = target.detach().double()
    angle = math.pi*(1-(target[:, None]-target[None, :]).abs()/(target.max()-target.min()))
    generator = torch.Generator(device='cpu').manual_seed(seed+128000)
    permutation = torch.arange(len(target)).expand(len(target), -1).clone()
    for i, row in enumerate(negative.cpu()):
        indices = torch.where(row)[0]
        order = torch.randperm(len(indices), generator=generator)
        permutation[i, indices[order]] = indices[order.roll(1)]
    assert (positive.sum(1) > 0).all() and (negative.sum(1) > 1).all()
    return positive, negative, angle, permutation.to(target.device)


def pair_loss(feature, positive, negative, angle, permutation, arm):
    z = F.normalize(feature.double(), dim=1)
    cosine = (z@z.T).clamp(-1, 1)
    phi = angle.gather(1, permutation) if arm == 'PC' else angle
    adjusted = cosine*phi.cos()-((1-cosine.square()).clamp_min(0)+1e-6).sqrt()*phi.sin().abs()
    similarity = torch.where(negative, adjusted, cosine) if arm != 'SC' else cosine
    logits = (similarity/.1).masked_fill(torch.eye(len(z), device=z.device, dtype=torch.bool), -torch.inf)
    loss = (torch.logsumexp(logits, dim=1)-(cosine.masked_fill(~positive, 0)/.1).sum(1)/positive.sum(1)).mean()
    return loss, cosine, similarity


class Optimizer(sphere.Optimizer):
    def step(self):
        gradient = self.model.layer_log_metric.grad
        self.model.angle_feedback.append({'step': len(self.model.angle_feedback)+1,
            'geometry_gradient_norm': float(gradient.norm()), 'backbone_gradient_finite':
            all(bool(torch.isfinite(p.grad).all()) for p in self.model.backbone.parameters() if p.grad is not None)})
        super().step()


class AngleSAGE(previous.CurvatureSAGE):
    training_objective = 'native_SG_MSE_and_spherical_terms;pair_contrast_with_correct_none_or_permuted_negative_angle'

    def __init__(self, arm, config):
        super().__init__('JA', config)
        self.angle_arm, self.angle_feedback, self.executed_optimizer = arm, [], None

    def fit_relations(self, x, target, edges):
        history = super().fit_relations(x, target, edges)
        self.pair_data = pair_context(target, self.label_bins, self.config['seed'])
        return history

    def graph(self, u, log_metric, dtype, support=None):
        distance, diagonal = self.corrected_distance(u, log_metric)
        n = len(self.field_support)
        mask = torch.arange(len(u), device=u.device)[:, None] == torch.arange(n, device=u.device)[None, :]
        if support is not None: mask = mask | (~support)[None, :]
        distance = distance.masked_fill(mask, torch.inf)
        near = torch.argsort(distance, dim=1, stable=True)[:, :8]
        weights = torch.softmax(-distance.gather(1, near)/self.temperature, dim=1).to(dtype)
        return near, weights, diagonal

    def training_loss(self, x, target, edges):
        if self.angle_arm == 'N': return sphere.SphericalSAGE.training_loss(self, x, target, edges)
        prediction = self(x, len(target), edges)[0]
        self.last_training_prediction = prediction
        mse = F.mse_loss(prediction, target)
        envelope, homogeneous, prototype, centers = sphere.spherical_terms(
            self.geometry_feature, self.label_bins, self.label_values, self.sphere_points)
        contrast, cosine, adjusted = pair_loss(self.geometry_feature, *self.pair_data, self.angle_arm)
        warmup = not self.geometry_history
        loss = mse if warmup else mse+contrast+.01*(envelope+homogeneous)
        self.geometry_history.append({'step': len(self.geometry_history)+1, 'mse': float(mse.detach()),
            'contrast': float(contrast.detach()), 'native_prototype_contrast': float(prototype.detach()),
            'envelope': float(envelope.detach()), 'homogeneous': float(homogeneous.detach()),
            'centroid_norm_error': float((centers.detach().norm(dim=1)-1).abs().max()),
            'negative_logit_adjustment_mean': float((adjusted-cosine)[self.pair_data[1]].detach().mean()),
            'first_update_MSE_only': warmup})
        return loss

    def make_optimizer(self, config):
        self.executed_optimizer = Optimizer(self, config)
        return self.executed_optimizer


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train'] = int(data['n_train'])
    r.fix_seed(config['seed'], 1)
    return AngleSAGE(arm, config).to(device)


def directory(output, region, fold, arm):
    root, name = CACHED[arm] if arm in CACHED else (output, arm)
    return root/region/f'fold-{fold:02d}'/name


def math_checks():
    generator = torch.Generator().manual_seed(128000)
    feature = torch.randn((6, 4), dtype=torch.float64, generator=generator, requires_grad=True)
    target, bins = torch.tensor([.1, .2, 1.1, 1.3, 2.4, 3.5], dtype=torch.float64), torch.arange(3).repeat_interleave(2)
    old_rng = torch.get_rng_state().clone(); data = pair_context(target, bins, 10)
    assert torch.equal(old_rng, torch.get_rng_state())
    pos, neg, angle, permutation = data
    assert (permutation[neg] != torch.arange(6).expand(6, -1)[neg]).all()
    assert all(torch.equal(angle[i, neg[i]].sort().values, angle.gather(1, permutation)[i, neg[i]].sort().values) for i in range(6))
    loss, cosine, adjusted = pair_loss(feature, *data, 'AC')
    literal = []
    for i in range(6):
        logits, numerator = [], []
        for j in range(6):
            if i == j: continue
            value = cosine[i, j]
            if neg[i, j]: value = value*torch.cos(angle[i, j])-torch.sqrt((1-value.square()).clamp_min(0)+1e-6)*torch.sin(angle[i, j]).abs()
            logits.append(value/.1)
            if pos[i, j]: numerator.append(cosine[i, j]/.1)
        literal.append(torch.logsumexp(torch.stack(logits), 0)-torch.stack(numerator).mean())
    literal_error = float((loss-torch.stack(literal).mean()).abs().detach())
    assert literal_error < 1e-12
    identity, _, _ = pair_loss(feature, pos, neg, torch.zeros_like(angle), permutation, 'AC')
    simple, _, _ = pair_loss(feature, *data, 'SC')
    assert torch.equal(identity, simple)
    affine = pair_context(target*3.7+2.1, bins, 10)
    affine_error = float((loss-pair_loss(feature, *affine, 'AC')[0]).abs().detach())
    rotation = torch.linalg.qr(torch.randn((4, 4), dtype=torch.float64, generator=generator))[0]
    rotation_error = float((loss-pair_loss(feature@rotation, *data, 'AC')[0]).abs().detach())
    assert affine_error < 1e-12 and rotation_error < 1e-12
    gradient = torch.autograd.grad(loss, feature)[0]
    plus, minus = feature.detach().clone(), feature.detach().clone(); plus[1, 2] += 1e-6; minus[1, 2] -= 1e-6
    fd = (pair_loss(plus, *data, 'AC')[0]-pair_loss(minus, *data, 'AC')[0])/2e-6
    fd_error = float(abs(fd-gradient[1, 2])); assert fd_error < 1e-6 and torch.isfinite(gradient).all()
    edge = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]], dtype=torch.float64, requires_grad=True)
    edge_loss = pair_loss(edge, *pair_context(torch.arange(4.), torch.tensor([0, 0, 1, 1]), 1), 'AC')[0]
    assert torch.isfinite(torch.autograd.grad(edge_loss, edge)[0]).all()
    return {'literal_loss_error': literal_error, 'zero_angle_exact_SupCon': True, 'label_affine_error': affine_error,
        'rotation_error': rotation_error, 'finite_difference_error': fd_error, 'collinear_finite': True,
        'negative_angle_multiset_and_derangement': True, 'default_RNG_preserved': True}


def source_hashes():
    return {**archive.source_hashes(), str(Path(__file__).resolve()): r.digest(Path(__file__))}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False); tick = time.perf_counter(); math_result = math_checks(); cases = []
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
                'query_isolated': True, 'restore_without_training_runtime_exact': True})
            model.executed_optimizer.model = None; model.executed_optimizer = None
        assert all(not torch.equal(second_gradients[i], second_gradients[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    r.write_json(output/'preflight.json', {'math': math_result, 'cases': cases, 'native_SG4steps_all_provinces_exact': True,
        'source_sha256': r.digest(Path(__file__)), 'elapsed_seconds': time.perf_counter()-tick})


@torch.no_grad()
def export(model, data, pre, output, restore):
    model.eval(); n = int(data['n_train']); x, _, ee, target = r.tensors(data, pre, 'cuda')
    actual, embedding, near, weight, diagonal = model(x, n, ee)
    feature = model.geometry_feature[:n].clone(); graphs = model.layer_graphs(x, n)
    prediction = actual.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    with np.load(output/'responses.npz') as saved: assert np.array_equal(prediction, saved['prediction'])
    metric = json.loads((output/'metrics.json').read_text(encoding='utf-8')); best = int(metric['best_epoch'])
    theta = model.layer_log_metric.clone(); model.layer_log_metric.zero_(); model._relation_cache.clear()
    bypass = model(x, n, ee)[0]
    model.layer_log_metric.copy_(theta); model._relation_cache.clear()
    difference = float((actual-bypass).abs().max())
    with np.load(directory(output, model.config['region'], model.config['fold'], 'SG')/'responses.npz') as saved:
        sg_difference = float(abs(prediction-saved['prediction']).max())
    exposed = sum(row['geometry_gradient_norm'] > 1e-12 for row in model.angle_feedback[1:best])
    pair, cosine, adjusted = pair_loss(feature, *model.pair_data, model.angle_arm)
    env, homo, proto, _ = sphere.spherical_terms(feature, model.label_bins, model.label_values, model.sphere_points)
    report = {'selected_epoch': best, 'selected_objective_exposed_steps': exposed,
        'selected_deployment_active': bool(exposed > 0 and difference > 1e-6 and sg_difference > 1e-6 and theta.norm() > 1e-10),
        'zero_log_bypass_prediction_max_standardized_difference': difference, 'SG_prediction_max_absolute_difference': sg_difference,
        'selected_pair_contrast': float(pair), 'selected_native_prototype_contrast': float(proto),
        'selected_envelope': float(env), 'selected_homogeneous': float(homo),
        'selected_negative_adjustment_mean': float((adjusted-cosine)[model.pair_data[1]].mean()),
        'training_context_checkpoint_parameter': False, 'restore_checked': restore}
    if restore:
        fresh = construct(model.angle_arm, model.config, 'cuda').eval(); fresh.load_state_dict(model.state_dict())
        with torch.no_grad(): assert torch.equal(fresh(x, n, ee)[0], actual)
        report['restore_exact'] = True
    arrays = {'sample_ids': data['sample_ids']}
    for i, graph in enumerate(graphs, 1):
        arrays[f'layer{i}_neighbors'], arrays[f'layer{i}_weights'] = graph[0].cpu().numpy(), graph[1].cpu().numpy()
    np.savez_compressed(output/'angle_graphs.npz', **arrays)
    np.savez_compressed(output/'angle_context.npz', sample_ids=data['sample_ids'][:n], feature=feature.cpu().numpy(),
        target=target.cpu().numpy(), bins=model.label_bins.cpu().numpy(), permutation=model.pair_data[3].cpu().numpy())
    r.write_json(output/'angle_diagnostics.json', report)
    r.write_json(output/'angle_history.json', model.geometry_history)
    r.write_json(output/'angle_feedback.json', model.angle_feedback)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 9 and check['native_SG4steps_all_provinces_exact']
    assert check['source_sha256'] == r.digest(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    hashes = source_hashes(); hashes[str((output/'task_frozen.md').resolve())] = r.digest(output/'task_frozen.md')
    started, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', {'task_id': TASK, 'branch': 'RELATION/v1.28', 'parent': 'v1.27-r1',
        'change_type': 'pivot', 'started_at': started, 'source_input_hashes': hashes, 'new_fits': 45, 'cached_fits': 45,
        'training_runtime_pair_context_not_in_checkpoint': True, 'final_holdout_access': False})
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                export(holder[0], data, pre, output/region/f'fold-{fold:02d}'/arm, arm == 'AC' and fold == 1)
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
                    diagnostic = json.loads((path/'angle_diagnostics.json').read_text(encoding='utf-8'))
                    row.update(selected_deployment_active=diagnostic['selected_deployment_active'],
                               selected_objective_exposed_steps=diagnostic['selected_objective_exposed_steps'])
                    updates = json.loads((path/'angle_history.json').read_text(encoding='utf-8'))
                    feedback = json.loads((path/'angle_feedback.json').read_text(encoding='utf-8'))
                    row['training_changed_geometry_steps'] = sum(not update['first_update_MSE_only'] and abs(update['contrast']-update['native_prototype_contrast']) > 1e-12 and gradient['geometry_gradient_norm'] > 1e-12 for update, gradient in zip(updates, feedback))
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
    p = p.join(t.groupby('sample_id').agg(AC_train_MSE=('AC_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean')),
               on='sample_id', validate='one_to_one')
    assert p.feature1.notna().all() and p.AC_train_MSE.notna().all()
    p['AC_minus_SG_SSE'] = p.AC_squared_error-p.SG_squared_error
    p['training_AC_minus_SG_MSE'] = p.AC_train_MSE-p.SG_train_MSE
    p['failure_phase'] = np.select([
        (p.AC_minus_SG_SSE > 0)&(p.training_AC_minus_SG_MSE > 0),
        (p.AC_minus_SG_SSE > 0)&(p.training_AC_minus_SG_MSE <= 0), p.AC_minus_SG_SSE < 0],
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
                    'AC_minus_SG_SSE': float(selected.AC_minus_SG_SSE.sum()),
                    'AC_bias': float((selected.AC_prediction-selected.observed_agb).mean()),
                    'SG_bias': float((selected.SG_prediction-selected.observed_agb).mean())})
    features = []
    for region in r.REGIONS:
        group = p[p.region == region]; raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold); n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]): data['raw_features'][i] for i in range(n, len(data['y']))})
        matrix = np.stack([raw[identity] for identity in group.sample_id]); change = group.AC_minus_SG_SSE.to_numpy()
        for index, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, index]; valid = np.isfinite(values)&(values != -9999)
            correlation = pd.Series(values[valid]).corr(pd.Series(change[valid]), method='spearman')
            features.append({'region': region, 'feature': name, 'valid_n': int(valid.sum()),
                'spearman_feature_vs_AC_SSE_change': float(correlation) if np.isfinite(correlation) else None,
                'missing_SSE_change': float(change[~valid].sum()), 'observed_SSE_change': float(change[valid].sum())})
    assert len(features) == 555
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False); t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False)
    pd.DataFrame(features).to_csv(output/'all_555_feature_error_context.csv', index=False)
    pd.concat([p.nsmallest(20, 'AC_minus_SG_SSE'), p.nlargest(20, 'AC_minus_SG_SSE')]).to_csv(output/'forty_explanatory_cases.csv', index=False)
    pivot = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    gains = {arm: float((100*(1-pivot[arm]/pivot.G0)).mean()) for arm in (*CACHED, *NEW) if arm != 'G0'}
    comparisons = {arm: float((100*(1-pivot.AC/pivot[arm])).mean()) for arm in ('SC', 'PC', 'SG', 'GR')}
    province = f.groupby(['region', 'arm'])[['rmse', 'r2']].mean()
    noninferior = all(province.loc[(region, 'AC'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*(1+1e-6)
        and province.loc[(region, 'AC'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-1e-6 for region in r.REGIONS)
    cap = all(province.loc[(region, 'AC'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*1.01
        and province.loc[(region, 'AC'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-.01 for region in r.REGIONS)
    main = f[f.arm == 'AC']; active = main.set_index(['region', 'fold']).selected_deployment_active.astype(bool)
    deployment = int(active.sum()) >= 9 and all(int(active.loc[region].sum()) >= 2 for region in r.REGIONS)
    valid = bool((main.training_changed_geometry_steps > 0).all() and (main.training_nonzero_geometry_feedback_steps > 0).all())
    full = gains['AC'] >= 10 and comparisons['SC'] >= .5 and comparisons['PC'] >= .5 and comparisons['SG'] > 0 and comparisons['GR'] > 0 and noninferior and deployment
    pareto = gains['AC'] >= 3 and comparisons['SG'] >= .5 and all(value > 0 for value in comparisons.values()) and cap and deployment
    r.write_json(output/'summary.json', {'candidate_G0_RMSE_gains_percent': gains, 'AC_paired_RMSE_reductions_percent': comparisons,
        'province_G0_noninferiority': bool(noninferior), 'active_selected_AC_folds': int(active.sum()), 'deployment_gate': bool(deployment),
        'intervention_valid': valid, 'FULL': bool(full and valid), 'PARETO': bool(pareto and valid),
        'classification': 'INVALID' if not valid else 'FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        'AC_helped_SG': int((p.AC_minus_SG_SSE < 0).sum()), 'AC_harmed_SG': int((p.AC_minus_SG_SSE > 0).sum()),
        'AC_minus_SG_SSE': float(p.AC_minus_SG_SSE.sum()), 'all_1421_IDs': True,
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
