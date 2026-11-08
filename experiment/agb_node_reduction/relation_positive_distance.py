"""Numerical repair of scalar SG endpoint distances, without changing its model."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
import relation_backbone_curvature as previous
import relation_ordinal_field as ordinal
import relation_adapt_combine as archive

r, sphere, layer = previous.r, previous.sphere, previous.layer
context, CACHED = archive.context, previous.CACHED
TASK, NEW = 'CARBON-RELATION-POSITIVE-DISTANCE-001', ('NS', 'NA', 'NF')


def endpoint_weights(model, u, log_metric):
    root = (.5*log_metric).exp()
    origin = model.field_support.mean(0)
    query, support = (u-origin)*root, (model.field_support-origin)*root
    reference = torch.cdist(support, support, compute_mode='donot_use_mm_for_euclid_dist').square()
    mask = torch.eye(len(support), device=u.device, dtype=torch.bool)
    tau = reference.masked_fill(mask, torch.inf).topk(8, largest=False).values.mean().clamp_min(1e-8)
    distance = torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist').square()
    return torch.softmax(-distance/tau+model.field_log_norm, 1), torch.softmax(-reference/tau+model.field_log_norm, 1)


def stable_distance(model, u, theta, statistics=None):
    w, wt = endpoint_weights(model, u, theta)
    root = (.5*theta).exp()
    origin = model.field_support.mean(0)
    query, support = (u-origin)*root, (model.field_support-origin)*root
    a, b = query@model.field_directions.T, support@model.field_directions.T
    positive_left = (w*a.square()).sum(1)[:, None]+w@b.square().T
    positive_right = a.square()@wt.T+(wt*b.square()).sum(1)[None, :]
    left = positive_left-2*(w*a)@b.T
    right = positive_right-2*a@(wt*b).T
    base = torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist').square()
    distance = (185*(left+right)*.5+.01*base)/1.01
    scale = (185*(positive_left+positive_right)*.5
             +.01*(query.square().sum(1)[:, None]+support.square().sum(1)[None, :]))/1.01
    eps = torch.finfo(u.dtype).eps; terms = len(model.field_directions)+u.shape[1]
    gamma = terms*eps/(1-terms*eps)
    pairs = torch.nonzero(distance.detach() <= 8*gamma*scale.detach()/1e-8)
    if len(pairs):
        direct = []
        for block in pairs.split(256):
            i, j = block.unbind(1)
            delta = (u[i]-model.field_support[j])*root
            projected = delta@model.field_directions.T
            direct.append((185*(projected.square()*(w[i]+wt[j])*.5).sum(1)
                           +.01*delta.square().sum(1))/1.01)
        distance = distance.index_put(tuple(pairs.unbind(1)), torch.cat(direct))
    if statistics is not None:
        statistics.update(direct_fallback_pairs=len(pairs), total_pairs=distance.numel())
    diagonal = (185*(w@model.field_directions.square())+.01)/1.01*theta.exp()
    return distance.clamp_min(0), diagonal


def direct_distance(model, u, theta):
    """Independent, nonnegative difference-before-projection reference."""
    w, wt = endpoint_weights(model, u, theta)
    values = []
    for i in range(len(u)):
        delta = (u[i]-model.field_support)*(.5*theta).exp()
        projected = delta@model.field_directions.T
        values.append((185*(projected.square()*(w[i]+wt)*.5).sum(1)+.01*delta.square().sum(1))/1.01)
    return torch.stack(values)


class Optimizer:
    def __init__(self, model, config):
        self.model = model
        self.adam = torch.optim.Adam(model.backbone.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.geometry = (None if model.numeric_arm == 'NF' else
                         torch.optim.Adam([model.layer_log_metric], lr=.001, weight_decay=0) if model.numeric_arm == 'NA' else
                         layer.RDoG(model.layer_log_metric))
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.model.layer_log_metric.grad = None

    def step(self):
        self.adam.step()
        if self.geometry is not None:
            self.geometry.step()
        self.steps += 1

    def diagnostics(self):
        return dict(geometry_optimizer='fixed_zero' if self.geometry is None else 'Adam' if self.model.numeric_arm == 'NA' else 'native_RDoG_flat_log',
                    steps=self.steps, geometry_state_scope='training_end_not_selected_checkpoint',
                    numerical_repair='original_ordinal_v58_arithmetic_with_original_scalar_SG_field', new_parameters=0)


class PositiveDistanceSAGE(previous.CurvatureSAGE):
    def __init__(self, arm, config):
        super().__init__('JA', config)
        self.numeric_arm = arm
        self.layer_log_metric.requires_grad_(arm != 'NF')

    def endpoint_weights(self, u, theta):
        if self.numeric_arm == 'NC':
            return super().endpoint_weights(u, theta)
        return endpoint_weights(self, u, theta)

    def corrected_distance(self, u, theta):
        if self.numeric_arm == 'NC':
            return super().corrected_distance(u, theta)
        return stable_distance(self, u, theta)

    def make_optimizer(self, config):
        return Optimizer(self, config)


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as saved:
            config['relation_n_train'] = int(saved['n_train'])
    r.fix_seed(config['seed'], 1)
    return PositiveDistanceSAGE(arm, config).to(device)


def source_hashes():
    hashes = archive.source_hashes()
    paths = [Path(__file__).resolve(), Path(previous.__file__), Path(sphere.__file__), Path(ordinal.__file__)]
    for region in r.REGIONS:
        for fold in range(1, 6):
            for source, alias in CACHED.values():
                paths += [source/region/f'fold-{fold:02d}'/alias/name for name in ('best.pt', 'responses.npz', 'metrics.json')]
    for path in paths:
        hashes[str(path.resolve())] = r.digest(path)
    return hashes


def directory(output, region, fold, arm):
    return CACHED[arm][0]/region/f'fold-{fold:02d}'/CACHED[arm][1] if arm in CACHED else output/region/f'fold-{fold:02d}'/arm


def math_checks():
    generator = torch.Generator().manual_seed(20261020)
    support = torch.randn(11, 185, generator=generator, dtype=torch.float64)
    directions = torch.randn(11, 185, generator=generator, dtype=torch.float64)
    directions /= directions.norm(dim=1, keepdim=True)
    model = SimpleNamespace(field_support=support, field_directions=directions,
                            field_log_norm=torch.randn(11, generator=generator, dtype=torch.float64))
    model.endpoint_weights = lambda u, theta: endpoint_weights(model, u, theta)
    query = torch.randn(4, 185, generator=generator, dtype=torch.float64)
    theta = torch.randn(185, generator=generator, dtype=torch.float64).requires_grad_()
    stable, _ = stable_distance(model, query, theta)
    literal = direct_distance(model, query, theta)
    value_error = float((stable-literal).abs().max().detach())
    assert torch.allclose(stable, literal, atol=1e-9, rtol=1e-10)
    cotangent = torch.randn(4, 11, generator=generator, dtype=torch.float64)
    first = torch.autograd.grad((stable*cotangent).sum(), theta)[0]
    second = torch.autograd.grad((literal*cotangent).sum(), theta)[0]
    gradient_error = float((first-second).abs().max())
    assert torch.allclose(first, second, atol=1e-8, rtol=1e-9)
    indices = [0, 37, 184]
    differences = []
    for index in indices:
        offset = torch.zeros_like(theta); offset[index] = 1e-5
        positive = (stable_distance(model, query, theta.detach()+offset)[0]*cotangent).sum()
        negative = (stable_distance(model, query, theta.detach()-offset)[0]*cotangent).sum()
        differences.append(float(abs((positive-negative)/2e-5-first[index])))
    assert max(differences) < 1e-5
    model.field_support = support+1e9
    large_query = query+1e9
    large_theta = theta.detach().clone().requires_grad_()
    positive = stable_distance(model, large_query, large_theta)[0]
    reference = direct_distance(model, large_query, large_theta)
    large_error = float((positive-reference).abs().max().detach())
    assert torch.allclose(positive, reference, atol=1e-8, rtol=1e-9)
    gp = torch.autograd.grad((positive*cotangent).sum(), large_theta)[0]
    gr = torch.autograd.grad((reference*cotangent).sum(), large_theta)[0]
    assert torch.allclose(gp, gr, atol=1e-7, rtol=1e-8)
    return dict(positive_identity_max_abs_error=value_error, gradient_max_abs_error=gradient_error,
                finite_difference_indices=indices, finite_difference_errors=differences,
                large_common_offset=1e9, large_offset_positive_error=large_error,
                large_offset_gradient_max_abs_error=float((gp-gr).abs().max()))


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter(); cases = []; native_cases = []; timings = []
    math_result = math_checks()
    for region in r.REGIONS:
        _, cfg, data, pre = r.load_fold(region, 1); n = int(data['n_train'])
        x, te, ee, y = r.tensors(data, pre, 'cuda')
        original = previous.construct('JA', cfg, 'cuda'); original.fit_reference(x[:n], te)
        disabled = construct('NC', cfg, 'cuda'); disabled.fit_reference(x[:n], te)
        first, second = sphere.Optimizer(original, cfg), disabled.make_optimizer(cfg)
        original.train(); disabled.train()
        for _ in range(4):
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            first.zero_grad(); a = original.training_loss(x[:n], y, te); a.backward()
            nn.utils.clip_grad_norm_(original.parameters(), cfg['gradient_clip']); first.step()
            after = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
            second.zero_grad(); b = disabled.training_loss(x[:n], y, te); b.backward()
            nn.utils.clip_grad_norm_(disabled.parameters(), cfg['gradient_clip']); second.step()
            assert torch.equal(a, b)
            assert all(torch.equal(value, disabled.state_dict()[key]) for key, value in original.state_dict().items())
            assert torch.equal(after[0], torch.get_rng_state()) and torch.equal(after[1], torch.cuda.get_rng_state())
        native_cases.append(dict(region=region, disabled_native_four_steps_bitexact=True))
        for arm in ('NC', *NEW):
            model = construct(arm, cfg, 'cuda'); model.fit_reference(x[:n], te)
            before = r.cpu_state(model); optimizer = model.make_optimizer(cfg)
            assert len(list(model.named_buffers())) == 16 and sum(p.numel() for p in model.parameters()) == 28275
            with torch.no_grad():
                model.eval()
                initial_prediction = model(x, n, ee)[0]
                baseline = construct('NC', cfg, 'cuda').eval(); baseline.fit_reference(x[:n], te)
                assert torch.allclose(initial_prediction, baseline(x, n, ee)[0], atol=3e-6, rtol=3e-6)
                del baseline
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            for _ in range(4):
                model.train(); optimizer.zero_grad(); loss = model.training_loss(x[:n], y, te)
                assert torch.isfinite(loss); loss.backward()
                if arm != 'NF':
                    assert model.layer_log_metric.grad is not None and torch.isfinite(model.layer_log_metric.grad).all()
                nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); optimizer.step()
                model.eval()
                with torch.no_grad():
                    model(x, n, ee)
            torch.cuda.synchronize()
            timings.append(dict(region=region, arm=arm, four_train_and_eval_seconds=time.perf_counter()-started,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
            assert all(torch.equal(value.cpu(), before[name]) for name, value in model.named_buffers())
            if arm == 'NF':
                assert torch.count_nonzero(model.layer_log_metric) == 0
            else:
                assert model.layer_log_metric.norm(dim=1).min() > 0
            with torch.no_grad():
                full = model(x, n, ee)[0]
                prefix = model(x[:n], n, te)[0]
                single = model(torch.cat([x[:n], x[n:n+1]]), n, ee)[0]
                extreme = model(torch.cat([x, torch.full_like(x[n:n+1], 1e6)]), n, ee)[0]
                assert torch.allclose(full[:n], prefix, atol=3e-6, rtol=3e-6)
                assert torch.allclose(full[n], single[n], atol=3e-6, rtol=3e-6)
                assert torch.allclose(full, extreme[:len(x)], atol=3e-6, rtol=3e-6)
                restored = construct(arm, cfg, 'cuda').eval(); restored.load_state_dict(r.cpu_state(model))
                assert torch.equal(full, restored(x, n, ee)[0])
                if arm != 'NC':
                    u = model.coordinates(x[n:n+3]); theta = model.layer_log_metric[0]
                    actual = model.corrected_distance(u, theta)[0]
                    literal = direct_distance(model, u, theta)
                    assert torch.allclose(actual, literal, atol=1e-6, rtol=1e-8)
            cases.append(dict(region=region, arm=arm, four_real_updates=True, native16buffers_fixed=True,
                              query_prefix_single_extreme_independent=True, restore_bitexact=True))
            del model, restored, optimizer
    _, cfg, data, pre = r.load_fold('hunan', 3); n = int(data['n_train'])
    x, te, ee, y = r.tensors(data, pre, 'cuda')
    faulty = construct('NS', cfg, 'cuda').eval()
    state = torch.load(faulty.reference, map_location='cuda', weights_only=False)['state_dict']
    faulty.load_state_dict(state)
    with torch.no_grad():
        u = faulty.coordinates(x); theta = faulty.layer_log_metric[0]
        indices = [156, 158, 392]
        actual = faulty.corrected_distance(u[indices], theta)[0]
        literal = direct_distance(faulty, u[indices], theta)
        assert torch.allclose(actual, literal, atol=1e-5, rtol=1e-8)
        regression = dict(region='hunan', fold=3, layer=0, internal_rows=indices,
                          theta_max=float(theta.max()), relative_distance_error=float(((actual-literal).abs()/literal.abs().clamp_min(1)).max()))
    r.write_json(output/'preflight.json', dict(source_sha256=r.digest(Path(__file__)), source_input_hashes=source_hashes(),
                 math=math_result, cases=cases, native_cases=native_cases, known_fault_regression=regression,
                 timings=timings, wall_seconds=time.perf_counter()-tick))
    print('PREFLIGHT PASS', len(cases), json.dumps(timings), flush=True)


@torch.no_grad()
def export(model, data, pre, output, restore=False):
    n = int(data['n_train']); x, te, ee, y = r.tensors(data, pre, 'cuda')
    model.eval(); full = model(x, n, ee)[0]; arrays = {}
    numeric = []
    for index, (near, weights, diagonal) in enumerate(model.layer_graphs(x, n)):
        arrays.update({f'layer{index}_near': near.cpu().numpy(), f'layer{index}_weights': weights.cpu().numpy(),
                       f'layer{index}_diagonal': diagonal.cpu().numpy()})
        stats = {}; u = model.coordinates(x); theta = model.layer_log_metric[index]
        stable_distance(model, u, theta, stats)
        rows = [0, n//2, n-1, n, len(x)-1]
        actual = model.corrected_distance(u[rows], theta)[0]
        literal = direct_distance(model, u[rows], theta)
        error = float(((actual-literal).abs()/literal.abs().clamp_min(1)).max())
        assert error <= 1e-8
        numeric.append(dict(layer=index, reference_internal_rows=rows, relative_positive_reference_error=error, **stats))
    np.savez_compressed(output/'two_layer_graphs.npz', **arrays)
    r.write_json(output/'spherical_loss_history.json', model.geometry_history)
    if restore:
        restored = construct(model.numeric_arm, model.config, 'cuda').eval(); restored.load_state_dict(r.cpu_state(model))
        assert torch.equal(full, restored(x, n, ee)[0])
    with np.load(output/'responses.npz') as saved:
        raw = full.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
        assert np.array_equal(raw, saved['prediction'])
    r.write_json(output/'numeric_diagnostics.json', dict(arm=model.numeric_arm,
        layer_log_norms=model.layer_log_metric.norm(dim=1).tolist(), theta_min=float(model.layer_log_metric.min()),
        theta_max=float(model.layer_log_metric.max()), parameter_count=28275, native16buffers_fixed=True,
        restored_prediction_bitexact=bool(restore), scalar_field_directions=len(model.field_directions), selected_numeric_checks=numeric))


def analyze(output):
    folds, points, training = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, cfg, data, pre = r.load_fold(region, fold); n = int(data['n_train']); y = data['y']; predictions = {}
            for arm in (*CACHED, *NEW):
                path = directory(output, region, fold, arm)
                metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                with np.load(path/'responses.npz') as saved:
                    assert np.array_equal(saved['sample_ids'], data['sample_ids'])
                    predictions[arm] = saved['prediction'].copy()
                mt, mv = r.metrics(y[:n], predictions[arm][:n]), r.metrics(y[n:], predictions[arm][n:])
                assert all(abs(values[key]-metric[role][key]) < 1e-7 for role, values in [('train', mt), ('validation', mv)] for key in values)
                row = dict(region=region, fold=fold, arm=arm, **mv, train_rmse=mt['rmse'], best_epoch=metric['best_epoch'],
                           epochs_run=metric['epochs_run'], elapsed_seconds=metric['elapsed_seconds'])
                if arm in NEW:
                    diag = json.loads((path/'numeric_diagnostics.json').read_text())
                    row['geometry_active'] = min(diag['layer_log_norms']) > 1e-10
                folds.append(row)
            for i, identity in enumerate(data['sample_ids']):
                row = dict(sample_id=str(identity), region=region, fold=fold, observed_agb=float(y[i]),
                    high_agb=int(y[i] >= np.quantile(y[:n], .8)), missing_features_current=int(data['missing_mask'][i].sum()))
                for arm, pred in predictions.items():
                    row[arm+'_prediction'], row[arm+'_squared_error'] = float(pred[i]), float((pred[i]-y[i])**2)
                (training if i < n else points).append(row)
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    for path, keys in [(context.PAIN/'all_1421_ids.csv', ['sample_id', 'region', 'fold', 'observed_agb']),
                       (context.NAMED, ['sample_id', 'region', 'fold']), (context.TAIL, ['sample_id', 'region'])]:
        p = p.merge(pd.read_csv(path, float_precision='round_trip'), on=keys, how='left', validate='one_to_one', suffixes=('', '_archive'))
    p = p.join(t.groupby('sample_id').agg(NS_train_MSE=('NS_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean')), on='sample_id', validate='one_to_one')
    assert p.feature1.notna().all() and p.NS_train_MSE.notna().all()
    p['NS_minus_SG_SSE'] = p.NS_squared_error-p.SG_squared_error
    p['training_NS_minus_SG_MSE'] = p.NS_train_MSE-p.SG_train_MSE
    p['failure_phase'] = np.select([(p.NS_minus_SG_SSE > 0)&(p.training_NS_minus_SG_MSE > 0),
        (p.NS_minus_SG_SSE > 0)&(p.training_NS_minus_SG_MSE <= 0), p.NS_minus_SG_SSE < 0],
        ['training_and_OOF_worse', 'training_better_or_tied_OOF_worse', 'OOF_better'], default='OOF_tied')
    strata, features = [], []
    for region, group in p.groupby('region'):
        masks = dict(all=np.ones(len(group), bool), high=group.high_agb == 1, ordinary=group.high_agb == 0,
            missing=group.missing_features_current > 0, complete=group.missing_features_current == 0,
            complete_typical=(group.missing_features_current == 0)&(group.max_abs_training_standardized_feature <= 3), persistent_tail=group.tail_appearances >= 3)
        for label, mask in masks.items():
            selected = group[mask]
            if len(selected):
                strata.append(dict(region=region, stratum=label, n=len(selected), NS_minus_SG_SSE=float(selected.NS_minus_SG_SSE.sum()),
                    NS_bias=float((selected.NS_prediction-selected.observed_agb).mean()), SG_bias=float((selected.SG_prediction-selected.observed_agb).mean())))
        raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold); n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]): data['raw_features'][i] for i in range(n, len(data['y']))})
        matrix = np.stack([raw[identity] for identity in group.sample_id]); changes = group.NS_minus_SG_SSE.to_numpy()
        for index, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, index]; valid = np.isfinite(values)&(values != -9999)
            corr = pd.Series(values[valid]).corr(pd.Series(changes[valid]), method='spearman')
            features.append(dict(region=region, feature=name, valid_n=int(valid.sum()), spearman_feature_vs_NS_SSE_change=float(corr) if np.isfinite(corr) else None,
                missing_SSE_change=float(changes[~valid].sum()), observed_SSE_change=float(changes[valid].sum())))
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False); t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False); pd.DataFrame(features).to_csv(output/'all_555_feature_contexts.csv', index=False)
    pd.concat([p.nlargest(20, 'NS_minus_SG_SSE'), p.nsmallest(20, 'NS_minus_SG_SSE')]).to_csv(output/'forty_cases.csv', index=False)
    wide = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    gains = {arm: float((100*(1-wide[arm]/wide.G0)).mean()) for arm in (*CACHED, *NEW) if arm != 'G0'}
    compared = {arm: float((100*(1-wide.NS/wide[arm])).mean()) for arm in ('NA', 'NF', 'SG', 'GR')}
    province_gate = province_cap = True
    for region in r.REGIONS:
        base, candidate = [f[(f.region == region)&(f.arm == arm)][['rmse', 'r2']].mean() for arm in ('G0', 'NS')]
        province_gate &= candidate.rmse <= base.rmse+1e-6 and candidate.r2 >= base.r2-1e-6
        province_cap &= candidate.rmse <= 1.01*base.rmse and candidate.r2 >= base.r2-.01
    activity = {region: int(f[(f.region == region)&(f.arm == 'NS')].geometry_active.sum()) for region in r.REGIONS}
    active = sum(activity.values()) >= 9 and min(activity.values()) >= 2
    full = gains['NS'] >= 10 and min(compared[a] for a in ('NA', 'NF')) >= .5 and min(compared['SG'], compared['GR']) > 0 and province_gate and active
    pareto = gains['NS'] >= 3 and compared['SG'] >= .5 and min(compared.values()) > 0 and province_cap and active
    summary = dict(task_id=TASK, G0_RMSE_gains_percent=gains, NS_control_gains_percent=compared,
        province_G0_noninferiority=bool(province_gate), province_cap=bool(province_cap), selected_NS_activity=activity,
        training_intervention_gate=bool(active), FULL_GAIN=bool(full), PARETO_GAIN=bool(pareto),
        classification='FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN', all_development_IDs=1421,
        training_appearances_per_method=5684, feature_contexts=555, NS_helped_SG=int((p.NS_minus_SG_SSE < 0).sum()),
        NS_harmed_SG=int((p.NS_minus_SG_SSE > 0).sum()), NS_minus_SG_total_SSE=float(p.NS_minus_SG_SSE.sum()),
        failure_phase_counts=p.failure_phase.value_counts().to_dict(), final_holdout_access=False)
    r.write_json(output/'summary.json', summary); print(json.dumps(summary), flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 12 and all(row['disabled_native_four_steps_bitexact'] for row in check['native_cases'])
    assert check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(path)) == value for path, value in check['source_input_hashes'].items())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/Path(__file__).name); shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    hashes = source_hashes()
    for path in (output/'task_frozen.md', preflight_dir/'preflight.json'):
        hashes[str(path.resolve())] = r.digest(path)
    started = datetime.now().astimezone().isoformat(); tick = time.perf_counter()
    r.write_json(output/'manifest.json', dict(task_id=TASK, branch='RELATION/v1.44', parent='RELATION/v1.43', change_type='repair',
        started_at=started, source_input_hashes=hashes, new_fits=45, cached_fits=45, device=torch.cuda.get_device_name(),
        total_memory_bytes=torch.cuda.get_device_properties(0).total_memory, parameter_count=28275, final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                export(holder[0], data, pre, directory(output, region, fold, arm), restore=arm == 'NS' and fold == 1)
                holder.clear()
    elapsed = time.perf_counter()-tick
    analyze(output)
    assert all(r.digest(Path(path)) == value for path, value in hashes.items())
    r.write_json(output/'finished.json', dict(started_at=started, finished_at=datetime.now().astimezone().isoformat(),
        fit_export_seconds=elapsed, wall_seconds=time.perf_counter()-tick, protected_hashes_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('math', 'preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path); parser.add_argument('--preflight-dir', type=Path); args = parser.parse_args()
    r.setup()
    if args.mode == 'math': print(json.dumps(math_checks()))
    elif args.mode == 'preflight': preflight(args.output)
    elif args.mode == 'run': run(args.output, args.preflight_dir)
    else: analyze(args.output)
