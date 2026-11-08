"""Training-only continuous transport Fenchel-Young gap for original SG."""
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
import relation_readout_sphere as prior
import relation_balance as balance

base = prior.base
r, previous, sphere = prior.r, prior.previous, prior.sphere
context, CACHED, directory = prior.context, prior.CACHED, prior.directory
TASK, NEW = 'CARBON-RELATION-TRANSPORT-GAP-001', ('BT', 'RT', 'PT')
GAMMA = .1


def entropy(plan):
    positive = plan > 0
    return (plan[positive]*plan[positive].log()).sum()


def feature_cost(raw):
    unit = F.normalize(raw.double(), dim=1)
    cost = 1-unit@unit.T
    return (.5*(cost+cost.T)).clamp(0, 2)


@torch.no_grad()
def plan(cost, domain):
    n = len(cost)
    log_kernel = -cost.detach()/GAMMA
    log_kernel.fill_diagonal_(-torch.inf)
    if domain == 'RT':
        value = torch.softmax(log_kernel, 1)/n
        iterations, residual = 0, float((value.sum(1)*n-1).abs().max())
        scaling = torch.empty(0, device=cost.device, dtype=torch.float64)
    else:
        scaling, iterations, residual = balance.symmetric_balance(log_kernel, tolerance=1e-8)
        value = torch.exp(log_kernel+scaling[:, None]+scaling[None, :])/n
    assert torch.isfinite(value).all() and (value >= 0).all() and torch.count_nonzero(value.diag()) == 0
    assert float((value.sum(1)*n-1).abs().max()) <= 1.01e-8
    if domain != 'RT':
        assert float((value.sum(0)*n-1).abs().max()) <= 1.01e-8
    return value, scaling, iterations, residual


def gap(cost, target_plan, current_plan):
    return (cost*(target_plan-current_plan)).sum()+GAMMA*(entropy(target_plan)-entropy(current_plan))


@torch.no_grad()
def targets(target, bins, seed):
    values = target.detach().double()
    cost = (values[:, None]-values[None, :]).abs()/(values.max()-values.min())
    balanced, scaling, iterations, residual = plan(cost, 'BT')
    row = plan(cost, 'RT')[0]
    generator = torch.Generator(device='cpu').manual_seed(seed+134000)
    permutation = torch.arange(len(values))
    for group in torch.unique(bins.cpu()):
        ids = torch.where(bins.cpu() == group)[0]
        assert len(ids) >= 2
        order = ids[torch.randperm(len(ids), generator=generator)]
        permutation[order] = order.roll(1)
    permutation = permutation.to(target.device)
    assert torch.equal(bins[permutation], bins) and (permutation != torch.arange(len(values), device=values.device)).all()
    return dict(cost=cost, BT=balanced, RT=row, PT=balanced[permutation][:, permutation], permutation=permutation,
                target_scaling=scaling, target_iterations=iterations, target_residual=residual)


class TransportSAGE(base.AngleSAGE):
    training_objective = 'native_SG_plus_continuous_signed_sphere_transport_Fenchel_Young_gap'

    def fit_relations(self, x, target, edges):
        history = previous.CurvatureSAGE.fit_relations(self, x, target, edges)
        if self.angle_arm != 'N':
            self.transport_targets = targets(target, self.label_bins, self.config['seed'])
        return history

    def forward(self, x, n_train, edges):
        value = super().forward(x, n_train, edges)
        if self.training and torch.is_grad_enabled():
            self.last_training_prediction = value[0]
        return value

    def training_loss(self, x, target, edges):
        native = sphere.SphericalSAGE.training_loss(self, x, target, edges)
        if self.angle_arm == 'N':
            return native
        row = self.geometry_history[-1]
        if row['first_update_MSE_only']:
            row.update(transport_gap=0., transport_KL=0., plan_iterations=0, plan_residual=0.,
                       target_plan_entropy=0., source_plan_entropy=0.)
            return native
        tick = time.perf_counter()
        cost = feature_cost(self.geometry_feature)
        source, _, iterations, residual = plan(cost, self.angle_arm)
        target_plan = self.transport_targets[self.angle_arm]
        extra = gap(cost, target_plan, source)
        positive = target_plan > 0
        kl = (target_plan[positive]*(target_plan[positive]/source[positive]).log()).sum()
        assert torch.isfinite(extra) and extra >= -1e-8
        self.last_transport = dict(cost=cost, source=source, target=target_plan, extra=extra)
        if row['step'] == 2:
            geometry, backbone, head = torch.autograd.grad(extra,
                (self.layer_log_metric, self.backbone.first.lin_l.weight, self.backbone.head.weight),
                retain_graph=True, allow_unused=True)
            self.transport_step2 = dict(extra_geometry_norms=geometry.detach().norm(dim=1).tolist(),
                extra_backbone_norm=float(backbone.detach().norm()), extra_head_unused_or_zero=head is None or bool((head == 0).all()))
        row.update(transport_gap=float(extra.detach()), transport_KL=float(kl), plan_iterations=iterations,
            plan_residual=residual, target_plan_entropy=float(entropy(target_plan)), source_plan_entropy=float(entropy(source)),
            transport_seconds=time.perf_counter()-tick)
        return native+extra


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train'] = int(data['n_train'])
    r.fix_seed(config['seed'], 1)
    return TransportSAGE(arm, config).to(device)


def source_hashes():
    hashes = prior.source_hashes()
    paths = [Path(__file__), Path(balance.__file__)]
    for name in ('RANK', 'RANK-DECOUPLE', 'BALANCE', 'MASS', 'BREGMAN', 'ANGLE-COMPENSATION', 'EDGE-CONFUSION', 'TAIL-SEPARABILITY-AUDIT'):
        paths.extend((r.HERE/'RESULTS').glob('CARBON-RELATION-'+name+'-001.attempt-*.md'))
    tail_root = r.HERE/'runs/CARBON-RELATION-TAIL-SEPARABILITY-AUDIT-001'
    expected = json.loads((tail_root/'artifact_hashes.json').read_text(encoding='utf-8'))
    for name in ('attempt-01/summary.json', 'attempt-01/all_1421_point_review.csv', 'history_followup.md'):
        path = tail_root/name
        assert r.digest(path) == expected[str(path.resolve())]
        paths.append(path)
    hashes.update({str(path.resolve()): r.digest(path) for path in paths})
    return hashes


def math_checks():
    generator = torch.Generator().manual_seed(134000)
    raw = torch.randn((12, 5), generator=generator, dtype=torch.float64)
    values = torch.arange(12, dtype=torch.float64).square()
    bins = torch.arange(6).repeat_interleave(2)
    rng = torch.get_rng_state().clone()
    target_data = targets(values, bins, 77)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.allclose(target_data['cost'], targets(3*values+7, bins, 77)['cost'], atol=1e-15, rtol=0)
    assert torch.allclose(feature_cost(raw*3), feature_cost(raw), atol=1e-14, rtol=0)
    errors, derivatives = [], []
    for arm in NEW:
        cost = feature_cost(raw).requires_grad_()
        source = plan(cost, arm)[0]
        target = target_data[arm]
        loss = gap(cost, target, source)
        derivative = torch.autograd.grad(loss, cost)[0]
        assert torch.equal(derivative, target-source) and loss >= -1e-8
        positive = target > 0
        errors.append(float(abs(loss-GAMMA*(target[positive]*(target[positive]/source[positive]).log()).sum()).detach()))
        assert errors[-1] < 1e-8
        direction = torch.randn((12, 12), generator=generator, dtype=torch.float64)
        direction = .5*(direction+direction.T); direction.fill_diagonal_(0); direction /= direction.norm()
        def objective(c): return gap(c, target, plan(c, arm)[0])
        epsilon = 1e-5
        finite = (objective(cost.detach()+epsilon*direction)-objective(cost.detach()-epsilon*direction))/(2*epsilon)
        derivatives.append(float(abs(finite-(derivative*direction).sum())))
        assert derivatives[-1] < 2e-6
        zero = gap(cost, source, source)
        assert zero == 0 and torch.count_nonzero(torch.autograd.grad(zero, cost)[0]) == 0
    assert abs(float(entropy(target_data['BT'])-entropy(target_data['PT']))) < 1e-12
    assert torch.allclose(torch.linalg.eigvalsh(target_data['BT']), torch.linalg.eigvalsh(target_data['PT']), atol=1e-12, rtol=0)
    uniform = torch.ones((12, 12), dtype=torch.float64)
    assert torch.allclose(plan(uniform, 'BT')[0], plan(uniform, 'RT')[0], atol=1e-12, rtol=0)
    return dict(FY_gradient_exact=True, zero_gap_exact=True, max_KL_error=max(errors), max_FD_error=max(derivatives),
                permutation_entropy_spectrum=True, row_and_balanced_uniform_exact=True, default_RNG_unchanged=True)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    hashes, checks, cases = source_hashes(), math_checks(), []
    tick = time.perf_counter()
    for region in r.REGIONS:
        _, config, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, target = r.tensors(data, pre, 'cuda')
        native = previous.construct('JA', config, 'cuda'); native.fit_reference(x[:n], te); native.fit_relations(x[:n], target, te)
        control = construct('N', config, 'cuda'); control.fit_reference(x[:n], te); control.fit_relations(x[:n], target, te)
        native_optimizer = sphere.Optimizer(native, config); control_optimizer = control.make_optimizer(config)
        initial, first_state = r.cpu_state(native), None
        native.eval()
        with torch.no_grad(): initial_prediction = native(x, n, ee)[0]
        for step in range(4):
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            for model, optimizer in ((native, native_optimizer), (control, control_optimizer)):
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
                model.train(); optimizer.zero_grad()
                loss = model.training_loss(x[:n], target, te); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optimizer.step()
            native_state, control_state = r.cpu_state(native), r.cpu_state(control)
            assert all(torch.equal(value, control_state[key]) for key, value in native_state.items())
            if step == 0: first_state = native_state
        for arm in NEW:
            model = construct(arm, config, 'cuda'); model.fit_reference(x[:n], te)
            before_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            model.fit_relations(x[:n], target, te)
            assert all(torch.equal(a, b) for a, b in zip(before_rng, (torch.get_rng_state(), torch.cuda.get_rng_state())))
            assert len(list(model.named_buffers())) == 16 and sum(p.numel() for p in model.parameters()) == 28275
            initial_state = r.cpu_state(model)
            assert all(torch.equal(value, initial_state[key]) for key, value in initial.items())
            model.eval()
            with torch.no_grad(): assert torch.equal(model(x, n, ee)[0], initial_prediction)
            optimizer = model.make_optimizer(config)
            for step in range(6):
                model.train(); optimizer.zero_grad(); loss = model.training_loss(x[:n], target, te)
                if step == 1:
                    evidence = model.transport_step2
                    assert min(evidence['extra_geometry_norms']) > 1e-12 and evidence['extra_backbone_norm'] > 1e-12
                    assert evidence['extra_head_unused_or_zero']
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                assert torch.isfinite(norm); optimizer.step()
                if step == 0:
                    updated = r.cpu_state(model)
                    assert all(torch.equal(value, updated[key]) for key, value in first_state.items())
            model.eval()
            with torch.no_grad():
                full = model(x, n, ee)[0]; alone = model(x[:n], n, te)[0]
                single = model(torch.cat((x[:n], x[n:n+1])), n, ee)[0][-1]
            assert torch.allclose(full[:n], alone, atol=3e-6, rtol=3e-6) and torch.allclose(full[n], single, atol=3e-6, rtol=3e-6)
            path = output/f'{region}_{arm}.pt'
            torch.save(dict(state_dict=r.cpu_state(model), config=model.config), path)
            restored = construct(arm, config, 'cuda').eval(); restored.load_state_dict(model.state_dict())
            with torch.no_grad(): assert torch.equal(restored(x, n, ee)[0], full)
            cases.append(dict(region=region, arm=arm, native_first_MSE_exact=True, six_updates=True,
                query_independent=True, restore_without_runtime_exact=True, **evidence))
            print(f'PREFLIGHT {region}/{arm} extra_geometry={evidence["extra_geometry_norms"]}', flush=True)
    assert all(r.digest(Path(path)) == expected for path, expected in hashes.items())
    r.write_json(output/'preflight.json', dict(source_sha256=r.digest(Path(__file__)), source_input_hashes=hashes,
        math=checks, cases=cases, native_SG4steps_all_provinces_exact=True, elapsed_seconds=time.perf_counter()-tick))


@torch.no_grad()
def export(model, data, pre, output, restore):
    model.eval(); n = int(data['n_train']); x, _, ee, target = r.tensors(data, pre, 'cuda')
    actual = model(x, n, ee)[0]; raw = model.geometry_feature[:n].clone(); graphs = model.layer_graphs(x, n)
    prediction = actual.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    with np.load(output/'responses.npz') as saved: assert np.array_equal(prediction, saved['prediction'])
    cost = feature_cost(raw); source, scaling, iterations, residual = plan(cost, model.angle_arm)
    target_plan = model.transport_targets[model.angle_arm]
    value = float(gap(cost, target_plan, source)); positive = target_plan > 0
    kl = float((target_plan[positive]*(target_plan[positive]/source[positive]).log()).sum())
    target_difference = float((model.transport_targets['BT']-model.transport_targets['PT']).abs().sum())
    row_difference = float((plan(cost, 'BT')[0]-plan(cost, 'RT')[0]).abs().sum())
    theta = model.layer_log_metric.clone(); model.layer_log_metric.zero_(); model._relation_cache.clear()
    bypass = model(x, n, ee)[0]; model.layer_log_metric.copy_(theta); model._relation_cache.clear()
    bypass_difference = float((actual-bypass).abs().max())
    with np.load(directory(None, model.config['region'], model.config['fold'], 'SG')/'responses.npz') as saved:
        sg_difference = float(abs(prediction-saved['prediction']).max())
    best = int(json.loads((output/'metrics.json').read_text(encoding='utf-8'))['best_epoch'])
    exposed = sum(row['transport_gap'] > 1e-12 for row in model.geometry_history[:best])
    step2 = getattr(model, 'transport_step2', dict(extra_geometry_norms=[0., 0.], extra_backbone_norm=0., extra_head_unused_or_zero=True))
    active = exposed > 0 and min(step2['extra_geometry_norms']) > 1e-12 and theta.norm() > 1e-10 and bypass_difference > 1e-6 and sg_difference > 1e-6
    if model.angle_arm == 'BT': active = active and target_difference >= .01 and row_difference >= .01
    diagnostic = dict(selected_epoch=best, selected_deployment_active=bool(active), selected_objective_exposed_steps=exposed,
        selected_transport_gap=value, selected_transport_KL=kl, selected_plan_iterations=iterations, selected_plan_residual=residual,
        target_permutation_L1_difference=target_difference, source_balanced_row_L1_difference=row_difference,
        zero_log_bypass_prediction_max_standardized_difference=bypass_difference, SG_prediction_max_absolute_difference=sg_difference,
        restore_checked=restore, **step2)
    if restore:
        restored = construct(model.angle_arm, model.config, 'cuda').eval(); restored.load_state_dict(model.state_dict())
        with torch.no_grad(): assert torch.equal(restored(x, n, ee)[0], actual)
        diagnostic['restore_exact'] = True
    np.savez_compressed(output/'transport_context.npz', sample_ids=data['sample_ids'][:n], target_standardized=target.cpu().numpy(),
        raw=raw.cpu().numpy(), cost=cost.cpu().numpy(), target_cost=model.transport_targets['cost'].cpu().numpy(),
        target_plan=target_plan.cpu().numpy(), source_plan=source.cpu().numpy(), source_scaling=scaling.cpu().numpy(),
        permutation=model.transport_targets['permutation'].cpu().numpy(), bins=model.label_bins.cpu().numpy(),
        target_balanced=model.transport_targets['BT'].cpu().numpy(), target_row=model.transport_targets['RT'].cpu().numpy(),
        target_scaling=model.transport_targets['target_scaling'].cpu().numpy(), gamma=GAMMA)
    graph_arrays = dict(sample_ids=data['sample_ids'])
    for index, graph in enumerate(graphs, 1):
        graph_arrays[f'layer{index}_neighbors'], graph_arrays[f'layer{index}_weights'] = graph[0].cpu().numpy(), graph[1].cpu().numpy()
        assert (graph[0] < n).all() and (graph[1] >= 0).all() and torch.isfinite(graph[2]).all()
    np.savez_compressed(output/'transport_graphs.npz', **graph_arrays)
    r.write_json(output/'transport_diagnostics.json', diagnostic)
    r.write_json(output/'transport_history.json', model.geometry_history)
    r.write_json(output/'transport_feedback.json', model.angle_feedback)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 9 and check['native_SG4steps_all_provinces_exact']
    assert check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(path)) == expected for path, expected in check['source_input_hashes'].items())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    hashes = source_hashes()
    for path in (output/'task_frozen.md', preflight_dir/'preflight.json'):
        hashes[str(path.resolve())] = r.digest(path)
    started, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', dict(task_id=TASK, branch='RELATION/v1.34', parent='RELATION/v1.33-audit',
        change_type='native_adaptation', started_at=started, source_input_hashes=hashes, new_fits=45, cached_fits=45,
        gamma=GAMMA, gap_coefficient=1., final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                export(holder[0], data, pre, output/region/f'fold-{fold:02d}'/arm, arm == 'BT' and fold == 1)
                holder[0].executed_optimizer.model = None; holder[0].executed_optimizer = None; holder.clear()
    fit_seconds = time.perf_counter()-tick
    analyze(output)
    assert all(r.digest(Path(path)) == expected for path, expected in hashes.items())
    r.write_json(output/'finished.json', dict(started_at=started, finished_at=datetime.now().astimezone().isoformat(),
        fit_export_seconds=fit_seconds, wall_seconds=time.perf_counter()-tick, protected_hashes_unchanged=True))


def analyze(output):
    folds, points, training = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, config, data, _ = r.load_fold(region, fold)
            n, y, predictions = int(data['n_train']), data['y'], {}
            for arm in (*CACHED, *NEW):
                path = directory(output, region, fold, arm); metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                expected_arm = CACHED[arm][1] if arm in CACHED else arm
                assert (metric['region'], metric['fold'], metric['arm'], metric['seed']) == (region, fold, expected_arm, config['seed'])
                with np.load(path/'responses.npz') as saved:
                    assert np.array_equal(saved['sample_ids'], data['sample_ids'])
                    prediction = saved['prediction'].copy(); predictions[arm] = prediction
                mt, mv = r.metrics(y[:n], prediction[:n]), r.metrics(y[n:], prediction[n:])
                assert all(abs(values[key]-metric[role][key]) < 1e-7 for role, values in [('train', mt), ('validation', mv)] for key in values)
                row = dict(region=region, fold=fold, arm=arm, **mv, train_rmse=mt['rmse'], best_epoch=metric['best_epoch'],
                           epochs_run=metric['epochs_run'], elapsed_seconds=metric['elapsed_seconds'])
                if arm in NEW:
                    diagnostic = json.loads((path/'transport_diagnostics.json').read_text(encoding='utf-8'))
                    row.update(selected_deployment_active=diagnostic['selected_deployment_active'],
                               selected_objective_exposed_steps=diagnostic['selected_objective_exposed_steps'])
                folds.append(row)
            for i, identity in enumerate(data['sample_ids']):
                row = dict(sample_id=str(identity), region=region, fold=fold, observed_agb=float(y[i]),
                           high_agb=int(y[i] >= np.quantile(y[:n], .8)), missing_features_current=int(data['missing_mask'][i].sum()))
                for arm, prediction in predictions.items():
                    row[arm+'_prediction'], row[arm+'_squared_error'] = float(prediction[i]), float((prediction[i]-y[i])**2)
                (training if i < n else points).append(row)
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    for path, keys in [(context.PAIN/'all_1421_ids.csv', ['sample_id', 'region', 'fold', 'observed_agb']),
                       (context.NAMED, ['sample_id', 'region', 'fold']), (context.TAIL, ['sample_id', 'region'])]:
        p = p.merge(pd.read_csv(path, float_precision='round_trip'), on=keys, how='left', validate='one_to_one', suffixes=('', '_archive'))
    p = p.join(t.groupby('sample_id').agg(BT_train_MSE=('BT_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean')), on='sample_id', validate='one_to_one')
    assert p.feature1.notna().all() and p.BT_train_MSE.notna().all()
    p['BT_minus_SG_SSE'] = p.BT_squared_error-p.SG_squared_error
    p['training_BT_minus_SG_MSE'] = p.BT_train_MSE-p.SG_train_MSE
    p['failure_phase'] = np.select([(p.BT_minus_SG_SSE > 0)&(p.training_BT_minus_SG_MSE > 0),
        (p.BT_minus_SG_SSE > 0)&(p.training_BT_minus_SG_MSE <= 0), p.BT_minus_SG_SSE < 0],
        ['training_and_OOF_worse', 'training_better_or_tied_OOF_worse', 'OOF_better'], default='OOF_tied')
    strata, features = [], []
    for region, group in p.groupby('region'):
        masks = dict(all=np.ones(len(group), dtype=bool), high=group.high_agb == 1, ordinary=group.high_agb == 0,
            missing=group.missing_features_current > 0, complete=group.missing_features_current == 0,
            complete_typical=(group.missing_features_current == 0)&(group.max_abs_training_standardized_feature <= 3), persistent_tail=group.tail_appearances >= 3)
        for label, mask in masks.items():
            selected = group[mask]
            if len(selected): strata.append(dict(region=region, stratum=label, n=len(selected),
                BT_minus_SG_SSE=float(selected.BT_minus_SG_SSE.sum()), BT_bias=float((selected.BT_prediction-selected.observed_agb).mean()),
                SG_bias=float((selected.SG_prediction-selected.observed_agb).mean())))
        raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold); n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]): data['raw_features'][i] for i in range(n, len(data['y']))})
        matrix = np.stack([raw[identity] for identity in group.sample_id]); changes = group.BT_minus_SG_SSE.to_numpy()
        for index, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, index]; valid = np.isfinite(values)&(values != -9999)
            coefficient = pd.Series(values[valid]).corr(pd.Series(changes[valid]), method='spearman')
            features.append(dict(region=region, feature=name, valid_n=int(valid.sum()),
                spearman_feature_vs_BT_SSE_change=float(coefficient) if np.isfinite(coefficient) else None,
                missing_SSE_change=float(changes[~valid].sum()), observed_SSE_change=float(changes[valid].sum())))
    assert len(features) == 555
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False); t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False); pd.DataFrame(features).to_csv(output/'all_555_feature_contexts.csv', index=False)
    pd.concat((p.nlargest(20, 'BT_minus_SG_SSE'), p.nsmallest(20, 'BT_minus_SG_SSE'))).to_csv(output/'forty_cases.csv', index=False)
    wide = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    improvements = {arm: float((100*(1-wide[arm]/wide.G0)).mean()) for arm in (*CACHED, *NEW) if arm != 'G0'}
    compared = {arm: float((100*(1-wide.BT/wide[arm])).mean()) for arm in ('RT', 'PT', 'SG', 'GR')}
    province_gate, province_cap = True, True
    for region in r.REGIONS:
        baseline, candidate = [f[(f.region == region)&(f.arm == arm)][['rmse', 'r2']].mean() for arm in ('G0', 'BT')]
        province_gate &= candidate.rmse <= baseline.rmse+1e-6 and candidate.r2 >= baseline.r2-1e-6
        province_cap &= candidate.rmse <= 1.01*baseline.rmse and candidate.r2 >= baseline.r2-.01
    activity = {region: int(f[(f.region == region)&(f.arm == 'BT')].selected_deployment_active.sum()) for region in r.REGIONS}
    deployed = sum(activity.values()) >= 9 and min(activity.values()) >= 2
    full = improvements['BT'] >= 10 and min(compared['RT'], compared['PT']) >= .5 and min(compared['SG'], compared['GR']) > 0 and province_gate and deployed
    pareto = improvements['BT'] >= 3 and compared['SG'] >= .5 and min(compared.values()) > 0 and province_cap and deployed
    summary = dict(task_id=TASK, G0_RMSE_gains_percent=improvements, BT_control_gains_percent=compared,
        province_G0_noninferiority=bool(province_gate), province_cap=bool(province_cap), selected_BT_activity=activity,
        deployment_gate=deployed, FULL_GAIN=bool(full), PARETO_GAIN=bool(pareto),
        classification='FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        all_development_IDs=1421, training_appearances_per_method=5684, feature_contexts=555,
        BT_helped_SG=int((p.BT_minus_SG_SSE < 0).sum()), BT_harmed_SG=int((p.BT_minus_SG_SSE > 0).sum()),
        BT_minus_SG_total_SSE=float(p.BT_minus_SG_SSE.sum()), failure_phase_counts=p.failure_phase.value_counts().to_dict(), final_holdout_access=False)
    r.write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('preflight', 'run'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight': preflight(args.output)
    else: run(args.output, args.preflight_dir)
