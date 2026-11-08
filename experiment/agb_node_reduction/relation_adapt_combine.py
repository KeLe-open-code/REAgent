"""Private native SG predictors with adapt-then-combine log-SPD relations."""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import relation_backbone_curvature as previous
import relation_transport_gap as reporting

r, sphere, layer = previous.r, previous.sphere, previous.layer
context, CACHED = reporting.context, previous.CACHED
TASK, NEW = 'CARBON-RELATION-ADAPT-COMBINE-001', ('RD', 'TG', 'IE')
OFFSETS = (0, 1009, 2018)


class Group(torch.nn.Module):
    def __init__(self, arm, config, device):
        super().__init__()
        self.variant, self.config = arm, dict(config)
        members = []
        for offset in OFFSETS:
            cfg = {**config, 'seed': config['seed']+offset}
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg['seed'])
                model = previous.CurvatureSAGE('JA', cfg)
            members.append(model.to(device))
        self.members = torch.nn.ModuleList(members)
        if arm == 'TG':
            for member in self.members[1:]:
                member.layer_log_metric = self.members[0].layer_log_metric
        body_ids = [{id(p) for p in m.backbone.parameters()} for m in self.members]
        assert all(body_ids[i].isdisjoint(body_ids[j]) for i in range(3) for j in range(i))
        geometry_ids = {id(m.layer_log_metric) for m in self.members}
        assert len(geometry_ids) == (1 if arm == 'TG' else 3)
        assert sum(p.numel() for p in self.parameters()) == (84085 if arm == 'TG' else 84825)

    def fit_reference(self, x, edges, target):
        for member in self.members:
            member.fit_reference(x, edges)
            member.fit_relations(x, target, edges)
            assert len(list(member.named_buffers())) == 16
        first = dict(self.members[0].named_buffers())
        assert all(torch.equal(value, first[name]) for member in self.members for name, value in member.named_buffers())

    def clear_caches(self):
        for member in self.members:
            member._relation_cache.clear()

    def load_state_dict(self, *args, **kwargs):
        self.clear_caches()
        return super().load_state_dict(*args, **kwargs)

    def outputs(self, x, n, edges):
        return [member(x, n, edges) for member in self.members]

    def forward(self, x, n, edges):
        values = self.outputs(x, n, edges)
        return (torch.stack([v[0] for v in values]).mean(0),
                torch.stack([v[1] for v in values]).mean(0), *values[0][2:])


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as z:
            config['relation_n_train'] = int(z['n_train'])
    r.fix_seed(config['seed'], 1)
    return Group(arm, config, device)


class Optimizer:
    """Keep each native loss and clipping scale; cooperate only in geometry."""
    def __init__(self, model, config):
        self.model, self.config, self.steps = model, config, 0
        self.bodies = [torch.optim.Adam(m.backbone.parameters(), lr=config['learning_rate'],
                                       weight_decay=config['weight_decay']) for m in model.members]
        selected = model.members[:1] if model.variant == 'TG' else model.members
        self.geometry = [layer.RDoG(m.layer_log_metric) for m in selected]
        self.rng = [torch.Generator(device='cuda').manual_seed(config['seed']+offset).get_state()
                    for offset in OFFSETS]
        parameter = model.members[0].layer_log_metric
        self.shadow_radius, self.shadow_accumulator = parameter.new_full((2,), .01), parameter.new_zeros(2)
        self.history, self.trace = [], []

    def advance(self, x, target, edges):
        model = self.model
        model.train()
        for body in self.bodies:
            body.zero_grad(set_to_none=True)
        losses, raw_norms, clipped_norms, gradients = [], [], [], []
        ambient = torch.cuda.get_rng_state().clone()
        before = torch.stack([m.layer_log_metric.detach().clone() for m in model.members])
        try:
            for index, member in enumerate(model.members):
                # The TG parameter is aliased; clear it before every private backward.
                member.layer_log_metric.grad = None
                torch.cuda.set_rng_state(self.rng[index])
                loss = member.training_loss(x, target, edges)
                assert torch.isfinite(loss)
                loss.backward()
                self.rng[index] = torch.cuda.get_rng_state().clone()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in member.parameters())
                norm = torch.nn.utils.clip_grad_norm_(member.parameters(), self.config['gradient_clip'])
                assert torch.isfinite(norm)
                raw_norms.append(float(norm))
                clipped_norms.append(float(torch.stack([p.grad.detach().norm() for p in member.parameters()]).norm()))
                gradients.append(member.layer_log_metric.grad.detach().clone())
                losses.append(float(loss.detach()))
        finally:
            torch.cuda.set_rng_state(ambient)
        g = torch.stack(gradients)
        for body in self.bodies:
            body.step()
        with torch.no_grad():
            if model.variant == 'TG':
                model.members[0].layer_log_metric.grad = g.mean(0)
                self.geometry[0].step()
            else:
                for member, optimizer, gradient in zip(model.members, self.geometry, gradients):
                    member.layer_log_metric.grad = gradient
                    optimizer.step()
            candidates = torch.stack([m.layer_log_metric.detach().clone() for m in model.members])
            shadow = candidates[0].clone()
            if model.variant == 'RD':
                assert torch.equal(before[0], before[1]) and torch.equal(before[0], before[2])
                mean_g = g.mean(0)
                self.shadow_radius = torch.maximum(self.shadow_radius, before[0].norm(dim=1))
                self.shadow_accumulator += mean_g.square().sum(1)
                rate = torch.where(self.shadow_accumulator > 0,
                                   self.shadow_radius/self.shadow_accumulator.sqrt(), 0.)
                # This uses the actual RD current state, never the separate TG trajectory.
                shadow = before[0]-rate[:, None]*mean_g
                consensus = candidates.mean(0)
                for member in model.members:
                    member.layer_log_metric.copy_(consensus)
            after = torch.stack([m.layer_log_metric.detach().clone() for m in model.members])
            disagreement = (candidates-candidates.mean(0)).square().sum(2).mean(0).sqrt()
            difference = (after[0]-shadow).norm(dim=1) if model.variant == 'RD' else before.new_zeros(2)
            record = dict(step=self.steps+1, loss_mean=float(np.mean(losses)), member_losses=losses,
                member_MSE=[m.geometry_history[-1]['mse'] for m in model.members],
                per_member_raw_gradient_norm=raw_norms, per_member_clipped_gradient_norm=clipped_norms,
                clipped_geometry_gradient_norms=g.norm(dim=2).tolist(),
                local_candidate_disagreement=disagreement.tolist(),
                postcombine_max_member_distance=float((after-after[0]).abs().max()),
                same_state_mean_gradient_update_difference=difference.tolist(),
                geometry_radius=[o.radius.tolist() for o in self.geometry],
                geometry_accumulator=[o.accumulator.tolist() for o in self.geometry])
            arrays = dict(theta_before=before, clipped_geometry_gradients=g, local_candidates=candidates,
                          theta_after=after, same_state_TG_candidate=shadow,
                          radius=torch.stack([o.radius for o in self.geometry]),
                          accumulator=torch.stack([o.accumulator for o in self.geometry]))
            self.trace.append({k: v.cpu().numpy().copy() for k, v in arrays.items()})
        self.steps += 1
        self.history.append(record)
        model.clear_caches()
        return record


def directory(output, region, fold, arm):
    root, name = CACHED[arm] if arm in CACHED else (output, arm)
    return root/region/f'fold-{fold:02d}'/name


def source_hashes():
    hashes = reporting.source_hashes()
    names = ('MIXTURE-FIELD', 'TAIL-MAGNITUDE-AUDIT', 'MULTICHART', 'EFFICIENT-ENSEMBLE',
             'HETEROGENEOUS-JOINT', 'AVERAGING', 'SEED-VARIANCE-AUDIT', 'DIFFUSION')
    paths = [Path(__file__)]
    for name in names:
        paths.extend((r.HERE/'RESULTS').glob('CARBON-RELATION-'+name+'-001.attempt-*.md'))
    for region in r.REGIONS:
        for fold in range(1, 6):
            source, _, _, _ = r.load_fold(region, fold)
            paths.extend(source/name for name in ('config.json', 'data/development.npz',
                                                'data/preprocessor.npz', 'data/split.csv'))
            for arm in CACHED:
                root = directory(None, region, fold, arm)
                paths.extend(root/name for name in ('metrics.json', 'responses.npz', 'best.pt'))
    paths.extend([context.PAIN/'all_1421_ids.csv', context.PAIN/'feature_diagnostics.csv', context.NAMED, context.TAIL])
    hashes.update({str(path.resolve()): r.digest(path) for path in paths})
    return hashes


def math_checks():
    theta = torch.tensor([[[.2, -.4, .1], [-.3, .1, .2]],
                          [[-.2, .2, .5], [.4, .2, -.1]],
                          [[.1, .3, -.2], [.2, -.4, .3]]], dtype=torch.float64)
    diagonal_mean = theta.mean(0).exp()
    literal = theta.exp().prod(0).pow(1/3)
    assert torch.allclose(diagonal_mean, literal, atol=1e-14, rtol=0)
    gradients = torch.tensor([[[1., 2., 3.], [-1., 2., 0.]],
                              [[3., 0., 0.], [.2, -.1, 3.]],
                              [[-.2, 1., .3], [3., 1., 2.]]], dtype=torch.float64)
    candidates = -.01*gradients/gradients.norm(dim=2, keepdim=True)
    mean = gradients.mean(0)
    merged_first = -.01*mean/mean.norm(dim=1, keepdim=True)
    assert (candidates.mean(0)-merged_first).norm() > 1e-5
    same = gradients[:1].repeat(3, 1, 1)
    assert torch.allclose((-.01*same/same.norm(dim=2, keepdim=True)).mean(0),
                          -.01*same[0]/same[0].norm(dim=1, keepdim=True), atol=1e-14, rtol=0)
    p = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float64))
    opt = layer.RDoG(p)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.count_nonzero(p) == 0 and torch.isfinite(p).all()
    return dict(diagonal_SPD_geometric_mean_literal=True, adapt_combine_order_distinct=True,
                equal_gradient_identity=True, zero_gradient_stable=True)


@torch.no_grad()
def export(model, optimizer, data, pre, output, best_epoch, restore=False):
    model.eval()
    n = int(data['n_train'])
    x, te, ee, _ = r.tensors(data, pre, 'cuda')
    values = model.outputs(x, n, ee)
    predictions = torch.stack([value[0] for value in values])
    deployed = predictions.mean(0)
    embeddings = torch.stack([value[1] for value in values])
    graphs = [member.layer_graphs(x, n) for member in model.members]
    neighbors = torch.stack([torch.stack([g[0] for g in pair]) for pair in graphs])
    weights = torch.stack([torch.stack([g[1] for g in pair]) for pair in graphs])
    diagonal = torch.stack([torch.stack([g[2] for g in pair]) for pair in graphs])
    assert neighbors.shape == (3, 2, len(x), 8) and int(neighbors.max()) < n
    assert torch.isfinite(diagonal).all() and (diagonal > 0).all()
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6, rtol=0)
    alone = model(x[:n], n, te)[0]
    single = model(torch.cat((x[:n], x[n:n+1])), n, ee)[0]
    train_effect, query_effect = float((deployed[:n]-alone).abs().max()), float((deployed[n]-single[n]).abs())
    assert torch.allclose(deployed[:n], alone, atol=3e-6, rtol=3e-6)
    assert torch.allclose(deployed[n], single[n], atol=3e-6, rtol=3e-6)
    original = torch.load(model.members[0].reference, map_location='cpu', weights_only=False)['state_dict']
    assert all(torch.equal(value.cpu(), original[name]) for member in model.members for name, value in member.named_buffers())
    theta = torch.stack([m.layer_log_metric.detach().clone() for m in model.members])
    try:
        for member in model.members:
            member.layer_log_metric.zero_()
        model.clear_caches()
        counter = model(x, n, ee)[0]
    finally:
        for member, value in zip(model.members, theta):
            member.layer_log_metric.copy_(value)
        model.clear_caches()
    effect = float((deployed[n:]-counter[n:]).abs().max())
    exposed = [row for row in optimizer.history[:best_epoch]
        if min(v for member in row['clipped_geometry_gradient_norms'] for v in member) > 0
        and min(row['local_candidate_disagreement']) > 1e-10
        and row['postcombine_max_member_distance'] == 0
        and min(row['same_state_mean_gradient_update_difference']) > 1e-10]
    active = bool(model.variant == 'RD' and exposed and theta.norm(dim=2).min() > 0 and effect > 1e-6)
    if restore:
        loaded = construct(model.variant, model.config, 'cuda').eval()
        loaded.load_state_dict(r.cpu_state(model))
        assert torch.equal(deployed, loaded(x, n, ee)[0])
        del loaded
    raw = deployed.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    member_raw = predictions.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    near0, weight0 = neighbors[0, 0].cpu().numpy(), weights[0, 0].cpu().numpy()
    edges = np.stack((near0.ravel(), np.repeat(np.arange(len(x)), 8)))
    np.savez_compressed(output/'responses.npz', sample_ids=data['sample_ids'], prediction=raw,
        absolute_residual=np.abs(raw-data['y']), embedding=embeddings.mean(0).cpu().numpy(),
        neighbors=near0, weights=weight0, local_metric=diagonal[0, 0].cpu().numpy(),
        train_edges=edges[:, edges[1] < n], eval_edges=edges)
    np.savez_compressed(output/'member_responses.npz', sample_ids=data['sample_ids'], n_train=n,
        predictions=member_raw, prediction_standardized=predictions.cpu().numpy(),
        embeddings=embeddings.cpu().numpy(), neighbors=neighbors.cpu().numpy(), weights=weights.cpu().numpy(),
        layer_log_metric=theta.cpu().numpy(), zero_geometry_prediction_standardized=counter.cpu().numpy())
    # These full arrays permit CPU reconstruction of every native RDoG update.
    np.savez_compressed(output/'geometry_optimizer_trace.npz',
        **{key: np.stack([row[key] for row in optimizer.trace]) for key in optimizer.trace[0]})
    r.write_json(output/'geometry_optimizer_history.json', optimizer.history)
    r.write_json(output/'spherical_loss_histories.json', [m.geometry_history for m in model.members])
    jensen = {}
    for label, take in [('train', slice(0, n)), ('validation', slice(n, None))]:
        observed, array = data['y'][take], member_raw[:, take]
        mse_members = float(np.mean((array-observed)**2))
        mse_mean = float(np.mean((array.mean(0)-observed)**2))
        variance = float(np.mean((array-array.mean(0))**2))
        assert abs(mse_members-mse_mean-variance) < 1e-8
        jensen[label] = dict(mean_member_MSE=mse_members, mean_prediction_MSE_float64=mse_mean,
                            member_prediction_variance=variance,
                            deployed_float32_mean_MSE=float(np.mean((raw[take]-observed)**2)))
    diagnostic = dict(variant=model.variant, parameters=sum(p.numel() for p in model.parameters()),
        effective_parameters=84085 if model.variant in ('RD', 'TG') else 84825,
        member_initialization_dropout_seeds=[model.config['seed']+v for v in OFFSETS],
        member_fitted_buffer_source='same_original_root_seed_SG', fixed_prediction_weights=[1/3]*3,
        selected_deployment_active=active, selected_adaptive_combination_steps=len(exposed),
        selected_query_geometry_effect_max_standardized=effect, selected_geometry_norms=theta.norm(dim=2).tolist(),
        max_train_query_addition_difference=train_effect, single_query_difference=query_effect,
        exact_restore=restore, optimizer_group_steps=optimizer.steps, jensen=jensen,
        response_graph_scope='member0_layer0_only;all270graphs_in_member_responses',
        checkpoint_scope='inference_weights_only_not_training_resume')
    r.write_json(output/'combination_diagnostics.json', diagnostic)
    return raw, diagnostic


def fit(arm, region, fold, output, budget=400, restore=False):
    source, cfg, data, pre = r.load_fold(region, fold)
    path = directory(output, region, fold, arm)
    assert not path.exists()
    path.mkdir(parents=True)
    started = time.perf_counter()
    model = construct(arm, cfg, 'cuda')
    n = int(data['n_train'])
    x, te, ee, target = r.tensors(data, pre, 'cuda')
    model.fit_reference(x[:n], te, target)
    optimizer = Optimizer(model, cfg)
    fingerprint = hashlib.sha256()
    for key, value in r.cpu_state(model).items():
        fingerprint.update(key.encode())
        fingerprint.update(value.numpy().tobytes())
    model.eval()
    with torch.no_grad():
        initial_predictions = torch.stack([v[0] for v in model.outputs(x, n, ee)]).cpu().numpy()
    np.savez_compressed(path/'initial_predictions.npz', predictions_standardized=initial_predictions)
    best, stale, best_epoch, best_state = float('inf'), 0, 0, None
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, budget+1):
        update = optimizer.advance(x[:n], target, te)
        model.eval()
        with torch.no_grad():
            scaled = model(x, n, ee)[0].cpu().numpy().astype(float)
        prediction = scaled*pre['y_scale']+pre['y_mean']
        mt, mv = r.metrics(data['y'][:n], prediction[:n]), r.metrics(data['y'][n:], prediction[n:])
        history.append(dict(epoch=epoch, train_loss_standardized=update['loss_mean'],
            native_member_MSE_standardized=float(np.mean(update['member_MSE'])),
            train_rmse=mt['rmse'], validation_rmse=mv['rmse'], validation_r2=mv['r2'],
            maximum_member_raw_gradient_norm=max(update['per_member_raw_gradient_norm'])))
        if mv['rmse'] < best-cfg['min_delta']:
            best, stale, best_epoch, best_state = mv['rmse'], 0, epoch, r.cpu_state(model)
        else:
            stale += 1
        if epoch % 50 == 0:
            r.write_csv(path/'history.csv', history)
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    raw, diagnostic = export(model, optimizer, data, pre, path, best_epoch, restore)
    torch.save(dict(task_id=TASK, arm=arm, state_dict=best_state, config=model.config, epoch=best_epoch,
        device='cuda', max_gnn_epochs=budget, member_seed_offsets=OFFSETS,
        split_sha256=r.digest(source/'data/split.csv'), source_sha256=r.digest(Path(__file__))), path/'best.pt')
    r.write_csv(path/'history.csv', history)
    result = dict(task_id=TASK, arm=arm, region=region, fold=fold, seed=cfg['seed'], device='cuda',
        train=r.metrics(data['y'][:n], raw[:n]), validation=r.metrics(data['y'][n:], raw[n:]),
        best_epoch=best_epoch, epochs_run=epoch, total_updates=epoch, member_updates=3*epoch,
        parameters=diagnostic['parameters'], effective_parameters=diagnostic['effective_parameters'],
        max_gnn_epochs=budget, patience=cfg['patience'], training_objective='three_independent_native_SG_losses',
        gradient_clipping='per_member_5_before_geometry_combination',
        selected_deployment_active=diagnostic['selected_deployment_active'],
        initial_state_sha256=fingerprint.hexdigest(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.perf_counter()-started)
    r.write_json(path/'metrics.json', result)
    print('FIT', region, fold, arm, 'epoch', best_epoch, '/', epoch,
          'validation_RMSE', result['validation']['rmse'], 'seconds', round(result['elapsed_seconds'], 3), flush=True)
    return result


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    started, cases, native_checks = time.perf_counter(), [], []
    checks = math_checks()
    for region in r.REGIONS:
        _, cfg, data, pre = r.load_fold(region, 1)
        n = int(data['n_train'])
        x, te, _, target = r.tensors(data, pre, 'cuda')
        model = construct('IE', cfg, 'cuda')
        model.fit_reference(x[:n], te, target)
        optimizer = Optimizer(model, cfg)
        native = previous.construct('JA', cfg, 'cuda')
        native.fit_reference(x[:n], te)
        native.fit_relations(x[:n], target, te)
        native_optimizer = sphere.Optimizer(native, cfg)
        assert all(torch.equal(value, native.state_dict()[key]) for key, value in model.members[0].state_dict().items())
        for _ in range(4):
            torch.cuda.set_rng_state(optimizer.rng[0])
            native.train()
            native_optimizer.zero_grad(set_to_none=True)
            loss = native.training_loss(x[:n], target, te)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(native.parameters(), cfg['gradient_clip'])
            native_optimizer.step()
            record = optimizer.advance(x[:n], target, te)
            assert record['member_losses'][0] == float(loss.detach())
            assert all(torch.equal(value, native.state_dict()[key]) for key, value in model.members[0].state_dict().items())
            assert torch.equal(optimizer.geometry[0].radius, native_optimizer.geometry.radius)
            assert torch.equal(optimizer.geometry[0].accumulator, native_optimizer.geometry.accumulator)
        native_checks.append(dict(region=region, member0_native_SG_four_updates_bitexact=True))
        del optimizer, model, native, native_optimizer
        common_hash, common_prediction = None, None
        for arm in NEW:
            result = fit(arm, region, 1, output, budget=4, restore=True)
            path = directory(output, region, 1, arm)
            with np.load(path/'initial_predictions.npz') as z:
                initial = z['predictions_standardized'].copy()
            if common_hash is None:
                common_hash, common_prediction = result['initial_state_sha256'], initial
            assert common_hash == result['initial_state_sha256'] and np.array_equal(common_prediction, initial)
            if arm == 'RD':
                trace = json.loads((path/'geometry_optimizer_history.json').read_text(encoding='utf-8'))
                assert max(max(v['local_candidate_disagreement']) for v in trace) > 1e-10
                assert max(max(v['same_state_mean_gradient_update_difference']) for v in trace) > 1e-10
                assert all(v['postcombine_max_member_distance'] == 0 for v in trace)
            cases.append(dict(region=region, arm=arm, four_group_updates=True,
                initial_state_and_prediction_paired=True, query_independence=True, restore_exact=True,
                parameters=result['parameters'], peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes']))
    r.write_json(output/'preflight.json', dict(task_id=TASK, cases=cases, mathematics=checks,
        native_checks=native_checks, native_SG4steps_all_provinces_exact=True,
        wall_seconds=time.perf_counter()-started, source_sha256=r.digest(Path(__file__)),
        source_input_hashes=source_hashes(), final_holdout_access=False))
    print('PREFLIGHT_OK', len(cases), 'groups', flush=True)


def analyze(output):
    folds, points, training = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, cfg, data, _ = r.load_fold(region, fold)
            n, y, predictions = int(data['n_train']), data['y'], {}
            for arm in (*CACHED, *NEW):
                path = directory(output, region, fold, arm)
                metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                expected = CACHED[arm][1] if arm in CACHED else arm
                assert (metric['region'], metric['fold'], metric['arm'], metric['seed']) == (region, fold, expected, cfg['seed'])
                with np.load(path/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    prediction = z['prediction'].copy()
                predictions[arm] = prediction
                mt, mv = r.metrics(y[:n], prediction[:n]), r.metrics(y[n:], prediction[n:])
                assert all(abs(value[key]-metric[role][key]) < 1e-7
                    for role, value in [('train', mt), ('validation', mv)] for key in value)
                row = dict(region=region, fold=fold, arm=arm, **mv, train_rmse=mt['rmse'],
                    best_epoch=metric['best_epoch'], epochs_run=metric['epochs_run'], elapsed_seconds=metric['elapsed_seconds'])
                if arm in NEW:
                    d = json.loads((path/'combination_diagnostics.json').read_text(encoding='utf-8'))
                    row.update(selected_deployment_active=d['selected_deployment_active'],
                        selected_adaptive_combination_steps=d['selected_adaptive_combination_steps'],
                        member_prediction_variance=d['jensen']['validation']['member_prediction_variance'])
                folds.append(row)
            for i, identity in enumerate(data['sample_ids']):
                row = dict(sample_id=str(identity), region=region, fold=fold, observed_agb=float(y[i]),
                    high_agb=int(y[i] >= np.quantile(y[:n], .8)), missing_features_current=int(data['missing_mask'][i].sum()))
                for arm, prediction in predictions.items():
                    row[arm+'_prediction'] = float(prediction[i])
                    row[arm+'_squared_error'] = float((prediction[i]-y[i])**2)
                (training if i < n else points).append(row)
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    for path, keys in [(context.PAIN/'all_1421_ids.csv', ['sample_id', 'region', 'fold', 'observed_agb']),
                       (context.NAMED, ['sample_id', 'region', 'fold']), (context.TAIL, ['sample_id', 'region'])]:
        p = p.merge(pd.read_csv(path, float_precision='round_trip'), on=keys, how='left',
                    validate='one_to_one', suffixes=('', '_archive'))
    p = p.join(t.groupby('sample_id').agg(RD_train_MSE=('RD_squared_error', 'mean'),
                                        SG_train_MSE=('SG_squared_error', 'mean')), on='sample_id', validate='one_to_one')
    assert p.feature1.notna().all() and p.RD_train_MSE.notna().all()
    p['RD_minus_SG_SSE'] = p.RD_squared_error-p.SG_squared_error
    p['training_RD_minus_SG_MSE'] = p.RD_train_MSE-p.SG_train_MSE
    p['failure_phase'] = np.select([(p.RD_minus_SG_SSE > 0)&(p.training_RD_minus_SG_MSE > 0),
        (p.RD_minus_SG_SSE > 0)&(p.training_RD_minus_SG_MSE <= 0), p.RD_minus_SG_SSE < 0],
        ['training_and_OOF_worse', 'training_better_or_tied_OOF_worse', 'OOF_better'], default='OOF_tied')
    strata, features = [], []
    for region, group in p.groupby('region'):
        masks = dict(all=np.ones(len(group), dtype=bool), high=group.high_agb == 1, ordinary=group.high_agb == 0,
            missing=group.missing_features_current > 0, complete=group.missing_features_current == 0,
            complete_typical=(group.missing_features_current == 0)&(group.max_abs_training_standardized_feature <= 3),
            persistent_tail=group.tail_appearances >= 3)
        for label, mask in masks.items():
            selected = group[mask]
            if len(selected):
                strata.append(dict(region=region, stratum=label, n=len(selected),
                    RD_minus_SG_SSE=float(selected.RD_minus_SG_SSE.sum()),
                    RD_bias=float((selected.RD_prediction-selected.observed_agb).mean()),
                    SG_bias=float((selected.SG_prediction-selected.observed_agb).mean())))
        raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold)
            n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]): data['raw_features'][i] for i in range(n, len(data['y']))})
        matrix = np.stack([raw[identity] for identity in group.sample_id])
        changes = group.RD_minus_SG_SSE.to_numpy()
        for index, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, index]
            valid = np.isfinite(values)&(values != -9999)
            coefficient = pd.Series(values[valid]).corr(pd.Series(changes[valid]), method='spearman')
            features.append(dict(region=region, feature=name, valid_n=int(valid.sum()),
                spearman_feature_vs_RD_SSE_change=float(coefficient) if np.isfinite(coefficient) else None,
                missing_SSE_change=float(changes[~valid].sum()), observed_SSE_change=float(changes[valid].sum())))
    assert len(features) == 555
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False)
    t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False)
    pd.DataFrame(features).to_csv(output/'all_555_feature_contexts.csv', index=False)
    pd.concat((p.nlargest(20, 'RD_minus_SG_SSE'), p.nsmallest(20, 'RD_minus_SG_SSE'))).to_csv(output/'forty_cases.csv', index=False)
    wide = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    improvements = {arm: float((100*(1-wide[arm]/wide.G0)).mean()) for arm in (*CACHED, *NEW) if arm != 'G0'}
    compared = {arm: float((100*(1-wide.RD/wide[arm])).mean()) for arm in ('TG', 'IE', 'SG', 'GR')}
    province_gate, province_cap = True, True
    for region in r.REGIONS:
        baseline, candidate = [f[(f.region == region)&(f.arm == arm)][['rmse', 'r2']].mean() for arm in ('G0', 'RD')]
        province_gate &= candidate.rmse <= baseline.rmse+1e-6 and candidate.r2 >= baseline.r2-1e-6
        province_cap &= candidate.rmse <= 1.01*baseline.rmse and candidate.r2 >= baseline.r2-.01
    activity = {region: int(f[(f.region == region)&(f.arm == 'RD')].selected_deployment_active.sum()) for region in r.REGIONS}
    deployed = sum(activity.values()) >= 9 and min(activity.values()) >= 2
    full = improvements['RD'] >= 10 and min(compared['TG'], compared['IE']) >= .5 and min(compared['SG'], compared['GR']) > 0 and province_gate and deployed
    pareto = improvements['RD'] >= 3 and compared['SG'] >= .5 and min(compared.values()) > 0 and province_cap and deployed
    summary = dict(task_id=TASK, G0_RMSE_gains_percent=improvements, RD_control_gains_percent=compared,
        province_G0_noninferiority=bool(province_gate), province_cap=bool(province_cap), selected_RD_activity=activity,
        deployment_gate=deployed, FULL_GAIN=bool(full), PARETO_GAIN=bool(pareto),
        classification='FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        new_group_fits=45, member_trajectories=135, cached_fits=45, method_folds=90,
        all_development_IDs=1421, training_appearances_per_method=5684, feature_contexts=555,
        RD_helped_SG=int((p.RD_minus_SG_SSE < 0).sum()), RD_harmed_SG=int((p.RD_minus_SG_SSE > 0).sum()),
        RD_minus_SG_total_SSE=float(p.RD_minus_SG_SSE.sum()),
        failure_phase_counts=p.failure_phase.value_counts().to_dict(), final_holdout_access=False)
    r.write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


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
    r.write_json(output/'manifest.json', dict(task_id=TASK, branch='RELATION/v1.37', parent='RELATION/v1.36',
        change_type='pivot', started_at=started, source_input_hashes=hashes, new_group_fits=45,
        member_trajectories=135, cached_fits=45, method_folds=90, member_seed_offsets=OFFSETS,
        geometry_combination_weights=[1/3]*3, geometry_combination_step=1,
        max_group_updates=400, patience=80, clipping='native_per_member_5',
        device=torch.cuda.get_device_name(), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1, 6):
            for arm in NEW:
                fit(arm, region, fold, output, restore=arm == 'RD' and fold == 1)
    fit_seconds = time.perf_counter()-tick
    analyze(output)
    assert all(r.digest(Path(path)) == expected for path, expected in hashes.items())
    r.write_json(output/'finished.json', dict(started_at=started, finished_at=datetime.now().astimezone().isoformat(),
        fit_export_seconds=fit_seconds, wall_seconds=time.perf_counter()-tick, protected_hashes_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('math', 'preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args()
    r.setup()
    if args.mode == 'math':
        print(json.dumps(math_checks()))
    elif args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output, args.preflight_dir)
    else:
        analyze(args.output)
