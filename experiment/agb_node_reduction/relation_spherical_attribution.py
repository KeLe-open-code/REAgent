"""Complete the missing local-contrast control of the existing SG2x2 study."""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import relation_backbone_curvature as previous
import numpy as np
import pandas as pd
import torch

r, layer, sphere = previous.r, previous.layer, previous.sphere
TASK = 'CARBON-RELATION-SPHERICAL-ATTRIBUTION-001'
SPHERE = previous.CACHED['SG'][0]
CACHED = {**previous.CACHED, 'CE': (SPHERE, 'CE'), 'GE': (SPHERE, 'GE')}
PAIN = r.HERE/'runs/CARBON-GAIN10-PAIN-AUDIT-001/attempt-02'
NAMED = r.HERE/'runs/CARBON-RELATION-INCIDENT-SPD-001/attempt-01/all_1421_named_input_extremes.csv'
TAIL = r.HERE/'runs/CARBON-RELATION-TAIL-RISK-001/attempt-01/training_all1421_tail_persistence.csv'


class LocalContrast(previous.CurvatureSAGE):
    training_objective = 'local_SG_MSE_plus_original_prototype_contrast_without_spherical_penalties'

    def __init__(self, config):
        super().__init__('JA', config)
        # Only the inherited loss switch changes; the constructed GL field stays local.
        self.variant = 'CE'
        assert self.arm == 'GL' and self.layer_log_metric.requires_grad

    def make_optimizer(self, config):
        self.executed_optimizer = layer.JointOptimizer(self, config)
        return self.executed_optimizer


def construct(arm, config, device):
    assert arm == 'LC'
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as z:
            config['relation_n_train'] = int(z['n_train'])
    r.fix_seed(config['seed'], 1)
    return LocalContrast(config).to(device)


def source_hashes():
    paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
             if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
             and Path(m.__file__).is_file() and Path(m.__file__).resolve().parent == r.HERE}
    paths.add(r.BASE/'baseline.py')
    return {str(p): r.digest(p) for p in sorted(paths)}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    started, cases = time.perf_counter(), []
    for region in r.REGIONS:
        _, config, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, target = r.tensors(data, pre, 'cuda')
        model = construct('LC', config, 'cuda')
        native = previous.construct('JA', config, 'cuda')
        for current in (model, native):
            current.fit_reference(x[:n], te); current.fit_relations(x[:n], target, te)
        assert all(torch.equal(value, native.state_dict()[key]) for key, value in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == 28275
        model.eval(); native.eval()
        with torch.no_grad():
            assert torch.equal(model(x, n, ee)[0], native(x, n, ee)[0])
        buffers = {key: value.detach().clone() for key, value in model.named_buffers()}
        optimizers = [model.make_optimizer(config), layer.JointOptimizer(native, config)]
        error, loss_error = 0., 0.
        for step in range(4):
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            gradients, losses, end_rng = [], [], None
            for current, optimizer in zip((model, native), optimizers):
                torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
                current.train(); optimizer.zero_grad()
                loss = current.training_loss(x[:n], target, te)
                if step == 1:
                    gradients.append(torch.autograd.grad(loss, tuple(current.parameters()), retain_graph=True))
                    losses.append(float(loss.detach()))
                    if current is native:
                        env, homo, _, _ = sphere.spherical_terms(native.geometry_feature, native.label_bins,
                            native.label_values, native.sphere_points)
                        penalty = .01*(env+homo)
                        pg = torch.autograd.grad(penalty, tuple(native.parameters()), allow_unused=True, retain_graph=True)
                        for lc, sg, geometry in zip(gradients[0], gradients[1], pg):
                            expected = lc if geometry is None else lc+geometry
                            error = max(error, float((sg-expected).abs().max()))
                            assert torch.allclose(sg, expected, atol=5e-6, rtol=2e-4)
                        loss_error = abs(losses[1]-losses[0]-float(penalty.detach()))
                        assert loss_error < 1e-6
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters())
                assert (current.layer_log_metric.grad.norm(dim=1) > 0).all()
                torch.nn.utils.clip_grad_norm_(current.parameters(), config['gradient_clip']); optimizer.step()
                state = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                if end_rng is None:
                    end_rng = state
                else:
                    assert all(torch.equal(a, b) for a, b in zip(state, end_rng))
            if step == 0:
                assert all(torch.equal(value, native.state_dict()[key]) for key, value in model.state_dict().items())
        assert all(torch.equal(value, buffers[key]) for key, value in model.named_buffers())
        model.eval()
        with torch.no_grad():
            full = model(x, n, ee)[0]
            alone = model(x[:n], n, te)[0]
            indices = torch.tensor([len(x)-1, n, n+1], device=x.device)
            reordered = model(torch.cat([x[:n], x[indices]]), n, ee)[0][n:]
            singleton = model(torch.cat([x[:n], x[n:n+1]]), n, ee)[0][-1]
            assert torch.allclose(full[:n], alone, atol=3e-6, rtol=3e-6)
            assert torch.allclose(full[indices], reordered, atol=3e-6, rtol=3e-6)
            assert torch.allclose(full[n], singleton, atol=3e-6, rtol=3e-6)
            restored = construct('LC', config, 'cuda'); restored.load_state_dict(r.cpu_state(model)); restored.eval()
            assert torch.equal(restored(x, n, ee)[0], full)
        case = {'region': region, 'LC_steps': 4, 'SG_steps': 4, 'local_GL_preserved': True,
            'initial_and_first_update_exact': True, 'matched_gradient_difference_max_error': error,
            'matched_loss_difference_error': loss_error, 'buffers_frozen': True,
            'query_independent': True, 'checkpoint_exact': True}
        cases.append(case); print(json.dumps(case), flush=True)
    r.write_json(output/'preflight.json', {'cases': cases, 'source_sha256': r.digest(Path(__file__)),
        'source_hashes': source_hashes(), 'seconds': time.perf_counter()-started})


def directory(output, region, fold, arm):
    root, source_arm = CACHED[arm] if arm in CACHED else (output, arm)
    return root/region/f'fold-{fold:02d}'/source_arm


@torch.no_grad()
def export(model, data, pre, output, restore):
    model.eval(); n = int(data['n_train'])
    x, _, ee, _ = r.tensors(data, pre, 'cuda')
    prediction = model(x, n, ee)[0]
    with np.load(output/'responses.npz') as z:
        assert np.array_equal(z['prediction'], prediction.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean'])
    graphs = model.layer_graphs(x, n)
    checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] >= 2 and len(model.geometry_history) >= checkpoint['epoch']
    arrays = {'sample_ids': data['sample_ids']}
    for index, graph in enumerate(graphs, 1):
        near, weights = graph[:2]
        assert near.shape == weights.shape == (len(x), 8)
        assert (near < n).all() and not (near == torch.arange(len(x), device=x.device)[:, None]).any()
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        assert torch.allclose(weights.sum(1), torch.ones(len(x), device=x.device), atol=1e-6, rtol=0)
        arrays[f'layer{index}_neighbors'], arrays[f'layer{index}_weights'] = near.cpu().numpy(), weights.cpu().numpy()
    np.savez_compressed(output/'local_graphs.npz', **arrays)
    r.write_json(output/'contrast_loss_history.json', model.geometry_history)
    report = {'local_arm': model.arm, 'loss_switch': model.variant, 'first_MSE_only': model.geometry_history[0]['first_update_MSE_only'],
        'active_prefix_steps': checkpoint['epoch']-1, 'geometry_parameter_norms': model.layer_log_metric.norm(dim=1).tolist(),
        'spherical_coefficients': [0, 0], 'restore_checked': restore, 'parameters': sum(p.numel() for p in model.parameters())}
    if restore:
        fresh = construct('LC', checkpoint['config'], 'cuda'); fresh.load_state_dict(checkpoint['state_dict']); fresh.eval()
        assert torch.equal(fresh(x, n, ee)[0], prediction)
        report['restore_exact'] = True
    r.write_json(output/'attribution_diagnostics.json', report)


def analyze(output):
    started, folds, points, training = time.perf_counter(), [], [], []
    methods = (*CACHED, 'LC')
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, config, data, _ = r.load_fold(region, fold)
            n, y = int(data['n_train']), data['y']; q80 = np.quantile(y[:n], .8)
            predictions = {}
            for arm in methods:
                path = directory(output, region, fold, arm)
                metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                expected_arm = CACHED[arm][1] if arm in CACHED else arm
                assert (metric['region'], metric['fold'], metric['arm'], metric['seed']) == (region, fold, expected_arm, config['seed'])
                with np.load(path/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    predictions[arm] = z['prediction'].copy()
                mt, mv = r.metrics(y[:n], predictions[arm][:n]), r.metrics(y[n:], predictions[arm][n:])
                assert all(abs(computed[k]-metric[role][k]) < 1e-7 for role, computed in [('train', mt), ('validation', mv)] for k in computed)
                folds.append({'region': region, 'fold': fold, 'arm': arm, **mv,
                    'train_rmse': mt['rmse'], 'best_epoch': metric['best_epoch'], 'epochs_run': metric['epochs_run'],
                    'elapsed_seconds': metric['elapsed_seconds'], 'parameters': metric['parameters']})
            for i in range(n, len(y)):
                row = {'sample_id': str(data['sample_ids'][i]), 'region': region, 'fold': fold, 'observed_agb': float(y[i]),
                    'high_agb': int(y[i] >= q80), 'missing_features_current': int(data['missing_mask'][i].sum())}
                for arm, pred in predictions.items():
                    row[arm+'_prediction'], row[arm+'_squared_error'] = float(pred[i]), float((pred[i]-y[i])**2)
                row['LC_minus_SG_SSE'] = row['LC_squared_error']-row['SG_squared_error']
                points.append(row)
            for i in range(n):
                training.append({'sample_id': str(data['sample_ids'][i]), 'region': region, 'fold': fold,
                    'observed_agb': float(y[i]), 'high_agb': int(y[i] >= q80),
                    'LC_squared_error': float((predictions['LC'][i]-y[i])**2), 'SG_squared_error': float((predictions['SG'][i]-y[i])**2)})
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    p = p.merge(pd.read_csv(PAIN/'all_1421_ids.csv', float_precision='round_trip'),
        on=['sample_id', 'region', 'fold', 'observed_agb'], how='left', validate='one_to_one', suffixes=('', '_archive'))
    p = p.merge(pd.read_csv(NAMED, float_precision='round_trip'), on=['sample_id', 'region', 'fold'], how='left', validate='one_to_one')
    p = p.merge(pd.read_csv(TAIL, float_precision='round_trip'), on=['sample_id', 'region'], how='left', validate='one_to_one', suffixes=('', '_tail'))
    aggregate = t.groupby('sample_id', as_index=False).agg(LC_train_MSE=('LC_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean'))
    p = p.merge(aggregate, on='sample_id', validate='one_to_one')
    assert len(p) == 1421 and p.feature1.notna().all()
    f.to_csv(output/'all_90_fold_metrics.csv', index=False); p.to_csv(output/'all_1421_point_analysis.csv', index=False)
    t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pivot = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    effects = pd.DataFrame({'local_spherical_gain_percent': 100*(1-pivot.SG/pivot.LC),
        'global_spherical_gain_percent': 100*(1-pivot.GE/pivot.CE),
        'interaction_log_ratio_x100': 100*(np.log(pivot.LC/pivot.SG)-np.log(pivot.CE/pivot.GE))})
    effects.to_csv(output/'all_15_factorial_effects.csv')
    province = f.groupby(['region', 'arm'], as_index=False)[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean()
    province.to_csv(output/'province_metrics.csv', index=False)
    local = effects.local_spherical_gain_percent
    wins = {region: int(local.loc[region].gt(0).sum()) for region in r.REGIONS}
    priority = local.mean() >= .5 and local.gt(0).sum() >= 9 and min(wins.values()) >= 2
    gains = {arm: float((100*(1-pivot[arm]/pivot.G0)).mean()) for arm in methods if arm != 'G0'}
    report = {'candidate_G0_RMSE_gains_percent': gains, 'local_spherical_gain_percent': float(local.mean()),
        'global_spherical_gain_percent': float(effects.global_spherical_gain_percent.mean()),
        'interaction_log_ratio_x100': float(effects.interaction_log_ratio_x100.mean()),
        'local_spherical_wins': int(local.gt(0).sum()), 'province_local_wins': wins, 'geometry_priority_gate': bool(priority),
        'LC_helped_SG': int(p.LC_minus_SG_SSE.lt(0).sum()), 'LC_harmed_SG': int(p.LC_minus_SG_SSE.gt(0).sum()),
        'LC_minus_SG_SSE': float(p.LC_minus_SG_SSE.sum()), 'final_holdout_access': False,
        'diagnostic_only': True, 'all_1421_IDs': True, 'analysis_seconds': time.perf_counter()-started}
    r.write_json(output/'summary.json', report); print(json.dumps(report), flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 3 and check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(p)) == h for p, h in check['source_hashes'].items())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md'); shutil.copy2(Path(__file__), output/Path(__file__).name)
    files = [output/'task_frozen.md', preflight_dir/'preflight.json', PAIN/'all_1421_ids.csv', PAIN/'feature_diagnostics.csv', NAMED, TAIL]
    for region in r.REGIONS:
        for fold in range(1, 6):
            source, _, _, _ = r.load_fold(region, fold)
            files.extend([source/'config.json', *[source/'data'/v for v in ('development.npz', 'preprocessor.npz', 'split.csv')]])
            for cache, arm in CACHED.values():
                files.extend(cache/region/f'fold-{fold:02d}'/arm/v for v in ('best.pt', 'metrics.json', 'responses.npz'))
    hashes = {**source_hashes(), **{str(p.resolve()): r.digest(p) for p in files}}
    start, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', {'task_id': TASK, 'started_at': start, 'source_input_hashes': hashes,
        'new_fits': 15, 'cached_fits': 75, 'device': torch.cuda.get_device_name(), 'final_holdout_access': False})
    export_seconds = 0.
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold); holder = []
            def factory(arm, config, device):
                model = construct(arm, config, device); holder.append(model); return model
            r.fit('LC', region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
            mark = time.perf_counter(); export(holder[0], data, pre, directory(output, region, fold, 'LC'), fold == 1)
            export_seconds += time.perf_counter()-mark; holder.clear()
    fit_seconds, fitting_finished = time.perf_counter()-tick, datetime.now().astimezone().isoformat()
    analyze(output)
    assert all(r.digest(Path(p)) == h for p, h in hashes.items())
    r.write_json(output/'finished.json', {'started_at': start, 'fitting_finished_at': fitting_finished,
        'fitting_export_seconds': fit_seconds, 'graph_export_seconds': export_seconds,
        'finished_at': datetime.now().astimezone().isoformat(), 'wall_seconds': time.perf_counter()-tick, 'protected_hashes_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight': preflight(args.output)
    elif args.mode == 'run': run(args.output, args.preflight_dir)
    else: analyze(args.output)
