"""Sequential prediction-feedback agent on physically reduced RP reference graphs."""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import relation_agop as agop
import relation_local_egop as local
import relation_precision as rp
from baseline import fit_preprocessor, transform, graph_edges
from rp_agent_reference import ReferencePredictor, RP_ROOT

r = rp.r
TASK = 'CARBON-RP-AGENT-PREDICTION-001'
STEPS = (32, 96)
EPISODES = 4
TARGET = .99**2
FEATURES = 128
METHODS = ('main', 'shuffled', 'uniform')


def now():
    return datetime.now().astimezone().isoformat()


def require_running():
    body = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    assert json.loads(body.split('<!-- TASK_META', 1)[1].split('TASK_META -->', 1)[0])['status'] == 'RUNNING'


def protected_inputs():
    archived = json.loads((r.HERE/'runs/CARBON-RP-AGENT-REFERENCE-001/original_rp_source_check.json').read_text(encoding='utf-8'))['hashes']
    assert all(r.digest(Path(path)) == value for path, value in archived.items())
    paths = [Path(__file__), r.HERE/'rp_agent_reference.py']
    for region in r.REGIONS:
        for fold in range(1, 6):
            source = r.SOURCE/region/f'fold-{fold:02d}'
            paths.extend([source/'config.json', *[source/'data'/name for name in ('development.npz', 'preprocessor.npz', 'split.csv')],
                          RP_ROOT/region/f'fold-{fold:02d}'/'RP/responses.npz',
                          r.HERE/'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01'/region/f'fold-{fold:02d}'/'G0/responses.npz'])
    return {**archived, **{str(path.resolve()): r.digest(path) for path in paths}}


def inner_split(data, seed):
    n = int(data['n_train'])
    rng = np.random.default_rng(seed+310000)
    ordered = np.argsort(data['y'][:n], kind='stable')
    feedback = np.sort([group[int(rng.integers(len(group)))] for group in np.array_split(ordered, max(1, n//5))])
    source = np.setdiff1d(np.arange(n), feedback)
    assert not np.intersect1d(source, feedback).size and len(source)+len(feedback) == n
    return source, feedback


def source_data(region, fold):
    source_path, config, data, _ = r.load_fold(region, fold)
    source, feedback = inner_split(data, config['seed'])
    pre = fit_preprocessor(data['raw_features'][source], data['y'][source])
    x, qx = transform(data['raw_features'][source], pre), transform(data['raw_features'][feedback], pre)
    cfg = {**config, 'relation_n_train': len(source)}
    return source_path, cfg, data, pre, source, feedback, x, qx


def fit_source_model(region, fold, output, cap=400):
    _, config, data, pre, source, feedback, x, qx = source_data(region, fold)
    output.mkdir(parents=True, exist_ok=False)
    model = rp.construct('RP', config, 'cuda').eval()
    tx = torch.as_tensor(x, device='cuda')
    edges = torch.as_tensor(graph_edges(x, np.empty((0, 185)), 8)[0], device='cuda')
    target = torch.tensor((data['y'][source]-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device='cuda')
    feedback_target = (data['y'][feedback]-pre['y_mean'])/pre['y_scale']
    combined = torch.cat((tx, torch.as_tensor(qx, device='cuda')))
    started, tick = now(), time.perf_counter()
    # Explicitly refit every label-dependent buffer; native RP would load a full-prefix cache.
    agop.AgopSAGE.fit_reference(model, tx, edges)
    model.arm = 'GE'
    history_field = agop.AgopSAGE.fit_relations(model, tx, target, edges, rounds=5)
    model.arm = 'GL'
    history_field += local.LocalEgopSAGE.fit_relations(model, tx, target, edges)
    with torch.no_grad():
        model.label_edges.copy_(torch.quantile(target, torch.linspace(.1, .9, 9, device='cuda')))
        bins = torch.bucketize(target.contiguous(), model.label_edges)
        assert (torch.bincount(bins, minlength=10) > 0).all()
        model.label_bins.copy_(bins)
        model.label_values.copy_(torch.stack([target[bins == k].double().mean() for k in range(10)]))
    fitted = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = model.make_optimizer(config)
    best, stale, best_state, best_epoch, history = float('inf'), 0, None, 0, []
    for epoch in range(1, cap+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(tx, target, edges)
        assert torch.isfinite(loss)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
        assert torch.isfinite(norm)
        optimizer.step(); model.eval()
        with torch.no_grad():
            pred = model(combined, len(source), None)[0][len(source):].cpu().numpy().astype(float)
        mse = float(np.mean((pred-feedback_target)**2))
        history.append(dict(epoch=epoch, loss=float(loss.detach()), feedback_mse=mse, gradient_norm=float(norm)))
        if mse < best-1e-6:
            best, stale, best_state, best_epoch = mse, 0, r.cpu_state(model), epoch
        else:
            stale += 1
        if stale >= 80:
            break
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in fitted.items())
    assert best_state is not None
    model.load_state_dict(best_state); model.eval()
    bundle = dict(state_dict=best_state, config=config, preprocessor=pre, source_indices=source,
                  feedback_indices=feedback, source_ids=data['sample_ids'][source], feedback_ids=data['sample_ids'][feedback],
                  reference_x=x, feedback_x=qx, feedback_y_scaled=feedback_target,
                  best_epoch=best_epoch, feedback_mse=best, epochs_run=epoch, started_at=started, finished_at=now(),
                  seconds=time.perf_counter()-tick, fitted_buffers_source_only=True, outer_label_selection=False)
    torch.save(bundle, output/'warmup.pt')
    r.write_csv(output/'history.csv', history)
    r.write_json(output/'field_fit.json', history_field)
    r.write_json(output/'split.json', dict(source_ids=bundle['source_ids'].tolist(), feedback_ids=bundle['feedback_ids'].tolist(),
        outer_ids=data['sample_ids'][int(data['n_train']):].tolist(), source_count=len(source), feedback_count=len(feedback),
        preprocessor_fit='source only', label_use='source: gradients/fields; feedback: internal early stop/reward; outer: final evaluation only'))
    return bundle, data


def load_engine(bundle):
    return ReferencePredictor(bundle['state_dict'], bundle['config'], bundle['reference_x'], bundle['source_ids'], bundle['preprocessor'], 'cuda')


def coverage(coords, alpha):
    lon, lat = np.deg2rad(coords).T
    value = np.sin((lat[:, None]-lat)/2)**2
    value += np.cos(lat[:, None])*np.cos(lat)*np.sin((lon[:, None]-lon)/2)**2
    distance = 2*np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    np.fill_diagonal(distance, np.inf)
    r0 = float(np.median(distance.min(1)))
    assert r0 > 0
    np.fill_diagonal(distance, 0)
    return distance/r0 <= alpha+1e-10, r0


class Actor(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeedbackEnvironment:
    def __init__(self, engine, bundle, data, alpha):
        self.engine, self.bundle = engine, bundle
        self.n, self.alpha = engine.n, alpha
        self.cover, self.r0 = coverage(data['coords'][bundle['source_indices']], alpha)
        self.full = np.ones(self.n, dtype=bool)
        prediction = engine.predict(self.full, bundle['feedback_x'], bundle['feedback_ids'])
        self.base_mse = self.mse(prediction['scaled'])
        assert self.base_mse > 0 and abs(self.base_mse-bundle['feedback_mse']) < 2e-5
        own = engine.predict(self.full, bundle['reference_x'], bundle['source_ids'])
        pre = bundle['preprocessor']
        sy = (data['y'][bundle['source_indices']]-pre['y_mean'])/pre['y_scale']
        missing = data['missing_mask'][bundle['source_indices']].mean(1)
        self.nodes = np.column_stack((own['embedding'], own['scaled'], sy, own['scaled']-sy,
                                     missing, np.log1p(self.cover.sum(1)), np.ones(self.n))).astype(np.float32)
        assert self.nodes.shape == (self.n, 38)
        scale = self.nodes.std(0); scale[scale < 1e-6] = 1
        self.nodes = (self.nodes-self.nodes.mean(0))/scale
        self.memo = {self.full.tobytes(): self.base_mse}
        self.logical_queries, self.real_queries = 0, 0

    def mse(self, scaled):
        return float(np.mean((np.asarray(scaled, dtype=float)-self.bundle['feedback_y_scaled'])**2))

    def evaluate(self, keep):
        self.logical_queries += 1
        key = keep.tobytes()
        if key not in self.memo:
            result = self.engine.predict(keep, self.bundle['feedback_x'], self.bundle['feedback_ids'])
            self.memo[key] = self.mse(result['scaled'])
            self.real_queries += 1
        return self.memo[key]

    def feasible(self, mse):
        return mse <= TARGET*self.base_mse+1e-12

    def potential(self, keep, mse, phase):
        return -mse/self.base_mse if phase == 0 else 1-keep.mean()-10*max(0., mse/self.base_mse-TARGET)

    def propose(self, keep, rng, phase, valid):
        counts = self.cover[:, keep].sum(1)
        critical = self.cover[counts == 1].any(0)
        deletable = np.flatnonzero(keep & ~critical) if keep.sum() > 9 else np.empty(0, dtype=int)
        removed, retained = np.flatnonzero(~keep), np.flatnonzero(keep)
        take = lambda values: rng.choice(values, min(16, len(values)), replace=False)
        actions = [(int(i), -1, 0) for i in take(deletable)]
        actions += [(-1, int(i), 1) for i in take(removed)]
        if len(removed):
            seen = set()
            for _ in range(64):
                a, b = int(rng.choice(retained)), int(rng.choice(removed))
                if (a, b) not in seen and np.all(counts-self.cover[:, a]+self.cover[:, b] >= 1):
                    seen.add((a, b)); actions.append((a, b, 2))
                    if len(seen) == 16:
                        break
        if phase == 1 and valid and 1-keep.mean() >= .2:
            actions.append((-1, -1, 3))
        if not actions:
            actions.append((-1, -1, 3))
        return actions, counts

    def features(self, actions, keep, counts, mse, phase, remaining, last_reward, best_cr, since_best):
        global_values = np.array([keep.mean(), mse/self.base_mse, phase, remaining, last_reward,
                                  best_cr, since_best, counts.min()/self.n], dtype=np.float32)
        values = []
        for deleted, restored, kind in actions:
            left = self.nodes[deleted] if deleted >= 0 else np.zeros(38, dtype=np.float32)
            right = self.nodes[restored] if restored >= 0 else np.zeros(38, dtype=np.float32)
            affected = counts[self.cover[:, deleted]] if deleted >= 0 else counts
            redundancy = np.array([affected.min()/self.n, affected.mean()/self.n], dtype=np.float32)
            values.append(np.r_[left, right, np.abs(left-right), global_values, redundancy, np.eye(4, dtype=np.float32)[kind]])
        x = np.asarray(values, dtype=np.float32)
        assert x.shape == (len(actions), FEATURES) and np.isfinite(x).all()
        types = np.array([a[2] for a in actions])
        unique, count = np.unique(types, return_counts=True)
        prior = np.array([-np.log(count[np.flatnonzero(unique == k)[0]])-np.log(len(unique)) for k in types], dtype=np.float32)
        return x, prior

    def move(self, keep, action):
        deleted, restored, kind = action
        trial = keep.copy()
        if deleted >= 0:
            assert trial[deleted]; trial[deleted] = False
        if restored >= 0:
            assert not trial[restored]; trial[restored] = True
        assert trial.sum() >= 9 and self.cover[:, trial].any(1).all()
        return trial


def episode(env, model, seed, steps=STEPS, strong_uniform=False):
    rng = np.random.default_rng(seed)
    keep, error = env.full.copy(), env.base_mse
    best_mask, best_error, valid = keep.copy(), error, False
    accuracy_mask, accuracy_error = keep.copy(), error
    records, events, last_reward, since_best = [], [], 0., 0
    for phase, budget in enumerate(steps):
        if phase == 1:
            keep, error = accuracy_mask.copy(), accuracy_error
        for step in range(budget):
            actions, counts = env.propose(keep, rng, phase, valid)
            x, prior = env.features(actions, keep, counts, error, phase, 1-step/budget, last_reward,
                                    1-best_mask.mean() if valid else 0., since_best/max(1, sum(steps)))
            with torch.no_grad():
                logits = torch.from_numpy(prior)+(model(torch.from_numpy(x)) if model is not None else 0)
                if strong_uniform:
                    for i, action in enumerate(actions):
                        if action[2] == 3 and len(actions) > 1:
                            logits[i] = -1e9
                logp = F.log_softmax(logits, dim=0).numpy()
            probabilities = np.exp(logp.astype(float)); probabilities /= probabilities.sum()
            chosen = int(rng.choice(len(actions), p=probabilities))
            action = actions[chosen]
            if action[2] == 3:
                events.append(dict(phase=phase, step=step, action=list(action), stopped=True, mse=error, retained=int(keep.sum())))
                break
            before = env.potential(keep, error, phase)
            trial = env.move(keep, action)
            new_error = env.evaluate(trial)
            reward = env.potential(trial, new_error, phase)-before
            keep, error, last_reward = trial, new_error, reward
            since_best += 1
            if error < accuracy_error:
                accuracy_mask, accuracy_error = keep.copy(), error
            if env.feasible(error) and (not valid or keep.sum() < best_mask.sum() or (keep.sum() == best_mask.sum() and error < best_error)):
                best_mask, best_error, valid, since_best = keep.copy(), error, True, 0
            records.append(dict(x=x, prior=prior, picked=chosen, old_logp=float(logp[chosen]), reward=reward, phase=phase))
            events.append(dict(phase=phase, step=step, action=list(action), stopped=False, mse=error,
                               retained=int(keep.sum()), reward=reward, best_retained=int(best_mask.sum()), feasible=env.feasible(error)))
    return best_mask, best_error, valid, records, events


def logprob(model, records):
    width = max(len(row['prior']) for row in records)
    x = np.zeros((len(records), width, FEATURES), dtype=np.float32)
    base = np.full((len(records), width), -1e9, dtype=np.float32)
    for i, row in enumerate(records):
        n = len(row['prior']); x[i, :n] = row['x']; base[i, :n] = row['prior']
    base = torch.from_numpy(base)
    logits = model(torch.from_numpy(x.reshape(-1, FEATURES))).reshape(len(records), width)+base
    logp = F.log_softmax(logits, dim=1)
    chosen = logp[torch.arange(len(records)), torch.tensor([row['picked'] for row in records])]
    return chosen, (logp.exp()*(logp-base)).sum(1).mean()


def update_actor(model, optimizer, records, rng, shuffled):
    if not records:
        return dict(updates=0, gradient_norms=[], losses=[])
    returns, value, last = np.zeros(len(records), dtype=np.float32), 0., -1
    for i in range(len(records)-1, -1, -1):
        if records[i]['phase'] != last:
            value = 0.; last = records[i]['phase']
        value = records[i]['reward']+.95*value
        returns[i] = value
    for phase in (0, 1):
        ids = np.array([i for i, row in enumerate(records) if row['phase'] == phase])
        if len(ids):
            values = returns[ids].copy()
            if shuffled:
                values = values[rng.permutation(len(values))]
            returns[ids] = (values-values.mean())/max(float(values.std()), 1e-6)
    losses, norms = [], []
    for _ in range(4):
        ids = rng.integers(len(records), size=min(32, len(records)))
        batch = [records[i] for i in ids]
        chosen, kl = logprob(model, batch)
        ratio = (chosen-torch.tensor([row['old_logp'] for row in batch])).exp()
        advantages = torch.from_numpy(returns[ids])
        loss = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages).mean()+.01*kl
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
        assert torch.isfinite(norm)
        optimizer.step()
        losses.append(float(loss.detach())); norms.append(float(norm))
    return dict(updates=len(losses), gradient_norms=norms, losses=losses)


def select_references(engine, bundle, data, alpha, method, output, episodes=EPISODES, steps=STEPS):
    output.mkdir(parents=True, exist_ok=False)
    seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])+100000*alpha
    env = FeedbackEnvironment(engine, bundle, data, alpha)
    model = None if method == 'uniform' else Actor(seed+310000)
    initial = None if model is None else {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = None if model is None else torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed+410000)
    best, best_error, valid = env.full.copy(), env.base_mse, False
    all_events, fits = [], []
    start, calls = time.perf_counter(), engine.calls
    for iteration in range(episodes):
        mask, error, feasible, records, events = episode(env, model, seed+1000*iteration, steps, strong_uniform=model is None)
        if model is not None:
            fits.append(update_actor(model, optimizer, records, rng, method == 'shuffled'))
        elif feasible and (not valid or mask.sum() < best.sum() or (mask.sum() == best.sum() and error < best_error)):
            best, best_error, valid = mask.copy(), error, True
        all_events.extend([{**row, 'episode': iteration, 'role': 'training' if model is not None else 'search'} for row in events])
    deployed_state = None if model is None else {k: v.clone() for k, v in model.state_dict().items()}
    # A fresh deployment starts from all source nodes; never from a training best mask.
    mask, error, feasible, _, events = episode(env, model, seed+900000, steps, strong_uniform=model is None)
    assert model is None or all(torch.equal(value, model.state_dict()[name]) for name, value in deployed_state.items())
    if model is not None or (feasible and (not valid or mask.sum() < best.sum() or (mask.sum() == best.sum() and error < best_error))):
        best, best_error, valid = mask.copy(), error, feasible
    all_events.extend([{**row, 'episode': episodes, 'role': 'deployment'} for row in events])
    assert env.cover[:, best].any(1).all() and len(best) == engine.n
    if not valid:
        assert best.all()
    prediction = engine.predict(best, bundle['feedback_x'], bundle['feedback_ids'])
    assert abs(env.mse(prediction['scaled'])-best_error) <= 1e-10
    np.savez_compressed(output/'selection.npz', keep=best, feedback_prediction=prediction['prediction'], feedback_mse=best_error,
                        feedback_target_met=valid, source_ids=bundle['source_ids'])
    engine.export(best, output/'reduced.pt')
    change = 0. if model is None else max(float((v-initial[k]).abs().max()) for k, v in model.state_dict().items())
    if model is not None:
        torch.save({'state_dict': model.state_dict(), 'seed': seed+310000, 'features': FEATURES}, output/'policy.pt')
    r.write_json(output/'trajectory.json', all_events)
    report = dict(method=method, alpha=alpha, original=engine.n, retained=int(best.sum()), cr_percent=100*(1-best.mean()),
                  feedback_rmse_ratio=float(np.sqrt(best_error/env.base_mse)), target_met=bool(valid),
                  logical_queries=env.logical_queries, actual_gpu_feedback_queries=env.real_queries,
                  total_engine_calls=engine.calls-calls, seconds=time.perf_counter()-start,
                  optimizer_updates=sum(f['updates'] for f in fits), parameter_max_change=change, fits=fits,
                  r0_angle=env.r0, source_universe='inner_source_only', query_labels_used='inner_feedback_only',
                  training_best_mask_used_in_deployment=False, coverage_pass=True,
                  deployment_policy_weights_frozen=True, method_local_exact_state_memoization=True,
                  logical_query_cap=(episodes+1)*sum(steps), reduced_bundle_bytes=(output/'reduced.pt').stat().st_size,
                  parameters=0 if model is None else sum(p.numel() for p in model.parameters()))
    assert env.logical_queries <= report['logical_query_cap']
    r.write_json(output/'summary.json', report)
    print(json.dumps({k: report[k] for k in ('method', 'alpha', 'cr_percent', 'feedback_rmse_ratio', 'optimizer_updates', 'seconds')}), flush=True)
    return report


def preflight(output):
    require_running(); output.mkdir(parents=True, exist_ok=False)
    protected = protected_inputs()
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    cases = []
    for region in r.REGIONS:
        bundle, data = fit_source_model(region, 1, output/region/'warmup', cap=4)
        n = int(data['n_train'])
        assert np.all(bundle['source_indices'] < n) and np.all(bundle['feedback_indices'] < n)
        assert not set(bundle['source_ids']) & set(bundle['feedback_ids'])
        assert not (set(bundle['source_ids']) | set(bundle['feedback_ids'])) & set(data['sample_ids'][n:])
        engine = load_engine(bundle)
        summary = select_references(engine, bundle, data, 3, 'main', output/region/'agent', episodes=1, steps=(4, 8))
        assert summary['optimizer_updates'] > 0 and summary['parameter_max_change'] > 1e-10
        assert max(summary['fits'][0]['gradient_norms']) > 0
        with np.load(output/region/'agent/selection.npz') as saved:
            keep = saved['keep'].copy()
        restored = ReferencePredictor.restore(output/region/'agent/reduced.pt', 'cuda')
        a = engine.predict(keep, bundle['feedback_x'], bundle['feedback_ids'])
        b = restored.predict(np.ones(restored.n, dtype=bool), bundle['feedback_x'], bundle['feedback_ids'])
        error = float(np.max(np.abs(a['scaled']-b['scaled'])))
        assert error <= 3e-6 and engine.unchanged_parameters()
        cases.append(dict(region=region, source=len(bundle['source_indices']), feedback=len(bundle['feedback_indices']),
                          actual_rp_updates=bundle['epochs_run'], actual_actor_updates=summary['optimizer_updates'],
                          actor_change=summary['parameter_max_change'], reduced_restore_error=error,
                          source_only_fitted=True, outer_labels_in_reward=False, final_holdout_access=False))
        del engine, restored
    assert all(r.digest(Path(path)) == value for path, value in protected.items())
    report = dict(valid=True, cases=cases, source_sha256=r.digest(Path(__file__)), protected_inputs=protected, stages=STEPS, episodes=EPISODES,
                  formal_outer_query_evaluation=False, finished_at=now())
    r.write_json(output/'preflight.json', report)
    print(json.dumps(report), flush=True)


def run(output, preflight_dir):
    require_running()
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert check['valid'] and check['source_sha256'] == r.digest(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(r.HERE/'rp_agent_reference.py', output/'rp_agent_reference.py')
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    protected = protected_inputs()
    assert protected == check['protected_inputs']
    started, tick = now(), time.perf_counter()
    r.write_json(output/'manifest.json', dict(task_id=TASK, started_at=started, seed='20261014+fold-1', stages=STEPS,
        episodes=EPISODES, feedback_target_rmse_ratio=.99, source_fraction=.8, protected_hashes=protected,
        outer_labels_for_rewards=False, final_holdout_access=False, device=torch.cuda.get_device_name()))
    for region in r.REGIONS:
        for fold in range(1, 6):
            if time.perf_counter()-tick > 180*60:
                r.write_json(output/'budget_incomplete.json', dict(finished_at=now(), wall_seconds=time.perf_counter()-tick,
                    next_region=region, next_fold=fold, no_outer_evaluation=True))
                return
            dest = output/region/f'fold-{fold:02d}'
            bundle, data = fit_source_model(region, fold, dest/'warmup')
            print(json.dumps({'stage': 'warmup_done', 'region': region, 'fold': fold, 'epochs': bundle['epochs_run'], 'seconds': bundle['seconds']}), flush=True)
            engine = load_engine(bundle)
            for alpha in (1, 2, 3):
                for method in METHODS:
                    select_references(engine, bundle, data, alpha, method, dest/f'alpha-{alpha}'/method)
            assert engine.unchanged_parameters()
            del engine
            r.write_json(dest/'finished.json', dict(finished_at=now(), all_nine_conditions_done=True))
    # No outer metric is evaluated until every policy and selected set is frozen.
    r.write_json(output/'selections_frozen.json', dict(time=now(), contexts=15, conditions=45))
    analyze(output)
    assert all(r.digest(Path(path)) == value for path, value in protected.items())
    r.write_json(output/'finished.json', dict(started_at=started, finished_at=now(), wall_seconds=time.perf_counter()-tick,
        source_predictor_fits=15, policy_fits=90, final_holdout_access=False, protected_inputs_unchanged=True))


def analyze(output):
    assert (output/'selections_frozen.json').exists()
    rows, points, identities = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            dest = output/region/f'fold-{fold:02d}'
            bundle = torch.load(dest/'warmup/warmup.pt', map_location='cpu', weights_only=False)
            _, _, data, _ = r.load_fold(region, fold)
            n = int(data['n_train'])
            query_x, ids, y = transform(data['raw_features'][n:], bundle['preprocessor']), data['sample_ids'][n:], data['y'][n:]
            engine = load_engine(bundle)
            full = engine.predict(np.ones(engine.n, dtype=bool), query_x, ids)['prediction']
            with np.load(RP_ROOT/region/f'fold-{fold:02d}'/'RP/responses.npz') as saved:
                assert np.array_equal(saved['sample_ids'], data['sample_ids'])
                original = saved['prediction'][n:].copy()
            with np.load(r.HERE/'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01'/region/f'fold-{fold:02d}'/'G0/responses.npz') as saved:
                assert np.array_equal(saved['sample_ids'], data['sample_ids'])
                g0 = saved['prediction'][n:].copy()
            high = y >= np.quantile(data['y'][bundle['source_indices']], .8)
            for alpha in (1, 2, 3):
                predictions = {'paired_RP': full, 'original_RP': original, 'G0': g0}
                for method in METHODS:
                    path = dest/f'alpha-{alpha}'/method
                    summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
                    with np.load(path/'selection.npz') as selected:
                        keep = selected['keep'].copy()
                    prediction = engine.predict(keep, query_x, ids)['prediction']
                    predictions[method] = prediction
                    np.savez_compressed(path/'outer_predictions.npz', sample_ids=ids, target=y, prediction=prediction)
                    row = {**{k: summary[k] for k in ('cr_percent', 'retained', 'original', 'target_met', 'optimizer_updates', 'parameter_max_change', 'seconds')},
                           'region': region, 'fold': fold, 'alpha': alpha, 'method': method, **r.metrics(y, prediction),
                           'high_rmse': float(np.sqrt(np.mean((prediction[high]-y[high])**2))),
                           'ordinary_rmse': float(np.sqrt(np.mean((prediction[~high]-y[~high])**2)))}
                    rows.append(row)
                    for j, sid in enumerate(bundle['source_ids']):
                        identities.append(dict(region=region, fold=fold, alpha=alpha, method=method,
                                               sample_id=str(sid), role='source', retained=bool(keep[j])))
                    for sid in bundle['feedback_ids']:
                        identities.append(dict(region=region, fold=fold, alpha=alpha, method=method,
                                               sample_id=str(sid), role='feedback', retained=False))
                for method in ('paired_RP', 'original_RP', 'G0'):
                    prediction = predictions[method]
                    rows.append(dict(region=region, fold=fold, alpha=alpha, method=method, cr_percent=0., **r.metrics(y, prediction),
                        high_rmse=float(np.sqrt(np.mean((prediction[high]-y[high])**2))),
                        ordinary_rmse=float(np.sqrt(np.mean((prediction[~high]-y[~high])**2)))))
                for i, sid in enumerate(ids):
                    points.append(dict(region=region, fold=fold, alpha=alpha, sample_id=str(sid), target=float(y[i]),
                        high=bool(high[i]), missing_count=int(data['missing_mask'][n+i].sum()),
                        max_abs_source_z=float(np.abs(query_x[i]).max()), **{method: float(pred[i]) for method, pred in predictions.items()},
                        main_minus_paired_squared_error=float((predictions['main'][i]-y[i])**2-(full[i]-y[i])**2)))
            del engine
    import pandas as pd
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index=['region', 'fold', 'alpha'], columns='method', values='rmse')
    cr = frame.pivot(index=['region', 'fold', 'alpha'], columns='method', values='cr_percent')
    reduction = {method: float((100*(1-pivot['main']/pivot[method])).mean()) for method in ('paired_RP', 'original_RP', 'G0', 'uniform', 'shuffled')}
    main = frame[frame.method == 'main']
    high = frame.pivot(index=['region', 'fold', 'alpha'], columns='method', values='high_rmse')
    province = {region: float((100*(1-pivot.loc[region, 'main']/pivot.loc[region, 'paired_RP'])).mean()) for region in r.REGIONS}
    checks = dict(accuracy=reduction['paired_RP'] >= 1, strong_original_rp=reduction['original_RP'] >= 0,
        compression=main.cr_percent.mean() >= 20,
        uniform_compression=float((cr['main']-cr['uniform']).mean()) >= 1,
        uniform_accuracy=reduction['uniform'] >= -.1,
        shuffled_compression=float((cr['main']-cr['shuffled']).mean()) > 0,
        shuffled_accuracy=reduction['shuffled'] >= -.1,
        province_noninferior=min(province.values()) >= -1,
        high_noninferior=float((100*(1-high['main']/high['paired_RP'])).mean()) >= -1,
        active_policy=int((main.parameter_max_change > 1e-10).sum()) >= 30)
    full_gain = all(checks.values())
    pareto = (not full_gain and reduction['paired_RP'] >= .5 and main.cr_percent.mean() >= 10
              and not (reduction['uniform'] < -.1 and float((cr['main']-cr['uniform']).mean()) < -1)
              and checks['province_noninferior'] and checks['high_noninferior'] and checks['active_policy'])
    counts = pd.DataFrame(points)
    assert len(counts) == 1421*3 and counts.groupby('alpha').sample_id.nunique().eq(1421).all()
    r.write_csv(output/'all_outer_fold_metrics.csv', rows)
    r.write_csv(output/'all_1421_outer_queries_three_scales.csv', points)
    r.write_csv(output/'all_training_reference_identities.csv', identities)
    forty = counts.sort_values('main_minus_paired_squared_error', ascending=False).head(20)
    forty = pd.concat((forty, counts.sort_values('main_minus_paired_squared_error').head(20)))
    forty.to_csv(output/'forty_cases.csv', index=False, encoding='utf-8')
    aggregate = frame.groupby('method')[['rmse', 'mae', 'r2', 'bias', 'cr_percent', 'high_rmse', 'ordinary_rmse']].mean().to_dict('index')
    summary = dict(task_id=TASK, gain_class='FULL_GAIN' if full_gain else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        decision='PROMOTE' if full_gain else 'RETAIN' if pareto else 'REJECT', checks=checks,
        main_paired_rmse_reductions_percent=reduction, province_reduction_percent=province,
        compression_advantage_uniform_pp=float((cr['main']-cr['uniform']).mean()), aggregate=aggregate,
        query_ids=1421, query_scale_rows=len(points), contexts=15, conditions=45, policy_models=90,
        final_holdout_access=False, added_seed_confirmation=False, original_rp_reference_development_selected=True)
    r.write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output, args.preflight_dir)
    else:
        analyze(args.output)
