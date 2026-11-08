"""Replay-trained sequential reference editing on unchanged full-training RP."""
import argparse
import copy
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import rp_prediction_agent as old
from rp_agent_reference import ReferencePredictor, RP_ROOT

r = old.r
TASK = 'CARBON-RP-AGENT-VALUE-001'
FEATURES, ALPHA = 144, 2
METHODS = ('main', 'shuffled', 'uniform', 'legacy')
TRAIN_STEPS, DEPLOY_STEPS = (16, 47), (64, 192)
DEPTHS = (0., 0., .1, .2, .3, .4)
CACHE_FILE = r.HERE/'runs/CARBON-RP-AGENT-PREDICTION-001/cache_prototype.py'
spec = importlib.util.spec_from_file_location('value_agent_geometry', CACHE_FILE)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
CachedReference = cache_module.CachedReference


def write(path, obj):
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=default)+'\n', encoding='utf-8')


def require_running():
    body = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    assert json.loads(body.split('<!-- TASK_META', 1)[1].split('TASK_META -->', 1)[0])['status'] == 'RUNNING'


def original_bundle(region, fold):
    engine, data, checkpoint, source = ReferencePredictor.original(region, fold, 'cuda')
    n, pre = engine.n, engine.pre
    x, ids = data['x'][:n].copy(), data['sample_ids'][:n].copy()
    target = (data['y'][:n]-pre['y_mean'])/pre['y_scale']
    initial = engine.predict(np.ones(n, dtype=bool), x, ids)
    bundle = dict(state_dict=engine.full_state, config=engine.config, preprocessor=pre,
        reference_x=x, source_ids=ids, source_indices=np.arange(n), feedback_indices=np.arange(n),
        feedback_x=x, feedback_ids=ids, feedback_y_scaled=target,
        feedback_mse=float(np.mean((initial['scaled'].astype(float)-target)**2)),
        checkpoint=str(checkpoint), checkpoint_sha256=r.digest(checkpoint),
        feedback_contract='all original training labels; already seen by RP; NOT independent validation',
        original_rp_historical_outer_early_stopping=True, new_predictor_training=False)
    assert not set(ids) & set(data['sample_ids'][n:])
    return bundle, data


class ValueAgent(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.Linear(FEATURES, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DynamicEnvironment(old.FeedbackEnvironment):
    def __init__(self, engine, bundle, data):
        super().__init__(engine, bundle, data, ALPHA)
        full = engine.predict(self.full, bundle['feedback_x'], bundle['feedback_ids'], graphs=True)
        self.observations = {self.full.tobytes(): full}
        self.original_graphs = full['graphs']
        self.original_gather = full['packed_query_positions']
        self.original_source_prediction = full['scaled'].astype(float)
        assert np.array_equal(bundle['source_ids'], bundle['feedback_ids'])
        initial = self.credit(full, self.original_graphs, self.original_gather)
        self.credit_scale = np.sqrt(np.mean(initial**2, axis=0))
        self.credit_scale[self.credit_scale < 1e-6] = 1

    def credit(self, observation, graphs, gather):
        prediction = observation['scaled'].astype(float)
        residual = prediction-self.bundle['feedback_y_scaled']
        values = np.zeros((self.n, 8), dtype=np.float64)
        for layer, graph in enumerate(graphs):
            near, weights = graph['neighbors'][gather], graph['weights'][gather].astype(float)
            difference = self.original_source_prediction[near]-prediction[:, None]
            for j, term in enumerate((np.ones_like(weights), np.broadcast_to(residual[:, None], weights.shape),
                                      np.broadcast_to(np.abs(residual[:, None]), weights.shape), residual[:, None]*difference)):
                np.add.at(values[:, layer*4+j], near.ravel(), (weights*term).ravel())
        return values*self.n/len(prediction)

    def evaluate(self, keep):
        self.logical_queries += 1
        key = keep.tobytes()
        if key not in self.memo:
            result = self.engine.predict(keep, self.bundle['feedback_x'], self.bundle['feedback_ids'], graphs=True)
            self.memo[key] = self.mse(result['scaled'])
            self.observations[key] = result
            self.real_queries += 1
        assert np.isfinite(self.memo[key])
        return self.memo[key]

    def features(self, actions, keep, counts, mse, phase, remaining, last_reward, best_cr, since_best):
        static, prior = super().features(actions, keep, counts, mse, phase, remaining, last_reward, best_cr, since_best)
        current = self.observations[keep.tobytes()]
        dynamic = self.credit(current, current['graphs'], current['packed_query_positions'])
        potential = self.credit(current, self.original_graphs, self.original_gather)
        dynamic[~keep] = potential[~keep]
        dynamic = (dynamic/self.credit_scale).astype(np.float32)
        extra = np.stack([np.r_[dynamic[a] if a >= 0 else np.zeros(8),
                                  dynamic[b] if b >= 0 else np.zeros(8)] for a, b, _ in actions]).astype(np.float32)
        result = np.column_stack((static, extra))
        assert result.shape == (len(actions), FEATURES) and np.isfinite(result).all()
        return result, prior


class ReplayLearner:
    def __init__(self, model, seed, shuffled):
        self.model, self.target = model, copy.deepcopy(model).eval()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        self.rng, self.shuffled = np.random.default_rng(seed), shuffled
        self.memory, self.priorities, self.history = [], [], []

    def add(self, x, next_x, reward, phase, terminal):
        self.memory.append((x.copy(), next_x.copy(), float(reward), phase, terminal))
        self.priorities.append(max(self.priorities, default=1.))
        if len(self.memory) < 32:
            return
        for _ in range(4):
            probability = np.asarray(self.priorities, dtype=float)**.6
            probability /= probability.sum()
            ids = self.rng.choice(len(self.memory), 32, replace=True, p=probability)
            rows = [self.memory[i] for i in ids]
            width = max(len(row[1]) for row in rows)
            next_x = np.zeros((32, width, FEATURES), dtype=np.float32)
            valid = np.zeros((32, width), dtype=bool)
            for j, row in enumerate(rows):
                next_x[j, :len(row[1])] = row[1]; valid[j, :len(row[1])] = True
            rewards = np.array([row[2] for row in rows], dtype=np.float32)
            phases = np.array([row[3] for row in rows])
            if self.shuffled:
                for phase_value in (0, 1):
                    positions = np.flatnonzero(phases == phase_value)
                    rewards[positions] = rewards[self.rng.permutation(positions)]
            with torch.no_grad():
                inputs = torch.from_numpy(next_x.reshape(-1, FEATURES))
                online = self.model(inputs).reshape(32, width).masked_fill(~torch.from_numpy(valid), -1e9)
                chosen = online.argmax(1)
                target = self.target(inputs).reshape(32, width)[torch.arange(32), chosen]
                terminal = torch.tensor([row[4] for row in rows], dtype=torch.float32)
                target = torch.from_numpy(rewards)+.95*(1-terminal)*target
            value = self.model(torch.from_numpy(np.stack([row[0] for row in rows])))
            error = target-value
            importance = (len(self.memory)*probability[ids])**(-.4); importance /= importance.max()
            loss = (F.smooth_l1_loss(value, target, reduction='none')*torch.tensor(importance, dtype=torch.float32)).mean()
            assert torch.isfinite(loss)
            self.optimizer.zero_grad(); loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            assert torch.isfinite(norm)
            self.optimizer.step()
            errors = error.detach().abs().numpy()
            for index, priority in zip(ids, errors):
                self.priorities[int(index)] = float(priority)+.01
            self.history.append(dict(loss=float(loss.detach()), gradient_norm=float(norm), mean_abs_TD=float(errors.mean())))
            if len(self.history) % 64 == 0:
                self.target.load_state_dict(self.model.state_dict())


def initial_mask(env, depth, rng):
    keep = env.full.copy()
    counts = env.cover.sum(1)
    desired = max(9, round(env.n*(1-depth)))
    while keep.sum() > desired:
        critical = env.cover[counts == 1].any(0)
        candidates = np.flatnonzero(keep & ~critical)
        if not len(candidates):
            break
        chosen = int(rng.choice(candidates))
        keep[chosen] = False; counts -= env.cover[:, chosen]
    assert env.cover[:, keep].any(1).all()
    return keep


def value_episode(env, model, seed, steps, keep, error, learner=None, uniform=False):
    rng = np.random.default_rng(seed)
    keep = keep.copy()
    accuracy, accuracy_error = keep.copy(), error
    valid = env.feasible(error)
    best, best_error = (keep.copy(), error) if valid else (env.full.copy(), env.base_mse)
    events, last_reward, since_best = [], 0., 0
    for phase, budget in enumerate(steps):
        if phase == 1:
            keep, error = accuracy.copy(), accuracy_error
        actions, counts = env.propose(keep, rng, phase, valid)
        features = (env.features(actions, keep, counts, error, phase, 1., last_reward,
                    1-best.mean() if valid else 0., since_best/max(1, sum(steps)))[0]
                    if model is not None else np.zeros((len(actions), FEATURES), dtype=np.float32))
        for step in range(budget):
            epsilon = 1. if model is None else max(.1, .5-.4*len(learner.memory)/378) if learner is not None else 0.
            with torch.no_grad():
                q = np.zeros(len(actions)) if model is None else model(torch.from_numpy(features)).numpy().astype(float)
            assert np.isfinite(q).all()
            if model is None or (epsilon > 0 and rng.random() < epsilon):
                options = [i for i, action in enumerate(actions) if not (uniform and action[2] == 3 and len(actions) > 1)]
            else:
                options = np.flatnonzero(np.isclose(q, q.max(), rtol=0, atol=1e-12))
            chosen = int(rng.choice(options))
            action = actions[chosen]
            terminal = step+1 == budget or action[2] == 3
            reward = 0.
            if action[2] != 3:
                before = env.potential(keep, error, phase)
                keep = env.move(keep, action)
                error = env.evaluate(keep)
                reward = env.potential(keep, error, phase)-before
                since_best += 1
                if error < accuracy_error:
                    accuracy, accuracy_error = keep.copy(), error
                if env.feasible(error) and (not valid or keep.sum() < best.sum() or (keep.sum() == best.sum() and error < best_error)):
                    best, best_error, valid, since_best = keep.copy(), error, True, 0
            next_actions, next_counts = ([], None) if terminal else env.propose(keep, rng, phase, valid)
            next_features = (env.features(next_actions, keep, next_counts, error, phase, 1-(step+1)/budget, reward,
                             1-best.mean() if valid else 0., since_best/max(1, sum(steps)))[0]
                             if not terminal and model is not None else np.zeros((max(1, len(next_actions)), FEATURES), dtype=np.float32))
            if learner is not None:
                learner.add(features[chosen], next_features, 100*reward, phase, terminal)
            events.append(dict(phase=phase, step=step, action=list(action), stopped=action[2] == 3,
                mse=error, retained=int(keep.sum()), reward=reward, feasible=env.feasible(error),
                epsilon=epsilon, q_min=float(q.min()), q_max=float(q.max())))
            last_reward = reward
            if action[2] == 3:
                break
            actions, counts, features = next_actions, next_counts, next_features
    return best, best_error, valid, events


def select(engine, bundle, data, method, output, small=False):
    output.mkdir(parents=True, exist_ok=False)
    seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])+100000*ALPHA
    started, call_start = time.perf_counter(), engine.calls
    if method == 'legacy':
        env = old.FeedbackEnvironment(engine, bundle, data, ALPHA)
        model = old.Actor(seed+310000)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        actor_rng = np.random.default_rng(seed+410000)
        steps, repetitions = ((8, 12), 2) if small else ((32, 96), 4)
        initial = copy.deepcopy(model.state_dict())
        events, training = [], []
        for iteration in range(repetitions):
            _, _, _, records, trace = old.episode(env, model, seed+1000*iteration, steps)
            training.append(old.update_actor(model, optimizer, records, actor_rng, False))
            events.extend([{**row, 'episode': iteration, 'role': 'training'} for row in trace])
        frozen = copy.deepcopy(model.state_dict())
        keep, error, valid, _, trace = old.episode(env, model, seed+900000, steps)
        events.extend([{**row, 'episode': repetitions, 'role': 'deployment'} for row in trace])
        updates = sum(row['updates'] for row in training)
        norms = [x for row in training for x in row['gradient_norms']]
        learning = dict(kind='legacy_PPO_style', updates=updates, gradient_norms=norms)
        feature_count, query_cap = 128, (repetitions+1)*sum(steps)
    else:
        env = DynamicEnvironment(engine, bundle, data)
        model = None if method == 'uniform' else ValueAgent(seed+320000)
        initial = None if model is None else copy.deepcopy(model.state_dict())
        learner = None if model is None else ReplayLearner(model, seed+420000, method == 'shuffled')
        train_steps, deploy_steps = ((8, 12), (8, 12)) if small else (TRAIN_STEPS, DEPLOY_STEPS)
        depths = DEPTHS[:2] if small else DEPTHS
        events, keep, error, valid = [], env.full.copy(), env.base_mse, False
        for iteration, depth in enumerate(depths):
            start = initial_mask(env, depth, np.random.default_rng(seed+510000+iteration))
            start_error = env.evaluate(start)
            mask, score, feasible, trace = value_episode(env, model, seed+1000*iteration, train_steps,
                                                        start, start_error, learner=learner, uniform=model is None)
            if model is None and feasible and (not valid or mask.sum() < keep.sum() or (mask.sum() == keep.sum() and score < error)):
                keep, error, valid = mask.copy(), score, True
            events.extend([{**row, 'episode': iteration, 'role': 'search' if model is None else 'training',
                            'initial_retained': int(start.sum())} for row in trace])
        frozen = None if model is None else copy.deepcopy(model.state_dict())
        mask, score, feasible, trace = value_episode(env, model, seed+900000, deploy_steps,
                                                    env.full, env.base_mse, uniform=model is None)
        if model is not None or (feasible and (not valid or mask.sum() < keep.sum() or (mask.sum() == keep.sum() and score < error))):
            keep, error, valid = mask.copy(), score, feasible
        events.extend([{**row, 'episode': len(depths), 'role': 'deployment'} for row in trace])
        updates = 0 if learner is None else len(learner.history)
        norms = [] if learner is None else [row['gradient_norm'] for row in learner.history]
        learning = dict(kind='uniform' if learner is None else 'Double_DQN_prioritized_replay', updates=updates,
                        history=[] if learner is None else learner.history)
        if learner is not None:
            sample = learner.memory[:16]+learner.memory[-16:]
            np.savez_compressed(output/'replay_witness.npz', features=np.stack([row[0] for row in sample]),
                reward=np.array([row[2] for row in sample]), phase=np.array([row[3] for row in sample]))
        feature_count, query_cap = FEATURES, len(depths)*(1+sum(train_steps))+sum(deploy_steps)
    assert model is None or all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())
    assert env.logical_queries <= query_cap and env.cover[:, keep].any(1).all()
    assert valid or keep.all()
    prediction = engine.predict(keep, bundle['feedback_x'], bundle['feedback_ids'])
    assert abs(env.mse(prediction['scaled'])-error) <= 1e-10
    np.savez_compressed(output/'selection.npz', keep=keep, source_ids=bundle['source_ids'], feedback_prediction=prediction['prediction'],
                        feedback_mse=error, feedback_target_met=valid)
    engine.export(keep, output/'reduced.pt')
    change = 0. if model is None else max(float((value-initial[key]).abs().max()) for key, value in model.state_dict().items())
    if model is not None:
        torch.save(dict(state_dict=model.state_dict(), seed=seed+(310000 if method == 'legacy' else 320000), features=feature_count), output/'policy.pt')
    report = dict(method=method, alpha=ALPHA, original=engine.n, retained=int(keep.sum()), cr_percent=100*(1-keep.mean()),
        feedback_rmse_ratio=float(np.sqrt(error/env.base_mse)), target_met=bool(valid), optimizer_updates=updates,
        parameter_max_change=change, maximum_gradient_norm=max(norms, default=0.), logical_queries=env.logical_queries,
        logical_query_cap=query_cap, actual_gpu_feedback_queries=env.real_queries, total_engine_calls=engine.calls-call_start,
        seconds=time.perf_counter()-started, parameters=0 if model is None else sum(value.numel() for value in model.parameters()),
        deployment_policy_weights_frozen=True, deployment_from_full_source=True, cached_training_mask_used=False,
        feedback_contract=bundle['feedback_contract'], full_original_RP_unchanged=engine.unchanged_parameters(),
        reduced_bundle_bytes=(output/'reduced.pt').stat().st_size)
    write(output/'trajectory.json', events); write(output/'learning.json', learning); write(output/'summary.json', report)
    print(json.dumps({key: report[key] for key in ('method', 'cr_percent', 'feedback_rmse_ratio', 'optimizer_updates', 'seconds')}), flush=True)
    return report


def hashes():
    values = old.protected_inputs()
    paths = [Path(__file__), CACHE_FILE]
    paths += [RP_ROOT/region/f'fold-{fold:02d}'/'RP/best.pt' for region in r.REGIONS for fold in range(1, 6)]
    return {**values, **{str(path.resolve()): r.digest(path) for path in paths}}


def preflight(output):
    require_running(); output.mkdir(parents=True, exist_ok=False)
    protected = hashes()
    shutil.copy2(__file__, output/Path(__file__).name)
    cases = []
    for region in r.REGIONS:
        bundle, data = original_bundle(region, 1)
        native = old.load_engine(bundle)
        cached = CachedReference(bundle, 'cuda')
        equivalence = []
        rng = np.random.default_rng(20261021)
        for fraction in (1., .9, .6):
            keep = np.zeros(native.n, dtype=bool)
            keep[rng.choice(native.n, max(9, round(native.n*fraction)), replace=False)] = True
            a = native.predict(keep, bundle['feedback_x'], bundle['feedback_ids'], graphs=True)
            b = cached.predict(keep, bundle['feedback_x'], bundle['feedback_ids'], graphs=True)
            error = float(np.max(np.abs(a['scaled']-b['scaled'])))
            same = all(np.array_equal(x['neighbors'], y['neighbors']) for x, y in zip(a['graphs'], b['graphs']))
            assert error <= 3e-6 and same
            equivalence.append(error)
        env = DynamicEnvironment(cached, bundle, data)
        trial = initial_mask(env, .2, rng); env.evaluate(trial)
        actions, counts = env.propose(trial, rng, 0, False)
        observed, _ = env.features(actions, trial, counts, env.memo[trial.tobytes()], 0, 1, 0, 0, 0)
        assert np.isfinite(observed).all() and np.max(np.abs(observed[:, 128:])) > 1e-6
        report = select(cached, bundle, data, 'main', output/region/'main', small=True)
        assert report['parameter_max_change'] > 1e-10 and report['maximum_gradient_norm'] > 0 and report['optimizer_updates'] > 0
        restored = ReferencePredictor.restore(output/region/'main/reduced.pt', 'cuda')
        with np.load(output/region/'main/selection.npz') as saved:
            keep = saved['keep'].copy()
        a = native.predict(keep, bundle['feedback_x'], bundle['feedback_ids'])
        b = restored.predict(np.ones(restored.n, dtype=bool), bundle['feedback_x'], bundle['feedback_ids'])
        restore_error = float(np.max(np.abs(a['scaled']-b['scaled'])))
        assert restore_error <= 3e-6 and native.unchanged_parameters() and cached.unchanged_parameters()
        if region == 'hubei':
            for method in ('shuffled', 'uniform', 'legacy'):
                select(cached, bundle, data, method, output/region/method, small=True)
        cases.append(dict(region=region, source_feedback_overlap_declared=True, query_ids=len(bundle['source_ids']),
            original_RP_unchanged=True, native_cached_errors=equivalence, restore_error=restore_error,
            actor_updates=report['optimizer_updates'], actor_change=report['parameter_max_change']))
    assert all(r.digest(Path(path)) == value for path, value in protected.items())
    write(output/'preflight.json', dict(valid=True, source_sha256=r.digest(Path(__file__)), protected_inputs=protected,
        cases=cases, outer_evaluation=False, final_holdout_access=False))
    print(json.dumps(dict(preflight_valid=True, cases=cases)), flush=True)


def run(output, preflight_dir):
    require_running()
    checked = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    protected = hashes()
    assert checked['valid'] and checked['protected_inputs'] == protected
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    started, tick = old.now(), time.perf_counter()
    write(output/'manifest.json', dict(task_id=TASK, started_at=started, protected_hashes=protected, alpha=ALPHA,
        full_training_reference=True, feedback_training_labels_already_seen_by_RP=True,
        original_RP_early_stopped_on_historical_development=True, new_outer_label_reward=False, final_holdout_access=False,
        total_logical_query_cap=640, train_steps=TRAIN_STEPS, deploy_steps=DEPLOY_STEPS, initial_depths=DEPTHS,
        methods=METHODS, device=torch.cuda.get_device_name()))
    for region in r.REGIONS:
        for fold in range(1, 6):
            if time.perf_counter()-tick > 90*60:
                write(output/'budget_incomplete.json', dict(next_region=region, next_fold=fold, no_outer_evaluation=True))
                return
            dest = output/region/f'fold-{fold:02d}'; dest.mkdir(parents=True)
            bundle, data = original_bundle(region, fold)
            torch.save(bundle, dest/'reference.pt')
            write(dest/'source_contract.json', dict(source_ids=bundle['source_ids'].tolist(), feedback_ids=bundle['feedback_ids'].tolist(),
                outer_ids=data['sample_ids'][len(bundle['source_ids']):].tolist(), source_feedback_overlap=True,
                feedback_is_training_not_validation=True, checkpoint=bundle['checkpoint'], checkpoint_sha256=bundle['checkpoint_sha256']))
            engine = CachedReference(bundle, 'cuda')
            print(json.dumps(dict(region=region, fold=fold, stage='original_RP_loaded')), flush=True)
            for method in METHODS:
                select(engine, bundle, data, method, dest/method)
            assert engine.unchanged_parameters()
            del engine
            write(dest/'finished.json', dict(time=old.now(), methods=4))
    write(output/'selections_frozen.json', dict(time=old.now(), contexts=15, alpha=ALPHA, methods=4))
    analyze(output)
    assert all(r.digest(Path(path)) == value for path, value in protected.items())
    write(output/'finished.json', dict(started_at=started, finished_at=old.now(), wall_seconds=time.perf_counter()-tick,
        original_RP_reused=15, new_RP_fits=0, policy_fits=45, final_holdout_access=False, protected_inputs_unchanged=True))


def analyze(output):
    assert (output/'selections_frozen.json').exists()
    rows, points, sources = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            dest = output/region/f'fold-{fold:02d}'
            bundle = torch.load(dest/'reference.pt', map_location='cpu', weights_only=False)
            _, _, data, _ = r.load_fold(region, fold)
            n = len(bundle['source_ids']); y = data['y'][n:]; ids = data['sample_ids'][n:]
            x = old.transform(data['raw_features'][n:], bundle['preprocessor'])
            high = y >= np.quantile(data['y'][:n], .8)
            native = old.load_engine(bundle)
            predictions = {'RP': native.predict(np.ones(n, dtype=bool), x, ids)['prediction']}
            with np.load(RP_ROOT/region/f'fold-{fold:02d}'/'RP/responses.npz') as original:
                assert np.array_equal(original['sample_ids'], data['sample_ids'])
                assert np.max(np.abs((predictions['RP']-original['prediction'][n:])/bundle['preprocessor']['y_scale'])) <= 3e-6
            with np.load(r.HERE/'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01'/region/f'fold-{fold:02d}'/'G0/responses.npz') as original:
                assert np.array_equal(original['sample_ids'], data['sample_ids'])
                predictions['G0'] = original['prediction'][n:].copy()
            for method in METHODS:
                folder = dest/method
                info = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                with np.load(folder/'selection.npz') as selected:
                    keep = selected['keep'].copy()
                    stored_feedback = selected['feedback_prediction'].copy()
                actual_feedback = native.predict(keep, bundle['feedback_x'], bundle['feedback_ids'])
                native_mse = float(np.mean((actual_feedback['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))
                assert np.max(np.abs((actual_feedback['prediction']-stored_feedback)/bundle['preprocessor']['y_scale'])) <= 3e-6
                assert not info['target_met'] or np.sqrt(native_mse/bundle['feedback_mse']) <= .99+1e-6
                predictions[method] = native.predict(keep, x, ids)['prediction']
                np.savez_compressed(folder/'outer_predictions.npz', sample_ids=ids, target=y, prediction=predictions[method])
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]),
                    **{key: info[key] for key in ('cr_percent', 'retained', 'original', 'parameter_max_change', 'optimizer_updates', 'seconds', 'target_met')},
                    high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
                for i, sid in enumerate(bundle['source_ids']):
                    sources.append(dict(region=region, fold=fold, method=method, sample_id=str(sid), retained=bool(keep[i]),
                        target=float(data['y'][i]), missing_count=int(data['missing_mask'][i].sum())))
            for method in ('RP', 'G0'):
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]), cr_percent=0.,
                    high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
            names = np.asarray(data['feature_names']).astype(str)
            for i, sid in enumerate(ids):
                extreme = int(np.abs(x[i]).argmax())
                points.append(dict(region=region, fold=fold, sample_id=str(sid), target=float(y[i]), high=bool(high[i]),
                    missing_count=int(data['missing_mask'][n+i].sum()), extreme_feature=str(names[extreme]), extreme_z=float(x[i, extreme]),
                    **{method: float(value[i]) for method, value in predictions.items()},
                    main_minus_RP_squared_error=float((predictions['main'][i]-y[i])**2-(predictions['RP'][i]-y[i])**2)))
            assert native.unchanged_parameters()
            del native
    import pandas as pd
    frame, per_point = pd.DataFrame(rows), pd.DataFrame(points)
    assert len(per_point) == 1421 and per_point.sample_id.nunique() == 1421 and len(rows) == 90
    pivot = frame.pivot(index=['region', 'fold'], columns='method', values='rmse')
    cr = frame.pivot(index=['region', 'fold'], columns='method', values='cr_percent')
    high = frame.pivot(index=['region', 'fold'], columns='method', values='high_rmse')
    reduction = {method: float((100*(1-pivot['main']/pivot[method])).mean()) for method in ('RP', 'G0', 'uniform', 'shuffled', 'legacy')}
    province = {region: float((100*(1-pivot.loc[region, 'main']/pivot.loc[region, 'RP'])).mean()) for region in r.REGIONS}
    main = frame[frame.method == 'main']
    checks = dict(accuracy=reduction['RP'] >= 1, compression=bool(main.cr_percent.mean() >= 20),
        uniform_CR=float((cr['main']-cr['uniform']).mean()) >= 1, uniform_accuracy=reduction['uniform'] >= -.1,
        shuffled_CR=float((cr['main']-cr['shuffled']).mean()) > 0, shuffled_accuracy=reduction['shuffled'] >= -.1,
        province_noninferior=min(province.values()) >= -1,
        high_noninferior=float((100*(1-high['main']/high['RP'])).mean()) >= -1,
        active_policy=int((main.parameter_max_change > 1e-10).sum()) >= 10)
    full = all(checks.values())
    pareto = (not full and reduction['RP'] >= .5 and float(main.cr_percent.mean()) >= 10
        and not (reduction['uniform'] < -.1 and float((cr['main']-cr['uniform']).mean()) < -1)
        and checks['province_noninferior'] and checks['high_noninferior'] and checks['active_policy'])
    r.write_csv(output/'all_outer_fold_metrics.csv', rows)
    r.write_csv(output/'all1421_query_diagnosis.csv', points)
    r.write_csv(output/'all_training_reference_memberships.csv', sources)
    aggregate = frame.groupby('method')[['rmse', 'mae', 'r2', 'bias', 'cr_percent', 'high_rmse']].mean().to_dict('index')
    summary = dict(task_id=TASK, gain_class='FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        decision='PROMOTE' if full else 'RETAIN' if pareto else 'REJECT', checks=checks, aggregate=aggregate,
        main_rmse_reduction_percent=reduction, province_reduction_percent=province,
        uniform_CR_advantage_pp=float((cr['main']-cr['uniform']).mean()), source_protocol='original full training pool',
        feedback_is_training_resubstitution=True, original_RP_historical_outer_selection=True,
        query_ids=1421, contexts=15, alpha=ALPHA, policy_fits=45, final_holdout_access=False,
        additional_seeds=False, other_scales=False)
    write(output/'summary.json', summary)
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
