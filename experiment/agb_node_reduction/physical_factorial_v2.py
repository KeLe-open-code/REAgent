"""Four frozen-method arms and deployment-step matched random effectiveness."""
import argparse
import ctypes
import gc
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import physical_baseline_v2 as b

p, r = b.agent, b.r
TASK = b.TASK
ROOT = Path('runs')/TASK
NATIVE_PREFLIGHT = ROOT/'preflight-02'
RANDOM_DRAWS = 3
PREFIXES = (64, 128, 256)


def cleanup():
    with b.GPU_ACCESS():
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def save_endpoint(env, keep, output, method, details, curves=None, trace=None):
    output.mkdir(parents=True, exist_ok=False)
    key = keep.tobytes()
    error = env.memo[key]
    assert env.feasible(error) and env.utility(keep) >= -1e-12
    assert env.cover[:,keep].any(1).all() and keep.sum() >= 10
    assert env.engine.unchanged_parameters()
    np.savez_compressed(output/'selection.npz', keep=keep, source_ids=env.bundle['source_ids'],
        mse=error, utility=env.utility(keep), drift=env.drift(keep)[1])
    env.engine.export(keep, output/'reduced.pt')
    if curves is not None:
        np.savez_compressed(output/'prefix_masks.npz', **{str(k):v for k,v in curves.items()})
    if trace is not None:
        r.write_json(output/'trajectory.json', trace)
    report = dict(method=method, original=env.n, retained=int(keep.sum()),
        cr_percent=100*(1-keep.mean()), source_RMSE_ratio=float(np.sqrt(error/env.base_mse)),
        utility=env.utility(keep), drift=env.drift(keep)[1], logical_queries=env.logical_queries,
        actual_new_feedback=env.real_queries, engine_calls=env.engine.calls, **details)
    r.write_json(output/'summary.json', report)
    return report


def score_select(bundle, output):
    tick = time.perf_counter()
    env = b.environment(bundle)
    full = env.observations[env.full.tobytes()]
    importance = np.zeros(env.n)
    for graph in full['graphs']:
        np.add.at(importance, graph['neighbors'][full['packed_query_positions']].ravel(),
                  graph['weights'][full['packed_query_positions']].ravel())
    order = np.lexsort((bundle['source_ids'].astype(str), importance))
    keep, trace, prefixes = env.full.copy(), [], {}
    geometry_skips, attempts = 0, 0
    while attempts < 256:
        accepted_this_pass = False
        for index in order:
            if attempts >= 256:
                break
            if not keep[index]:
                continue
            if keep.sum() <= 10 or np.min(env.cover[:,keep].sum(1)[env.cover[:,index]]) <= 1:
                geometry_skips += 1
                continue
            trial = keep.copy(); trial[index] = False
            error = env.evaluate(trial)
            attempts += 1
            accepted = env.feasible(error) and env.utility(trial) >= -1e-12
            if accepted:
                keep = trial
                accepted_this_pass = True
            trace.append(dict(step=attempts, deleted=int(index), accepted=bool(accepted),
                retained=int(keep.sum()), trial_error=error, trial_U=env.utility(trial)))
            if attempts in PREFIXES:
                prefixes[attempts] = keep.copy()
        if not accepted_this_pass:
            break
    for cap in PREFIXES:
        prefixes.setdefault(cap, keep.copy())
    assert attempts == env.logical_queries
    save_endpoint(env, env.full, output.parent/'full', 'full', dict(seconds=0., selection_feedback=0))
    report = save_endpoint(env, keep, output, 'score', dict(seconds=time.perf_counter()-tick,
        selection_feedback=attempts, geometry_skips=geometry_skips), prefixes, trace)
    del env
    cleanup()
    return report


def deployment(env, model, seed, steps, uniform=False):
    """Capture prefix incumbents without looking at any outer predictions."""
    original = env.evaluate
    best, best_error, requests = env.full.copy(), env.base_mse, 0
    curves = {}
    def feedback(keep):
        nonlocal best, best_error, requests
        error = original(keep)
        requests += 1
        if env.feasible(error) and (env.utility(keep) > env.utility(best)+1e-12 or
                (abs(env.utility(keep)-env.utility(best)) <= 1e-12 and error < best_error)):
            best, best_error = keep.copy(), error
        if requests in PREFIXES:
            curves[requests] = best.copy()
        return error
    env.evaluate = feedback
    try:
        keep, error, trace = p.episode(env, model, seed, steps, env.full, env.base_mse, uniform=uniform)
    finally:
        env.evaluate = original
    assert np.array_equal(keep, best)
    counts = [sum(row['phase'] == phase and row['action'][2] != 3 for row in trace) for phase in (0,1)]
    assert sum(counts) == requests
    for cap in PREFIXES:
        curves.setdefault(cap, keep.copy())
    return keep, error, trace, curves, counts


def agent_select(bundle, output, small=False):
    tick = time.perf_counter()
    env = b.environment(bundle)
    config = bundle['config']
    seed = int(config['seed'])+10000*r.REGIONS.index(config['region'])+200000
    model = p.StopAgent(seed+320000, anchored=True)
    initial = {key:value.clone() for key,value in model.state_dict().items()}
    learner = p.ReplayLearner(model, seed+320001, False, 1.)
    events = []
    depths = (0., .2) if small else p.v.DEPTHS
    train_steps = (20,15) if small else p.v.TRAIN_STEPS
    for i,depth in enumerate(depths):
        start = p.v.initial_mask(env, depth, np.random.default_rng(seed+400+i))
        error = env.evaluate(start)
        _, _, trace = p.episode(env, model, seed+500+i, train_steps, start, error, learner)
        events.extend(dict(row, role='training', episode=i) for row in trace)
    trained = {key:value.clone() for key,value in model.state_dict().items()}
    change = max(float((value-initial[key]).abs().max()) for key,value in trained.items())
    assert learner.history and change > 0
    env.group_rng = np.random.default_rng(seed+460000)
    before_requests, before_new = env.logical_queries, env.real_queries
    keep, error, trace, curves, counts = deployment(env, model, seed+900, (5,15) if small else (64,192))
    events.extend(dict(row, role='deployment', episode=len(depths)) for row in trace)
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in trained.items())
    report = save_endpoint(env, keep, output, 'agent', dict(seconds=time.perf_counter()-tick,
        training_feedback=before_requests, deployment_feedback=env.logical_queries-before_requests,
        deployment_new_feedback=env.real_queries-before_new, phase_feedback=counts,
        optimizer_updates=len(learner.history), actor_parameter_change=change,
        actor_parameters=sum(value.numel() for value in model.parameters()),
        maximum_gradient=max(row['gradient_norm'] for row in learner.history),
        deployment_policy_frozen=True, training_mask_used=False), curves, events)
    torch.save(dict(state_dict=trained, seed=seed+320000), output/'policy.pt')
    r.write_json(output/'learning.json', learner.history)
    del env, learner, model
    cleanup()
    return report


def random_select(bundle, phase_steps, draw, output):
    tick = time.perf_counter()
    env = b.environment(bundle)
    config = bundle['config']
    region_index = r.REGIONS.index(config['region'])
    seed = 8000000+10000*region_index+100*config['fold']+draw
    env.group_rng = np.random.default_rng(seed+460000)
    keep, error, trace, curves, counts = deployment(env, None, seed, phase_steps, uniform=True)
    assert counts == phase_steps, (counts, phase_steps)
    report = save_endpoint(env, keep, output, 'random', dict(seconds=time.perf_counter()-tick,
        draw=draw, random_seed=seed, phase_feedback=counts, matched_phase_feedback=phase_steps,
        deployment_feedback=sum(counts), restarts=0, training_candidates_used=False), curves, trace)
    del env
    cleanup()
    return report


def run_case(predictor_path, output):
    output = Path(output)
    finished = output/'case_finished.json'
    if finished.exists():
        return json.loads(finished.read_text(encoding='utf-8'))
    bundle = torch.load(predictor_path, map_location='cpu', weights_only=False)
    outputs = {}
    for name, function in (('score', score_select), ('agent', agent_select)):
        folder = output/name
        if (folder/'summary.json').exists() and (folder/'reduced.pt').exists():
            outputs[name] = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        else:
            outputs[name] = function(bundle, folder)
    for draw in range(RANDOM_DRAWS):
        name = f'random-{draw:02d}'
        folder = output/name
        if (folder/'summary.json').exists() and (folder/'reduced.pt').exists():
            outputs[name] = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        else:
            outputs[name] = random_select(bundle, outputs['agent']['phase_feedback'], draw, folder)
    result = dict(region=bundle['config']['region'], fold=bundle['config']['fold'], arm=bundle['arm'],
                  methods={key:dict(cr_percent=value['cr_percent'], seconds=value['seconds']) for key,value in outputs.items()})
    r.write_json(finished, result)
    return result


def initialize_worker():
    r.setup()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    lock = kernel.CreateMutexW(None, False, 'Local\\CarbonPhysicalV2GPU')
    @contextmanager
    def access():
        status = kernel.WaitForSingleObject(lock, 300000)
        assert status in (0, 128), status
        started = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                if torch.cuda.memory_reserved()-torch.cuda.memory_allocated() > 128*1024**2:
                    torch.cuda.empty_cache()
            ended = time.perf_counter()
            with (ROOT/f'gpu-access-{os.getpid()}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(pid=os.getpid(), start=started, end=ended))+'\n')
            assert kernel.ReleaseMutex(lock)
    b.GPU_ACCESS = access


def agent_preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for arm in b.ARMS:
        bundle = torch.load(NATIVE_PREFLIGHT/'hubei'/arm/'predictor.pt', map_location='cpu', weights_only=False)
        score = score_select(bundle, output/arm/'score')
        learned = agent_select(bundle, output/arm/'agent', small=True)
        random = random_select(bundle, learned['phase_feedback'], 0, output/arm/'random-00')
        assert learned['phase_feedback'] == random['phase_feedback']
        assert learned['actor_parameters'] == 11393 and learned['deployment_policy_frozen']
        cases.append(dict(arm=arm, score_CR=score['cr_percent'], agent_CR=learned['cr_percent'],
                          matched_steps=learned['phase_feedback'], valid=True))
    r.write_json(output/'preflight.json', dict(valid=True, cases=cases, protected_inputs=b.protected(),
                                             final_holdout_access=False))
    print(json.dumps(dict(agent_preflight='VALID', cases=cases)), flush=True)


def run(output):
    task_text = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    assert json.loads(task_text.split('<!-- TASK_META',1)[1].split('TASK_META -->',1)[0])['status'] == 'RUNNING'
    protected = b.protected()
    for path in (NATIVE_PREFLIGHT/'preflight.json', ROOT/'agent-preflight-01/preflight.json'):
        evidence = json.loads(path.read_text(encoding='utf-8'))
        assert evidence['valid'] and evidence['protected_inputs'] == protected
    output.mkdir(parents=True, exist_ok=False)
    for path in (Path(b.__file__), Path(__file__)):
        shutil.copy2(path, output/path.name)
    (output/'task_frozen.md').write_text(task_text, encoding='utf-8')
    started, tick = b.now(), time.perf_counter()
    r.write_json(output/'manifest.json', dict(task_id=TASK, started_at=started, protected_inputs=protected,
        seed_offset=0, random_draws=RANDOM_DRAWS, new_backbone_fits=60, new_agent_fits=30,
        physical_endpoints=180, final_holdout_access=False))
    jobs = []
    for region in r.REGIONS:
        for fold in range(1,6):
            for arm in b.ARMS:
                folder = output/region/f'fold-{fold:02d}'/arm
                b.fit_predictor(region, fold, arm, folder/'predictor')
                jobs.append((folder/'predictor/predictor.pt', folder))
    r.write_json(output/'predictors_finished.json', dict(time=b.now(), predictors=30, fits=60,
                                                        seconds=time.perf_counter()-tick))
    print(json.dumps(dict(event='all_predictors_ready', fits=60)), flush=True)
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker) as pool:
        futures = {pool.submit(run_case, str(path), str(folder)):folder for path,folder in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps(dict(event='selection_case_done', completed=number, total=30, **result)), flush=True)
    assert b.protected() == protected
    selected = []
    for _, folder in jobs:
        for method in ('full','score','agent', *(f'random-{i:02d}' for i in range(RANDOM_DRAWS))):
            selected.append(folder/method/'selection.npz')
            if method != 'full':
                selected.append(folder/method/'prefix_masks.npz')
    assert len([p for p in selected if p.name=='selection.npz']) == 180
    r.write_json(output/'selections_frozen.json', dict(time=b.now(), files={str(path):r.digest(path) for path in selected},
                                                      outer_labels_used=False))
    analyze(output)
    r.write_json(output/'finished.json', dict(started_at=started, finished_at=b.now(), seconds=time.perf_counter()-tick,
        new_backbone_fits=60, new_agent_fits=30, random_deployments=90, protected_inputs_unchanged=b.protected()==protected,
        final_holdout_access=False))


def restore(path):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle.get('arm') != 'G0':
        bundle = {**bundle, 'arm':'RP', 'source_ids':bundle['reference_ids']}
    return b.engine(bundle)


def analyze(output):
    frozen = json.loads((output/'selections_frozen.json').read_text(encoding='utf-8'))
    assert all(r.digest(Path(path))==value for path,value in frozen['files'].items())
    metrics, points, memberships, curve_rows, restores = [], [], [], [], []
    for region in r.REGIONS:
        for fold in range(1,6):
            _, _, cached, _ = r.load_fold(region, fold)
            n = int(cached['n_train'])
            y, query_ids = cached['y'][n:], cached['sample_ids'][n:]
            for arm in b.ARMS:
                folder = output/region/f'fold-{fold:02d}'/arm
                bundle = torch.load(folder/'predictor/predictor.pt', map_location='cpu', weights_only=False)
                qx = b.transform(cached['raw_features'][n:], bundle['preprocessor'])
                assert not set(query_ids) & set(bundle['source_ids'])
                predictor = b.engine(bundle)
                high_threshold = float(np.quantile(bundle['source']['y'], .8))
                high = y >= high_threshold
                for method in ('full','score','agent', *(f'random-{i:02d}' for i in range(RANDOM_DRAWS))):
                    target = folder/method
                    with np.load(target/'selection.npz') as saved:
                        keep = saved['keep'].copy()
                    result = predictor.predict(keep, qx, query_ids)
                    physical = restore(target/'reduced.pt')
                    restored = physical.predict(np.ones(physical.n,dtype=bool), qx, query_ids)
                    delta = float(np.max(np.abs(result['scaled']-restored['scaled'])))
                    assert np.allclose(result['scaled'], restored['scaled'], atol=3e-6, rtol=3e-6)
                    assert physical.n == int(keep.sum()) and physical.unchanged_parameters()
                    restores.append(dict(region=region, fold=fold, arm=arm, method=method, max_scaled_error=delta))
                    row = dict(region=region, fold=fold, arm=arm, method=method,
                        **r.metrics(y, result['prediction']), cr_percent=100*(1-keep.mean()), retained=int(keep.sum()),
                        high_rmse=float(np.sqrt(np.mean((result['prediction'][high]-y[high])**2))))
                    metrics.append(row)
                    np.savez_compressed(target/'outer_predictions.npz', sample_ids=query_ids, target=y,
                                        prediction=result['prediction'], high=high)
                    for i,sid in enumerate(query_ids):
                        points.append(dict(sample_id=str(sid), region=region, fold=fold, arm=arm, method=method,
                            target=float(y[i]), prediction=float(result['prediction'][i]), squared_error=float((result['prediction'][i]-y[i])**2),
                            high=bool(high[i]), missing_features=int(cached['missing_mask'][n+i].sum()),
                            max_abs_z=float(np.abs(qx[i]).max())))
                    for sid, kept in zip(bundle['source_ids'], keep):
                        memberships.append(dict(sample_id=str(sid),region=region,fold=fold,arm=arm,method=method,retained=bool(kept)))
                    if method != 'full':
                        with np.load(target/'prefix_masks.npz') as prefixes:
                            for cap in PREFIXES:
                                mask = prefixes[str(cap)]
                                pred = predictor.predict(mask, qx, query_ids)['prediction']
                                curve_rows.append(dict(region=region,fold=fold,arm=arm,method=method,cap=cap,
                                    **r.metrics(y,pred),cr_percent=100*(1-mask.mean())))
                    del physical
                    cleanup()
                del predictor, bundle
                cleanup()
    assert len(metrics)==180 and len(restores)==180
    frame = pd.DataFrame(metrics)
    numeric = ['rmse','mae','r2','bias','high_rmse','cr_percent','retained']
    main_rows = frame[~frame.method.str.startswith('random-')]
    random = frame[frame.method.str.startswith('random-')].groupby(['region','fold','arm'])[numeric].mean().reset_index()
    random['method'] = 'random_mean'
    primary = pd.concat((main_rows, random),ignore_index=True)
    means = primary.groupby(['arm','method'])[numeric].mean().reset_index()
    wide = primary.pivot(index=['region','fold'],columns=['arm','method'],values='rmse')
    cr = primary.pivot(index=['region','fold'],columns=['arm','method'],values='cr_percent')
    effects = dict(RP_full_gain_percent=float((100*(1-wide['RP','full']/wide['G0','full'])).mean()))
    for arm in b.ARMS:
        effects[arm+'_agent_vs_score_RMSE_gain_percent'] = float((100*(1-wide[arm,'agent']/wide[arm,'score'])).mean())
        effects[arm+'_agent_vs_random_RMSE_gain_percent'] = float((100*(1-wide[arm,'agent']/wide[arm,'random_mean'])).mean())
        effects[arm+'_agent_vs_random_CR_pp'] = float((cr[arm,'agent']-cr[arm,'random_mean']).mean())
        effects[arm+'_agent_vs_full_RMSE_gain_percent'] = float((100*(1-wide[arm,'agent']/wide[arm,'full'])).mean())
    effects['CR_interaction_pp'] = float(((cr['RP','agent']-cr['RP','score'])-(cr['G0','agent']-cr['G0','score'])).mean())
    pd.DataFrame(points).to_csv(output/'all_query_predictions.csv',index=False)
    pd.DataFrame(memberships).to_csv(output/'all_reference_memberships.csv',index=False)
    frame.to_csv(output/'all180_fold_metrics.csv',index=False)
    primary.to_csv(output/'primary120_fold_metrics.csv',index=False)
    means.to_csv(output/'method_means.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(output/'budget_curves.csv',index=False)
    pd.DataFrame(restores).to_csv(output/'physical_restore_checks.csv',index=False)
    assert len(set(row['sample_id'] for row in points)) == 1421
    report = dict(task_id=TASK, measurement_complete=True, methods=means.to_dict('records'), effects=effects,
        primary_groups=4, matched_random_controls=2, full_references=2, real_query_IDs=1421,
        physical_endpoints=180, seed_offset=0, random_draws_per_fold=3, full_pipeline_three_seeds=False,
        final_holdout_access=False, historical_scores_used=False)
    r.write_json(output/'summary.json',report)
    print(json.dumps(dict(event='analysis_complete', **report)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight_agents','run','analyze'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r.setup()
    {'preflight_agents':agent_preflight,'run':run,'analyze':analyze}[args.command](args.output)
