"""Learn variable-granularity RP reference edits, then refine using feedback."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import rp_fidelity_agent as f

r, v, old = f.r, f.v, f.old
latent_drift = f.latent_drift
TASK = 'CARBON-RP-AGENT-COARSE-001'
METHODS = ('main','shuffled','uniform','fine_only')
FRACTIONS = (.05,.1,.2)
GROUPS_PER_FRACTION = 2


class CoarseWorld(f.FidelityWorld):
    def __init__(self,engine,bundle,data,strength=1.,coarse=True):
        super().__init__(engine,bundle,data,strength)
        self.coarse = coarse
        seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])
        self.group_rng = np.random.default_rng(seed+430000)
        self.proposals_with_macro = 0

    def propose(self,keep,rng,phase,valid):
        singles,counts = old.FeedbackEnvironment.propose(self,keep,rng,phase,valid)
        actions = [(tuple([a]) if a>=0 else (),tuple([b]) if b>=0 else (),kind) for a,b,kind in singles]
        if self.coarse and phase==1 and keep.sum()>10:
            seen = set()
            retained = np.flatnonzero(keep)
            for fraction in FRACTIONS:
                target = min(max(2,round(self.n*fraction)),int(keep.sum())-9)
                for _ in range(GROUPS_PER_FRACTION):
                    remaining,deleted = counts.copy(),[]
                    for index in self.group_rng.permutation(retained):
                        if np.min(remaining[self.cover[:,index]]) > 1:
                            deleted.append(int(index)); remaining -= self.cover[:,index]
                            if len(deleted)==target:
                                break
                    group = tuple(sorted(deleted))
                    if len(group)>1 and group not in seen:
                        seen.add(group); actions.append((group,(),0))
            if seen:
                self.proposals_with_macro += 1
        return actions,counts

    def move(self,keep,action):
        deleted,restored,_ = action
        assert len(deleted)==len(set(deleted)) and len(restored)==len(set(restored))
        assert not set(deleted)&set(restored)
        trial = keep.copy()
        if deleted:
            assert trial[list(deleted)].all(); trial[list(deleted)] = False
        if restored:
            assert not trial[list(restored)].any(); trial[list(restored)] = True
        assert trial.sum()>=9 and self.cover[:,trial].any(1).all()
        return trial

    def features(self,actions,keep,counts,mse,phase,remaining,last_reward,best_utility,since_best):
        observation = self.observations[keep.tobytes()]
        dynamic = self.credit(observation,observation['graphs'],observation['packed_query_positions'])
        potential = self.credit(observation,self.original_graphs,self.original_gather)
        dynamic[~keep] = potential[~keep]
        dynamic = dynamic/self.credit_scale
        dz,tail = self.drift(keep)
        global_values = np.array([keep.mean(),mse/self.base_mse,phase,remaining,last_reward,
            best_utility,since_best,tail],dtype=np.float32)
        values = []
        for deleted,restored,kind in actions:
            deleted,restored = sorted(deleted),sorted(restored)
            left = self.nodes[deleted].mean(0) if deleted else np.zeros(38)
            maximum = self.nodes[deleted].max(0) if deleted else np.zeros(38)
            right = self.nodes[restored].mean(0) if restored else np.zeros(38)
            after = counts.copy()
            if deleted:
                after -= self.cover[:,deleted].sum(1)
            if restored:
                after += self.cover[:,restored].sum(1)
            geometry = np.array([(len(deleted)-len(restored))/self.n,after.min()/self.n])
            dl = dynamic[deleted].mean(0) if deleted else np.zeros(8)
            dr = dynamic[restored].mean(0) if restored else np.zeros(8)
            dl,dr = dl.copy(),dr.copy()
            dl[-1] = float(dz[deleted].max()) if deleted else 0.
            dr[-1] = float(dz[restored].max()) if restored else 0.
            values.append(np.r_[left,maximum,right,global_values,geometry,np.eye(4)[kind],dl,dr])
        result = np.asarray(values,dtype=np.float32)
        assert result.shape==(len(actions),v.FEATURES) and np.isfinite(result).all()
        return result,np.zeros(len(actions),dtype=np.float32)


def select(bundle,data,method,output,small=False):
    output.mkdir(parents=True,exist_ok=False)
    engine = v.CachedReference(bundle,'cuda')
    tick,call_start = time.perf_counter(),engine.calls
    env = CoarseWorld(engine,bundle,data,coarse=method!='fine_only')
    seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])+100000*v.ALPHA
    model = None if method=='uniform' else v.ValueAgent(seed+320000)
    initial = None if model is None else {key:value.clone() for key,value in model.state_dict().items()}
    learner = None if model is None else v.ReplayLearner(model,seed+320001,method=='shuffled')
    depths,train_steps,deploy_steps = ((0.,.2),(20,15),(5,15)) if small else (v.DEPTHS,v.TRAIN_STEPS,v.DEPLOY_STEPS)
    events,candidates = [],[]
    for i,depth in enumerate(depths):
        start = v.initial_mask(env,depth,np.random.default_rng(seed+400+i))
        error = env.evaluate(start)
        best,best_error,trace = f.episode(env,model,seed+500+i,train_steps,start,error,learner,method=='uniform')
        for row in trace:
            row.update(role='training',episode=i,initial_retained=int(start.sum()))
        events.extend(trace); candidates.append((best,best_error))
    frozen = None if model is None else {key:value.clone() for key,value in model.state_dict().items()}
    best,best_error,trace = f.episode(env,model,seed+900,deploy_steps,env.full,env.base_mse,uniform=method=='uniform')
    for row in trace:
        row.update(role='deployment',episode=len(depths))
    events.extend(trace); candidates.append((best,best_error))
    if method=='uniform':
        best,best_error = max(candidates,key=lambda pair:(env.utility(pair[0]),-pair[1]))
    assert model is None or all(torch.equal(value,model.state_dict()[key]) for key,value in frozen.items())
    cap = len(depths)*(1+sum(train_steps))+sum(deploy_steps)
    assert env.logical_queries<=cap and env.feasible(best_error) and env.utility(best)>=-1e-12
    observation = env.observations[best.tobytes()]
    dz,tail = env.drift(best)
    np.savez_compressed(output/'selection.npz',keep=best,source_ids=bundle['source_ids'],mse=best_error,
        training_prediction=observation['prediction'],latent_drift=dz,drift_tail=tail,utility=env.utility(best))
    engine.export(best,output/'reduced.pt')
    if model is not None:
        torch.save(dict(state_dict=model.state_dict(),seed=seed+320000),output/'policy.pt')
    history = [] if learner is None else learner.history
    change = 0. if model is None else max(float((value-initial[key]).abs().max()) for key,value in model.state_dict().items())
    macro = sum(len(row['action'][0])>1 or len(row['action'][1])>1 for row in events)
    assert method!='fine_only' or macro==env.proposals_with_macro==0
    report = dict(method=method,original=engine.n,retained=int(best.sum()),cr_percent=100*(1-best.mean()),
        training_RMSE_ratio=float(np.sqrt(best_error/env.base_mse)),drift_tail=tail,utility=env.utility(best),strength=1.,
        macro_actions=macro,proposals_with_macro=env.proposals_with_macro,
        deployment_macro_actions=sum(row['role']=='deployment' and len(row['action'][0])>1 for row in events),
        optimizer_updates=len(history),parameter_max_change=change,
        maximum_gradient_norm=max((row['gradient_norm'] for row in history),default=0.),
        logical_queries=env.logical_queries,logical_query_cap=cap,actual_new_feedback=env.real_queries,
        total_predictor_calls_including_setup=engine.calls-call_start,seconds=time.perf_counter()-tick,
        deployment_policy_weights_frozen=True,deployment_from_full_source=True,cached_training_mask_used=False,
        full_original_RP_unchanged=engine.unchanged_parameters(),parameters=0 if model is None else sum(p.numel() for p in model.parameters()),
        reduced_bundle_bytes=(output/'reduced.pt').stat().st_size)
    v.write(output/'trajectory.json',events); v.write(output/'learning.json',history); v.write(output/'summary.json',report)
    print(json.dumps({key:report[key] for key in ('method','cr_percent','training_RMSE_ratio','drift_tail','macro_actions','seconds')}),flush=True)
    return report


def protected_inputs():
    return {**f.protected_inputs(),str(Path(__file__).resolve()):r.digest(Path(__file__))}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    protected, checks = protected_inputs(), []
    for region in r.REGIONS:
        full, data = v.original_bundle(region, 1)
        native = old.load_engine(full)
        base = native.predict(np.ones(native.n, dtype=bool), full['feedback_x'], full['feedback_ids'])
        cached = v.CachedReference(full, 'cuda')
        probe = CoarseWorld(cached, full, data, 1.)
        probe_keep = v.initial_mask(probe, .2, np.random.default_rng(9101))
        assert not probe_keep.all()
        probe_mse = probe.evaluate(probe_keep)
        probe_prediction = native.predict(probe_keep, full['feedback_x'], full['feedback_ids'])
        probe_native_mse = np.mean((probe_prediction['scaled'].astype(float)-full['feedback_y_scaled'])**2)
        probe_dz, probe_tail = latent_drift(probe_prediction['embedding'], base['embedding'])
        assert abs(probe_mse-probe_native_mse) <= 3e-6
        assert np.max(np.abs(probe_dz-probe.drift(probe_keep)[0])) <= 3e-5 and probe_tail > 0
        (output/region).mkdir()
        native.export(probe_keep, output/region/'probe-reduced.pt')
        probe_restored = old.ReferencePredictor.restore(output/region/'probe-reduced.pt', 'cuda')
        probe_again = probe_restored.predict(np.ones(probe_restored.n,dtype=bool), full['feedback_x'], full['feedback_ids'])
        assert np.max(np.abs(probe_again['scaled']-probe_prediction['scaled'])) <= 3e-6
        probe_actions, counts = probe.propose(probe_keep, np.random.default_rng(9102), 1, True)
        probe_features = probe.features(probe_actions,probe_keep,counts,probe_mse,1,1.,0.,0.,0.)[0]
        macro = next(action for action in probe_actions if len(action[0])>1)
        changed = probe.move(probe_keep,macro)
        macro_mse = probe.evaluate(changed)
        macro_native = native.predict(changed,full['feedback_x'],full['feedback_ids'])
        expected_mse = np.mean((macro_native['scaled'].astype(float)-full['feedback_y_scaled'])**2)
        assert abs(macro_mse-expected_mse)<=3e-6
        permuted = (tuple(reversed(macro[0])),macro[1],macro[2])
        original_features = probe.features([macro],probe_keep,counts,probe_mse,1,1.,0.,0.,0.)[0]
        permuted_features = probe.features([permuted],probe_keep,counts,probe_mse,1,1.,0.,0.,0.)[0]
        assert np.array_equal(original_features,permuted_features)
        assert np.max(np.abs(probe_features[:,121]-probe_tail)) <= 3e-5
        checks.append(dict(region=region,method='nontrivial_probe',retained=int(probe_keep.sum()),
            native_mse_delta=float(abs(probe_mse-probe_native_mse)),drift_tail=probe_tail,
            drift_delta=float(np.max(np.abs(probe_dz-probe.drift(probe_keep)[0])))))
        del cached, probe, probe_restored
        for method in METHODS:
            if region != 'hubei' and method != 'main':
                continue
            folder = output/region/method
            info = select(full, data, method, folder, small=True)
            with np.load(folder/'selection.npz') as saved:
                keep = saved['keep'].copy(); risk = float(saved['mse']); drift = saved['latent_drift'].copy()
            measured = native.predict(keep, full['feedback_x'], full['feedback_ids'])
            native_mse = np.mean((measured['scaled'].astype(float)-full['feedback_y_scaled'])**2)
            dz, tail = latent_drift(measured['embedding'], base['embedding'])
            assert abs(native_mse-risk) <= 3e-6 and np.max(np.abs(dz-drift)) <= 3e-5
            restored = old.ReferencePredictor.restore(folder/'reduced.pt', 'cuda')
            again = restored.predict(np.ones(restored.n, dtype=bool), full['feedback_x'], full['feedback_ids'])
            delta = float(np.max(np.abs(again['scaled']-measured['scaled'])))
            assert delta <= 3e-6
            assert abs(1-keep.mean()-info['strength']*tail-info['utility']) <= 3e-5
            assert method == 'uniform' or info['optimizer_updates'] > 0 and info['parameter_max_change'] > 0
            checks.append(dict(region=region, method=method, mse_delta=float(abs(native_mse-risk)),
                drift_delta=float(np.max(np.abs(dz-drift))), restore_delta=delta,
                utility_delta=float(abs(1-keep.mean()-info['strength']*tail-info['utility']))))
            del restored
        assert native.unchanged_parameters()
        del native
    assert protected_inputs() == protected
    v.write(output/'preflight.json', dict(valid=True, checks=checks, protected_inputs=protected,
        feedback_training_label_contact=True, final_holdout_access=False))

def analyze(output):
    assert (output/'selections_frozen.json').exists()
    rows, points, members = [], [], []
    for region in r.REGIONS:
        for fold in range(1, 6):
            dest = output/region/f'fold-{fold:02d}'
            full = torch.load(dest/'reference.pt', map_location='cpu', weights_only=False)
            _, _, data, _ = r.load_fold(region, fold)
            n = len(full['source_ids']); y, ids = data['y'][n:], data['sample_ids'][n:]
            x = old.transform(data['raw_features'][n:], full['preprocessor'])
            engine = old.load_engine(full)
            base = engine.predict(np.ones(n, dtype=bool), full['feedback_x'], full['feedback_ids'])
            predictions = {'RP': engine.predict(np.ones(n, dtype=bool), x, ids)['prediction']}
            high = y >= np.quantile(data['y'][:n], .8)
            for method, parent in (('RP', v.RP_ROOT), ('G0', r.HERE/'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01')):
                with np.load(parent/region/f'fold-{fold:02d}'/method/'responses.npz') as saved:
                    assert np.array_equal(saved['sample_ids'], data['sample_ids'])
                    if method == 'RP':
                        assert np.max(np.abs((predictions['RP']-saved['prediction'][n:])/full['preprocessor']['y_scale'])) <= 3e-6
                    else:
                        predictions[method] = saved['prediction'][n:].copy()
            for method in METHODS:
                folder = dest/method
                info = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                with np.load(folder/'selection.npz') as saved:
                    keep = saved['keep'].copy(); stored_mse = float(saved['mse']); stored_drift = float(saved['drift_tail'])
                train = engine.predict(keep, full['feedback_x'], full['feedback_ids'])
                mse = float(np.mean((train['scaled'].astype(float)-full['feedback_y_scaled'])**2))
                drift, tail = latent_drift(train['embedding'], base['embedding'])
                assert abs(mse-stored_mse) <= 3e-6 and abs(tail-stored_drift) <= 3e-5
                assert mse <= full['feedback_mse']+3e-6
                assert abs(1-keep.mean()-info['strength']*tail-info['utility']) <= 3e-5
                predictions[method] = engine.predict(keep, x, ids)['prediction']
                np.savez_compressed(folder/'outer_predictions.npz', sample_ids=ids, target=y, prediction=predictions[method])
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]),
                    **{key: info[key] for key in ('cr_percent','retained','original','parameter_max_change','optimizer_updates','seconds','drift_tail','utility','macro_actions','deployment_macro_actions')},
                    native_training_mse=mse, high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
                for i, sid in enumerate(full['source_ids']):
                    members.append(dict(region=region, fold=fold, method=method, sample_id=str(sid),
                        target=float(data['y'][i]), retained=bool(keep[i]), latent_drift=float(drift[i]),
                        missing_count=int(data['missing_mask'][i].sum())))
            for method in ('RP', 'G0'):
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]), cr_percent=0.,
                    high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
            names = np.asarray(data['feature_names']).astype(str)
            for i, sid in enumerate(ids):
                extreme = int(np.abs(x[i]).argmax())
                points.append(dict(region=region, fold=fold, sample_id=str(sid), target=float(y[i]), high=bool(high[i]),
                    missing_count=int(data['missing_mask'][n+i].sum()), extreme_feature=str(names[extreme]), extreme_z=float(x[i,extreme]),
                    **{method: float(value[i]) for method,value in predictions.items()},
                    main_minus_RP_squared_error=float((predictions['main'][i]-y[i])**2-(predictions['RP'][i]-y[i])**2)))
            assert engine.unchanged_parameters()
            del engine
    import pandas as pd
    frame = pd.DataFrame(rows)
    assert len(points) == len({row['sample_id'] for row in points}) == 1421 and len(frame) == 90
    rmse = frame.pivot(index=['region','fold'], columns='method', values='rmse')
    cr = frame.pivot(index=['region','fold'], columns='method', values='cr_percent')
    high = frame.pivot(index=['region','fold'], columns='method', values='high_rmse')
    reduction = {method: float((100*(1-rmse.main/rmse[method])).mean()) for method in ('RP','G0','uniform','shuffled','fine_only')}
    province = {region: float((100*(1-rmse.loc[region,'main']/rmse.loc[region,'RP'])).mean()) for region in r.REGIONS}
    main = frame[frame.method == 'main']
    checks = dict(accuracy=reduction['RP'] >= 1, compression=bool(main.cr_percent.mean() >= 20),
        uniform_CR=float((cr.main-cr.uniform).mean()) >= 1, uniform_accuracy=reduction['uniform'] >= -.1,
        shuffled_CR=float((cr.main-cr.shuffled).mean()) > 0, shuffled_accuracy=reduction['shuffled'] >= -.1,
        province_noninferior=min(province.values()) >= -1,
        high_noninferior=float((100*(1-high.main/high.RP)).mean()) >= -1,
        active_policy=int((main.parameter_max_change > 1e-10).sum()) >= 10,
        macro_active=int((main.deployment_macro_actions > 0).sum()) >= 10,
        fine_only_CR=float((cr.main-cr.fine_only).mean()) >= 1,
        fine_only_accuracy=reduction['fine_only'] >= -.1)
    full_gain = all(checks.values())
    pareto = (not full_gain and reduction['RP'] >= .5 and float(main.cr_percent.mean()) >= 10
        and not (reduction['uniform'] < -.1 and float((cr.main-cr.uniform).mean()) < -1)
        and checks['province_noninferior'] and checks['high_noninferior'] and checks['active_policy'] and checks['macro_active'])
    r.write_csv(output/'all_outer_fold_metrics.csv', rows)
    r.write_csv(output/'all1421_query_diagnosis.csv', points)
    r.write_csv(output/'all_training_reference_memberships.csv', members)
    report = dict(task_id=TASK, gain_class='FULL_GAIN' if full_gain else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        decision='PROMOTE' if full_gain else 'RETAIN' if pareto else 'REJECT', checks=checks,
        aggregate=frame.groupby('method')[['rmse','mae','r2','bias','cr_percent','high_rmse']].mean().to_dict('index'),
        main_rmse_reduction_percent=reduction, province_reduction_percent=province,
        uniform_CR_advantage_pp=float((cr.main-cr.uniform).mean()), contexts=15, query_ids=1421,
        first_innovation_parameters_unchanged=True, feedback_is_independent_validation=False,
        original_RP_historical_outer_selection=True, final_holdout_access=False, additional_seeds=False)
    v.write(output/'summary.json', report)
    print(json.dumps(report), flush=True)

def run(output, preflight_dir):
    text = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    assert json.loads(text.split('<!-- TASK_META',1)[1].split('TASK_META -->',1)[0])['status'] == 'RUNNING'
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    protected = protected_inputs()
    assert check['valid'] and check['protected_inputs'] == protected
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    started, tick = old.now(), time.perf_counter()
    v.write(output/'manifest.json', dict(task_id=TASK, started_at=started, protected_hashes=protected,
        methods=METHODS, alpha=v.ALPHA, world_feedback_cap=640, original_RP_training_labels=True,
        internal_training_target_RMSE_ratio=1., representation_tail_fraction=.2, main_fidelity_strength=1., macro_fractions=FRACTIONS, groups_per_fraction=GROUPS_PER_FRACTION,
        original_RP_refit=False, outer_label_reward=False, final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1,6):
            assert time.perf_counter()-tick < 90*60
            destination = output/region/f'fold-{fold:02d}'; destination.mkdir(parents=True)
            bundle, data = v.original_bundle(region, fold)
            torch.save(bundle, destination/'reference.pt')
            print(json.dumps(dict(region=region,fold=fold,stage='original_RP_ready')),flush=True)
            for method in METHODS:
                select(bundle, data, method, destination/method)
            v.write(destination/'finished.json',dict(time=old.now(),methods=4))
    v.write(output/'selections_frozen.json',dict(time=old.now(),contexts=15,methods=4))
    analyze(output)
    assert protected_inputs() == protected
    v.write(output/'finished.json',dict(started_at=started,finished_at=old.now(),wall_seconds=time.perf_counter()-tick,
        policy_fits=45,new_RP_fits=0,protected_inputs_unchanged=True,final_holdout_access=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight','run','analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output,args.preflight_dir)
    else:
        analyze(args.output)
