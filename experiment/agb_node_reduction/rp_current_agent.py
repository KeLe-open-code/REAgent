"""Feed current native RP node representations to the sequential editing agent."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import rp_stop_agent as s

r,v,old,f=s.r,s.v,s.old,s.f
MessageReference,StopAgent=s.MessageReference,s.StopAgent
latent_drift,native_message=s.latent_drift,s.native_message
FRACTIONS,GROUPS_PER_FRACTION=s.FRACTIONS,s.GROUPS_PER_FRACTION
TASK='CARBON-RP-AGENT-CURRENT-001'
METHODS=('main','shuffled','uniform','static_state')
PARENT_ROOT=r.HERE/'runs/CARBON-RP-AGENT-STOP-001'
CONTROL_FILES=('selection.npz','components.npz','reduced.pt','trajectory.json','learning.json','summary.json')


class CurrentWorld(s.StopWorld):
    def __init__(self,engine,bundle,data,current=True):
        super().__init__(engine,bundle,data,message_objective=True)
        self.current=current
        full=self.observations[self.full.tobytes()]
        sy=(data['y'][bundle['source_indices']]-bundle['preprocessor']['y_mean'])/bundle['preprocessor']['y_scale']
        missing=data['missing_mask'][bundle['source_indices']].mean(1)
        self.node_base=np.column_stack((full['embedding'],full['scaled'],sy,full['scaled']-sy,
            missing,np.log1p(self.cover.sum(1)),np.ones(self.n))).astype(np.float32)
        self.node_mean=self.node_base.mean(0)
        self.node_scale=self.node_base.std(0);self.node_scale[self.node_scale<1e-6]=1
        self.node_target=sy
        assert np.max(np.abs((self.node_base-self.node_mean)/self.node_scale-self.nodes))<=1e-6

    def current_nodes(self,keep):
        observation=self.observations[keep.tobytes()]
        nodes=self.node_base.copy()
        nodes[:,:32]=observation['embedding']
        nodes[:,32]=observation['scaled']
        nodes[:,34]=observation['scaled']-self.node_target
        return (nodes-self.node_mean)/self.node_scale

    def features(self,actions,keep,counts,mse,phase,remaining,last_reward,best_utility,since_best):
        previous=self.nodes
        if self.current:
            self.nodes=self.current_nodes(keep)
        try:
            return super().features(actions,keep,counts,mse,phase,remaining,last_reward,best_utility,since_best)
        finally:
            self.nodes=previous


def protected_inputs():
    paths=[PARENT_ROOT/'completion_record.json']
    paths += [path/name for path in sorted((PARENT_ROOT/'attempt-01').glob('*/fold-*/uniform')) for name in CONTROL_FILES]
    paths += [path/name for path in sorted((PARENT_ROOT/'attempt-01').glob('*/fold-*/main')) for name in (*CONTROL_FILES,'policy.pt')]
    assert len(paths)==196
    return {**s.protected_inputs(),str(Path(__file__).resolve()):r.digest(Path(__file__)),
        **{str(path.resolve()):r.digest(path) for path in paths}}


def reuse_control(bundle,method,output):
    parent_method='uniform' if method=='uniform' else 'main'
    source=PARENT_ROOT/'attempt-01'/bundle['config']['region']/output.parent.name/parent_method
    assert json.loads((PARENT_ROOT/'completion_record.json').read_text(encoding='utf-8'))['status']=='NO_GAIN'
    output.mkdir(parents=True,exist_ok=False)
    names=CONTROL_FILES if method=='uniform' else (*CONTROL_FILES,'policy.pt')
    for name in names:
        if name!='summary.json':
            shutil.copy2(source/name,output/name)
    info=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    info.update(method=method,current_node_state=False,reused_control=True,reused_from=str(source),new_control_selection_seconds=0.)
    v.write(output/'summary.json',info)
    v.write(output/'reuse_receipt.json',{str((source/name).resolve()):r.digest(source/name) for name in names})
    print(json.dumps(dict(method=method,reused_matched_control=True,cr_percent=info['cr_percent'])),flush=True)
    return info


def select(bundle,data,method,output,small=False):
    if method in ('uniform','static_state') and not small:
        return reuse_control(bundle,method,output)
    output.mkdir(parents=True,exist_ok=False)
    engine = MessageReference(bundle,'cuda')
    tick,call_start = time.perf_counter(),engine.calls
    env = CurrentWorld(engine,bundle,data,current=method!='static_state')
    seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])+100000*v.ALPHA
    model = None if method=='uniform' else StopAgent(seed+320000,anchored=True)
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
    dm,tm=env.message_drift(best)
    d2,t2=f.FidelityWorld.drift(env,best)
    np.savez_compressed(output/'components.npz',message_drift=dm,layer2_drift=d2,message_tail=tm,layer2_tail=t2)
    engine.export(best,output/'reduced.pt')
    if model is not None:
        torch.save(dict(state_dict=model.state_dict(),seed=seed+320000,anchored=True),output/'policy.pt')
    history = [] if learner is None else learner.history
    change = 0. if model is None else max(float((value-initial[key]).abs().max()) for key,value in model.state_dict().items())
    macro = sum(len(row['action'][0])>1 or len(row['action'][1])>1 for row in events)
    report = dict(stop_always_available=True,anchored_critic=True,current_node_state=env.current,method=method,original=engine.n,retained=int(best.sum()),cr_percent=100*(1-best.mean()),
        training_RMSE_ratio=float(np.sqrt(best_error/env.base_mse)),drift_tail=tail,utility=env.utility(best),strength=1.,message_objective=env.message_objective,message_tail=tm,layer2_tail=t2,
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

def preflight(output):
    output.mkdir(parents=True,exist_ok=False)
    protected,checks=protected_inputs(),[]
    for region in r.REGIONS:
        bundle,data=v.original_bundle(region,1)
        native=old.load_engine(bundle)
        full=np.ones(native.n,dtype=bool)
        base=native.predict(full,bundle['feedback_x'],bundle['feedback_ids'])
        base_message=native_message(native,full,bundle['feedback_x'],bundle['feedback_ids'])
        engine=MessageReference(bundle,'cuda')
        world=CurrentWorld(engine,bundle,data)
        keep=v.initial_mask(world,.2,np.random.default_rng(9101))
        assert not keep.all()
        mse=world.evaluate(keep)
        measured=native.predict(keep,bundle['feedback_x'],bundle['feedback_ids'])
        message=native_message(native,keep,bundle['feedback_x'],bundle['feedback_ids'])
        dm,tm=latent_drift(message,base_message)
        d2,t2=latent_drift(measured['embedding'],base['embedding'])
        joint=np.maximum(dm,d2)
        tail=float(np.sort(joint)[-max(1,int(np.ceil(.2*len(keep)))):].mean())
        assert abs(mse-np.mean((measured['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))<=3e-6
        assert np.max(np.abs(message-world.observations[keep.tobytes()]['message']))<=3e-6
        assert np.max(np.abs(joint-world.drift(keep)[0]))<=3e-5
        actions,counts=world.propose(keep,np.random.default_rng(9102),1,True)
        assert any(action[2]==3 for action in actions)
        exact_stop=world.features([((),(),3)],keep,counts,mse,1,1.,0.,0.,0.)[0]
        all_features=world.features(actions,keep,counts,mse,1,1.,0.,0.,0.)[0]
        assert np.array_equal(StopAgent.stop_input(torch.from_numpy(all_features)).numpy(),np.repeat(exact_stop,len(actions),axis=0))
        macro=next(action for action in actions if len(action[0])>1)
        trial=world.move(keep,macro)
        trial_mse=world.evaluate(trial)
        exact=native.predict(trial,bundle['feedback_x'],bundle['feedback_ids'])
        assert abs(trial_mse-np.mean((exact['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))<=3e-6
        features=world.features(actions,keep,counts,mse,1,1.,0.,0.,0.)[0]
        world.current=False
        static_features=world.features(actions,keep,counts,mse,1,1.,0.,0.,0.)[0]
        world.current=True
        feature_change=float(np.max(np.abs(features[:,:114]-static_features[:,:114])))
        assert feature_change>1e-7
        assert np.max(np.abs(world.current_nodes(world.full)-world.nodes))<=1e-6
        assert np.max(np.abs(features[:,118]-tm))<=3e-5 and np.max(np.abs(features[:,121]-t2))<=3e-5
        permuted=(tuple(reversed(macro[0])),macro[1],macro[2])
        assert np.array_equal(world.features([macro],keep,counts,mse,1,1.,0.,0.,0.)[0],
            world.features([permuted],keep,counts,mse,1,1.,0.,0.,0.)[0])
        checks.append(dict(region=region,method='nontrivial_probe',retained=int(keep.sum()),
            message_tail=tm,layer2_tail=t2,joint_tail=tail,macro_cardinality=len(macro[0]),current_feature_change=feature_change))
        del world,engine
        for method in METHODS:
            if region!='hubei' and method!='main':
                continue
            folder=output/region/method
            info=select(bundle,data,method,folder,small=True)
            with np.load(folder/'selection.npz') as saved:
                keep=saved['keep'].copy(); stored_mse=float(saved['mse']); stored_drift=saved['latent_drift'].copy()
            measured=native.predict(keep,bundle['feedback_x'],bundle['feedback_ids'])
            dm,tm=latent_drift(native_message(native,keep,bundle['feedback_x'],bundle['feedback_ids']),base_message)
            d2,t2=latent_drift(measured['embedding'],base['embedding'])
            joint=np.maximum(dm,d2) if info['message_objective'] else d2
            tail=float(np.sort(joint)[-max(1,int(np.ceil(.2*len(keep)))):].mean())
            mse=float(np.mean((measured['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))
            assert abs(mse-stored_mse)<=3e-6 and np.max(np.abs(joint-stored_drift))<=3e-5
            assert abs(1-keep.mean()-tail-info['utility'])<=3e-5
            restored=old.ReferencePredictor.restore(folder/'reduced.pt','cuda')
            again=restored.predict(np.ones(restored.n,dtype=bool),bundle['feedback_x'],bundle['feedback_ids'])
            delta=float(np.max(np.abs(again['scaled']-measured['scaled'])))
            assert delta<=3e-6 and restored.n==keep.sum()
            assert method=='uniform' or info['optimizer_updates']>0 and info['parameter_max_change']>0
            if method!='uniform':
                policy=torch.load(folder/'policy.pt',map_location='cpu',weights_only=False)
                fitted=StopAgent(policy['seed']);fitted.load_state_dict(policy['state_dict'])
                assert float(fitted(torch.from_numpy(exact_stop))[0].detach())==0.
            checks.append(dict(region=region,method=method,mse_delta=abs(mse-stored_mse),
                drift_delta=float(np.max(np.abs(joint-stored_drift))),restore_delta=delta,
                optimizer_updates=info['optimizer_updates']))
            del restored
        for current_method,parent_method in (('uniform','uniform'),('static_state','main')):
            new_folder,prior_folder=output/region/(current_method+'-new'),output/region/(current_method+'-parent')
            select(bundle,data,current_method,new_folder,small=True)
            s.select(bundle,data,parent_method,prior_folder,small=True)
            with np.load(new_folder/'selection.npz') as new,np.load(prior_folder/'selection.npz') as prior:
                assert all(np.array_equal(new[key],prior[key]) for key in new.files)
            assert json.loads((new_folder/'trajectory.json').read_text())==json.loads((prior_folder/'trajectory.json').read_text())
            if current_method=='static_state':
                new=torch.load(new_folder/'policy.pt',map_location='cpu',weights_only=False)
                prior=torch.load(prior_folder/'policy.pt',map_location='cpu',weights_only=False)
                assert all(torch.equal(value,prior['state_dict'][key]) for key,value in new['state_dict'].items())
            checks.append(dict(region=region,method=current_method+'_reuse_compatibility',same_entire_small_trajectory=True))
        assert native.unchanged_parameters()
        del native
    assert protected_inputs()==protected
    v.write(output/'preflight.json',dict(valid=True,checks=checks,protected_inputs=protected,
        feedback_training_label_contact=True,final_holdout_access=False))

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
            baseline_message=native_message(engine,np.ones(n,dtype=bool),full['feedback_x'],full['feedback_ids'])
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
                d2,t2=latent_drift(train['embedding'],base['embedding'])
                dm,tm=latent_drift(native_message(engine,keep,full['feedback_x'],full['feedback_ids']),baseline_message)
                drift=np.maximum(dm,d2) if info['message_objective'] else d2
                tail=float(np.sort(drift)[-max(1,int(np.ceil(.2*n))):].mean())
                with np.load(folder/'components.npz') as components:
                    assert np.max(np.abs(components['message_drift']-dm))<=3e-5
                    assert np.max(np.abs(components['layer2_drift']-d2))<=3e-5
                with np.load(folder/'selection.npz') as saved:
                    assert np.array_equal(saved['source_ids'],full['source_ids'])
                    assert np.max(np.abs(saved['latent_drift']-drift))<=3e-5
                assert abs(mse-stored_mse) <= 3e-6 and abs(tail-stored_drift) <= 3e-5
                assert mse <= full['feedback_mse']+3e-6
                assert abs(1-keep.mean()-info['strength']*tail-info['utility']) <= 3e-5
                predictions[method] = engine.predict(keep, x, ids)['prediction']
                np.savez_compressed(folder/'outer_predictions.npz', sample_ids=ids, target=y, prediction=predictions[method])
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]),
                    **{key: info[key] for key in ('cr_percent','retained','original','parameter_max_change','optimizer_updates','seconds','drift_tail','utility','macro_actions','deployment_macro_actions','message_tail','layer2_tail')},
                    native_training_mse=mse, high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
                for i, sid in enumerate(full['source_ids']):
                    members.append(dict(region=region, fold=fold, method=method, sample_id=str(sid),
                        target=float(data['y'][i]), retained=bool(keep[i]), latent_drift=float(drift[i]),message_drift=float(dm[i]),layer2_drift=float(d2[i]),
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
    reduction = {method: float((100*(1-rmse.main/rmse[method])).mean()) for method in ('RP','G0','uniform','shuffled','static_state')}
    province = {region: float((100*(1-rmse.loc[region,'main']/rmse.loc[region,'RP'])).mean()) for region in r.REGIONS}
    main = frame[frame.method == 'main']
    checks = dict(accuracy=reduction['RP'] >= 1, compression=bool(main.cr_percent.mean() >= 20),
        uniform_CR=float((cr.main-cr.uniform).mean()) >= 1, uniform_accuracy=reduction['uniform'] >= -.1,
        shuffled_CR=float((cr.main-cr.shuffled).mean()) > 0, shuffled_accuracy=reduction['shuffled'] >= -.1,
        province_noninferior=min(province.values()) >= -1,
        high_noninferior=float((100*(1-high.main/high.RP)).mean()) >= -1,
        active_policy=int((main.parameter_max_change > 1e-10).sum()) >= 10,
        macro_active=int((main.deployment_macro_actions > 0).sum()) >= 10,
        current_control_accuracy=reduction['static_state'] >= -.1)
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
        methods=METHODS, alpha=v.ALPHA, world_feedback_cap=640, original_RP_training_labels=True,current_node_observation=True,reused_uniform_source=str(PARENT_ROOT),stop_always_available=True,main_Q_parameterization='g(s,a)-g(s,STOP)',
        internal_training_target_RMSE_ratio=1., representation_tail_fraction=.2, main_fidelity_strength=1., message_cost='CVaR20_pointwise_max_first_message_second_embedding',message_objective_weight=1.,macro_fractions=FRACTIONS, groups_per_fraction=GROUPS_PER_FRACTION,
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
        policy_fits=30,reused_policy_fits=15,new_RP_fits=0,reused_uniform_controls=15,protected_inputs_unchanged=True,final_holdout_access=False))
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','run','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args(); r.setup()
    if args.mode=='preflight':
        preflight(args.output)
    elif args.mode=='run':
        run(args.output,args.preflight_dir)
    else:
        analyze(args.output)
