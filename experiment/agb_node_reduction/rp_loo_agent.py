"""Genuine sequential RP editing with target-isolated training feedback."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import rp_conservative_agent as p
from rp_loo_reference import LeaveOneOutReference
from rp_agent_reference import ReferencePredictor

r,v,old,f=p.r,p.v,p.old,p.f
MessageReference,StopAgent,ReplayLearner=p.MessageReference,p.StopAgent,p.ReplayLearner
episode,latent_drift,native_message=p.episode,p.latent_drift,p.native_message
TASK='CARBON-RP-AGENT-LOO-001'
METHODS=('main','shuffled','uniform','native_feedback')
FRACTIONS,GROUPS_PER_FRACTION=p.FRACTIONS,p.GROUPS_PER_FRACTION


class MinimumWorld(p.StopWorld):
    def propose(self,keep,rng,phase,valid):
        actions,counts=super().propose(keep,rng,phase,valid)
        return [a for a in actions if keep.sum()-len(a[0])+len(a[1])>=10],counts


def protected_inputs():
    paths=[Path(__file__).resolve(),r.HERE/'rp_loo_reference.py',
        r.HERE/'runs/CARBON-RP-AGENT-EGO-AUDIT-001/completion_record.json']
    return {**p.protected_inputs(),**{str(x.resolve()):r.digest(x) for x in paths}}


def prepare_bundle(region,fold):
    bundle,data=v.original_bundle(region,fold)
    engine=LeaveOneOutReference(bundle,'cuda')
    full=engine.predict(np.ones(engine.n,dtype=bool),bundle['feedback_x'],bundle['feedback_ids'])
    bundle['loo_feedback_mse']=float(np.mean((full['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))
    bundle['loo_feedback_contract']='target i uses S minus i; original frozen RP already trained on all T labels'
    return bundle,data


def physical_oracle(native,keep,bundle,result,leave_out=True):
    retained=np.flatnonzero(keep);removed=np.flatnonzero(~keep)
    chosen=sorted(set([int(retained[0]),int(removed[0]) if len(removed) else len(keep)//2,len(keep)-1]))
    checks=[]
    for i in chosen:
        excluded=keep.copy()
        if leave_out:excluded[i]=False
        expected=native.predict(excluded,bundle['feedback_x'][i:i+1],bundle['feedback_ids'][i:i+1])
        deltas={name:float(np.max(np.abs(result[name][i]-expected[name][0]))) for name in ('scaled','embedding','message')}
        assert deltas['scaled']<=3e-6 and max(deltas.values())<=3e-5
        checks.append(dict(index=i,was_retained=bool(keep[i]),**deltas))
    return checks


def preflight(output):
    output.mkdir(parents=True,exist_ok=False);protected=protected_inputs();checks=[]
    for region in r.REGIONS:
        bundle,data=prepare_bundle(region,1);engine=LeaveOneOutReference(bundle,'cuda');native=MessageReference(bundle,'cuda')
        full=np.ones(engine.n,dtype=bool)
        for fraction in (0.,.2):
            cover,_=old.coverage(data['coords'][:engine.n],v.ALPHA)
            from types import SimpleNamespace
            keep=v.initial_mask(SimpleNamespace(n=engine.n,full=full,cover=cover),fraction,np.random.default_rng(9101))
            torch.cuda.synchronize();tick=time.perf_counter()
            result=engine.predict(keep,bundle['feedback_x'],bundle['feedback_ids'],graphs=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            oracle=physical_oracle(native,keep,bundle,result)
            for graph in result['graphs']:
                near=graph['neighbors'][result['packed_query_positions']]
                assert not np.any(near==np.arange(engine.n)[:,None])
                assert np.isin(near,np.flatnonzero(keep)).all()
            checks.append(dict(region=region,fraction=fraction,seconds=seconds,oracle=oracle))
        folder=output/region/'main';info=select(bundle,data,'main',folder,small=True)
        assert info['optimizer_updates']>0 and info['parameter_max_change']>1e-10 and info['regularizer_gradient_norm']>1e-10
        restored=ReferencePredictor.restore(folder/'reduced.pt','cuda')
        with np.load(folder/'selection.npz') as selected:
            keep=selected['keep'].copy()
        expected=native.predict(keep,bundle['feedback_x'],bundle['feedback_ids'])
        again=restored.predict(np.ones(restored.n,dtype=bool),bundle['feedback_x'],bundle['feedback_ids'])
        assert np.max(np.abs(expected['scaled']-again['scaled']))<=3e-6
        assert engine.unchanged_parameters() and native.unchanged_parameters()
        checks.append(dict(region=region,small_training=info,physical_restore=True))
        print(json.dumps(dict(region=region,preflight_complete=True)),flush=True)
        del engine,native,restored
    assert protected_inputs()==protected
    v.write(output/'preflight.json',dict(valid=True,checks=checks,protected_inputs=protected,
        feedback_labels_already_seen_by_RP=True,final_holdout_access=False))


def run(output,preflight_dir):
    task=(r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    assert json.loads(task.split('<!-- TASK_META',1)[1].split('TASK_META -->',1)[0])['status']=='RUNNING'
    protected=protected_inputs();check=json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert check['valid'] and check['protected_inputs']==protected
    output.mkdir(parents=True,exist_ok=False)
    for name in ('rp_loo_agent.py','rp_loo_reference.py'):
        shutil.copy2(r.HERE/name,output/name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md',output/'task_frozen.md')
    started,tick=old.now(),time.perf_counter()
    v.write(output/'manifest.json',dict(task_id=TASK,started_at=started,protected_hashes=protected,
        methods=METHODS,world_feedback_cap=640,original_RP_training_labels=True,conservative_weight=1.,
        minimum_references=10,loo_batch_size=128,feedback_targets='allT with per-target S minus i',
        original_RP_refit=False,outer_label_reward=False,final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1,6):
            assert time.perf_counter()-tick<120*60
            destination=output/region/f'fold-{fold:02d}';destination.mkdir(parents=True)
            bundle,data=prepare_bundle(region,fold);torch.save(bundle,destination/'reference.pt')
            print(json.dumps(dict(region=region,fold=fold,stage='original_RP_and_LOO_baseline_ready')),flush=True)
            for method in METHODS:
                select(bundle,data,method,destination/method)
            v.write(destination/'finished.json',dict(time=old.now(),methods=4))
    v.write(output/'selections_frozen.json',dict(time=old.now(),contexts=15,methods=4))
    analyze(output)
    assert protected_inputs()==protected
    v.write(output/'finished.json',dict(started_at=started,finished_at=old.now(),wall_seconds=time.perf_counter()-tick,
        policy_fits=45,uniform_searches=15,new_RP_fits=0,protected_inputs_unchanged=True,final_holdout_access=False))


def select(bundle,data,method,output,small=False):
    output.mkdir(parents=True,exist_ok=False)
    engine = (MessageReference if method=='native_feedback' else LeaveOneOutReference)(bundle,'cuda')
    if method!='native_feedback':
        bundle={**bundle,'feedback_mse':bundle['loo_feedback_mse']}
    tick,call_start = time.perf_counter(),engine.calls
    env = MinimumWorld(engine,bundle,data,message_objective=True)
    seed = int(bundle['config']['seed'])+10000*r.REGIONS.index(bundle['config']['region'])+100000*v.ALPHA
    model = None if method=='uniform' else StopAgent(seed+320000,anchored=True)
    initial = None if model is None else {key:value.clone() for key,value in model.state_dict().items()}
    learner = None if model is None else ReplayLearner(model,seed+320001,method=='shuffled',1.)
    depths,train_steps,deploy_steps = ((0.,.2),(20,15),(5,15)) if small else (v.DEPTHS,v.TRAIN_STEPS,v.DEPLOY_STEPS)
    events,candidates = [],[]
    for i,depth in enumerate(depths):
        start = v.initial_mask(env,depth,np.random.default_rng(seed+400+i))
        error = env.evaluate(start)
        best,best_error,trace = episode(env,model,seed+500+i,train_steps,start,error,learner,method=='uniform')
        for row in trace:
            row.update(role='training',episode=i,initial_retained=int(start.sum()))
        events.extend(trace); candidates.append((best,best_error))
    frozen = None if model is None else {key:value.clone() for key,value in model.state_dict().items()}
    best,best_error,trace = episode(env,model,seed+900,deploy_steps,env.full,env.base_mse,uniform=method=='uniform')
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
    report = dict(feedback_mode='native' if method=='native_feedback' else 'leave_one_reference_out',feedback_base_mse=env.base_mse,leave_one_out_cases=getattr(engine,'leave_one_out_cases',0),conservative_weight=0. if learner is None else learner.weight,
        regularizer_gradient_norm=max((row['regularizer_gradient_norm'] for row in history),default=0.),
        mean_conservative_penalty=float(np.mean([row['conservative_penalty'] for row in history])) if history else 0.,stop_always_available=True,anchored_critic=True,method=method,original=engine.n,retained=int(best.sum()),cr_percent=100*(1-best.mean()),
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
            native_feedback=MessageReference(full,'cuda');loo_feedback=LeaveOneOutReference(full,'cuda')
            baselines={key:obj.predict(np.ones(n,dtype=bool),full['feedback_x'],full['feedback_ids']) for key,obj in [('native',native_feedback),('loo',loo_feedback)]}
            np.savez_compressed(dest/'feedback_baselines.npz',**{key+'_'+name:val[name] for key,val in baselines.items() for name in ('scaled','embedding','message')})
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
                key='native' if method=='native_feedback' else 'loo'
                feedback_engine=native_feedback if key=='native' else loo_feedback
                base=baselines[key];baseline_message=base['message']
                feedback_base=float(np.mean((base['scaled'].astype(float)-full['feedback_y_scaled'])**2))
                train=feedback_engine.predict(keep,full['feedback_x'],full['feedback_ids'])
                oracle=physical_oracle(native_feedback,keep,full,train,leave_out=key=='loo')
                v.write(folder/'feedback_oracle.json',oracle)
                mse = float(np.mean((train['scaled'].astype(float)-full['feedback_y_scaled'])**2))
                d2,t2=latent_drift(train['embedding'],base['embedding'])
                dm,tm=latent_drift(train['message'],baseline_message)
                drift=np.maximum(dm,d2) if info['message_objective'] else d2
                tail=float(np.sort(drift)[-max(1,int(np.ceil(.2*n))):].mean())
                with np.load(folder/'components.npz') as components:
                    assert np.max(np.abs(components['message_drift']-dm))<=3e-5
                    assert np.max(np.abs(components['layer2_drift']-d2))<=3e-5
                with np.load(folder/'selection.npz') as saved:
                    assert np.array_equal(saved['source_ids'],full['source_ids'])
                    assert np.max(np.abs(saved['latent_drift']-drift))<=3e-5
                assert abs(mse-stored_mse) <= 3e-6 and abs(tail-stored_drift) <= 3e-5
                assert mse <= feedback_base+3e-6
                assert abs(1-keep.mean()-info['strength']*tail-info['utility']) <= 3e-5
                predictions[method] = engine.predict(keep, x, ids)['prediction']
                np.savez_compressed(folder/'outer_predictions.npz', sample_ids=ids, target=y, prediction=predictions[method])
                rows.append(dict(region=region, fold=fold, method=method, **r.metrics(y, predictions[method]),
                    **{key: info[key] for key in ('cr_percent','retained','original','parameter_max_change','optimizer_updates','seconds','drift_tail','utility','macro_actions','deployment_macro_actions','message_tail','layer2_tail','regularizer_gradient_norm')},
                    feedback_training_mse=mse, high_rmse=float(np.sqrt(np.mean((predictions[method][high]-y[high])**2)))))
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
    reduction = {method: float((100*(1-rmse.main/rmse[method])).mean()) for method in ('RP','G0','uniform','shuffled','native_feedback')}
    province = {region: float((100*(1-rmse.loc[region,'main']/rmse.loc[region,'RP'])).mean()) for region in r.REGIONS}
    main = frame[frame.method == 'main']
    checks = dict(accuracy=reduction['RP'] >= 1, compression=bool(main.cr_percent.mean() >= 20),
        uniform_CR=float((cr.main-cr.uniform).mean()) >= 1, uniform_accuracy=reduction['uniform'] >= -.1,
        shuffled_CR=float((cr.main-cr.shuffled).mean()) > 0, shuffled_accuracy=reduction['shuffled'] >= -.1,
        province_noninferior=min(province.values()) >= -1,
        high_noninferior=float((100*(1-high.main/high.RP)).mean()) >= -1,
        active_policy=int((main.parameter_max_change > 1e-10).sum()) >= 10,
        conservative_active=int((main.regularizer_gradient_norm>1e-10).sum())>=10,
        macro_active=int((main.deployment_macro_actions > 0).sum()) >= 10,
        native_feedback_control_accuracy=reduction['native_feedback'] >= -.1)
    full_gain = all(checks.values())
    pareto = (not full_gain and reduction['RP'] >= .5 and float(main.cr_percent.mean()) >= 10
        and not (reduction['uniform'] < -.1 and float((cr.main-cr.uniform).mean()) < -1)
        and checks['province_noninferior'] and checks['high_noninferior'] and checks['active_policy'] and checks['macro_active'] and checks['conservative_active'])
    r.write_csv(output/'all_outer_fold_metrics.csv', rows)
    r.write_csv(output/'all1421_query_diagnosis.csv', points)
    r.write_csv(output/'all_training_reference_memberships.csv', members)
    report = dict(task_id=TASK, gain_class='FULL_GAIN' if full_gain else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        decision='PROMOTE' if full_gain else 'RETAIN' if pareto else 'REJECT', checks=checks,
        aggregate=frame.groupby('method')[['rmse','mae','r2','bias','cr_percent','high_rmse']].mean().to_dict('index'),
        main_rmse_reduction_percent=reduction, province_reduction_percent=province,
        uniform_CR_advantage_pp=float((cr.main-cr.uniform).mean()), contexts=15, query_ids=1421,
        first_innovation_parameters_unchanged=True,per_target_reference_exclusion=True,feedback_is_independent_validation=False,
        original_RP_historical_outer_selection=True, final_holdout_access=False, additional_seeds=False)
    v.write(output/'summary.json', report)
    print(json.dumps(report), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run','analyze'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    elif args.mode=='run':run(args.output,args.preflight_dir)
    else:analyze(args.output)
