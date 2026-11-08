"""Train-only symmetric kernel balancing before sparse inductive aggregation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import relation_graphsage as r
import relation_agop as a
import relation_local_egop as local

TASK='CARBON-RELATION-BALANCE-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-LOCAL-EGOP-001/attempt-01'
ARMS=('G0','GB','GS','GU','GE','GL')


def symmetric_balance(log_kernel,tolerance=1e-6,max_iterations=2000):
    assert log_kernel.ndim==2 and log_kernel.shape[0]==log_kernel.shape[1]
    assert torch.equal(log_kernel,log_kernel.T)
    z=-.5*torch.logsumexp(log_kernel,dim=1)
    for iteration in range(max_iterations+1):
        row_log_mass=z+torch.logsumexp(log_kernel+z[None,:],dim=1)
        residual=float(torch.expm1(row_log_mass).abs().max())
        if residual<=tolerance:return z,iteration,residual
        if iteration==max_iterations:break
        if iteration<100:
            logu=-torch.logsumexp(log_kernel+z[None,:],dim=1)
            logv=-torch.logsumexp(log_kernel+logu[None,:],dim=1)
            z=.5*(logu+logv)
        else:
            # Same convex scaling problem; Newton acceleration for nearly isolated pairs.
            p=torch.exp(log_kernel+z[:,None]+z[None,:]);row=p.sum(1);gradient=row-1
            hessian=torch.diag(row)+p+1e-10*torch.eye(len(z),device=z.device,dtype=z.dtype)
            step=torch.linalg.solve(hessian,-gradient);objective=.5*p.sum()-z.sum()
            directional=gradient@step
            for backtrack in range(30):
                amount=2.**(-backtrack);trial=z+amount*step
                value=.5*torch.exp(log_kernel+trial[:,None]+trial[None,:]).sum()-trial.sum()
                if bool(torch.isfinite(value)) and value<=objective+1e-4*amount*directional:break
            else:raise RuntimeError('Balancing Newton line search failed')
            z=trial
    raise RuntimeError(f'Symmetric balancing did not converge: residual={residual}, iterations={iteration}')


class BalancedEgopSAGE(local.LocalEgopSAGE):
    def __init__(self,arm,config):
        super().__init__(arm,config)
        self.register_buffer('balance_log_scaling',torch.zeros(config['relation_n_train'],dtype=torch.float64))
        self.register_buffer('balance_ready',torch.tensor(False))

    @torch.no_grad()
    def fit_reference(self,x,edges):
        super().fit_reference(x,edges)
        self.balance_ready.fill_(False);self.balance_log_scaling.zero_()

    @torch.no_grad()
    def fit_relations(self,x,target,edges):
        history=super().fit_relations(x,target,edges)
        distance,_=self.field_distance(self.coordinates(x))
        log_kernel=-a.symmetric(distance)/self.temperature;log_kernel.fill_diagonal_(-torch.inf)
        z,iterations,residual=symmetric_balance(log_kernel)
        self.balance_log_scaling.copy_(z);self.balance_ready.fill_(True);self._relation_cache.clear()
        history[0].update(balance_iterations=iterations,dense_marginal_residual=residual,
                          log_scaling_min=float(z.min()),log_scaling_max=float(z.max()))
        return history

    @torch.no_grad()
    def relations(self,x,n_train,edges):
        if not bool(self.balance_ready):return super().relations(x,n_train,edges)
        assert n_train==len(self.field_support)
        for stored,version,n,result in self._relation_cache:
            if stored is x and version==x._version and n==n_train:return result
        distance,diagonal=self.field_distance(self.coordinates(x))
        scores=-distance/self.temperature+self.balance_log_scaling[None,:]
        ids=torch.arange(n_train,device=x.device);scores[ids,ids]=-torch.inf
        near=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:8]
        weight=torch.softmax(scores.gather(1,near),dim=1).float()
        result=(near,weight,diagonal.float())
        self._relation_cache.append((x,x._version,n_train,result));self._relation_cache=self._relation_cache[-2:]
        return result


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return BalancedEgopSAGE(arm,config).to(device)


def preflight(output):
    known=[]
    for diagonal_scale in (torch.zeros(17,dtype=torch.float64),torch.linspace(-50,50,17,dtype=torch.float64)):
        kernel=diagonal_scale[:,None]+diagonal_scale[None,:];kernel.fill_diagonal_(-torch.inf)
        z,it,residual=symmetric_balance(kernel)
        p=torch.exp(kernel+z[:,None]+z[None,:]);expected=torch.full_like(p,1/16);expected.fill_diagonal_(0)
        assert torch.allclose(p,expected,atol=1e-7,rtol=1e-6)
        assert torch.allclose(p,p.T,atol=1e-13,rtol=1e-13)
        known.append({'iterations':it,'residual':residual,'max_known_error':float((p-expected).abs().max())})
    local.preflight(output,model_factory=construct)
    checks=[]
    for region in r.REGIONS:
        _,cfg,data,pre=r.load_fold(region,1);n=int(data['n_train']);x,te,ee,y=r.tensors(data,pre,'cuda')
        for arm in ('GE','GL'):
            model=construct(arm,cfg,'cuda').eval()
            state=torch.load(output/f'{region}_{arm}.pt',map_location='cuda',weights_only=False)['state_dict']
            model.load_state_dict(state)
            parent=local.construct(arm,cfg,'cuda');parent.fit_reference(x[:n],te);parent.fit_relations(x[:n],y,te)
            assert all(torch.equal(value,model.state_dict()[name]) for name,value in parent.named_buffers())
            d,_=model.field_distance(model.coordinates(x[:n]));lk=-a.symmetric(d)/model.temperature;lk.fill_diagonal_(-torch.inf)
            z=model.balance_log_scaling;p=torch.exp(lk+z[:,None]+z[None,:])
            marginal_error=float((p.sum(1)-1).abs().max());assert marginal_error<=1e-6
            assert torch.allclose(p,p.T,atol=1e-13,rtol=1e-13)
            near,weight,_=model.relations(x,n,ee);oldnear,_,_=parent.relations(x,n,ee)
            changed=float((near!=oldnear).float().mean());assert changed>0
            checks.append({'region':region,'arm':arm,'source_field_messages_exact':True,
                           'dense_marginal_error':marginal_error,'changed_neighbor_entries':changed})
    r.write_json(output/'balance_checks.json',{'known_cases':known,'real_folds':checks})
    print('PASS: symmetric balancing, unchanged source field/messages, sparse intervention',flush=True)


def summarize(output):
    summary=json.loads((output/'summary.json').read_text());gains=[];rows=[];incoming=[];lookup={}
    for region in r.REGIONS:
        gu_metrics=[]
        for fold in range(1,6):
            _,_,data,_=r.load_fold(region,fold);n=int(data['n_train']);measured={}
            for arm in ARMS:
                directory=output/region/f'fold-{fold:02d}'/arm
                m=json.loads((directory/'metrics.json').read_text())
                with np.load(directory/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'],data['sample_ids'])
                    measured[arm]=r.metrics(data['y'][n:],z['prediction'][n:])
                    assert all(abs(measured[arm][key]-m['validation'][key])<1e-10 for key in measured[arm])
                    near=z['neighbors'][:n];weights=z['weights'][:n]
                    mass=np.bincount(near.ravel(),weights=weights.ravel(),minlength=n);sorted_mass=np.sort(mass)
                    gini=float((2*np.arange(1,n+1)-n-1)@sorted_mass/(n*sorted_mass.sum()))
                    incoming.append({'region':region,'fold':fold,'arm':arm,'incoming_gini':gini,
                                     'top10_mass_fraction':float(sorted_mass[-int(np.ceil(.1*n)):].sum()/mass.sum()),
                                     'zero_incoming_fraction':float((mass==0).mean()),
                                     'sparse_max_marginal_error':float(np.abs(mass-1).max())})
                    for i in range(n,len(data['y'])):
                        lookup.setdefault((region,str(data['sample_ids'][i])),{})[arm]=float(z['prediction'][i])
                rows.append({'region':region,'fold':fold,'arm':arm,**measured[arm],
                             'best_epoch':m['best_epoch'],'elapsed_seconds':m['elapsed_seconds']})
            gu_metrics.append(measured['GU']);gains.append(100*(1-measured['GL']['rmse']/measured['GU']['rmse']))
        summary['province_means'][region]['GU']={k:float(np.mean([row[k] for row in gu_metrics])) for k in ('rmse','mae','r2')}
    summary['GL_vs_GU_gain_percent']=float(np.mean(gains))
    if summary['GL_vs_GU_gain_percent']<=0:summary['classification']='NO_GAIN'
    summary.update(fit_records=90,new_fits=30,cached_fits=60)
    with (output/'all_1421_development_ids.csv').open(newline='',encoding='utf-8-sig') as f:points=list(csv.DictReader(f))
    for point in points:
        observed=float(point['observed_agb'])
        for arm,pred in lookup[(point['region'],point['sample_id'])].items():
            point[arm+'_prediction']=pred;point[arm+'_absolute_error']=abs(pred-observed)
        point['GL_absolute_error_gain_vs_GU']=point['GU_absolute_error']-point['GL_absolute_error']
    summary['GL_better_IDs_vs_GU']=sum(p['GL_absolute_error_gain_vs_GU']>0 for p in points)
    assert len(rows)==90 and len(points)==1421 and len(lookup)==1421
    r.write_csv(output/'all_fold_metrics.csv',rows);r.write_csv(output/'all_1421_development_ids.csv',points)
    r.write_csv(output/'incoming_message_diagnostics.csv',incoming);r.write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


def run(output,preflight_dir):
    assert (preflight_dir/'balance_checks.json').is_file()
    source=Path(local.__file__);value=r.digest(source)
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.24',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GL'},new_arms=('GE','GL'),
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,
                             'frozen_preprocessing_kernel_solves':25,'field_reference':'unit_group_balanced',
                             'message_source':'v0.20 ordinary GE','localization_iterations':1,
                             'balance_tolerance':1e-6,'balance_max_iterations':2000,
                             'balance_solver':'100 symmetric steps then damped Newton on unchanged dual objective',
                             'additional_source_hashes':{str(source.relative_to(r.ROOT)):value}})
    assert r.digest(source)==value;(output/source.name).write_bytes(source.read_bytes())
    summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run','summarize'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    elif args.mode=='summarize':summarize(args.output)
    else:run(args.output,args.preflight_dir)
