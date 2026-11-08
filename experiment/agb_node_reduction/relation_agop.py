"""Cross-fitted input-gradient moments for GraphSAGE relations.

RFM/AGOP: Radhakrishnan et al., Science 2024; xRFM, arXiv:2508.10053.
BW covariance mean: Alvarez-Esteban et al. 2016, Chewi et al., COLT 2020.
This is a native adaptation, not a reproduction of stock RFM or xRFM.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import relation_graphsage as r
import relation_bilevel as b
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TASK='CARBON-RELATION-AGOP-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-BILEVEL-001/attempt-01'
NEW=('GB','GE','GL')
CACHED={'G0':'G0','GS':'GE'}


def symmetric(a):return (a+a.T)*.5


def powers(a):
    values,vectors=torch.linalg.eigh(symmetric(a))
    assert float(values.min())>0
    root=(vectors*values.sqrt()[None,:])@vectors.T
    inverse=(vectors*values.rsqrt()[None,:])@vectors.T
    return root,inverse


def bw_mean(matrices,weights,steps=10):
    current=sum(w*a for w,a in zip(weights,matrices));change=0.
    for _ in range(steps):
        root,inverse=powers(current)
        average=sum(w*powers(root@a@root)[0] for w,a in zip(weights,matrices))
        updated=symmetric(inverse@average@average@inverse)
        change=float((updated-current).norm()/current.norm())
        current=updated
    return current,change


def distance(q,s,metric):
    qm=q@metric;sm=s@metric
    return ((qm*q).sum(1)[:,None]+(sm*s).sum(1)[None,:]-2*qm@s.T).clamp_min(0)


def bandwidth(u,metric):
    d=distance(u,u,metric);d.fill_diagonal_(torch.inf)
    return d.topk(8,largest=False).values.mean().clamp_min(1e-8)


def kernel_gradient(support,target,query,metric,tau):
    kernel=torch.exp(-distance(support,support,metric)/tau)
    alpha=torch.linalg.solve(kernel+.01*len(support)*torch.eye(len(support),device=support.device,dtype=support.dtype),target-target.mean())
    cross=torch.exp(-distance(query,support,metric)/tau)
    coeff=cross*alpha[None,:]
    gradient=(2/tau)*(coeff@support-coeff.sum(1)[:,None]*query)@metric
    return target.mean()+cross@alpha,gradient,alpha


class AgopSAGE(r.RelationSAGE):
    def __init__(self,arm,config):
        nn.Module.__init__(self);self.arm=arm;self.k=8;self.seed=config['seed']
        self.backbone=r.Regressor('graphsage',config)
        self.register_buffer('group_scale',torch.repeat_interleave(torch.tensor([7,10,56,56,56]),torch.tensor([7,10,56,56,56])).float().sqrt())
        self.register_buffer('center',torch.zeros(185))
        self.register_buffer('metric',torch.eye(185,dtype=torch.float64))
        self.register_buffer('temperature',torch.ones((),dtype=torch.float64))
        self.register_buffer('input_mean',torch.zeros(185,dtype=torch.float64))
        self.register_buffer('input_scale',torch.ones(185,dtype=torch.float64))
        # Fitted buffers, not parameters updated by the GNN optimizer.
        self.projection=nn.Module();self.projection.register_buffer('weight',torch.eye(185,dtype=torch.float64))

    def coordinates(self,x):return (x/self.group_scale-self.center).double()

    @torch.no_grad()
    def fit_reference(self,x,edges):
        assert edges.max()<len(x)
        self.center.copy_((x/self.group_scale).mean(0))
        self.temperature.copy_(bandwidth(self.coordinates(x),self.metric))
        self.normalize_messages(x)

    def filtered(self,x):
        return (self.coordinates(x)@self.projection.weight.T+self.center)*self.group_scale

    def regularize_moment(self,a,eye):
        a=a*(185/a.trace())
        return symmetric((a+.01*eye)/1.01)

    @torch.no_grad()
    def normalize_messages(self,x):
        values=self.filtered(x);self.input_mean.copy_(values.mean(0))
        sd=values.std(0,correction=0);self.input_scale.copy_(torch.where(sd<1e-6,torch.ones_like(sd),sd))

    @torch.no_grad()
    def fit_relations(self,x,target,edges,rounds=5):
        assert len(x)==len(target) and int(edges.max())<len(x)
        groups=torch.as_tensor(b.assignment(len(x),self.seed),device=x.device)
        u=self.coordinates(x);y=target.double();eye=torch.eye(185,device=x.device,dtype=torch.float64)
        history=[]
        for iteration in range(rounds):
            tau=bandwidth(u,self.metric);matrices=[];weights=[];square_error=0.
            for fold in range(5):
                support=groups!=fold;query=~support
                pred,gradient,_=kernel_gradient(u[support],y[support],u[query],self.metric,tau)
                a=gradient.T@gradient/int(query.sum());assert a.trace()>0
                matrices.append(self.regularize_moment(a,eye));weights.append(float(query.sum())/len(x))
                square_error+=float((pred-y[query]).square().sum())
            if self.arm=='GL':updated,change=bw_mean(matrices,weights)
            else:updated=sum(w*a for w,a in zip(weights,matrices));change=0.
            updated=updated*(185/updated.trace())
            if self.arm!='GB':self.metric.copy_(symmetric(updated))
            values=torch.linalg.eigvalsh(self.metric);assert values.min()>0
            history.append({'iteration':iteration+1,'training_internal_query_rmse':(square_error/len(x))**.5,
                            'kernel_solves':5,'input_gradient_observations':len(x),
                            'metric_min_eigenvalue':float(values.min()),'metric_max_eigenvalue':float(values.max()),
                            'metric_distance_from_identity':float((self.metric-eye).norm()),
                            'barycenter_last_relative_change':change})
        self.projection.weight.copy_(powers(self.metric)[0])
        self.temperature.copy_(bandwidth(u,self.metric));self.normalize_messages(x)
        return history

    def relations(self,x,n_train,edges):
        u=self.coordinates(x)
        d=distance(u,u[:n_train],self.metric)
        mask=torch.arange(len(x),device=x.device)[:,None]==torch.arange(n_train,device=x.device)
        d=d.masked_fill(mask,torch.inf)
        near=torch.argsort(d,dim=1,stable=True)[:,:8]
        weight=torch.softmax(-d.gather(1,near)/self.temperature,dim=1).float()
        return near,weight,self.metric.diag().float().expand(len(x),185)

    def forward(self,x,n_train,edges):
        near,weight,g=self.relations(x,n_train,edges)
        features=((self.filtered(x)-self.input_mean)/self.input_scale).float()
        h=self.weighted_layer(self.backbone.first,features,near,weight)
        h=self.backbone.dropout(torch.relu(h))
        embedding=torch.relu(self.weighted_layer(self.backbone.second,h,near,weight))
        return self.backbone.head(embedding).squeeze(-1),embedding,near,weight,g

    def make_optimizer(self,config):
        return torch.optim.Adam(self.backbone.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    return AgopSAGE(arm,config).to(device)


def preflight(output,model_factory=construct):
    output.mkdir(parents=True,exist_ok=False);cases=[]
    gen=torch.Generator().manual_seed(59)
    support=torch.randn((15,7),generator=gen,dtype=torch.float64)
    query=torch.randn((6,7),generator=gen,dtype=torch.float64,requires_grad=True)
    y=torch.randn(15,generator=gen,dtype=torch.float64)
    a=torch.randn((7,7),generator=gen,dtype=torch.float64);metric=a@a.T+torch.eye(7)
    tau=torch.tensor(3.,dtype=torch.float64)
    pred,analytic,_=kernel_gradient(support,y,query,metric,tau)
    numeric=torch.autograd.grad(pred.sum(),query)[0]
    error=float((numeric-analytic).abs().max().detach());assert error<1e-10
    matrices=[torch.diag(torch.tensor([1.,4.,9.],dtype=torch.float64)),torch.diag(torch.tensor([9.,1.,4.],dtype=torch.float64))]
    result,_=bw_mean(matrices,[.4,.6]);exact=(.4*matrices[0].diag().sqrt()+.6*matrices[1].diag().sqrt()).square()
    assert torch.allclose(result,torch.diag(exact),atol=1e-10)
    same,_=bw_mean([metric,metric],[.4,.6]);assert torch.allclose(same,metric,atol=1e-10)
    for region in r.REGIONS:
        _,cfg,d,p=r.load_fold(region,1);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda');first=None
        for arm in NEW:
            model=model_factory(arm,cfg,'cuda').eval();model.fit_reference(x[:n],te)
            state=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            with torch.no_grad():value=model(x,n,ee)[0]
            if first is None:first=(state,value)
            else:assert all(torch.equal(a,bb) for a,bb in zip(first[0],state)) and torch.equal(first[1],value)
            cases.append({'region':region,'arm':arm,'paired_backbone_rng':True})
    checks={}
    _,cfg,d,p=r.load_fold('hunan',1);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda')
    for arm in NEW:
        model=model_factory(arm,cfg,'cuda');model.fit_reference(x[:n],te);before=r.cpu_state(model)
        history=model.fit_relations(x[:n],y,te,rounds=2)
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items() if k.startswith('backbone'))
        changed=float((model.metric.cpu()-before['metric']).norm());assert (changed==0) if arm=='GB' else (changed>0)
        trained=r.cpu_state(model);optimizer=model.make_optimizer(cfg)
        for _ in range(3):
            model.train();optimizer.zero_grad();F.mse_loss(model(x[:n],n,te)[0],y).backward();optimizer.step()
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in trained.items() if not k.startswith('backbone'))
        model.eval()
        with torch.no_grad():full=model(x,n,ee);alone=model(x[:n],n,te)
        assert torch.allclose(full[0][:n],alone[0],atol=3e-6,rtol=3e-6)
        path=output/(arm+'.pt');torch.save(r.cpu_state(model),path)
        restored=model_factory(arm,cfg,'cuda').eval();restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():assert torch.equal(full[0],restored(x,n,ee)[0])
        checks[arm]={'metric_change':changed,'history':history,'frozen_during_gnn':True,'checkpoint_exact':True}
    r.write_json(output/'preflight.json',{'cases':cases,'input_gradient_max_error':error,'barycenter_known_cases':True,'learning':checks})
    print(json.dumps(checks),flush=True)


def summarize(output,new_fits=45,cached_fits=30):
    r.summarize(output)
    summary=json.loads((output/'summary.json').read_text());gains={a:[] for a in ('GB','GS')}
    for region in r.REGIONS:
        for arm in gains:
            rows=[]
            for fold in range(1,6):
                directory=output/region/f'fold-{fold:02d}';_,_,d,_=r.load_fold(region,fold);n=int(d['n_train'])
                control=json.loads((directory/arm/'metrics.json').read_text());gl=json.loads((directory/'GL/metrics.json').read_text())
                with np.load(directory/arm/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'],d['sample_ids']);measured=r.metrics(d['y'][n:],z['prediction'][n:])
                assert all(abs(measured[k]-control['validation'][k])<1e-10 for k in measured)
                gains[arm].append(100*(1-gl['validation']['rmse']/measured['rmse']));rows.append(measured)
            summary['province_means'][region][arm]={k:float(np.mean([row[k] for row in rows])) for k in ('rmse','mae','r2')}
    for arm,values in gains.items():summary[f'GL_vs_{arm}_gain_percent']=float(np.mean(values))
    if any(np.mean(v)<=0 for v in gains.values()):summary['classification']='NO_GAIN'
    summary.pop('fits');summary.update(fit_records=75,new_fits=new_fits,cached_fits=cached_fits)
    write_all_metrics(output)
    summary['complete_fold_metrics_table']='all_fold_metrics.csv'
    r.write_json(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


def write_all_metrics(output):
    rows=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            for arm in ('G0','GB','GS','GE','GL'):
                m=json.loads((output/region/f'fold-{fold:02d}'/arm/'metrics.json').read_text())
                rows.append({'region':region,'fold':fold,'arm':arm,**m['validation'],
                             'best_epoch':m['best_epoch'],'elapsed_seconds':m['elapsed_seconds']})
    assert len(rows)==75
    r.write_csv(output/'all_fold_metrics.csv',rows)


def run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.20',model_factory=construct,
        previous=PREVIOUS,cache_aliases=CACHED,new_arms=NEW,relation_metadata=None,max_gnn_epochs=400):
    check=json.loads((preflight_dir/'preflight.json').read_text());assert len(check['cases'])==9 and len(check['learning'])==3
    old=json.loads((previous/'manifest.json').read_text());count=0
    for path,value in old['source_hashes'].items():
        if '/data/' in path.replace('\\','/') or path.endswith('config.json'):
            assert r.digest(r.ROOT/path)==value;count+=1
    assert count==60
    output.mkdir(parents=True,exist_ok=False)
    sources=list(dict.fromkeys([Path(__file__).resolve(),Path(model_factory.__code__.co_filename).resolve(),
                                Path(r.__file__),Path(b.__file__),r.BASE/'baseline.py']))
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources};cached={}
    for region in r.REGIONS:
        for fold in range(1,6):
            source=r.SOURCE/region/f'fold-{fold:02d}'
            for name in ('config.json','data/development.npz','data/preprocessor.npz','data/split.csv'):
                p=source/name;hashes[str(p.relative_to(r.ROOT))]=r.digest(p)
            for alias,original in cache_aliases.items():
                src=previous/region/f'fold-{fold:02d}'/original
                for p in src.iterdir():
                    if p.is_file():cached[str(p.relative_to(r.ROOT))]=r.digest(p)
                shutil.copytree(src,output/region/f'fold-{fold:02d}'/alias)
            shutil.copyfile(previous/region/f'fold-{fold:02d}'/'internal_split.csv',output/region/f'fold-{fold:02d}'/'internal_split.csv')
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    new_fits,cached_fits=15*len(new_arms),15*len(cache_aliases)
    r.write_json(output/'manifest.json',{'task_id':task_id,'branch':branch,'device':'cuda',
                 'started_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'source_hashes':hashes,'cached_hashes':cached,
                 'cache_source':str(previous.relative_to(r.ROOT)),'cache_aliases':cache_aliases,'metric_rounds':5,'BW_iterations':10,
                 'preflight_sha256':r.digest(preflight_dir/'preflight.json'),'new_fits':new_fits,'cached_fits':cached_fits,'final_holdout_accessed':False,
                 'final_gnn_max_epochs':max_gnn_epochs,
                 **(relation_metadata or {})})
    start=time.perf_counter()
    for fold in range(1,6):
        for region in r.REGIONS:
            for arm in new_arms:r.fit(arm,region,fold,output,'cuda',model_factory=model_factory,task_id=task_id,max_gnn_epochs=max_gnn_epochs)
    for path,value in {**hashes,**cached}.items():assert r.digest(r.ROOT/path)==value,path
    r.write_json(output/'execution.json',{'finished_at':datetime.now().astimezone().isoformat(),'wall_seconds':time.perf_counter()-start,
                 'new_fits':new_fits,'cached_fits':cached_fits,'all_inputs_unchanged':True})
    summarize(output,new_fits,cached_fits)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)
