"""Train-internal kernel hypergradients for a frozen GraphSAGE relation subspace."""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import relation_graphsage as r
import relation_latent as latent
import relation_stiefel as stiefel
import relation_subspace as subspace
import torch
from torch.nn import functional as F

TASK='CARBON-RELATION-BILEVEL-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-LATENT-001/attempt-02'
NEW=('GB','GE','GL')


def assignment(n,seed):
    groups=np.empty(n,dtype=np.int64)
    groups[np.random.default_rng(seed+200000).permutation(n)]=np.arange(n)%5
    return groups


def outer_loss(u,y,p,tau,groups,fold,detach_solution=False):
    z=u@p.T;support=groups!=fold;query=~support
    zs,zq=z[support],z[query]
    ds=(zs.square().sum(1)[:,None]+zs.square().sum(1)[None,:]-2*zs@zs.T).clamp_min(0)
    dq=(zq.square().sum(1)[:,None]+zs.square().sum(1)[None,:]-2*zq@zs.T).clamp_min(0)
    kernel=torch.exp(-ds/tau);cross=torch.exp(-dq/tau)
    mean=y[support].mean();target=y[support]-mean
    alpha=torch.linalg.solve(kernel+.01*len(zs)*torch.eye(len(zs),device=u.device,dtype=u.dtype),target)
    if detach_solution:alpha=alpha.detach()
    return F.mse_loss(mean+cross@alpha,y[query])


class StagedOptimizer:
    def __init__(self,model,config):
        self.model=model;self.p=model.projection.weight
        self.adam=torch.optim.Adam(model.backbone.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        self.steps=0;self.warmup_limit=100;self.projection_steps=0;self.frozen=None;self.maximum_error=0.

    def zero_grad(self,set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none);self.p.grad=None

    @torch.no_grad()
    def step(self):
        if self.steps<self.warmup_limit:
            if self.model.arm!='GB':
                x=self.p.T.double();gradient=self.p.grad.T.double()
                if self.model.arm=='GL':gradient=stiefel.tangent(x,gradient)
                direction=-gradient/torch.linalg.matrix_norm(gradient,ord=2).clamp_min(1e-30)
                updated=x+.01*direction
                if self.model.arm=='GL':
                    q,rr=torch.linalg.qr(updated,mode='reduced');updated=q*torch.sign(rr.diag())[None,:]
                self.p.copy_(updated.T.to(self.p.dtype));self.projection_steps+=1
                error=(self.p@self.p.T-torch.eye(len(self.p),device=self.p.device)).abs().max()
                self.maximum_error=max(self.maximum_error,float(error))
            if self.steps+1==self.warmup_limit:
                self.p.requires_grad_(False);self.p.grad=None;self.frozen=self.p.detach().clone()
        else:self.adam.step()
        self.steps+=1

    def diagnostics(self):
        return {'mode':'training_internal_bilevel_'+self.model.arm,'proxy_solves':self.model.meta_calls,
                'projection_steps':self.projection_steps,'gnn_updates':self.steps-self.warmup_limit,
                'maximum_orthogonality_error':self.maximum_error,
                'projection_frozen_during_gnn':self.frozen is not None and torch.equal(self.frozen,self.p)}


class BilevelSAGE(subspace.SubspaceSAGE):
    warmup_loss_name='training_internal_query_mse'

    def __init__(self,arm,config):
        super().__init__(arm,config)
        # GB computes the same hypergradient but its optimizer discards the update.
        self.projection.requires_grad_(True);self.seed=config['seed'];self.meta_calls=0

    def make_optimizer(self,config):return StagedOptimizer(self,config)

    def metric_loss(self,x,target,edges):
        assert len(x)==len(target) and int(edges.max())<len(x)
        groups=torch.as_tensor(assignment(len(x),self.seed),device=x.device)
        u=(x/self.group_scale-self.center).double()
        value=outer_loss(u,target.double(),self.projection.weight.double(),self.temperature.double(),groups,self.meta_calls%5)
        self.meta_calls+=1
        return value


def construct(arm,config,device):
    assert arm in NEW
    r.fix_seed(config['seed'],1)
    return BilevelSAGE(arm,config).to(device)


def preflight(output):
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,cfg,d,p=r.load_fold(region,fold);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda')
            fixed=construct('GB',cfg,'cuda').eval();fixed.fit_reference(x[:n],te)
            cpu,cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            with torch.no_grad():base=fixed(x,n,ee)
            for arm in ('GE','GL'):
                model=construct(arm,cfg,'cuda').eval();model.fit_reference(x[:n],te)
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
                with torch.no_grad():assert torch.equal(base[0],model(x,n,ee)[0])
                cases.append({'region':region,'fold':fold,'arm':arm,'paired_initial_graph_rng':True})
    _,cfg,d,p=r.load_fold('hunan',1);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda');checks={}
    for arm in NEW:
        model=construct(arm,cfg,'cuda');model.fit_reference(x[:n],te);before=r.cpu_state(model)
        optimizer=model.make_optimizer(cfg);optimizer.warmup_limit=3
        for _ in range(3):
            optimizer.zero_grad();model.metric_loss(x[:n],y,te).backward()
            assert model.projection.weight.grad.norm()>0
            assert all(q.grad is None for q in model.backbone.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items() if k.startswith('backbone'))
        change=float((model.projection.weight.detach().cpu()-before['projection.weight']).abs().max())
        assert (change==0) if arm=='GB' else (change>0)
        for _ in range(3):
            optimizer.zero_grad();F.mse_loss(model(x[:n],n,te)[0],y).backward()
            assert model.projection.weight.grad is None
            optimizer.step()
        model.eval()
        with torch.no_grad():full=model(x,n,ee);alone=model(x[:n],n,te)
        assert torch.allclose(full[0][:n],alone[0],atol=3e-6,rtol=3e-6)
        assert optimizer.diagnostics()['projection_frozen_during_gnn']
        if arm=='GL':assert optimizer.maximum_error<2e-6
        path=output/(arm+'.pt');torch.save(r.cpu_state(model),path)
        restored=construct(arm,cfg,'cuda').eval();restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():assert torch.equal(full[0],restored(x,n,ee)[0])
        checks[arm]={'projection_change':change,'query_isolation':True,'checkpoint_exact':True,**optimizer.diagnostics()}
    u=(x[:32]/model.group_scale-model.center).cpu().double();target=y[:32].cpu().double()
    pp=before['projection.weight'].double().requires_grad_();tau=model.temperature.cpu().double()
    groups=torch.as_tensor(assignment(32,cfg['seed']))
    loss=outer_loss(u,target,pp,tau,groups,0);grad=torch.autograd.grad(loss,pp)[0]
    stopped=outer_loss(u,target,pp,tau,groups,0,detach_solution=True)
    partial=torch.autograd.grad(stopped,pp)[0];assert (grad-partial).norm()>1e-8
    direction=torch.randn(pp.shape,generator=torch.Generator().manual_seed(701),dtype=torch.float64);direction/=direction.norm()
    eps=1e-5
    numeric=(outer_loss(u,target,pp+eps*direction,tau,groups,0)-outer_loss(u,target,pp-eps*direction,tau,groups,0))/(2*eps)
    analytic=(grad*direction).sum();assert torch.allclose(numeric,analytic,atol=1e-7,rtol=1e-5)
    hyper={'finite_difference':float(numeric.detach()),'autograd':float(analytic),'solution_derivative_norm':float((grad-partial).norm())}
    r.write_json(output/'preflight.json',{'cases':cases,'learning':checks,'hypergradient':hyper});print(json.dumps({'learning':checks,'hypergradient':hyper}),flush=True)


def summarize_controls(output):
    r.summarize(output);latent.summarize_fixed(output)
    summary=json.loads((output/'summary.json').read_text());summary.pop('fits')
    summary.update(fit_records=60,new_fits=45,cached_fits=15)
    for region in r.REGIONS:
        for fold in range(1,6):
            for arm in NEW:
                m=json.loads((output/region/f'fold-{fold:02d}'/arm/'metrics.json').read_text())
                assert m['optimizer_diagnostics']['proxy_solves']==100 and m['optimizer_diagnostics']['projection_frozen_during_gnn']
                if arm=='GL':assert m['projection_orthogonality_error']<2e-6
    r.write_json(output/'summary.json',summary)


def run(output,preflight_dir):
    checked=json.loads((preflight_dir/'preflight.json').read_text());assert len(checked['cases'])==30 and len(checked['learning'])==3
    old=json.loads((PREVIOUS/'manifest.json').read_text());input_count=0
    for path,value in old['source_hashes'].items():
        if '/data/' in path.replace('\\','/') or path.endswith('config.json'):
            assert r.digest(r.ROOT/path)==value,path;input_count+=1
    assert input_count==60
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),Path(r.__file__),Path(latent.__file__),Path(stiefel.__file__),Path(subspace.__file__),r.BASE/'baseline.py']
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources};cached={}
    for region in r.REGIONS:
        for fold in range(1,6):
            source,cfg,data,_=r.load_fold(region,fold);n=int(data['n_train'])
            for name in ('config.json','data/development.npz','data/preprocessor.npz','data/split.csv'):
                p=source/name;hashes[str(p.relative_to(r.ROOT))]=r.digest(p)
            src=PREVIOUS/region/f'fold-{fold:02d}'/'G0'
            for p in src.iterdir():
                if p.is_file():cached[str(p.relative_to(r.ROOT))]=r.digest(p)
            directory=output/region/f'fold-{fold:02d}';shutil.copytree(src,directory/'G0')
            r.write_csv(directory/'internal_split.csv',[{'sample_id':str(sid),'internal_query_group':int(g)} for sid,g in zip(data['sample_ids'][:n],assignment(n,cfg['seed']))])
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    r.write_json(output/'manifest.json',{'task_id':TASK,'branch':'RELATION/v0.9','device':'cuda',
                'started_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'source_hashes':hashes,
                'cached_hashes':cached,'cache_source':str(PREVIOUS.relative_to(r.ROOT)),
                'preflight_sha256':r.digest(preflight_dir/'preflight.json'),
                'new_fits':45,'cached_fits':15,'final_holdout_accessed':False})
    start=time.perf_counter()
    for fold in range(1,6):
        for region in r.REGIONS:
            for arm in NEW:r.fit(arm,region,fold,output,'cuda',model_factory=construct,task_id=TASK,warmup_steps=100)
    for path,value in {**hashes,**cached}.items():assert r.digest(r.ROOT/path)==value,path
    r.write_json(output/'execution.json',{'finished_at':datetime.now().astimezone().isoformat(),
                'wall_seconds':time.perf_counter()-start,'new_fits':45,'cached_fits':15,'all_inputs_unchanged':True})
    summarize_controls(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)


if __name__=='__main__':main()
