"""Train-only PCA initialization, task-learned coordinates and local metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import relation_graphsage as r
import relation_refinement as ref
import torch
from torch import nn
from torch.nn import functional as F

TASK='CARBON-RELATION-LATENT-001'


class LatentSAGE(r.RelationSAGE):
    def __init__(self,arm,config):
        nn.Module.__init__(self)
        self.arm,self.k=arm,config['knn_k']
        self.backbone=r.Regressor('graphsage',config)
        self.register_buffer('group_scale',torch.repeat_interleave(torch.tensor([7,10,56,56,56]),torch.tensor([7,10,56,56,56])).float().sqrt())
        self.register_buffer('center',torch.zeros(185))
        self.register_buffer('temperature',torch.ones(()))
        self.register_buffer('explained_variance',torch.zeros(()))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['seed']+100000)
            self.projection=nn.Linear(185,16,bias=False)
            if arm=='GB':self.projection.requires_grad_(False)
            else:
                self.local=nn.Sequential(nn.Linear(16,16),nn.ReLU(),nn.Linear(16,16))
                nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    @torch.no_grad()
    def fit_reference(self,x,edges):
        assert edges.max()<len(x) and edges.shape[1]==len(x)*8
        data=(x/self.group_scale).cpu().numpy().astype(np.float64)
        center=data.mean(0);u,s,vt=np.linalg.svd(data-center,full_matrices=False)
        basis=vt[:16].copy()
        signs=np.sign(basis[np.arange(16),np.abs(basis).argmax(1)])
        basis*=signs[:,None]
        self.center.copy_(torch.as_tensor(center,device=x.device,dtype=x.dtype))
        self.projection.weight.copy_(torch.as_tensor(basis,device=x.device,dtype=x.dtype))
        self.explained_variance.fill_(float((s[:16]**2).sum()/(s**2).sum()))
        distance,_=self.distances(x,len(x))
        self.temperature.copy_(distance.topk(self.k,largest=False).values.mean().clamp_min(1e-8))

    def distances(self,x,n_train):
        z=self.projection(x/self.group_scale-self.center)
        if self.arm=='GB':g=torch.ones_like(z)
        else:
            a=.5*torch.tanh(self.local(F.normalize(z,dim=1)))
            g=torch.exp(a-a.mean(1,keepdim=True))
        zt,gt=z[:n_train],g[:n_train]
        if self.arm=='GL':
            distance=(.5*(g[:,None,:]+gt[None,:,:])*(z[:,None,:]-zt[None,:,:]).square()).sum(-1)
        else:
            q=z*g.sqrt();qt=q[:n_train]
            distance=(q[:,None,:]-qt[None,:,:]).square().sum(-1)
        mask=torch.arange(len(x),device=x.device)[:,None]==torch.arange(n_train,device=x.device)
        return distance.clamp_min(0).masked_fill(mask,torch.inf),g

    def relations(self,x,n_train,edges):
        distance,g=self.distances(x,n_train)
        near=torch.argsort(distance,dim=1,stable=True)[:,:self.k]
        weight=torch.softmax(-distance.gather(1,near)/self.temperature,dim=1)
        return near,weight,g

    def metric_loss(self,x,target,edges):
        assert len(x)==len(target) and edges.shape[1]==len(x)*self.k
        distance,_=self.distances(x,len(x))
        return F.mse_loss(torch.softmax(-distance/self.temperature,dim=1)@target,target)


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    return LatentSAGE(arm,config).to(device)


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
                assert all(torch.equal(v,model.backbone.state_dict()[k]) for k,v in fixed.backbone.state_dict().items())
                with torch.no_grad():v=model(x,n,ee)
                assert torch.equal(v[2],base[2])
                assert torch.allclose(v[0],base[0],atol=3e-6,rtol=3e-6)
                cases.append({'region':region,'fold':fold,'arm':arm,'paired_initial_graph_rng':True,
                              'explained_variance':float(model.explained_variance),'temperature':float(model.temperature)})
    _,cfg,d,p=r.load_fold('hunan',1);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda');checks={}
    for arm in ('GE','GL'):
        model=construct(arm,cfg,'cuda');model.fit_reference(x[:n],te);before=r.cpu_state(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True);model.metric_loss(x[:n],y,te).backward()
            assert model.projection.weight.grad.norm()>0 and model.local[-1].weight.grad.norm()>0
            optimizer.step()
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items() if k.startswith('backbone'))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(model(x[:n],n,te)[0],y)+.2*model.metric_loss(x[:n],y,te)
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():v=model(x,n,ee);t=model(x[:n],n,te)
        assert torch.allclose(v[0][:n],t[0],atol=3e-6,rtol=3e-6)
        assert torch.all(v[4]>0) and torch.isfinite(v[0]).all()
        path=output/(arm+'.pt');torch.save(r.cpu_state(model),path)
        restored=construct(arm,cfg,'cuda').eval();restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():assert torch.equal(v[0],restored(x,n,ee)[0])
        checks[arm]={'projection_change':float((model.projection.weight.detach().cpu()-before['projection.weight']).abs().max()),
                     'trainable_parameters':sum(q.numel() for q in model.parameters() if q.requires_grad),
                     'query_isolated':True,'checkpoint_includes_reference':True}
    r.write_json(output/'preflight.json',{'cases':cases,'learning':checks});print(json.dumps(checks),flush=True)


def summarize_fixed(output):
    summary=json.loads((output/'summary.json').read_text());gains=[]
    for region in r.REGIONS:
        rows=[]
        for fold in range(1,6):
            path=output/region/f'fold-{fold:02d}'
            gb=json.loads((path/'GB/metrics.json').read_text());gl=json.loads((path/'GL/metrics.json').read_text())
            _,_,d,_=r.load_fold(region,fold);n=int(d['n_train'])
            with np.load(path/'GB/responses.npz') as z:
                assert np.array_equal(z['sample_ids'],d['sample_ids'])
                measured=r.metrics(d['y'][n:],z['prediction'][n:])
            assert all(abs(measured[k]-gb['validation'][k])<1e-10 for k in measured)
            gains.append(100*(1-gl['validation']['rmse']/gb['validation']['rmse']));rows.append(measured)
        summary['province_means'][region]['GB']={k:float(np.mean([row[k] for row in rows])) for k in ('rmse','mae','r2')}
    summary['GL_vs_GB_gain_percent']=float(np.mean(gains))
    if summary['GL_vs_GB_gain_percent']<=0:summary['classification']='NO_GAIN'
    r.write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:
        ref.run(args.output,args.preflight_dir,task_id=TASK,model_factory=construct,warmup_steps=100,
                auxiliary_weight=.2,branch='RELATION/v0.5',static_arms=('GB',))
        summarize_fixed(args.output)


if __name__=='__main__':main()
