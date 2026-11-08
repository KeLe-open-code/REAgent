"""One local EGOP field, with matched global field and frozen AGOP messages."""
import argparse
import json
from pathlib import Path

import relation_agop as a
import relation_graphsage as r
import numpy as np
import torch
from torch.nn import functional as F

TASK='CARBON-RELATION-LOCAL-EGOP-001'
SOURCE=r.HERE/'runs/CARBON-RELATION-AGOP-001/attempt-01'


class LocalEgopSAGE(a.AgopSAGE):
    def __init__(self,arm,config):
        super().__init__(arm,config)
        self.reference_path=SOURCE/config['region']/f"fold-{config['fold']:02d}"/'GE/best.pt'
        n=config['relation_n_train']
        self.register_buffer('field_support',torch.zeros((n,185),dtype=torch.float64))
        self.register_buffer('field_directions',torch.zeros((n,185),dtype=torch.float64))
        self.register_buffer('field_log_norm',torch.zeros(n,dtype=torch.float64))
        self.register_buffer('field_temperature',torch.ones((),dtype=torch.float64))
        self.register_buffer('field_ready',torch.tensor(False))
        self._relation_cache=[]

    @torch.no_grad()
    def fit_reference(self,x,edges):
        assert len(x)==len(self.field_support) and edges.max()<len(x)
        state=torch.load(self.reference_path,map_location=x.device,weights_only=False)['state_dict']
        # The backbone remains freshly initialized; only fitted input geometry is reused.
        for name,value in self.named_buffers():
            if name in state:value.copy_(state[name])
        self.field_ready.fill_(False);self._relation_cache.clear()

    def field_weights(self,u):
        if self.arm=='GE':return torch.softmax(self.field_log_norm,dim=0).expand(len(u),-1)
        eye=torch.eye(185,device=u.device,dtype=u.dtype)
        return torch.softmax(-a.distance(u,self.field_support,eye)/self.field_temperature+self.field_log_norm[None,:],dim=1)

    def field_distance(self,u):
        w=self.field_weights(u);wt=self.field_weights(self.field_support)
        scores=u@self.field_directions.T;ts=self.field_support@self.field_directions.T
        left=(w*scores.square()).sum(1)[:,None]+w@ts.square().T-2*(w*scores)@ts.T
        right=scores.square()@wt.T+(wt*ts.square()).sum(1)[None,:]-2*scores@(wt*ts).T
        eye=torch.eye(185,device=u.device,dtype=u.dtype)
        base=a.distance(u,self.field_support,eye)
        d=((185*(left+right)*.5+.01*base)/1.01).clamp_min(0)
        diagonal=(185*(w@self.field_directions.square())+.01)/1.01
        return d,diagonal

    def reference_gradients(self,u,target,tau):
        eye=torch.eye(185,device=u.device,dtype=u.dtype)
        pred,gradient,_=a.kernel_gradient(u,target.double(),u,eye,tau)
        return pred,gradient,1

    @torch.no_grad()
    def fit_relations(self,x,target,edges):
        assert len(x)==len(target)==len(self.field_support) and edges.max()<len(x)
        u=self.coordinates(x);eye=torch.eye(185,device=x.device,dtype=torch.float64)
        tau=a.bandwidth(u,eye);pred,g,solves=self.reference_gradients(u,target,tau)
        norm=g.square().sum(1);positive=norm>0;assert positive.any()
        self.field_support.copy_(u)
        self.field_directions.copy_(g/norm.sqrt().clamp_min(1e-150)[:,None])
        self.field_log_norm.copy_(torch.where(positive,norm.log(),-torch.inf))
        self.field_temperature.copy_(tau);self.field_ready.fill_(True)
        d,diagonal=self.field_distance(u);d.fill_diagonal_(torch.inf)
        self.temperature.copy_(d.topk(8,largest=False).values.mean().clamp_min(1e-8))
        self._relation_cache.clear()
        return [{'kernel_solves':solves,'zero_gradient_references':int((~positive).sum()),
                 'training_reference_rmse':float((pred-target).square().mean().sqrt()),
                 'field_min_diagonal':float(diagonal.min()),'graph_temperature':float(self.temperature),
                 'reference_temperature':float(tau),'localized':self.arm=='GL'}]

    @torch.no_grad()
    def relations(self,x,n_train,edges):
        if not bool(self.field_ready):return super().relations(x,n_train,edges)
        assert n_train==len(self.field_support)
        for stored,version,n,result in self._relation_cache:
            if stored is x and version==x._version and n==n_train:return result
        d,diagonal=self.field_distance(self.coordinates(x))
        assert torch.isfinite(d).all() and torch.isfinite(diagonal).all()
        # x always starts with training nodes under the frozen data contract.
        ids=torch.arange(n_train,device=x.device);d[ids,ids]=torch.inf
        near=torch.argsort(d,dim=1,stable=True)[:,:8]
        weight=torch.softmax(-d.gather(1,near)/self.temperature,dim=1).float()
        result=(near,weight,diagonal.float())
        self._relation_cache.append((x,x._version,n_train,result));self._relation_cache=self._relation_cache[-2:]
        return result

    def load_state_dict(self,*args,**kwargs):
        self._relation_cache.clear()
        return super().load_state_dict(*args,**kwargs)


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        source=r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz'
        with np.load(source) as data:config['relation_n_train']=int(data['n_train'])
    return LocalEgopSAGE(arm,config).to(device)


def preflight(output,model_factory=construct):
    output.mkdir(parents=True,exist_ok=False);cases=[];learning={}
    for region in r.REGIONS:
        _,cfg,data,pre=r.load_fold(region,1);n=int(data['n_train']);x,te,ee,y=r.tensors(data,pre,'cuda')
        original_x=x
        first=None;results={}
        for arm in ('GB','GE','GL'):
            model=model_factory(arm,cfg,'cuda').eval()
            x=model.prepare_inputs(original_x,data['missing_mask']) if hasattr(model,'prepare_inputs') else original_x
            model.fit_reference(x[:n],te)
            rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            with torch.no_grad():initial=model(x,n,ee)[0]
            if first is None:first=(rng,initial)
            else:assert torch.equal(first[1],initial) and all(torch.equal(p,q) for p,q in zip(first[0],rng))
            cases.append({'region':region,'arm':arm,'paired_initial_prediction_and_rng':True})
            if arm=='GB':continue
            before=r.cpu_state(model);history=model.fit_relations(x[:n],y,te)
            assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items() if k.startswith('backbone'))
            u=model.coordinates(x);d,diag=model.field_distance(u);q=n;j=0
            ws=model.field_weights(u[[q,j]]);g=model.field_directions
            matrices=[(185*g.T@(w[:,None]*g)+.01*torch.eye(185,device=x.device))/1.01 for w in ws]
            assert all(float(torch.linalg.eigvalsh(a.symmetric(m)).min())>0 for m in matrices)
            delta=u[q]-u[j];exact=delta@((matrices[0]+matrices[1])*.5)@delta
            assert torch.allclose(exact,d[q,j],atol=1e-8,rtol=1e-8)
            frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('backbone')}
            optimizer=model.make_optimizer(cfg)
            for _ in range(3):
                model.train();optimizer.zero_grad();F.mse_loss(model(x[:n],n,te)[0],y).backward();optimizer.step()
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
            model.eval()
            with torch.no_grad():
                full=model(x,n,ee);alone=model(x[:n],n,te)
                single=model(torch.cat([x[:n],x[n:n+1]]),n,ee)
            assert torch.equal(full[2][:n],alone[2])
            assert torch.allclose(full[0][:n],alone[0],atol=3e-6,rtol=3e-6)
            assert torch.equal(full[2][n],single[2][n]) and torch.allclose(full[0][n],single[0][n],atol=3e-6,rtol=3e-6)
            path=output/f'{region}_{arm}.pt';torch.save({'state_dict':r.cpu_state(model),'config':cfg},path)
            saved=torch.load(path,weights_only=False);restored=type(model)(arm,saved['config']).to('cuda').eval()
            restored.load_state_dict(saved['state_dict'])
            with torch.no_grad():assert torch.equal(full[0],restored(x,n,ee)[0])
            results[arm]=full[2].cpu()
            learning[f'{region}_{arm}']={'history':history,'checkpoint_exact':True,'inductive_queries':True}
        change=float((results['GE']!=results['GL']).float().mean());assert change>0
        learning[region]={'changed_neighbor_entries':change}
    # The parent runner expects three region-level learning records.
    by_region={region:{k:v for k,v in learning.items() if k.startswith(region)} for region in r.REGIONS}
    r.write_json(output/'preflight.json',{'cases':cases,'learning':by_region,'selected_device':'cuda'})
    print(json.dumps(by_region),flush=True)


def run(output,preflight_dir):
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.22',model_factory=construct,
          previous=SOURCE,cache_aliases={'G0':'G0','GB':'GB','GS':'GE'},new_arms=('GE','GL'),
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,
                             'frozen_preprocessing_kernel_solves':25,'field_reference':'unit_group_balanced',
                             'message_source':'v0.20 ordinary GE','localization_iterations':1})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)
