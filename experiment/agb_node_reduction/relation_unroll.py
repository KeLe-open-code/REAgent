"""Short unrolled GraphSAGE feedback for a positive diagonal metric congruence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import relation_graphsage as r
import relation_bilevel as b
import relation_agop as a
import relation_local_egop as local
import relation_balance as balance

TASK='CARBON-RELATION-UNROLL-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-MASS-001/attempt-01'


def predictor(parameters,features,near,weights):
    values=features
    for name in ('first','second'):
        aggregate=(values[near]*weights[:,:,None]).sum(1)
        values=F.relu(F.linear(aggregate,parameters[name+'.lin_l.weight'],parameters[name+'.lin_l.bias'])
                      +F.linear(values,parameters[name+'.lin_r.weight']))
    return F.linear(values,parameters['head.weight'],parameters['head.bias']).squeeze(-1)


def unroll(parameters,features,near,weights,target,support,steps=4,detach_inner=False):
    current=parameters
    for _ in range(steps):
        pred=predictor(current,features,near,weights)
        loss=F.mse_loss(pred[support],target[support])
        gradients=torch.autograd.grad(loss,tuple(current.values()),create_graph=True)
        norm=torch.sqrt(sum(g.square().sum() for g in gradients));clip=(5/(norm+1e-6)).clamp_max(1)
        current={key:value-.01*(clip*gradient+1e-4*value) for (key,value),gradient in zip(current.items(),gradients)}
        if detach_inner:current={key:value.detach().requires_grad_() for key,value in current.items()}
    prediction=predictor(current,features,near,weights)
    return F.mse_loss(prediction[~support],target[~support]),current


class UnrolledSAGE(local.LocalEgopSAGE):
    def __init__(self,arm,config):
        super().__init__(arm,config)
        self.register_buffer('local_task_log_metric',torch.zeros(185,dtype=torch.float64))
        self.probe=config.get('relation_probe',False)
        self.config=config

    def initial_accumulator(self,gradient_norm):
        return 1.

    def corrected_distance(self,u,log_metric):
        w=self.field_weights(u).detach();wt=self.field_weights(self.field_support).detach()
        root=(.5*log_metric).exp();q=u*root;s=self.field_support*root
        scores=q@self.field_directions.T;ts=s@self.field_directions.T
        left=(w*scores.square()).sum(1)[:,None]+w@ts.square().T-2*(w*scores)@ts.T
        right=scores.square()@wt.T+(wt*ts.square()).sum(1)[None,:]-2*scores@(wt*ts).T
        base=a.distance(q,s,torch.eye(185,device=u.device,dtype=u.dtype))
        distance=((185*(left+right)*.5+.01*base)/1.01).clamp_min(0)
        diagonal=(185*(w@self.field_directions.square())+.01)/1.01*log_metric.exp()
        return distance,diagonal

    def graph(self,u,log_metric,dtype,support=None):
        distance,diagonal=self.corrected_distance(u,log_metric);n=len(self.field_support)
        mask=torch.arange(len(u),device=u.device)[:,None]==torch.arange(n,device=u.device)[None,:]
        if support is not None:mask=mask|(~support)[None,:]
        distance=distance.masked_fill(mask,torch.inf)
        near=torch.argsort(distance,dim=1,stable=True)[:,:8]
        weights=torch.softmax(-distance.gather(1,near)/self.temperature,dim=1).to(dtype)
        return near,weights,diagonal.to(dtype)

    @torch.no_grad()
    def relations(self,x,n_train,edges):
        if not bool(self.field_ready):return super().relations(x,n_train,edges)
        assert n_train==len(self.field_support)
        for stored,version,n,result in self._relation_cache:
            if stored is x and version==x._version and n==n_train:return result
        result=self.graph(self.coordinates(x),self.local_task_log_metric,torch.float32)
        self._relation_cache.append((x,x._version,n_train,result));self._relation_cache=self._relation_cache[-2:]
        return result

    def fit_relations(self,x,target,edges):
        base=super().fit_relations(x,target,edges)[0]
        initial={key:value.detach().clone() for key,value in self.backbone.state_dict().items()}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(x.device) if x.is_cuda else None
        teacher_steps,outer_steps=(2,1) if self.probe else (100,25)
        inner_steps=1 if self.probe else 4
        optimizer=torch.optim.Adam(self.backbone.parameters(),lr=self.config['learning_rate'],weight_decay=self.config['weight_decay'])
        for _ in range(teacher_steps):
            self.train();optimizer.zero_grad();loss=F.mse_loss(self(x,len(x),edges)[0],target)
            loss.backward();torch.nn.utils.clip_grad_norm_(self.backbone.parameters(),5);optimizer.step()
        teacher_loss=float(loss.detach());self.eval()
        features=((self.filtered(x)-self.input_mean)/self.input_scale).float().detach();u=self.coordinates(x).detach()
        groups=torch.as_tensor(b.assignment(len(x),self.seed),device=x.device)
        logarithm=self.local_task_log_metric.detach().clone().requires_grad_();accumulator=None;history=[]
        for iteration in range(outer_steps):
            support=groups!=iteration%5
            near,weight,_=self.graph(u,logarithm,features.dtype,support)
            assert support[near].all()
            parameters={key:value.detach().requires_grad_() for key,value in self.backbone.named_parameters()}
            value,adapted=unroll(parameters,features,near,weight,target,support,steps=inner_steps)
            gradient=torch.autograd.grad(value,logarithm)[0]
            assert torch.isfinite(gradient).all() and gradient.norm()>0
            norm=float(gradient.norm())
            if accumulator is None:accumulator=self.initial_accumulator(norm)
            accumulator+=norm**2
            logarithm=(logarithm-gradient/accumulator**.5).detach().requires_grad_()
            with torch.no_grad():
                for key,parameter in self.backbone.named_parameters():parameter.copy_(adapted[key].detach())
            history.append({**base,'outer_step':iteration+1,'teacher_steps':teacher_steps,'inner_steps':inner_steps,
                            'teacher_final_loss':teacher_loss,'training_meta_query_mse':float(value.detach()),
                            'hypergradient_norm':norm,'adaptive_accumulator':accumulator,
                            'log_metric_min':float(logarithm.detach().min()),'log_metric_max':float(logarithm.detach().max())})
        with torch.no_grad():self.local_task_log_metric.copy_(logarithm.detach())
        assert torch.isfinite(self.local_task_log_metric).all()
        self.backbone.load_state_dict(initial)
        for parameter in self.backbone.parameters():parameter.grad=None
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,x.device)
        self._relation_cache.clear()
        assert all(torch.equal(value,self.backbone.state_dict()[key]) for key,value in initial.items())
        return history


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return UnrolledSAGE(arm,config).to(device)


def probe_construct(arm,config,device):
    model=construct(arm,config,device);model.probe=True;return model


def preflight(output,model_factory=construct,probe_factory=probe_construct):
    local.preflight(output,model_factory=probe_factory)
    _,cfg,data,pre=r.load_fold('hubei',1);n=int(data['n_train']);x,te,ee,y=r.tensors(data,pre,'cuda')
    model=model_factory('GL',cfg,'cuda');model.fit_reference(x[:n],te)
    local.LocalEgopSAGE.fit_relations(model,x[:n],y,te)
    before=r.cpu_state(model);rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
    model.probe=True;model.fit_relations(x[:n],y,te)
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state())
    assert all(torch.equal(value,model.state_dict()[key].cpu()) for key,value in before.items() if key.startswith('backbone'))
    u=model.coordinates(x[:n]);features=((model.filtered(x[:n])-model.input_mean)/model.input_scale).double().detach()
    parameters={key:value.detach().double().requires_grad_() for key,value in model.backbone.named_parameters()}
    support=torch.as_tensor(b.assignment(n,cfg['seed'])!=0,device=x.device)
    logarithm=model.local_task_log_metric.detach().clone().requires_grad_()
    def objective(s,stop=False):
        near,weight,_=model.graph(u,s,torch.float64,support)
        value,_=unroll(parameters,features,near,weight,y.double(),support,steps=4,detach_inner=stop)
        return value,near
    value,near=objective(logarithm);gradient=torch.autograd.grad(value,logarithm)[0]
    direction=torch.linspace(-1,1,185,device=x.device,dtype=torch.float64);direction/=direction.norm()
    eps=1e-6;plus,np1=objective(logarithm.detach()+eps*direction);minus,nm1=objective(logarithm.detach()-eps*direction)
    assert torch.equal(near,np1) and torch.equal(near,nm1)
    finite=(plus-minus)/(2*eps);analytic=gradient@direction
    error=abs(float(finite.detach()-analytic));assert error<1e-6
    stopped,_=objective(logarithm,True);partial=torch.autograd.grad(stopped,logarithm)[0]
    indirect=float((gradient-partial).norm());assert indirect>1e-10
    q=0;j=1;w=model.field_weights(u[[q,j]]);g=model.field_directions;root=torch.diag((.5*logarithm).exp())
    matrices=[root@((185*g.T@(wi[:,None]*g)+.01*torch.eye(185,device=x.device))/1.01)@root for wi in w]
    delta=u[q]-u[j];exact=delta@((matrices[0]+matrices[1])*.5)@delta
    distance,_=model.corrected_distance(u,logarithm)
    assert torch.allclose(exact,distance[q,j],atol=1e-8,rtol=1e-8)
    assert all(float(torch.linalg.eigvalsh(a.symmetric(matrix)).min().detach())>0 for matrix in matrices)
    r.write_json(output/'hypergradient_checks.json',{'finite_difference_absolute_error':error,
                 'gradient_norm':float(gradient.norm()),'stop_inner_gradient_difference':indirect,
                 'original_backbone_and_rng_restored':True,'explicit_congruence_distance':True})
    print((output/'hypergradient_checks.json').read_text(),flush=True)


def run(output,preflight_dir):
    assert (preflight_dir/'hypergradient_checks.json').is_file()
    sources=[Path(local.__file__),Path(balance.__file__)];hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources}
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.26',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GU'},new_arms=('GE','GL'),max_gnn_epochs=200,
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,
                             'frozen_preprocessing_kernel_solves':25,'message_source':'v0.20 ordinary GE',
                             'teacher_Adam_steps':100,'geometry_updates':25,'inner_SGD_steps_per_update':4,
                             'final_Adam_maximum':200,'total_backbone_update_maximum':400,'fitted_metric_degrees':185,
                             'internal_query_is_independent_validation':False,'additional_source_hashes':hashes})
    for path,value in hashes.items():
        p=r.ROOT/path;assert r.digest(p)==value;(output/p.name).write_bytes(p.read_bytes())
    balance.summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)
