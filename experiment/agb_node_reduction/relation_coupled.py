"""Local field weights follow each layer's learned positive diagonal geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import relation_graphsage as r
import relation_agop as a
import relation_layer as layer

TASK='CARBON-RELATION-COUPLED-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-LAYER-001/attempt-01'
ARMS=('G0','GB','GS','GU','GE','GP','GJ','GL')


class CoupledLayerSAGE(layer.LayerSAGE):
    def endpoint_weights(self,u,log_metric):
        if self.arm=='GE':return self.field_weights(u),self.field_weights(self.field_support)
        root=(.5*log_metric).exp();q=u*root;s=self.field_support*root
        eye=torch.eye(185,device=u.device,dtype=u.dtype)
        reference=a.distance(s,s,eye)
        mask=torch.eye(len(s),device=u.device,dtype=torch.bool)
        tau=reference.masked_fill(mask,torch.inf).topk(8,largest=False).values.mean().clamp_min(1e-8)
        w=torch.softmax(-a.distance(q,s,eye)/tau+self.field_log_norm[None,:],dim=1)
        wt=torch.softmax(-reference/tau+self.field_log_norm[None,:],dim=1)
        return w,wt

    def corrected_distance(self,u,log_metric):
        w,wt=self.endpoint_weights(u,log_metric)
        root=(.5*log_metric).exp();q=u*root;s=self.field_support*root
        scores=q@self.field_directions.T;ts=s@self.field_directions.T
        left=(w*scores.square()).sum(1)[:,None]+w@ts.square().T-2*(w*scores)@ts.T
        right=scores.square()@wt.T+(wt*ts.square()).sum(1)[None,:]-2*scores@(wt*ts).T
        base=a.distance(q,s,torch.eye(185,device=u.device,dtype=u.dtype))
        distance=((185*(left+right)*.5+.01*base)/1.01).clamp_min(0)
        diagonal=(185*(w@self.field_directions.square())+.01)/1.01*log_metric.exp()
        return distance,diagonal


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return CoupledLayerSAGE(arm,config).to(device)


def preflight(output):
    layer.preflight(output,model_factory=construct)
    for region in r.REGIONS:
        current=torch.load(output/f'{region}_GE.pt',map_location='cpu',weights_only=False)['state_dict']
        old=torch.load(PREVIOUS.parent/'preflight-01'/f'{region}_GE.pt',map_location='cpu',weights_only=False)['state_dict']
        assert all(torch.equal(value,old[name]) for name,value in current.items())
    _,cfg,data,pre=r.load_fold('hunan',1);n=int(data['n_train']);x,te,_,y=r.tensors(data,pre,'cuda')
    model=construct('GL',cfg,'cuda').eval();saved=torch.load(output/'hunan_GL.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['state_dict']);u=model.coordinates(x[:n]).detach()
    features=((model.filtered(x[:n])-model.input_mean)/model.input_scale).double().detach()
    parameters={name:value.detach().double() for name,value in model.backbone.named_parameters()}
    logarithm=model.layer_log_metric.detach().clone().requires_grad_()
    def objective(s):
        graphs=[model.graph(u,block,torch.float64) for block in s];values=features
        for name,graph in zip(('first','second'),graphs):
            near,weights,_=graph;aggregate=(values[near]*weights[:,:,None]).sum(1)
            values=F.relu(F.linear(aggregate,parameters[name+'.lin_l.weight'],parameters[name+'.lin_l.bias'])
                          +F.linear(values,parameters[name+'.lin_r.weight']))
        pred=F.linear(values,parameters['head.weight'],parameters['head.bias']).squeeze(-1)
        return F.mse_loss(pred,y.double()),[g[0] for g in graphs]
    value,near=objective(logarithm);gradient=torch.autograd.grad(value,logarithm)[0]
    direction=torch.linspace(-1,1,370,device=x.device,dtype=torch.float64).reshape(2,185);direction/=direction.norm()
    epsilon=1e-6;plus,nplus=objective(logarithm.detach()+epsilon*direction);minus,nminus=objective(logarithm.detach()-epsilon*direction)
    assert all(torch.equal(p,q) and torch.equal(p,z) for p,q,z in zip(near,nplus,nminus))
    finite=(plus-minus)/(2*epsilon);analytic=(gradient*direction).sum();error=float((finite-analytic).abs().detach());assert error<1e-6
    weights,_=model.endpoint_weights(u[:3],logarithm[0]);index=torch.linspace(-1,1,n,device=x.device,dtype=torch.float64)
    field_gradient=torch.autograd.grad((weights*index[None,:]).sum(),logarithm)[0]
    field_norm=float(field_gradient.norm());assert field_norm>1e-10
    r.write_json(output/'coupled_checks.json',{'global_GE_three_real_steps_exact_all_three_provinces':True,
                 'field_weight_gradient_norm':field_norm,'full_loss_directional_derivative_error':error,
                 'gradient_norm':float(gradient.norm()),'no_query_features_in_field_bandwidth':True})
    print((output/'coupled_checks.json').read_text(),flush=True)


@torch.no_grad()
def field_diagnostics(output):
    rows=[];points=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,pre=r.load_fold(region,fold);n=int(data['n_train']);x,_,_,_=r.tensors(data,pre,'cuda')
            for arm in ('GJ','GL'):
                directory=output/region/f'fold-{fold:02d}'/arm
                saved=torch.load(directory/'best.pt',map_location='cuda',weights_only=False)
                model=construct(arm,saved['config'],'cuda').eval();model.load_state_dict(saved['state_dict'])
                u=model.coordinates(x);original=model.field_weights(u);changes=[]
                for s in model.layer_log_metric:
                    weights,_=model.endpoint_weights(u,s);changes.append(.5*(weights-original).abs().sum(1).cpu().numpy())
                if len(changes)==1:changes*=2
                rows.append({'region':region,'fold':fold,'arm':arm,'layer1_field_total_variation':float(changes[0][n:].mean()),
                             'layer2_field_total_variation':float(changes[1][n:].mean())})
                for i in range(n,len(x)):
                    points.append({'region':region,'fold':fold,'arm':arm,'sample_id':str(data['sample_ids'][i]),
                                   'layer1_field_total_variation':float(changes[0][i]),'layer2_field_total_variation':float(changes[1][i])})
    assert len(rows)==30 and len(points)==2842
    r.write_csv(output/'field_weight_diagnostics.csv',rows);r.write_csv(output/'all_ids_field_weight_changes.csv',points)


def run(output,preflight_dir):
    checks=json.loads((preflight_dir/'coupled_checks.json').read_text());assert checks['global_GE_three_real_steps_exact_all_three_provinces']
    sources=[Path(layer.__file__),Path(layer.local.__file__),Path(layer.unroll.__file__)]
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources}
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.31',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GU','GE':'GE','GP':'GL'},new_arms=('GJ','GL'),
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,'frozen_preprocessing_kernel_solves':25,
                             'maximum_joint_updates':400,'teacher_steps':0,'RDoG_epsilon':.01,
                             'field_localization':'learned layer diagonal coordinates with differentiable train-only8neighbor mean squared bandwidth',
                             'original_feature_count':185,'graph_scope_of_responses':'first_layer_only','additional_source_hashes':hashes})
    for path,value in hashes.items():
        p=r.ROOT/path;assert r.digest(p)==value;(output/p.name).write_bytes(p.read_bytes())
    layer.export_graphs(output,model_factory=construct)
    layer.summarize(output,arms=ARMS,new_fits=30,cached_fits=90)
    field_diagnostics(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)
