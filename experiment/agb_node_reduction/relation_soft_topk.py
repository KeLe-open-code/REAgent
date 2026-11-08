"""Hard eight-neighbor GraphSAGE with a soft membership geometry surrogate.

The custom mask has its exact implicit gradient. The model's straight-through
backward is intentionally biased and is not the derivative of hard top-k.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import relation_graphsage as r
import relation_agop as a
import relation_layer as layer
import relation_coupled as coupled

TASK='CARBON-RELATION-SOFT-TOPK-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-COUPLED-001/attempt-01'
ARMS=('G0','GB','GS','GU','GQ','GP','GE','GJ','GL')


class SigmoidTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,k):
        valid=torch.isfinite(scores)
        count=valid.sum(1,keepdim=True).to(scores.dtype)
        offset=torch.log(k/(count-k))
        low=offset-scores.amax(1,keepdim=True)
        high=offset-scores.masked_fill(~valid,torch.inf).amin(1,keepdim=True)
        for _ in range(48):
            middle=(low+high)*.5
            below=torch.sigmoid(scores+middle).sum(1,keepdim=True)<k
            low=torch.where(below,middle,low)
            high=torch.where(below,high,middle)
        membership=torch.sigmoid(scores+(low+high)*.5)
        ctx.save_for_backward(membership)
        return membership

    @staticmethod
    def backward(ctx,upstream):
        (membership,)=ctx.saved_tensors
        derivative=membership*(1-membership)
        correction=(derivative*upstream).sum(1,keepdim=True)/derivative.sum(1,keepdim=True).clamp_min(1e-30)
        return derivative*(upstream-correction),None


def relaxed_weights(scores):
    membership=SigmoidTopK.apply(scores,8)
    weights=membership*torch.softmax(scores,dim=1)
    return weights/weights.sum(1,keepdim=True),membership


def surrogate_layer(module,features,near,hard_weights,soft_weights):
    hard=(features[near]*hard_weights.detach()[:,:,None]).sum(1)
    soft=soft_weights.to(features.dtype)@features[:soft_weights.shape[1]].detach()
    aggregate=hard+(soft-soft.detach())
    return module.lin_l(aggregate)+module.lin_r(features)


class SoftTopKSAGE(coupled.CoupledLayerSAGE):
    def soft_graph(self,u,log_metric):
        distance,diagonal=self.corrected_distance(u,log_metric)
        n=len(self.field_support)
        mask=torch.arange(len(u),device=u.device)[:,None]==torch.arange(n,device=u.device)[None,:]
        distance=distance.masked_fill(mask,torch.inf)
        near=torch.argsort(distance,dim=1,stable=True)[:,:8]
        scores=-distance/self.temperature
        hard=torch.softmax(scores.gather(1,near),dim=1).float()
        soft,membership=relaxed_weights(scores)
        return near,hard,diagonal.float(),soft,membership

    def forward(self,x,n_train,edges):
        if not bool(self.field_ready) or not self.training or not torch.is_grad_enabled():
            return super().forward(x,n_train,edges)
        assert len(x)==n_train==len(self.field_support)
        u=self.coordinates(x)
        first=self.soft_graph(u,self.layer_log_metric[0])
        second=first if self.arm=='GJ' else self.soft_graph(u,self.layer_log_metric[1])
        features=((self.filtered(x)-self.input_mean)/self.input_scale).float()
        hidden=surrogate_layer(self.backbone.first,features,*first[:2],first[3])
        hidden=self.backbone.dropout(torch.relu(hidden))
        embedding=torch.relu(surrogate_layer(self.backbone.second,hidden,*second[:2],second[3]))
        return self.backbone.head(embedding).squeeze(-1),embedding,*first[:3]


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return SoftTopKSAGE(arm,config).to(device)


def preflight(output):
    layer.preflight(output,model_factory=construct)
    sample=torch.linspace(-3,4,33,device='cuda',dtype=torch.float64).reshape(3,11).requires_grad_()
    assert torch.autograd.gradcheck(lambda value:SigmoidTopK.apply(value,3),(sample,),eps=1e-5,atol=1e-6,rtol=1e-4)
    inclusion=SigmoidTopK.apply(sample,3)
    mass_error=float((inclusion.sum(1)-3).abs().max().detach())
    assert mass_error<1e-10
    assert torch.allclose(inclusion,SigmoidTopK.apply(sample+5,3),atol=1e-12,rtol=0)
    probes=[]
    for region in r.REGIONS:
        _,cfg,data,pre=r.load_fold(region,1);n=int(data['n_train']);x,te,_,y=r.tensors(data,pre,'cuda')
        saved=torch.load(output/f'{region}_GL.pt',map_location='cuda',weights_only=False)
        model=construct('GL',cfg,'cuda').train();model.load_state_dict(saved['state_dict'])
        parent=coupled.construct('GL',cfg,'cuda').train();parent.load_state_dict(saved['state_dict'])
        rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        prediction=model(x[:n],n,te)[0]
        F.mse_loss(prediction,y).backward()
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1])
        reference=parent(x[:n],n,te)[0]
        F.mse_loss(reference,y).backward()
        assert torch.equal(prediction,reference)
        backbone_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(model.backbone.parameters(),parent.backbone.parameters()))
        assert backbone_error==0
        geometry_change=float((model.layer_log_metric.grad-parent.layer_log_metric.grad).norm())
        assert geometry_change>1e-10
        # Isolate direct candidate-message derivatives from indirect field paths.
        model.eval();u=model.coordinates(x[:n]);graphs=[];score_leaves=[]
        for block in model.layer_log_metric:
            distance,_=model.corrected_distance(u,block)
            scores=(-distance.detach()/model.temperature).masked_fill(torch.eye(n,device=x.device,dtype=torch.bool),-torch.inf).requires_grad_()
            near=torch.argsort(-scores,dim=1,stable=True)[:,:8]
            hard=torch.softmax(scores.gather(1,near),dim=1).float()
            soft,membership=relaxed_weights(scores)
            assert float((membership.sum(1)-8).abs().max())<1e-8
            assert torch.equal(membership.diag(),torch.zeros(n,device=x.device,dtype=torch.float64))
            graphs.append((near,hard,soft));score_leaves.append(scores)
        features=((model.filtered(x[:n])-model.input_mean)/model.input_scale).float().detach()
        hidden=torch.relu(surrogate_layer(model.backbone.first,features,*graphs[0]))
        embedding=torch.relu(surrogate_layer(model.backbone.second,hidden,*graphs[1]))
        loss=F.mse_loss(model.backbone.head(embedding).squeeze(-1),y)
        gradients=torch.autograd.grad(loss,score_leaves)
        for index,(gradient,graph) in enumerate(zip(gradients,graphs),1):
            outside=torch.ones_like(gradient,dtype=torch.bool).scatter_(1,graph[0],False)
            outside.fill_diagonal_(False)
            nonselected_norm=float(gradient[outside].norm())
            assert nonselected_norm>1e-10 and torch.isfinite(gradient).all()
            probes.append({'region':region,'layer':index,'hard_forward_exact':True,'backbone_gradient_max_error':backbone_error,
                           'geometry_gradient_difference_norm':geometry_change,'nonselected_direct_score_gradient_norm':nonselected_norm,
                           'nonselected_gradient_fraction':float(gradient[outside].abs().sum()/gradient.abs().sum()),
                           'nonselected_nonzero_fraction':float((gradient[outside]!=0).double().mean())})
    r.write_json(output/'soft_topk_checks.json',{'implicit_mask_gradcheck':True,'mask_mass_error':mass_error,
                 'straight_through_is_biased_not_hard_derivative':True,'probes':probes})
    print((output/'soft_topk_checks.json').read_text(),flush=True)


@torch.no_grad()
def relaxation_diagnostics(output):
    rows=[];points=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,pre=r.load_fold(region,fold);n=int(data['n_train']);x,_,_,_=r.tensors(data,pre,'cuda')
            for arm in ('GE','GJ','GL'):
                saved=torch.load(output/region/f'fold-{fold:02d}'/arm/'best.pt',map_location='cuda',weights_only=False)
                model=construct(arm,saved['config'],'cuda').eval();model.load_state_dict(saved['state_dict'])
                u=model.coordinates(x);outside_mass=[]
                for index,block in enumerate(model.layer_log_metric,1):
                    near,_,_,soft,membership=model.soft_graph(u,block)
                    error=float((membership.sum(1)-8).abs().max());assert error<1e-6
                    outside=1-soft.gather(1,near).sum(1);outside_mass.append(outside.cpu().numpy())
                    rows.append({'region':region,'fold':fold,'arm':arm,'layer':index,'membership_mass_max_error':error,
                                 'train_soft_mass_outside_hard8':float(outside[:n].mean()),'validation_soft_mass_outside_hard8':float(outside[n:].mean())})
                if len(outside_mass)==1:outside_mass*=2
                for i in range(n,len(x)):
                    points.append({'region':region,'fold':fold,'arm':arm,'sample_id':str(data['sample_ids'][i]),
                                   'layer1_soft_mass_outside_hard8':float(outside_mass[0][i]),'layer2_soft_mass_outside_hard8':float(outside_mass[1][i])})
    assert len(rows)==75 and len(points)==4263
    r.write_csv(output/'relaxation_diagnostics.csv',rows);r.write_csv(output/'all_ids_relaxation_diagnostics.csv',points)


def run(output,preflight_dir):
    checks=json.loads((preflight_dir/'soft_topk_checks.json').read_text());assert checks['implicit_mask_gradcheck']
    sources=[Path(coupled.__file__),Path(layer.__file__),Path(layer.local.__file__),Path(layer.unroll.__file__)]
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources}
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.32',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GU','GQ':'GE','GP':'GL'},new_arms=('GE','GJ','GL'),
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,'frozen_preprocessing_kernel_solves':25,
                             'maximum_joint_updates':400,'teacher_steps':0,'RDoG_epsilon':.01,'soft_membership_bisection_steps':48,
                             'geometry_backward':'biased cardinality constrained sigmoid-topk straight-through',
                             'backbone_backward':'actual hard8message only','training_and_inference_forward':'actual hard8message',
                             'original_feature_count':185,'graph_scope_of_responses':'first_layer_only','additional_source_hashes':hashes})
    for path,value in hashes.items():
        p=r.ROOT/path;assert r.digest(p)==value;(output/p.name).write_bytes(p.read_bytes())
    layer.export_graphs(output,model_factory=construct)
    layer.summarize(output,arms=ARMS,new_fits=45,cached_fits=90)
    relaxation_diagnostics(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run','summarize'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    elif args.mode=='run':run(args.output,args.preflight_dir)
    else:layer.summarize(args.output,arms=ARMS,new_fits=45,cached_fits=90)
