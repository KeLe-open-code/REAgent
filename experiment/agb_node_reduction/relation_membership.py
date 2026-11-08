"""Preserve hard edge-weight derivatives and add only membership derivatives."""
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
import relation_soft_topk as soft

TASK='CARBON-RELATION-MEMBERSHIP-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-SOFT-TOPK-001/attempt-01'
ARMS=('G0','GB','GS','GU','GQ','GP','GR','GE','GJ','GL')


class StableSigmoidTopK(soft.SigmoidTopK):
    """Same implicit mask; center at the cutoff to avoid a huge root bracket."""
    @staticmethod
    def forward(ctx,scores,k):
        cutoff=torch.topk(scores,k,dim=1).values[:,-1:]
        centered=scores-cutoff
        limit=torch.isfinite(scores).sum(1,keepdim=True).to(scores.dtype).log()+40
        low=-limit;high=limit
        for _ in range(48):
            middle=(low+high)*.5
            below=torch.sigmoid(centered+middle).sum(1,keepdim=True)<k
            low=torch.where(below,middle,low)
            high=torch.where(below,high,middle)
        membership=torch.sigmoid(centered+(low+high)*.5)
        ctx.save_for_backward(membership)
        return membership


def membership_aggregate(features,near,hard_weights,membership,coefficient):
    hard=(features[near]*hard_weights[:,:,None]).sum(1)
    tangent=((membership-membership.detach())*coefficient.detach()).to(features.dtype)
    extra=tangent@features[:tangent.shape[1]].detach()-tangent.sum(1,keepdim=True)*hard.detach()
    return hard+extra


class MembershipSAGE(soft.SoftTopKSAGE):
    def membership_graph(self,u,log_metric):
        distance,diagonal=self.corrected_distance(u,log_metric)
        n=len(self.field_support)
        mask=torch.arange(len(u),device=u.device)[:,None]==torch.arange(n,device=u.device)[None,:]
        distance=distance.masked_fill(mask,torch.inf)
        near=torch.argsort(distance,dim=1,stable=True)[:,:8]
        scores=-distance/self.temperature
        hard=torch.softmax(scores.gather(1,near),dim=1).float()
        membership=StableSigmoidTopK.apply(scores,8)
        exponential=torch.softmax(scores,dim=1)
        coefficient=exponential/exponential.gather(1,near).sum(1,keepdim=True)
        return near,hard,diagonal.float(),membership,coefficient

    def forward(self,x,n_train,edges):
        if not bool(self.field_ready) or not self.training or not torch.is_grad_enabled():
            return super().forward(x,n_train,edges)
        assert len(x)==n_train==len(self.field_support)
        u=self.coordinates(x)
        first=self.membership_graph(u,self.layer_log_metric[0])
        second=first if self.arm=='GJ' else self.membership_graph(u,self.layer_log_metric[1])
        features=((self.filtered(x)-self.input_mean)/self.input_scale).float()
        def apply(module,values,graph):
            near,hard,_,membership,coefficient=graph
            aggregate=membership_aggregate(values,near,hard,membership,coefficient)
            return module.lin_l(aggregate)+module.lin_r(values)
        hidden=self.backbone.dropout(torch.relu(apply(self.backbone.first,features,first)))
        embedding=torch.relu(apply(self.backbone.second,hidden,second))
        return self.backbone.head(embedding).squeeze(-1),embedding,*first[:3]


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return MembershipSAGE(arm,config).to(device)


def product_rule_check():
    scores=torch.linspace(-2,3,33,device='cuda',dtype=torch.float64).reshape(3,11).requires_grad_()
    features=torch.linspace(-1,2,44,device='cuda',dtype=torch.float64).reshape(11,4).requires_grad_()
    near=torch.argsort(-scores,dim=1,stable=True)[:,:8]
    hard=torch.softmax(scores.gather(1,near),dim=1)
    membership=StableSigmoidTopK.apply(scores,8)
    exponential=torch.softmax(scores,dim=1)
    coefficient=exponential/exponential.gather(1,near).sum(1,keepdim=True)
    custom=membership_aggregate(features,near,hard,membership,coefficient)
    mask=torch.zeros_like(scores).scatter_(1,near,1.)+(membership-membership.detach())
    weights=mask*exponential
    explicit=(weights/weights.sum(1,keepdim=True))@features
    direction=torch.linspace(-2,1,12,device='cuda',dtype=torch.float64).reshape(3,4)
    one=torch.autograd.grad((custom*direction).sum(),(scores,features),retain_graph=True)
    two=torch.autograd.grad((explicit*direction).sum(),(scores,features))
    errors={'forward':float((custom-explicit).abs().max().detach()),
            'score_gradient':float((one[0]-two[0]).abs().max()),'feature_gradient':float((one[1]-two[1]).abs().max())}
    assert max(errors.values())<1e-10
    outside=torch.ones_like(scores,dtype=torch.bool).scatter_(1,near,False)
    assert float(one[0][outside].norm())>1e-10
    return errors


def preflight(output):
    layer.preflight(output,model_factory=construct)
    checks=product_rule_check();probes=[]
    sample=torch.linspace(-3,4,33,device='cuda',dtype=torch.float64).reshape(3,11).requires_grad_()
    assert torch.autograd.gradcheck(lambda value:StableSigmoidTopK.apply(value,3),(sample,),eps=1e-5,atol=1e-6,rtol=1e-4)
    fixture=r.HERE/'runs/CARBON-RELATION-MEMBERSHIP-001/attempt-01/numerical_failure_hunan_3_GE_1.pt'
    scores=torch.load(fixture,map_location='cuda',weights_only=False)['scores']
    repaired=StableSigmoidTopK.apply(scores,8)
    repaired_error=float((repaired.sum(1)-8).abs().max());assert repaired_error<1e-9
    # Equal scores and extreme separated tails have the same required mass.
    cases=torch.cat([torch.zeros((1,20),device='cuda',dtype=torch.float64),
                     torch.cat([torch.zeros(8,device='cuda',dtype=torch.float64),torch.full((12,),-1e20,device='cuda',dtype=torch.float64)])[None,:]])
    assert float((StableSigmoidTopK.apply(cases,8).sum(1)-8).abs().max())<1e-9
    for region in r.REGIONS:
        _,cfg,data,pre=r.load_fold(region,1);n=int(data['n_train']);x,te,_,y=r.tensors(data,pre,'cuda')
        saved=torch.load(output/f'{region}_GL.pt',map_location='cuda',weights_only=False)
        model=construct('GL',cfg,'cuda').train();model.load_state_dict(saved['state_dict'])
        parent=coupled.construct('GL',cfg,'cuda').train();parent.load_state_dict(saved['state_dict'])
        rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
        prediction=model(x[:n],n,te)[0];F.mse_loss(prediction,y).backward()
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1])
        reference=parent(x[:n],n,te)[0];F.mse_loss(reference,y).backward()
        assert torch.equal(prediction,reference)
        error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(model.backbone.parameters(),parent.backbone.parameters()))
        assert error==0
        difference=float((model.layer_log_metric.grad-parent.layer_log_metric.grad).norm());assert difference>1e-10
        probes.append({'region':region,'hard_prediction_exact':True,'backbone_gradient_error':error,
                       'membership_geometry_gradient_difference':difference})
    r.write_json(output/'membership_checks.json',{'product_rule_checks':checks,'probes':probes,'surrogate_is_biased':True,
                 'stable_mask_gradcheck':True,'failed_checkpoint_mass_error_after_repair':repaired_error})
    print((output/'membership_checks.json').read_text(),flush=True)


@torch.no_grad()
def membership_diagnostics(output):
    rows=[];points=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,pre=r.load_fold(region,fold);n=int(data['n_train']);x,_,_,_=r.tensors(data,pre,'cuda')
            for arm in ('GE','GJ','GL'):
                saved=torch.load(output/region/f'fold-{fold:02d}'/arm/'best.pt',map_location='cuda',weights_only=False)
                model=construct(arm,saved['config'],'cuda').eval();model.load_state_dict(saved['state_dict']);outside_rows=[]
                u=model.coordinates(x)
                for index,block in enumerate(model.layer_log_metric,1):
                    near,_,_,membership,_=model.membership_graph(u,block)
                    error=float((membership.sum(1)-8).abs().max());assert error<1e-6
                    outside=1-membership.gather(1,near).sum(1)/8;outside_rows.append(outside.cpu().numpy())
                    rows.append({'region':region,'fold':fold,'arm':arm,'layer':index,'membership_mass_max_error':error,
                                 'train_membership_outside_hard8':float(outside[:n].mean()),'validation_membership_outside_hard8':float(outside[n:].mean())})
                if len(outside_rows)==1:outside_rows*=2
                for i in range(n,len(x)):
                    points.append({'region':region,'fold':fold,'arm':arm,'sample_id':str(data['sample_ids'][i]),
                                   'layer1_membership_outside_hard8':float(outside_rows[0][i]),'layer2_membership_outside_hard8':float(outside_rows[1][i])})
    assert len(rows)==75 and len(points)==4263
    r.write_csv(output/'membership_diagnostics.csv',rows);r.write_csv(output/'all_ids_membership_diagnostics.csv',points)


def run(output,preflight_dir):
    checks=json.loads((preflight_dir/'membership_checks.json').read_text());assert len(checks['probes'])==3
    assert checks.get('stable_mask_gradcheck') is True
    assert checks['failed_checkpoint_mass_error_after_repair']<1e-9
    fixture=r.HERE/'runs/CARBON-RELATION-MEMBERSHIP-001/attempt-01/numerical_failure_hunan_3_GE_1.pt'
    numerical_gates={str(p.relative_to(r.ROOT)):r.digest(p) for p in (preflight_dir/'membership_checks.json',fixture)}
    sources=[Path(soft.__file__),Path(coupled.__file__),Path(layer.__file__),Path(layer.local.__file__),Path(layer.unroll.__file__)]
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources}
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.33',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GU','GQ':'GQ','GP':'GP','GR':'GL'},new_arms=('GE','GJ','GL'),
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,'frozen_preprocessing_kernel_solves':25,
                             'maximum_joint_updates':400,'teacher_steps':0,'RDoG_epsilon':.01,'soft_membership_bisection_steps':48,
                             'soft_membership_numerical_root':'cutoff-centered bracket plus log(candidate_count)+40 floating-point tail bound',
                             'numerical_preflight_hashes':numerical_gates,
                             'geometry_backward':'hard edge-weight derivative plus product-rule membership surrogate',
                             'backbone_backward':'actual hard8message only','training_and_inference_forward':'actual hard8message',
                             'original_feature_count':185,'graph_scope_of_responses':'first_layer_only','additional_source_hashes':hashes})
    for path,value in hashes.items():
        p=r.ROOT/path;assert r.digest(p)==value;(output/p.name).write_bytes(p.read_bytes())
    for path,value in numerical_gates.items():assert r.digest(r.ROOT/path)==value
    layer.export_graphs(output,model_factory=construct)
    layer.summarize(output,arms=ARMS,new_fits=45,cached_fits=105)
    membership_diagnostics(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run','summarize'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    elif args.mode=='run':run(args.output,args.preflight_dir)
    else:layer.summarize(args.output,arms=ARMS,new_fits=45,cached_fits=105)
