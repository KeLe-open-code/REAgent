"""Signed positive-precision adaptation for the native SG relation coordinates."""
import argparse
import itertools
import json
import shutil
import time
import weakref
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import relation_positive_distance as p

r,layer = p.r,p.layer
TASK,NEW = 'CARBON-RELATION-PRECISION-001',('RP','IP','FP')
ROOT = r.HERE/'runs'/TASK
CACHED = {**p.CACHED,'NS':(r.HERE/'runs/CARBON-RELATION-POSITIVE-DISTANCE-001/attempt-01','NS')}
DELTA,RHO,BETA = 1e-4,.01,.9
TRACE_NAMES = ('mean_before','sample','sigma_before','gradient','momentum_after',
               'precision_before','precision_after','direction','mean_after')


def assert_running():
    text = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    meta = json.loads(text.split('<!-- TASK_META',1)[1].split('TASK_META -->',1)[0])
    assert meta['status'] == 'RUNNING',meta['status']


def positive_update(precision, estimate):
    a = RHO*(estimate+DELTA-precision)/precision
    return .5*precision*((a+1).square()+1)


def normalize(direction, gradient):
    size = direction.norm(dim=1)
    return direction*(gradient.norm(dim=1)/torch.where(size>0,size,torch.ones_like(size)))[:,None]


class Optimizer:
    def __init__(self, model, config):
        self.model = weakref.proxy(model)
        self.body = torch.optim.Adam(model.backbone.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        self.mean = model.layer_log_metric.detach().clone()
        self.precision = torch.full_like(self.mean,1+DELTA)
        self.momentum = torch.zeros_like(self.mean)
        self.radius = self.mean.new_full((2,),.01)
        self.accumulator = self.mean.new_zeros(2)
        self.n = int(model.config['relation_n_train'])
        self.rng = torch.Generator(device='cpu').manual_seed(config['seed']+149000)
        self.steps,self.training = 0,False
        self.trace,self.scales,self.history = [],[],[]
        self.sample,self.sample_sigma = None,None

    @torch.no_grad()
    def mode(self, training):
        if training == self.training:
            return
        if training:
            self.sample_sigma = (self.n*self.precision).rsqrt()
            assert torch.isfinite(self.sample_sigma).all() and (self.sample_sigma>0).all()
            epsilon = torch.randn(self.mean.shape,generator=self.rng,dtype=torch.float64).to(self.mean.device)
            self.sample = self.mean+self.sample_sigma*epsilon
            assert torch.isfinite(self.sample).all()
            self.model.layer_log_metric.copy_(self.sample)
        else:
            self.model.layer_log_metric.copy_(self.mean)
        self.model._relation_cache.clear()
        self.training = training

    def zero_grad(self, set_to_none=True):
        self.body.zero_grad(set_to_none=set_to_none)
        self.model.layer_log_metric.grad = None

    @torch.no_grad()
    def step(self):
        assert self.training and self.sample is not None
        gradient = self.model.layer_log_metric.grad.detach().clone()
        assert torch.isfinite(gradient).all()
        self.body.step()
        before,old_precision = self.mean.clone(),self.precision.clone()
        estimate = gradient*(self.sample-before)/self.sample_sigma.square()
        if self.model.precision_arm == 'IP':
            estimate = estimate.mean(dim=1,keepdim=True).expand_as(estimate)
        if self.model.precision_arm != 'FP':
            self.precision = positive_update(old_precision,estimate)
        assert torch.isfinite(self.precision).all() and (self.precision>0).all()
        self.steps += 1
        self.momentum.mul_(BETA).add_(gradient,alpha=1-BETA)
        numerator = self.momentum/(1-BETA**self.steps)+DELTA*before
        direction = normalize(numerator/self.precision,gradient)
        isotropic = normalize(numerator/self.precision.mean(dim=1,keepdim=True),gradient)
        self.radius = torch.maximum(self.radius,before.norm(dim=1))
        self.accumulator += gradient.square().sum(1)
        factor = torch.where(self.accumulator>0,self.radius/self.accumulator.sqrt(),torch.zeros_like(self.radius))
        self.mean = before-factor[:,None]*direction
        assert torch.isfinite(self.mean).all()
        packed = torch.stack((before,self.sample,self.sample_sigma,gradient,self.momentum,
                              old_precision,self.precision,direction,self.mean))
        self.trace.append(packed.cpu().numpy())
        self.scales.append(torch.stack((self.radius,self.accumulator)).cpu().numpy())
        norm = gradient.norm(dim=1)
        difference = (direction-isotropic).norm(dim=1)/torch.where(norm>0,norm,torch.ones_like(norm))
        self.history.append(dict(step=self.steps,precision_min=float(self.precision.min()),precision_max=float(self.precision.max()),
            precision_CV=(self.precision.std(dim=1,unbiased=False)/self.precision.mean(1)).cpu().tolist(),
            isotropic_direction_difference=difference.cpu().tolist(),mean_norm=self.mean.norm(dim=1).cpu().tolist(),
            sample_displacement_norm=(self.sample-before).norm(dim=1).cpu().tolist(),
            sigma_min=float(self.sample_sigma.min()),sigma_max=float(self.sample_sigma.max()),
            direction_gradient_cosine=((direction*gradient).sum(1)/torch.where(norm>0,norm.square(),torch.ones_like(norm))).cpu().tolist()))

    def snapshot(self):
        return dict(step=self.steps,mean=self.mean.detach().cpu().clone(),precision=self.precision.detach().cpu().clone(),
            momentum=self.momentum.detach().cpu().clone(),radius=self.radius.detach().cpu().clone(),
            accumulator=self.accumulator.detach().cpu().clone(),training_count=self.n,
            body={name:value.detach().cpu().clone() for name,value in self.model.backbone.named_parameters()},
            latest_history=dict(self.history[-1]))

    def diagnostics(self):
        return dict(geometry_optimizer='signed_positive_precision_RDoG_direction_adaptation',arm=self.model.precision_arm,
            body_optimizer='native_Adam',body_lr=.001,body_weight_decay=1e-4,rho=RHO,beta1=BETA,delta=DELTA,
            precision_initial=1+DELTA,effective_sample_count=self.n,initial_radius=.01,
            steps=self.steps,inference_is_optimized_mean_not_temporal_average=True,
            state_scope='end_of_training;selected_precision.pt binds selected state',new_inference_parameters=0)


class PrecisionSAGE(p.PositiveDistanceSAGE):
    def __init__(self, arm, config):
        super().__init__('NS',config)
        self.precision_arm,self._optimizer = arm,None
        self._capture_candidate,self.selected_precision = False,None

    def make_optimizer(self, config):
        if self.precision_arm == 'NC':
            return p.Optimizer(self,config)
        self._optimizer = Optimizer(self,config)
        return self._optimizer

    def train(self, mode=True):
        optimizer = getattr(self,'_optimizer',None)
        if optimizer is not None:
            optimizer.mode(mode)
            if mode:
                self._capture_candidate = True
        return super().train(mode)

    def state_dict(self,*args,**kwargs):
        result = super().state_dict(*args,**kwargs)
        optimizer = getattr(self,'_optimizer',None)
        if optimizer is not None and optimizer.steps and not self.training and self._capture_candidate:
            self.selected_precision = optimizer.snapshot()
            self._capture_candidate = False
        return result

    def load_state_dict(self,*args,**kwargs):
        self._capture_candidate = False
        return super().load_state_dict(*args,**kwargs)


def construct(arm, config, device):
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as saved:
            config['relation_n_train'] = int(saved['n_train'])
    r.fix_seed(config['seed'],1)
    return PrecisionSAGE(arm,config).to(device)


def directory(output, region, fold, arm):
    if arm in CACHED:
        root,alias = CACHED[arm]
        return root/region/f'fold-{fold:02d}'/alias
    return output/region/f'fold-{fold:02d}'/arm


def source_hashes():
    hashes = p.source_hashes()
    paths = [Path(__file__),ROOT/'design_evidence.json',
        r.HERE/'RESULTS/CARBON-RELATION-BODY-SCHEDULE-FREE-001.attempt-01.md',
        r.HERE/'runs/CARBON-RELATION-BODY-SCHEDULE-FREE-001/body_diagnosis.json']
    for region in r.REGIONS:
        for fold in range(1,6):
            paths.extend(directory(None,region,fold,'NS')/name for name in ('best.pt','responses.npz','metrics.json'))
    for path in paths:
        hashes[str(path.resolve())] = r.digest(path)
    return hashes


def math_checks():
    before_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(149000)
    precision = torch.logspace(-8,8,19,dtype=torch.float64)[:,None]
    a = torch.tensor([-1e6,-100.,-2.,-1.,0.,1.,100.,1e6],dtype=torch.float64)[None,:]
    estimate = precision*(1+a/RHO)-DELTA
    updated = positive_update(precision,estimate)
    literal = .5*precision*((a+1).square()+1)
    assert torch.isfinite(updated).all() and (updated>=.5*precision).all()
    assert torch.allclose(updated,literal,atol=1e-12,rtol=1e-12)
    # Quadratic oracle: a signed Hessian diagonal is recovered by exact second moments.
    hessian = torch.randn(5,5,generator=generator,dtype=torch.float64)
    hessian = (hessian+hessian.T)/2
    mean = torch.randn(5,generator=generator,dtype=torch.float64)
    sigma = torch.linspace(.1,.5,5,dtype=torch.float64)
    xi = torch.tensor(list(itertools.product((-1.,1.),repeat=5)),dtype=torch.float64)
    sample = mean+sigma*xi
    gradients = sample@hessian.T
    recovered = (gradients*(sample-mean)/sigma.square()).mean(0)
    assert torch.allclose(recovered,hessian.diag(),atol=1e-13,rtol=1e-13)
    # Direction normalization restores the native RDoG update for scalar precision/no momentum/prior.
    parameter = nn.Parameter(torch.randn(2,185,generator=generator,dtype=torch.float64))
    native = layer.RDoG(parameter)
    mean_step = parameter.detach().clone(); radius = mean_step.new_full((2,),.01); accumulator = mean_step.new_zeros(2)
    errors = []
    for _ in range(5):
        gradient = torch.randn(2,185,generator=generator,dtype=torch.float64)
        radius = torch.maximum(radius,mean_step.norm(dim=1)); accumulator += gradient.square().sum(1)
        mean_step -= (radius/accumulator.sqrt())[:,None]*normalize(gradient/7,gradient)
        parameter.grad = gradient; native.step()
        errors.append(float((mean_step-parameter).abs().max()))
    assert max(errors) < 1e-13 and torch.equal(before_rng,torch.get_rng_state())
    return dict(valid=True,positive_identity=True,negative_curvature_allowed=True,
        quadratic_signed_diagonal_error=float((recovered-hessian.diag()).abs().max()),
        native_RDoG_limit_error=max(errors),default_rng_unchanged=True)


def prepared(arm, config, x, edges, target):
    model = construct(arm,config,'cuda')
    model.fit_reference(x,edges)
    optimizer = model.make_optimizer(config)
    model.eval(); model.fit_relations(x,target,edges)
    return model,optimizer


def train_step(model, optimizer, x, edges, target):
    model.train(); optimizer.zero_grad()
    loss = model.training_loss(x,target,edges)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in model.parameters())
    nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step(); model.eval()
    return float(loss.detach())


@torch.no_grad()
def sample_consumption(model, optimizer, x, edges):
    cpu,cuda = torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
    model.train()
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    private = optimizer.rng.get_state().clone()
    sample = model.layer_log_metric.detach().clone()
    model.train()
    assert torch.equal(private,optimizer.rng.get_state()) and torch.equal(sample,model.layer_log_metric)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        first = model(x,len(x),edges)
        torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda)
        model.layer_log_metric.copy_(optimizer.mean); model._relation_cache.clear()
        second = model(x,len(x),edges)
    model.layer_log_metric.copy_(sample); model._relation_cache.clear()
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    difference = float((first[0]-second[0]).abs().max())
    assert difference > 1e-12
    return dict(private_noise_did_not_advance_dropout_rng=True,repeated_train_did_not_resample=True,
                sampled_vs_mean_prediction_difference=difference,changed_neighbors=int((first[2]!=second[2]).sum()))


def preflight(output):
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter(); cases,native_cases = [],[]
    for region in r.REGIONS:
        _,cfg,data,pre = r.load_fold(region,1)
        n = int(data['n_train']); x,te,ee,target = r.tensors(data,pre,'cuda')
        sequences = []
        for disabled in (False,True):
            model = construct('NC',cfg,'cuda') if disabled else p.construct('NS',cfg,'cuda')
            model.fit_reference(x[:n],te); optimizer = model.make_optimizer(cfg)
            model.eval(); model.fit_relations(x[:n],target,te)
            losses = [train_step(model,optimizer,x[:n],te,target) for _ in range(4)]
            sequences.append((losses,r.cpu_state(model),torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()))
        first,second = sequences
        assert first[0] == second[0] and all(torch.equal(first[1][key],second[1][key]) for key in first[1])
        assert torch.equal(first[2],second[2]) and torch.equal(first[3],second[3])
        native_cases.append(dict(region=region,disabled_native4steps_bitexact=True))
        native = p.construct('NS',cfg,'cuda'); native.fit_reference(x[:n],te); native.eval()
        native.fit_relations(x[:n],target,te); native_state = r.cpu_state(native)
        with torch.no_grad():
            native_prediction = native(x,n,ee)[0].clone()
        for arm in NEW:
            clock = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            model,opt = prepared(arm,cfg,x[:n],te,target)
            assert all(torch.equal(value.detach().cpu(),native_state[name]) for name,value in model.state_dict().items())
            with torch.no_grad():
                assert torch.equal(model(x,n,ee)[0],native_prediction)
            noise = sample_consumption(model,opt,x[:n],te)
            opt.rng.manual_seed(cfg['seed']+149000); model.eval()
            for _ in range(7):
                train_step(model,opt,x[:n],te,target)
            selected = r.cpu_state(model); selected_runtime = model.selected_precision
            assert selected_runtime['step'] == 7
            with torch.no_grad():
                before = model(x,n,ee)[0].clone()
            train_step(model,opt,x[:n],te,target)
            model.load_state_dict(selected); model.eval()
            with torch.no_grad():
                full = model(x,n,ee)[0]
                assert torch.equal(full,before)
                assert torch.allclose(full[:n],model(x[:n],n,te)[0],atol=3e-6,rtol=3e-6)
                assert torch.allclose(full[n:n+1],model(torch.cat((x[:n],x[n:n+1])),n,ee)[0][n:],atol=3e-6,rtol=3e-6)
            assert torch.equal(selected['layer_log_metric'],selected_runtime['mean'])
            assert (selected_runtime['precision']>0).all()
            for name,_ in model.backbone.named_parameters():
                assert torch.equal(selected['backbone.'+name],selected_runtime['body'][name])
            other = construct(arm,cfg,'cuda').eval(); other.load_state_dict(selected)
            with torch.no_grad():
                assert torch.equal(full,other(x,n,ee)[0])
            if arm == 'IP':
                assert torch.equal(opt.precision,opt.precision[:,:1].expand_as(opt.precision))
            if arm == 'FP':
                assert torch.equal(opt.precision,torch.full_like(opt.precision,1+DELTA))
            cases.append(dict(region=region,arm=arm,updates=8,initial_mean_native_bitexact=True,
                selected7_restores_after8=True,precision_CV=opt.history[-1]['precision_CV'],
                direction_difference=opt.history[-1]['isotropic_direction_difference'],
                seconds=time.perf_counter()-clock,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),**noise))
    result = dict(valid=True,cases=cases,native_cases=native_cases,source_sha256=r.digest(Path(__file__)),
        source_input_hashes=source_hashes(),wall_seconds=time.perf_counter()-started,device='cuda')
    r.write_json(output/'preflight.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='source_input_hashes'}),flush=True)


@torch.no_grad()
def export(model, data, pre, output, restore=False):
    n = int(data['n_train']); x,te,ee,_ = r.tensors(data,pre,'cuda')
    model.eval(); prediction = model(x,n,ee)[0]
    selected = model.selected_precision
    record = json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    epoch = record['best_epoch']; assert selected['step'] == epoch
    assert torch.equal(selected['mean'],checkpoint['state_dict']['layer_log_metric'])
    for name,_ in model.backbone.named_parameters():
        assert torch.equal(checkpoint['state_dict']['backbone.'+name],selected['body'][name])
    torch.save(selected,output/'selected_precision.pt')
    np.savez_compressed(output/'precision_trace.npz',packed=np.stack(model._optimizer.trace),
        scales=np.stack(model._optimizer.scales),names=np.array(TRACE_NAMES),seed=model.config['seed']+149000,
        n_train=n,rho=RHO,beta1=BETA,delta=DELTA)
    r.write_json(output/'precision_history.json',model._optimizer.history)
    r.write_json(output/'spherical_loss_history.json',model.geometry_history)
    arrays = {}
    for index,(near,weight,diagonal) in enumerate(model.layer_graphs(x,n)):
        arrays.update({f'layer{index}_near':near.cpu().numpy(),f'layer{index}_weights':weight.cpu().numpy(),f'layer{index}_diagonal':diagonal.cpu().numpy()})
    np.savez_compressed(output/'two_layer_graphs.npz',**arrays)
    with np.load(output/'responses.npz') as saved:
        raw = prediction.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
        assert np.array_equal(raw,saved['prediction'])
    if restore:
        other = construct(model.precision_arm,model.config,'cuda').eval()
        other.load_state_dict(checkpoint['state_dict'])
        assert torch.equal(prediction,other(x,n,ee)[0])
    history = selected['latest_history']
    active = bool(epoch>=2 and min(history['mean_norm'])>1e-10 and
        min(history['precision_CV'])>=1e-4 and min(history['isotropic_direction_difference'])>=1e-4)
    r.write_json(output/'precision_diagnostics.json',dict(arm=model.precision_arm,selected_active=active,
        best_epoch=epoch,precision_CV=history['precision_CV'],direction_difference=history['isotropic_direction_difference'],
        mean_norm=history['mean_norm'],parameter_count=28275,buffer_count=len(list(model.buffers())),
        precision_inference_requirement=False,selected_snapshot_matches_checkpoint=True,
        restored_prediction_exact=bool(restore)))


def analyze(output):
    folds,points,training = [],[],[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,_ = r.load_fold(region,fold); n = int(data['n_train']); y = data['y']; predictions = {}
            for arm in (*CACHED,*NEW):
                path = directory(output,region,fold,arm); metric = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                with np.load(path/'responses.npz') as saved:
                    assert np.array_equal(saved['sample_ids'],data['sample_ids']); predictions[arm] = saved['prediction'].copy()
                mt,mv = r.metrics(y[:n],predictions[arm][:n]),r.metrics(y[n:],predictions[arm][n:])
                assert all(abs(values[key]-metric[role][key]) < 1e-7 for role,values in [('train',mt),('validation',mv)] for key in values)
                row = dict(region=region,fold=fold,arm=arm,**mv,train_rmse=mt['rmse'],best_epoch=metric['best_epoch'],epochs_run=metric['epochs_run'],elapsed_seconds=metric['elapsed_seconds'])
                if arm in NEW:
                    row['selected_active'] = json.loads((path/'precision_diagnostics.json').read_text())['selected_active']
                folds.append(row)
            for i,sid in enumerate(data['sample_ids']):
                row = dict(sample_id=str(sid),region=region,fold=fold,observed_agb=float(y[i]),high_agb=int(y[i] >= np.quantile(y[:n],.8)))
                for arm,pred in predictions.items():
                    row[arm+'_prediction'],row[arm+'_SE'] = float(pred[i]),float((pred[i]-y[i])**2)
                (training if i < n else points).append(row)
    f,q,t = pd.DataFrame(folds),pd.DataFrame(points),pd.DataFrame(training)
    assert len(f) == 105 and len(q) == q.sample_id.nunique() == 1421 and len(t) == 5684
    old = pd.read_csv(p.context.PAIN/'all_1421_ids.csv',float_precision='round_trip')
    columns = ['sample_id','region','fold','missing_features','max_abs_training_standardized_feature']
    q = q.merge(old[columns],on=['sample_id','region','fold'],validate='one_to_one')
    q = q.merge(pd.read_csv(p.context.NAMED),on=['sample_id','region','fold'],validate='one_to_one')
    q = q.merge(pd.read_csv(p.context.TAIL)[['sample_id','region','training_appearances','tail_appearances']],on=['sample_id','region'],validate='one_to_one')
    for arm in ('NS',*NEW):
        q[arm+'_train_MSE'] = q.sample_id.map(t.groupby('sample_id')[arm+'_SE'].mean())
    q['RP_minus_NS_SSE'] = q.RP_SE-q.NS_SE
    q['training_RP_minus_NS_MSE'] = q.RP_train_MSE-q.NS_train_MSE
    q['failure_phase'] = np.select([(q.RP_minus_NS_SSE>0)&(q.training_RP_minus_NS_MSE>0),
        (q.RP_minus_NS_SSE>0)&(q.training_RP_minus_NS_MSE<=0),q.RP_minus_NS_SSE<0],
        ['training_and_OOF_worse','training_better_or_tied_OOF_worse','OOF_better'],default='OOF_tied')
    strata,features = [],[]
    for region,group in q.groupby('region'):
        for label,mask in dict(all=np.ones(len(group),bool),high=group.high_agb==1,ordinary=group.high_agb==0,
            missing=group.missing_features>0,complete=group.missing_features==0,
            complete_typical=(group.missing_features==0)&(group.max_abs_training_standardized_feature<=3),persistent_tail=group.tail_appearances>=3).items():
            selected = group[mask]
            if len(selected):
                strata.append(dict(region=region,stratum=label,n=len(selected),RP_minus_NS_SSE=float(selected.RP_minus_NS_SSE.sum()),
                    RP_bias=float((selected.RP_prediction-selected.observed_agb).mean()),NS_bias=float((selected.NS_prediction-selected.observed_agb).mean())))
        raw = {}
        for fold in range(1,6):
            _,_,data,_ = r.load_fold(region,fold); n = int(data['n_train'])
            raw.update({str(data['sample_ids'][i]):data['raw_features'][i] for i in range(n,len(data['y']))})
        matrix = np.stack([raw[sid] for sid in group.sample_id]); changes = group.RP_minus_NS_SSE.to_numpy()
        for j,name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:,j]; valid = np.isfinite(values)&(values != -9999)
            corr = pd.Series(values[valid]).corr(pd.Series(changes[valid]),method='spearman')
            features.append(dict(region=region,feature=name,valid_n=int(valid.sum()),spearman_RP_SSE_change=float(corr) if np.isfinite(corr) else None))
    assert len(features) == 555
    f.to_csv(output/'all_105_fold_metrics.csv',index=False)
    province = f.groupby(['region','arm'])[['rmse','mae','r2','bias','train_rmse']].mean()
    province.to_csv(output/'province_metrics.csv')
    q.to_csv(output/'all_1421_point_analysis.csv',index=False); t.to_csv(output/'all_5684_training_appearances.csv',index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv',index=False); pd.DataFrame(features).to_csv(output/'all_555_feature_contexts.csv',index=False)
    pd.concat([q.nlargest(20,'RP_minus_NS_SSE'),q.nsmallest(20,'RP_minus_NS_SSE')]).to_csv(output/'forty_cases.csv',index=False)
    wide = f.pivot(index=['region','fold'],columns='arm',values='rmse')
    gains = {arm:float((100*(1-wide[arm]/wide.G0)).mean()) for arm in (*CACHED,*NEW) if arm != 'G0'}
    compared = {arm:float((100*(1-wide.RP/wide[arm])).mean()) for arm in ('IP','FP','NS','SG','GR')}
    noninferior = cap = True
    for region in r.REGIONS:
        candidate,base = province.loc[region,'RP'],province.loc[region,'G0']
        noninferior &= candidate.rmse <= base.rmse+1e-6 and candidate.r2 >= base.r2-1e-6
        cap &= candidate.rmse <= base.rmse*1.01 and candidate.r2 >= base.r2-.01
    activity = {reg:int(f[(f.region==reg)&(f.arm=='RP')].selected_active.sum()) for reg in r.REGIONS}
    active = sum(activity.values()) >= 9 and min(activity.values()) >= 2
    full = gains['RP'] >= 10 and min(compared['IP'],compared['FP']) >= .5 and min(compared[a] for a in ('NS','SG','GR')) > 0 and noninferior and active
    pareto = gains['RP'] >= 3 and min(compared['NS'],compared['SG']) >= .5 and min(compared.values()) > 0 and cap and active
    summary = dict(task_id=TASK,G0_RMSE_gains_percent=gains,RP_control_gains_percent=compared,
        province_G0_noninferiority=bool(noninferior),province_cap=bool(cap),selected_RP_activity=activity,
        training_intervention_gate=bool(active),FULL_GAIN=bool(full),PARETO_GAIN=bool(pareto),
        classification='FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',all_development_IDs=1421,
        training_appearances=5684,feature_contexts=555,RP_helped_NS=int((q.RP_minus_NS_SSE<0).sum()),
        RP_harmed_NS=int((q.RP_minus_NS_SSE>0).sum()),RP_minus_NS_total_SSE=float(q.RP_minus_NS_SSE.sum()),
        failure_phase_counts=q.failure_phase.value_counts().to_dict(),final_holdout_access=False)
    r.write_json(output/'summary.json',summary); print(json.dumps(summary),flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert check['valid'] and len(check['cases']) == 9 and len(check['native_cases']) == 3
    assert check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(path)) == value for path,value in check['source_input_hashes'].items())
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,output/Path(__file__).name); shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md',output/'task_frozen.md')
    hashes = source_hashes()
    for path in (output/'task_frozen.md',preflight_dir/'preflight.json'):
        hashes[str(path.resolve())] = r.digest(path)
    started,tick = datetime.now().astimezone().isoformat(),time.perf_counter()
    r.write_json(output/'manifest.json',dict(task_id=TASK,branch='RELATION/v1.49',parent='RELATION/v1.48',change_type='native_adaptation',
        started_at=started,source_input_hashes=hashes,new_fits=45,cached_records=60,parameter_count=28275,
        device=torch.cuda.get_device_name(),rho=RHO,beta1=BETA,delta=DELTA,final_holdout_access=False))
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,pre = r.load_fold(region,fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name,config,device); holder.append(model); return model
                r.fit(arm,region,fold,output,'cuda',model_factory=factory,task_id=TASK,max_gnn_epochs=400)
                export(holder[0],data,pre,directory(output,region,fold,arm),restore=arm=='RP' and fold==1)
                holder.clear()
    elapsed = time.perf_counter()-tick; analyze(output)
    assert all(r.digest(Path(path)) == value for path,value in hashes.items())
    r.write_json(output/'finished.json',dict(started_at=started,finished_at=datetime.now().astimezone().isoformat(),
        fit_export_seconds=elapsed,wall_seconds=time.perf_counter()-tick,protected_hashes_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=('math','preflight','run','analyze'))
    parser.add_argument('--output',type=Path); parser.add_argument('--preflight-dir',type=Path); args = parser.parse_args()
    if args.mode != 'analyze': assert_running()
    r.setup()
    if args.mode == 'math': print(json.dumps(math_checks()),flush=True)
    elif args.mode == 'preflight': preflight(args.output)
    elif args.mode == 'run': run(args.output,args.preflight_dir)
    else: analyze(args.output)
