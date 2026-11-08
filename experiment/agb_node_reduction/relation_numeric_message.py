"""Train-only numerical features before neighbor averaging, with matched capacity."""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import relation_membership as member

r, layer = member.r, member.layer
TASK = 'CARBON-RELATION-NUMERIC-MESSAGE-001'
NEW = ('OP', 'LIN', 'GE', 'GL')
BASE = r.HERE/'runs/CARBON-RELATION-POSTERIOR-MOMENT-001/attempt-01'
MATCH = r.HERE/'runs/CARBON-RELATION-MEMBERSHIP-001/attempt-02'
CACHED = {'G0': (BASE,'G0'), 'GR': (MATCH,'GE'), 'L0': (MATCH,'GL'),
          'GP': (BASE,'GP'), 'GN': (BASE,'GN'),
          'SG': (r.HERE/'runs/CARBON-RELATION-SPHERICAL-TRACE-001/attempt-01','GL')}


def ple(values, knots):
    left = knots[:, :-1]
    width = knots[:, 1:] - left
    active = width > 1e-12
    constant = ~active.any(1)
    width = width.clone(); active = active.clone()
    width[constant, 0] = 1; active[constant, 0] = True
    first = active.long().argmax(1)
    last = 3 - active.flip(1).long().argmax(1)
    position = (values.double()[:, :, None] - left[None]) / width.clamp_min(1e-12)[None]
    index = torch.arange(4, device=values.device)[None, :]
    encoded = position.clamp(0, 1)
    encoded = torch.where((index == first[:, None])[None], position.clamp_max(1), encoded)
    encoded = torch.where((index == last[:, None])[None], position.clamp_min(0), encoded)
    encoded = torch.where(((first == last)[:, None] & (index == first[:, None]))[None], position, encoded)
    return torch.where(active[None], encoded, torch.zeros_like(encoded)), active, width


class MessageEncoding:
    def initialize_encoding(self, mode):
        self.message_mode = mode
        self.register_buffer('message_knots', torch.zeros(185, 5, dtype=torch.float64))
        self.register_buffer('message_zero', torch.zeros(185, 4, dtype=torch.float64))
        old = self.backbone.first.lin_l
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(185*5, old.weight.shape[0], bias=old.bias is not None)
        with torch.no_grad():
            expanded.weight.zero_(); expanded.weight[:, :185].copy_(old.weight)
            if old.bias is not None: expanded.bias.copy_(old.bias)
        self.backbone.first.lin_l = expanded

    @torch.no_grad()
    def fit_encoding(self, features):
        quantiles = torch.linspace(0, 1, 5, device=features.device, dtype=torch.float64)
        self.message_knots.copy_(torch.quantile(features.double(), quantiles, dim=0).T)
        zero = torch.zeros((1, 185), device=features.device)
        self.message_zero.copy_(ple(zero, self.message_knots)[0][0])

    def message_features(self, features):
        encoded, active, width = ple(features, self.message_knots)
        if self.message_mode == 'linear':
            extra = (features.double()/width.sum(1))[:, :, None].expand(-1, -1, 4) * active[None]
        else:
            extra = encoded - self.message_zero[None]
        return torch.cat([features, extra.flatten(1).float()], dim=1)


class OriginalMessageSAGE(MessageEncoding, r.RelationSAGE):
    def __init__(self, config):
        super().__init__('G0', config)
        self.initialize_encoding('ple')

    def fit_reference(self, x, edges):
        self.fit_encoding(x)

    def forward(self, x, n_train, edges):
        near, weight, metric = self.relations(x, n_train, edges)
        values = self.message_features(x)
        aggregate = (values[near] * weight[:, :, None]).sum(1)
        hidden = self.backbone.first.lin_l(aggregate) + self.backbone.first.lin_r(x)
        hidden = self.backbone.dropout(torch.relu(hidden))
        embedding = torch.relu(self.weighted_layer(self.backbone.second, hidden, near, weight))
        return self.backbone.head(embedding).squeeze(-1), embedding, near, weight, metric


class NumericMessageSAGE(MessageEncoding, member.MembershipSAGE):
    def __init__(self, variant, config):
        super().__init__('GL' if variant == 'GL' else 'GE', config)
        self.initialize_encoding('linear' if variant == 'LIN' else 'ple')

    def features(self, x):
        return ((self.filtered(x)-self.input_mean)/self.input_scale).float()

    def fit_reference(self, x, edges):
        super().fit_reference(x, edges)
        self.fit_encoding(self.features(x))

    def forward(self, x, n_train, edges):
        hard_gradient = bool(self.field_ready) and self.training and torch.is_grad_enabled()
        if hard_gradient:
            u = self.coordinates(x)
            first = self.membership_graph(u, self.layer_log_metric[0])
            second = self.membership_graph(u, self.layer_log_metric[1])
        elif bool(self.field_ready):
            first, second = self.layer_graphs(x, n_train)
        else:
            first = second = self.relations(x, n_train, edges)
        features = self.features(x)
        encoded = self.message_features(features)
        if hard_gradient:
            def aggregate(values, graph):
                near, weights, _, membership, coefficient = graph
                return member.membership_aggregate(values, near, weights, membership, coefficient)
        else:
            def aggregate(values, graph):
                return (values[graph[0]] * graph[1][:, :, None]).sum(1)
        hidden = self.backbone.first.lin_l(aggregate(encoded, first)) + self.backbone.first.lin_r(features)
        hidden = self.backbone.dropout(torch.relu(hidden))
        raw = self.backbone.second.lin_l(aggregate(hidden, second)) + self.backbone.second.lin_r(hidden)
        embedding = torch.relu(raw)
        return self.backbone.head(embedding).squeeze(-1), embedding, *first[:3]


def construct(arm, config, device):
    r.fix_seed(config['seed'], 1)
    if arm == 'OP': return OriginalMessageSAGE(config).to(device)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as z:
            config = {**config, 'relation_n_train': int(z['n_train'])}
    return NumericMessageSAGE(arm, config).to(device)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    knots = torch.tensor([[0.,1.,2.,3.,4.]], dtype=torch.float64)
    values = torch.tensor([[-1.],[0.],[2.],[4.],[5.]], dtype=torch.float64)
    encoded, active, width = ple(values, knots)
    assert torch.allclose((encoded*width).sum(2)+knots[:, 0], values, atol=1e-12, rtol=0)
    assert not torch.equal(encoded[[1,3]].mean(0), encoded[2])
    repeated = knots.new_tensor([[0.,0.,0.,1.,1.],[2.,2.,2.,2.,2.]])
    assert torch.isfinite(ple(torch.tensor([[0.,2.],[1.,3.]]), repeated)[0]).all()
    cases = []
    for region in r.REGIONS:
        _, cfg, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, y = r.tensors(data, pre, 'cuda')
        for arm in NEW:
            model = construct(arm, cfg, 'cuda').eval()
            reference = (r.construct('G0', cfg, 'cuda') if arm == 'OP' else
                         member.construct('GL' if arm == 'GL' else 'GE', cfg, 'cuda')).eval()
            model.fit_reference(x[:n], te)
            if arm != 'OP':
                reference.fit_reference(x[:n], te)
                model.fit_relations(x[:n], y, te); reference.fit_relations(x[:n], y, te)
            with torch.no_grad():
                initial = model(x,n,ee)[0]
                original = reference(x,n,ee)[0]
                initial_diff = float((initial-original).abs().max())
                assert torch.allclose(initial, original, atol=3e-6, rtol=3e-6)
                assert model.backbone.first.lin_l.weight[:,185:].count_nonzero() == 0
            optimizer = model.make_optimizer(cfg) if hasattr(model,'make_optimizer') else torch.optim.Adam(
                model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            for _ in range(3):
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(x[:n],n,te)[0], y); loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                assert model.backbone.first.lin_l.weight.grad[:,185:].norm() > 0
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
            assert model.backbone.first.lin_l.weight[:,185:].norm() > 0
            model.eval()
            with torch.no_grad():
                full=model(x,n,ee)[0]
                train=model(x[:n],n,te)[0]
                single=model(torch.cat([x[:n],x[n:n+1]]),n,ee[:,:(n+1)*8])[0]
                assert torch.allclose(full[:n],train,atol=3e-6,rtol=3e-6)
                assert torch.allclose(full[n],single[n],atol=3e-6,rtol=3e-6)
            cases.append({'region':region,'arm':arm,'initial_prediction_difference':initial_diff,
                'parameters':sum(p.numel() for p in model.parameters()),'three_updates':True,
                'expanded_weight_gradient_and_update':True,'query_independence':True})
    r.write_json(output/'preflight.json',{'cases':cases,'PLE_literal_and_degenerate':True,
        'nonlinear_mean_noncommutation':True,'source_sha256':r.digest(Path(__file__))})
    print('PREFLIGHT',len(cases),'cases',flush=True)


def directory(output, region, fold, arm):
    root,name=CACHED[arm] if arm in CACHED else (output,arm)
    return root/region/f'fold-{fold:02d}'/name


def analyze(output):
    methods=(*CACHED,*NEW);points=[];folds=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,_=r.load_fold(region,fold);n=int(data['n_train']);y=data['y'];predictions={}
            for arm in methods:
                path=directory(output,region,fold,arm);saved=json.loads((path/'metrics.json').read_text())
                with np.load(path/'responses.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['sample_ids'],data['sample_ids']) and (z['neighbors']<n).all()
                    prediction=z['prediction'];predictions[arm]=prediction.copy()
                mt,mv=r.metrics(y[:n],prediction[:n]),r.metrics(y[n:],prediction[n:])
                assert abs(mt['rmse']-saved['train']['rmse'])<1e-7 and abs(mv['rmse']-saved['validation']['rmse'])<1e-7
                folds.append({'region':region,'fold':fold,'arm':arm,'rmse':mv['rmse'],'mae':mv['mae'],
                    'r2':mv['r2'],'train_rmse':mt['rmse'],'prediction_sd_ratio':float(prediction[n:].std()/y[n:].std()),
                    'parameters':saved['parameters'],'best_epoch':saved['best_epoch'],'epochs_run':saved['epochs_run'],
                    'elapsed_seconds':saved['elapsed_seconds']})
            q80=float(np.quantile(y[:n],.8))
            for i in range(n,len(y)):
                point={'sample_id':str(data['sample_ids'][i]),'region':region,'fold':fold,'observed_agb':float(y[i]),
                    'high_agb':int(y[i]>=q80),'missing_features':int(data['missing_mask'][i].sum()),
                    'max_abs_training_z':float(np.abs(data['x'][i]).max())}
                for arm,prediction in predictions.items():
                    point[arm+'_prediction']=float(prediction[i]);point[arm+'_squared_error']=float((prediction[i]-y[i])**2)
                points.append(point)
    assert len(points)==len({v['sample_id'] for v in points})==1421 and len(folds)==150
    lookup={(v['region'],v['fold'],v['arm']):v for v in folds};comparisons={};provinces={}
    for arm in NEW:
        comparisons[arm]={other:float(np.mean([100*(1-lookup[(reg,f,arm)]['rmse']/lookup[(reg,f,other)]['rmse'])
            for reg in r.REGIONS for f in range(1,6)])) for other in methods if other!=arm}
        provinces[arm]={reg:float(np.mean([100*(1-lookup[(reg,f,arm)]['rmse']/lookup[(reg,f,'G0')]['rmse'])
            for f in range(1,6)])) for reg in r.REGIONS}
    summary={'comparisons':comparisons,'province_gains':provinces,'new_fits':60,'cached_fits':90,'method_folds':150,'IDs':1421}
    r.write_csv(output/'all_1421_ids.csv',points);r.write_csv(output/'all_fold_metrics.csv',folds)
    r.write_json(output/'summary.json',summary);print('SUMMARY',json.dumps(summary),flush=True)


def run(output,preflight_dir):
    check=json.loads((preflight_dir/'preflight.json').read_text())
    assert check['source_sha256']==r.digest(Path(__file__)) and len(check['cases'])==12
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(r.HERE/'TASKS'/(TASK+'.md'),output/'task_frozen.md');shutil.copy2(Path(__file__),output/Path(__file__).name)
    sources=[Path(__file__),r.HERE/'relation_graphsage.py',r.HERE/'relation_membership.py',r.HERE/'relation_coupled.py',
             r.HERE/'relation_layer.py',r.HERE/'relation_local_egop.py',r.HERE/'relation_agop.py',r.HERE/'relation_soft_topk.py',r.HERE/'relation_unroll.py']
    for region in r.REGIONS:
        for fold in range(1,6): sources.extend(r.SOURCE/region/f'fold-{fold:02d}'/'data'/name for name in ('development.npz','preprocessor.npz'))
    hashes={str(path):r.digest(path) for path in sources}
    r.write_json(output/'manifest.json',{'task_id':TASK,'started_at':datetime.now().astimezone().isoformat(),
        'source_hashes':hashes,'new_fits':60,'cached_fits':90,'seed_offsets':[0],'final_holdout_accessed':False})
    started=time.perf_counter()
    for region in r.REGIONS:
        for fold in range(1,6):
            for arm in NEW: r.fit(arm,region,fold,output,'cuda',model_factory=construct,task_id=TASK,max_gnn_epochs=500 if arm=='OP' else 400)
    analyze(output)
    assert all(r.digest(Path(path))==value for path,value in hashes.items())
    r.write_json(output/'finished.json',{'finished_at':datetime.now().astimezone().isoformat(),
        'wall_seconds':time.perf_counter()-started,'source_hashes_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','run','analyze'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight': preflight(args.output)
    elif args.mode=='run':
        assert args.preflight_dir
        run(args.output,args.preflight_dir)
    else: analyze(args.output)
