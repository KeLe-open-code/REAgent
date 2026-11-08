"""Identity-initialized local metric refinements of the original feature graph."""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import relation_graphsage as r
import torch
from torch import nn
from torch.nn import functional as F

TASK = 'CARBON-RELATION-REFINE-001'
PREVIOUS = r.HERE / 'runs/CARBON-RELATION-001/attempt-02'


class RefinedSAGE(r.RelationSAGE):
    def __init__(self, arm, config):
        nn.Module.__init__(self)
        self.arm, self.k = arm, config['knn_k']
        self.backbone = r.Regressor('graphsage', config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['seed'] + 100000)
            self.conditioner = nn.Sequential(nn.Linear(185, 16), nn.ReLU(), nn.Linear(16, 185))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)

    def distance_parts(self, x, n_train):
        a = .5 * torch.tanh(self.conditioner(F.normalize(x, dim=1)))
        g = torch.exp(a - a.mean(dim=1, keepdim=True))
        xt, gt = x[:n_train], g[:n_train]
        base = (x.square().sum(1)[:, None] + xt.square().sum(1)[None, :] - 2*x@xt.T).clamp_min(0)
        if self.arm == 'GL':
            delta, dt = g-1, gt-1
            correction = .5 * (
                (delta*x.square()).sum(1)[:, None] + (dt*xt.square()).sum(1)[None, :]
                + delta@xt.square().T + x.square()@dt.T - 2*(delta*x)@xt.T - 2*x@(dt*xt).T)
        else:
            assert self.arm == 'GE'
            u = x * (torch.sqrt(g)-1)
            ut = u[:n_train]
            q = (2*x*u+u.square()).sum(1)
            correction = q[:, None] + q[:n_train][None, :] - 2*(x@ut.T + u@xt.T + u@ut.T)
        return base, correction, g

    def relations(self, x, n_train, edges):
        base, correction, g = self.distance_parts(x, n_train)
        distance = (base + correction).clamp_min(0)
        self_mask = torch.arange(len(x), device=x.device)[:, None] == torch.arange(n_train, device=x.device)
        distance = distance.masked_fill(self_mask, torch.inf)
        neighbors = torch.argsort(distance, dim=1, stable=True)[:, :self.k]
        original = edges[0, :n_train*self.k].reshape(n_train, self.k)
        tau = base[:n_train].gather(1, original).mean().clamp_min(1e-8).detach()
        weights = torch.softmax(-correction.gather(1, neighbors)/tau, dim=1)
        return neighbors, weights, g


def construct(arm, config, device):
    r.fix_seed(config['seed'], 1)
    return RefinedSAGE(arm, config).to(device)


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, cfg, data, pre = r.load_fold(region, fold)
            x, te, ee, target = r.tensors(data, pre, 'cuda')
            n = int(data['n_train'])
            original = r.construct('G0', cfg, 'cuda').eval()
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            with torch.no_grad():
                original_pred = original(x, n, ee)[0]
            for arm in ('GE', 'GL'):
                model = construct(arm, cfg, 'cuda').eval()
                assert torch.equal(torch.get_rng_state(), cpu_rng)
                assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
                for key, value in original.backbone.state_dict().items():
                    assert torch.equal(value, model.backbone.state_dict()[key])
                with torch.no_grad():
                    initial = model(x, n, ee)
                original_near = ee[0].reshape(len(x), 8)
                assert torch.equal(initial[2].sort(1).values, original_near.sort(1).values), (region, fold, arm)
                assert torch.equal(initial[3], torch.full_like(initial[3], .125))
                assert torch.allclose(initial[0], original_pred, atol=3e-6, rtol=3e-6)
                cases.append({'region':region, 'fold':fold, 'arm':arm,
                              'initial_max_scaled_difference':float((initial[0]-original_pred).abs().max()),
                              'initial_neighbor_sets_equal':True,'paired_cpu_cuda_rng':True})
    _, cfg, data, pre = r.load_fold('hunan', 1)
    x, te, ee, target = r.tensors(data, pre, 'cuda')
    n = int(data['n_train'])
    learned = {}
    for arm in ('GE','GL'):
        model = construct(arm,cfg,'cuda')
        before = r.cpu_state(model)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
        for step in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            pred = model(x[:n],n,te)[0]
            loss = F.mse_loss(pred,target); loss.backward()
            grad = float(model.conditioner[-1].weight.grad.norm())
            assert grad > 0 and torch.isfinite(loss)
            nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        model.eval()
        with torch.no_grad():
            full = model(x,n,ee)
            train = model(x[:n],n,te)
        assert torch.allclose(full[0][:n],train[0],atol=3e-6,rtol=3e-6)
        assert torch.isfinite(full[4]).all() and torch.all(full[4]>0)
        assert float(torch.log(full[4]).sum(1).abs().max()) < 2e-5
        checkpoint = output/(arm+'.pt')
        torch.save(r.cpu_state(model),checkpoint)
        restored = construct(arm,cfg,'cuda').eval()
        restored.load_state_dict(torch.load(checkpoint,weights_only=True))
        with torch.no_grad():
            assert torch.equal(restored(x,n,ee)[0],full[0])
        learned[arm]={'parameters':sum(p.numel() for p in model.parameters()),'local_gradient_norm':grad,
                      'parameter_max_change':float((model.conditioner[-1].weight.detach().cpu()-before['conditioner.2.weight']).abs().max()),
                      'query_train_max_difference':float((full[0][:n]-train[0]).abs().max()),'checkpoint_exact':True}
    result={'cases':cases,'learning':learned,'device':'cuda','recorded_at':datetime.now().astimezone().isoformat()}
    r.write_json(output/'preflight.json',result)
    print(json.dumps({'initial_graph_checks':len(cases),'learning':learned}),flush=True)


def run(output, preflight_dir, task_id=TASK, model_factory=construct, warmup_steps=0,
        auxiliary_weight=0, branch='RELATION/v0.2', static_arms=()):
    checked = json.loads((preflight_dir/'preflight.json').read_text())
    assert len(checked['cases'])==30
    output.mkdir(parents=True,exist_ok=False)
    hashes={str(Path(__file__).relative_to(r.ROOT)):r.digest(Path(__file__)),
            str(Path(r.__file__).relative_to(r.ROOT)):r.digest(Path(r.__file__))}
    factory_file = Path(sys.modules[model_factory.__module__].__file__).resolve()
    hashes[str(factory_file.relative_to(r.ROOT))] = r.digest(factory_file)
    cached={}
    for region in r.REGIONS:
        for fold in range(1,6):
            source=r.SOURCE/region/f'fold-{fold:02d}'
            for name in ('config.json','data/development.npz','data/preprocessor.npz','data/split.csv'):
                p=source/name; hashes[str(p.relative_to(r.ROOT))]=r.digest(p)
            previous=PREVIOUS/region/f'fold-{fold:02d}'/'G0'
            for p in previous.iterdir():
                if p.is_file(): cached[str(p.relative_to(r.ROOT))]=r.digest(p)
            destination=output/region/f'fold-{fold:02d}'/'G0'
            shutil.copytree(previous,destination)
    old_manifest=json.loads((PREVIOUS/'manifest.json').read_text())
    for path,value in old_manifest['source_hashes'].items():
        if '/data/' in path.replace('\\','/') or path.endswith('config.json') or path.endswith('baseline.py'):
            assert r.digest(r.ROOT/path)==value,path
    assert old_manifest['device']=='cuda'
    new_fits = 15 * (2 + len(static_arms))
    r.write_json(output/'manifest.json',{'task_id':task_id,'branch':branch,'device':'cuda',
                 'started_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,
                 'source_hashes':hashes,'cached_G0_hashes':cached,'cached_source':str(PREVIOUS.relative_to(r.ROOT)),
                 'new_fits':new_fits,'cached_fits':15,'warmup_steps':warmup_steps,'static_arms':static_arms,
                 'auxiliary_weight':auxiliary_weight,'final_holdout_accessed':False})
    for source in {Path(__file__).resolve(),Path(r.__file__).resolve(),factory_file}:
        (output/source.name).write_bytes(source.read_bytes())
    started=time.perf_counter()
    for fold in range(1,6):
        for region in r.REGIONS:
            for arm in (*static_arms,'GE','GL'):
                r.fit(arm,region,fold,output,'cuda',model_factory=model_factory,task_id=task_id,
                      warmup_steps=0 if arm in static_arms else warmup_steps,
                      auxiliary_weight=0 if arm in static_arms else auxiliary_weight)
    for path,value in {**hashes,**cached}.items():
        assert r.digest(r.ROOT/path)==value,path
    r.write_json(output/'execution.json',{'finished_at':datetime.now().astimezone().isoformat(),
                 'wall_seconds':time.perf_counter()-started,'new_fits':new_fits,'cached_fits':15,'all_inputs_unchanged':True})
    r.summarize(output)
    summary=json.loads((output/'summary.json').read_text())
    summary['fit_records']=summary.pop('fits') + 15 * len(static_arms)
    summary['new_fits']=new_fits; summary['cached_fits']=15
    r.write_json(output/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args(); r.setup()
    if args.mode=='preflight': preflight(args.output)
    else: run(args.output,args.preflight_dir)


if __name__=='__main__': main()
