"""A shared subspace for feature messages and train-only graph relations."""
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
import torch
from torch.nn import functional as F

TASK='CARBON-RELATION-SUBSPACE-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-LATENT-001/attempt-02'
NEW=('GB','GE','GR','GL')


class SubspaceSAGE(latent.LatentSAGE):
    def __init__(self,arm,config):
        super().__init__('GB',config)
        self.arm=arm
        self.projection.requires_grad_(arm!='GB')
        self.optimizer_mode='skewon' if arm=='GL' else 'normalized_riemannian_gradient'

    def distances(self,x,n_train):
        z=self.projection(x/self.group_scale-self.center)
        distance=(z[:,None,:]-z[None,:n_train,:]).square().sum(-1)
        mask=torch.arange(len(x),device=x.device)[:,None]==torch.arange(n_train,device=x.device)
        return distance.masked_fill(mask,torch.inf),torch.ones_like(z)

    def forward(self,x,n_train,edges):
        near,weight,g=self.relations(x,n_train,edges)
        z=self.projection(x/self.group_scale-self.center)
        filtered=(z@self.projection.weight+self.center)*self.group_scale
        h=self.weighted_layer(self.backbone.first,filtered,near,weight)
        h=self.backbone.dropout(torch.relu(h))
        embedding=torch.relu(self.weighted_layer(self.backbone.second,h,near,weight))
        return self.backbone.head(embedding).squeeze(-1),embedding,near,weight,g

    def make_optimizer(self,config):
        if self.arm in ('GR','GL'):return stiefel.StiefelOptimizer(self,config)
        return torch.optim.Adam(self.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])


def construct(arm,config,device):
    assert arm in NEW
    r.fix_seed(config['seed'],1)
    return SubspaceSAGE(arm,config).to(device)


def preflight(output):
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,cfg,d,p=r.load_fold(region,fold);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda')
            fixed=construct('GB',cfg,'cuda').eval();fixed.fit_reference(x[:n],te)
            cpu,cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
            with torch.no_grad():base=fixed(x,n,ee)
            for arm in ('GE','GR','GL'):
                model=construct(arm,cfg,'cuda').eval();model.fit_reference(x[:n],te)
                assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
                with torch.no_grad():v=model(x,n,ee)
                assert torch.equal(v[0],base[0]) and torch.equal(v[2],base[2])
                assert sum(q.numel() for q in model.parameters() if q.requires_grad)==30865
                cases.append({'region':region,'fold':fold,'arm':arm,'paired_initial_graph_rng':True})
    _,cfg,d,p=r.load_fold('hunan',1);n=int(d['n_train']);x,te,ee,y=r.tensors(d,p,'cuda');checks={}
    for arm in NEW:
        model=construct(arm,cfg,'cuda');model.fit_reference(x[:n],te);before=r.cpu_state(model)
        optimizer=model.make_optimizer(cfg)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True);F.mse_loss(model(x[:n],n,te)[0],y).backward()
            if arm=='GB':assert model.projection.weight.grad is None
            else:assert model.projection.weight.grad.norm()>0
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():full=model(x,n,ee);alone=model(x[:n],n,te)
        assert torch.allclose(full[0][:n],alone[0],atol=3e-6,rtol=3e-6)
        change=float((model.projection.weight.detach().cpu()-before['projection.weight']).abs().max())
        assert (change==0) if arm=='GB' else (change>0)
        if arm in ('GR','GL'):assert optimizer.maximum_error<2e-6
        path=output/(arm+'.pt');torch.save(r.cpu_state(model),path)
        restored=construct(arm,cfg,'cuda').eval();restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():assert torch.equal(full[0],restored(x,n,ee)[0])
        checks[arm]={'projection_change':change,'query_isolation':True,'checkpoint_exact':True}
        if hasattr(optimizer,'diagnostics'):checks[arm].update(optimizer.diagnostics())
    r.write_json(output/'preflight.json',{'cases':cases,'learning':checks});print(json.dumps(checks),flush=True)


def summarize_controls(output):
    r.summarize(output)
    summary=json.loads((output/'summary.json').read_text());gains={a:[] for a in ('GB','GR')}
    for region in r.REGIONS:
        for arm in gains:
            rows=[]
            for fold in range(1,6):
                directory=output/region/f'fold-{fold:02d}'
                control=json.loads((directory/arm/'metrics.json').read_text())
                gl=json.loads((directory/'GL/metrics.json').read_text())
                _,_,d,_=r.load_fold(region,fold);n=int(d['n_train'])
                with np.load(directory/arm/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'],d['sample_ids'])
                    measured=r.metrics(d['y'][n:],z['prediction'][n:])
                assert all(abs(measured[k]-control['validation'][k])<1e-10 for k in measured)
                if arm=='GR':assert control['projection_orthogonality_error']<2e-6
                assert gl['projection_orthogonality_error']<2e-6
                gains[arm].append(100*(1-gl['validation']['rmse']/measured['rmse']));rows.append(measured)
            summary['province_means'][region][arm]={k:float(np.mean([row[k] for row in rows])) for k in ('rmse','mae','r2')}
    for arm,values in gains.items():summary[f'GL_vs_{arm}_gain_percent']=float(np.mean(values))
    if any(np.mean(v)<=0 for v in gains.values()):summary['classification']='NO_GAIN'
    summary.pop('fits');summary.update(fit_records=75,new_fits=60,cached_fits=15)
    r.write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


def run(output,preflight_dir):
    checked=json.loads((preflight_dir/'preflight.json').read_text());assert len(checked['cases'])==45 and len(checked['learning'])==4
    old=json.loads((PREVIOUS/'manifest.json').read_text())
    for path,value in old['source_hashes'].items():
        if '/data/' in path.replace('\\','/') or path.endswith('config.json'):assert r.digest(r.ROOT/path)==value,path
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),Path(r.__file__),Path(latent.__file__),Path(stiefel.__file__),r.BASE/'baseline.py']
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources};cached={}
    for region in r.REGIONS:
        for fold in range(1,6):
            source=r.SOURCE/region/f'fold-{fold:02d}'
            for name in ('config.json','data/development.npz','data/preprocessor.npz','data/split.csv'):
                p=source/name;hashes[str(p.relative_to(r.ROOT))]=r.digest(p)
            src=PREVIOUS/region/f'fold-{fold:02d}'/'G0'
            for p in src.iterdir():
                if p.is_file():cached[str(p.relative_to(r.ROOT))]=r.digest(p)
            shutil.copytree(src,output/region/f'fold-{fold:02d}'/'G0')
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    r.write_json(output/'manifest.json',{'task_id':TASK,'branch':'RELATION/v0.8','device':'cuda',
                'started_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'source_hashes':hashes,
                'cached_hashes':cached,'cache_source':str(PREVIOUS.relative_to(r.ROOT)),
                'preflight_sha256':r.digest(preflight_dir/'preflight.json'),
                'new_fits':60,'cached_fits':15,'final_holdout_accessed':False})
    start=time.perf_counter()
    for fold in range(1,6):
        for region in r.REGIONS:
            for arm in NEW:r.fit(arm,region,fold,output,'cuda',model_factory=construct,task_id=TASK)
    for path,value in {**hashes,**cached}.items():assert r.digest(r.ROOT/path)==value,path
    r.write_json(output/'execution.json',{'finished_at':datetime.now().astimezone().isoformat(),
                'wall_seconds':time.perf_counter()-start,'new_fits':60,'cached_fits':15,'all_inputs_unchanged':True})
    summarize_controls(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)


if __name__=='__main__':main()
