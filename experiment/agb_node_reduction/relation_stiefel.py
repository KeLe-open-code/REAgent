"""Stiefel-constrained relation projection with Skewon and normalized gradient controls."""
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
import torch
from torch.nn import functional as F

TASK='CARBON-RELATION-STIEFEL-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-LATENT-001/attempt-02'
CACHED={'G0':'G0','GB':'GB','GE':'GE','GA':'GL'}


def tangent(x,m):
    a=x.T@m
    return m-x@((a+a.T)/2)


def polar_ranked(a):
    u,s,vh=torch.linalg.svd(a,full_matrices=False)
    active=s>torch.clamp(s.max()*1e-10,min=1e-30)
    return (u*active.to(a.dtype)[None,:])@vh


def skewon_direction(x,m):
    # Algorithm 2 of arXiv:2608.06218v1, exact thresholded polar factor.
    q,rr=torch.linalg.qr(torch.cat((-x,m),dim=1),mode='reduced')
    p=x.shape[1];a,b=rr[:,:p],rr[:,p:]
    small=.5*(a@b.T-b@a.T)
    polar=polar_ranked(small);direction=-.5*(polar-polar.T)
    return q@(direction@(q.T@x))


class StiefelOptimizer:
    def __init__(self,model,config):
        self.p=model.projection.weight
        self.mode=model.optimizer_mode
        other=[v for v in model.parameters() if v is not self.p]
        self.adam=torch.optim.Adam(other,lr=config['learning_rate'],weight_decay=config['weight_decay'])
        self.m=torch.zeros_like(self.p.T,dtype=torch.float64)
        self.steps=0;self.maximum_error=0.;self.maximum_step=0.

    def zero_grad(self,set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.p.grad=None

    @torch.no_grad()
    def step(self):
        if self.p.grad is not None:
            x=self.p.T.double();self.m.mul_(.9).add_(self.p.grad.T.double(),alpha=.1)
            if self.mode=='skewon':direction=skewon_direction(x,self.m)
            else:
                gradient=tangent(x,self.m)
                direction=-gradient/torch.linalg.matrix_norm(gradient,ord=2).clamp_min(1e-30)
            q,rr=torch.linalg.qr(x+.01*direction,mode='reduced')
            q=q*torch.sign(rr.diag())[None,:]
            self.p.copy_(q.T.to(self.p.dtype))
            err=(self.p@self.p.T-torch.eye(len(self.p),device=self.p.device)).abs().max()
            self.maximum_error=max(self.maximum_error,float(err))
            self.maximum_step=max(self.maximum_step,float(torch.linalg.matrix_norm(direction,ord=2))*.01)
            self.steps+=1
        self.adam.step()

    def diagnostics(self):
        return {'mode':self.mode,'projection_steps':self.steps,'projection_step_size':.01,
                'momentum':.9,'maximum_orthogonality_error':self.maximum_error,
                'maximum_direction_step_spectral_norm':self.maximum_step}


class StiefelSAGE(latent.LatentSAGE):
    def __init__(self,arm,config):
        super().__init__('GL',config)
        self.optimizer_mode='skewon' if arm=='GL' else 'normalized_riemannian_gradient'

    def make_optimizer(self,config):
        return StiefelOptimizer(self,config)


def construct(arm,config,device):
    assert arm in ('GR','GL')
    r.fix_seed(config['seed'],1)
    return StiefelSAGE(arm,config).to(device)


def preflight(output):
    # Reuse the unchanged forward/PCA contract checks, then test new optimization.
    latent.preflight(output)
    generator=torch.Generator().manual_seed(801)
    xx=torch.randn(11,3,generator=generator,dtype=torch.float64)
    x=torch.linalg.qr(xx,mode='reduced').Q
    m=torch.randn(11,3,generator=generator,dtype=torch.float64)
    b=skewon_direction(x,m);n=.5*(m@x.T-x@m.T)
    pp=polar_ranked(n);direct=-.5*(pp-pp.T)@x
    assert torch.allclose(b,direct,atol=1e-9,rtol=1e-9)
    assert float((x.T@b+b.T@x).abs().max())<1e-9
    assert float(torch.linalg.matrix_norm(b,ord=2))<=1+1e-9
    ordinary=-tangent(x,m);ordinary/=torch.linalg.matrix_norm(ordinary,ord=2)
    assert float((m*b).sum())<=float((m*ordinary).sum())+1e-9
    assert torch.allclose((m*b).sum(),-torch.linalg.svdvals(n).sum(),atol=1e-9,rtol=1e-9)
    _,cfg,d,p=r.load_fold('hunan',1);nt=int(d['n_train']);features,te,ee,y=r.tensors(d,p,'cuda');checks={}
    for arm in ('GR','GL'):
        model=construct(arm,cfg,'cuda');model.fit_reference(features[:nt],te)
        before=r.cpu_state(model);optimizer=model.make_optimizer(cfg)
        for step in range(6):
            optimizer.zero_grad()
            loss=model.metric_loss(features[:nt],y,te)
            if step>=3:loss=.2*loss+F.mse_loss(model(features[:nt],nt,te)[0],y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():full=model(features,nt,ee);alone=model(features[:nt],nt,te)
        assert torch.allclose(full[0][:nt],alone[0],atol=3e-6,rtol=3e-6)
        assert optimizer.maximum_error<2e-6 and optimizer.maximum_step<=.01000001
        change=float((model.projection.weight.detach().cpu()-before['projection.weight']).abs().max())
        assert change>0
        path=output/(arm+'_stiefel.pt');torch.save(r.cpu_state(model),path)
        restored=construct(arm,cfg,'cuda').eval();restored.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad():assert torch.equal(restored(features,nt,ee)[0],full[0])
        checks[arm]={**optimizer.diagnostics(),'projection_change':change,'query_isolation':True,'checkpoint_exact':True}
    report=json.loads((output/'preflight.json').read_text());report['stiefel_checks']=checks
    report['optimizer_algebra_checks']=True;r.write_json(output/'preflight.json',report)
    print(json.dumps(checks),flush=True)


def summarize_controls(output):
    r.summarize(output)
    summary=json.loads((output/'summary.json').read_text());gains={a:[] for a in ('GB','GA','GR')}
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
    summary.pop('fits');summary.update(fit_records=90,new_fits=30,cached_fits=60)
    r.write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


def run(output,preflight_dir):
    checked=json.loads((preflight_dir/'preflight.json').read_text());assert checked['optimizer_algebra_checks']
    assert len(checked['stiefel_checks'])==2
    old=json.loads((PREVIOUS/'manifest.json').read_text())
    for path,value in old['source_hashes'].items():
        if '/data/' in path.replace('\\','/') or path.endswith('config.json'):assert r.digest(r.ROOT/path)==value,path
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),Path(r.__file__),Path(latent.__file__),r.BASE/'baseline.py']
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources};cached={}
    for region in r.REGIONS:
        for fold in range(1,6):
            source=r.SOURCE/region/f'fold-{fold:02d}'
            for name in ('config.json','data/development.npz','data/preprocessor.npz','data/split.csv'):
                p=source/name;hashes[str(p.relative_to(r.ROOT))]=r.digest(p)
            for alias,original in CACHED.items():
                src=PREVIOUS/region/f'fold-{fold:02d}'/original
                for p in src.iterdir():
                    if p.is_file():cached[str(p.relative_to(r.ROOT))]=r.digest(p)
                shutil.copytree(src,output/region/f'fold-{fold:02d}'/alias)
    for p in sources:(output/p.name).write_bytes(p.read_bytes())
    r.write_json(output/'manifest.json',{'task_id':TASK,'branch':'RELATION/v0.6','device':'cuda',
                'started_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'source_hashes':hashes,
                'cached_hashes':cached,'cache_source':str(PREVIOUS.relative_to(r.ROOT)),'cache_aliases':CACHED,
                'new_fits':30,'cached_fits':60,'final_holdout_accessed':False})
    start=time.perf_counter()
    for fold in range(1,6):
        for region in r.REGIONS:
            for arm in ('GR','GL'):
                r.fit(arm,region,fold,output,'cuda',model_factory=construct,task_id=TASK,warmup_steps=100,auxiliary_weight=.2)
    for path,value in {**hashes,**cached}.items():assert r.digest(r.ROOT/path)==value,path
    r.write_json(output/'execution.json',{'finished_at':datetime.now().astimezone().isoformat(),
                'wall_seconds':time.perf_counter()-start,'new_fits':30,'cached_fits':60,'all_inputs_unchanged':True})
    summarize_controls(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)


if __name__=='__main__':main()
