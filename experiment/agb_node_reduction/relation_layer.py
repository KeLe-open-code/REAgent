"""Joint layer-specific diagonal congruences of a frozen local SPD field."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import relation_graphsage as r
import relation_agop as a
import relation_local_egop as local
import relation_unroll as unroll

TASK='CARBON-RELATION-LAYER-001'
PREVIOUS=r.HERE/'runs/CARBON-RELATION-CONDITIONAL-001/attempt-02'
ARMS=('G0','GB','GS','GU','GE','GJ','GL')
NEW=('GE','GJ','GL')


class RDoG:
    """RDoG in flat log coordinates, one independent schedule per diagonal block."""
    def __init__(self,parameter):
        self.parameter=parameter
        self.radius=parameter.new_full((len(parameter),),.01)
        self.accumulator=parameter.new_zeros(len(parameter))
        self.steps=0

    @torch.no_grad()
    def step(self):
        gradient=self.parameter.grad
        if gradient is None:return
        assert torch.isfinite(gradient).all()
        self.radius=torch.maximum(self.radius,self.parameter.norm(dim=1))
        self.accumulator+=gradient.square().sum(1)
        factor=torch.where(self.accumulator>0,self.radius/self.accumulator.sqrt(),0.)
        self.parameter.add_(-factor[:,None]*gradient)
        assert torch.isfinite(self.parameter).all()
        self.steps+=1


class JointOptimizer:
    def __init__(self,model,config):
        self.adam=torch.optim.Adam(model.backbone.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        self.geometry=RDoG(model.layer_log_metric)

    def zero_grad(self,set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.geometry.parameter.grad=None

    def step(self):
        self.adam.step();self.geometry.step()

    def diagnostics(self):
        return {'geometry_optimizer':'RDoG_flat_log_coordinates','initial_radius':.01,
                'steps':self.geometry.steps,'last_radius':self.geometry.radius.tolist(),
                'last_squared_gradient_sum':self.geometry.accumulator.tolist(),
                'schedule_state_is_end_of_training_not_best_checkpoint':True}


class LayerSAGE(local.LocalEgopSAGE):
    corrected_distance=unroll.UnrolledSAGE.corrected_distance
    graph=unroll.UnrolledSAGE.graph

    def __init__(self,arm,config):
        super().__init__(arm,config)
        self.layer_log_metric=torch.nn.Parameter(torch.zeros((1 if arm=='GJ' else 2,185),dtype=torch.float64))

    def layer_graphs(self,x,n_train):
        assert n_train==len(self.field_support) and bool(self.field_ready)
        u=self.coordinates(x)
        first=self.graph(u,self.layer_log_metric[0],torch.float32)
        second=first if self.arm=='GJ' else self.graph(u,self.layer_log_metric[1],torch.float32)
        return first,second

    def relations(self,x,n_train,edges):
        if not bool(self.field_ready):return super().relations(x,n_train,edges)
        assert n_train==len(self.field_support)
        return self.graph(self.coordinates(x),self.layer_log_metric[0],torch.float32)

    def fit_relations(self,x,target,edges):
        history=super().fit_relations(x,target,edges)
        history[0]['localized']=self.arm!='GE'
        return history

    def forward(self,x,n_train,edges):
        if not bool(self.field_ready):return super().forward(x,n_train,edges)
        first,second=self.layer_graphs(x,n_train)
        features=((self.filtered(x)-self.input_mean)/self.input_scale).float()
        h=self.weighted_layer(self.backbone.first,features,*first[:2])
        h=self.backbone.dropout(torch.relu(h))
        embedding=torch.relu(self.weighted_layer(self.backbone.second,h,*second[:2]))
        return self.backbone.head(embedding).squeeze(-1),embedding,*first

    def make_optimizer(self,config):return JointOptimizer(self,config)


def construct(arm,config,device):
    r.fix_seed(config['seed'],1)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as data:
            config['relation_n_train']=int(data['n_train'])
    return LayerSAGE(arm,config).to(device)


def preflight(output,model_factory=construct):
    output.mkdir(parents=True,exist_ok=False);cases=[];learning={};started=time.perf_counter()
    parameter=torch.nn.Parameter(torch.zeros((2,3),dtype=torch.float64))
    optimizer=RDoG(parameter);parameter.grad=torch.tensor([[1.,2.,3.],[0.,0.,0.]],dtype=torch.float64)
    optimizer.step();assert abs(float(parameter[0].detach().norm())-.01)<1e-14
    assert torch.equal(parameter[1],torch.zeros(3,dtype=torch.float64))
    before=parameter.detach().clone();parameter.grad.zero_();optimizer.step();assert torch.equal(parameter,before)
    for region in r.REGIONS:
        _,cfg,data,pre=r.load_fold(region,1);n=int(data['n_train']);x,te,ee,y=r.tensors(data,pre,'cuda')
        common=None
        for arm in ('GB','GE','GL'):
            model=model_factory(arm,cfg,'cuda').eval();model.fit_reference(x[:n],te)
            rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            with torch.no_grad():prediction=model(x,n,ee)[0]
            if common is None:common=(rng,prediction)
            else:assert torch.equal(prediction,common[1]) and all(torch.equal(p,q) for p,q in zip(rng,common[0]))
            cases.append({'region':region,'arm':arm,'paired_initial_prediction_and_rng':True})
        learning[region]={};fresh={}
        for arm in NEW:
            model=model_factory(arm,cfg,'cuda').eval();model.fit_reference(x[:n],te)
            original=r.cpu_state(model);model.fit_relations(x[:n],y,te)
            assert all(torch.equal(v,model.state_dict()[key].cpu()) for key,v in original.items() if key.startswith('backbone'))
            parent=local.construct('GE' if arm=='GE' else 'GL',cfg,'cuda').eval()
            parent.fit_reference(x[:n],te);parent.fit_relations(x[:n],y,te)
            with torch.no_grad():
                assert torch.equal(model(x,n,ee)[0],parent(x,n,ee)[0])
                initial_graphs=model.layer_graphs(x,n)
            fresh[arm]=r.cpu_state(model)
            frozen={key:value.clone() for key,value in model.named_buffers()}
            optimizer=model.make_optimizer(cfg);gradient_norms=[]
            for _ in range(3):
                model.train();optimizer.zero_grad();loss=F.mse_loss(model(x[:n],n,te)[0],y);loss.backward()
                norms=model.layer_log_metric.grad.norm(dim=1);assert torch.isfinite(norms).all() and (norms>0).all()
                gradient_norms.append(norms.tolist());torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);optimizer.step()
            assert all(torch.equal(value,dict(model.named_buffers())[key]) for key,value in frozen.items())
            assert (model.layer_log_metric.detach().norm(dim=1)>0).all()
            model.eval()
            with torch.no_grad():
                full=model(x,n,ee);alone=model(x[:n],n,te)
                single_x=torch.cat([x[:n],x[n:n+1]]);single=model(single_x,n,ee)
                graphs=model.layer_graphs(x,n);alone_graphs=model.layer_graphs(x[:n],n);single_graphs=model.layer_graphs(single_x,n)
                assert torch.allclose(full[0][:n],alone[0],atol=3e-6,rtol=3e-6)
                assert torch.allclose(full[0][n],single[0][n],atol=3e-6,rtol=3e-6)
                for layer,(near,weight,diagonal) in enumerate(graphs):
                    assert (near<n).all() and (near[:n]!=torch.arange(n,device=x.device)[:,None]).all()
                    assert torch.equal(near[:n],alone_graphs[layer][0]) and torch.equal(near[n],single_graphs[layer][0][n])
                    assert torch.allclose(weight.sum(1),torch.ones(len(x),device=x.device),atol=1e-6)
                    assert torch.isfinite(diagonal).all() and (diagonal>0).all()
                    assert not torch.equal(weight,initial_graphs[layer][1])
                    u=model.coordinates(x);g=model.field_directions
                    log_metric=model.layer_log_metric[0 if arm=='GJ' else layer]
                    w=model.endpoint_weights(u[[n,0]],log_metric)[0] if hasattr(model,'endpoint_weights') else model.field_weights(u[[n,0]])
                    root=torch.diag((.5*model.layer_log_metric[0 if arm=='GJ' else layer]).exp())
                    matrices=[root@((185*g.T@(wi[:,None]*g)+.01*torch.eye(185,device=x.device))/1.01)@root for wi in w]
                    assert all(torch.linalg.eigvalsh(a.symmetric(m)).min()>0 for m in matrices)
                    delta=u[n]-u[0];exact=delta@((matrices[0]+matrices[1])*.5)@delta
                    distance,_=model.corrected_distance(u,model.layer_log_metric[0 if arm=='GJ' else layer])
                    assert torch.allclose(exact,distance[n,0],atol=1e-8,rtol=1e-8)
            path=output/f'{region}_{arm}.pt';torch.save({'state_dict':r.cpu_state(model),'config':cfg},path)
            saved=torch.load(path,weights_only=False);restored=model_factory(arm,saved['config'],'cuda').eval();restored.load_state_dict(saved['state_dict'])
            with torch.no_grad():assert torch.equal(full[0],restored(x,n,ee)[0])
            count=sum(p.numel() for p in model.parameters());assert count==(28090 if arm=='GJ' else 28275)
            learning[region][arm]={'zero_correction_parent_exact':True,'gradient_norms':gradient_norms,
                                  'geometry_norms':model.layer_log_metric.detach().norm(dim=1).tolist(),
                                  'parameters':count,'inductive_both_layers':True,'checkpoint_exact':True}
        gradients={}
        for arm in ('GJ','GL'):
            model=model_factory(arm,cfg,'cuda').eval();model.load_state_dict(fresh[arm])
            loss=F.mse_loss(model(x[:n],n,te)[0],y)
            gradients[arm]=torch.autograd.grad(loss,model.layer_log_metric)[0]
        discrepancy=float((gradients['GJ'][0]-gradients['GL'].sum(0)).abs().max())
        assert torch.allclose(gradients['GJ'][0],gradients['GL'].sum(0),atol=1e-8,rtol=1e-5)
        learning[region]['shared_gradient_sum_error']=discrepancy
        learning[region]['initial_layer_gradient_cosine']=float(F.cosine_similarity(gradients['GL'][0],gradients['GL'][1],dim=0))
    r.write_json(output/'preflight.json',{'cases':cases,'learning':learning,'selected_device':'cuda',
                 'RDoG_first_step_and_zero_gradient':True,'wall_seconds':time.perf_counter()-started})
    print(json.dumps(learning),flush=True)


def summarize(output,arms=ARMS,new_fits=45,cached_fits=60):
    rows=[];points=[];long_rows=[];pairs=[];grouped={};groups=[]
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,_=r.load_fold(region,fold);n=int(data['n_train']);predictions={};measured={}
            for arm in arms:
                directory=output/region/f'fold-{fold:02d}'/arm
                m=json.loads((directory/'metrics.json').read_text())
                with np.load(directory/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'],data['sample_ids']);predictions[arm]=z['prediction'].copy()
                    measured[arm]=r.metrics(data['y'][n:],z['prediction'][n:])
                    assert all(abs(measured[arm][k]-m['validation'][k])<1e-10 for k in measured[arm])
                    for i in range(n,len(data['y'])):
                        near=z['neighbors'][i];weights=z['weights'][i];observed=float(data['y'][i])
                        long_rows.append({'region':region,'fold':fold,'sample_id':str(data['sample_ids'][i]),'arm':arm,
                            'observed_agb':observed,'prediction':float(z['prediction'][i]),'absolute_error':abs(z['prediction'][i]-observed),
                            'missing_features':int(data['missing_mask'][i].sum()),'high_agb_train_q80':bool(observed>=np.quantile(data['y'][:n],.8)),
                            'first_layer_neighbor_agb_mean':float(weights@data['y'][near]),
                            'graph_scope':'first_layer' if (directory/'layer_graphs.npz').is_file() else 'shared_both_layers'})
                rows.append({'region':region,'fold':fold,'arm':arm,**measured[arm]})
                rows[-1].update(best_epoch=m['best_epoch'],epochs_run=m['epochs_run'],train_rmse=m['train']['rmse'],
                                elapsed_seconds=m['elapsed_seconds'],parameters=m['parameters'])
            pairs.append({'region':region,'fold':fold,**{f'GL_vs_{arm}_gain_percent':100*(1-measured['GL']['rmse']/measured[arm]['rmse']) for arm in arms if arm!='GL'}})
            for i in range(n,len(data['y'])):
                observed=float(data['y'][i]);point={'region':region,'fold':fold,'sample_id':str(data['sample_ids'][i]),
                    'observed_agb':observed,'missing_features':int(data['missing_mask'][i].sum()),'high_agb_train_q80':bool(observed>=np.quantile(data['y'][:n],.8))}
                for arm in arms:
                    point[arm+'_prediction']=float(predictions[arm][i]);point[arm+'_absolute_error']=abs(predictions[arm][i]-observed)
                for arm in arms:
                    if arm!='GL':point[f'GL_absolute_error_gain_vs_{arm}']=point[arm+'_absolute_error']-point['GL_absolute_error']
                points.append(point)
        grouped[region]={arm:{key:float(np.mean([p[key] for p in rows if p['region']==region and p['arm']==arm]))
                       for key in ('rmse','mae','r2','train_rmse','best_epoch','elapsed_seconds')} for arm in arms}
        for name,condition in [('all',lambda p:True),('missing',lambda p:p['missing_features']>0),('complete',lambda p:p['missing_features']==0),
                               ('high_agb',lambda p:p['high_agb_train_q80']),('ordinary_agb',lambda p:not p['high_agb_train_q80'])]:
            selected=[p for p in points if p['region']==region and condition(p)]
            if not selected:continue
            for arm in arms:
                groups.append({'region':region,'group':name,'arm':arm,'n':len(selected),
                               **r.metrics(np.array([p['observed_agb'] for p in selected]),np.array([p[arm+'_prediction'] for p in selected]))})
    assert len(rows)==15*len(arms) and len(long_rows)==1421*len(arms) and len(points)==1421
    assert len({(p['region'],p['sample_id']) for p in points})==1421
    gains={f'GL_vs_{arm}_gain_percent':float(np.mean([p[f'GL_vs_{arm}_gain_percent'] for p in pairs])) for arm in arms if arm!='GL'}
    province_gain={region:float(np.mean([p['GL_vs_G0_gain_percent'] for p in pairs if p['region']==region])) for region in r.REGIONS}
    province_r2={region:grouped[region]['GL']['r2']-grouped[region]['G0']['r2'] for region in r.REGIONS}
    controls_positive=all(gains[f'GL_vs_{arm}_gain_percent']>0 for arm in arms if arm not in ('G0','GE','GL'))
    full=gains['GL_vs_G0_gain_percent']>=3 and gains['GL_vs_GE_gain_percent']>=.5 and controls_positive and min(province_gain.values())>=-.0001 and min(province_r2.values())>=-1e-6
    pareto=gains['GL_vs_G0_gain_percent']>=1 and gains['GL_vs_GE_gain_percent']>0 and controls_positive and min(province_gain.values())>=-1 and min(province_r2.values())>=-.01
    summary={'province_means':grouped,**gains,'province_GL_vs_G0_gain_percent':province_gain,'province_GL_vs_G0_r2_change':province_r2,
             'classification':'FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
             'folds':15,'fit_records':len(rows),'new_fits':new_fits,'cached_fits':cached_fits,'development_ids':1421,'method_id_records':len(long_rows),'final_test_used':False,
             'both_layer_graphs':'Each new model has layer_graphs.npz; responses.neighbors/weights/edges are first layer only',
             **{f'GL_better_IDs_vs_{arm}':int(sum(p[f'GL_absolute_error_gain_vs_{arm}']>0 for p in points)) for arm in arms if arm!='GL'}}
    for name,records in [('all_fold_metrics',rows),('all_1421_development_ids',points),('all_ids_graph_diagnostics',long_rows),
                         ('paired_fold_gains',pairs),('subgroup_metrics',groups)]:r.write_csv(output/(name+'.csv'),records)
    r.write_json(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='province_means'}),flush=True)


@torch.no_grad()
def export_graphs(output,model_factory=construct):
    started=time.perf_counter();rows=[];points=[];files={}
    for region in r.REGIONS:
        for fold in range(1,6):
            _,_,data,pre=r.load_fold(region,fold);n=int(data['n_train']);x,te,ee,_=r.tensors(data,pre,'cuda')
            for arm in NEW:
                directory=output/region/f'fold-{fold:02d}'/arm
                saved=torch.load(directory/'best.pt',map_location='cuda',weights_only=False)
                model=model_factory(arm,saved['config'],'cuda').eval();model.load_state_dict(saved['state_dict'])
                full=model(x,n,ee);graphs=model.layer_graphs(x,n);u=model.coordinates(x)
                original=model.graph(u,torch.zeros(185,dtype=torch.float64,device=x.device),torch.float32)
                original_near=original[0].cpu().numpy()
                with np.load(directory/'responses.npz') as z:
                    scaled=(z['prediction']-pre['y_mean'])/pre['y_scale']
                    assert np.allclose(full[0].cpu().numpy(),scaled,atol=3e-6,rtol=3e-6)
                    assert np.array_equal(full[2].cpu().numpy(),z['neighbors']) and np.array_equal(full[3].cpu().numpy(),z['weights'])
                parent_arm='GE' if arm=='GE' else 'GL'
                old_path=r.HERE/'runs/CARBON-RELATION-LOCAL-EGOP-001/attempt-01'/region/f'fold-{fold:02d}'/parent_arm/'best.pt'
                parent=torch.load(old_path,map_location='cuda',weights_only=False)['state_dict']
                assert all(torch.equal(value,parent[name]) for name,value in model.named_buffers())
                arrays={'sample_ids':data['sample_ids'],'n_train':n,'layer_log_metric':model.layer_log_metric.detach().cpu().numpy()}
                near=[];replacement=[]
                for layer,(neighbors,weights,diagonal) in enumerate(graphs,1):
                    nn=neighbors.cpu().numpy();ww=weights.cpu().numpy();assert (nn<n).all() and np.allclose(ww.sum(1),1,atol=1e-6)
                    edges=np.stack([nn.ravel(),np.repeat(np.arange(len(x)),8)])
                    arrays.update({f'layer{layer}_neighbors':nn,f'layer{layer}_weights':ww,f'layer{layer}_local_metric_diagonal':diagonal.cpu().numpy(),
                                   f'layer{layer}_train_edges':edges[:,:n*8],f'layer{layer}_eval_edges':edges})
                    near.append(nn);replacement.append(np.array([1-len(set(p)&set(q))/8 for p,q in zip(nn,original_near)]))
                union=np.array([len(set(p)|set(q)) for p,q in zip(*near)])
                if arm=='GJ':assert np.array_equal(near[0],near[1]) and np.array_equal(arrays['layer1_weights'],arrays['layer2_weights'])
                path=directory/'layer_graphs.npz';np.savez_compressed(path,**arrays);files[str(path.relative_to(output))]=r.digest(path)
                s=model.layer_log_metric.detach()
                rows.append({'region':region,'fold':fold,'arm':arm,'blocks':len(s),'norm_layer1':float(s[0].norm()),'norm_layer2':float(s[-1].norm()),
                             'log_scale_min':float(s.min()),'log_scale_max':float(s.max()),'layer_parameter_difference':float((s[0]-s[-1]).norm()),
                             'validation_layer1_replacement':float(replacement[0][n:].mean()),'validation_layer2_replacement':float(replacement[1][n:].mean()),
                             'validation_unique_union_neighbors':float(union[n:].mean()),'source_field_messages_exact':True})
                for i in range(n,len(x)):
                    points.append({'region':region,'fold':fold,'arm':arm,'sample_id':str(data['sample_ids'][i]),
                                   'layer1_replacement':float(replacement[0][i]),'layer2_replacement':float(replacement[1][i]),
                                   'unique_union_neighbors':int(union[i])})
    assert len(rows)==45 and len(points)==4263
    r.write_csv(output/'joint_geometry_diagnostics.csv',rows);r.write_csv(output/'all_ids_layer_geometry.csv',points)
    r.write_json(output/'graph_export.json',{'models':45,'method_id_rows':4263,'wall_seconds':time.perf_counter()-started,
                 'both_layers_train_only':True,'responses_graph_scope':'first_layer','all_source_fields_messages_exact':True,'graph_hashes':files})


def run(output,preflight_dir):
    checks=json.loads((preflight_dir/'preflight.json').read_text());assert checks['RDoG_first_step_and_zero_gradient']
    sources=[Path(local.__file__),Path(unroll.__file__)]
    hashes={str(p.relative_to(r.ROOT)):r.digest(p) for p in sources}
    a.run(output,preflight_dir,task_id=TASK,branch='RELATION/v0.30',model_factory=construct,
          previous=PREVIOUS,cache_aliases={'G0':'G0','GB':'GB','GS':'GS','GU':'GU'},new_arms=NEW,
          relation_metadata={'metric_rounds':0,'BW_iterations':0,'field_kernel_solves':1,'frozen_preprocessing_kernel_solves':25,
                             'message_source':'v0.20 ordinary GE','maximum_joint_updates':400,'teacher_steps':0,
                             'geometry_optimizer':'RDoG per layer in flat log coordinates','RDoG_epsilon':.01,
                             'geometry_parameters_GE_GL':370,'geometry_parameters_GJ':185,'graph_scope_of_responses':'first_layer_only',
                             'additional_source_hashes':hashes})
    for path,value in hashes.items():
        p=r.ROOT/path;assert r.digest(p)==value;(output/p.name).write_bytes(p.read_bytes())
    export_graphs(output);summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight-dir',type=Path)
    args=parser.parse_args();r.setup()
    if args.mode=='preflight':preflight(args.output)
    else:run(args.output,args.preflight_dir)
