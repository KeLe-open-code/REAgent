"""CPU integration check on generated data; no measured datasets or training run."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
NODE=ROOT/'experiment/agb_node_reduction'
sys.path.insert(0,str(NODE))
import numpy as np
import torch
from physical_baseline_v2 import prepare_model
from rp_agent_reference import ReferencePredictor
from rp_stop_agent import StopAgent
from evaluation.eco_calibration import EcologicalOOFCalibrator
from evaluation.metrics import compression_metrics

def main():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(20,185)).astype(np.float32)
    y=np.linspace(-2,2,20,dtype=np.float32)
    config={'region':'hunan','fold':1,'seed':20261014,'relation_n_train':20,
            'hidden_dims':[64,32],'dropout':.2,'knn_k':8,
            'learning_rate':.001,'weight_decay':1e-4}
    model,optimizer,xt,edges,target,_=prepare_model('RP',config,x,y,'cpu')
    assert sum(p.numel() for p in model.parameters())==28275
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    ids=np.asarray([f'synthetic-{i}' for i in range(20)])
    engine=ReferencePredictor(state,config,x,ids,{'y_mean':0.,'y_scale':1.},'cpu')
    query=rng.normal(size=(4,185)).astype(np.float32)
    query_ids=np.asarray([f'query-{i}' for i in range(4)])
    full=engine.predict(np.ones(20,dtype=bool),query,query_ids)['scaled']
    keep=np.arange(20)<12
    compressed=engine.predict(keep,query,query_ids)['scaled']
    assert full.shape==compressed.shape==(4,) and np.isfinite(compressed).all()
    assert len(engine.model.field_support)==12
    policy=StopAgent(17)
    features=torch.from_numpy(rng.normal(size=(4,144)).astype(np.float32))
    assert sum(p.numel() for p in policy.parameters())==11393
    with torch.no_grad():
        policy.net[-1].weight.fill_(.01)
        anchored=policy(policy.stop_input(features))
    assert torch.equal(anchored,torch.zeros_like(anchored))
    oof=np.linspace(10,100,40)
    ecology=np.asarray(['class-a']*20+['class-b']*20)
    calibration=EcologicalOOFCalibrator.fit(ecology,np.sin(oof/20),oof)
    corrected,offset=calibration.transform(['class-a','unseen'],[25.,75.])
    assert np.isfinite(corrected).all() and len(offset)==2
    metrics=compression_metrics(np.arange(4,dtype=float),full,compressed,20,12)
    assert abs(metrics['CR_percent']-40)<1e-10 and metrics['query_count']==4
    print(json.dumps({'status':'PASS','generated_inputs_only':True,'RP_parameters':28275,
                      'Agent_parameters':11393,'retained_references':12,'query_count':4,
                      'zero_STOP_anchor':True,'ECRC_unseen_class_fallback':True},indent=2))

if __name__=='__main__':main()
