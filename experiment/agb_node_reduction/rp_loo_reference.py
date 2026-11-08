"""Exact per-target leave-one-reference-out feedback, batched over targets."""
import numpy as np
import torch

from rp_message_agent import MessageReference


class LeaveOneOutReference(MessageReference):
    def __init__(self,bundle,device='cuda',batch_size=128):
        super().__init__(bundle,device)
        self.batch_size=batch_size
        self.leave_one_out_cases=0

    def weights(self,table,s,excluded):
        base=table['distance'][s[:,None],s]
        diagonal=torch.eye(len(s),device=self.device,dtype=torch.bool)
        values,near=base.masked_fill(diagonal,torch.inf).topk(9,largest=False)
        candidates=values.unsqueeze(0).expand(len(excluded),-1,-1).clone()
        candidates.masked_fill_(near.unsqueeze(0)==excluded[:,None,None],torch.inf)
        totals=candidates.topk(8,largest=False).values.sum(-1)
        totals.scatter_(1,excluded[:,None],0.)
        tau=(totals.sum(1)/((len(s)-1)*8)).clamp_min(1e-8)
        logits=-base.unsqueeze(0)/tau[:,None,None]+self.fields['field_log_norm'][s][None,None,:]
        logits.scatter_(2,excluded[:,None,None].expand(-1,len(s),1),-torch.inf)
        return torch.softmax(logits,dim=-1),base

    def row_distances(self,table,s,excluded,rows,weights,base):
        b=table['scores'][s[:,None],s]
        batch=torch.arange(len(excluded),device=self.device)[:,None]
        a=b[rows];w=weights[batch,rows]
        positive_left=(w*a.square()).sum(-1).unsqueeze(-1)+w@b.square().T
        positive_right=a.square()@weights.transpose(1,2)+(weights*b.square()).sum(-1).unsqueeze(1)
        distance=(185*(positive_left+positive_right-2*((w*a)@b.T+a@(weights*b).transpose(1,2)))*.5+
            .01*base[rows])/1.01
        scale=(185*(positive_left+positive_right)*.5+
            .01*(table['norms'][s[rows]][...,None]+table['norms'][s][None,None,:]))/1.01
        columns=torch.arange(len(s),device=self.device)
        forbidden=(columns[None,None,:]==excluded[:,None,None]) | (columns[None,None,:]==rows[:,:,None])
        terms=len(s)-1+185;eps=torch.finfo(distance.dtype).eps
        pairs=torch.nonzero((distance<=8*(terms*eps/(1-terms*eps))*scale/1e-8)&~forbidden)
        for block in pairs.split(256):
            bi,ri,ci=block.unbind(1)
            delta=(self.canonical_u[s[rows[bi,ri]]]-self.canonical_u[s[ci]])*table['root']
            projected=delta@self.fields['field_directions'][s].T
            direct=(185*(projected.square()*(w[bi,ri]+weights[bi,ci])*.5).sum(1)+.01*delta.square().sum(1))/1.01
            distance[bi,ri,ci]=direct
        return distance.clamp_min(0).masked_fill(forbidden,torch.inf)

    def neighbors(self,distance):
        near=torch.argsort(distance,dim=-1,stable=True)[...,:8]
        weight=torch.softmax(-distance.gather(-1,near)/self.model.temperature,dim=-1).float()
        return near,weight

    @torch.inference_mode()
    def predict(self,keep,query_x,query_ids,graphs=False):
        query_ids=np.asarray(query_ids).astype(str)
        assert np.array_equal(query_ids,self.ids), 'LOO feedback requires the entire original training query set'
        assert np.array_equal(np.asarray(query_x),self.x.cpu().numpy())
        assert int(np.sum(keep))>=10
        result=super().predict(keep,query_x,query_ids,graphs=graphs)
        s=torch.as_tensor(np.flatnonzero(keep),device=self.device)
        features=self.features[s]
        model=self.model
        for start in range(0,len(s),self.batch_size):
            excluded=torch.arange(start,min(start+self.batch_size,len(s)),device=self.device)
            rows=excluded[:,None]
            weights2,base2=self.weights(self.tables[1],s,excluded)
            distance2=self.row_distances(self.tables[1],s,excluded,rows,weights2,base2)
            near2,weight2=self.neighbors(distance2);near2,weight2=near2[:,0],weight2[:,0]
            rows=torch.cat((rows,near2),dim=1)
            weights1,base1=self.weights(self.tables[0],s,excluded)
            distance1=self.row_distances(self.tables[0],s,excluded,rows,weights1,base1)
            near1,weight1=self.neighbors(distance1)
            aggregate=(features[near1]*weight1.unsqueeze(-1)).sum(-2)
            message=model.backbone.first.lin_l(aggregate)
            hidden=torch.relu(message+model.backbone.first.lin_r(features[rows]))
            embedding=torch.relu(model.backbone.second.lin_l((hidden[:,1:]*weight2.unsqueeze(-1)).sum(1))+
                model.backbone.second.lin_r(hidden[:,0]))
            scaled=model.backbone.head(embedding).squeeze(-1)
            positions=s[excluded].cpu().numpy()
            result['scaled'][positions]=scaled.cpu().numpy()
            result['embedding'][positions]=embedding.cpu().numpy()
            result['message'][positions]=message[:,0].cpu().numpy()
            if graphs:
                gather=result['packed_query_positions'][positions]
                result['graphs'][0]['neighbors'][gather]=s[near1[:,0]].cpu().numpy()
                result['graphs'][0]['weights'][gather]=weight1[:,0].cpu().numpy()
                result['graphs'][1]['neighbors'][gather]=s[near2].cpu().numpy()
                result['graphs'][1]['weights'][gather]=weight2.cpu().numpy()
            self.leave_one_out_cases+=len(excluded)
        result['prediction']=result['scaled'].astype(float)*self.pre['y_scale']+self.pre['y_mean']
        result['feedback_mode']='each training target predicted using S minus itself'
        return result
