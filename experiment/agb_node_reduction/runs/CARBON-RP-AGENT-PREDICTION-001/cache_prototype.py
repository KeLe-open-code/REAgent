"""CPU-only preparation: cache frozen geometry primitives, never predictions."""
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rp_prediction_agent as p
from rp_agent_reference import ReferencePredictor


class CachedReference(ReferencePredictor):
    def __init__(self, bundle, device='cpu'):
        super().__init__(bundle['state_dict'], bundle['config'], bundle['reference_x'], bundle['source_ids'], bundle['preprocessor'], device)
        full_x = torch.cat((self.x, torch.as_tensor(bundle['feedback_x'], device=device)))
        self.canonical_u = self.model.coordinates(full_x).detach()
        assert torch.equal(self.canonical_u[:self.n], self.fields['field_support'])
        all_ids = list(self.ids)+list(bundle['feedback_ids'])
        self.positions = {str(sid): i for i, sid in enumerate(all_ids)}
        self.tables = []
        for theta in self.model.layer_log_metric:
            root = (.5*theta).exp()
            shifted = (self.canonical_u-self.fields['field_support'].mean(0))*root
            self.tables.append(dict(theta=theta.detach().clone(), root=root,
                distance=torch.cdist(shifted, shifted[:self.n], compute_mode='donot_use_mm_for_euclid_dist').square(),
                scores=shifted@self.fields['field_directions'].T,
                norms=shifted.square().sum(1)))
        def corrected(model, u, theta):
            return self.cached_distance(u, theta)
        self.model.corrected_distance = MethodType(corrected, self.model)

    def packed(self, keep, query_x, query_ids):
        result = super().packed(keep, query_x, query_ids)
        retained = set(self.ids[np.asarray(keep)])
        extra = [self.positions[str(sid)] for sid in query_ids if str(sid) not in retained]
        self.source_positions = result[0]
        self.row_positions = torch.as_tensor(list(np.flatnonzero(keep))+extra, device=self.device)
        return result

    def cached_distance(self, u, theta):
        table = next(row for row in self.tables if torch.equal(row['theta'], theta))
        s, q = self.source_positions, self.row_positions
        reference = table['distance'][s[:, None], s]
        tau = reference.masked_fill(torch.eye(len(s), device=self.device, dtype=torch.bool), torch.inf).topk(8, largest=False).values.mean().clamp_min(1e-8)
        base = table['distance'][q[:, None], s]
        w = torch.softmax(-base/tau+self.model.field_log_norm, 1)
        wt = torch.softmax(-reference/tau+self.model.field_log_norm, 1)
        a, b = table['scores'][q[:, None], s], table['scores'][s[:, None], s]
        positive_left = (w*a.square()).sum(1)[:, None]+w@b.square().T
        positive_right = a.square()@wt.T+(wt*b.square()).sum(1)[None, :]
        distance = (185*(positive_left-2*(w*a)@b.T+positive_right-2*a@(wt*b).T)*.5+.01*base)/1.01
        scale = (185*(positive_left+positive_right)*.5+.01*(table['norms'][q, None]+table['norms'][s][None, :]))/1.01
        terms, eps = len(s)+185, torch.finfo(u.dtype).eps
        pairs = torch.nonzero(distance.detach() <= 8*(terms*eps/(1-terms*eps))*scale.detach()/1e-8)
        if len(pairs):
            direct = []
            for block in pairs.split(256):
                i, j = block.unbind(1)
                delta = (u[i]-self.model.field_support[j])*table['root']
                projected = delta@self.model.field_directions.T
                direct.append((185*(projected.square()*(w[i]+wt[j])*.5).sum(1)+.01*delta.square().sum(1))/1.01)
            distance = distance.index_put(tuple(pairs.unbind(1)), torch.cat(direct))
        diagonal = (185*(w@self.model.field_directions.square())+.01)/1.01*theta.exp()
        return distance.clamp_min(0), diagonal


def main():
    p.r.setup()
    root = p.r.HERE/'runs'/p.TASK
    output = root/'cache-cpu-diagnostic-01'
    output.mkdir(exist_ok=False)
    rows = []
    for region in p.r.REGIONS:
        bundle = torch.load(root/'preflight-02'/region/'warmup/warmup.pt', map_location='cpu', weights_only=False)
        native = ReferencePredictor(bundle['state_dict'], bundle['config'], bundle['reference_x'], bundle['source_ids'], bundle['preprocessor'], 'cpu')
        cached = CachedReference(bundle)
        rng = np.random.default_rng(20261020)
        for fraction in (1., .9, .75, .5):
            keep = np.zeros(native.n, dtype=bool)
            keep[rng.choice(native.n, max(9, round(native.n*fraction)), replace=False)] = True
            start = time.perf_counter()
            expected = native.predict(keep, bundle['feedback_x'], bundle['feedback_ids'], graphs=True)
            native_seconds = time.perf_counter()-start
            start = time.perf_counter()
            actual = cached.predict(keep, bundle['feedback_x'], bundle['feedback_ids'], graphs=True)
            cached_seconds = time.perf_counter()-start
            error = float(np.max(np.abs(expected['scaled']-actual['scaled'])))
            same_edges = all(np.array_equal(a['neighbors'], b['neighbors']) for a, b in zip(expected['graphs'], actual['graphs']))
            assert error <= 3e-6 and same_edges
            rows.append(dict(region=region, fraction=fraction, scaled_prediction_error=error, same_neighbors=same_edges,
                native_cpu_seconds=native_seconds, cached_cpu_seconds=cached_seconds))
        assert native.unchanged_parameters() and cached.unchanged_parameters()
    report = dict(valid=True, CPU_only=True, formal_core_unchanged=True, cached_predictions=False, cases=rows)
    p.r.write_json(output/'summary.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
