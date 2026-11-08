"""Source-only epoch selection and native G0/RP physical reference interfaces."""
import argparse
import gc
import json
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import relation_precision as rp
import relation_agop as agop
import relation_local_egop as local
import rp_loo_agent as agent
from baseline import fit_preprocessor, transform, graph_edges

r = rp.r
TASK = 'CARBON-PHYSICAL-FACTORIAL-001'
ARMS = ('G0', 'RP')
GPU_ACCESS = nullcontext


def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def source_data(region, fold):
    source, config, cached, _ = r.load_fold(region, fold)
    n = int(cached['n_train'])
    data = {key: cached[key][:n].copy() for key in
            ('raw_features', 'y', 'coords', 'sample_ids', 'missing_mask')}
    config = {**config, 'max_epochs': 400, 'threads': 1, 'relation_n_train': n}
    assert data['raw_features'].shape == (n, 185)
    return source, config, data


def prepare_model(arm, config, x, y, device):
    config = {**config, 'relation_n_train': len(x)}
    model = (r.construct('G0', config, device) if arm == 'G0' else rp.construct('RP', config, device))
    edges, _ = graph_edges(x, x[:0], 8)
    edges = torch.as_tensor(edges, device=device)
    xt = torch.as_tensor(x, device=device)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    field_tick = time.perf_counter()
    if arm == 'RP':
        # The historical RP method loads a full-fold field cache. Refit its
        # original 25 AGOP + 1 local-field kernels using only this training set.
        agop.AgopSAGE.fit_reference(model, xt, edges)
        model.arm = 'GE'
        agop.AgopSAGE.fit_relations(model, xt, target, edges, rounds=5)
        model.arm = 'GL'
        local.LocalEgopSAGE.fit_relations(model, xt, target, edges)
        with torch.no_grad():
            model.label_edges.copy_(torch.quantile(target, torch.linspace(.1,.9,9,device=device)))
            bins = torch.bucketize(target.contiguous(), model.label_edges)
            assert (torch.bincount(bins, minlength=10)>0).all()
            model.label_bins.copy_(bins)
            model.label_values.copy_(torch.stack([target[bins==k].double().mean() for k in range(10)]))
        assert torch.equal(model.coordinates(xt), model.field_support)
    optimizer = (model.make_optimizer(config) if hasattr(model, 'make_optimizer') else
                 torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4))
    model.eval()
    setup_seconds = time.perf_counter() - field_tick
    expected = 27905 if arm == 'G0' else 28275
    assert sum(p.numel() for p in model.parameters()) == expected
    return model, optimizer, xt, edges, target, setup_seconds


def train_update(model, optimizer, x, edges, target):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (model.training_loss(x, target, edges) if hasattr(model, 'training_loss') else
            F.mse_loss(model(x, len(x), edges)[0], target))
    assert torch.isfinite(loss)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
    assert torch.isfinite(norm)
    optimizer.step()
    model.eval()
    return float(loss.detach()), float(norm)


def fit_predictor(region, fold, arm, output, device='cuda', small=False):
    output.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    source, config, data = source_data(region, fold)
    raw, y, n = data['raw_features'], data['y'], len(data['y'])
    order = np.random.default_rng(20261024 + 100*r.REGIONS.index(region) + fold).permutation(n)
    val = np.sort(order[:int(np.ceil(.2*n))])
    train = np.sort(order[len(val):])
    assert not set(data['sample_ids'][train]) & set(data['sample_ids'][val])
    pre = fit_preprocessor(raw[train], y[train])
    x, qx = transform(raw[train], pre), transform(raw[val], pre)
    model, optimizer, xt, edges, target, field_seconds = prepare_model(
        arm, config, x, (y[train]-pre['y_mean'])/pre['y_scale'], device)
    combined = torch.as_tensor(np.concatenate((x, qx)), device=device)
    _, ev = graph_edges(x, qx, 8)
    ev = torch.as_tensor(ev, device=device)
    history, best, stale, chosen = [], float('inf'), 0, 0
    for epoch in range(1, (3 if small else 400)+1):
        loss, norm = train_update(model, optimizer, xt, edges, target)
        with torch.inference_mode():
            pred = model(combined, len(train), ev)[0][len(train):].cpu().numpy().astype(float)
        rmse = float(np.sqrt(np.mean((pred*pre['y_scale']+pre['y_mean']-y[val])**2)))
        history.append(dict(epoch=epoch, loss=loss, internal_RMSE=rmse, gradient=norm))
        if rmse < best-1e-6:
            best, stale, chosen = rmse, 0, epoch
        else:
            stale += 1
        if stale >= 80:
            break
    assert chosen > 0
    r.write_csv(output/'inner_history.csv', history)
    del model, optimizer, xt, target, edges, combined, ev
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    pre = fit_preprocessor(raw, y)
    x = transform(raw, pre)
    model, optimizer, xt, edges, target, full_field_seconds = prepare_model(
        arm, config, x, (y-pre['y_mean'])/pre['y_scale'], device)
    initial = {key: value.detach().clone() for key, value in model.backbone.named_parameters()}
    full_history = []
    for epoch in range(1, chosen+1):
        loss, norm = train_update(model, optimizer, xt, edges, target)
        full_history.append(dict(epoch=epoch, loss=loss, gradient=norm))
    state = r.cpu_state(model)
    change = max(float((value-initial[key]).detach().abs().max()) for key, value in model.backbone.named_parameters())
    assert change > 0
    bundle = dict(schema=2, arm=arm, state_dict=state, config=config, preprocessor=pre,
        reference_x=x, source_ids=data['sample_ids'], source_indices=np.arange(n),
        feedback_indices=np.arange(n), feedback_x=x, feedback_ids=data['sample_ids'],
        feedback_y_scaled=(y-pre['y_mean'])/pre['y_scale'], source=data,
        epoch_selection_ids=data['sample_ids'][val], gradient_selection_ids=data['sample_ids'][train],
        selected_epoch=chosen, internal_epochs=len(history), full_epochs=len(full_history),
        outer_label_selection=False, full_source_refit=True, original_rp_historical_outer_early_stopping=False,
        label_dependent_fields_refitted=True, historical_field_cache_loaded=False)
    torch.save(bundle, output/'predictor.pt')
    r.write_csv(output/'full_history.csv', full_history)
    report = dict(region=region, fold=fold, arm=arm, selected_epoch=chosen,
        internal_epochs=len(history), full_epochs=len(full_history), parameters=sum(p.numel() for p in model.parameters()),
        source_count=n, body_parameter_change=change, seconds=time.perf_counter()-tick,
        inner_field_seconds=field_seconds, full_field_seconds=full_field_seconds,
        source_only_epoch_selection=True, full_source_refit=True, new_backbone_fits=2)
    r.write_json(output/'training.json', report)
    del model, optimizer, xt, target, edges
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(json.dumps(dict(event='predictor_fit', **report)), flush=True)
    return bundle


class PlainReference:
    """Only standard mean GraphSAGE weights and physical reference rows."""
    def __init__(self, bundle, device='cuda', loo=False):
        self.bundle, self.device, self.loo = bundle, device, loo
        self.config, self.pre = bundle['config'], bundle['preprocessor']
        self.model = r.construct('G0', self.config, device).eval()
        self.model.load_state_dict(bundle['state_dict'])
        self.model.requires_grad_(False)
        self.ids = np.asarray(bundle['source_ids']).astype(str)
        self.x = torch.as_tensor(bundle['reference_x'], device=device)
        self.features = self.x
        self.n, self.calls = len(self.ids), 0
        self.full_state = r.cpu_state(self.model)
        assert sum(p.numel() for p in self.model.parameters()) == 27905

    def packed(self, keep, qx, qids):
        s = np.flatnonzero(keep)
        assert len(s) >= 9
        locations = {sid:i for i, sid in enumerate(self.ids[s])}
        extras, gather = [], []
        for i, sid in enumerate(np.asarray(qids).astype(str)):
            if sid in locations:
                gather.append(locations[sid])
            else:
                gather.append(len(s)+len(extras)); extras.append(i)
        indices = torch.as_tensor(s, device=self.device)
        x = torch.cat((self.x[indices], torch.as_tensor(np.asarray(qx)[extras], device=self.device)))
        near = self.neighbors(x, len(s))
        return indices, x, near, torch.as_tensor(gather, device=self.device)

    @staticmethod
    def neighbors(x, n):
        a, b = x.double(), x[:n].double()
        distance = (a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*a@b.T).clamp_min(0)
        diagonal = torch.arange(n, device=x.device)
        distance[diagonal, diagonal] = torch.inf
        return torch.argsort(distance, dim=-1, stable=True)[:,:8]

    @torch.inference_mode()
    def native(self, keep, qx, qids):
        indices, x, _, gather = self.packed(keep, qx, qids)
        _, edges = graph_edges(x[:len(indices)].cpu().numpy(), x[len(indices):].cpu().numpy(), 8)
        edges = torch.as_tensor(edges, device=self.device)
        return self.model.backbone(x, edges)[0][gather].cpu().numpy()

    @torch.inference_mode()
    def predict(self, keep, qx, qids, graphs=False):
        indices, x, near, gather = self.packed(keep, qx, qids)
        body = self.model.backbone
        message = body.first.lin_l(x[near].mean(1))
        hidden = torch.relu(message+body.first.lin_r(x))
        embedding = torch.relu(body.second.lin_l(hidden[near].mean(1))+body.second.lin_r(hidden))
        scaled = body.head(embedding).squeeze(-1)[gather]
        result = dict(scaled=scaled.cpu().numpy(), embedding=embedding[gather].cpu().numpy(),
            message=message[gather].cpu().numpy(), packed_query_positions=gather.cpu().numpy(),
            reference_count=len(indices), query_count=len(gather))
        if graphs:
            mapped = indices[near].cpu().numpy()
            result['graphs'] = [dict(neighbors=mapped.copy(), weights=np.full(mapped.shape, .125, dtype=np.float32)) for _ in range(2)]
        if self.loo:
            assert np.array_equal(np.asarray(qids).astype(str), self.ids) and len(indices) >= 10
            s = indices
            source = self.x[s]
            a = source.double()
            distance = (a.square().sum(1)[:,None]+a.square().sum(1)[None,:]-2*a@a.T).clamp_min(0)
            for start in range(0, len(s), 128):
                excluded = torch.arange(start, min(start+128, len(s)), device=self.device)
                d2 = distance[excluded].clone()
                d2.scatter_(1, excluded[:,None], torch.inf)
                near2 = torch.argsort(d2, dim=-1, stable=True)[:,:8]
                rows = torch.cat((excluded[:,None], near2), 1)
                d1 = distance[rows].clone()
                columns = torch.arange(len(s), device=self.device)
                forbidden = ((columns[None,None,:] == excluded[:,None,None]) |
                             (columns[None,None,:] == rows[:,:,None]))
                near1 = torch.argsort(d1.masked_fill(forbidden, torch.inf), dim=-1, stable=True)[...,:8]
                first_message = body.first.lin_l(source[near1].mean(-2))
                h = torch.relu(first_message+body.first.lin_r(source[rows]))
                z = torch.relu(body.second.lin_l(h[:,1:].mean(1))+body.second.lin_r(h[:,0]))
                positions = s[excluded].cpu().numpy()
                result['scaled'][positions] = body.head(z).squeeze(-1).cpu().numpy()
                result['embedding'][positions] = z.cpu().numpy()
                result['message'][positions] = first_message[:,0].cpu().numpy()
                if graphs:
                    packed = result['packed_query_positions'][positions]
                    result['graphs'][0]['neighbors'][packed] = s[near1[:,0]].cpu().numpy()
                    result['graphs'][1]['neighbors'][packed] = s[near2].cpu().numpy()
        result['prediction'] = result['scaled'].astype(float)*self.pre['y_scale']+self.pre['y_mean']
        self.calls += 1
        return result

    def unchanged_parameters(self):
        return all(torch.equal(value, self.model.state_dict()[key].cpu()) for key,value in self.full_state.items())

    def export(self, keep, path):
        reduced = dict(schema=2, arm='G0', config=self.config, state_dict=self.full_state,
            preprocessor=self.pre, reference_x=self.x[torch.as_tensor(keep, device=self.device)].cpu().numpy(),
            source_ids=self.ids[np.asarray(keep)], original_reference_count=self.n,
            semantics='physical reference subset; all prediction queries retained')
        torch.save(reduced, path)


def engine(bundle, device='cuda', loo=False):
    with GPU_ACCESS():
        if bundle['arm'] == 'G0':
            result = PlainReference(bundle, device, loo)
        elif loo:
            result = agent.LeaveOneOutReference(bundle, device)
        else:
            from rp_agent_reference import ReferencePredictor
            result = ReferencePredictor(bundle['state_dict'], bundle['config'], bundle['reference_x'],
                                        bundle['source_ids'], bundle['preprocessor'], device)
    def guarded(function):
        def call(*args, **kwargs):
            with GPU_ACCESS():
                return function(*args, **kwargs)
        return call
    for name in ('predict', 'native', 'export', 'unchanged_parameters'):
        setattr(result, name, guarded(getattr(result, name)))
    return result


def environment(bundle, device='cuda'):
    with GPU_ACCESS():
        predictor = engine(bundle, device, loo=True)
        full = predictor.predict(np.ones(predictor.n, dtype=bool), bundle['feedback_x'], bundle['feedback_ids'])
        value = float(np.mean((full['scaled'].astype(float)-bundle['feedback_y_scaled'])**2))
        bundle = {**bundle, 'feedback_mse': value, 'loo_feedback_mse': value}
        return agent.MinimumWorld(predictor, bundle, bundle['source'], message_objective=True)


def protected():
    paths = list(r.HERE.glob('*.py'))
    paths.append(r.HERE/'runs/CARBON-BASELINE-DESIGN-002/completion_record.json')
    for region in r.REGIONS:
        for fold in range(1,6):
            folder = r.SOURCE/region/f'fold-{fold:02d}'
            paths.extend((folder/'config.json', folder/'data/development.npz', folder/'data/preprocessor.npz'))
    return {str(path.resolve()): r.digest(path) for path in paths}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = protected()
    for path in (Path(__file__), Path('physical_factorial_v2.py'), Path('TASKS')/(TASK+'.md')):
        shutil.copy2(path, output/path.name)
    r.write_json(output/'input_manifest.json', dict(started_at=now(), protected_inputs=inputs))
    cases = []
    for region in r.REGIONS:
        for arm in ARMS:
            bundle = fit_predictor(region, 1, arm, output/region/arm, small=True)
            predictor = engine(bundle)
            feedback = engine(bundle, loo=True)
            n = predictor.n
            keep = np.ones(n, dtype=bool)
            keep[np.random.default_rng(2401).choice(n, n//5, replace=False)] = False
            qx, ids = bundle['reference_x'], bundle['source_ids']
            actual = predictor.predict(keep, qx, ids)
            native = predictor.native(keep, qx, ids)
            assert np.allclose(actual['scaled'], native, atol=3e-6, rtol=3e-6)
            observed = feedback.predict(keep, qx, ids, graphs=True)
            deltas = []
            for i in (np.flatnonzero(keep)[0], np.flatnonzero(keep)[-1], np.flatnonzero(~keep)[0]):
                trial = keep.copy(); trial[i] = False
                oracle = predictor.native(trial, qx[i:i+1], ids[i:i+1])
                deltas.append(float(abs(oracle[0]-observed['scaled'][i])))
                assert np.allclose(oracle, observed['scaled'][i:i+1], atol=3e-6, rtol=3e-6)
            path = output/region/arm/'reduced.pt'
            predictor.export(keep, path)
            stored = torch.load(path, map_location='cpu', weights_only=False)
            if arm == 'G0':
                restored = engine(stored)
            else:
                stored = {**stored, 'arm':'RP', 'source_ids':stored['reference_ids']}
                restored = engine(stored)
            again = restored.predict(np.ones(restored.n,dtype=bool), qx[:3], ids[:3])
            assert np.allclose(again['scaled'], actual['scaled'][:3], atol=3e-6, rtol=3e-6)
            one = restored.predict(np.ones(restored.n,dtype=bool), qx[1:2], ids[1:2])
            assert np.allclose(one['scaled'], again['scaled'][1:2], atol=3e-6, rtol=3e-6)
            unseen = np.flatnonzero(~keep)[:2]
            new_ids = np.array(['unseen-preflight-'+str(i) for i in unseen])
            unknown = restored.predict(np.ones(restored.n,dtype=bool), qx[unseen], new_ids)
            oracle = restored.native(np.ones(restored.n,dtype=bool), qx[unseen], new_ids)
            assert np.allclose(unknown['scaled'], oracle, atol=3e-6, rtol=3e-6)
            alone = restored.predict(np.ones(restored.n,dtype=bool), qx[unseen[:1]], new_ids[:1])
            assert np.allclose(alone['scaled'], unknown['scaled'][:1], atol=3e-6, rtol=3e-6)
            assert predictor.unchanged_parameters() and feedback.unchanged_parameters()
            cases.append(dict(region=region, arm=arm, valid=True, maximum_LOO_error=max(deltas),
                              plain_RP_parameters=sum(p.numel() for p in predictor.model.parameters())))
            del predictor, feedback, restored, bundle
            gc.collect(); torch.cuda.empty_cache()
    assert protected() == inputs
    r.write_json(output/'preflight.json', dict(valid=True, cases=cases, protected_inputs=inputs,
                                             final_holdout_access=False))
    print(json.dumps(dict(preflight='VALID', cases=len(cases))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r.setup()
    preflight(args.output)
