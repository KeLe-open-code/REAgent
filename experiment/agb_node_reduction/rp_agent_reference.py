"""Physical reference-node reduction for frozen RP, with all queries retained."""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import relation_precision as rp

r = rp.r
TASK = 'CARBON-RP-AGENT-REFERENCE-001'
RP_ROOT = r.HERE/'runs/CARBON-RELATION-PRECISION-001/attempt-01'
NODE_BUFFERS = ('field_support', 'field_directions', 'field_log_norm')


class ReferencePredictor:
    def __init__(self, state, config, reference_x, reference_ids, pre, device):
        self.config = {**config, 'relation_n_train': len(reference_ids)}
        self.model = rp.construct('RP', self.config, device).eval()
        self.model.load_state_dict(state)
        self.model.requires_grad_(False)
        self.device = device
        self.x = torch.as_tensor(reference_x, device=device)
        self.ids = np.asarray(reference_ids).astype(str)
        self.n = len(self.ids)
        self.pre = {k: np.asarray(v).copy() for k, v in pre.items()}
        self.fields = {name: getattr(self.model, name).detach().clone() for name in NODE_BUFFERS}
        self.full_state = r.cpu_state(self.model)
        self.features = self.transform(self.x)
        self.calls = 0

    @classmethod
    def original(cls, region, fold, device):
        source, config, data, pre = r.load_fold(region, fold)
        checkpoint = RP_ROOT/region/f'fold-{fold:02d}'/'RP/best.pt'
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        n = int(data['n_train'])
        return cls(saved['state_dict'], config, data['x'][:n], data['sample_ids'][:n], pre, device), data, checkpoint, source

    def transform(self, x):
        return ((self.model.filtered(x)-self.model.input_mean)/self.model.input_scale).float()

    def select(self, keep):
        keep = np.asarray(keep, dtype=bool)
        assert keep.shape == (self.n,) and keep.sum() >= 9
        indices = torch.as_tensor(np.flatnonzero(keep), device=self.device)
        for name, full in self.fields.items():
            setattr(self.model, name, full.index_select(0, indices))
        self.model._relation_cache.clear()
        return indices

    def packed(self, keep, query_x, query_ids):
        indices = self.select(keep)
        query_x = torch.as_tensor(query_x, device=self.device)
        query_ids = np.asarray(query_ids).astype(str)
        assert len(query_x) == len(query_ids) and len(set(query_ids)) == len(query_ids)
        positions = {sid: i for i, sid in enumerate(self.ids[np.asarray(keep)])}
        extras, gather = [], []
        for i, sid in enumerate(query_ids):
            if sid in positions:
                gather.append(positions[sid])
            else:
                gather.append(len(indices)+len(extras))
                extras.append(i)
        extra_x = query_x[extras]
        x = torch.cat((self.x.index_select(0, indices), extra_x))
        features = torch.cat((self.features.index_select(0, indices), self.transform(extra_x)))
        return indices, x, features, torch.as_tensor(gather, device=self.device)

    @torch.inference_mode()
    def predict(self, keep, query_x, query_ids, graphs=False):
        indices, x, features, gather = self.packed(keep, query_x, query_ids)
        first, second = self.model.layer_graphs(x, len(indices))
        hidden = torch.relu(self.model.weighted_layer(self.model.backbone.first, features, *first[:2]))
        embedding = torch.relu(self.model.weighted_layer(self.model.backbone.second, hidden, *second[:2]))
        scaled = self.model.backbone.head(embedding).squeeze(-1).index_select(0, gather)
        result = {'prediction': scaled.cpu().numpy().astype(float)*self.pre['y_scale']+self.pre['y_mean'],
                  'scaled': scaled.cpu().numpy(), 'embedding': embedding.index_select(0, gather).cpu().numpy(),
                  'reference_count': len(indices), 'query_count': len(gather)}
        if graphs:
            result['graphs'] = [dict(neighbors=indices[g[0]].cpu().numpy(), weights=g[1].cpu().numpy()) for g in (first, second)]
            result['packed_query_positions'] = gather.cpu().numpy()
        self.calls += 1
        return result

    @torch.inference_mode()
    def native(self, keep, query_x, query_ids):
        indices, x, _, gather = self.packed(keep, query_x, query_ids)
        return self.model(x, len(indices), None)[0].index_select(0, gather).cpu().numpy()

    def export(self, keep, path):
        self.select(keep)
        state = r.cpu_state(self.model)
        # These labels are unused at inference; trim them to avoid retaining deleted rows.
        state['label_bins'] = self.full_state['label_bins'][np.asarray(keep)]
        config = {**self.config, 'relation_n_train': int(np.sum(keep))}
        bundle = {'schema': 1, 'state_dict': state, 'config': config, 'preprocessor': self.pre,
                  'reference_x': self.x[torch.as_tensor(keep, device=self.device)].cpu().numpy(), 'reference_ids': self.ids[np.asarray(keep)],
                  'original_reference_count': self.n, 'reference_compression_percent': 100*(1-np.mean(keep)),
                  'semantics': 'message references AND local-field anchors physically reduced; query set unchanged',
                  'parameter_count': sum(p.numel() for p in self.model.parameters()), 'query_labels_required': False}
        torch.save(bundle, path)
        return bundle

    @classmethod
    def restore(cls, path, device):
        bundle = torch.load(path, map_location=device, weights_only=False)
        return cls(bundle['state_dict'], bundle['config'], bundle['reference_x'], bundle['reference_ids'], bundle['preprocessor'], device)

    def unchanged_parameters(self):
        current = r.cpu_state(self.model)
        return all(torch.equal(value, current[name]) for name, value in self.full_state.items() if name not in NODE_BUFFERS)


def metadata_status():
    body = (r.HERE/'TASKS'/f'{TASK}.md').read_text(encoding='utf-8')
    return json.loads(body.split('<!-- TASK_META', 1)[1].split('TASK_META -->', 1)[0])['status']


def preflight(output):
    assert metadata_status() == 'RUNNING'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output/Path(__file__).name)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    started, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    manifest = {str(Path(__file__).resolve()): r.digest(Path(__file__))}
    cases, timings = [], []
    for region in r.REGIONS:
        engine, data, checkpoint, source = ReferencePredictor.original(region, 1, 'cuda')
        for path in (checkpoint, source/'data/development.npz', source/'data/preprocessor.npz'):
            manifest[str(path.resolve())] = r.digest(path)
        n, full = engine.n, np.ones(engine.n, dtype=bool)
        qx, qi = data['x'][:n], data['sample_ids'][:n]
        native_full = engine.native(full, qx, qi)
        ours = engine.predict(full, qx, qi, graphs=True)
        error = float(np.max(np.abs(native_full-ours['scaled'])))
        assert error <= 3e-6
        with np.load(checkpoint.parent/'responses.npz') as saved:
            original_scaled = (saved['prediction'][:n]-engine.pre['y_mean'])/engine.pre['y_scale']
        original_error = float(np.max(np.abs(original_scaled-ours['scaled'])))
        assert original_error <= 3e-6
        rng = np.random.default_rng(20261020)
        keep = full.copy()
        keep[rng.choice(n, n//3, replace=False)] = False
        reduced = engine.predict(keep, qx, qi, graphs=True)
        native_reduced = engine.native(keep, qx, qi)
        reduced_error = float(np.max(np.abs(reduced['scaled']-native_reduced)))
        assert reduced_error <= 3e-6
        kept = np.flatnonzero(keep)
        for graph in reduced['graphs']:
            assert np.isin(graph['neighbors'], kept).all()
            assert not np.any(graph['neighbors'][:len(kept)] == kept[:, None])
            assert np.allclose(graph['weights'].sum(1), 1, atol=1e-6)
        assert reduced['query_count'] == n and np.isfinite(reduced['scaled']).all()
        change = float(np.max(np.abs(reduced['scaled']-ours['scaled'])))
        assert change > 1e-6
        chosen = np.flatnonzero(~keep)[:3]
        alone = engine.predict(keep, qx[chosen[:1]], qi[chosen[:1]])
        multiple = engine.predict(keep, qx[chosen], qi[chosen])
        isolation = float(abs(alone['scaled'][0]-multiple['scaled'][0]))
        assert isolation <= 3e-6
        bundle_path = output/f'{region}_reduced.pt'
        bundle = engine.export(keep, bundle_path)
        restored = ReferencePredictor.restore(bundle_path, 'cuda')
        restored_result = restored.predict(np.ones(restored.n, dtype=bool), qx, qi)
        restore_error = float(np.max(np.abs(restored_result['scaled']-reduced['scaled'])))
        assert restore_error <= 3e-6 and restored.n == keep.sum()
        assert engine.unchanged_parameters()
        assert all(len(bundle['state_dict'][key]) == keep.sum() for key in NODE_BUFFERS)
        assert len(bundle['state_dict']['label_bins']) == keep.sum()
        mask_inputs = []
        for fraction in (1., 2/3, 1/2):
            mask = np.zeros(n, dtype=bool)
            mask[:max(9, round(n*fraction))] = True
            mask_inputs.append(mask)
        for device in ('cpu', 'cuda'):
            bench = engine if device == 'cuda' else ReferencePredictor.original(region, 1, device)[0]
            bench.predict(mask_inputs[0], qx, qi)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for mask in mask_inputs:
                bench.predict(mask, qx, qi)
            if device == 'cuda':
                torch.cuda.synchronize()
            timings.append({'region': region, 'device': device, 'three_real_masks_seconds': time.perf_counter()-t0,
                            'query_count': n, 'reference_counts': [int(m.sum()) for m in mask_inputs]})
            del bench
        case = dict(region=region, fold=1, original_references=n, reduced_references=int(keep.sum()),
                    all_queries_retained=n, native_full_scaled_error=error, original_saved_scaled_error=original_error,
                    native_reduced_scaled_error=reduced_error, query_isolation_scaled_error=isolation,
                    reduced_bundle_restore_scaled_error=restore_error, actual_prediction_change=change,
                    removed_anchors_and_sources_absent=True, no_self_reference=True, original_parameters_unchanged=True,
                    parameter_count=bundle['parameter_count'], final_holdout_access=False, outer_label_reward=False)
        cases.append(case)
        print(json.dumps(case), flush=True)
        del engine, restored
    assert all(r.digest(Path(path)) == value for path, value in manifest.items())
    report = dict(valid=True, cases=cases, timings=timings, started_at=started,
                  finished_at=datetime.now().astimezone().isoformat(), wall_seconds=time.perf_counter()-tick,
                  protected_inputs_unchanged=True, source_hashes=manifest, new_predictor_fits=0, new_agent_fits=0,
                  final_holdout_access=False, outer_validation_metrics_computed=False,
                  interpretation='ENGINEERING ONLY; no new predictive gain or agent benefit established')
    r.write_json(output/'preflight.json', report)
    print(json.dumps({'valid': True, 'timings': timings, 'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight',))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r.setup()
    preflight(args.output)
