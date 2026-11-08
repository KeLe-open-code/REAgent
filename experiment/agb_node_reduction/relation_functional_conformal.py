"""Function-space weak-tree updates of SG endpoint conformal metric factors."""
import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeRegressor
import relation_spherical_attribution as context

r, previous, sphere, layer = context.r, context.previous, context.sphere, context.layer
member = sphere.member
TASK = 'CARBON-RELATION-FUNCTIONAL-CONFORMAL-001'
NEW = ('TF', 'TA', 'SF')
CACHED = previous.CACHED
MAX_TREES, MAX_NODES = 400, 7


class FunctionalOptimizer:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.native = layer.JointOptimizer(model, config)
        self.steps = 0
        self.radius = np.full(2, .01)
        self.accumulator = np.zeros(2)
        self.mean = np.zeros((len(model.field_support), 2))
        self.variance = np.zeros_like(self.mean)
        self.history = []

    def zero_grad(self, set_to_none=True):
        self.native.zero_grad(set_to_none=set_to_none)
        self.model.field_values = None

    @torch.no_grad()
    def step(self):
        self.steps += 1
        model = self.model
        values = model.field_values
        assert values is not None and values.grad is not None
        n = len(model.field_support)
        gradient = n * values.grad.detach().double().cpu().numpy()
        gradient -= gradient.mean(0)
        assert gradient.shape == (n, 2) and np.isfinite(gradient).all()
        self.native.step()
        if self.steps == 1 or model._disable_functional:
            self.history.append({'step': self.steps, 'native_only': True, 'weak_trees': int(model.weak_count)})
            return
        function = values.detach().double().cpu().numpy()
        rms = np.sqrt(np.square(function).mean(0))
        self.radius = np.maximum(self.radius, rms)
        self.accumulator += np.square(gradient).mean(0)
        if model.functional_arm == 'TA':
            self.mean = .9*self.mean + .1*gradient
            self.variance = .999*self.variance + .001*np.square(gradient)
            count = self.steps - 1
            target = (self.mean/(1-.9**count))/(np.sqrt(self.variance/(1-.999**count))+1e-8)
            target -= target.mean(0)
            factor = np.full(2, self.config['learning_rate'])
        else:
            target = gradient.copy()
            if model.functional_arm == 'SF':
                permutation = np.random.default_rng(self.config['seed']+90000+self.steps).permutation(n)
                target = target[permutation]
            factor = np.divide(self.radius, np.sqrt(self.accumulator),
                               out=np.zeros(2), where=self.accumulator > 0)
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(5, math.ceil(.05*n)),
                                     random_state=self.config['seed']+self.steps)
        tree.fit(model.training_features, target)
        fitted = tree.predict(model.training_features)
        assert np.isfinite(fitted).all()
        alignment = np.mean(gradient*fitted, axis=0)
        model.append_tree(tree, -factor)
        self.last_tree, self.last_factor = tree, -factor
        self.history.append({'step': self.steps, 'native_only': False,
            'weak_trees': int(model.weak_count), 'tree_nodes': int(tree.tree_.node_count),
            'tree_leaves': int(tree.tree_.n_leaves), 'field_rms_before': rms.tolist(),
            'surrogate_gradient_rms': np.sqrt(np.square(gradient).mean(0)).tolist(),
            'weak_direction_alignment_to_actual_gradient': alignment.tolist(),
            'factor': factor.tolist(), 'radius': self.radius.tolist(),
            'squared_gradient_accumulator': self.accumulator.tolist(),
            'weak_target_MSE': np.square(fitted-target).mean(0).tolist()})

    def diagnostics(self):
        return {**self.native.diagnostics(), 'functional_arm': self.model.functional_arm,
                'functional_history_at_training_end_not_selected_checkpoint': self.history}


class ConformalSAGE(previous.CurvatureSAGE):
    training_objective = 'native_SG_loss_with_functional_endpoint_full_SPD_conformal_factors'

    def __init__(self, arm, config):
        super().__init__('JA', config)
        self.functional_arm = arm
        self.register_buffer('weak_feature', torch.zeros((MAX_TREES, MAX_NODES), dtype=torch.long))
        self.register_buffer('weak_left', torch.full((MAX_TREES, MAX_NODES), -1, dtype=torch.long))
        self.register_buffer('weak_right', torch.full((MAX_TREES, MAX_NODES), -1, dtype=torch.long))
        self.register_buffer('weak_threshold', torch.zeros((MAX_TREES, MAX_NODES), dtype=torch.float64))
        self.register_buffer('weak_value', torch.zeros((MAX_TREES, MAX_NODES, 2), dtype=torch.float64))
        self.register_buffer('weak_nodes', torch.zeros(MAX_TREES, dtype=torch.long))
        self.register_buffer('weak_center', torch.zeros(2, dtype=torch.float64))
        self.register_buffer('weak_count', torch.zeros((), dtype=torch.long))
        self.field_values, self.training_features = None, None
        self._disable_functional = False
        self.executed_optimizer = None

    @torch.no_grad()
    def fit_reference(self, x, edges):
        saved = torch.load(self.reference, map_location=x.device, weights_only=False)['state_dict']
        fresh = r.cpu_state(self)
        for name, value in self.named_buffers():
            if name in saved:
                value.copy_(saved[name])
        for name, parameter in self.named_parameters():
            assert torch.equal(parameter.cpu(), fresh[name])
        assert int(self.weak_count) == 0 and torch.count_nonzero(self.layer_log_metric) == 0
        assert edges.max() < len(x) and torch.equal(self.coordinates(x), self.field_support)
        assert bool(self.field_ready)
        self.training_features = x.detach().double().cpu().numpy().copy()
        self._relation_cache.clear()

    @torch.no_grad()
    def append_tree(self, tree, factor):
        index, source = int(self.weak_count), tree.tree_
        assert index < MAX_TREES and source.node_count <= MAX_NODES
        size = source.node_count
        for name, value in [('weak_feature', source.feature.clip(0)), ('weak_left', source.children_left),
                            ('weak_right', source.children_right), ('weak_threshold', source.threshold)]:
            destination = getattr(self, name)
            destination[index, :size].copy_(torch.as_tensor(value, dtype=destination.dtype, device=destination.device))
        self.weak_value[index, :size].copy_(torch.as_tensor(
            source.value[:, :, 0]*factor[None, :], dtype=torch.float64, device=self.weak_value.device))
        self.weak_nodes[index] = size
        center = (tree.predict(self.training_features)*factor[None, :]).mean(0)
        self.weak_center.add_(torch.as_tensor(center, device=self.weak_center.device, dtype=torch.float64))
        self.weak_count.add_(1)

    def function(self, x, n_train):
        count = int(self.weak_count)
        if not count or self._disable_functional:
            return torch.zeros((len(x), 2), device=x.device, dtype=torch.float64)
        nodes = torch.zeros((count, len(x)), device=x.device, dtype=torch.long)
        trees = torch.arange(count, device=x.device)[:, None]
        samples = torch.arange(len(x), device=x.device)[None, :]
        for _ in range(2):
            left = self.weak_left[trees, nodes]
            right = self.weak_right[trees, nodes]
            feature = self.weak_feature[trees, nodes]
            threshold = self.weak_threshold[trees, nodes]
            # sklearn CART compares inputs after conversion to float32.
            values = x.float()[samples, feature].double()
            next_node = torch.where(values <= threshold, left, right)
            nodes = torch.where(left < 0, nodes, next_node)
        result = self.weak_value[trees, nodes].sum(0)
        return result - self.weak_center[None, :]

    def distances(self, u, index):
        logarithm = self.layer_log_metric[index]
        weights, source_weights = self.endpoint_weights(u, logarithm)
        root = (.5*logarithm).exp()
        query, support = u*root, self.field_support*root
        scores, reference = query@self.field_directions.T, support@self.field_directions.T
        left = (weights*scores.square()).sum(1)[:, None]+weights@reference.square().T-2*(weights*scores)@reference.T
        right = scores.square()@source_weights.T+(source_weights*reference.square()).sum(1)[None, :]-2*scores@(source_weights*reference).T
        base = member.a.distance(query, support, torch.eye(185, device=u.device, dtype=u.dtype))
        native = (185*(left+right)*.5+.01*base)/1.01
        values = self.field_values[:, index]
        n = len(support)
        correction = ((values.expm1()[:, None]*(185*left+.01*base)
                      +values[:n].expm1()[None, :]*(185*right+.01*base))*.5)/1.01
        distance = (native+correction).clamp_min(0)
        diagonal = ((185*(weights@self.field_directions.square())+.01)/1.01
                    *logarithm.exp()*values.exp()[:, None])
        return distance, diagonal.float()

    def graph_for_layer(self, u, index, training):
        distance, diagonal = self.distances(u, index)
        n = len(self.field_support)
        mask = torch.arange(len(u), device=u.device)[:, None] == torch.arange(n, device=u.device)[None, :]
        distance = distance.masked_fill(mask, torch.inf)
        near = torch.argsort(distance, dim=1, stable=True)[:, :8]
        scores = -distance/self.temperature
        hard = torch.softmax(scores.gather(1, near), dim=1).float()
        if not training:
            return near, hard, diagonal
        membership = member.StableSigmoidTopK.apply(scores, 8)
        exponential = torch.softmax(scores, dim=1)
        coefficient = exponential/exponential.gather(1, near).sum(1, keepdim=True)
        return near, hard, diagonal, membership, coefficient

    def layer_graphs(self, x, n_train):
        assert n_train == len(self.field_support) and bool(self.field_ready)
        self.field_values = self.function(x, n_train)
        u = self.coordinates(x)
        return tuple(self.graph_for_layer(u, index, False) for index in range(2))

    def relations(self, x, n_train, edges):
        if not bool(self.field_ready):
            return super().relations(x, n_train, edges)
        return self.layer_graphs(x, n_train)[0]

    def forward(self, x, n_train, edges):
        if not bool(self.field_ready):
            return super().forward(x, n_train, edges)
        training = self.training and torch.is_grad_enabled()
        self.field_values = self.function(x, n_train).detach().requires_grad_(training)
        u = self.coordinates(x)
        graphs = [self.graph_for_layer(u, index, training) for index in range(2)]
        def apply(module, values, graph):
            if training:
                near, hard, _, membership, coefficient = graph
                aggregate = member.membership_aggregate(values, near, hard, membership, coefficient)
                return module.lin_l(aggregate)+module.lin_r(values)
            return self.weighted_layer(module, values, graph[0], graph[1])
        features = ((self.filtered(x)-self.input_mean)/self.input_scale).float()
        hidden = self.backbone.dropout(torch.relu(apply(self.backbone.first, features, graphs[0])))
        raw = apply(self.backbone.second, hidden, graphs[1])
        self.geometry_feature = raw
        embedding = torch.relu(raw)
        return self.backbone.head(embedding).squeeze(-1), embedding, *graphs[0][:3]

    def make_optimizer(self, config):
        self.executed_optimizer = FunctionalOptimizer(self, config)
        return self.executed_optimizer


def construct(arm, config, device):
    assert arm in NEW
    config = dict(config)
    if 'relation_n_train' not in config:
        with np.load(r.SOURCE/config['region']/f"fold-{config['fold']:02d}"/'data/development.npz') as z:
            config['relation_n_train'] = int(z['n_train'])
    r.fix_seed(config['seed'], 1)
    return ConformalSAGE(arm, config).to(device)


def math_checks():
    generator = torch.Generator().manual_seed(20261019)
    u = torch.randn(5, 4, dtype=torch.float64, generator=generator)
    atoms = torch.randn(5, 4, 4, dtype=torch.float64, generator=generator)
    matrices = atoms@atoms.transpose(1, 2)+.01*torch.eye(4, dtype=torch.float64)
    delta = u[:, None]-u[None, :]
    left = torch.einsum('ijd,idk,ijk->ij', delta, matrices, delta)
    right = torch.einsum('ijd,jdk,ijk->ij', delta, matrices, delta)
    f = torch.linspace(-.4, .4, 5, dtype=torch.float64).requires_grad_()
    def energy(values):
        return .5*(left+right)+.5*(values.expm1()[:, None]*left+values.expm1()[None, :]*right)
    direct = .5*(f.exp()[:, None]*left+f.exp()[None, :]*right)
    error = float((energy(f)-direct).abs().max().detach())
    assert error < 1e-10 and torch.equal(energy(f*0), .5*(left+right))
    assert torch.autograd.gradcheck(energy, (f,), eps=1e-6, atol=1e-6, rtol=1e-5)
    weight = torch.rand((5, 5), dtype=torch.float64, generator=generator)
    actual = torch.autograd.grad((weight*energy(f)).sum(), f)[0]
    receiver = .5*f.exp()*(weight*left).sum(1)
    source = .5*f.exp()*(weight*right).sum(0)
    assert torch.allclose(actual, receiver+source, atol=1e-12, rtol=1e-12)
    assert receiver.norm() > 0 and source.norm() > 0
    return {'full_SPD_factor_formula_max_error': error, 'factor_gradcheck': True,
            'receiver_and_source_incident_gradient': True, 'zero_factor_native_algebra_exact': True}


def source_hashes():
    return context.source_hashes()


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    start, algebra, cases, identities = time.perf_counter(), math_checks(), [], []
    for region in r.REGIONS:
        _, config, data, pre = r.load_fold(region, 1)
        n = int(data['n_train']); x, te, ee, target = r.tensors(data, pre, 'cuda')
        for arm in NEW:
            model = construct(arm, config, 'cuda')
            native = previous.construct('JA', config, 'cuda')
            for current in (model, native):
                current.fit_reference(x[:n], te); current.fit_relations(x[:n], target, te); current.eval()
            with torch.no_grad():
                assert torch.equal(model(x, n, ee)[0], native(x, n, ee)[0])
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in native.state_dict().items())
            optimizer, reference_optimizer = model.make_optimizer(config), layer.JointOptimizer(native, config)
            serialization_error, first_exact = 0., False
            for step in range(8):
                model.train(); optimizer.zero_grad()
                rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                loss = model.training_loss(x[:n], target, te)
                assert torch.isfinite(loss)
                loss.backward()
                assert torch.isfinite(model.field_values.grad).all()
                old_field = model.field_values.detach().cpu().numpy().copy()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                if step == 0:
                    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
                    native.train(); reference_optimizer.zero_grad()
                    other = native.training_loss(x[:n], target, te); other.backward()
                    assert torch.equal(loss, other)
                    torch.nn.utils.clip_grad_norm_(native.parameters(), config['gradient_clip'])
                    assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), native.parameters()))
                    reference_optimizer.step()
                optimizer.step()
                if step == 0:
                    assert all(torch.equal(value, model.state_dict()[key]) for key, value in native.state_dict().items())
                    assert int(model.weak_count) == 0
                    first_exact = True
                else:
                    increment = optimizer.last_tree.predict(model.training_features)*optimizer.last_factor[None, :]
                    expected = old_field+increment
                    expected -= expected.mean(0)
                    got = model.function(x[:n], n).detach().cpu().numpy()
                    serialization_error = max(serialization_error, float(np.abs(expected-got).max()))
                    assert np.allclose(expected, got, atol=1e-10, rtol=1e-10)
            model.eval()
            with torch.no_grad():
                full = model(x, n, ee)[0]
                field = model.function(x, n)
                alone = model(x[:n], n, te)[0]
                single = model(torch.cat([x[:n], x[n:n+1]]), n, ee)[0]
                assert torch.allclose(full[:n], alone, atol=3e-6, rtol=3e-6)
                assert torch.allclose(full[n], single[n], atol=3e-6, rtol=3e-6)
                assert field[:n].mean(0).abs().max() < 1e-10 and torch.isfinite(field.exp()).all()
            restored = construct(arm, config, 'cuda').eval(); restored.load_state_dict(r.cpu_state(model))
            with torch.no_grad():
                assert torch.equal(restored(x, n, ee)[0], full)
            cases.append({'region': region, 'arm': arm, 'real_updates': 8,
                'initial_and_native_first_update_exact': first_exact, 'query_isolation': True,
                'checkpoint_exact': True, 'tree_serialization_max_error': serialization_error,
                'weak_trees': int(model.weak_count), 'field_rms': field[:n].square().mean(0).sqrt().tolist(),
                'two_layers_feedback_nonzero': all(any(v.get('surrogate_gradient_rms', [0, 0])[i] > 0 for v in optimizer.history) for i in range(2))})
            assert cases[-1]['two_layers_feedback_nonzero']
            json.dumps(optimizer.diagnostics(), allow_nan=False)
        model = construct('TF', config, 'cuda'); model._disable_functional = True
        native = previous.construct('JA', config, 'cuda')
        for current in (model, native):
            current.fit_reference(x[:n], te); current.fit_relations(x[:n], target, te)
        optimizers = model.make_optimizer(config), layer.JointOptimizer(native, config)
        for _ in range(4):
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            for index, (current, optimizer) in enumerate(zip((model, native), optimizers)):
                torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
                current.train(); optimizer.zero_grad()
                loss = current.training_loss(x[:n], target, te); loss.backward()
                torch.nn.utils.clip_grad_norm_(current.parameters(), config['gradient_clip']); optimizer.step()
            assert all(torch.equal(value, model.state_dict()[key]) for key, value in native.state_dict().items())
        identities.append({'region': region, 'four_native_steps_exact_with_function_disabled': True})
    r.write_json(output/'preflight.json', {'cases': cases, 'identities': identities, 'algebra': algebra,
        'source_sha256': r.digest(Path(__file__)), 'source_hashes': source_hashes(),
        'seconds': time.perf_counter()-start})
    print('PREFLIGHT PASS', len(cases), flush=True)


def directory(output, region, fold, arm):
    root, name = CACHED[arm] if arm in CACHED else (output, arm)
    return root/region/f'fold-{fold:02d}'/name


@torch.no_grad()
def export(model, data, pre, output, restore):
    n = int(data['n_train']); x, te, ee, _ = r.tensors(data, pre, 'cuda')
    model.eval()
    actual = model(x, n, ee)[0]
    prediction = actual.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean']
    with np.load(output/'responses.npz') as z:
        assert np.array_equal(z['prediction'], prediction)
    field = model.function(x, n)
    graphs = model.layer_graphs(x, n)
    model._disable_functional = True
    bypass = model(x, n, ee)[0]
    bypass_graphs = model.layer_graphs(x, n)
    model._disable_functional = False
    arrays = {'sample_ids': data['sample_ids'], 'field': field.cpu().numpy()}
    changes = []
    for index, (graph, reference) in enumerate(zip(graphs, bypass_graphs), 1):
        near, weights = graph[:2]
        assert near.shape == weights.shape == (len(x), 8) and (near < n).all()
        assert not (near == torch.arange(len(x), device=x.device)[:, None]).any()
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        assert torch.allclose(weights.sum(1), torch.ones(len(x), device=x.device), atol=1e-6, rtol=0)
        arrays[f'layer{index}_neighbors'], arrays[f'layer{index}_weights'] = near.cpu().numpy(), weights.cpu().numpy()
        changes.append({'layer': index, 'changed_neighbor_entries': int((near != reference[0]).sum()),
                        'sorted_weight_max_difference': float((weights-reference[1]).abs().max())})
    assert field[:n].mean(0).abs().max() < 1e-10 and torch.isfinite(field.exp()).all()
    count = int(model.weak_count)
    leaves = sum(int((model.weak_left[i, :int(model.weak_nodes[i])] < 0).sum()) for i in range(count))
    nodes = int(model.weak_nodes[:count].sum())
    active = bool(field.abs().max() > 1e-10 and any(
        d['changed_neighbor_entries'] > 0 or d['sorted_weight_max_difference'] > 1e-8 for d in changes))
    np.savez_compressed(output/'functional_graphs.npz', **arrays)
    np.savez_compressed(output/'functional_counterfactual.npz', sample_ids=data['sample_ids'],
        actual=prediction, bypass=bypass.cpu().numpy().astype(float)*pre['y_scale']+pre['y_mean'])
    checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
    assert count == checkpoint['epoch']-1
    report = {'selected_epoch': checkpoint['epoch'], 'selected_trees': count,
        'selected_tree_nodes': nodes, 'learned_leaf_output_coefficients': 2*leaves,
        'learned_internal_thresholds': nodes-leaves,
        'original_backprop_parameter_count': sum(p.numel() for p in model.parameters()),
        'field_train_rms': field[:n].square().mean(0).sqrt().tolist(),
        'field_max_abs': float(field.abs().max()), 'training_mean_error': float(field[:n].mean(0).abs().max()),
        'positive_factor_min': float(field.exp().min()), 'positive_factor_max': float(field.exp().max()),
        'actual_vs_disabled_scaled_prediction_max_difference': float((actual-bypass).abs().max()),
        'selected_deployment_active': active, 'graph_changes': changes, 'restore_checked': restore}
    if restore:
        fresh = construct(model.functional_arm, checkpoint['config'], 'cuda').eval()
        fresh.load_state_dict(checkpoint['state_dict'])
        assert torch.equal(fresh(x, n, ee)[0], actual)
        assert torch.allclose(fresh.function(x[n:n+1], n)[0], field[n], atol=1e-12, rtol=1e-12)
        report['restore_exact'] = True
    r.write_json(output/'functional_diagnostics.json', report)
    r.write_json(output/'functional_history.json', model.executed_optimizer.history)
    r.write_json(output/'geometry_loss_history.json', model.geometry_history)


def analyze(output):
    folds, points, training = [], [], []
    methods = (*CACHED, *NEW)
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, config, data, _ = r.load_fold(region, fold)
            n, y = int(data['n_train']), data['y']
            predictions = {}
            for arm in methods:
                path = directory(output, region, fold, arm)
                saved = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
                name = CACHED[arm][1] if arm in CACHED else arm
                assert (saved['region'], saved['fold'], saved['arm'], saved['seed']) == (region, fold, name, config['seed'])
                with np.load(path/'responses.npz') as z:
                    assert np.array_equal(z['sample_ids'], data['sample_ids'])
                    prediction = z['prediction'].copy(); predictions[arm] = prediction
                mt, mv = r.metrics(y[:n], prediction[:n]), r.metrics(y[n:], prediction[n:])
                assert all(abs(value[key]-saved[role][key]) < 1e-7 for role, value in [('train', mt), ('validation', mv)] for key in value)
                row = {'region': region, 'fold': fold, 'arm': arm, **mv, 'train_rmse': mt['rmse'],
                    'best_epoch': saved['best_epoch'], 'epochs_run': saved['epochs_run'], 'elapsed_seconds': saved['elapsed_seconds']}
                if arm in NEW:
                    row.update(json.loads((path/'functional_diagnostics.json').read_text(encoding='utf-8')))
                    row.pop('graph_changes')
                folds.append(row)
            q80 = np.quantile(y[:n], .8)
            for i, identity in enumerate(data['sample_ids']):
                row = {'sample_id': str(identity), 'region': region, 'fold': fold,
                    'observed_agb': float(y[i]), 'high_agb': int(y[i] >= q80),
                    'missing_features_current': int(data['missing_mask'][i].sum())}
                for arm, prediction in predictions.items():
                    row[arm+'_prediction'], row[arm+'_squared_error'] = float(prediction[i]), float((prediction[i]-y[i])**2)
                (training if i < n else points).append(row)
    f, p, t = pd.DataFrame(folds), pd.DataFrame(points), pd.DataFrame(training)
    assert len(f) == 90 and len(p) == p.sample_id.nunique() == 1421 and len(t) == 5684
    p = p.merge(pd.read_csv(context.PAIN/'all_1421_ids.csv', float_precision='round_trip'),
        on=['sample_id', 'region', 'fold', 'observed_agb'], how='left', validate='one_to_one', suffixes=('', '_archive'))
    p = p.merge(pd.read_csv(context.NAMED, float_precision='round_trip'),
        on=['sample_id', 'region', 'fold'], how='left', validate='one_to_one')
    p = p.merge(pd.read_csv(context.TAIL, float_precision='round_trip'),
        on=['sample_id', 'region'], how='left', validate='one_to_one', suffixes=('', '_tail'))
    agg = t.groupby('sample_id').agg(TF_train_MSE=('TF_squared_error', 'mean'), SG_train_MSE=('SG_squared_error', 'mean'))
    p = p.join(agg, on='sample_id', validate='one_to_one')
    assert len(p) == 1421 and p.feature1.notna().all() and p.TF_train_MSE.notna().all()
    p['TF_minus_SG_SSE'] = p.TF_squared_error-p.SG_squared_error
    strata = []
    for region, group in p.groupby('region'):
        masks = {'all': np.ones(len(group), dtype=bool), 'high': group.high_agb == 1, 'ordinary': group.high_agb == 0,
            'missing': group.missing_features_current > 0, 'complete': group.missing_features_current == 0,
            'complete_typical': (group.missing_features_current == 0)&(group.max_abs_training_standardized_feature <= 3),
            'persistent_tail': group.tail_appearances >= 3}
        for label, mask in masks.items():
            selected = group[mask]
            if len(selected):
                strata.append({'region': region, 'stratum': label, 'n': len(selected),
                    'TF_minus_SG_SSE': float(selected.TF_minus_SG_SSE.sum()),
                    'TF_bias': float((selected.TF_prediction-selected.observed_agb).mean()),
                    'SG_bias': float((selected.SG_prediction-selected.observed_agb).mean())})
    feature_rows = []
    for region in r.REGIONS:
        group = p[p.region == region]; raw = {}
        for fold in range(1, 6):
            _, _, data, _ = r.load_fold(region, fold); n = int(data['n_train'])
            for i in range(n, len(data['y'])):
                raw[str(data['sample_ids'][i])] = data['raw_features'][i]
        matrix = np.stack([raw[identity] for identity in group.sample_id])
        change = group.TF_minus_SG_SSE.to_numpy()
        for i, name in enumerate(data['feature_names'].astype(str)):
            values = matrix[:, i]; valid = np.isfinite(values)&(values != -9999)
            correlation = pd.Series(values[valid]).corr(pd.Series(change[valid]), method='spearman')
            feature_rows.append({'region': region, 'feature': name, 'valid_n': int(valid.sum()),
                'spearman_feature_vs_TF_SSE_change': float(correlation) if np.isfinite(correlation) else None,
                'missing_SSE_change': float(change[~valid].sum()), 'observed_SSE_change': float(change[valid].sum())})
    assert len(feature_rows) == 555
    f.to_csv(output/'all_90_fold_metrics.csv', index=False)
    f.groupby(['region', 'arm'])[['rmse', 'mae', 'r2', 'bias', 'train_rmse']].mean().to_csv(output/'province_metrics.csv')
    p.to_csv(output/'all_1421_point_analysis.csv', index=False); t.to_csv(output/'all_5684_training_appearances.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'failure_strata.csv', index=False)
    pd.DataFrame(feature_rows).to_csv(output/'all_555_feature_error_context.csv', index=False)
    pd.concat([p.nsmallest(20, 'TF_minus_SG_SSE'), p.nlargest(20, 'TF_minus_SG_SSE')]).to_csv(output/'forty_explanatory_cases.csv', index=False)
    pivot = f.pivot(index=['region', 'fold'], columns='arm', values='rmse')
    gains = {arm: float((100*(1-pivot[arm]/pivot.G0)).mean()) for arm in methods if arm != 'G0'}
    comparisons = {arm: float((100*(1-pivot.TF/pivot[arm])).mean()) for arm in ('TA', 'SF', 'SG', 'GR')}
    province = f.groupby(['region', 'arm'])[['rmse', 'r2']].mean()
    noninferior = all(province.loc[(region, 'TF'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*(1+1e-6)
        and province.loc[(region, 'TF'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-1e-6 for region in r.REGIONS)
    cap = all(province.loc[(region, 'TF'), 'rmse'] <= province.loc[(region, 'G0'), 'rmse']*1.01
        and province.loc[(region, 'TF'), 'r2'] >= province.loc[(region, 'G0'), 'r2']-.01 for region in r.REGIONS)
    active = f[f.arm == 'TF'].set_index(['region', 'fold']).selected_deployment_active.astype(bool)
    deployment = int(active.sum()) >= 9 and all(int(active.loc[region].sum()) >= 2 for region in r.REGIONS)
    full = gains['TF'] >= 10 and comparisons['TA'] >= .5 and comparisons['SF'] >= .5 and comparisons['SG'] > 0 and comparisons['GR'] > 0 and noninferior and deployment
    pareto = gains['TF'] >= 3 and comparisons['SG'] >= .5 and all(value > 0 for value in comparisons.values()) and cap and deployment
    report = {'candidate_G0_RMSE_gains_percent': gains, 'TF_paired_RMSE_reductions_percent': comparisons,
        'province_G0_noninferiority': bool(noninferior), 'active_selected_TF_folds': int(active.sum()),
        'deployment_gate': bool(deployment), 'FULL': bool(full), 'PARETO': bool(pareto),
        'classification': 'FULL_GAIN' if full else 'PARETO_GAIN' if pareto else 'NO_GAIN',
        'TF_helped_SG': int((p.TF_minus_SG_SSE < 0).sum()), 'TF_harmed_SG': int((p.TF_minus_SG_SSE > 0).sum()),
        'TF_minus_SG_SSE': float(p.TF_minus_SG_SSE.sum()), 'all_1421_IDs': True,
        'training_appearances_per_new_method': 5684, 'feature_context_rows': 555, 'final_holdout_access': False}
    r.write_json(output/'summary.json', report)
    print(json.dumps(report), flush=True)


def run(output, preflight_dir):
    check = json.loads((preflight_dir/'preflight.json').read_text(encoding='utf-8'))
    assert len(check['cases']) == 9 and check['source_sha256'] == r.digest(Path(__file__))
    assert all(r.digest(Path(p)) == h for p, h in check['source_hashes'].items())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(r.HERE/'TASKS'/f'{TASK}.md', output/'task_frozen.md')
    shutil.copy2(Path(__file__), output/Path(__file__).name)
    files = [output/'task_frozen.md', preflight_dir/'preflight.json', context.PAIN/'all_1421_ids.csv',
        context.PAIN/'feature_diagnostics.csv', context.NAMED, context.TAIL]
    for region in r.REGIONS:
        for fold in range(1, 6):
            source, _, _, _ = r.load_fold(region, fold)
            files.extend([source/'config.json', *[source/'data'/v for v in ('development.npz', 'preprocessor.npz', 'split.csv')]])
            for cache, arm in CACHED.values():
                files.extend(cache/region/f'fold-{fold:02d}'/arm/v for v in ('best.pt', 'metrics.json', 'responses.npz'))
    hashes = {**source_hashes(), **{str(p.resolve()): r.digest(p) for p in files}}
    started, tick = datetime.now().astimezone().isoformat(), time.perf_counter()
    r.write_json(output/'manifest.json', {'task_id': TASK, 'started_at': started,
        'source_input_hashes': hashes, 'new_fits': 45, 'cached_fits': 45,
        'device': torch.cuda.get_device_name(), 'final_holdout_access': False})
    export_seconds = 0.
    for region in r.REGIONS:
        for fold in range(1, 6):
            _, _, data, pre = r.load_fold(region, fold)
            for arm in NEW:
                holder = []
                def factory(name, config, device):
                    model = construct(name, config, device); holder.append(model); return model
                r.fit(arm, region, fold, output, 'cuda', model_factory=factory, task_id=TASK, max_gnn_epochs=400)
                mark = time.perf_counter()
                export(holder[0], data, pre, directory(output, region, fold, arm), arm == 'TF' and fold == 1)
                export_seconds += time.perf_counter()-mark
                holder.clear()
    fitting = time.perf_counter()-tick
    analyze(output)
    assert all(r.digest(Path(p)) == h for p, h in hashes.items())
    r.write_json(output/'finished.json', {'started_at': started, 'finished_at': datetime.now().astimezone().isoformat(),
        'fitting_export_seconds': fitting, 'graph_export_seconds': export_seconds,
        'wall_seconds': time.perf_counter()-tick, 'protected_hashes_unchanged': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-dir', type=Path)
    args = parser.parse_args(); r.setup()
    if args.mode == 'preflight':
        preflight(args.output)
    elif args.mode == 'run':
        run(args.output, args.preflight_dir)
    else:
        analyze(args.output)
