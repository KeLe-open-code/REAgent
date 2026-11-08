import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.compression import ReducedReference, rebuild_reference_graph
from evaluation.dynamic_agent import DynamicFunctionAgent
from evaluation.dynamic_rp import DynamicPrecisionSAGE, DynamicReferencePredictor
from evaluation.eco_calibration import EcologicalOOFCalibrator
from evaluation.metrics import compression_metrics, regression_metrics
from evaluation.public_models import construct_public_model


class ParentEngine:
    def __init__(self, kind, model, reference_x, reference_ids, coords, ecology, preprocessor, device='cpu'):
        self.kind = kind
        self.model = model
        self.x = np.asarray(reference_x, dtype=np.float32)
        self.ids = np.asarray(reference_ids).astype(str)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.ecology = np.asarray(ecology).astype(str)
        self.pre = {key: np.asarray(value).copy() for key, value in preprocessor.items()}
        self.device = device
        self.n = len(self.ids)
        if kind == 'RP':
            self.engine = DynamicReferencePredictor(model, self.x, self.ids, self.pre, device)
        else:
            self.engine = None

    def parent_state(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            if name.split('.')[-1] in ('field_support', 'field_directions', 'field_log_norm'):
                continue
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def predict(self, keep, query_x, query_ids):
        keep = np.asarray(keep, dtype=bool)
        query_x = np.asarray(query_x, dtype=np.float32)
        query_ids = np.asarray(query_ids).astype(str)
        if self.kind == 'RP':
            return self.engine.predict(keep, query_x, query_ids)
        selected = np.flatnonzero(keep)
        reduced = ReducedReference(self.x[selected], self.ids[selected], self.coords[selected],
                                   self.ecology[selected], 'G0', self.n)
        adjacency = rebuild_reference_graph(reduced, query_x, k=8)
        edges = torch.as_tensor(np.array(adjacency.nonzero()), dtype=torch.long, device=self.device)
        x = torch.as_tensor(np.concatenate((reduced.reference_x, query_x)), device=self.device)
        with torch.inference_mode():
            scaled = self.model(x, edges)[len(selected):].cpu().numpy().astype(float)
        raw = scaled*self.pre['y_scale']+self.pre['y_mean']
        return {'prediction': raw, 'reference_count': len(selected), 'query_count': len(query_ids)}


def fit_preprocessor(raw, target):
    raw = np.asarray(raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    median = np.nanmedian(raw, axis=0)
    imputed = np.where(np.isfinite(raw), raw, median[None, :])
    mean = imputed.mean(axis=0)
    scale = np.where(imputed.std(axis=0) > 1e-8, imputed.std(axis=0), 1.)
    return {'median': median, 'mean': mean, 'scale': scale,
            'y_mean': np.asarray(float(target.mean())), 'y_scale': np.asarray(max(float(target.std()), 1e-8))}


def transform(raw, pre):
    raw = np.asarray(raw, dtype=np.float64)
    imputed = np.where(np.isfinite(raw), raw, pre['median'][None, :])
    return ((imputed-pre['mean'])/pre['scale']).astype(np.float32)


def fit_parent(kind, raw, ids, target, coords, ecology, seed, config, device='cpu', epochs=None, selection=None):
    assert kind in ('G0', 'RP')
    raw = np.asarray(raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert len(raw) == len(ids) == len(target) == len(coords) == len(ecology)
    pre = fit_preprocessor(raw, target)
    x = transform(raw, pre)
    n = len(x)
    model_config = {**config, 'input_dim': x.shape[1], 'relation_n_train': n, 'seed': int(seed)}
    torch.manual_seed(seed)
    if kind == 'RP':
        model = DynamicPrecisionSAGE(model_config).to(device)
        model.fit_reference(torch.as_tensor(x, device=device))
        model.fit_relations(torch.as_tensor(x, device=device),
                            torch.as_tensor((target-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device=device))
        optimizer = model.make_optimizer(model_config)
    else:
        model = construct_public_model('graphsage', x.shape[1], seed).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    adjacency = rebuild_reference_graph(ReducedReference(x, ids, coords, ecology, kind, n), x[:0], k=8)
    edges = torch.as_tensor(np.array(adjacency.nonzero()), dtype=torch.long, device=device)
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor((target-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device=device)
    if selection is not None:
        selection_x = transform(selection['raw_features'], pre)
        selection_adjacency = rebuild_reference_graph(ReducedReference(x, ids, coords, ecology, kind, n), selection_x, k=8)
        selection_edges = torch.as_tensor(np.array(selection_adjacency.nonzero()), dtype=torch.long, device=device)
        all_x = torch.as_tensor(np.concatenate((x, selection_x)), device=device)
        selection_target = torch.as_tensor((selection['y']-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device=device)
    epochs = int(config.get('max_epochs', 400) if epochs is None else epochs)
    assert epochs >= 1
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == 'RP':
            loss = model.training_loss(x_tensor, y_tensor, edges)
        else:
            loss = torch.nn.functional.mse_loss(model(x_tensor, edges), y_tensor)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 5.))
        optimizer.step()
        record = {'epoch': epoch, 'loss': float(loss.detach())}
        if selection is not None:
            model.eval()
            with torch.inference_mode():
                if kind == 'RP':
                    selection_prediction = model(all_x, n, selection_edges)[0][n:]
                else:
                    selection_prediction = model(all_x, selection_edges)[n:]
                record['selection_loss'] = float(torch.nn.functional.mse_loss(selection_prediction, selection_target))
        history.append(record)
    model.eval()
    engine = ParentEngine(kind, model, x, ids, coords, ecology, pre, device)
    return engine, history


def fit_agent(engine, coords, seed, k=None, episodes=4, candidates_per_step=4):
    k = engine.n//2 if k is None else int(k)
    agent = DynamicFunctionAgent(engine, engine.x, engine.ids, coords, seed=seed+500000)
    report = agent.fit(k, episodes=episodes, candidates_per_step=candidates_per_step)
    selected, diagnostics = agent.deploy(k)
    keep = np.zeros(engine.n, dtype=bool)
    keep[selected] = True
    return agent, keep, {**report, **diagnostics}


def oof_predictions(kind, use_agent, raw, ids, target, coords, ecology, seed, config, device, epochs, agent_episodes):
    n = len(raw)
    rng = np.random.default_rng(seed+31337)
    permutation = rng.permutation(n)
    directions = ((permutation[n//2:], permutation[:n//2]), (permutation[:n//2], permutation[n//2:]))
    result = np.empty(n, dtype=np.float64)
    coverage = np.zeros(n, dtype=np.int64)
    for held, retained in directions:
        engine, _ = fit_parent(kind, raw[retained], ids[retained], target[retained], coords[retained], ecology[retained],
                               seed+len(held), config, device, epochs)
        if use_agent:
            _, keep, _ = fit_agent(engine, coords[retained], seed+len(held), episodes=agent_episodes)
        else:
            keep = np.ones(len(retained), dtype=bool)
        prediction = engine.predict(keep, transform(raw[held], engine.pre), ids[held])['prediction']
        result[held] = prediction
        coverage[held] += 1
    assert np.array_equal(coverage, np.ones(n, dtype=np.int64)) and np.isfinite(result).all()
    return result


def run_project_case(data, config, arm, seed, output, device='cpu', epochs=None, agent_episodes=4):
    valid_arms = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
    assert arm in valid_arms
    use_rp = arm in ('A1', 'A4', 'A5', 'A7')
    use_agent = arm in ('A2', 'A4', 'A6', 'A7')
    use_eco = arm in ('A3', 'A5', 'A6', 'A7')
    kind = 'RP' if use_rp else 'G0'
    n = int(data['n_train'])
    raw, ids = data['raw_features'], data['sample_ids'].astype(str)
    target, coords = data['y'].astype(float), data['coords'].astype(float)
    ecology = data['ecology'].astype(str)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selection = data.get('selection')
    selected_epoch = None
    if selection is not None:
        _, selection_history = fit_parent(kind, raw[:n], ids[:n], target[:n], coords[:n], ecology[:n],
                                          seed, config, device, epochs, selection)
        selected_epoch = int(min(selection_history, key=lambda row: row['selection_loss'])['epoch'])
        pd.DataFrame(selection_history).to_csv(output/'epoch_selection.csv', index=False)
        epochs = selected_epoch
    oof = None
    if use_eco:
        oof = oof_predictions(kind, use_agent, raw[:n], ids[:n], target[:n], coords[:n], ecology[:n],
                              seed, config, device, epochs, agent_episodes)
    engine, history = fit_parent(kind, raw[:n], ids[:n], target[:n], coords[:n], ecology[:n], seed, config, device, epochs)
    full = engine.predict(np.ones(n, dtype=bool), transform(raw[n:], engine.pre), ids[n:])['prediction']
    agent, keep, agent_report = None, np.ones(n, dtype=bool), None
    if use_agent:
        agent, keep, agent_report = fit_agent(engine, coords[:n], seed, episodes=agent_episodes)
        agent.save(output/'actor.pt', int(keep.sum()))
    compressed = engine.predict(keep, transform(raw[n:], engine.pre), ids[n:])['prediction']
    correction = np.zeros(len(compressed))
    if use_eco:
        calibrator = EcologicalOOFCalibrator.fit(ecology[:n], target[:n]-oof, oof)
        compressed, correction = calibrator.transform(ecology[n:], compressed)
        categories = list(calibrator.coefficients)
        np.savez_compressed(output/'calibration.npz', categories=np.asarray(categories),
                            coefficients=np.asarray([calibrator.coefficients[value] for value in categories]),
                            center=np.asarray(calibrator.center), scale=np.asarray(calibrator.scale),
                            minimum_support=np.asarray(calibrator.minimum_support))
    metrics = compression_metrics(target[n:], full, compressed, n, int(keep.sum()))
    pd.DataFrame({'sample_id': ids[n:], 'target': target[n:], 'full_prediction': full,
                  'compressed_prediction': compressed-correction, 'ecological_offset': correction,
                  'prediction': compressed}).to_csv(output/'predictions.csv', index=False)
    pd.DataFrame(history).to_csv(output/'training_history.csv', index=False)
    parent_state = engine.engine.full_state if kind == 'RP' else {name: value.detach().cpu().clone() for name, value in engine.model.state_dict().items()}
    torch.save({'kind': kind, 'state_dict': parent_state, 'preprocessor': engine.pre,
                'config': {**config, 'input_dim': raw.shape[1], 'relation_n_train': n, 'seed': seed},
                'adaptation': 'dynamic public native port'}, output/'parent.pt')
    np.savez_compressed(output/'selection.npz', keep=keep, source_ids=ids[:n], retained=np.flatnonzero(keep))
    summary = {'arm': arm, 'kind': kind, 'agent': use_agent, 'ecological_calibration': use_eco,
               'seed': int(seed), 'metrics': metrics, 'agent_report': agent_report,
               'selected_epoch_from_official_dev': selected_epoch,
               'oof_source_labels_only': use_eco, 'outer_labels_used_for_selection': False,
               'final_holdout_access': False, 'adaptation': 'dynamic public native port'}
    (output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def run_external_model(name, data, config, seed, output, device='cpu', epochs=None):
    n = int(data['n_train'])
    raw, ids, target = data['raw_features'], data['sample_ids'].astype(str), data['y'].astype(float)
    pre = fit_preprocessor(raw[:n], target[:n])
    x, q = transform(raw[:n], pre), transform(raw[n:], pre)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected_epoch = None
    if name in ('linear_regression', 'random_forest', 'xgboost'):
        if name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=1)
        else:
            from evaluation.public_models import construct_xgboost
            model = construct_xgboost(seed)
        model.fit(x, target[:n])
        prediction = model.predict(q)
        history = []
    else:
        adjacency = rebuild_reference_graph(ReducedReference(x, ids[:n], data['coords'][:n], data['ecology'][:n], name, n), x[:0])
        edges = torch.as_tensor(np.array(adjacency.nonzero()), dtype=torch.long, device=device)
        x_tensor = torch.as_tensor(x, device=device)
        y_tensor = torch.as_tensor((target[:n]-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device=device)
        selection = data.get('selection')
        if selection is not None:
            selection_x = transform(selection['raw_features'], pre)
            all_x = torch.cat((x_tensor, torch.as_tensor(selection_x, device=device)))
            all_adjacency = rebuild_reference_graph(ReducedReference(x, ids[:n], data['coords'][:n], data['ecology'][:n], name, n), selection_x)
            all_edges = torch.as_tensor(np.array(all_adjacency.nonzero()), dtype=torch.long, device=device)
            selection_target = torch.as_tensor((selection['y']-pre['y_mean'])/pre['y_scale'], dtype=torch.float32, device=device)
        selected_epoch = None
        for training_pass in range(2 if selection is not None else 1):
            model = construct_public_model(name, x.shape[1], seed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            history = []
            for epoch in range(1, int(config.get('max_epochs', 400) if epochs is None else epochs)+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if name == 'semi_arid_transformer':
                    estimate = model(x_tensor.unsqueeze(1)).squeeze(-1).squeeze(-1)
                else:
                    estimate = model(x_tensor, edges)
                loss = torch.nn.functional.mse_loss(estimate, y_tensor)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                record = {'epoch': epoch, 'loss': float(loss.detach())}
                if selection is not None:
                    model.eval()
                    with torch.inference_mode():
                        if name == 'semi_arid_transformer':
                            selection_prediction = model(all_x[n:].unsqueeze(1)).squeeze(-1).squeeze(-1)
                        else:
                            selection_prediction = model(all_x, all_edges)[n:]
                        record['selection_loss'] = float(torch.nn.functional.mse_loss(selection_prediction, selection_target))
                history.append(record)
            if selection is not None:
                selected_epoch = int(min(history, key=lambda row: row['selection_loss'])['epoch'])
                pd.DataFrame(history).to_csv(output/'epoch_selection.csv', index=False)
                epochs = selected_epoch
                selection = None
        model.eval()
        with torch.inference_mode():
            q_tensor = torch.as_tensor(q, device=device)
            if name == 'semi_arid_transformer':
                scaled = model(q_tensor.unsqueeze(1)).squeeze(-1).squeeze(-1)
            else:
                all_x = torch.cat((x_tensor, q_tensor))
                all_adjacency = rebuild_reference_graph(ReducedReference(x, ids[:n], data['coords'][:n], data['ecology'][:n], name, n), q)
                all_edges = torch.as_tensor(np.array(all_adjacency.nonzero()), dtype=torch.long, device=device)
                scaled = model(all_x, all_edges)[n:]
            prediction = scaled.cpu().numpy()*pre['y_scale']+pre['y_mean']
        torch.save(model.state_dict(), output/'model.pt')
    metrics = regression_metrics(target[n:], prediction)
    pd.DataFrame({'sample_id': ids[n:], 'target': target[n:], 'prediction': prediction}).to_csv(output/'predictions.csv', index=False)
    pd.DataFrame(history).to_csv(output/'training_history.csv', index=False)
    summary = {'model': name, 'seed': int(seed), 'metrics': metrics, 'compression': 'not_applicable',
               'selected_epoch_from_official_dev': selected_epoch,
               'outer_labels_used_for_selection': False, 'final_holdout_access': False}
    (output/'case_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
