import types

import numpy as np
import torch
from torch.func import functional_call, vmap


def _layer_graphs(self, x, n_train):
    u = self.coordinates(x)
    graphs = []
    for layer_metric in self.layer_log_metric:
        distance = self.stable_distance(u, layer_metric)
        mask = (torch.arange(len(x), device=x.device)[:, None]
                == torch.arange(n_train, device=x.device))
        distance = distance.masked_fill(mask, torch.inf)
        near = torch.argsort(distance, dim=1, stable=True)[:, :min(self.k, n_train)]
        values = distance.gather(1, near)
        finite = torch.where(torch.isfinite(values), values, torch.inf).reshape(-1)
        tau = finite.topk(min(8, int(finite.numel())), largest=False).values.mean().clamp_min(1e-8)
        weights = torch.softmax(-values/tau, dim=1).float()
        graphs.append((near, weights))
    return graphs


def _batched_rp_predictions(agent, candidates, keep):
    engine = agent.engine.engine
    model = engine.model
    model.layer_graphs = types.MethodType(_layer_graphs, model)
    if not hasattr(agent, '_speedup_base_state'):
        agent._speedup_base_state = {name: value.to(engine.device)
                                     for name, value in engine.full_state.items()}
    supports, directions, log_norms, inputs, gathers = [], [], [], [], []
    for candidate in candidates:
        updated = keep.copy()
        updated[candidate] = False
        indices = torch.as_tensor(np.flatnonzero(updated), device=engine.device)
        supports.append(engine.fields['field_support'].index_select(0, indices))
        directions.append(engine.fields['field_directions'].index_select(0, indices))
        log_norms.append(engine.fields['field_log_norm'].index_select(0, indices))
        retained = np.flatnonzero(updated)
        deleted = np.flatnonzero(~updated)
        inputs.append(torch.cat((engine.x.index_select(0, indices), engine.x[torch.as_tensor(deleted, device=engine.device)])))
        gather = np.empty(agent.n, dtype=np.int64)
        gather[retained] = np.arange(len(retained))
        gather[deleted] = np.arange(len(retained), agent.n)
        gathers.append(torch.as_tensor(gather, device=engine.device))
    supports, directions, log_norms, inputs, gathers = [
        torch.stack(values) for values in (supports, directions, log_norms, inputs, gathers)]
    base = agent._speedup_base_state
    reference_count = int(keep.sum()-1)

    def predict_one(support, direction, log_norm, x, gather):
        state = {**base, 'field_support': support, 'field_directions': direction,
                 'field_log_norm': log_norm}
        return functional_call(model, state, (x, reference_count, None), tie_weights=False)[0].index_select(0, gather)

    with torch.inference_mode():
        prediction = vmap(predict_one, (0, 0, 0, 0, 0))(
            supports, directions, log_norms, inputs, gathers).cpu().numpy()
    return prediction*engine.pre['y_scale']+engine.pre['y_mean']


def _candidate_predictions(agent, candidates, keep, batch_rp):
    if batch_rp and agent.engine.kind == 'RP' and agent.engine.device == 'cuda':
        return _batched_rp_predictions(agent, candidates, keep)
    predictions = []
    for candidate in candidates:
        updated = keep.copy()
        updated[candidate] = False
        predictions.append(agent._predict(updated))
    return np.asarray(predictions)


def fit_with_prediction_cache(agent, k, episodes=4, candidates_per_step=4, batch_rp=True):
    assert 0 < k < agent.n and episodes >= 1 and candidates_per_step >= 1
    rng = np.random.default_rng(agent.seed+991)
    projected = agent.x@agent.projection
    coord_scale = np.where(agent.coords.std(0) > 0, agent.coords.std(0), 1)
    coordinates = (agent.coords-agent.coords.mean(0))/coord_scale
    for episode in range(episodes):
        keep = np.ones(agent.n, dtype=bool)
        current = agent.teacher.copy()
        previous = float(np.mean((current-agent.teacher)**2)/agent.teacher_variance)
        agent.feedback_count += 1
        while keep.sum() > k:
            agent.feedback_count += 1
            prediction = (current-current.mean())/max(current.std(), 1e-12)
            ratio = np.full(agent.n, keep.mean())
            features = np.column_stack((projected, coordinates, prediction, ratio)).astype(np.float32)
            scores = agent.policy(torch.as_tensor(features)).detach().numpy()
            scores[~keep] = -np.inf
            ranked = np.argsort(-scores)
            exploratory = rng.choice(np.flatnonzero(keep), min(candidates_per_step, int(keep.sum())), replace=False)
            candidates = np.unique(np.concatenate((ranked[:candidates_per_step], exploratory)))
            candidate_predictions = _candidate_predictions(agent, candidates, keep, batch_rp)
            agent.feedback_count += len(candidates)
            errors = np.mean((candidate_predictions-agent.teacher[None, :])**2, axis=1)/agent.teacher_variance
            targets = previous-errors
            candidate_features = [features[candidate] for candidate in candidates]
            target = np.asarray(targets, dtype=np.float32)
            center, scale = float(target.mean()), max(float(target.std()), 1e-6)
            normalized = (target-center)/scale
            for _ in range(2):
                agent.optimizer.zero_grad(set_to_none=True)
                estimate = agent.policy(torch.as_tensor(np.asarray(candidate_features, dtype=np.float32)))
                loss = torch.nn.functional.mse_loss(estimate, torch.as_tensor(normalized))
                assert torch.isfinite(loss)
                loss.backward()
                agent.optimizer.step()
            best_position = int(np.argmax(target))
            best = int(candidates[best_position])
            keep[best] = False
            current = candidate_predictions[best_position]
            previous = float(errors[best_position])
            agent.feedback_count += 1
            agent.history.append({'episode': episode, 'retained': int(keep.sum()), 'action': best,
                                  'observed_reward': float(target.max()), 'loss': float(loss.detach()),
                                  'candidates': len(candidates)})
    assert agent.history and agent.feedback_count > 0
    if agent.parent_before is not None:
        assert agent.engine.parent_state() == agent.parent_before
    return {'episodes': episodes, 'updates': 2*len(agent.history), 'feedback_count': agent.feedback_count,
            'parent_unchanged': True}
