import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DynamicAgentPolicy(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class DynamicFunctionAgent:
    adaptation = 'native_dynamic_port_of_real_function_feedback_agent'

    def __init__(self, engine, reference_x, reference_ids, coords, seed=0, projection=None):
        self.engine = engine
        self.x = np.asarray(reference_x, dtype=np.float64)
        self.ids = np.asarray(reference_ids).astype(str)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n = len(self.ids)
        self.seed = int(seed)
        assert self.x.shape == (self.n, engine.x.shape[1]) and self.coords.shape == (self.n, 2)
        rng = np.random.default_rng(seed+7171)
        self.projection = rng.normal(size=(self.x.shape[1], 8))/np.sqrt(self.x.shape[1]) if projection is None else projection
        self.policy = DynamicAgentPolicy()
        self.history = []
        self.feedback_count = 0
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.teacher = self._predict(np.ones(self.n, dtype=bool))
        variance = float(np.var(self.teacher))
        assert variance > 1e-12
        self.teacher_variance = variance
        self.parent_before = self.engine.parent_state() if hasattr(self.engine, 'parent_state') else None

    def _predict(self, keep):
        return np.asarray(self.engine.predict(keep, self.x, self.ids)['prediction'], dtype=np.float64)

    def _error(self, keep):
        prediction = self._predict(keep)
        self.feedback_count += 1
        return float(np.mean((prediction-self.teacher)**2)/self.teacher_variance)

    def features(self, keep):
        projected = self.x@self.projection
        coord_scale = np.where(self.coords.std(0) > 0, self.coords.std(0), 1)
        coordinates = (self.coords-self.coords.mean(0))/coord_scale
        current = self._predict(keep)
        self.feedback_count += 1
        prediction = (current-current.mean())/max(current.std(), 1e-12)
        ratio = np.full(self.n, keep.mean())
        return np.column_stack((projected, coordinates, prediction, ratio)).astype(np.float32)

    def fit(self, k, episodes=4, candidates_per_step=4):
        assert 0 < k < self.n and episodes >= 1 and candidates_per_step >= 1
        rng = np.random.default_rng(self.seed+991)
        for episode in range(episodes):
            keep = np.ones(self.n, dtype=bool)
            previous = self._error(keep)
            while keep.sum() > k:
                features = self.features(keep)
                scores = self.policy(torch.as_tensor(features)).detach().numpy()
                scores[~keep] = -np.inf
                ranked = np.argsort(-scores)
                exploratory = rng.choice(np.flatnonzero(keep), min(candidates_per_step, int(keep.sum())), replace=False)
                candidates = np.unique(np.concatenate((ranked[:candidates_per_step], exploratory)))
                targets, candidate_features = [], []
                for candidate in candidates:
                    updated = keep.copy()
                    updated[candidate] = False
                    error = self._error(updated)
                    targets.append(previous-error)
                    candidate_features.append(features[candidate])
                target = np.asarray(targets, dtype=np.float32)
                center, scale = float(target.mean()), max(float(target.std()), 1e-6)
                normalized = (target-center)/scale
                for _ in range(2):
                    self.optimizer.zero_grad(set_to_none=True)
                    estimate = self.policy(torch.as_tensor(np.asarray(candidate_features, dtype=np.float32)))
                    loss = nn.functional.mse_loss(estimate, torch.as_tensor(normalized))
                    assert torch.isfinite(loss)
                    loss.backward()
                    self.optimizer.step()
                best = int(candidates[int(np.argmax(target))])
                keep[best] = False
                previous = self._error(keep)
                self.history.append({'episode': episode, 'retained': int(keep.sum()), 'action': best,
                                     'observed_reward': float(target.max()), 'loss': float(loss.detach()),
                                     'candidates': len(candidates)})
        assert self.history and self.feedback_count > 0
        if self.parent_before is not None:
            assert self.engine.parent_state() == self.parent_before
        return {'episodes': episodes, 'updates': 2*len(self.history), 'feedback_count': self.feedback_count,
                'parent_unchanged': True}

    def deploy(self, k):
        assert 0 < k < self.n
        keep = np.ones(self.n, dtype=bool)
        schedule = []
        with torch.inference_mode():
            while keep.sum() > k:
                features = self.features(keep)
                scores = self.policy(torch.as_tensor(features)).numpy()
                scores[~keep] = -np.inf
                action = int(np.argmax(scores))
                keep[action] = False
                schedule.append(1)
        selected = np.flatnonzero(keep)
        return selected, {'seed': self.seed, 'deletion_schedule': schedule, 'deployment_feedback': 0,
                          'training_feedback': self.feedback_count, 'adaptation': self.adaptation,
                          'real_T_nodes_only': True}

    def save(self, path, k):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'schema': 'dynamic_agent_1', 'seed': self.seed, 'projection': self.projection,
                    'state_dict': self.policy.state_dict(), 'history': self.history,
                    'feedback_count': self.feedback_count, 'k': int(k), 'adaptation': self.adaptation}, path)
        (path.parent/'learning.json').write_text(json.dumps({'history': self.history,
            'feedback_count': self.feedback_count, 'adaptation': self.adaptation}, indent=2), encoding='utf-8')

    @classmethod
    def restore(cls, path, engine, reference_x, reference_ids, coords):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['schema'] == 'dynamic_agent_1'
        agent = cls(engine, reference_x, reference_ids, coords, seed=saved['seed'], projection=saved['projection'])
        agent.policy.load_state_dict(saved['state_dict'])
        agent.policy.eval()
        agent.history = saved['history']
        agent.feedback_count = int(saved['feedback_count'])
        return agent
