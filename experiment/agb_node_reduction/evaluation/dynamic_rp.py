from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from relation_layer import RDoG


class DynamicPrecisionSAGE(nn.Module):
    adaptation = 'native_dynamic_feature_port_of_RP_positive_distance'

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.k = int(config.get('knn_k', 8))
        input_dim = int(config['input_dim'])
        self.input_dim = input_dim
        hidden = config.get('hidden_dims', [32, 32])
        self.first = SAGEConv(input_dim, hidden[0], aggr='mean')
        self.second = SAGEConv(hidden[0], hidden[1], aggr='mean')
        self.backbone = nn.Module()
        self.backbone.first, self.backbone.second = self.first, self.second
        self.backbone.dropout = nn.Dropout(config.get('dropout', .1))
        self.backbone.head = nn.Linear(hidden[1], 1)
        n = int(config['relation_n_train'])
        self.register_buffer('center', torch.zeros(input_dim, dtype=torch.float64))
        self.register_buffer('group_scale', torch.ones(input_dim))
        self.register_buffer('input_mean', torch.zeros(input_dim, dtype=torch.float64))
        self.register_buffer('input_scale', torch.ones(input_dim, dtype=torch.float64))
        self.register_buffer('field_support', torch.zeros(n, input_dim, dtype=torch.float64))
        self.register_buffer('field_directions', torch.zeros(n, input_dim, dtype=torch.float64))
        self.register_buffer('field_log_norm', torch.zeros(n, dtype=torch.float64))
        self.register_buffer('label_bins', torch.zeros(n, dtype=torch.long))
        self.register_buffer('label_values', torch.zeros(10, dtype=torch.float64))
        self.register_buffer('label_edges', torch.zeros(9, dtype=torch.float32))
        self.layer_log_metric = nn.Parameter(torch.zeros(2, input_dim, dtype=torch.float64))
        self.geometry_history = []
        self.geometry_feature = None
        self._optimizer = None

    def coordinates(self, x):
        return (x/self.group_scale-self.center).double()

    def filtered(self, x):
        return x

    @torch.no_grad()
    def fit_reference(self, x, edges=None):
        assert len(x) == len(self.field_support)
        self.center.copy_((x/self.group_scale).mean(0))
        coordinates = self.coordinates(x)
        self.field_support.copy_(coordinates)
        norms = coordinates.norm(dim=1, keepdim=True)
        fallback = torch.full_like(coordinates, 1/np.sqrt(self.input_dim))
        self.field_directions.copy_(torch.where(norms > 1e-12, coordinates/norms.clamp_min(1e-12), fallback))
        values = self.filtered(x)
        self.input_mean.copy_(values.mean(0))
        scale = values.std(0, correction=0)
        self.input_scale.copy_(torch.where(scale < 1e-6, torch.ones_like(scale), scale))

    @torch.no_grad()
    def fit_relations(self, x, target, edges=None):
        assert len(x) == len(target) == len(self.field_support)
        quantiles = torch.linspace(.1, .9, 9, device=target.device)
        self.label_edges.copy_(torch.quantile(target, quantiles))
        bins = torch.bucketize(target.contiguous(), self.label_edges)
        counts = torch.bincount(bins, minlength=10)
        assert (counts > 0).all(), 'RP label bins require at least ten supported training groups'
        self.label_bins.copy_(bins)
        self.label_values.copy_(torch.stack([target[bins == index].double().mean() for index in range(10)]))
        return [{'dynamic_port': True, 'field_support_fit': 'training coordinates',
                 'field_directions': 'normalized per-node coordinate initialization', 'label_bin_counts': counts.cpu().tolist()}]

    def endpoint_weights(self, u, log_metric):
        root = (.5*log_metric).exp()
        origin = self.field_support.mean(0)
        query, support = (u-origin)*root, (self.field_support-origin)*root
        reference = torch.cdist(support, support, compute_mode='donot_use_mm_for_euclid_dist').square()
        mask = torch.eye(len(support), device=u.device, dtype=torch.bool)
        tau = reference.masked_fill(mask, torch.inf).topk(min(8, len(support)-1), largest=False).values.mean().clamp_min(1e-8)
        distance = torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist').square()
        return torch.softmax(-distance/tau+self.field_log_norm, 1), torch.softmax(-reference/tau+self.field_log_norm, 1)

    def stable_distance(self, u, theta):
        w, wt = self.endpoint_weights(u, theta)
        root = (.5*theta).exp()
        origin = self.field_support.mean(0)
        query, support = (u-origin)*root, (self.field_support-origin)*root
        a, b = query@self.field_directions.T, support@self.field_directions.T
        positive_left = (w*a.square()).sum(1)[:, None]+w@b.square().T
        positive_right = a.square()@wt.T+(wt*b.square()).sum(1)[None, :]
        left = positive_left-2*(w*a)@b.T
        right = positive_right-2*a@(wt*b).T
        base = torch.cdist(query, support, compute_mode='donot_use_mm_for_euclid_dist').square()
        d = self.input_dim
        return ((d*(left+right)*.5+.01*base)/1.01).clamp_min(0)

    def layer_graphs(self, x, n_train):
        assert n_train == len(self.field_support)
        u = self.coordinates(x)
        graphs = []
        for layer_metric in self.layer_log_metric:
            distance = self.stable_distance(u, layer_metric)
            mask = torch.arange(len(x), device=x.device)[:, None] == torch.arange(n_train, device=x.device)
            distance = distance.masked_fill(mask, torch.inf)
            near = torch.argsort(distance, dim=1, stable=True)[:, :min(self.k, n_train)]
            values = distance.gather(1, near)
            positive = values[torch.isfinite(values)]
            tau = positive.topk(min(8, positive.numel()), largest=False).values.mean().clamp_min(1e-8)
            weights = torch.softmax(-values/tau, dim=1).float()
            graphs.append((near, weights))
        return graphs

    @staticmethod
    def weighted_layer(layer, x, neighbors, weights):
        aggregate = (x[neighbors]*weights[:, :, None]).sum(1)
        return layer.lin_l(aggregate)+layer.lin_r(x)

    def forward(self, x, n_train, edges=None):
        first, second = self.layer_graphs(x, n_train)
        features = ((self.filtered(x)-self.input_mean)/self.input_scale).float()
        hidden = self.backbone.dropout(torch.relu(self.weighted_layer(self.backbone.first, features, *first)))
        raw = self.weighted_layer(self.backbone.second, hidden, *second)
        self.geometry_feature = raw
        embedding = torch.relu(raw)
        return self.backbone.head(embedding).squeeze(-1), embedding, *first

    def training_loss(self, x, target, edges=None):
        prediction = self(x, len(target), edges)[0]
        mse = F.mse_loss(prediction, target)
        normalized = F.normalize(self.geometry_feature, dim=1)
        centers = F.normalize(torch.stack([normalized[self.label_bins == index].mean(0) for index in range(10)]), dim=1)
        contrast = F.cross_entropy(normalized@centers.detach().T/.1, self.label_bins)
        spacing = self.label_values[1:]-self.label_values[:-1]
        homogeneous = ((centers[1:]-centers[:-1]).square().sum(1)/spacing).sum()
        sphere = F.normalize(torch.randn(1000, centers.shape[1], generator=torch.Generator(device='cpu').manual_seed(self.config['seed']+87000)).to(x.device), dim=1)
        envelope = 1-(centers@sphere.T).max(dim=0).values.mean()
        warmup = not self.geometry_history
        loss = mse if warmup else mse+contrast+.01*(envelope+homogeneous)
        self.geometry_history.append({'step': len(self.geometry_history)+1, 'mse': float(mse.detach()),
                                      'contrast': float(contrast.detach()), 'first_update_MSE_only': warmup})
        return loss

    def make_optimizer(self, config):
        model = self

        class Optimizer:
            def __init__(self):
                self.adam = torch.optim.Adam(model.backbone.parameters(), lr=config['learning_rate'],
                                             weight_decay=config['weight_decay'])
                self.geometry = RDoG(model.layer_log_metric)

            def zero_grad(self, set_to_none=True):
                self.adam.zero_grad(set_to_none=set_to_none)
                model.layer_log_metric.grad = None

            def step(self):
                self.adam.step()
                self.geometry.step()

        self._optimizer = Optimizer()
        return self._optimizer


class DynamicReferencePredictor:
    def __init__(self, model, reference_x, reference_ids, preprocessor, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device
        self.x = torch.as_tensor(reference_x, device=device)
        self.ids = np.asarray(reference_ids).astype(str)
        self.n = len(self.ids)
        self.pre = {key: np.asarray(value).copy() for key, value in preprocessor.items()}
        self.fields = {name: getattr(model, name).detach().clone() for name in
                       ('field_support', 'field_directions', 'field_log_norm')}
        self.full_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    @torch.inference_mode()
    def predict(self, keep, query_x, query_ids):
        keep = np.asarray(keep, dtype=bool)
        assert keep.shape == (self.n,) and keep.any()
        indices = torch.as_tensor(np.flatnonzero(keep), device=self.device)
        for name, full in self.fields.items():
            if full.shape[0] == self.n:
                setattr(self.model, name, full.index_select(0, indices))
        query_x = torch.as_tensor(query_x, device=self.device)
        query_ids = np.asarray(query_ids).astype(str)
        kept_ids = self.ids[keep]
        positions = {value: index for index, value in enumerate(kept_ids)}
        extras, gather = [], []
        for index, value in enumerate(query_ids):
            if value in positions:
                gather.append(positions[value])
            else:
                gather.append(len(indices)+len(extras))
                extras.append(index)
        x = torch.cat((self.x.index_select(0, indices), query_x[extras]))
        prediction = self.model(x, len(indices), None)[0].index_select(0, torch.as_tensor(gather, device=self.device, dtype=torch.long))
        raw = prediction.cpu().numpy().astype(float)*self.pre['y_scale']+self.pre['y_mean']
        return {'prediction': raw, 'query_count': len(query_ids), 'reference_count': len(indices)}

    def export(self, keep, path):
        self.predict(keep, self.x[:0], np.asarray([], dtype=str))
        state = {name: value.detach().cpu() for name, value in self.model.state_dict().items()}
        state['label_bins'] = self.full_state['label_bins'][np.asarray(keep)]
        config = {**self.model.config, 'relation_n_train': int(np.sum(keep))}
        bundle = {'schema': 'dynamic_rp_1', 'state_dict': state, 'config': config,
                  'reference_x': self.x[torch.as_tensor(keep, device=self.device)].cpu().numpy(),
                  'reference_ids': self.ids[np.asarray(keep)], 'preprocessor': self.pre,
                  'query_labels_required': False, 'adaptation': self.model.adaptation}
        torch.save(bundle, path)
        return bundle

    @classmethod
    def restore(cls, path, device='cpu'):
        bundle = torch.load(path, map_location=device, weights_only=False)
        assert bundle['schema'] == 'dynamic_rp_1'
        model = DynamicPrecisionSAGE(bundle['config'])
        model.load_state_dict(bundle['state_dict'])
        return cls(model, bundle['reference_x'], bundle['reference_ids'], bundle['preprocessor'], device)
