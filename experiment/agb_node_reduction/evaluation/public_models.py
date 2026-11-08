import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 32), dropout=.1):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.Linear(hidden_dims[1], 1))

    def forward(self, x, edge_index=None):
        return self.network(x).squeeze(-1)


class GraphRegressor(nn.Module):
    def __init__(self, kind, input_dim, hidden_dims=(32, 32), dropout=.1):
        super().__init__()
        self.kind = kind
        convolution = {'gcn': GCNConv, 'gat': GATConv, 'graphsage': SAGEConv}[kind]
        self.first = convolution(input_dim, hidden_dims[0])
        self.second = convolution(hidden_dims[0], hidden_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dims[1], 1)

    def forward(self, x, edge_index):
        hidden = self.dropout(torch.relu(self.first(x, edge_index)))
        embedding = torch.relu(self.second(hidden, edge_index))
        return self.head(embedding).squeeze(-1)


def load_semi_arid_transformer(input_dim=539):
    root = Path(__file__).resolve().parents[3]
    path = root/'01_数据集/public/semi_arid/author_code/code/model/Transformer.py'
    spec = importlib.util.spec_from_file_location('semi_arid_public_transformer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TabularRegressionTransformer(input_dim=input_dim, output_dim=1, d_model=72, nhead=8,
                                                dim_feedforward=76, num_layers=2)


def construct_public_model(name, input_dim, seed=0, hidden_dims=(32, 32), dropout=.1):
    torch.manual_seed(seed)
    if name == 'mlp':
        return MLPRegressor(input_dim, hidden_dims, dropout)
    if name in ('gcn', 'gat', 'graphsage'):
        return GraphRegressor(name, input_dim, hidden_dims, dropout)
    if name == 'semi_arid_transformer':
        assert input_dim > 0
        return load_semi_arid_transformer(input_dim)
    raise KeyError(name)


def construct_xgboost(seed=0):
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError('XGBoost is declared in the evaluation environment but is not installed locally') from error
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05, subsample=.8,
                        colsample_bytree=.8, objective='reg:squarederror', random_state=seed, n_jobs=1)
