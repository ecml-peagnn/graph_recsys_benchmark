import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros

from .base import GraphRecsysModel


class GCNRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(GCNRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_index = self.update_graph_input(kwargs['dataset'])

        self.conv1 = GCNConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = GCNConv(kwargs['hidden_size'], kwargs['hidden_size'] // 2)

        self.fc1 = torch.nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        self.fc2 = torch.nn.Linear(kwargs['hidden_size'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GCNInnerRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(GCNInnerRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_type = kwargs['entity_aware_type']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']

        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')
        self.edge_index = self.update_graph_input(kwargs['dataset'])

        self.conv1 = GCNConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = GCNConv(kwargs['hidden_size'], kwargs['hidden_size'] // 2)

        self.bias = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        zeros(self.bias)

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1) + self.bias