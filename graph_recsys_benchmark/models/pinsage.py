import torch
import torch.nn.functional as F
from graph_recsys_benchmark.nn import PinSAGEConv
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add


from .base import GraphRecsysModel


class SAGERecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(SAGERecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.if_use_features = kwargs['if_use_features']
        self.dropout = kwargs['dropout']
        self.margin = kwargs['margin']

        if not self.if_use_features:
            self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')
        self.x, self.edge_index = self.update_graph_input(kwargs['dataset'])

        self.conv1 = PinSAGEConv(kwargs['emb_dim'], kwargs['hidden_size'])
        self.conv2 = PinSAGEConv(kwargs['hidden_size'], kwargs['hidden_size'] // 2)

        edge_weight = torch.ones((self.edge_index.size(1),), device=self.edge_index.device)
        deg = scatter_add(edge_weight, self.edge_index[1], dim=0, dim_size=kwargs['dataset'].num_nodes)
        sum_deg = []
        for i in range(kwargs['dataset'].num_nodes):
            sum_deg.append(torch.sum(deg[self.edge_index[0, self.edge_index[1, :] == i]]).view(1))
        sum_deg = torch.cat(sum_deg)
        self.edge_weight = deg[self.edge_index[0, :].long()] / sum_deg[self.edge_index[1, :].long()]

        self.fc1 = torch.nn.Linear(kwargs['hidden_size'], kwargs['hidden_size'])
        self.fc2 = torch.nn.Linear(kwargs['hidden_size'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self):
        x, edge_index = self.x, self.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)
