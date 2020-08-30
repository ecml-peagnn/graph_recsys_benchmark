import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

from .base import MFRecsysModel


class MCFKGRecsysModel(MFRecsysModel):
    def __init__(self, **kwargs):
        super(MCFKGRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.dropout = kwargs['dropout']

        self.x = torch.nn.Embedding(kwargs['dataset']['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        self.r = torch.nn.Embedding(kwargs['dataset'].num_edge_types, kwargs['emb_dim'], max_norm=1).weight

        self.fc1 = torch.nn.Linear(kwargs['emb_dim'] * 2, kwargs['hidden_size'])
        self.fc2 = torch.nn.Linear(kwargs['hidden_size'], 1)

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        glorot(self.r)

    def forward(self, unids, inids):
        u_repr = self.x[unids].detach()
        i_repr = self.x[inids].detach()
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
