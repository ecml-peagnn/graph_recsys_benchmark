import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import JumpingKnowledge

from .base import GraphRecsysModel


class PEAGCNChannel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PEAGCNChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        if kwargs['jump_mode'] == 'lstm':
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'], channels=kwargs['jump_channels'], num_layers=kwargs['jump_num_layers'])
        else:
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'])

        self.gcn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] >= 2:
            self.gcn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['hidden_size']))
            for i in range(kwargs['num_steps'] - 1):
                self.gcn_layers.append(GCNConv(kwargs['hidden_size'], kwargs['hidden_size']))
        else:
            self.gcn_layers.append(GCNConv(kwargs['emb_dim'], kwargs['hidden_size']))

        if kwargs['jump_mode'] == 'cat':
            self.lin = Linear(kwargs['num_steps'] * kwargs['hidden_size'], kwargs['repr_dim'])
        else:
            self.lin = Linear(kwargs['hidden_size'], kwargs['repr_dim'])

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gcn_layers:
            module.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for step_idx in range(self.num_steps):
            x = F.relu(self.gcn_layers[step_idx](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return self.lin(self.jump(xs))


class PEAGCNRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEAGCNRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.channel_aggr = kwargs['channel_aggr']

        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')
        self.meta_path_edge_indices = self.update_graph_input(kwargs['dataset'])

        self.peagcn_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.peagcn_channels.append(PEAGCNChannel(**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = torch.nn.Linear(kwargs['repr_dim'], 1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')

    def reset_parameters(self):
        glorot(self.x)
        for module in self.peagcn_channels:
            module.reset_parameters()
        if self.channel_aggr == 'att':
            glorot(self.att.weight)

    def forward(self):
        x = self.x
        x = [module(x, self.meta_path_edge_indices[idx]).unsqueeze(-2) for idx, module in enumerate(self.peagcn_channels)]
        x = torch.cat(x, dim=-2)
        x = F.normalize(x, dim=-2)
        if self.channel_aggr == 'cat':
            x = x.view(x.shape[0], -1)
            x = F.normalize(x)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=-2)
        elif self.channel_aggr == 'att':
            atts = F.softmax(self.att(x).squeeze(-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=-2)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1)

