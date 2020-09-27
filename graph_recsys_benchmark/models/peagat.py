import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn.models import JumpingKnowledge

from .base import PEABaseChannel, PEABaseRecsysModel, PEAJKBaseChannel, PEAJKBaseRecsysModel


class PEAGATChannel(PEABaseChannel):
    def __init__(self, **kwargs):
        super(PEAGATChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        self.gnn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] == 1:
            self.gnn_layers.append(GATConv(kwargs['emb_dim'], kwargs['repr_dim'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
        else:
            self.gnn_layers.append(GATConv(kwargs['emb_dim'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            for i in range(kwargs['num_steps'] - 2):
                self.gnn_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            self.gnn_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))

        self.reset_parameters()


class PEAGATRecsysModel(PEABaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEAGATChannel
        super(PEAGATRecsysModel, self).__init__(**kwargs)


class PEAGATJKChannel(PEAJKBaseChannel):
    def __init__(self, **kwargs):
        super(PEAGATJKChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        if kwargs['jump_mode'] == 'lstm':
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'], channels=kwargs['jump_channels'], num_layers=kwargs['jump_num_layers'])
        else:
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'])

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(GATConv(kwargs['emb_dim'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
        if kwargs['num_steps'] > 1:
            for i in range(kwargs['num_steps'] - 1):
                self.gnn_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))

        if kwargs['jump_mode'] == 'cat':
            self.lin = Linear(kwargs['num_steps'] * kwargs['hidden_size'], kwargs['hidden_size'])
        else:
            self.lin = Linear(kwargs['hidden_size'], kwargs['hidden_size'])

        self.reset_parameters()


class PEAGATJKRecsysModel(PEAJKBaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEAGATJKChannel
        super(PEAGATJKRecsysModel, self).__init__(**kwargs)