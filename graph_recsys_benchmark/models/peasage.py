import torch
from torch.nn import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import SAGEConv

from .base import PEABaseChannel, PEABaseRecsysModel, PEAJKBaseChannel, PEAJKBaseRecsysModel


class PEASageChannel(PEABaseChannel):
    def __init__(self, **kwargs):
        super(PEASageChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        self.gnn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] == 1:
            self.gnn_layers.append(SAGEConv(kwargs['emb_dim'], kwargs['repr_dim']))
        else:
            self.gnn_layers.append(SAGEConv(kwargs['emb_dim'], kwargs['hidden_size']))
            for i in range(kwargs['num_steps'] - 2):
                self.gnn_layers.append(SAGEConv(kwargs['hidden_size'], kwargs['hidden_size']))
            self.gnn_layers.append(SAGEConv(kwargs['hidden_size'], kwargs['repr_dim']))

        self.reset_parameters()


class PEASageRecsysModel(PEABaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEASageChannel
        super(PEASageRecsysModel, self).__init__(**kwargs)


class PEASageJKChannel(PEAJKBaseChannel):
    def __init__(self, **kwargs):
        super(PEASageJKChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        if kwargs['jump_mode'] == 'lstm':
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'], channels=kwargs['jump_channels'], num_layers=kwargs['jump_num_layers'])
        else:
            self.jump = JumpingKnowledge(mode=kwargs['jump_mode'])

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(SAGEConv(kwargs['emb_dim'], kwargs['hidden_size']))
        if kwargs['num_steps'] > 1:
            for i in range(kwargs['num_steps'] - 1):
                self.gnn_layers.append(SAGEConv(kwargs['hidden_size'], kwargs['hidden_size']))

        if kwargs['jump_mode'] == 'cat':
            self.lin = Linear(kwargs['num_steps'] * kwargs['hidden_size'], kwargs['hidden_size'])
        else:
            self.lin = Linear(kwargs['hidden_size'], kwargs['hidden_size'])

        self.reset_parameters()


class PEASageJKRecsysModel(PEAJKBaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEASageJKChannel
        super(PEASageJKRecsysModel, self).__init__(**kwargs)
