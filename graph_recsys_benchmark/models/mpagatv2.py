import inspect

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

from .base import GraphRecsysModel


class MPARelationProp(MessagePassing):
    def __init__(self, edge_index, feat_dim, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super_args = {k: v for k, v in kwargs.items() if
                                  k in inspect.signature(super(MPARelationProp, self).__init__).parameters}
        if not 'aggr' in super_args.keys():
            super_args['aggr'] = 'add'
        super(MPARelationProp, self).__init__(**super_args)

        self.edge_index = edge_index
        self.feat_dim = feat_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(feat_dim, heads * feat_dim))
        self.att = Parameter(torch.Tensor(1, heads, 2 * feat_dim))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * feat_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(feat_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(self.edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(self.edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.feat_dim)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.feat_dim:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.feat_dim)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.feat_dim)
        else:
            aggr_out = aggr_out.view(-1, self.heads, self.feat_dim).mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MPAGATRecsysModelV2(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(MPAGATRecsysModelV2, self).__init__(**kwargs)

    def _init(self, **kwargs):
        ##################### parse the model input #####################
        self.channel_aggr = kwargs['channel_aggr']

        if kwargs['activation'] == 'relu':
            self.act_func = torch.nn.ReLU()
        else:
            raise NotImplementedError('{} activation not implemented!'.format(kwargs['activaion']))

        ##################### update the metapath #####################
        self.edge_index_dict, self.metapaths = self.update_graph_input(kwargs['dataset'])

        ##################### Create trainable parameters #####################
        # create relation propagation
        self.mpa_relation_props = torch.nn.ModuleDict()
        for relationship, edge_index in self.edge_index_dict.items():
            self.mpa_relation_props[str(relationship) + '_1'] = MPARelationProp(
                edge_index=edge_index,
                feat_dim=kwargs['emb_dim'],
                **kwargs
            )
        for relationship, edge_index in self.edge_index_dict.items():
            self.mpa_relation_props[str(relationship) + '_2'] = MPARelationProp(
                edge_index=edge_index,
                feat_dim=kwargs['hidden_size'],
                **kwargs
            )

        # Create transform layer between metapath propagation layer
        self.transform1 = torch.nn.Linear(kwargs['emb_dim'], kwargs['hidden_size'])
        self.transform2 = torch.nn.Linear(kwargs['hidden_size'], kwargs['repr_dim'])

        # Create feature in case of no given node features
        if not kwargs['if_use_features']:
            self.x = torch.nn.Embedding(kwargs['num_nodes'], kwargs['emb_dim'], max_norm=1).weight
        else:
            raise NotImplementedError('Feature not implemented!')

        # Create the misc layers
        if self.channel_aggr == 'concat':
            self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        elif self.channel_aggr == 'mean':
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        elif self.channel_aggr == 'att':
            self.att = torch.nn.Linear(kwargs['repr_dim'], 1)
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

        ##################### Create computational graph #####################
        self.metapath_channels = []
        for metapath in self.metapaths:
            layer_1_props = [self.mpa_relation_props[str(relationship) + '_1'] for relationship in metapath]
            layer_2_props = [self.mpa_relation_props[str(relationship) + '_2'] for relationship in metapath]
            self.metapath_channels.append([layer_1_props, layer_2_props])
        self.transforms = [self.transform1, self.transform2]

    def reset_parameters(self):
        for module in list(self.mpa_relation_props.values()):
            module.reset_parameters()
        glorot(self.transform1.weight)
        glorot(self.transform2.weight)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att.weight)

    def channel_forward(self, channel, x):
        for layer, transform in zip(channel, self.transforms):
            for relation_prop in layer:
                x = self.act_func(relation_prop(x))
            x = self.act_func(transform(x))
        return x

    def forward(self):
        x = [self.channel_forward(channel, self.x).unsqueeze(-2) for channel in self.metapath_channels]
        x = torch.cat(x, dim=-2)
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=-2)
        elif self.channel_aggr == 'att':
            atts = F.softmax(self.att(x).squeeze(-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=-2)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        x = F.normalize(x)
        return x
