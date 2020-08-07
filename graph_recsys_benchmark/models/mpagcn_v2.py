import inspect
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from .base import GraphRecsysModel


class MPARelationProp(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, edge_index, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super_args = {k: v for k, v in kwargs.items() if
                      k in inspect.signature(super(MPARelationProp, self).__init__).parameters}
        if not 'aggr' in super_args.keys():
            super_args['aggr'] = 'add'
        super(MPARelationProp, self).__init__(**super_args)

        self.edge_index = edge_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if self.edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, self.edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = self.edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(self.edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MPAGCNRecsysModelV2(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(MPAGCNRecsysModelV2, self).__init__(**kwargs)

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
                in_channels=kwargs['emb_dim'],
                out_channels=kwargs['emb_dim'],
                **kwargs
            )
        for relationship, edge_index in self.edge_index_dict.items():
            self.mpa_relation_props[str(relationship) + '_2'] = MPARelationProp(
                edge_index=edge_index,
                in_channels=kwargs['hidden_size'],
                out_channels=kwargs['hidden_size'],
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
