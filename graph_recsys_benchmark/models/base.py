import torch
from torch.nn import functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import glorot, zeros


class BaseRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError


class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        if self.entity_aware and self.training:
            pos_item_entity, neg_item_entity = pos_neg_pair_t[:, 3], pos_neg_pair_t[:, 4]
            pos_user_entity, neg_user_entity = pos_neg_pair_t[:, 6], pos_neg_pair_t[:, 7]
            item_entity_mask, user_entity_mask = pos_neg_pair_t[:, 5], pos_neg_pair_t[:, 8]

            # cos
            if hasattr(self, 'entity_aware_type') and self.entity_aware_type == 'cos':
                x = F.normalize(self.x)
                item_pos_reg = x[pos_neg_pair_t[:, 1]] * x[pos_item_entity]
                item_neg_reg = x[pos_neg_pair_t[:, 1]] * x[neg_item_entity]
                item_pos_reg = item_pos_reg.sum(dim=-1)
                item_neg_reg = item_neg_reg.sum(dim=-1)

                user_pos_reg = x[pos_neg_pair_t[:, 0]] * x[pos_user_entity]
                user_neg_reg = x[pos_neg_pair_t[:, 0]] * x[neg_user_entity]
                user_pos_reg = user_pos_reg.sum(dim=-1)
                user_neg_reg = user_neg_reg.sum(dim=-1)

                item_reg_los = -((item_pos_reg - item_neg_reg) * item_entity_mask).sigmoid().log().sum()
                user_reg_los = -((user_pos_reg - user_neg_reg) * user_entity_mask).sigmoid().log().sum()
                reg_los = item_reg_los + user_reg_los
            else:
                # l2 norm
                x = self.x
                item_pos_reg = (x[pos_neg_pair_t[:, 1]] - x[pos_item_entity]) * (
                            x[pos_neg_pair_t[:, 1]] - x[pos_item_entity])
                item_neg_reg = (x[pos_neg_pair_t[:, 1]] - x[neg_item_entity]) * (
                            x[pos_neg_pair_t[:, 1]] - x[neg_item_entity])
                item_pos_reg = item_pos_reg.sum(dim=-1)
                item_neg_reg = item_neg_reg.sum(dim=-1)

                user_pos_reg = (x[pos_neg_pair_t[:, 0]] - x[pos_user_entity]) * (
                            x[pos_neg_pair_t[:, 0]] - x[pos_user_entity])
                user_neg_reg = (x[pos_neg_pair_t[:, 0]] - x[neg_user_entity]) * (
                            x[pos_neg_pair_t[:, 0]] - x[neg_user_entity])
                user_pos_reg = user_pos_reg.sum(dim=-1)
                user_neg_reg = user_neg_reg.sum(dim=-1)

                item_reg_los = -((item_pos_reg - item_neg_reg) * item_entity_mask).sigmoid().log().sum()
                user_reg_los = -((user_pos_reg - user_neg_reg) * user_entity_mask).sigmoid().log().sum()
                reg_los = item_reg_los + user_reg_los

            # two parts of loss
            loss = cf_loss + self.entity_aware_coff * reg_los
        else:
            loss = cf_loss

        return loss

    def update_graph_input(self, dataset):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError

    def eval(self):
        super(GraphRecsysModel, self).eval()
        if self.__class__.__name__ not in ['KGATRecsysModel', 'KGCNRecsysModel']:
            with torch.no_grad():
                self.cached_repr = self.forward()


class MFRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MFRecsysModel, self).__init__()
        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        loss_func = torch.nn.BCEWithLogitsLoss()
        if self.training:
            pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
            label = pos_neg_pair_t[:, -1].float()
        else:
            pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])[:1]
            neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).float()

        loss = loss_func(pred, label)
        return loss

    def predict(self, unids, inids):
        return self.forward(unids, inids)


class PEABaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        assert len(edge_index_list) == self.num_steps

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gnn_layers[-1](x, edge_index_list[-1])
        return x


class PEABaseRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEABaseRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_type = kwargs['entity_aware_type']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.channel_aggr = kwargs['channel_aggr']

        # Create node embedding
        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')

        # Create graphs
        meta_path_edge_index_list = self.update_graph_input(kwargs['dataset'])
        assert len(meta_path_edge_index_list) == len(kwargs['meta_path_steps'])
        self.meta_path_edge_index_list = meta_path_edge_index_list

        # Create channels
        self.pea_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.pea_channels.append(kwargs_cpy['channel_class'](**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = Parameter(torch.Tensor(1, len(kwargs['meta_path_steps']), kwargs['repr_dim']))

        if self.channel_aggr == 'cat':
            self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        for module in self.pea_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att)

    def forward(self):
        x = self.x
        x = [module(x, self.meta_path_edge_index_list[idx]).unsqueeze(1) for idx, module in enumerate(self.pea_channels)]
        x = torch.cat(x, dim=1)
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PEAJKBaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for step_idx in range(self.num_steps):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return self.lin(self.jump(xs))


class PEAJKBaseRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEAJKBaseRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_type = kwargs['entity_aware_type']
        self.entity_aware_coff = kwargs['entity_aware_coff']
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.channel_aggr = kwargs['channel_aggr']

        # Create node embedding or use node features
        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')

        # Create graphs
        meta_path_edge_indices = self.update_graph_input(kwargs['dataset'])
        assert len(meta_path_edge_indices) == len(kwargs['meta_path_steps'])
        self.meta_path_edge_indices = meta_path_edge_indices

        self.gnn_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.gnn_channels.append(kwargs_cpy['channel_class'](**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = Parameter(torch.Tensor(1, len(kwargs['meta_path_steps']), kwargs['hidden_size']))
        self.fc = torch.nn.Linear(kwargs['hidden_size'], kwargs['repr_dim'])
        self.bias = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        for module in self.gnn_channels:
            module.reset_parameters()
        if self.channel_aggr == 'att':
            glorot(self.att)
        glorot(self.fc.weight)
        zeros(self.bias)

    def forward(self):
        x = self.x
        x = [F.relu(channel(x, self.meta_path_edge_indices[idx]).unsqueeze(1)) for idx, channel in enumerate(self.gnn_channels)]
        x = torch.cat(x, dim=1)
        if self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1), dim=1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        x = self.fc(x)
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids]
        i_repr = self.cached_repr[inids]
        return torch.sum(u_repr * i_repr, dim=-1) + self.bias
