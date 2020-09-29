import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import normal

from .base import BaseRecsysModel


class CKERecsysModel(BaseRecsysModel):
    def __init__(self, **kwargs):
        super(CKERecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))
        self.r = Parameter(torch.Tensor(kwargs['dataset'].num_edge_types, kwargs['emb_dim']))
        self.proj_mat = Parameter(torch.Tensor(kwargs['emb_dim'], kwargs['emb_dim']))
        self.eta = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], kwargs['emb_dim']))

        self.lambda_u = kwargs['lambda_u']
        self.lambda_v = kwargs['lambda_v']
        self.lambda_r = kwargs['lambda_r']
        self.lambda_m = kwargs['lambda_m']
        self.lambda_i = kwargs['lambda_i']

        self.acc_users = kwargs['dataset']['type_accs']['uid']
        self.acc_items = kwargs['dataset']['type_accs']['iid']
        self.num_users = kwargs['dataset']['num_uids']
        self.num_items = kwargs['dataset']['num_iids']

    def reset_parameters(self):
        normal(self.x[self.acc_users: self.acc_users + self.num_users], 0, self.lambda_u)
        normal(self.x[self.acc_items: self.acc_items + self.num_items], 0, self.lambda_v)
        normal(self.r, 0, self.lambda_r)
        normal(self.proj_mat, 0, self.lambda_m)
        normal(self.eta, 0, self.lambda_i)

    def forward(self, unids, inids):
        u_repr = self.x[unids]
        i_repr = self.x[inids] + self.eta[inids]
        return torch.sum(u_repr * i_repr, dim=-1)

    def predict(self, unids, inids):
        return self.forward(unids, inids)

    def loss(self, pos_neg_pair_t):
        # CF loss
        pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        # Reg loss
        eta = self.eta[self.acc_items: self.acc_items + self.num_items]
        reg_loss = 0.5 * self.lambda_i * torch.sum(torch.sum(eta * eta, dim=-1))
        reg_loss += 0.5 * self.lambda_r * torch.sum(torch.sum(self.r * self.r, dim=-1))
        reg_loss += 0.5 * self.lambda_m * torch.sum(torch.sum(self.proj_mat * self.proj_mat, dim=-1))

        loss = cf_loss + reg_loss

        if self.training:
            # KG loss
            h = self.x[pos_neg_pair_t[:, 3]]
            pos_t = self.x[pos_neg_pair_t[:, 4]]
            neg_t = self.x[pos_neg_pair_t[:, 5]]

            r = self.r[pos_neg_pair_t[:, 6]]
            pos_diff = torch.mm(h, self.proj_mat) + r - torch.mm(pos_t, self.proj_mat)
            neg_diff = torch.mm(h, self.proj_mat) + r - torch.mm(neg_t, self.proj_mat)

            pos_pred = (pos_diff * pos_diff).sum(-1)
            neg_pred = (neg_diff * neg_diff).sum(-1)

            kg_loss = -(pos_pred - neg_pred).sigmoid().log().sum()
            loss += kg_loss

        return loss
