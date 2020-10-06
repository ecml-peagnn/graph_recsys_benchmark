import torch
from torch.nn import Parameter
from torch import Tensor
from .base import BaseRecsysModel


class HetRecRecsysModel(BaseRecsysModel):
    def __init__(self, **kwargs):
        super(HetRecRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.factor_num = kwargs['factor_num']
        self.num_uids, self.num_iids = kwargs['dataset'].num_uids, kwargs['dataset'].num_iids
        self.acc_iids, self.acc_uids = kwargs['dataset'].type_accs['iid'], kwargs['dataset'].type_accs['uid']
        self.diffused_score_mats, self.num_metapaths = self.compute_diffused_score_mat(kwargs['dataset'])

        self.user_emb = Parameter(Tensor(self.num_uids, self.num_metapaths, self.factor_num))
        self.item_emb = Parameter(Tensor(self.num_iids, self.num_metapaths, self.factor_num))

        self.theta = Parameter(Tensor(1, self.num_metapaths))

    def reset_parameters(self):
        torch.nn.init.normal_(self.user_emb, std=0.01)
        torch.nn.init.normal_(self.item_emb, std=0.01)
        torch.nn.init.normal_(self.theta, std=0.01)

    def forward(self, unids, inids):
        ratings = torch.sum(
            torch.relu(self.user_emb[unids - self.acc_uids]) * torch.relu(self.item_emb[inids - self.acc_iids]),
            dim=-1
        )
        return ratings

    def predict(self, unids, inids):
        ratings = self.forward(unids, inids)
        ratings *= torch.sigmoid(self.theta)
        ratings = torch.sum(ratings, dim=-1)
        return ratings