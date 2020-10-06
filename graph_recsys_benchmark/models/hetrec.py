import torch
from torch.nn import functional as F
from torch.nn import Parameter
from torch import Tensor
from .base import BaseRecsysModel


class HetRecRecsysModel(BaseRecsysModel):
    def __init__(self, **kwargs):
        super(HetRecRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.num_metapaths = kwargs['num_metapaths']
        self.factor_num = kwargs['factor_num']
        self.user_emb = torch.nn.Embedding(kwargs['dataset'].num_uids, kwargs['num_metapaths'] * kwargs['factor_num'])
        self.item_emb = torch.nn.Embedding(kwargs['dataset'].num_iids, kwargs['num_metapaths'] * kwargs['factor_num'])

        self.theta = Parameter(Tensor(1, self.num_metapaths))

        self.diffused_score_mat = self.compute_diffused_score_mat(kwargs['dataset'])

    def reset_parameters(self):
        torch.nn.init.normal_(self.user_emb.weight, std=0.01)
        torch.nn.init.normal_(self.item_emb.weight, std=0.01)

    def update_graph_input(self, dataset):
        raise NotImplementedError

    def forward(self, uid, iid):
        ratings = torch.sum(
            torch.sigmoid(self.user_emb(uid)) * torch.sigmoid(self.item_emb(iid)).view(-1, self.num_metapaths, self.factor_num),
            dim=-1
        )
        return ratings

    def predict(self, uids, iids):
        ratings = self.forward(uids, iids)
        ratings *= torch.sigmoid(self.theta)
        ratings = torch.sum(ratings, dim=-1)
        return ratings