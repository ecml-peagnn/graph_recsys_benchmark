import torch
from .base import MFRecsysModel


class MFRecsysModel(MFRecsysModel):
    def __init__(self, **kwargs):
        super(MFRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.user_emb = torch.nn.Embedding(kwargs['num_users'], kwargs['factor_num'])
        self.item_emb = torch.nn.Embedding(kwargs['num_items'], kwargs['factor_num'])

    def reset_parameters(self):
        torch.nn.init.normal_(self.user_emb.weight, std=0.01)
        torch.nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, uid, iid):
        rating = torch.sum(self.user_emb(uid) * self.item_emb(iid), dim=-1)
        return rating