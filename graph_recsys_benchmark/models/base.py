import torch
from torch.nn import functional as F


class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()
        self.entity_aware = kwargs['entity_aware']
        self.entity_aware_coff = kwargs['entity_aware_coff']

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
            pos_entity, neg_entity = pos_neg_pair_t[:, 3], pos_neg_pair_t[:, 4]
            x = F.normalize(self.cached_repr)

            # l2 norm
            pos_reg = (x[pos_neg_pair_t[:, 1]] - x[pos_entity]) * (
                        x[pos_neg_pair_t[:, 1]] - x[pos_entity])
            neg_reg = (x[pos_neg_pair_t[:, 1]] - x[neg_entity]) * (
                        x[pos_neg_pair_t[:, 1]] - x[neg_entity])

            # # cos distance
            # pos_reg = - (x[pos_neg_pair_t[:, 1]] * x[pos_entity])
            # neg_reg = - (x[pos_neg_pair_t[:, 1]] * x[neg_entity])

            pos_reg = pos_reg.sum(dim=-1)
            neg_reg = neg_reg.sum(dim=-1)

            reg_los = -(pos_reg - neg_reg).sigmoid().log().sum()

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