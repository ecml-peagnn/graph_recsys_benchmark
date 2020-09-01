import torch


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
            pos_reg = (self.cached_repr[pos_neg_pair_t[:, 1]] - self.cached_repr[pos_entity]) * (
                        self.cached_repr[pos_neg_pair_t[:, 1]] - self.cached_repr[pos_entity])
            pos_reg = pos_reg.sum(dim=-1)
            neg_reg = (self.cached_repr[pos_neg_pair_t[:, 1]] - self.cached_repr[neg_entity]) * (
                        self.cached_repr[pos_neg_pair_t[:, 1]] - self.cached_repr[neg_entity])
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
        raise NotImplementedError

    def predict(self, unids, inids):
        return self.forward(unids, inids)