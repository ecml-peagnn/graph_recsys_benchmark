import torch
from torch.nn import functional as F


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
            if hasattr(self, 'entiy_aware_type') and self.entiy_aware_type == 'cos':
                item_pos_reg = self.cached_repr[pos_neg_pair_t[:, 1]] * self.cached_repr[pos_item_entity]
                item_neg_reg = self.cached_repr[pos_neg_pair_t[:, 1]] * self.cached_repr[neg_item_entity]
                item_pos_reg = item_pos_reg.sum(dim=-1)
                item_neg_reg = item_neg_reg.sum(dim=-1)

                user_pos_reg = self.cached_repr[pos_neg_pair_t[:, 0]] * self.cached_repr[pos_user_entity]
                user_neg_reg = self.cached_repr[pos_neg_pair_t[:, 0]] * self.cached_repr[neg_user_entity]
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