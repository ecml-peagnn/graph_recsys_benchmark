import argparse
import torch
import os
import numpy as np
import random as rd
import sys

sys.path.append('..')
from graph_recsys_benchmark.models import SAGERecsysModel
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'Graph'
LOSS_TYPE = 'BPR'
MODEL = 'PinSAGE'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Yelp', help='')
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
# Model params
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--repr_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--margin', type=float, default=1, help='')

# Train params
parser.add_argument('--init_eval', type=str, default='false', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=3, help='')
parser.add_argument('--epochs', type=int, default=20, help='')
parser.add_argument('--batch_size', type=int, default=1024, help='')
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='5,10,15', help='')
parser.add_argument('--save_every_epoch', type=int, default=16, help='')

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset, loss_type=LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset,
    'if_use_features': args.if_use_features.lower() == 'true', 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'cf_loss_type': LOSS_TYPE, 'type': GRAPH_TYPE
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features.lower() == 'true',
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout,
    'margin': args.margin
}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def _negative_sampling(u_nid, num_negative_samples, train_splition, item_nid_occs):
    '''
    The negative sampling methods used for generating the training batches
    :param u_nid:
    :return:
    '''
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = train_splition
    # negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    # nid_occs = np.array([item_nid_occs[nid] for nid in negative_inids])
    # nid_occs = nid_occs / np.sum(nid_occs)
    # negative_inids = rd.choices(population=negative_inids, weights=nid_occs, k=num_negative_samples)
    # negative_inids = negative_inids

    negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    negative_inids = rd.choices(population=negative_inids, k=num_negative_samples)

    return np.array(negative_inids).reshape(-1, 1)


class SAGERecsysModel(SAGERecsysModel):
    def cf_loss(self, batch):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])

        loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return loss

    def update_graph_input(self, dataset):
        edge_index_np = np.hstack(list(dataset.edge_index_nps.values()))
        edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, 0)])
        edge_index = torch.from_numpy(edge_index_np).long().to(train_args['device'])
        return self.x, edge_index


class SAGESolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(SAGESolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, u_nid):
        pos_inids = dataset.test_pos_unid_inid_map[u_nid]
        neg_iids = np.array(rd.sample(dataset.unique_iids, train_args['num_neg_candidates']), dtype=int)
        neg_inids = [dataset.e2nid_dict['iid'][iid] for iid in neg_iids]

        return pos_inids, list(neg_inids)

if __name__ == '__main__':
    dataset_args['_cf_negative_sampling'] = _negative_sampling
    solver = SAGESolver(SAGERecsysModel, dataset_args, model_args, train_args)
    solver.run()
