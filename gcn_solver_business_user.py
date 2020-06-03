import argparse
import torch
import os
import numpy as np
import random as rd

from graph_recsys_benchmark.models import GCNRecsysModel
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.solvers import BaseSolver


MODEL_TYPE = 'Graph'
MODEL = 'GCN'

parser = argparse.ArgumentParser()
# Dataset params
parser.add_argument("--dataset", type=str, default='Yelp', help="")
parser.add_argument("--if_use_features", type=bool, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
# Model params
parser.add_argument("--dropout", type=float, default=0.5, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
# Train params
parser.add_argument("--num_negative_samples", type=int, default=5, help="")
parser.add_argument("--init_eval", type=bool, default=True, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=20, help="")
parser.add_argument("--epochs", type=int, default=30, help="")
parser.add_argument("--batch_size", type=int, default=4096, help="")
parser.add_argument("--num_workers", type=int, default=12, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--lr", type=float, default=1e-3, help="")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="")
parser.add_argument("--early_stopping", type=int, default=20, help="")
parser.add_argument("--save_epochs", type=list, default=[5, 10, 15, 20, 25], help="")
parser.add_argument("--save_every_epoch", type=int, default=25, help="")

args = parser.parse_args()

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset,
    'if_use_features': args.if_use_features, 'emb_dim': args.emb_dim,
    'num_core': args.num_core, 'train_ratio': args.train_ratio
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features,
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout
}
train_args = {
    'init_eval': args.init_eval, 'num_negative_samples': args.num_negative_samples,
    'opt': args.opt, 'loss': args.loss,
    'runs': args.runs, 'epochs': args.epochs, 'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': args.save_epochs, 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def _negative_sampling(b_nid, num_negative_samples, train_splition, user_nid_occs):
    """
    The negative sampling methods used for generating the training batches
    :param u_nid:
    :return:
    """
    train_pos_bnid_unid_map, test_pos_bnid_unid_map, neg_bnid_unid_map = train_splition
    negative_unids = test_pos_bnid_unid_map[b_nid] + neg_bnid_unid_map[b_nid]
    nid_occs = np.array([user_nid_occs[nid] for nid in negative_unids])
    nid_occs = nid_occs / np.sum(nid_occs)
    negative_unids = rd.choices(population=negative_unids, weights=nid_occs, k=num_negative_samples)
    negative_unids = negative_unids

    return negative_unids


class BusUserGCNRecsysModel(GCNRecsysModel):
    def loss_func(self, pos_u_ratings, neg_u_ratings):
        return - (pos_u_ratings - neg_u_ratings).sigmoid().log().sum()

    def update_graph_input(self, dataset):
        edge_index_np = dataset.edge_index_nps['bus2user']
        edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, 0)])
        edge_index = torch.from_numpy(edge_index_np).long().to(train_args['device'])
        return edge_index


class GCNSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(GCNSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, b_nid):
        pos_u_nids = dataset.test_pos_bnid_unid_map[b_nid]
        neg_u_nids = np.array(dataset.neg_bnid_unid_map[b_nid])

        neg_u_nids_indices = np.array(rd.sample(range(neg_u_nids.shape[0]), 99), dtype=int)

        return pos_u_nids, list(neg_u_nids[neg_u_nids_indices])


if __name__ == '__main__':
    dataset_args['_negative_sampling'] = _negative_sampling
    solver = GCNSolver(BusUserGCNRecsysModel, dataset_args, model_args, train_args)
    solver.run()
