import argparse
import torch
import os
import numpy as np
import random as rd
import sys

sys.path.append('..')
from graph_recsys_benchmark.models import PAGAGCNRecsysModel
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.solvers import BaseSolver

MODEL_TYPE = 'Graph'
LOSS_TYPE = 'BPR'
MODEL = 'PAGAGCN'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='Yelp', help="")
parser.add_argument("--if_use_features", type=bool, default=False, help="")
parser.add_argument("--num_core", type=int, default=10, help="")
# Model params
parser.add_argument("--dropout", type=float, default=0, help="")
parser.add_argument("--emb_dim", type=int, default=64, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
parser.add_argument("--meta_path_steps", type=list, default=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], help="")
parser.add_argument("--aggr", type=str, default='concat', help="")

# Train params
parser.add_argument("--init_eval", type=bool, default=True, help="")
parser.add_argument("--num_negative_samples", type=int, default=4, help="")
parser.add_argument("--num_neg_candidates", type=int, default=99, help="")

parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--runs", type=int, default=20, help="")
parser.add_argument("--epochs", type=int, default=30, help="")
parser.add_argument("--batch_size", type=int, default=4096, help="")
parser.add_argument("--num_workers", type=int, default=12, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--lr", type=float, default=0.001, help="")
parser.add_argument("--weight_decay", type=float, default=0, help="")
parser.add_argument("--early_stopping", type=int, default=20, help="")
parser.add_argument("--save_epochs", type=list, default=[5, 10, 15, 20, 25], help="")
parser.add_argument("--save_every_epoch", type=int, default=25, help="")

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
    'if_use_features': args.if_use_features, 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'loss_type': LOSS_TYPE
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features,
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim, 'dropout': args.dropout,
    'meta_path_steps': args.meta_path_steps, 'aggr': args.aggr
}
train_args = {
    'init_eval': args.init_eval,
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
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
    :param b_nid:
    :return:
    """
    train_pos_bnid_unid_map, test_pos_bnid_unid_map, neg_bnid_unid_map = train_splition
    # negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    # nid_occs = np.array([item_nid_occs[nid] for nid in negative_inids])
    # nid_occs = nid_occs / np.sum(nid_occs)
    # negative_inids = rd.choices(population=negative_inids, weights=nid_occs, k=num_negative_samples)
    # negative_inids = negative_inids

    negative_unids = test_pos_bnid_unid_map[b_nid] + neg_bnid_unid_map[b_nid]
    negative_unids = rd.choices(population=negative_unids, k=num_negative_samples)

    return negative_unids


class PAGAGCNRecsysModel(PAGAGCNRecsysModel):
    def loss(self, batch):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])

        loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return loss

    def update_graph_input(self, dataset):
        bus2user_edge_index = torch.from_numpy(dataset.edge_index_nps['bus2user']).long().to(train_args['device'])
        names2user_edge_index = torch.from_numpy(dataset.edge_index_nps['names2user']).long().to(train_args['device'])
        reviewcount2user_edge_index = torch.from_numpy(dataset.edge_index_nps['reviewcount2user']).long().to(train_args['device'])
        startdate2user_edge_index = torch.from_numpy(dataset.edge_index_nps['startdate2user']).long().to(train_args['device'])
        friendcount2user_edge_index = torch.from_numpy(dataset.edge_index_nps['friendcount2user']).long().to(train_args['device'])
        # friends2user_edge_index = torch.from_numpy(dataset.edge_index_nps['friends2user']).long().to(train_args['device'])
        fans2user_edge_index = torch.from_numpy(dataset.edge_index_nps['fans2user']).long().to(train_args['device'])
        elite2user_edge_index = torch.from_numpy(dataset.edge_index_nps['elite2user']).long().to(train_args['device'])
        averagestars2user_edge_index = torch.from_numpy(dataset.edge_index_nps['averagestars2user']).long().to(train_args['device'])
        name2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['name2bus']).long().to(train_args['device'])
        city2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['city2bus']).long().to(train_args['device'])
        state2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['state2bus']).long().to(train_args['device'])
        stars2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['stars2bus']).long().to(train_args['device'])
        reviewcount2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['reviewcount2bus']).long().to(train_args['device'])
        isopen2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['isopen2bus']).long().to(train_args['device'])
        attributes2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['attributes2bus']).long().to(train_args['device'])
        categories2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['categories2bus']).long().to(train_args['device'])
        time2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['time2bus']).long().to(train_args['device'])
        checkincount2bus_edge_index = torch.from_numpy(dataset.edge_index_nps['checkincount2bus']).long().to(train_args['device'])
        meta_path_edge_indicis_1 = [bus2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_2 = [torch.flip(bus2user_edge_index, dims=[0]), bus2user_edge_index]
        meta_path_edge_indicis_3 = [names2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_4 = [reviewcount2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_5 = [startdate2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_6 = [friendcount2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        # meta_path_edge_indicis_6 = [friends2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_7 = [fans2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_8 = [elite2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_9 = [averagestars2user_edge_index, torch.flip(bus2user_edge_index, dims=[0])]
        meta_path_edge_indicis_10 = [name2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_11 = [city2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_12 = [state2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_13 = [stars2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_14 = [reviewcount2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_15 = [isopen2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_16 = [attributes2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_17 = [categories2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_18 = [time2bus_edge_index, bus2user_edge_index]
        meta_path_edge_indicis_19 = [checkincount2bus_edge_index, bus2user_edge_index]

        meta_path_edge_index_list = [
            meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3, meta_path_edge_indicis_4,
            meta_path_edge_indicis_5, meta_path_edge_indicis_6, meta_path_edge_indicis_7, meta_path_edge_indicis_8,
            meta_path_edge_indicis_9, meta_path_edge_indicis_10, meta_path_edge_indicis_11, meta_path_edge_indicis_12,
            meta_path_edge_indicis_13, meta_path_edge_indicis_14, meta_path_edge_indicis_15, meta_path_edge_indicis_16,
            meta_path_edge_indicis_17, meta_path_edge_indicis_18, meta_path_edge_indicis_19
        ]
        return self.x, meta_path_edge_index_list


class PAGAGCNSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(PAGAGCNSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, b_nid):
        pos_u_nids = dataset.test_pos_bnid_unid_map[b_nid]
        neg_u_nids = np.array(dataset.neg_bnid_unid_map[b_nid])

        neg_u_nids_indices = np.array(rd.sample(range(neg_u_nids.shape[0]), train_args['num_neg_candidates']), dtype=int)

        return pos_u_nids, list(neg_u_nids[neg_u_nids_indices])


if __name__ == '__main__':
    dataset_args['_negative_sampling'] = _negative_sampling
    solver = PAGAGCNSolver(PAGAGCNRecsysModel, dataset_args, model_args, train_args)
    solver.run()
