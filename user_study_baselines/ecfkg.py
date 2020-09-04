import argparse
import torch
import os
import numpy as np
import random as rd
import time
import tqdm
import sys
from GPUtil import showUtilization as gpu_usage

sys.path.append('..')
from torch.utils.data import DataLoader
from graph_recsys_benchmark.models import MCFKGRecsysModel
from graph_recsys_benchmark.solvers import BaseSolver
from graph_recsys_benchmark.utils import *

MODEL_TYPE = 'Graph'
GRAPH_TYPE = 'hete'
KG_LOSS_TYPE = 'BPR'
CF_LOSS_TYPE = 'BPR'
MODEL = 'MCFKG'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')
parser.add_argument('--dataset_name', type=str, default='1m', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')

# Model params
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--hidden_size', type=int, default=64, help='')

# Train params
parser.add_argument('--init_eval', type=str, default='true', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='7', help='')
parser.add_argument('--runs', type=int, default=5, help='')
parser.add_argument('--kg_epochs', type=int, default=30, help='')
parser.add_argument('--cf_epochs', type=int, default=30, help='')
parser.add_argument('--kg_batch_size', type=int, default=128, help='')
parser.add_argument('--cf_batch_size', type=int, default=1028, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--save_epochs', type=str, default='15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=1, help='')

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=CF_LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name, 'type': GRAPH_TYPE,
    'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'kg_loss_type': KG_LOSS_TYPE, 'cf_loss_type': CF_LOSS_TYPE
}
model_args = {
    'model_type': MODEL_TYPE,
    'emb_dim': args.emb_dim, 'hidden_size': args.hidden_size,
    'dropout': args.dropout,
}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs, 'kg_epochs': args.kg_epochs, 'cf_epochs': args.cf_epochs,
    'kg_batch_size': args.kg_batch_size, 'cf_batch_size': args.cf_batch_size,
    'num_workers': args.num_workers,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': os.path.join(weights_folder, str(model_args)),
    'logger_folder': os.path.join(logger_folder, str(model_args)),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))


def _cf_negative_sampling(u_nid, num_negative_samples, train_splition, item_nid_occs):
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


class MCFKGRecsysModel(MCFKGRecsysModel):
    def cf_loss(self, batch):
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])

        loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return loss

    def kg_loss(self, batch):
        h = self.x[batch[:, 0]]
        pos_t = self.x[batch[:, 1]]
        neg_t = self.x[batch[:, 2]]

        r = self.r[batch[:, 3]]
        pos_diff = (h + r) * pos_t
        neg_diff = (h + r) * neg_t

        loss = -(pos_diff - neg_diff).sigmoid().log().sum()

        return loss


class MCFKGRecsysModelSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(MCFKGRecsysModelSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, u_nid):
        pos_inids = dataset.test_pos_unid_inid_map[u_nid]
        neg_iids = np.array(rd.sample(dataset.unique_iids, train_args['num_neg_candidates']), dtype=int)
        neg_inids = [dataset.e2nid_dict['iid'][iid] for iid in neg_iids]

        return pos_inids, list(neg_inids)

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, \
        kg_train_loss_per_run_np, cf_train_loss_per_run_np, \
        kg_eval_loss_per_run_np, cf_eval_loss_per_run_np, last_run = \
            load_kgat_global_logger(global_logger_file_path)

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'w') as logger_file:
            start_run = last_run + 1
            if start_run <= self.train_args['runs']:
                for run in range(start_run, self.train_args['runs'] + 1):
                    # Fix the random seed
                    seed = 2019 + run
                    rd.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    # Create the dataset
                    dataset = load_dataset(self.dataset_args)

                    # Create model and optimizer
                    self.model_args['num_nodes'] = dataset.num_nodes
                    self.model_args['dataset'] = dataset

                    model = self.model_class(**self.model_args).to(self.train_args['device'])

                    opt_class = get_opt_class(self.train_args['opt'])
                    optimizer = opt_class(
                        params=model.parameters(),
                        lr=self.train_args['lr'],
                        weight_decay=self.train_args['weight_decay']
                    )

                    weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.train_args['device'])

                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, \
                    kg_train_loss_per_epoch_np, cf_train_loss_per_epoch_np, \
                    kg_eval_loss_per_epoch_np, cf_eval_loss_per_epoch_np = \
                        np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(
                            (0, 1)), np.zeros((0, 1)), np.zeros((0, 1))

                    t_start = time.perf_counter()
                    # Train KG embedding
                    for epoch in range(1, self.train_args['kg_epochs'] + 1):
                        # Train KG part
                        model.train()
                        kg_loss_per_batch = []
                        dataset.kg_negative_sampling()
                        dataloader = DataLoader(
                            dataset,
                            shuffle=True,
                            batch_size=self.train_args['kg_batch_size'],
                            num_workers=self.train_args['num_workers']
                        )
                        train_bar = tqdm.tqdm(dataloader, total=len(dataloader))
                        for _, batch in enumerate(train_bar):
                            batch = batch.to(self.train_args['device'])

                            optimizer.zero_grad()
                            loss = model.kg_loss(batch)
                            loss.backward()
                            optimizer.step()

                            kg_loss_per_batch.append(loss.detach().cpu().item())
                            kg_train_loss = np.mean(kg_loss_per_batch)
                            train_bar.set_description(
                                'Run: {}, epoch: {}, kg train loss: {:.4f}'.format(run, epoch, kg_train_loss)
                            )

                    for epoch in range(1, self.train_args['cf_epochs'] + 1):
                            # Train CF part
                            model.train()
                            cf_loss_per_batch = []
                            dataset.cf_negative_sampling()
                            dataloader = DataLoader(
                                dataset,
                                shuffle=True,
                                batch_size=self.train_args['cf_batch_size'],
                                num_workers=self.train_args['num_workers']
                            )
                            train_bar = tqdm.tqdm(dataloader, total=len(dataloader))
                            for _, batch in enumerate(train_bar):
                                batch = batch.to(self.train_args['device'])

                                optimizer.zero_grad()
                                loss = model.cf_loss(batch)
                                loss.backward()
                                optimizer.step()

                                cf_loss_per_batch.append(loss.detach().cpu().item())
                                cf_train_loss = np.mean(cf_loss_per_batch)
                                train_bar.set_description(
                                    'Run: {}, epoch: {}, cf train loss: {:.4f}'.format(run, epoch, cf_train_loss)
                                )

                            # Eval CF part
                            model.eval()
                            HRs, NDCGs, AUC, cf_eval_loss = self.metrics(run, epoch, model, dataset)

                            # Sumarize the epoch
                            HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                            NDCGs_per_epoch_np = np.vstack([NDCGs_per_epoch_np, NDCGs])
                            AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])

                            kg_train_loss_per_epoch_np = np.vstack([kg_train_loss_per_epoch_np, np.array([kg_train_loss])])
                            cf_train_loss_per_epoch_np = np.vstack([cf_train_loss_per_epoch_np, np.array([cf_train_loss])])
                            kg_eval_loss_per_epoch_np = np.vstack([kg_eval_loss_per_epoch_np, np.array([0])])
                            cf_eval_loss_per_epoch_np = np.vstack([cf_eval_loss_per_epoch_np, np.array([cf_eval_loss])])

                            print(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                    AUC[0], cf_train_loss, cf_eval_loss[0]
                                )
                            )
                            logger_file.write(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10], NDCGs[15],
                                    AUC[0], cf_train_loss, cf_eval_loss[0]
                                )
                            )
                            instantwrite(logger_file)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.train_args['device'])
                    t_end = time.perf_counter()

                    HRs_per_run_np = np.vstack([HRs_per_run_np, HRs_per_epoch_np[-1]])
                    NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, NDCGs_per_epoch_np[-1]])
                    AUC_per_run_np = np.vstack([AUC_per_run_np, AUC_per_epoch_np[-1]])
                    kg_train_loss_per_run_np = np.vstack([kg_train_loss_per_run_np, kg_train_loss_per_epoch_np[-1]])
                    cf_train_loss_per_run_np = np.vstack([cf_train_loss_per_run_np, cf_train_loss_per_epoch_np[-1]])
                    kg_eval_loss_per_run_np = np.vstack([kg_eval_loss_per_run_np, kg_eval_loss_per_epoch_np[-1]])
                    cf_eval_loss_per_run_np = np.vstack([cf_eval_loss_per_run_np, cf_eval_loss_per_epoch_np[-1]])

                    if epoch in self.train_args['save_epochs']:
                        weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                        save_kgat_model(
                            weightpath,
                            model, optimizer, epoch,
                            rec_metrics=(
                                HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np,
                                kg_train_loss_per_epoch_np, cf_train_loss_per_epoch_np,
                                kg_eval_loss_per_epoch_np, cf_eval_loss_per_epoch_np
                            )
                        )
                    if epoch > self.train_args['save_every_epoch']:
                        weightpath = os.path.join(weights_path, 'latest.pkl')
                        save_kgat_model(
                            weightpath,
                            model, optimizer, epoch,
                            rec_metrics=(
                                HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np,
                                kg_train_loss_per_epoch_np, cf_train_loss_per_epoch_np,
                                kg_eval_loss_per_epoch_np, cf_eval_loss_per_epoch_np
                            )
                        )

                    save_kg_global_logger(
                        global_logger_file_path,
                        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                        kg_train_loss_per_run_np, cf_train_loss_per_run_np,
                        kg_eval_loss_per_run_np, cf_eval_loss_per_run_np
                    )
                    print(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0],
                            np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10],
                            np.max(NDCGs_per_epoch_np, axis=0)[15], AUC_per_epoch_np[-1][0],
                            cf_train_loss_per_epoch_np[-1][0], cf_eval_loss_per_epoch_np[-1][0])
                    )
                    logger_file.write(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0],
                            np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10], np.max(NDCGs_per_epoch_np, axis=0)[15],
                            AUC_per_epoch_np[-1][0],
                            cf_train_loss_per_epoch_np[-1][0], cf_eval_loss_per_epoch_np[-1][0])
                    )
                    instantwrite(logger_file)

                    print("GPU Usage after each run")
                    gpu_usage()

                    del model, optimizer, loss, kg_loss_per_batch, cf_loss_per_batch
                    # clearcache()

                print(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                        NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                        NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                        cf_train_loss_per_run_np.mean(axis=0)[0], cf_eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                logger_file.write(
                    'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                        HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                        HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                        NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                        NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                        cf_train_loss_per_run_np.mean(axis=0)[0], cf_eval_loss_per_run_np.mean(axis=0)[0]
                    )
                )
                instantwrite(logger_file)


if __name__ == '__main__':
    dataset_args['_cf_negative_sampling'] = _cf_negative_sampling
    solver = MCFKGRecsysModelSolver(MCFKGRecsysModel, dataset_args, model_args, train_args)
    solver.run()
