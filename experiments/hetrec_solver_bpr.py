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
from graph_recsys_benchmark.models import HetRecRecsysModel
from graph_recsys_benchmark.solvers import BaseSolver
from graph_recsys_benchmark.utils import *

MODEL_TYPE = 'Graph'
CF_LOSS_TYPE = 'BPR'
MODEL = 'HetRec'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')		#Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')	#1m, 25m, latest-small
parser.add_argument('--num_core', type=int, default=10, help='')			#10(for others), 20(only for 25m)
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='random', help='')		#unseen(for 1m,latest-small), random(for Yelp,25m)
parser.add_argument('--entity_aware', type=str, default='false', help='')

# Model params
parser.add_argument('--num_metapaths', type=int, default=1, help='')
parser.add_argument('--factor_num', type=int, default=64, help='')		#64(for others), 32(only for 25m)

# Train params
parser.add_argument('--init_eval', type=str, default='false', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='3', help='')
parser.add_argument('--runs', type=int, default=5, help='')
parser.add_argument('--emb_iter', type=int, default=10000, help='')          #30(for others), 20(only for Yelp)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for others), 20(only for Yelp)
parser.add_argument('--batch_size', type=int, default=1024, help='')    #1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=26, help='')        #26(for others), 16(only for Yelp)

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
    'cf_loss_type': CF_LOSS_TYPE,
    'sampling_strategy': args.sampling_strategy, 'entity_aware': args.entity_aware.lower() == 'true'
}
model_args = {
    'model_type': MODEL_TYPE,
    'factor_num': args.factor_num,
    'num_metapaths': args.num_metapaths
}
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt, 'emb_iter': args.emb_iter,
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


class HetRecRecsysModel(HetRecRecsysModel):
    def loss(self, batch):
        pos_pred = self.predict(batch[:, 0], batch[:, 1])
        neg_pred = self.predict(batch[:, 0], batch[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return cf_loss

    def mse_loss(self):
        loss_func = torch.nn.MSELoss()

        user_emb = torch.relu(self.user_emb.view(self.num_metapaths, self.num_uids, self.factor_num))
        item_emb = torch.relu(self.item_emb.view(self.num_metapaths, self.factor_num, self.num_iids))

        mask = torch.ones_like(self.diffused_score_mats)
        mask[torch.where(self.diffused_score_mats == 0)] = 0

        loss = loss_func(torch.matmul(user_emb, item_emb) * mask, self.diffused_score_mats)

        return loss

    def compute_diffused_score_mat(self, dataset):
        item_similarity_mat = compute_item_similarity_mat(dataset, ['-user2item', 'user2item'])
        diffused_score_mat_1_t = torch.from_numpy(compute_diffused_score_mat(dataset, item_similarity_mat)).to(train_args['device']).unsqueeze(0)
        item_similarity_mat = compute_item_similarity_mat(dataset, ['-genre2item', 'genre2item'])
        diffused_score_mat_2_t = torch.from_numpy(compute_diffused_score_mat(dataset, item_similarity_mat)).to(train_args['device']).unsqueeze(0)
        item_similarity_mat = compute_item_similarity_mat(dataset, ['-director2item', 'director2item'])
        diffused_score_mat_3_t = torch.from_numpy(compute_diffused_score_mat(dataset, item_similarity_mat)).to(train_args['device']).unsqueeze(0)

        diffused_score_mats = torch.cat([diffused_score_mat_1_t, diffused_score_mat_2_t, diffused_score_mat_3_t], dim=0).float()
        # diffused_score_mats = torch.cat([diffused_score_mat_3_t], dim=0).float()

        return diffused_score_mats, 3


class HetRecSolver(BaseSolver):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        super(HetRecSolver, self).__init__(model_class, dataset_args, model_args, train_args)

    def generate_candidates(self, dataset, u_nid):
        pos_i_nids = dataset.test_pos_unid_inid_map[u_nid]
        neg_i_nids = list(np.random.choice(dataset.neg_unid_inid_map[u_nid], size=(self.train_args['num_neg_candidates'],)))

        return pos_i_nids, neg_i_nids

    def run(self):
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
            load_global_logger(global_logger_file_path)

        print("GPU Usage before data load")
        gpu_usage()

        # Create the dataset
        dataset = load_dataset(self.dataset_args)

        print("GPU Usage after data load")
        gpu_usage()

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'a') as logger_file:
            start_run = last_run + 1
            if start_run <= self.train_args['runs']:
                for run in range(start_run, self.train_args['runs'] + 1):
                    # Fix the random seed
                    seed = 2019 + run
                    rd.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

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

                    # Load models
                    weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)
                    weights_file = os.path.join(weights_path, 'latest.pkl')
                    model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                           self.train_args['device'])
                    HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np, eval_loss_per_epoch_np = \
                        rec_metrics

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.train_args['device'])

                    # Train embedding
                    mse_loss_per_iter = []
                    pbar = tqdm.tqdm(range(self.train_args['emb_iter']))
                    for iter in pbar:
                        optimizer.zero_grad()
                        mse_loss = model.mse_loss()
                        mse_loss.backward()
                        optimizer.step()

                        mse_loss_per_iter.append(mse_loss.detach().cpu().item())
                        mse_train_loss = np.mean(mse_loss_per_iter[-100:])
                        pbar.set_description(
                            'Run: {}, embedding iter: {}, mse train loss: {:.4f}'.format(run, iter, mse_train_loss)
                        )
                    model.user_emb.requires_grad = False
                    model.item_emb.requires_grad = False

                    start_epoch = last_epoch + 1
                    t_start = time.perf_counter()
                    if start_epoch <= self.train_args['epochs']:
                        # Start training model
                        for epoch in range(start_epoch, self.train_args['epochs'] + 1):
                            # Train CF part
                            model.train()
                            cf_loss_per_batch = []
                            dataset.cf_negative_sampling()
                            dataloader = DataLoader(
                                dataset,
                                shuffle=True,
                                batch_size=self.train_args['batch_size'],
                                num_workers=self.train_args['num_workers']
                            )
                            train_bar = tqdm.tqdm(dataloader, total=len(dataloader))
                            for _, batch in enumerate(train_bar):
                                batch = batch.to(self.train_args['device'])

                                optimizer.zero_grad()
                                loss = model.loss(batch)
                                loss.backward()
                                optimizer.step()

                                cf_loss_per_batch.append(loss.detach().cpu().item())
                                cf_train_loss = np.mean(cf_loss_per_batch)
                                train_bar.set_description(
                                    'Run: {}, epoch: {}, cf train loss: {:.4f}'.format(run, epoch, cf_train_loss)
                                )

                            model.eval()
                            with torch.no_grad():
                                HRs, NDCGs, AUC, eval_loss = self.metrics(run, epoch, model, dataset)

                            # Sumarize the epoch
                            HRs_per_epoch_np = np.vstack([HRs_per_epoch_np, HRs])
                            NDCGs_per_epoch_np = np.vstack([NDCGs_per_epoch_np, NDCGs])
                            AUC_per_epoch_np = np.vstack([AUC_per_epoch_np, AUC])
                            train_loss_per_epoch_np = np.vstack([train_loss_per_epoch_np, np.array([cf_train_loss])])
                            eval_loss_per_epoch_np = np.vstack([eval_loss_per_epoch_np, np.array([eval_loss])])

                            if epoch in self.train_args['save_epochs']:
                                weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                        HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np,
                                        eval_loss_per_epoch_np)
                                )
                            if epoch > self.train_args['save_every_epoch']:
                                weightpath = os.path.join(weights_path, 'latest.pkl')
                                save_model(
                                    weightpath,
                                    model, optimizer, epoch,
                                    rec_metrics=(
                                        HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np, train_loss_per_epoch_np,
                                        eval_loss_per_epoch_np)
                                )
                            print(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                    NDCGs[15],
                                    AUC[0], cf_train_loss, eval_loss[0]
                                )
                            )
                            logger_file.write(
                                'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                    run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                    NDCGs[15],
                                    AUC[0], cf_train_loss, eval_loss[0]
                                )
                            )
                            instantwrite(logger_file)
                            clearcache()

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_end = time.perf_counter()

                        HRs_per_run_np = np.vstack([HRs_per_run_np, np.max(HRs_per_epoch_np, axis=0)])
                        NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, np.max(NDCGs_per_epoch_np, axis=0)])
                        AUC_per_run_np = np.vstack([AUC_per_run_np, np.max(AUC_per_epoch_np, axis=0)])
                        train_loss_per_run_np = np.vstack(
                            [train_loss_per_run_np, np.mean(train_loss_per_epoch_np, axis=0)])
                        eval_loss_per_run_np = np.vstack(
                            [eval_loss_per_run_np, np.mean(eval_loss_per_epoch_np, axis=0)])

                        save_global_logger(
                            global_logger_file_path,
                            HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                            train_loss_per_run_np, eval_loss_per_run_np
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
                                np.max(NDCGs_per_epoch_np, axis=0)[15], np.max(AUC_per_epoch_np, axis=0)[0],
                                train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
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
                                np.max(AUC_per_epoch_np, axis=0)[0],
                                train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                        )
                        instantwrite(logger_file)

                        print("GPU Usage after each run")
                        gpu_usage()

                        del model, optimizer, loss, cf_loss_per_batch, rec_metrics, dataloader
                        clearcache()

                    print(
                        'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5],
                            HRs_per_run_np.mean(axis=0)[10],
                            HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                            NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                            NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                            train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                        )
                    )
                    logger_file.write(
                        'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5],
                            HRs_per_run_np.mean(axis=0)[10],
                            HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                            NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                            NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                            train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                        )
                    )
                    instantwrite(logger_file)


if __name__ == '__main__':
    solver = HetRecSolver(HetRecRecsysModel, dataset_args, model_args, train_args)
    solver.run()
