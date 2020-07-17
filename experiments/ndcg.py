import os
import sys
sys.path.append('..')
from graph_recsys_benchmark.utils import *
MODEL = 'MPAGAT'
DATASET = 'Yelp'
CF_LOSS_TYPE = 'BPR'
# model_args = { 'model_type': 'Graph', 'if_use_features': False, 'emb_dim': 62, 'hidden_size': 16, 'repr_dim': 62, 'dropout': 0, 'num_heads': 1, 'meta_path_steps':[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], 'aggr':'att' }
model_args = {'model_type': 'Graph', 'if_use_features': False, 'emb_dim': 62, 'hidden_size': 62, 'repr_dim': 16, 'dropout': 0, 'num_heads': 1, 'meta_path_steps': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'aggr': 'att'}
_, _, logger_folder = \
    get_folder_path(model=MODEL, dataset=DATASET, loss_type=CF_LOSS_TYPE)
global_logger_file_path = os.path.join(logger_folder, str(model_args), 'global_logger.pkl')
HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
    load_global_logger(global_logger_file_path)
print(HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run)