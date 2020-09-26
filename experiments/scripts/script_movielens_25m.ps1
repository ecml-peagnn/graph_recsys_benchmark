# Movielens25m

# GCN
# --entity_aware=false --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=1 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=2 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26

# GCNInner
# --entity_aware=false --dropout=0.5
python3 gcn_inner_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=16 --repr_dim=8 --hidden_size=32 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gcn_inner_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=16 --repr_dim=8 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=2 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26


# GAT
# --entity_aware=false --dropout=0.5
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=1 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# GATInner
# --entity_aware=false --dropout=0.5
python3 gat_inner_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=16 --repr_dim=8 --hidden_size=32 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=1 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gat_inner_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=1 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEAGCN
# --entity_aware=false --dropout=0
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=3 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26

# PEAGCNJK
# --entity_aware=false --dropout=0
python3 peagcn_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --jump_mode=cat --channel_aggr=att --dropout=0 --emb_dim=16 --repr_dim=8 --hidden_size=32 --meta_path_steps=2,2,2,2,2,2,2,2 --entity_aware_coff=0.01 --init_eval=false --gpu_idx=3 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagcn_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=true --jump_mode=cat --channel_aggr=att --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=3 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEAGAT
# --entity_aware=false --dropout=0
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# PEAGATInner
# --entity_aware=false --dropout=0
python3 peagat_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --jump_mode=cat --channel_aggr=att --dropout=0 --emb_dim=16 --repr_dim=8 --hidden_size=32 --meta_path_steps=2,2,2,2,2,2,2,2 --entity_aware_coff=0.01 --init_eval=false --gpu_idx=4 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagat_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# KGAT
# --entity_aware=false --dropout=0.1
python3 kgat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.1 --emb_dim=32 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# NCF
# --entity_aware=false
python3 ncf_solver_bce.py --dataset=Movielens --dataset_name=25m --num_core=10 --sampling_strategy=random --entity_aware=false --factor_num=8 --num_layers=4 --dropout=0.5 --init_eval=false --gpu_idx=2 --runs=2 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PinSAGE
# --entity_aware=false --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=16 --repr_dim=8 --hidden_size=32 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=2 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=1 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEASage
# --entity_aware=false --dropout=0
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

# PEASageJK
# --entity_aware=false --dropout=0
python3 peasage_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=10 --num_feat_core=10 --sampling_strategy=random --entity_aware=false --jump_mode=cat --channel_aggr=att --dropout=0 --emb_dim=16 --repr_dim=8 --hidden_size=32 --meta_path_steps=2,2,2,2,2,2,2,2 --entity_aware_coff=0.01 --init_eval=false --gpu_idx=5 --runs=5 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peasage_jk_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# Node2Vec
# --entity_aware=false
python3 node2vec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --emb_dim=32 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26


# Metapath2Vec
# --entity_aware=false
python3 metapath2vec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --emb_dim=32 --init_eval=false --gpu_idx=0 --runs=3 --epochs=30 --batch_size=4096 --save_every_epoch=26

