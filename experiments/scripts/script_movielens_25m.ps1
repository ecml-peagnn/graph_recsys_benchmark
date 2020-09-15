# Movielens25m

# GCN
# --entity_aware=false --dropout=0
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# GAT
# --entity_aware=false --dropout=0
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 gat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEAGCN
# --entity_aware=false --dropout=0
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 peagcn_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEAGAT
# --entity_aware=false --dropout=0
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 peagat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# KGAT
# --entity_aware=false --dropout=0.1
python3 kgat_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.1 --emb_dim=32 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# NCF
# --entity_aware=false
python3 ncf_solver_bce.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PinSAGE
# --entity_aware=false --dropout=0
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# PEASAGE
# --entity_aware=false --dropout=0
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=false --dropout=0.5
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

# --entity_aware=true --dropout=0.5
python3 peasage_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=32 --repr_dim=8 --hidden_size=16 --meta_path_steps=2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# Node2Vec
# --entity_aware=false
python3 node2vec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --emb_dim=32 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26


# Metapath2Vec
# --entity_aware=false
python3 metapath2vec_solver_bpr.py --dataset=Movielens --dataset_name=25m --num_core=20 --sampling_strategy=random --entity_aware=false --emb_dim=32 --init_eval=false --gpu_idx=0 --epochs=30 --batch_size=4096 --save_every_epoch=26

