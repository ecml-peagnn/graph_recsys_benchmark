# Yelp

# GCN
# --entity_aware=false --dropout=0
python3 gcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 gcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 gcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# GAT
# --entity_aware=false --dropout=0
python3 gat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 gat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 gat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 gat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# PEAGCN
# --entity_aware=false --dropout=0
python3 peagcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 peagcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 peagcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 peagcn_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# PEAGAT
# --entity_aware=false --dropout=0
python3 peagat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 peagat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 peagat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 peagat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# KGAT
# --entity_aware=false --dropout=0.1
python3 kgat_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.1 --emb_dim=64 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# NCF
# --entity_aware=false
python3 ncf_solver_bce.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# PinSAGE
# --entity_aware=false --dropout=0
python3 pinsage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 pinsage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 pinsage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# PEASAGE
# --entity_aware=false --dropout=0
python3 peasage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=false --dropout=0.5
python3 peasage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0
python3 peasage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

# --entity_aware=true --dropout=0.5
python3 peasage_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=true --dropout=0.5 --emb_dim=64 --repr_dim=16 --hidden_size=64 --meta_path_steps=2,2,2,2,2,2,2,2,2,2,2 --entity_aware_coff=0.1 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# Node2Vec
# --entity_aware=false
python3 node2vec_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --emb_dim=64 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16


# Metapath2Vec
# --entity_aware=false
python3 metapath2vec_solver_bpr.py --dataset=Yelp --num_core=10 --sampling_strategy=random --entity_aware=false --emb_dim=64 --init_eval=false --gpu_idx=0 --epochs=20 --batch_size=1024 --save_every_epoch=16

