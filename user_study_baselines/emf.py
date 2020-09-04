import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from collections import Counter
import os


SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DS = 'Movielens1m'
DEVICE = 'cuda:2'
EMBEDDING_DIM = 30
BATCH_SIZE = 1024
EPOCHS = 100
BETA = 0.01
LAMBDA = 0.1
LR = 0.001
K = 10
THETA = 0.0

embedding_file = 'checkpoint/weights/{}/EMF/embedding.pkl'.format(DS)

# Import rating
ratings_df = pd.read_csv('checkpoint/data/{}/processed/ratings.csv'.format(DS), sep=';')
ratings_df = ratings_df.reindex(np.random.permutation(ratings_df.index))
############### Uncomment when no filter out movies needed ###############
uids = np.sort(ratings_df.uid.unique())
iids = np.sort(ratings_df.iid.unique())
###########################################################################

########## Uncomment when Need filter out movies needed ###############
# movies = pd.read_csv('data/movies.csv', sep=';')
# movies = movies[movies.year > 1990]
# ratings = ratings[ratings.iid.isin(movies.iid)]
# # Reindex uid
# unique_uids = np.sort(ratings.uid.unique()).astype(np.int)
# uids = np.arange(unique_uids.shape[0]).astype(np.int)
# raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(unique_uids, uids)}
# ratings['uid'] = np.array([raw_uid2uid[raw_uid] for raw_uid in ratings.uid], dtype=np.int)
# # Reindex iid
# unique_iids = np.sort(ratings.iid.unique()).astype(np.int)
# iids = np.arange(unique_iids.shape[0]).astype(np.int)
# raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(unique_iids, iids)}
# ratings['iid'] = np.array([raw_iid2iid[raw_iid] for raw_iid in ratings.iid], dtype=np.int)
##########################################################################################

num_users = ratings_df.uid.unique().shape[0]
num_items = ratings_df.iid.unique().shape[0]
ALL_USER_INDICES_T = torch.arange(num_users, dtype=torch.long, device=DEVICE)
print(num_users, num_items, ratings_df.shape[0])

user_embedding = torch.nn.Embedding(num_users, EMBEDDING_DIM).to(DEVICE)
item_embedding = torch.nn.Embedding(num_items, EMBEDDING_DIM).to(DEVICE)


if not os.path.exists(embedding_file):
    print('No embedding file found! Initialization done.')
else:
    with open(embedding_file, mode='rb+') as f:
        checkpoint = torch.load(f, map_location=DEVICE)
    user_embedding.weight = checkpoint['user_embedding']
    item_embedding.weight = checkpoint['item_embedding']
    print('Embedding loaded!')

data = ratings_df[['uid', 'iid', 'rating']].to_numpy()
del ratings_df
train_data = torch.from_numpy(data[:int(data.shape[0] * 0.9)]).to(DEVICE)
test_data = torch.from_numpy(data[int(data.shape[0] * 0.9):]).to(DEVICE)

edge_index = train_data[:, :2].T


def compute_W():
    W = []
    all_neighbours = []
    pbar = tqdm.tqdm(uids, total=uids.shape[0])
    for uid in pbar:
        pbar.set_description('Computing W matrix...')
        batch_user_dists = torch.sum(user_embedding(ALL_USER_INDICES_T) * user_embedding(torch.tensor([uid], device=DEVICE)), dim=-1)
        neighbours = torch.topk(batch_user_dists, k=K + 1, largest=True)[1][:1]
        all_neighbours.append(neighbours.cpu().numpy())
        seen_iids = []
        for neighbour in neighbours:
            seen_iids.append(edge_index[1, edge_index[0, :] == neighbour])
        iid_dict = Counter(torch.cat(seen_iids).cpu().numpy())
        w = np.array([iid_dict.get(iid, 0) / K for iid in iids])  # theta == 0
        W.append(w)
    all_neighbours = np.array(all_neighbours)
    W = torch.tensor(W, device=DEVICE)

    return W, all_neighbours


opt = torch.optim.SGD(lr=LR, params=[user_embedding.weight, item_embedding.weight])

for epoch in range(EPOCHS):
    W, all_neighbours = compute_W()
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    losses = []
    for batch in pbar:
        uid_batch = batch[:, 0].long().to(DEVICE)
        iid_batch = batch[:, 1].long().to(DEVICE)
        rating_batch = batch[:, 2]

        user_repr = user_embedding(uid_batch)
        item_repr = item_embedding(iid_batch)

        cf_loss = rating_batch - torch.sum(user_repr * item_repr, dim=-1)
        cf_loss = torch.sum(cf_loss * cf_loss)

        reg_loss = 0.5 * BETA * torch.sum(torch.sum(user_repr * user_repr, dim=-1) + torch.sum(item_repr * item_repr, dim=-1))

        w = W[uid_batch, iid_batch]
        exp_loss = 0.5 * LAMBDA * torch.sum(torch.sum((user_repr - item_repr) * (user_repr - item_repr), dim=-1) * w)

        loss = cf_loss + reg_loss + exp_loss
        # loss = cf_loss + reg_loss
        losses.append(loss.detach().cpu().item())
        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_description('loss: {:.4f}'.format(np.mean(losses)))

    with torch.no_grad():
        mses = []
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
        pbar = tqdm.tqdm(test_dataloader, total=len(test_dataloader))
        for batch in pbar:
            uid_batch = batch[:, 0].long().to(DEVICE)
            iid_batch = batch[:, 1].long().to(DEVICE)
            rating_batch = batch[:, 2]

            user_repr = user_embedding(uid_batch)
            item_repr = item_embedding(iid_batch)

            mses.append(torch.nn.MSELoss()(torch.sum(user_repr * item_repr, dim=-1).view(-1), rating_batch).detach().cpu().item())
            pbar.set_description('MSE: {:.4f}'.format(np.mean(mses)))

# Save files
states = {'user_embedding': user_embedding, 'item_embedding': item_embedding, 'all_neighbours': all_neighbours}
with open(embedding_file, mode='wb+') as f:
    torch.save(states, f)
print('Embedding saved!')
