import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from collections import Counter
import pickle
import os

ds = 'Movielenslatest-small'
device = 'cuda:5'
embedding_dim = 30
batch_size = 1024
epochs = 100
beta = 0.01
lambd = 0.1
lr = 0.001
k = 10
theta = 0.0

embedding_file = 'checkpoint/weights/{}/EMF/embedding.pkl'.format(ds)
misc_folder = 'checkpoint/weights/{}/EMF'.format(ds)
misc_file = 'checkpoint/weights/{}/EMF/misc.pkl'.format(ds)
# Import rating
ratings = pd.read_csv('checkpoint/data/{}/processed/ratings.csv'.format(ds), sep=';')
############### Uncomment when no filter out movies needed ###############
uids = np.sort(ratings.uid.unique())
iids = np.sort(ratings.iid.unique())
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

num_users = ratings.uid.unique().shape[0]
num_items = ratings.iid.unique().shape[0]
print(num_users, num_items, ratings.shape[0])

user_embedding = torch.nn.Embedding(num_users, embedding_dim, max_norm=1).to(device).weight
item_embedding = torch.nn.Embedding(num_items, embedding_dim, max_norm=1).to(device).weight
if not os.path.exists(embedding_file):
    print('No embedding file found! Initialization done.')
else:
    with open(embedding_file, mode='rb+') as f:
        checkpoint = torch.load(f, map_location=device)
    user_embedding = checkpoint['user_embedding']
    item_embedding = checkpoint['item_embedding']
    print('Embedding loaded!')

data = ratings[['uid', 'iid', 'rating']].sample(frac=1).to_numpy()
del ratings
train_data = torch.from_numpy(data[:int(data.shape[0] * 0.9)]).to(device)
test_data = torch.from_numpy(data[int(data.shape[0] * 0.9):]).to(device)

edge_index = train_data[:, :2].T

if not os.path.exists(misc_file):
    os.makedirs(misc_folder, exist_ok=True)

    print('No W file found! Initializing...')
    W = []
    all_neighbours = []
    pbar = tqdm.tqdm(uids, total=uids.shape[0])
    for uid in pbar:
        batch_user_dists = torch.sum(user_embedding * user_embedding[uid].unsqueeze(0), dim=-1)
        _, neighbours = torch.topk(batch_user_dists, k=k, largest=False)
        all_neighbours.append(neighbours.cpu().numpy())
        seen_iids = []
        for neighbour in neighbours:
            seen_iids.append(edge_index[1, edge_index[0, :] == neighbour])
        iid_dict = Counter(torch.cat(seen_iids).cpu().numpy())
        w = np.array([iid_dict.get(iid, 0) / k for iid in iids])  # theta == 0
        W.append(w)
    all_neighbours = np.array(all_neighbours)
    W = torch.tensor(W).to(device)
    print('Initialization done.')

    with open(misc_file, mode='wb') as f:
        pickle.dump([W.cpu().numpy(), all_neighbours], f)
    print('W file saved!')
else:
    with open(misc_file, mode='rb') as f:
        [W, all_neighbours] = pickle.load(f)
        W = torch.from_numpy(W).to(device)
        print('W file loaded!')


opt = torch.optim.SGD(lr=lr, params=[user_embedding, item_embedding])

for epoch in range(epochs):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    losses = []
    for batch in pbar:
        uid_batch = batch[:, 0].long().to(device)
        iid_batch = batch[:, 1].long().to(device)
        rating_batch = batch[:, 2]

        user_repr = user_embedding[uid_batch]
        item_repr = item_embedding[iid_batch]

        cf_loss = rating_batch - torch.sum(user_repr * item_repr, dim=-1).view(-1)
        cf_loss = torch.sum(cf_loss * cf_loss)

        reg_loss = 0.5 * beta * torch.sum(torch.sum(user_repr * user_repr, dim=-1) + torch.sum(item_repr * item_repr, dim=-1))

        w = W[uid_batch, iid_batch]
        exp_loss = 0.5 * lambd * torch.sum(torch.sum((user_repr - item_repr) * (user_repr - item_repr), dim=-1) * w)

        loss = cf_loss + reg_loss + exp_loss
        # loss = cf_loss + reg_loss
        losses.append(loss.detach().cpu().item())
        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_description('loss: {:.4f}'.format(np.mean(losses)))

    with torch.no_grad():
        mses = []
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        pbar = tqdm.tqdm(test_dataloader, total=len(test_dataloader))
        for batch in pbar:
            uid_batch = batch[:, 0].long().to(device)
            iid_batch = batch[:, 1].long().to(device)
            rating_batch = batch[:, 2]

            user_repr = user_embedding[uid_batch]
            item_repr = item_embedding[iid_batch]

            mses.append(torch.nn.MSELoss()(torch.sum(user_repr * item_repr, dim=-1).view(-1), rating_batch).detach().cpu().item())
            pbar.set_description('MSE: {:.4f}'.format(np.mean(mses)))

# Save files
states = {'user_embedding': user_embedding, 'item_embedding': item_embedding}
with open(embedding_file, mode='wb+') as f:
    torch.save(states, f)
print('Embedding saved!')
