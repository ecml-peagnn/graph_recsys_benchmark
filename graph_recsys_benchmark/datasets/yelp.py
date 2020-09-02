import torch
from dateutil.parser import parser
from torch.utils.data import DataLoader
from os.path import join, isfile, isdir
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from iteration_utilities import unique_everseen
import tqdm
import pickle
import re
from shutil import copy

from .dataset import Dataset
from torch_geometric.data import extract_tar
from ..parser import parse_yelp


def reindex_df(business, user, reviewtip):
    """
    reindex business, user, reviewtip in case there are some values missing or duplicates in between
    :param business: pd.DataFrame
    :param user: pd.DataFrame
    :param reviewtip: pd.DataFrame
    :return: same
    """
    print('Reindexing dataframes...')
    unique_uids = user.user_id.unique()
    unique_iids = business.business_id.unique()

    num_users = unique_uids.shape[0]
    num_bus = unique_iids.shape[0]

    raw_uids = np.array(unique_uids, dtype=object)
    raw_iids = np.array(unique_iids, dtype=object)

    uids = np.arange(num_users)
    iids = np.arange(num_bus)

    user['user_id'] = uids
    business['business_id'] = iids

    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}

    review_uids = np.array(reviewtip.user_id, dtype=object)
    review_iids = np.array(reviewtip.business_id, dtype=object)
    review_uids = [raw_uid2uid[review_uid] for review_uid in review_uids]
    review_iids = [raw_iid2iid[review_iid] for review_iid in review_iids]
    reviewtip['user_id'] = review_uids
    reviewtip['business_id'] = review_iids

    print('Reindex done!')

    return business, user, reviewtip


def drop_infrequent_concept_from_str(df, concept_name):
    concept_strs = [concept_str if concept_str != None else '' for concept_str in df[concept_name]]
    duplicated_concept = [concept_str.split(', ') for concept_str in concept_strs]
    duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
    business_category_dict = Counter(duplicated_concept)
    del business_category_dict['']
    del business_category_dict['N/A']
    unique_concept = [k for k, v in business_category_dict.items() if
                      v >= 0.1 * np.max(list(business_category_dict.values()))]
    concept_strs = [
        ','.join([concept for concept in concept_str.split(', ') if concept in unique_concept])
        for concept_str in concept_strs
    ]
    df[concept_name] = concept_strs
    return df


def generate_graph_data(
        users, items, reviewtip
):
    """
    Entitiy node include (business, user, reviewtip)
    """

    def get_concept_num_from_str(df, concept_name):
        if (concept_name == 'friends'):
            concept_strs = [concept_str.split(', ') for concept_str in df[concept_name]]
            concepts = set(itertools.chain.from_iterable(concept_strs))
            unique_uids = list(df.user_id.unique())
            concepts = list(set(concepts).difference(unique_uids))
        else:
            concept_strs = [concept_str.split(',') for concept_str in df[concept_name]]
            concepts = set(itertools.chain.from_iterable(concept_strs))
            concepts.remove('')
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Discretized user reviewcount  #########################
    userreviewcount = users.review_count.to_numpy().astype(np.int)
    min_userreviewcount = min(userreviewcount)
    max_userreviewcount = max(userreviewcount)
    num_userreviewcount = (max_userreviewcount - min_userreviewcount) // 100
    discretized_userreviewcounts = [min_userreviewcount + i * 100 for i in range(num_userreviewcount + 1)]
    for i, discretized_userreviewcount in enumerate(discretized_userreviewcounts):
        if i != len(discretized_userreviewcounts) - 1:
            userreviewcount[(discretized_userreviewcount <= userreviewcount) & (
                        userreviewcount < discretized_userreviewcounts[i + 1])] = str(discretized_userreviewcount)
        else:
            userreviewcount[discretized_userreviewcount <= userreviewcount] = str(discretized_userreviewcount)
    users['review_count'] = userreviewcount

    #########################  Discretized user friendcount  #########################
    userfriends_count = users.friends_count.to_numpy().astype(np.int)
    min_userfriends_count = min(userfriends_count)
    max_userfriends_count = max(userfriends_count)
    num_userfriends_count = (max_userfriends_count - min_userfriends_count) // 500
    discretized_userfriends_counts = [min_userfriends_count + i * 500 for i in range(num_userfriends_count + 1)]
    for i, discretized_friend_count in enumerate(discretized_userfriends_counts):
        if i != len(discretized_userfriends_counts) - 1:
            userfriends_count[
                (discretized_friend_count <= userfriends_count) & (
                            userfriends_count < discretized_userfriends_counts[i + 1])] = str(
                discretized_friend_count)
        else:
            userfriends_count[discretized_friend_count <= userfriends_count] = str(discretized_friend_count)
    users['friends_count'] = userfriends_count

    #########################  Discretized user fans  #########################
    userfans = users.fans.to_numpy().astype(np.int)
    min_userfans = min(userfans)
    max_userfans = max(userfans)
    num_userfans = (max_userfans - min_userfans) // 100
    discretized_userfans = [min_userfans + i * 100 for i in range(num_userfans + 1)]
    for i, discretized_userfan in enumerate(discretized_userfans):
        if i != len(discretized_userfans) - 1:
            userfans[
                (discretized_userfan <= userfans) & (
                        userfans < discretized_userfans[i + 1])] = str(
                discretized_userfan)
        else:
            userfans[discretized_userfan <= userfans] = str(discretized_userfan)
    users['fans'] = userfans

    #########################  Discretized user stars  #########################
    userstars = users.average_stars.to_numpy().astype(np.float)
    min_userstars = min(userstars)
    max_userstars = max(userstars)
    num_userstars = ((max_userstars - min_userstars) // 0.5).astype(np.int)
    discretized_userstars = [min_userstars + i * 0.5 for i in range(num_userstars + 1)]
    for i, discretized_userstar in enumerate(discretized_userstars):
        if i != len(discretized_userstars) - 1:
            userstars[
                (discretized_userstar <= userstars) & (
                        userstars < discretized_userstars[i + 1])] = str(
                discretized_userstar)
        else:
            userstars[discretized_userstar <= userstars] = str(discretized_userstar)
    users['average_stars'] = userstars
    #########################  Discretized item reviewcount  #########################
    itemreviewcount = items.review_count.to_numpy().astype(np.int)
    min_itemreviewcount = min(itemreviewcount)
    max_itemreviewcount = max(itemreviewcount)
    num_itemreviewcount = (max_itemreviewcount - min_itemreviewcount) // 500
    discretized_itemreviewcounts = [min_itemreviewcount + i * 500 for i in range(num_itemreviewcount + 1)]
    for i, discretized_itemreviewcount in enumerate(discretized_itemreviewcounts):
        if i != len(discretized_itemreviewcounts) - 1:
            itemreviewcount[(discretized_itemreviewcount <= itemreviewcount) & (
                    itemreviewcount < discretized_itemreviewcounts[i + 1])] = str(discretized_itemreviewcount)
        else:
            itemreviewcount[discretized_itemreviewcount <= itemreviewcount] = str(discretized_itemreviewcount)
    items['review_count'] = itemreviewcount

    #########################  Discretized item checkincount  #########################
    itemcheckincount = items.checkin_count.to_numpy().astype(np.int)
    min_itemcheckincount = min(itemcheckincount)
    max_itemcheckincount = max(itemcheckincount)
    num_itemcheckincount = (max_itemcheckincount - min_itemcheckincount) // 1000
    discretized_itemcheckincounts = [min_itemcheckincount + i * 1000 for i in range(num_itemcheckincount + 1)]
    for i, discretized_itemcheckincount in enumerate(discretized_itemcheckincounts):
        if i != len(discretized_itemcheckincounts) - 1:
            itemcheckincount[(discretized_itemcheckincount <= itemcheckincount) & (
                    itemcheckincount < discretized_itemcheckincounts[i + 1])] = str(discretized_itemcheckincount)
        else:
            itemcheckincount[discretized_itemcheckincount <= itemcheckincount] = str(discretized_itemcheckincount)
    items['checkin_count'] = itemcheckincount

    #########################  Define entities  #########################
    unique_uids = list(np.sort(reviewtip.user_id.unique()))
    num_users = len(unique_uids)

    unique_iids = list(np.sort(reviewtip.business_id.unique()))
    num_items = len(unique_iids)

    unique_user_reviewcount = list(users.review_count.unique())
    num_user_reviewcount = len(unique_user_reviewcount)

    unique_user_friendcount = list(users.friends_count.unique())
    num_user_friendcount = len(unique_user_friendcount)

    unique_user_fans = list(users.fans.unique())
    num_user_fans = len(unique_user_fans)

    unique_user_stars = list(users.average_stars.unique())
    num_user_stars = len(unique_user_stars)

    unique_item_stars = list(items.stars.unique())
    num_item_stars = len(unique_item_stars)

    unique_item_reviewcount = list(items.review_count.unique())
    num_item_reviewcount = len(unique_item_reviewcount)

    unique_item_attributes, num_item_attributes = get_concept_num_from_str(items, 'attributes')
    unique_item_categories, num_item_categories = get_concept_num_from_str(items, 'categories')

    unique_item_checkincount = list(items.checkin_count.unique())
    num_item_checkincount = len(unique_item_checkincount)

    #########################  Create dataset property dict  #########################
    dataset_property_dict = {}
    dataset_property_dict['unique_uids'] = unique_uids
    dataset_property_dict['num_users'] = num_users
    dataset_property_dict['unique_iids'] = unique_iids
    dataset_property_dict['num_items'] = num_items
    dataset_property_dict['unique_user_reviewcount'] = unique_user_reviewcount
    dataset_property_dict['num_user_reviewcount'] = num_user_reviewcount
    dataset_property_dict['unique_user_friendcount'] = unique_user_friendcount
    dataset_property_dict['num_user_friendcount'] = num_user_friendcount
    dataset_property_dict['unique_user_fans'] = unique_user_fans
    dataset_property_dict['num_user_fans'] = num_user_fans
    dataset_property_dict['unique_user_stars'] = unique_user_stars
    dataset_property_dict['num_user_stars'] = num_user_stars
    dataset_property_dict['unique_item_stars'] = unique_item_stars
    dataset_property_dict['num_item_stars'] = num_item_stars
    dataset_property_dict['unique_item_reviewcount'] = unique_item_reviewcount
    dataset_property_dict['num_item_reviewcount'] = num_item_reviewcount
    dataset_property_dict['unique_item_attributes'] = unique_item_attributes
    dataset_property_dict['num_item_attributes'] = num_item_attributes
    dataset_property_dict['unique_item_categories'] = unique_item_categories
    dataset_property_dict['num_item_categories'] = num_item_categories
    dataset_property_dict['unique_item_checkincount'] = unique_item_checkincount
    dataset_property_dict['num_item_checkincount'] = num_item_checkincount

    #########################  Define number of entities  #########################
    num_nodes = num_users + num_items + num_user_reviewcount + num_user_friendcount + \
                num_user_fans + num_user_stars + num_item_stars + num_item_reviewcount + \
                num_item_attributes + num_item_categories + num_item_checkincount

    num_node_types = 11
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types
    types = ['users', 'items', 'userreviewcount', 'userfriendcount', 'userfans', 'userstars',
             'itemstars', 'itemreviewcount', 'itemattributes', 'itemcategories', 'itemcheckincount']
    num_nodes_dict = {'users': num_users, 'items': num_items, 'userreviewcount': num_user_reviewcount,
                      'userfriendcount': num_user_friendcount, 'userfans': num_user_fans, 'userstars': num_user_stars,
                      'itemstars': num_item_stars, 'itemreviewcount': num_item_reviewcount,
                      'itemattributes': num_item_attributes,
                      'itemcategories': num_item_categories, 'itemcheckincount': num_item_checkincount,
                      }
    #########################  Define entities to node id map  #########################
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs['users'] = acc
    uid2nid = {uid: i + acc for i, uid in enumerate(users['user_id'])}
    for i, uid in enumerate(users['user_id']):
        nid2e_dict[i + acc] = ('uid', uid)
    acc += num_users
    type_accs['items'] = acc
    iid2nid = {iid: i + acc for i, iid in enumerate(items['business_id'])}
    for i, iid in enumerate(items['business_id']):
        nid2e_dict[i + acc] = ('iid', iid)
    acc += num_items
    type_accs['userreviewcount'] = acc
    userreviewcount2nid = {userreviewcount: i + acc for i, userreviewcount in enumerate(unique_user_reviewcount)}
    for i, userreviewcount in enumerate(unique_user_reviewcount):
        nid2e_dict[i + acc] = ('userreviewcount', userreviewcount)
    acc += num_user_reviewcount
    type_accs['userfriendcount'] = acc
    userfriendcount2nid = {userfriendcount: i + acc for i, userfriendcount in enumerate(unique_user_friendcount)}
    for i, userfriendcount in enumerate(unique_user_friendcount):
        nid2e_dict[i + acc] = ('userfriendcount', userfriendcount)
    acc += num_user_friendcount
    type_accs['userfans'] = acc
    userfans2nid = {userfans: i + acc for i, userfans in enumerate(unique_user_fans)}
    for i, userfans in enumerate(unique_user_fans):
        nid2e_dict[i + acc] = ('userfans', userfans)
    acc += num_user_fans
    type_accs['userstars'] = acc
    userstars2nid = {userstars: i + acc for i, userstars in enumerate(unique_user_stars)}
    for i, userstars in enumerate(unique_user_stars):
        nid2e_dict[i + acc] = ('userstars', userstars)
    acc += num_user_stars
    type_accs['itemstars'] = acc
    itemstars2nid = {itemstars: i + acc for i, itemstars in enumerate(unique_item_stars)}
    for i, itemstars in enumerate(unique_item_stars):
        nid2e_dict[i + acc] = ('itemstars', itemstars)
    acc += num_item_stars
    type_accs['itemreviewcount'] = acc
    itemreviewcount2nid = {itemreviewcount: i + acc for i, itemreviewcount in enumerate(unique_item_reviewcount)}
    for i, itemreviewcount in enumerate(unique_item_reviewcount):
        nid2e_dict[i + acc] = ('itemreviewcount', itemreviewcount)
    acc += num_item_reviewcount
    type_accs['itemattributes'] = acc
    itemattributes2nid = {itemattributes: i + acc for i, itemattributes in enumerate(unique_item_attributes)}
    for i, itemattributes in enumerate(unique_item_attributes):
        nid2e_dict[i + acc] = ('itemattributes', itemattributes)
    acc += num_item_attributes
    type_accs['itemcategories'] = acc
    itemcategories2nid = {itemcategories: i + acc for i, itemcategories in enumerate(unique_item_categories)}
    for i, itemcategories in enumerate(unique_item_categories):
        nid2e_dict[i + acc] = ('itemcategories', itemcategories)
    acc += num_item_categories
    type_accs['itemcheckincount'] = acc
    itemcheckincount2nid = {itemcheckincount: i + acc for i, itemcheckincount in enumerate(unique_item_checkincount)}
    for i, itemcheckincount in enumerate(unique_item_checkincount):
        nid2e_dict[i + acc] = ('itemcheckincount', itemcheckincount)

    e2nid_dict = {'uid': uid2nid, 'iid': iid2nid,
                  'userreviewcount': userreviewcount2nid, 'userfriendcount': userfriendcount2nid,
                  'userfans': userfans2nid, 'userstars': userstars2nid,
                  'itemstars': itemstars2nid, 'itemreviewcount': itemreviewcount2nid,
                  'itemattributes': itemattributes2nid,
                  'itemcategories': itemcategories2nid, 'itemcheckincount': itemcheckincount2nid
                  }
    dataset_property_dict['e2nid_dict'] = e2nid_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print('Creating user property edges...')
    u_nids = [e2nid_dict['uid'][uid] for uid in users.user_id]
    userreviewcount_nids = [e2nid_dict['userreviewcount'][userreviewcount] for userreviewcount in users.review_count]
    reviewcount2user_edge_index_np = np.vstack((np.array(userreviewcount_nids), np.array(u_nids)))
    userfriendcount_nids = [e2nid_dict['userfriendcount'][userfriendcount] for userfriendcount in users.friends_count]
    friendcount2user_edge_index_np = np.vstack((np.array(userfriendcount_nids), np.array(u_nids)))
    userfans_nids = [e2nid_dict['userfans'][userfans] for userfans in users.fans]
    fans2user_edge_index_np = np.vstack((np.array(userfans_nids), np.array(u_nids)))
    userstars_nids = [e2nid_dict['userstars'][userstars] for userstars in users.average_stars]
    stars2user_edge_index_np = np.vstack((np.array(userstars_nids), np.array(u_nids)))

    edge_index_nps['reviewcount2user'] = reviewcount2user_edge_index_np
    edge_index_nps['friendcount2user'] = friendcount2user_edge_index_np
    edge_index_nps['fans2user'] = fans2user_edge_index_np
    edge_index_nps['stars2user'] = stars2user_edge_index_np

    print('Creating item property edges...')
    i_nids = [e2nid_dict['iid'][iid] for iid in items.business_id]
    itemstars_nids = [e2nid_dict['itemstars'][itemstars] for itemstars in items.stars]
    stars2item_edge_index_np = np.vstack((np.array(itemstars_nids), np.array(i_nids)))
    itemreviewcount_nids = [e2nid_dict['itemreviewcount'][itemreviewcount] for itemreviewcount in items.review_count]
    reviewcount2item_edge_index_np = np.vstack((np.array(itemreviewcount_nids), np.array(i_nids)))

    attributes_list = [
        [attribute for attribute in attributes.split(',') if attribute != '']
        for attributes in items.attributes
    ]
    itemattributes_nids = [[e2nid_dict['itemattributes'][attribute] for attribute in attributes] for attributes in
                           attributes_list]
    itemattributes_nids = list(itertools.chain.from_iterable(itemattributes_nids))
    a_i_nids = [[i_nid for _ in range(len(attributes_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    a_i_nids = list(itertools.chain.from_iterable(a_i_nids))
    attributes2item_edge_index_np = np.vstack((np.array(itemattributes_nids), np.array(a_i_nids)))

    categories_list = [
        [category for category in categories.split(',') if category != '']
        for categories in items.categories
    ]
    itemcategories_nids = [[e2nid_dict['itemcategories'][category] for category in categories] for categories in
                           categories_list]
    itemcategories_nids = list(itertools.chain.from_iterable(itemcategories_nids))
    c_i_nids = [[i_nid for _ in range(len(categories_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    c_i_nids = list(itertools.chain.from_iterable(c_i_nids))
    categories2item_edge_index_np = np.vstack((np.array(itemcategories_nids), np.array(c_i_nids)))

    itemcheckincount_nids = [e2nid_dict['itemcheckincount'][itemcheckincount] for itemcheckincount in
                             items.checkin_count]
    checkincount2item_edge_index_np = np.vstack((np.array(itemcheckincount_nids), np.array(i_nids)))

    edge_index_nps['stars2item'] = stars2item_edge_index_np
    edge_index_nps['reviewcount2item'] = reviewcount2item_edge_index_np
    edge_index_nps['attributes2item'] = attributes2item_edge_index_np
    edge_index_nps['categories2item'] = categories2item_edge_index_np
    edge_index_nps['checkincount2item'] = checkincount2item_edge_index_np

    print('Creating reviewtip property edges...')
    test_pos_unid_inid_map, neg_unid_inid_map = {}, {}

    user2item_edge_index_np = np.zeros((2, 0))
    pbar = tqdm.tqdm(unique_uids, total=len(unique_uids))
    sorted_reviewtip = reviewtip.sort_values(['bus_count', 'user_count'])
    for uid in pbar:
        pbar.set_description('Creating the edges for the user {}'.format(uid))
        uid_reviewtip = sorted_reviewtip[sorted_reviewtip.user_id == uid]
        uid_iids = uid_reviewtip.business_id.to_numpy()

        unid = e2nid_dict['uid'][uid]
        train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
        train_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in train_pos_uid_iids]
        test_pos_uid_iids = list(uid_iids[-1:])
        test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]
        neg_uid_iids = list(set(unique_iids) - set(uid_iids))
        neg_uid_inids = [e2nid_dict['iid'][iid] for iid in neg_uid_iids]

        test_pos_unid_inid_map[unid] = test_pos_uid_inids
        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack([user2item_edge_index_np, unid_user2item_edge_index_np])

    edge_index_nps['user2item'] = user2item_edge_index_np

    print('missing iids:',
          np.setdiff1d(np.unique(items.business_id), np.unique(user2item_edge_index_np[1].astype(int) - num_users)))

    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['test_pos_unid_inid_map'], dataset_property_dict['neg_unid_inid_map'] = \
        test_pos_unid_inid_map, neg_unid_inid_map

    print('Building edge type map...')
    edge_type_dict = {edge_type: edge_type_idx for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))}
    dataset_property_dict['edge_type_dict'] = edge_type_dict
    dataset_property_dict['num_edge_types'] = len(list(edge_index_nps.keys()))

    print('Building the item occurrence map...')
    item_nid_occs = {}
    for iid in items.business_id:
        item_nid_occs[e2nid_dict['iid'][iid]] = reviewtip[reviewtip.business_id == iid].iloc[0]['bus_count']

    dataset_property_dict['item_nid_occs'] = item_nid_occs

    # New functionality for pytorch geometric like dataset
    dataset_property_dict['types'] = types
    dataset_property_dict['num_nodes_dict'] = num_nodes_dict
    dataset_property_dict['type_accs'] = type_accs

    return dataset_property_dict


class Yelp(Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):

        self.type = kwargs['type']
        assert self.type in ['hete', 'bipartite']
        self.num_core = kwargs['num_core']
        self.entity_aware = kwargs['entity_aware']
        self.num_negative_samples = kwargs['num_negative_samples']
        self.sampling_strategy = kwargs['sampling_strategy']
        self.cf_loss_type = kwargs['cf_loss_type']
        self.kg_loss_type = kwargs.get('kg_loss_type', None)
        self.dataset = kwargs['dataset']

        super(Yelp, self).__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], 'rb') as f:  # Read the class property
            dataset_property_dict = pickle.load(f)
        for k, v in dataset_property_dict.items():
            self[k] = v

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'yelp_dataset.tar'

    @property
    def processed_file_names(self):
        return ['dataset{}.pkl'.format(self.build_suffix())]

    @property
    def untar_file_path(self):
        return join(self.raw_dir, 'yelp_dataset')

    def download(self):
        if isdir(self.untar_file_path):
            return

        tar_file_path = join(Path(__file__).parents[2], 'datasets', self.dataset, self.raw_file_names)
        copy(tar_file_path, self.raw_dir)
        extract_tar(join(self.raw_dir, self.raw_file_names), self.untar_file_path)

    def process(self):
        # parser files
        if isfile(join(self.processed_dir, 'business.pkl')) and isfile(join(self.processed_dir, 'user.pkl')) and isfile(
                join(self.processed_dir, 'reviewtip.pkl')):
            print('Read data frame!')
            business = pd.read_pickle(join(self.processed_dir, 'business.pkl'))
            user = pd.read_pickle(join(self.processed_dir, 'user.pkl'))
            reviewtip = pd.read_pickle(join(self.processed_dir, 'reviewtip.pkl'))
            business = business.fillna('')
            user = user.fillna('')
            reviewtip = reviewtip.fillna('')

        else:
            print('Data frame not found in {}! Read from raw data!'.format(self.processed_dir))
            business, user, review, tip, checkin = parse_yelp(self.untar_file_path)

            print('Preprocessing...')
            # Extract business hours
            hours = []
            for hr in business['hours']:
                hours.append(hr) if hr != None else hours.append({})

            df_hours = (
                pd.DataFrame(hours)
                    .fillna(False))

            # Replacing all times with True
            df_hours.where(df_hours == False, True, inplace=True)

            # Filter business categories > 1% of max value
            business = drop_infrequent_concept_from_str(business, 'categories')

            # Extract business attributes
            attributes = []
            for attr_list in business['attributes']:
                attr_dict = {}
                if attr_list != None:
                    for a, b in attr_list.items():
                        if (b.lower() == 'true' or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                                'outdoor', 'yes', 'allages', '21plus', '19plus', '18plus', 'full_bar', 'beer_and_wine',
                                'yes_free', 'yes_corkage', 'free', 'paid', 'quiet', 'average', 'loud', 'very_loud',
                                'casual',
                                'formal', 'dressy')):
                            attr_dict[a.strip()] = True
                        elif (b.lower() in ('false', 'none') or ''.join(re.findall(r"'(.*?)'", b)).lower() in (
                                'no', 'none')):
                            attr_dict[a.strip()] = False
                        elif (b[0] != '{'):
                            attr_dict[a.strip()] = True
                        else:
                            for c in b.split(","):
                                attr_dict[a.strip()] = False
                                if (c == '{}'):
                                    attr_dict[a.strip()] = False
                                    break
                                elif (c.split(":")[1].strip().lower() == 'true'):
                                    attr_dict[a.strip()] = True
                                    break
                attributes.append([k for k, v in attr_dict.items() if v == True])

            business['attributes'] = [','.join(map(str, l)) for l in attributes]

            # Concating business df
            business_concat = [business.iloc[:, :-1], df_hours]
            business = pd.concat(business_concat, axis=1)

            # Compute friend counts
            user['friends_count'] = [len(f.split(",")) if f != 'None' else 0 for f in user['friends']]

            # Compute checkin counts
            checkin['checkin_count'] = [len(f.split(",")) if f != 'None' else 0 for f in checkin['date']]

            # Extract business checkin times
            checkin_years = []
            checkin_months = []
            checkin_time = []
            for checkin_list in checkin['date']:
                checkin_years_ar = []
                checkin_months_ar = []
                checkin_time_ar = []
                if checkin_list != '':
                    for chk in checkin_list.split(","):
                        checkin_years_ar.append(chk.strip()[:4])
                        checkin_months_ar.append(chk.strip()[:7])

                        if int(chk.strip()[11:13]) in range(0, 4):
                            checkin_time_ar.append('00-03')
                        elif int(chk.strip()[11:13]) in range(3, 7):
                            checkin_time_ar.append('03-06')
                        elif int(chk.strip()[11:13]) in range(6, 10):
                            checkin_time_ar.append('06-09')
                        elif int(chk.strip()[11:13]) in range(9, 13):
                            checkin_time_ar.append('09-12')
                        elif int(chk.strip()[11:13]) in range(12, 16):
                            checkin_time_ar.append('12-15')
                        elif int(chk.strip()[11:13]) in range(15, 19):
                            checkin_time_ar.append('15-18')
                        elif int(chk.strip()[11:13]) in range(18, 22):
                            checkin_time_ar.append('18-21')
                        elif int(chk.strip()[11:13]) in range(21, 24):
                            checkin_time_ar.append('21-24')

                checkin_years.append(Counter(checkin_years_ar))
                checkin_months.append(Counter(checkin_months_ar))
                checkin_time.append(Counter(checkin_time_ar))

            df_checkin = (pd.concat([
                pd.DataFrame(checkin_years)
                    .fillna('0').sort_index(axis=1),
                pd.DataFrame(checkin_months)
                    .fillna('0').sort_index(axis=1),
                pd.DataFrame(checkin_time)
                    .fillna('0').sort_index(axis=1)], axis=1))

            # Concating checkin df
            checkin_concat = [checkin, df_checkin]
            checkin = pd.concat(checkin_concat, axis=1)

            # Merging business and checkin
            business = pd.merge(business, checkin, on='business_id', how='left').fillna(0)

            # Select only relevant columns of review and tip
            review = review.iloc[:, [1, 2]]
            tip = tip.iloc[:, [0, 1]]

            # Concat review and tips
            reviewtip = pd.concat([review, tip], axis=0)

            # remove duplications
            business = business.drop_duplicates()
            user = user.drop_duplicates()
            reviewtip = reviewtip.drop_duplicates()

            if business.shape[0] != business.business_id.unique().shape[0] or user.shape[0] != \
                    user.user_id.unique().shape[0]:
                raise ValueError('Duplicates in dfs.')

            # Compute the business counts for reviewtip
            bus_count = reviewtip['business_id'].value_counts()
            bus_count.name = 'bus_count'

            # Remove infrequent business in reviewtip
            reviewtip = reviewtip[reviewtip.join(bus_count, on='business_id').bus_count > (self.num_core + 40)]

            # Compute the user counts for reviewtip
            user_count = reviewtip['user_id'].value_counts()
            user_count.name = 'user_count'
            reviewtip = reviewtip.join(user_count, on='user_id')

            # Remove infrequent users in reviewtip
            reviewtip = reviewtip[
                (reviewtip.user_count > self.num_core) & (reviewtip.user_count <= (self.num_core + 10))]

            # Sync the business and user dataframe
            user = user[user.user_id.isin(reviewtip['user_id'].unique())]
            business = business[business.business_id.isin(reviewtip['business_id'].unique())]
            reviewtip = reviewtip[reviewtip.user_id.isin(user['user_id'].unique())]
            reviewtip = reviewtip[reviewtip.business_id.isin(business['business_id'].unique())]

            # Compute the updated business and user counts for reviewtip
            bus_count = reviewtip['business_id'].value_counts()
            user_count = reviewtip['user_id'].value_counts()
            bus_count.name = 'bus_count'
            user_count.name = 'user_count'
            reviewtip = reviewtip.iloc[:, [0, 1]].join(bus_count, on='business_id')
            reviewtip = reviewtip.join(user_count, on='user_id')

            # Reindex the bid and uid in case of missing values
            business, user, reviewtip = reindex_df(business, user, reviewtip)

            print('Preprocessing done.')

            business.to_pickle(join(self.processed_dir, 'business.pkl'))
            user.to_pickle(join(self.processed_dir, 'user.pkl'))
            reviewtip.to_pickle(join(self.processed_dir, 'reviewtip.pkl'))

        dataset_property_dict = generate_graph_data(user, business, reviewtip)

        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(dataset_property_dict, f)

    def build_suffix(self):
        return 'core_{}_type_{}'.format(self.num_core, self.type)

    def kg_negative_sampling(self):
        """
        Replace tail entities in existing triples with random entities
        """
        print('KG negative sampling...')
        pos_edge_index_r_nps = [
            (edge_index, np.ones((edge_index.shape[1], 1)) * self.edge_type_dict[edge_type])
            for edge_type, edge_index in self.edge_index_nps.items()
        ]
        pos_edge_index_trans_np = np.hstack([_[0] for _ in pos_edge_index_r_nps]).T
        pos_r_np = np.vstack([_[1] for _ in pos_edge_index_r_nps])
        neg_t_np = np.random.randint(low=0, high=self.num_nodes, size=(pos_edge_index_trans_np.shape[0], 1))
        train_data_np = np.hstack([pos_edge_index_trans_np, neg_t_np, pos_r_np])
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def cf_negative_sampling(self):
        """
        Replace positive items with random/unseen items
        """
        print('CF negative sampling...')
        pos_edge_index_trans_np = self.edge_index_nps['user2item'].T
        num_interactions = pos_edge_index_trans_np.shape[0]
        if self.cf_loss_type == 'BCE':
            pos_samples_np = np.hstack([pos_edge_index_trans_np, np.ones((pos_edge_index_trans_np.shape[0], 1))])
            if self.sampling_strategy == 'random':
                neg_samples_np = np.hstack(
                    [
                        np.repeat(pos_samples_np[:, 0].reshape(-1, 1), repeats=self.num_negative_samples, axis=0),
                        np.random.randint(
                            low=self.type_accs['items'],
                            high=self.type_accs['items'] + self.num_items,
                            size=(num_interactions * self.num_negative_samples, 1)
                        ),
                        torch.zeros((num_interactions * self.num_negative_samples, 1))
                    ]
                )
            elif self.sampling_strategy == 'unseen':
                neg_inids = []
                u_nids = pos_samples_np[:, 0]
                p_bar = tqdm.tqdm(u_nids)
                for u_nid in p_bar:
                    negative_inids = self.test_pos_unid_inid_map[u_nid] + self.neg_unid_inid_map[u_nid]
                    negative_inids = np.random.choice(negative_inids, size=(self.num_negative_samples, 1))
                    neg_inids.append(negative_inids)
                neg_samples_np = np.hstack(
                    [
                        np.repeat(pos_samples_np[:, 0].reshape(-1, 1), repeats=self.num_negative_samples, axis=0),
                        np.vstack(neg_inids),
                        np.zeros((num_interactions * self.num_negative_samples, 1))
                    ]
                )
            else:
                raise NotImplementedError
            train_data_np = np.vstack([pos_samples_np, neg_samples_np])
        elif self.cf_loss_type == 'BPR':
            train_data_np = np.repeat(pos_edge_index_trans_np, repeats=self.num_negative_samples, axis=0)
            if self.sampling_strategy == 'random':
                neg_inid_np = np.random.randint(
                            low=self.type_accs['items'],
                            high=self.type_accs['items'] + self.num_items,
                            size=(num_interactions * self.num_negative_samples, 1)
                        )
            elif self.sampling_strategy == 'unseen':
                neg_inids = []
                u_nids = pos_edge_index_trans_np[:, 0]
                p_bar = tqdm.tqdm(u_nids)
                for u_nid in p_bar:
                    negative_inids = self.test_pos_unid_inid_map[u_nid] + self.neg_unid_inid_map[u_nid]
                    negative_inids = np.random.choice(negative_inids, size=(self.num_negative_samples, 1))
                    neg_inids.append(negative_inids)
                neg_inid_np = np.vstack(neg_inids)
            else:
                raise NotImplementedError
            train_data_np = np.hstack([train_data_np, neg_inid_np])

            if self.entity_aware:
                # add entity aware data to batches
                if not hasattr(self, 'iid_feat_nids'):
                    business = pd.read_pickle(join(self.processed_dir, 'business.pkl')).fillna('')
                    iid_feat_nids = []
                    pbar = tqdm.tqdm(self.unique_iids, total=len(self.unique_iids))
                    for iid in pbar:
                            pbar.set_description('Sampling item entities...')
                            iid_attribute_nids = [self.e2nid_dict['itemattributes'][attribute] for attribute in
                                                  business[business['business_id'] == iid]['attributes'].item().split(',')
                                                  if attribute != '']
                            iid_category_nids = [self.e2nid_dict['itemcategories'][category] for category in
                                                 business[business['business_id'] == iid]['categories'].item().split(',') if
                                                 category != '']
                            feat_nids = iid_attribute_nids + iid_category_nids
                            iid_feat_nids.append(feat_nids)
                    self.iid_feat_nids = iid_feat_nids

                pos_entity_nids = []
                for inid in train_data_np[:, 1]:
                    # pos_entity_nids.append(np.random.choice(self.iid_feat_nids[int(inid - self.type_accs['items'])]))
                    try:
                        pos_entity_nids.append(
                            np.random.choice(self.iid_feat_nids[int(inid - self.type_accs['items'])]))
                    except:
                        import pdb
                        pdb.set_trace()
                pos_entity_nids = np.array(pos_entity_nids).reshape(-1, 1)
                neg_entity_nids = np.random.randint(
                    low=self.type_accs['itemattributes'],
                    high=self.type_accs['itemcheckincount'],
                    size=(train_data_np.shape[0], 1)
                )
                train_data_np = np.hstack([train_data_np, pos_entity_nids, neg_entity_nids])
        else:
            raise NotImplementedError
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def __len__(self):
        return self.train_data_length

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, str):
            return getattr(self, idx, None)
        else:
            idx = idx.to_list() if torch.is_tensor(idx) else idx
            return self.train_data[idx]

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            raise NotImplementedError('Assignment can\'t be done outside of constructor')

    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())
