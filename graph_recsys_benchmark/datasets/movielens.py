import torch
from torch.utils.data import DataLoader
from os.path import join
from os.path import isfile
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import tqdm
import pickle

from .dataset import Dataset
from torch_geometric.data import download_url, extract_zip
from ..parser import parse_ml1m, parse_ml25m


def save_df(df, path):
    df.to_csv(path, sep=';', index=False)


def reindex_df_ml1m(users, movies, ratings):
    """
    reindex users, items, interactions in case there are some values missing or duplicates in between
    :param users: pd.DataFrame
    :param movies: pd.DataFrame
    :param ratings: pd.DataFrame
    :return: same
    """
    unique_uids = np.sort(users.uid.unique()).astype(np.int)
    num_users = unique_uids.shape[0]
    raw_uids = np.array(unique_uids, dtype=np.int)
    uids = np.arange(num_users)
    users['uid'] = uids
    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(raw_uids, uids)}
    rating_uids = np.array(ratings.uid, dtype=np.int)
    rating_uids = [raw_uid2uid[rating_uid] for rating_uid in rating_uids]
    ratings['uid'] = rating_uids

    unique_iids = np.sort(movies.iid.unique()).astype(np.int)
    num_movies = unique_iids.shape[0]
    raw_iids = np.array(unique_iids, dtype=np.int)
    iids = np.arange(num_movies)
    movies['iid'] = iids
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(raw_iids, iids)}
    rating_iids = np.array(ratings.iid, dtype=np.int)
    rating_iids = [raw_iid2iid[rating_iid] for rating_iid in rating_iids]
    ratings['iid'] = rating_iids

    return users, movies, ratings


def reindex_df_ml25m(movies, ratings, tagging, genome_tagging, genome_tags):
    """

    Args:
        movies:
        ratings:
        tagging:
        genome_tagging:
        genome_tags:

    Returns:

    """
    # Reindex uid
    unique_uids = np.sort(ratings.uid.unique()).astype(np.int)
    uids = np.arange(unique_uids.shape[0]).astype(np.int)
    raw_uid2uid = {raw_uid: uid for raw_uid, uid in zip(unique_uids, uids)}
    ratings['uid'] = np.array([raw_uid2uid[raw_uid] for raw_uid in ratings.uid], dtype=np.int)

    # Reindex iid
    unique_iids = np.sort(movies.iid.unique()).astype(np.int)
    iids = np.arange(unique_iids.shape[0]).astype(np.int)
    raw_iid2iid = {raw_iid: iid for raw_iid, iid in zip(unique_iids, iids)}
    movies['iid'] = np.array([raw_iid2iid[raw_iid] for raw_iid in movies.iid], dtype=np.int)
    ratings['iid'] = np.array([raw_iid2iid[raw_iid] for raw_iid in ratings.iid], dtype=np.int)

    # Create tid
    unique_tags = np.sort(tagging.tag.unique()).astype(np.str)
    tids = np.arange(unique_tags.shape[0]).astype(np.int)
    tags = pd.DataFrame({'tid': tids, 'tag': unique_tags})
    tag2tid = {tag: tid for tag, tid in zip(unique_tags, tids)}
    tagging['tid'] = np.array([tag2tid[tag] for tag in tagging.tag], dtype=np.int)
    tagging.drop(columns=['tag'])

    # Reindex genome_tid
    unique_genome_tids = np.sort(genome_tags.genome_tid.unique()).astype(np.int)
    genome_tids = np.arange(unique_genome_tids.shape[0]).astype(np.int)
    raw_genome_tid2genome_tid = {raw_genome_tid: tid for raw_genome_tid, tid in zip(unique_genome_tids, genome_tids)}
    genome_tags['genome_tid'] = np.array([raw_genome_tid2genome_tid[raw_genome_tid] for raw_genome_tid in genome_tags.genome_tid], dtype=np.int)
    genome_tagging['genome_tid'] = np.array([raw_genome_tid2genome_tid[raw_genome_tid] for raw_genome_tid in genome_tagging.genome_tid])

    return movies, ratings, tagging, tags, genome_tagging, genome_tags


def drop_infrequent_concept_from_str(df, concept_name, num_occs):
    concept_strs = [concept_str for concept_str in df[concept_name]]
    duplicated_concept = [concept_str.split(',') for concept_str in concept_strs]
    duplicated_concept = list(itertools.chain.from_iterable(duplicated_concept))
    writer_counter_dict = Counter(duplicated_concept)
    del writer_counter_dict['']
    del writer_counter_dict['N/A']
    unique_concept = [k for k, v in writer_counter_dict.items() if v >= num_occs]
    concept_strs = [
        ','.join([concept for concept in concept_str.split(',') if concept in unique_concept])
        for concept_str in concept_strs
    ]
    df[concept_name] = concept_strs
    return df


def generate_ml1m_graph_data(
        users, items, ratings
):
    """
    Entitiy node include (gender, occupation, genres)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """

    def get_concept_num_from_str(df, concept_name):
        concept_strs = [concept_str.split(',') for concept_str in df[concept_name]]
        concepts = set(itertools.chain.from_iterable(concept_strs))
        concepts.remove('')
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Create dataset property dict  #########################
    dataset_property_dict = {'users': users, 'items': items, 'ratings': ratings}

    #########################  Define entities  #########################
    num_users = users.shape[0]
    num_items = items.shape[0]
    dataset_property_dict['num_users'] = num_users
    dataset_property_dict['num_items'] = num_items

    unique_genders = list(users.gender.unique())
    num_genders = len(unique_genders)

    unique_occupations = list(users.occupation.unique())
    num_occupations = len(unique_occupations)

    unique_ages = list(users.age.unique())
    num_ages = len(unique_ages)

    unique_genres = list(items.keys()[3:20])
    num_genres = len(unique_genres)

    unique_years = list(items.year.unique())
    num_years = len(unique_years)

    unique_directors, num_directors = get_concept_num_from_str(items, 'directors')
    unique_actors, num_actors = get_concept_num_from_str(items, 'actors')
    unique_writers, num_writers = get_concept_num_from_str(items, 'writers')

    dataset_property_dict['unique_genders'] = unique_genders
    dataset_property_dict['num_genders'] = num_genders
    dataset_property_dict['unique_occupations'] = unique_occupations
    dataset_property_dict['num_occupations'] = num_occupations
    dataset_property_dict['unique_ages'] = unique_ages
    dataset_property_dict['num_ages'] = num_ages
    dataset_property_dict['unique_genres'] = unique_genres
    dataset_property_dict['num_genres'] = num_genres
    dataset_property_dict['unique_years'] = unique_years
    dataset_property_dict['num_years'] = num_years
    dataset_property_dict['unique_directors'] = unique_directors
    dataset_property_dict['num_directors'] = num_directors
    dataset_property_dict['unique_actors'] = unique_actors
    dataset_property_dict['num_actors'] = num_actors
    dataset_property_dict['unique_writers'] = unique_writers
    dataset_property_dict['num_writers'] = num_writers

    #########################  Define number of entities  #########################
    num_nodes = num_users + num_items + num_genders + num_occupations + num_ages + num_genres + num_years + \
                num_directors + num_actors + num_writers
    num_node_types = 10
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types
    types = ['user', 'movie', 'gender', 'occupation', 'age', 'genre', 'year', 'director', 'actor', 'writer']
    num_nodes_dict = {'user': num_users, 'movie': num_items, 'gender': num_genders, 'occupation': num_occupations,
                      'age': num_ages, 'genre': num_genres, 'year': num_years, 'director': num_directors,
                      'actor': num_actors, 'writer': num_writers}

    #########################  Define entities to node id map  #########################
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs['user'] = acc
    uid2nid = {uid: i + acc for i, uid in enumerate(users['uid'])}
    for i, uid in enumerate(users['uid']):
        nid2e_dict[i + acc] = ('uid', uid)
    acc += num_users
    type_accs['movie'] = acc
    iid2nid = {iid: i + acc for i, iid in enumerate(items['iid'])}
    for i, iid in enumerate(items['iid']):
        nid2e_dict[i + acc] = ('iid', iid)
    acc += num_items
    type_accs['gender'] = acc
    gender2nid = {gender: i + acc for i, gender in enumerate(unique_genders)}
    for i, gender in enumerate(unique_genders):
        nid2e_dict[i + acc] = ('gender', gender)
    acc += num_genders
    type_accs['occupation'] = acc
    occ2nid = {occupation: i + acc for i, occupation in enumerate(unique_occupations)}
    for i, occ in enumerate(unique_occupations):
        nid2e_dict[i + acc] = ('occ', occ)
    acc += num_occupations
    type_accs['age'] = acc
    age2nid = {age: i + acc for i, age in enumerate(unique_ages)}
    for i, age in enumerate(unique_ages):
        nid2e_dict[i + acc] = ('age', age)
    acc += num_ages
    type_accs['genre'] = acc
    genre2nid = {genre: i + acc for i, genre in enumerate(unique_genres)}
    for i, genre in enumerate(unique_genres):
        nid2e_dict[i + acc] = ('genre', genre)
    acc += num_genres
    type_accs['year'] = acc
    year2nid = {year: i + acc for i, year in enumerate(unique_years)}
    for i, year in enumerate(unique_years):
        nid2e_dict[i + acc] = ('year', year)
    acc += num_years
    type_accs['director'] = acc
    director2nid = {director: i + acc for i, director in enumerate(unique_directors)}
    for i, director in enumerate(unique_directors):
        nid2e_dict[i + acc] = ('director', director)
    acc += num_directors
    type_accs['actor'] = acc
    actor2nid = {actor: i + acc for i, actor in enumerate(unique_actors)}
    for i, actor in enumerate(unique_actors):
        nid2e_dict[i + acc] = ('actor', actor)
    acc += num_actors
    type_accs['writer'] = acc
    writer2nid = {writer: i + acc for i, writer in enumerate(unique_writers)}
    for i, writer in enumerate(unique_writers):
        nid2e_dict[i + acc] = ('writer', writer)
    e2nid_dict = {'uid': uid2nid, 'iid': iid2nid, 'gender': gender2nid, 'occ': occ2nid, 'age': age2nid,
                  'genre': genre2nid,
                  'year': year2nid, 'director': director2nid, 'actor': actor2nid, 'writer': writer2nid}
    dataset_property_dict['e2nid_dict'] = e2nid_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print('Creating user property edges...')
    u_nids = [e2nid_dict['uid'][uid] for uid in users.uid]
    gender_nids = [e2nid_dict['gender'][gender] for gender in users.gender]
    gender2user_edge_index_np = np.vstack((np.array(gender_nids), np.array(u_nids)))
    occ_nids = [e2nid_dict['occ'][occ] for occ in users.occupation]
    occ2user_edge_index_np = np.vstack((np.array(occ_nids), np.array(u_nids)))
    age_nids = [e2nid_dict['age'][age] for age in users.age]
    age2user_edge_index_np = np.vstack((np.array(age_nids), np.array(u_nids)))
    edge_index_nps['gender2user'] = gender2user_edge_index_np
    edge_index_nps['occ2user'] = occ2user_edge_index_np
    edge_index_nps['age2user'] = age2user_edge_index_np

    print('Creating item property edges...')
    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    year_nids = [e2nid_dict['year'][year] for year in items.year]
    year2item_edge_index_np = np.vstack((np.array(year_nids), np.array(i_nids)))

    genre_nids = []
    i_nids = []
    for genre in unique_genres:
        iids = items.iid[items[genre]]
        i_nids += [e2nid_dict['iid'][iid] for iid in iids]
        genre_nids += [e2nid_dict['genre'][genre] for _ in range(iids.shape[0])]
    genre2item_edge_index_np = np.vstack((np.array(genre_nids), np.array(i_nids)))

    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    directors_list = [
        [director for director in directors.split(',') if director != '']
        for directors in items.directors
    ]
    directors_nids = [[e2nid_dict['director'][director] for director in directors] for directors in directors_list]
    directors_nids = list(itertools.chain.from_iterable(directors_nids))
    d_i_nids = [[i_nid for _ in range(len(directors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    d_i_nids = list(itertools.chain.from_iterable(d_i_nids))
    director2item_edge_index_np = np.vstack((np.array(directors_nids), np.array(d_i_nids)))

    actors_list = [
        [actor for actor in actors.split(',') if actor != '']
        for actors in items.actors
    ]
    actor_nids = [[e2nid_dict['actor'][actor] for actor in actors] for actors in actors_list]
    actor_nids = list(itertools.chain.from_iterable(actor_nids))
    a_i_nids = [[i_nid for _ in range(len(actors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    a_i_nids = list(itertools.chain.from_iterable(a_i_nids))
    actor2item_edge_index_np = np.vstack((np.array(actor_nids), np.array(a_i_nids)))

    writers_list = [
        [writer for writer in writers.split(',') if writer != '']
        for writers in items.writers
    ]
    writer_nids = [[e2nid_dict['writer'][writer] for writer in writers] for writers in writers_list]
    writer_nids = list(itertools.chain.from_iterable(writer_nids))
    w_i_nids = [[i_nid for _ in range(len(writers_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    w_i_nids = list(itertools.chain.from_iterable(w_i_nids))
    writer2item_edge_index_np = np.vstack((np.array(writer_nids), np.array(w_i_nids)))
    edge_index_nps['year2item'] = year2item_edge_index_np
    edge_index_nps['genre2item'] = genre2item_edge_index_np
    edge_index_nps['director2item'] = director2item_edge_index_np
    edge_index_nps['actor2item'] = actor2item_edge_index_np
    edge_index_nps['writer2item'] = writer2item_edge_index_np

    print('Creating rating property edges...')
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = {}, {}, {}

    user2item_edge_index_np = np.zeros((2, 0))
    pbar = tqdm.tqdm(users.uid, total=users.uid.shape[0])
    for uid in pbar:
        pbar.set_description('Creating the edges for the user {}'.format(uid))
        uid_ratings = ratings[ratings.uid == uid].sort_values('timestamp')
        uid_iids = uid_ratings[['iid']].to_numpy().reshape(-1)

        unid = e2nid_dict['uid'][uid]
        train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
        train_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in train_pos_uid_iids]
        test_pos_uid_iids = list(uid_iids[-1:])
        test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]
        neg_uid_iids = list(set(items.iid) - set(uid_iids))
        neg_uid_inids = [e2nid_dict['iid'][iid] for iid in neg_uid_iids]

        train_pos_unid_inid_map[unid] = train_pos_uid_inids
        test_pos_unid_inid_map[unid] = test_pos_uid_inids
        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack([user2item_edge_index_np, unid_user2item_edge_index_np])
    edge_index_nps['user2item'] = user2item_edge_index_np

    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['train_pos_unid_inid_map'], dataset_property_dict['test_pos_unid_inid_map'], \
    dataset_property_dict['neg_unid_inid_map'] = \
        train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map

    print('Building edge type map...')
    edge_type_dict = {edge_type: edge_type_idx for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))}
    dataset_property_dict['edge_type_dict'] = edge_type_dict
    dataset_property_dict['num_edge_types'] = len(list(edge_index_nps.keys()))

    print('Building the item occurrence map...')
    item_nid_occs = {}
    for iid in items.iid:
        item_nid_occs[e2nid_dict['iid'][iid]] = ratings[ratings.iid == iid].iloc[0]['movie_count']
    dataset_property_dict['item_nid_occs'] = item_nid_occs

    # New functionality for pytorch geometric like dataset
    dataset_property_dict['types'] = types
    dataset_property_dict['num_nodes_dict'] = num_nodes_dict
    dataset_property_dict['type_accs'] = type_accs

    return dataset_property_dict


def generate_ml25m_graph_data(
        movies, ratings,tags, tagging, genome_tags, genome_tagging
):
    """
    Entitiy node include (gender, occupation, genres)
    num_nodes = num_users + num_items + num_genders + num_occupation + num_ages + num_genres + num_years + num_directors + num_actors + num_writers
    """

    def get_concept_num_from_str(df, concept_name):
        concept_strs = [concept_str.split(',') for concept_str in df[concept_name]]
        concepts = set(itertools.chain.from_iterable(concept_strs))
        concepts.remove('')
        num_concepts = len(concepts)
        return list(concepts), num_concepts

    #########################  Create dataset property dict  #########################
    dataset_property_dict = {'items': items, 'ratings': ratings}
    if users is not None:
        dataset_property_dict['users'] = users
    else:
        dataset_property_dict['tags'] = tags
        dataset_property_dict['genome_scores'] = genome_scores
        dataset_property_dict['genome_tags'] = genome_tags

    #########################  Define entities  #########################
    num_users = users.shape[0] if users is not None else len(list(ratings.uid.unique()))
    num_items = items.shape[0]
    dataset_property_dict['num_users'] = num_users
    dataset_property_dict['num_items'] = num_items

    if users is not None:
        unique_genders = list(users.gender.unique())
        num_genders = len(unique_genders)

        unique_occupations = list(users.occupation.unique())
        num_occupations = len(unique_occupations)

        unique_ages = list(users.age.unique())
        num_ages = len(unique_ages)
    else:
        unique_tags = list(tags.tag.unique())
        num_tags = len(unique_tags)

        unique_genome_tags = list(genome_tags.tid.unique())
        num_genome_tags = len(unique_genome_tags)

    unique_genres = list(items.keys()[3:20])
    num_genres = len(unique_genres)

    unique_years = list(items.year.unique())
    num_years = len(unique_years)

    unique_directors, num_directors = get_concept_num_from_str(items, 'directors')
    unique_actors, num_actors = get_concept_num_from_str(items, 'actors')
    unique_writers, num_writers = get_concept_num_from_str(items, 'writers')

    if users is not None:
        dataset_property_dict['unique_genders'] = unique_genders
        dataset_property_dict['num_genders'] = num_genders
        dataset_property_dict['unique_occupations'] = unique_occupations
        dataset_property_dict['num_occupations'] = num_occupations
        dataset_property_dict['unique_ages'] = unique_ages
        dataset_property_dict['num_ages'] = num_ages
    else:
        dataset_property_dict['unique_tags'] = unique_tags
        dataset_property_dict['num_tags'] = num_tags
        dataset_property_dict['unique_genome_tags'] = unique_genome_tags
        dataset_property_dict['num_genome_tags'] = num_genome_tags
    dataset_property_dict['unique_genres'] = unique_genres
    dataset_property_dict['num_genres'] = num_genres
    dataset_property_dict['unique_years'] = unique_years
    dataset_property_dict['num_years'] = num_years
    dataset_property_dict['unique_directors'] = unique_directors
    dataset_property_dict['num_directors'] = num_directors
    dataset_property_dict['unique_actors'] = unique_actors
    dataset_property_dict['num_actors'] = num_actors
    dataset_property_dict['unique_writers'] = unique_writers
    dataset_property_dict['num_writers'] = num_writers


    #########################  Define number of entities  #########################
    num_nodes = num_users + num_items + num_genres + num_years + \
                num_directors + num_actors + num_writers
    num_node_types = 7
    if users is not None:
        num_nodes += num_genders + num_occupations + num_ages
        num_node_types += 3
    else:
        num_nodes += num_tags + num_genome_tags
        num_node_types += 2
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types
    if users is not None:
        types = ['user', 'movie', 'gender', 'occupation', 'age', 'genre', 'year', 'director', 'actor', 'writer']
    else:
        types = ['user', 'movie', 'genre', 'year', 'director', 'actor', 'writer', 'tag', 'genome_tag']
    if users is not None:
        num_nodes_dict = {'user': num_users, 'movie': num_items, 'gender': num_genders, 'occupation': num_occupations,
                      'age': num_ages, 'genre': num_genres, 'year': num_years, 'director': num_directors,
                      'actor': num_actors, 'writer': num_writers}
    else:
        num_nodes_dict = {'user': num_users, 'movie': num_items, 'genre': num_genres, 'year': num_years,
                          'director': num_directors, 'actor': num_actors, 'writer': num_writers,
                          'tags': num_tags, 'genome_tags': genome_tags}

    #########################  Define entities to node id map  #########################
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs['user'] = acc
    uid2nid = {uid: i + acc for i, uid in enumerate(users['uid'])}
    for i, uid in enumerate(users['uid']):
        nid2e_dict[i + acc] = ('uid', uid)
    acc += num_users
    type_accs['movie'] = acc
    iid2nid = {iid: i + acc for i, iid in enumerate(items['iid'])}
    for i, iid in enumerate(items['iid']):
        nid2e_dict[i + acc] = ('iid', iid)
    acc += num_items
    if users is not None:
        type_accs['gender'] = acc
        gender2nid = {gender: i + acc for i, gender in enumerate(unique_genders)}
        for i, gender in enumerate(unique_genders):
            nid2e_dict[i + acc] = ('gender', gender)
        acc += num_genders
        type_accs['occupation'] = acc
        occ2nid = {occupation: i + acc for i, occupation in enumerate(unique_occupations)}
        for i, occ in enumerate(unique_occupations):
            nid2e_dict[i + acc] = ('occ', occ)
        acc += num_occupations
        type_accs['age'] = acc
        age2nid = {age: i + acc for i, age in enumerate(unique_ages)}
        for i, age in enumerate(unique_ages):
            nid2e_dict[i + acc] = ('age', age)
        acc += num_ages
    else:
        acc += num_items
    type_accs['genre'] = acc
    genre2nid = {genre: i + acc for i, genre in enumerate(unique_genres)}
    for i, genre in enumerate(unique_genres):
        nid2e_dict[i + acc] = ('genre', genre)
    acc += num_genres
    type_accs['year'] = acc
    year2nid = {year: i + acc for i, year in enumerate(unique_years)}
    for i, year in enumerate(unique_years):
        nid2e_dict[i + acc] = ('year', year)
    acc += num_years
    type_accs['director'] = acc
    director2nid = {director: i + acc for i, director in enumerate(unique_directors)}
    for i, director in enumerate(unique_directors):
        nid2e_dict[i + acc] = ('director', director)
    acc += num_directors
    type_accs['actor'] = acc
    actor2nid = {actor: i + acc for i, actor in enumerate(unique_actors)}
    for i, actor in enumerate(unique_actors):
        nid2e_dict[i + acc] = ('actor', actor)
    acc += num_actors
    type_accs['writer'] = acc
    writer2nid = {writer: i + acc for i, writer in enumerate(unique_writers)}
    for i, writer in enumerate(unique_writers):
        nid2e_dict[i + acc] = ('writer', writer)
    if users is None:
        acc += num_writers
        type_accs['tag'] = acc
        tag2nid = {tag: i + acc for i, tag in enumerate(unique_tags)}
        for i, tag in enumerate(unique_tags):
            nid2e_dict[i + acc] = ('tag', tag)
        acc += num_tags
        type_accs['genome_tags'] = acc
        genometag2nid = {genome_tag: i + acc for i, genome_tag in enumerate(unique_genome_tags)}
        for i, genome_tag in enumerate(unique_genome_tags):
            nid2e_dict[i + acc] = ('genome_tag', genome_tag)
    e2nid_dict = {'uid': uid2nid, 'iid': iid2nid, 'gender': gender2nid, 'occ': occ2nid, 'age': age2nid, 'genre': genre2nid,
             'year': year2nid, 'director': director2nid, 'actor': actor2nid, 'writer': writer2nid}
    if users is None:
        e2nid_dict['tag'] = tag2nid
        e2nid_dict['genome_tag'] = genometag2nid
    dataset_property_dict['e2nid_dict'] = e2nid_dict

    #########################  create graphs  #########################
    edge_index_nps = {}
    print('Creating user property edges...')
    u_nids = [e2nid_dict['uid'][uid] for uid in users.uid]
    gender_nids = [e2nid_dict['gender'][gender] for gender in users.gender]
    gender2user_edge_index_np = np.vstack((np.array(gender_nids), np.array(u_nids)))
    occ_nids = [e2nid_dict['occ'][occ] for occ in users.occupation]
    occ2user_edge_index_np = np.vstack((np.array(occ_nids), np.array(u_nids)))
    age_nids = [e2nid_dict['age'][age] for age in users.age]
    age2user_edge_index_np = np.vstack((np.array(age_nids), np.array(u_nids)))
    edge_index_nps['gender2user'] = gender2user_edge_index_np
    edge_index_nps['occ2user'] = occ2user_edge_index_np
    edge_index_nps['age2user'] = age2user_edge_index_np

    print('Creating item property edges...')
    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    year_nids = [e2nid_dict['year'][year] for year in items.year]
    year2item_edge_index_np = np.vstack((np.array(year_nids), np.array(i_nids)))

    genre_nids = []
    i_nids = []
    for genre in unique_genres:
        iids = items.iid[items[genre]]
        i_nids += [e2nid_dict['iid'][iid] for iid in iids]
        genre_nids += [e2nid_dict['genre'][genre] for _ in range(iids.shape[0])]
    genre2item_edge_index_np = np.vstack((np.array(genre_nids), np.array(i_nids)))

    i_nids = [e2nid_dict['iid'][iid] for iid in items.iid]
    directors_list = [
        [director for director in directors.split(',') if director != '']
        for directors in items.directors
    ]
    directors_nids = [[e2nid_dict['director'][director] for director in directors] for directors in directors_list]
    directors_nids = list(itertools.chain.from_iterable(directors_nids))
    d_i_nids = [[i_nid for _ in range(len(directors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    d_i_nids = list(itertools.chain.from_iterable(d_i_nids))
    director2item_edge_index_np = np.vstack((np.array(directors_nids), np.array(d_i_nids)))

    actors_list = [
        [actor for actor in actors.split(',') if actor != '']
        for actors in items.actors
    ]
    actor_nids = [[e2nid_dict['actor'][actor] for actor in actors] for actors in actors_list]
    actor_nids = list(itertools.chain.from_iterable(actor_nids))
    a_i_nids = [[i_nid for _ in range(len(actors_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    a_i_nids = list(itertools.chain.from_iterable(a_i_nids))
    actor2item_edge_index_np = np.vstack((np.array(actor_nids), np.array(a_i_nids)))

    writers_list = [
        [writer for writer in writers.split(',') if writer != '']
        for writers in items.writers
    ]
    writer_nids = [[e2nid_dict['writer'][writer] for writer in writers] for writers in writers_list]
    writer_nids = list(itertools.chain.from_iterable(writer_nids))
    w_i_nids = [[i_nid for _ in range(len(writers_list[idx]))] for idx, i_nid in enumerate(i_nids)]
    w_i_nids = list(itertools.chain.from_iterable(w_i_nids))
    writer2item_edge_index_np = np.vstack((np.array(writer_nids), np.array(w_i_nids)))
    edge_index_nps['year2item'] = year2item_edge_index_np
    edge_index_nps['genre2item'] = genre2item_edge_index_np
    edge_index_nps['director2item'] = director2item_edge_index_np
    edge_index_nps['actor2item'] = actor2item_edge_index_np
    edge_index_nps['writer2item'] = writer2item_edge_index_np

    print('Creating rating property edges...')
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = {}, {}, {}

    user2item_edge_index_np = np.zeros((2, 0))
    pbar = tqdm.tqdm(users.uid, total=users.uid.shape[0])
    for uid in pbar:
        pbar.set_description('Creating the edges for the user {}'.format(uid))
        uid_ratings = ratings[ratings.uid == uid].sort_values('timestamp')
        uid_iids = uid_ratings[['iid']].to_numpy().reshape(-1)

        unid = e2nid_dict['uid'][uid]
        train_pos_uid_iids = list(uid_iids[:-1])  # Use leave one out setup
        train_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in train_pos_uid_iids]
        test_pos_uid_iids = list(uid_iids[-1:])
        test_pos_uid_inids = [e2nid_dict['iid'][iid] for iid in test_pos_uid_iids]
        neg_uid_iids = list(set(items.iid) - set(uid_iids))
        neg_uid_inids = [e2nid_dict['iid'][iid] for iid in neg_uid_iids]

        train_pos_unid_inid_map[unid] = train_pos_uid_inids
        test_pos_unid_inid_map[unid] = test_pos_uid_inids
        neg_unid_inid_map[unid] = neg_uid_inids

        unid_user2item_edge_index_np = np.array(
            [[unid for _ in range(len(train_pos_uid_inids))], train_pos_uid_inids]
        )
        user2item_edge_index_np = np.hstack([user2item_edge_index_np, unid_user2item_edge_index_np])
    edge_index_nps['user2item'] = user2item_edge_index_np

    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['train_pos_unid_inid_map'], dataset_property_dict['test_pos_unid_inid_map'], \
            dataset_property_dict['neg_unid_inid_map'] = \
        train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map

    print('Building edge type map...')
    edge_type_dict = {edge_type: edge_type_idx for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))}
    dataset_property_dict['edge_type_dict'] = edge_type_dict
    dataset_property_dict['num_edge_types'] = len(list(edge_index_nps.keys()))

    print('Building the item occurrence map...')
    item_nid_occs = {}
    for iid in items.iid:
        item_nid_occs[e2nid_dict['iid'][iid]] = ratings[ratings.iid == iid].iloc[0]['movie_count']
    dataset_property_dict['item_nid_occs'] = item_nid_occs

    # New functionality for pytorch geometric like dataset
    dataset_property_dict['types'] = types
    dataset_property_dict['num_nodes_dict'] = num_nodes_dict
    dataset_property_dict['type_accs'] = type_accs

    return dataset_property_dict


class MovieLens(Dataset):
    url = 'http://files.grouplens.org/datasets/movielens/'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):

        self.name = name.lower()
        assert self.name in ['1m', '25m']
        self.num_core = kwargs['num_core']
        self.num_feat_core = kwargs['num_feat_core']
        self.seed = kwargs['seed']
        self.num_negative_samples = kwargs['num_negative_samples']
        self.cf_loss_type = kwargs['cf_loss_type']
        self._cf_negative_sampling = kwargs['_cf_negative_sampling']
        self.kg_loss_type = kwargs.get('kg_loss_type', None)

        super(MovieLens, self).__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], 'rb') as f:  # Read the class property
            dataset_property_dict = pickle.load(f)
        for k, v in dataset_property_dict.items():
            self[k] = v

        print('Dataset loaded!')

    @property
    def raw_file_names(self):
        return 'ml-{}.zip'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return ['ml_{}_{}.pkl'.format(self.name, self.build_suffix())]

    def download(self):
        path = download_url(self.url + self.raw_file_names, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self):
        if self.name == '1m':
            try:
                users = pd.read_csv(join(self.processed_dir, 'users.csv'), sep=';')
                movies = pd.read_csv(join(self.processed_dir, 'movies.csv'), sep=';')
                ratings = pd.read_csv(join(self.processed_dir, 'ratings.csv'), sep=';')

                users = users.fillna('')
                movies = movies.fillna('')
                ratings = ratings.fillna('')
                print('Read data frame from {}!'.format(self.processed_dir))
            except:
                unzip_raw_dir = join(self.raw_dir, 'ml-{}'.format(self.name))
                print('Data frame not found in {}! Read from raw data and preprocessing from {}!'.format(self.processed_dir, unzip_raw_dir))
                users, movies, ratings = parse_ml1m(unzip_raw_dir)

                # Remove duplicates
                movies = movies.drop_duplicates()
                ratings = ratings.drop_duplicates()
                users = users.drop_duplicates()

                # Compute the movie and user counts
                item_count = ratings['iid'].value_counts()
                item_count.name = 'movie_count'
                ratings = ratings.join(item_count, on='iid')
                user_count = ratings['uid'].value_counts()
                user_count.name = 'user_count'
                ratings = ratings.join(user_count, on='uid')

                # Remove infrequent users and item in ratings
                ratings = ratings[ratings.user_count > self.num_core]
                ratings = ratings[ratings.movie_count > self.num_core]

                # Sync the user and item dataframe
                users = users[users.uid.isin(ratings['uid'].unique())]
                movies = movies[movies.iid.isin(ratings['iid'].unique())]
                ratings = ratings[ratings.uid.isin(users['uid'].unique())]
                ratings = ratings[ratings.iid.isin(movies['iid'].unique())]

                # Reindex
                users, movies, ratings = reindex_df_ml1m(users, movies, ratings)

                # Discretized year
                years = movies.year.to_numpy().astype(np.int)
                min_year = min(years)
                max_year = max(years)
                num_years = (max_year - min_year) // 10
                discretized_years = [min_year + i * 10 for i in range(num_years + 1)]
                for i, discretized_year in enumerate(discretized_years):
                    if i != len(discretized_years) - 1:
                        years[(discretized_year <= years) & (years < discretized_years[i + 1])] = str(discretized_year)
                    else:
                        years[discretized_year <= years] = str(discretized_year)
                movies['year'] = years

                # Drop the infrequent writer, actor and directors
                movies = drop_infrequent_concept_from_str(movies, 'writers', self.num_feat_core)
                movies = drop_infrequent_concept_from_str(movies, 'directors', self.num_feat_core)
                movies = drop_infrequent_concept_from_str(movies, 'actors', self.num_feat_core)

                # Save csv files
                print('Saving processed csv...')
                save_df(users, join(self.processed_dir, 'users.csv'))
                save_df(movies, join(self.processed_dir, 'movies.csv'))
                save_df(ratings, join(self.processed_dir, 'ratings.csv'))

            # Generate and save graph
            dataset_property_dict = generate_ml1m_graph_data(users, movies, ratings)
            with open(self.processed_paths[0], 'wb') as f:
                pickle.dump(dataset_property_dict, f)
        elif self.name == '25m':
            try:
                print('Read data frame from {}!'.format(self.processed_dir))
                movies = pd.read_csv(join(self.processed_dir, 'movies.csv'), sep=';')
                ratings = pd.read_csv(join(self.processed_dir, 'ratings.csv'), sep=';')
                tags = pd.read_csv(join(self.processed_dir, 'tags.csv'), sep=';')
                tagging = pd.read_csv(join(self.processed_dir, 'tagging.csv'), sep=';')
                genome_tags = pd.read_csv(join(self.processed_dir, 'genome_tags.csv'), sep=';')
                genome_tagging = pd.read_csv(join(self.processed_dir, 'genome_tagging.csv'), sep=';')
            except:
                unzip_raw_dir = join(self.raw_dir, 'ml-{}'.format(self.name))
                print('Data frame not found in {}! Read from raw data and preprocessing from {}!'.format(self.processed_dir, unzip_raw_dir))
                movies, ratings, tagging, genome_scores, genome_tags = parse_ml25m(unzip_raw_dir)

                # Remove duplicates
                movies = movies.drop_duplicates()
                ratings = ratings.drop_duplicates()
                tagging = tagging.drop_duplicates()
                genome_scores = genome_scores.drop_duplicates()
                genome_tags = genome_tags.drop_duplicates()

                # Compute the movie and user counts
                item_count = ratings['iid'].value_counts()
                item_count.name = 'movie_count'
                ratings = ratings.join(item_count, on='iid')
                user_count = ratings['uid'].value_counts()
                user_count.name = 'user_count'
                ratings = ratings.join(user_count, on='uid')

                # Remove infrequent users and item in ratings
                ratings = ratings[ratings.user_count > self.num_core]
                ratings = ratings[ratings.movie_count > self.num_core]

                # Sync the user and item dataframe
                movies = movies[movies.iid.isin(ratings['iid'].unique())]
                ratings = ratings[ratings.iid.isin(movies['iid'].unique())]

                # Remove infrequent tags
                tag_count = tagging['tag'].value_counts()
                tag_count.name = 'tag_count'
                tagging = tagging[tagging.join(tag_count, on='tag').tag_count > self.num_core]

                # Remove infrequent genome tags
                genome_scores = genome_scores[genome_scores.relevance > 0.5]
                genome_tagging = genome_scores.drop(columns=['relevance'])
                genome_tag_count = genome_tagging['genome_tid'].value_counts()
                genome_tag_count.name = 'genome_tag_count'
                genome_tagging = genome_tagging[
                    genome_tagging.join(genome_tag_count, 'genome_tid').genome_tag_count > self.num_core]
                genome_tags = genome_tags[genome_tags.genome_tid.isin(genome_tagging['genome_tid'].unique())]
                genome_tagging = genome_tagging[genome_tagging.genome_tid.isin(genome_tags['genome_tid'].unique())]

                # Reindex the uid and iid in case of missing values
                movies, ratings, tagging, tags, genome_tagging, genome_tags = reindex_df_ml25m(
                    movies, ratings, tagging, genome_tagging, genome_tags)

                # Discretized year
                years = movies.year.to_numpy().astype(np.int)
                min_year = min(years)
                max_year = max(years)
                num_years = (max_year - min_year) // 10
                discretized_years = [min_year + i * 10 for i in range(num_years + 1)]
                for i, discretized_year in enumerate(discretized_years):
                    if i != len(discretized_years) - 1:
                        years[(discretized_year <= years) & (years < discretized_years[i + 1])] = str(discretized_year)
                    else:
                        years[discretized_year <= years] = str(discretized_year)
                movies['year'] = years

                # Drop the infrequent writer, actor and directors
                movies = drop_infrequent_concept_from_str(movies, 'writers', self.num_feat_core)
                movies = drop_infrequent_concept_from_str(movies, 'directors', self.num_feat_core)
                movies = drop_infrequent_concept_from_str(movies, 'actors', self.num_feat_core)

                # save dfs
                print('Saving processed csv...')
                save_df(tags, join(self.processed_dir, 'tags.csv'))
                save_df(tagging, join(self.processed_dir, 'tagging.csv'))
                save_df(genome_tagging, join(self.processed_dir, 'genome_tagging.csv'))
                save_df(genome_tags, join(self.processed_dir, 'genome_tags.csv'))
                save_df(movies, join(self.processed_dir, 'movies.csv'))
                save_df(ratings, join(self.processed_dir, 'ratings.csv'))

            # Generate and save graph
            dataset_property_dict = generate_ml25m_graph_data(movies, ratings,tags, tagging, genome_tags, genome_tagging)
            with open(self.processed_paths[0], 'wb') as f:
                pickle.dump(dataset_property_dict, f)

    def build_suffix(self):
        suffixes = [
            'core_{}'.format(self.num_core),
            'featcore_{}'.format(self.num_feat_core),
            'seed_{}'.format(self.seed)
        ]
        if not suffixes:
            suffix = ''
        else:
            suffix = '_'.join(suffixes)
        return suffix

    def kg_negative_sampling(self):
        print('KG negative sampling...')
        pos_edge_index_r_nps = [
            (edge_index, np.ones((edge_index.shape[1], 1)) * self.edge_type_dict[edge_type])
            for edge_type, edge_index in self.edge_index_nps.items()
        ]
        pos_edge_index_trans_np = np.hstack([_[0] for _ in pos_edge_index_r_nps]).T
        pos_r_np = np.vstack([_[1] for _ in pos_edge_index_r_nps])
        neg_t_np = np.random.randint(low=0, high=self.num_nodes, size=(pos_edge_index_trans_np.shape[0], 1))
        if self.cf_loss_type == 'BCE':
            pos_samples_np = np.hstack([pos_edge_index_trans_np, pos_r_np])
            neg_samples_np = np.hstack([pos_edge_index_trans_np[:, 0], neg_t_np, pos_r_np])
            train_data_np = np.vstack([pos_samples_np, neg_samples_np])
        elif self.cf_loss_type == 'BPR':
            train_data_np = np.hstack([pos_edge_index_trans_np, neg_t_np, pos_r_np])
        else:
            raise NotImplementedError('KG loss type not specified or not implemented!')
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def cf_negative_sampling(self):
        print('CF negative sampling...')
        pos_edge_index_trans_np = self.edge_index_nps['user2item'].T
        if self.cf_loss_type == 'BCE':
            pos_samples_np = np.hstack([pos_edge_index_trans_np, np.ones((pos_edge_index_trans_np.shape[0], 1))])

            neg_inids = []
            u_nids = pos_samples_np[:, 0]
            p_bar = tqdm.tqdm(u_nids)
            for u_nid in p_bar:
                neg_inids.append(
                    self._cf_negative_sampling(
                        u_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_unid_inid_map,
                            self.test_pos_unid_inid_map,
                            self.neg_unid_inid_map
                        ),
                        self.item_nid_occs
                    )
                )
            neg_inids_np = np.vstack(neg_inids)
            neg_samples_np = np.hstack(
                [
                    np.repeat(pos_samples_np[:, 0].reshape(-1, 1), repeats=self.num_negative_samples, axis=0),
                    neg_inids_np,
                    torch.zeros((neg_inids_np.shape[0], 1)).long()
                ]
            )

            train_data_np = np.vstack([pos_samples_np, neg_samples_np])
        elif self.cf_loss_type == 'BPR':
            neg_inids = []
            u_nids = pos_edge_index_trans_np[:, 0]
            p_bar = tqdm.tqdm(u_nids)
            for u_nid in p_bar:
                neg_inids.append(
                    self._cf_negative_sampling(
                        u_nid,
                        self.num_negative_samples,
                        (
                            self.train_pos_unid_inid_map,
                            self.test_pos_unid_inid_map,
                            self.neg_unid_inid_map
                        ),
                        self.item_nid_occs
                    )
                )

            train_data_np = np.hstack(
                [
                    np.repeat(pos_edge_index_trans_np, repeats=self.num_negative_samples, axis=0),
                    np.vstack(neg_inids)
                ]
            )
        else:
            raise NotImplementedError('No negative sampling for loss type: {}.'.format(self.cf_loss_type))
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


if __name__ == '__main__':
    import os.path as osp

    root = osp.join('.', 'tmp', 'ml')
    name = '1m'
    seed = 2020
    dataset = MovieLens(root=root, name='1m', seed=seed)
    dataloader = DataLoader(dataset)
    for u_nids, pos_inids, neg_inids in dataloader:
        pass
    print('stop')

