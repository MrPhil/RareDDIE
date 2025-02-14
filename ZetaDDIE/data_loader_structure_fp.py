import json
import random
from tqdm import tqdm
import logging
import torch
from torch_geometric.data import Batch, Data
from data_preprocessing_t import *
class PairData(Data):

    def __init__(self, j_indices, i_indices, pair_edge_index):
        super().__init__()
        self.i_indices = i_indices
        self.j_indices = j_indices
        self.edge_index = pair_edge_index
        self.num_nodes = None

    def __inc__(self, key, value):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'edge_index':
            return torch.tensor([[self.j_indices.shape[0]], [self.i_indices.shape[0]]])
        if key in ('i_indices', 'j_indices'):
            return 1
        return super().__inc__(key, value)
            # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
            # Replace with "return super().__inc__(self, key, value, args, kwargs)"

def train_generate_simple(dataset, batch_size, few, symbol2id):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]
        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]  

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)
        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

        false_pairs = []
        for triple in query_triples:
            e_h = triple[0]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)
                if noise != e_t:
                    break
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])

        yield support_pairs, query_pairs, false_pairs

def drug_structure_construct(all_drug_data,support_triples_rel2id,drug_num_node_indices):
    def _get_new_batch_id_and_num_nodes(old_id, old_id_to_new_batch_id, batch_drug_feats):
        new_id = old_id_to_new_batch_id.get(old_id, -1)
        num_nodes = all_drug_data[old_id].x.size(0)
        if new_id == - 1:
            new_id = len(old_id_to_new_batch_id)
            old_id_to_new_batch_id[old_id] = new_id
            batch_drug_feats.append(all_drug_data[old_id])
            start = (node_ind_seqs[-1][-1] + 1) if len(node_ind_seqs) else 0
            node_ind_seqs.append(torch.arange(num_nodes) + start)
        return new_id, num_nodes

    def _get_combo_index(combo, old_combo, already_in_combo, unique_pairs, num_nodes):
        idx = already_in_combo.get(combo, -1)
        if idx == -1:
            idx = len(already_in_combo)
            already_in_combo[combo] = idx
            pair_edge_index = bipartite_edge_dict.get(old_combo)
            if pair_edge_index is None:
                index_j = torch.arange(num_nodes[0]).repeat_interleave(num_nodes[1])
                index_i = torch.arange(num_nodes[1]).repeat(num_nodes[0])
                pair_edge_index = torch.stack([index_j, index_i])
                bipartite_edge_dict[old_combo] = pair_edge_index
            j_num_indices, i_num_indices = drug_num_node_indices[old_combo[0]], drug_num_node_indices[old_combo[1]]
            unique_pairs.append(PairData(j_num_indices, i_num_indices, pair_edge_index))
            node_j_ind_seqs_for_pair.append(node_ind_seqs[combo[0]])
            node_i_ind_seqs_for_pair.append(node_ind_seqs[combo[1]])
            drug_node_num_pair_list.append([len(node_ind_seqs[combo[0]]),len(node_ind_seqs[combo[1]])])
            drug_pair_list.append([combo[0], combo[1]])
        return idx

    bipartite_edge_dict = dict()

    old_id_to_new_batch_id = {}
    batch_drug_feats = []
    node_ind_seqs = []
    node_i_ind_seqs_for_pair = []
    node_j_ind_seqs_for_pair = []
    drug_pair_list = []
    drug_node_num_pair_list = []

    combo_indices_pos = []
    already_in_combo = {}
    rels = []
    batch_unique_pairs = []

    for ind, pos_item in enumerate(support_triples_rel2id):
        h, r, t = pos_item[:3]
        idx_h, h_num_nodes = _get_new_batch_id_and_num_nodes(h, old_id_to_new_batch_id, batch_drug_feats)
        idx_t, t_num_nodes = _get_new_batch_id_and_num_nodes(t, old_id_to_new_batch_id, batch_drug_feats)
        combo_idx = _get_combo_index((idx_h, idx_t), (h, t), already_in_combo, batch_unique_pairs,
                                          (h_num_nodes, t_num_nodes))
        combo_indices_pos.append(combo_idx)
        rels.append(int(r))

    batch_drug_data = Batch.from_data_list(batch_drug_feats, follow_batch=['edge_index'])
    drug_pair_list = torch.tensor(drug_pair_list)
    drug_node_num_pair_list = torch.tensor(drug_node_num_pair_list)
    batch_drug_pair_indices = torch.LongTensor(combo_indices_pos)
    batch_unique_drug_pair = Batch.from_data_list(batch_unique_pairs, follow_batch=['edge_index'])
    node_j_for_pairs = torch.cat(node_j_ind_seqs_for_pair)
    node_i_for_pairs = torch.cat(node_i_ind_seqs_for_pair)
    rels = torch.LongTensor(rels)

    return (batch_drug_data, batch_unique_drug_pair, rels, batch_drug_pair_indices, node_j_for_pairs, node_i_for_pairs,drug_pair_list,drug_node_num_pair_list)

def train_generate(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2, all_drug_data, drug_num_node_indices):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0
    rel2id = json.load(open(dataset + '/relation2ids'))
    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]

        if len(candidates) <= 20:
            print('not enough candidates')
            continue

        train_and_test = train_tasks[query]

        random.shuffle(train_and_test)

        support_triples = train_and_test[:few]

        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        support_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in support_triples]
        support_batch = DrugDataset(support_triples_rel2id)
        support_batch_loader = DrugDataLoader(support_batch, batch_size=len(support_triples_rel2id), shuffle=False)
        support_batch=[]
        for batch in support_batch_loader:
            support_batch.append(batch)

        all_test_triples = train_and_test[few:]

        if len(all_test_triples) == 0:
            continue

        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        query_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in query_triples]
        query_batch = DrugDataset(query_triples_rel2id)
        query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id), shuffle=False)
        query_batch=[]
        for batch in query_batch_loader:
            query_batch.append(batch)

        false_pairs = []
        false_left = []
        false_right = []
        false_triples = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            while True:
                noise = random.choice(candidates)
                if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                    break
            false_triples.append([e_h,rel,noise])
            false_pairs.append([symbol2id[e_h], symbol2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])

        false_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in false_triples]
        false_batch = DrugDataset(false_triples_rel2id)
        false_batch_loader = DrugDataLoader(false_batch, batch_size=len(false_triples_rel2id), shuffle=False)
        false_batch=[]
        for batch in false_batch_loader:
            false_batch.append(batch)

        yield query, support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right,support_batch[0],query_batch[0],false_batch[0]

def train_generate_(dataset, batch_size, few, symbol2id, ent2id, e1rel_e2, num_neg=1):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]
        rel_idx += 1
        candidates = rel2candidates[query]
        train_and_test = train_tasks[query]

        random.shuffle(train_and_test)
        support_triples = train_and_test[:few]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
        support_left = [ent2id[triple[0]] for triple in support_triples]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        all_test_triples = train_and_test[few:]
        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]
        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        labels = [1] * len(query_triples)

        for triple in query_triples:
            e_h = triple[0]
            e_t = triple[2]

            if e_t in candidates: candidates.remove(e_t)

            if len(candidates) >=num_neg:
                noises = random.sample(candidates, num_neg)
            else:
                noises = candidates

            for noise in noises:
                query_pairs.append([symbol2id[e_h], symbol2id[noise]])
                query_left.append(ent2id[e_h])
                query_right.append(ent2id[noise])
                labels.append(0)

        yield support_pairs, query_pairs, support_left, support_right, query_left, query_right, labels





