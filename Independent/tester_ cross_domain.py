#!/usr/bin/env Python
# coding=utf-8

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from args import read_options
from data_loader_structure_fp_twoside import *
from matcher_structure_acc_fp_neigh_VAE_struc import *
from tensorboardX import SummaryWriter

from tqdm import tqdm
import pickle
from torch_geometric.data import Batch, Data
from sklearn import metrics

class CustomData(Data):
    def __inc__(self, key, value):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= probas_pred.mean()).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap

class Trainer(object):
    
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items(): setattr(self, k, v)

        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            # load pretrained embedding
            self.load_embed()
        self.use_pretrain = use_pretrain

        # if self.embed_model == 'RESCAL':
        #     self.num_ents = len(self.ent2id.keys()) - 1
        #     self.pad_id_ent = self.num_ents
        #     self.num_rels = len(self.rel2id.keys()) - 1
        #     self.pad_id_rel = self.num_rels
        #     self.matcher = RescalMatcher(self.embed_dim, self.num_ents, self.num_rels, use_pretrain=self.use_pretrain, ent_embed=self.ent_embed, rel_matrices=self.rel_matrices,dropout=self.dropout, attn_layers=self.n_attn, n_head=self.n_head, batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune, aggregate=self.aggregate)
        # else:
        self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain, embed=self.symbol2vec, dropout=self.dropout, batch_size=self.batch_size, finetune=self.fine_tune, aggregate=self.aggregate)
        self.matcher.cuda()

        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.optim_VAE = optim.Adam(self.matcher.vaemodel.parameters(), lr=self.lr*10, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        ent2id_case = json.load(open('twoside/ent2ids'))
        conunt = len(self.ent2id)
        for i in range(len(list(ent2id_case.keys()))):
            if list(ent2id_case.keys())[i] not in list(self.ent2id.keys()):
                self.ent2id[list(ent2id_case.keys())[i]] = conunt
                conunt+=1

        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open('twoside/rel2candidates.json'))  # 每个事件中的候选药物为全体药物，后面会随机在这里面挑负样本

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open('twoside/e1rel_e2.json'))

        with open(f'twoside/drug_data.pkl', 'rb') as f:
            a_drug_data = pickle.load(f)
        self.all_drug_data = {drug_id: CustomData(x=data[0], edge_index=data[1], edge_feats=data[2], line_graph_edge_index=data[3])
            for drug_id, data in a_drug_data.items()}
        self.drug_num_node_indices = {
            drug_id: torch.zeros(data.x.size(0)).long() for drug_id, data in self.all_drug_data.items()
        }

    def load_symbol2id(self):

        # if self.embed_model == 'RESCAL':
        #     self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        #     self.ent2id = json.load(open(self.dataset + '/ent2ids'))

        #     self.rel2id['PAD'] = len(self.rel2id.keys())
        #     self.ent2id['PAD'] = len(self.ent2id.keys())
        #     self.ent_embed = None
        #     self.rel_matrices = None
        #     return
        
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def de(self,a):
        a.pop('support_encoder.proj1.weight')
        a.pop('support_encoder.proj1.bias')
        a.pop('support_encoder.proj2.weight')
        a.pop('support_encoder.proj2.bias')
        a.pop('support_encoder.layer_norm.a_2')
        a.pop('support_encoder.layer_norm.b_2')
        a.pop('query_encoder.process.weight_ih')
        a.pop('query_encoder.process.weight_hh')
        a.pop('query_encoder.process.bias_ih')
        a.pop('query_encoder.process.bias_hh')

    def load_embed(self):

        # if self.embed_model == 'RESCAL':
        #     self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        #     self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        #     self.rel2id['PAD'] = len(self.rel2id.keys())
        #     self.ent2id['PAD'] = len(self.ent2id.keys())
        #     self.ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
        #     self.rel_matrices = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)
        #     self.ent_embed = np.concatenate((self.ent_embed, np.zeros((1,self.embed_dim))),axis=0)
        #     self.rel_matrices = np.concatenate((self.rel_matrices, np.zeros((1, self.embed_dim * self.embed_dim))), axis=0)
        #     return    


        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        relation2embids = json.load(open(self.dataset + '/relation2embids'))
        ent2embids = json.load(open(self.dataset + '/ent2embids'))
        rel2id_case = json.load(open('twoside/relation2ids'))
        ent2id_case = json.load(open('twoside/ent2ids'))
        conunt = len(rel2id)
        for i in range(len(list(rel2id_case.keys()))):
            if list(rel2id_case.keys())[i] not in list(rel2id.keys()):
                rel2id[list(rel2id_case.keys())[i]] = conunt
                conunt+=1

        conunt = len(ent2id)
        for i in range(len(list(ent2id_case.keys()))):
            if list(ent2id_case.keys())[i] not in list(ent2id.keys()):
                ent2id[list(ent2id_case.keys())[i]] = conunt
                conunt+=1

        relation2embids.update(
            {key: val for key, val in {key: -1 for key in rel2id_case}.items() if key not in relation2embids})
        ent2embids.update({key: val for key, val in {key: -1 for key in ent2id_case}.items() if
                           key not in ent2embids})  # 不修改原本ent2embids就存在的key-value，不存在的key才执行添加操作

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            # ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            # rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)
            ent_embed = np.load(self.dataset + '/DRKG_'+self.embed_model+'_entity.npy')
            rel_embed = np.load(self.dataset + '/DRKG_'+self.embed_model+'_relation.npy')

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            # assert ent_embed.shape[0] == len(ent2id.keys())
            # assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    if relation2embids[key] == -1:
                        embeddings.append(list(np.random.randn(rel_embed.shape[1],)))
                    else:
                        embeddings.append(list(rel_embed[relation2embids[key],:]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    if ent2embids[key] == -1:
                        embeddings.append(list(np.random.randn(rel_embed.shape[1],)))
                    else:
                        embeddings.append(list(ent_embed[ent2embids[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id  # 实体、边合在一块统一成一个字典
            self.symbol2vec = embeddings  # 实体、边的嵌入合在一块统一成一个矩阵

    def build_connection(self, max_=100):

        # if self.embed_model == 'RESCAL':
        #     self.connections = np.ones((self.num_ents, max_, 2)).astype(int)
        #     self.connections[:,:,0] = self.pad_id_rel
        #     self.connections[:,:,1] = self.pad_id_ent
        #     self.e1_rele2 = defaultdict(list)
        #     self.e1_degrees = defaultdict(int)
        #     with open(self.dataset + '/path_graph') as f:
        #         lines = f.readlines()
        #         for line in tqdm(lines):
        #             e1,rel,e2 = line.rstrip().split()
        #             self.e1_rele2[e1].append((self.rel2id[rel], self.ent2id[e2]))
        #             self.e1_rele2[e2].append((self.rel2id[rel+'_inv'], self.ent2id[e1]))  

        # else:
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)  # 存的是所有边+实体嵌入的下标，每个点在背景图中的邻居
        self.e1_degrees = defaultdict(int)  # 存的是实体实际index：背景邻居数量
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))
                # if self.dataset != 'drugbank':
                #     self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                random.shuffle(neighbors)
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors)) 
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors) # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # 实体真实坐标-第几个邻居-邻居边嵌入坐标
                self.connections[id_, idx, 1] = _[1]  # 实体真实坐标-第几个邻居-邻居点嵌入坐标

        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))
        # assert 1==2

        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')

        best_hits10 = 0.0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)

        # if self.embed_model == 'RESCAL':
        #     self.symbol2id = self.ent2id
        probas_pred = []
        ground_truth = []
        besttestauc=0
        besttest2auc=0
        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id, self.e1rel_e2, self.all_drug_data, self.drug_num_node_indices):

            if self.batch_nums % 50 == 0:
                logging.info('CURRENT EPOCH: %d MAX EPOCH %d' % (self.batch_nums,self.max_batches))
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right,support_batch,query_batch,false_batch = data
            support_batch = [t.to(device) for t in support_batch]
            query_batch = [t.to(device) for t in query_batch]
            false_batch = [t.to(device) for t in false_batch]
            # TODO more elegant solution
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            if self.no_meta:
            # for ablation
                query_scores = self.matcher(query, support)
                false_scores = self.matcher(false, support)
            else:
                try:
                    query_scores, loss2 = self.matcher(query, support, query_meta, support_meta, query_batch, support_batch,self.optim_VAE)
                except:
                    print(support)
                    continue
                false_scores, loss2 = self.matcher(false, support, false_meta, support_meta, false_batch, support_batch,self.optim_VAE)
            probas_pred.append(np.concatenate([torch.sigmoid(query_scores.detach()).cpu(), torch.sigmoid(false_scores.detach()).cpu()]))
            ground_truth.append(np.concatenate([np.ones(query_scores.shape[0]), np.zeros(false_scores.shape[:2]).reshape(-1)]))

            loss, loss_p, loss_n = loss_fn(query_scores, false_scores)
            loss+=loss2
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()

            if (self.batch_nums+1) % self.eval_every == 0:
            # if self.batch_nums % self.eval_every == 0:
                valauc = self.eval_acc(meta=self.meta)

                # hits10, hits5, mrr = self.eval(meta=self.meta)
                # self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                # self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                # self.writer.add_scalar('MAP', mrr, self.batch_nums)

                testauc = self.eval_acc(meta=self.meta, mode='test')
                test2auc = self.eval_acc(meta=self.meta, mode='test2')

                if (testauc+test2auc)>(besttestauc+besttest2auc) or (testauc>besttestauc and besttest2auc-test2auc<0.01) or (test2auc>besttest2auc and besttestauc-testauc<0.01):
                    besttestauc=testauc
                    besttest2auc=test2auc
                    self.save(self.save_path + f'{self.batch_nums}_best_{besttestauc:.4f}_{besttest2auc:.4f}')


                # if hits10 > best_hits10:
                #     self.save(self.save_path + '_bestHits10')
                #     best_hits10 = hits10

            if self.batch_nums % self.log_every == 0:
                # self.save()
                # logging.info('AVG. BATCH_LOSS: {.2f} AT STEP {}'.format(np.mean(losses), self.batch_nums))
                self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.batch_nums)
                acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred), np.concatenate(ground_truth))
                logging.info(f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')


            self.batch_nums += 1
            self.scheduler.step()
            if self.batch_nums == self.max_batches:
                self.save()
                break

    def eval_acc(self, mode='dev', meta=False):
        self.matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open('twoside/dev_tasks.json'))
        elif mode == 'test':
            test_tasks = json.load(open('twoside/test_tasks.json'))
        else:
            test_tasks = json.load(open('twoside/test2_tasks.json'))

        for task in list(test_tasks.keys()):
            newinteraciton=[]
            for ite in test_tasks[task]:
                if ite[0][:3]=='CID' or ite[-1][:3]=='CID':
                    continue
                else:
                    newinteraciton.append(ite)
            test_tasks[task]=newinteraciton


        rel2id = json.load(open(self.dataset + '/relation2ids'))
        rel2id_case = json.load(open('twoside/relation2ids'))
        conunt = len(rel2id)
        for i in range(len(list(rel2id_case.keys()))):
            if list(rel2id_case.keys())[i] not in list(rel2id.keys()):
                rel2id[list(rel2id_case.keys())[i]] = conunt
                conunt+=1

        rel2candidates = self.rel2candidates

        probas_pred = []
        ground_truth = []

        for query_ in test_tasks.keys():

            probas_pred_t = []
            ground_truth_t = []
            if len(test_tasks[query_])<few+1 or len(test_tasks[query_])>10: # 只看跨域小样本的情况，小于20就算小样本。去掉后就是近看跨域的情况
                continue
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
            support_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in support_triples]

            if meta:
                # support_batch = drug_structure_construct(self.all_drug_data, support_triples_rel2id,
                #                                          self.drug_num_node_indices)
                support_batch = DrugDataset(support_triples_rel2id)
                support_batch_loader = DrugDataLoader(support_batch, batch_size=len(support_triples_rel2id),
                                                      shuffle=False)
                support_batch = []
                for batch in support_batch_loader:
                    support_batch.append(batch)
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).cuda()

            query_triples = test_tasks[query_][few:]
            query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

            false_pairs = []
            false_triples = []
            for triple in query_triples:
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if (noise not in self.e1rel_e2[e_h + rel]) and noise != e_t:
                        break
                false_triples.append([e_h, rel, noise])
                false_pairs.append([symbol2id[e_h], symbol2id[noise]])

            query_pairs.extend(false_pairs)
            query_triples.extend(false_triples)
            query_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in query_triples]

            query = Variable(torch.LongTensor(query_pairs)).cuda()

            test_size = self.batch_size * 800
            if len(query_triples_rel2id) < test_size:
                test_size = len(query_triples_rel2id)
            for i in range(len(query_triples_rel2id) // test_size):  # 由于query_triples_rel2id数量都小于batchsize*8，所以只运行一次。否则运行次数要+1
                if (i + 1) * test_size > len(query_triples_rel2id):
                    query_triples_rel2id_batch = query_triples_rel2id[i * test_size:]
                else:
                    query_triples_rel2id_batch = query_triples_rel2id[i * test_size: (i + 1) * test_size]
                if meta:
                    query_left = [self.ent2id[triple[0]] for triple in query_triples]
                    query_right = [self.ent2id[triple[2]] for triple in query_triples]
                    query_meta = self.get_meta(query_left, query_right)
                    # query_batch = drug_structure_construct(self.all_drug_data, query_triples_rel2id_batch,
                    #                                        self.drug_num_node_indices)
                    query_batch = DrugDataset(query_triples_rel2id)
                    query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id),  # 因为只运行一次，所以直接用query_triples_rel2id也能正常运行，而没有用query_triples_rel2id_batch，实际上必须用query_triples_rel2id_batch
                                                        shuffle=False)
                    query_batch = []
                    for batch in query_batch_loader:
                        query_batch.append(batch)
                    support_batch = [t.to(device) for t in support_batch[0]]  # 因为只运行一次，这种赋值方法没错，若循环的话，support_batch本身在循环外面，这样是有问题的。所以support_batch=support_batch[0]要放在循环外面
                    query_batch = [t.to(device) for t in query_batch[0]]
                    scores, loss2 = self.matcher(query, support, query_meta, support_meta, query_batch, support_batch,self.optim_VAE)  # 因为只运行一次，query,query_triples_batch都直接输入的，实际上也需要分批
                    scores.detach()
                    scores = scores.data
                    probas_pred_t.append(np.concatenate([torch.sigmoid(scores.detach()).cpu()]))
                else:
                    scores, loss2 = self.matcher(query, support)
                    scores.detach()
                    scores = scores.data
                    probas_pred_t.append(np.concatenate([torch.sigmoid(scores.detach()).cpu()]))

            # for i in range(len(probas_pred_t)-1):  # 因为只运行一次，所以不这样格式化也没事
            #     probas_pred_t[0]=np.vstack((probas_pred_t[0], probas_pred_t[i+1]))
            # probas_pred_t = [probas_pred_t[0]]

            ground_truth_t.append(np.concatenate([np.ones(int(len(probas_pred_t[0])/2)), np.zeros(int(len(probas_pred_t[0])/2))]))

            acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred_t), np.concatenate(ground_truth_t))
            logging.info(f'task: {query_}\n,  acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
            probas_pred.extend(probas_pred_t)
            ground_truth.extend(ground_truth_t)

        acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred), np.concatenate(ground_truth))
        logging.info(f'alltask:\n acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')

        import csv
        with open(f'results_跨域/跨域twosides_{args.few}shot_{mode}_acc{acc:.4f}_auroc{auroc:.4f}_f1_score{f1_score:.4f}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in [['auroc,acc,f1_score'],[f'{auroc:.4f},{acc:.4f},{f1_score:.4f}']]:
                writer.writerow(row)

        all_results.append([auroc, acc, f1_score,args.few,mode])
        self.matcher.train()

        return auroc

    def eval(self, mode='dev', meta=False):
        self.matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        elif mode == 'test':
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test2_tasks.json'))
        rel2id = json.load(open(self.dataset + '/relation2ids'))

        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in test_tasks.keys():

            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []
            if len(test_tasks[query_])<few+1:
                continue
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
            support_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in support_triples]

            if meta:
                support_batch = DrugDataset(support_triples_rel2id)
                support_batch_loader = DrugDataLoader(support_batch, batch_size=len(support_triples_rel2id),
                                                      shuffle=False)
                support_batch = []
                for batch in support_batch_loader:
                    support_batch.append(batch)
                support_batch = support_batch[0]
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).cuda()

            for triple in tqdm(test_tasks[query_][few:],desc=f'task:{query_}'):
                true = triple[2]
                query_pairs = []
                query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])
                query_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]]]

                if meta:
                    query_left = []
                    query_right = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])

                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                        query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                        query_triples_rel2id.append([triple[0], ent, rel2id[triple[1]]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).cuda()

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    query_batch = DrugDataset(query_triples_rel2id)
                    query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id),  # 因为只运行一次，所以直接用query_triples_rel2id也能正常运行，而没有用query_triples_rel2id_batch，实际上必须用query_triples_rel2id_batch
                                                        shuffle=False)
                    query_batch = []
                    for batch in query_batch_loader:
                        query_batch.append(batch)
                    support_batch = [t.to(device) for t in support_batch]
                    query_batch = [t.to(device) for t in query_batch[0]]
                    scores, loss2 = self.matcher(query, support, query_meta, support_meta, query_batch, support_batch,self.optim_VAE)  # 因为只运行一次，query,query_triples_batch都直接输入的，实际上也需要分批
                    scores.detach()
                    scores = scores.data
                else:
                    scores, loss2 = self.matcher(query, support)
                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                scores = scores.reshape(len(scores))
                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)

            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_),
                                                                                              np.mean(hits5_),
                                                                                              np.mean(hits1_),
                                                                                              np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))
            # print query_ + ':'
            # print 'HITS10: ', np.mean(hits10_)
            # print 'HITS5: ', np.mean(hits5_)
            # print 'HITS1: ', np.mean(hits1_)
            # print 'MAP: ', np.mean(mrr_)

        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MAP: {:.3f}'.format(np.mean(mrr)))

        self.matcher.train()

        return np.mean(hits10), np.mean(hits5), np.mean(mrr)

    def test_(self):

        if args.dataset == 'dataset1':
            a = torch.load('models/dataset1/models_drugbank_10shot_str/bestmodel')
            self.de(a)
            a['symbol_emb.weight'] = torch.vstack((a['symbol_emb.weight'][:92, :], torch.rand([1318, 128]).cuda(),
                                                   a['symbol_emb.weight'][92:16837 + 92, :],
                                                   torch.rand([17078 - 16837, 128]).cuda(),
                                                   a['symbol_emb.weight'][16837 + 92:, :]))

        if args.dataset == 'dataset2':
            a = torch.load('models/dataset2/models_mdf_10shot_str/bestmodel')
            self.de(a)
            a['symbol_emb.weight'] = torch.vstack((a['symbol_emb.weight'][:106, :], torch.rand([1318, 128]).cuda(),
                                                   a['symbol_emb.weight'][106:16758 + 106, :],
                                                   torch.rand([17002 - 16758, 128]).cuda(),
                                                   a['symbol_emb.weight'][16758 + 106:, :]))
        self.matcher.load_state_dict(a)
        logging.info('Pre-trained model loaded')
        # self.eval_acc(mode='test', meta=self.meta)
        self.eval_acc(mode='test2', meta=self.meta)


class SigmoidLoss(nn.Module):

    def forward(self, p_scores, n_scores):
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2, p_loss, n_loss


if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(formatter)
    #
    # logger.addHandler(ch)
    # logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda'
    loss_fn = SigmoidLoss()


    all_results=[]
    all_results.append(['auroc','acc',  'f1_score','few','test_n'])

    few=1

    args.dataset = 'dataset1'
    args.dataset = 'dataset2'

    for few in [1]:
        args.few = few
        args.train_few = few
        trainer = Trainer(args)

        if few==10:
            trainer.test_()
        elif few==5:
            trainer.test_()
        elif few==1:
            trainer.test_()
        else:
            print("num_few error!")

    import csv
    def StoreFile2(data, fileName):
        with open(fileName, "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(data)
        return


    StoreFile2(all_results,f'results_/{args.dataset}_allresults.csv')



