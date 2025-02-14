#!/usr/bin/env Python
# coding=utf-8


from args import read_options
from data_loader_structure_fp import *
from matcher_structure_acc_fp_neigh_VAE_struc import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn import metrics

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
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
            self.load_embed()
        self.use_pretrain = use_pretrain
        self.num_symbols = len(self.symbol2id.keys()) - 1
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
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        self.all_drug_data = {}
        self.drug_num_node_indices = {}

    def load_symbol2id(self):
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

    def load_embed(self):

        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        relation2embids = json.load(open(self.dataset + '/relation2embids'))
        ent2embids = json.load(open(self.dataset + '/ent2embids'))


        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.load(self.dataset + '/DRKG_'+self.embed_model+'_entity.npy')
            rel_embed = np.load(self.dataset + '/DRKG_'+self.embed_model+'_relation.npy')

            if self.embed_model == 'ComplEx':
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

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

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def build_connection(self, max_=100):

        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2 = line.rstrip().split('\t')
                self.e1_rele2[e1[-7:]].append((self.symbol2id[rel], self.symbol2id[e2]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                random.shuffle(neighbors)
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        return degrees

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

        return a
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

    def eval_acc(self, mode='dev', meta=False):
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

        probas_pred = []
        ground_truth = []

        for query_ in test_tasks.keys():

            probas_pred_t = []
            ground_truth_t = []
            if len(test_tasks[query_])<few+1:
                continue
            candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]
            support_triples_rel2id = [[triple[0], triple[2], rel2id[triple[1]]] for triple in support_triples]

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
            for i in range(len(query_triples_rel2id) // test_size):
                query_left = [self.ent2id[triple[0]] for triple in query_triples]
                query_right = [self.ent2id[triple[2]] for triple in query_triples]
                query_meta = self.get_meta(query_left, query_right)
                query_batch = DrugDataset(query_triples_rel2id)
                query_batch_loader = DrugDataLoader(query_batch, batch_size=len(query_triples_rel2id), shuffle=False)
                query_batch = []
                for batch in query_batch_loader:
                    query_batch.append(batch)
                support_batch = [t.to(device) for t in support_batch[0]]
                query_batch = [t.to(device) for t in query_batch[0]]
                scores, loss2 = self.matcher(query, support, query_meta, support_meta, query_batch, support_batch,self.optim_VAE)
                scores.detach()
                scores = scores.data
                probas_pred_t.append(np.concatenate([torch.sigmoid(scores.detach()).cpu()]))

            ground_truth_t.append(np.concatenate([np.ones(int(len(probas_pred_t[0])/2)), np.zeros(int(len(probas_pred_t[0])/2))]))
            loss, loss_p, loss_n = loss_fn(scores[:int(len(probas_pred_t[0])/2)], scores[int(len(probas_pred_t[0])/2):])
            acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred_t), np.concatenate(ground_truth_t))
            logging.info(f'task: {query_}\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')
            probas_pred.extend(probas_pred_t)
            ground_truth.extend(ground_truth_t)

        acc, auroc, f1_score, precision, recall, int_ap, ap = do_compute_metrics(np.concatenate(probas_pred), np.concatenate(ground_truth))
        logging.info(f'alltask:\n loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')

        import csv
        with open(f'results/{args.dataset}_{args.few}shot_{mode}_acc{acc:.4f}_auroc{auroc:.4f}_f1_score{f1_score:.4f}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in [['acc,auroc,f1_score'],[f'{acc:.4f},{auroc:.4f},{f1_score:.4f}']]:
                writer.writerow(row)

        all_results.append([acc, auroc, f1_score,args.few,mode])
        self.matcher.train()

        return auroc
    def test_(self,model_P):
        self.matcher.load_state_dict(torch.load(model_P))
        logging.info('Pre-trained model loaded')
        self.eval_acc(mode='test', meta=self.meta)
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

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = 'cuda'
    loss_fn = SigmoidLoss()


    all_results=[]
    all_results.append(['acc', 'auroc', 'f1_score','few','test_n'])

    few=1

    for few in [1,5,10]:
        args.few = few
        args.train_few = few
        args.dataset = 'toy_dataset'
        trainer = Trainer(args)

        if few==1:
            trainer.test_('models/intialbestmodel')
        elif few==5:
            trainer.test_('models/intialbestmodel')
        elif few==10:
            trainer.test_('models/intialbestmodel')
        else:
            print("num_few error!")

    import csv
    def StoreFile2(data, fileName):
        with open(fileName, "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(data)
        return


    StoreFile2(all_results,f'results/{args.dataset}_allresults.csv')

