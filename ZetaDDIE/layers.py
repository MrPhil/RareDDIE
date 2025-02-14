import math
import datetime
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv,TransformerConv


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):      

        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)
        

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim,dp,head,edge,head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,head_out_feats//2,head,edge_dim=edge,dropout=dp)
    
    def forward(self,data):
        input_feature,edge_index = data.x, data.edge_index
        input_feature = F.relu(input_feature)
        intra_rep = self.intra(input_feature,edge_index,data.edge_attr)
        return intra_rep

# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim,dp,head,edge,head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim,input_dim),head_out_feats//2,head,dropout=dp)
    
    def forward(self,h_data,t_data,b_graph):
        edge_index = b_graph.edge_index
        h_input = F.relu(h_data.x)
        t_input = F.relu(t_data.x)
        t_rep = self.inter((h_input,t_input),edge_index)
        h_rep = self.inter((t_input,h_input),edge_index[[1,0]])
        return h_rep,t_rep

class MergeFD(nn.Module):
    def __init__(self, in_features_fp, in_features_desc, kge_dim):
        super().__init__()
        self.in_features_fp = in_features_fp
        self.in_features_desc = in_features_desc
        self.kge_dim = kge_dim
        self.reduction_fp = nn.Sequential(nn.Linear(self.in_features_fp, 512),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(512, self.kge_dim),
                                          nn.ELU(),
                                          nn.Dropout(0.3)
                                          )
        self.reduction_desc = nn.Sequential(nn.Linear(self.in_features_desc, 256),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(256, self.kge_dim),
                                          nn.ELU(),
                                          nn.Dropout(0.3))
        self.merge_fd = nn.Sequential(nn.Linear(self.kge_dim, self.kge_dim),
                                   nn.ELU())
        
    def forward(self,h_data_fin,h_data_desc,t_data_fin,t_data_desc):
        h_data_fin = F.normalize(h_data_fin, 2, 1)
        t_data_fin = F.normalize(t_data_fin, 2 ,1)
        h_data_fin = self.reduction_fp(h_data_fin)
        t_data_fin = self.reduction_fp(t_data_fin)
        h_fdmerge = h_data_fin
        h_fdmerge = F.normalize(h_fdmerge, 2, 1)
        h_fdmerge = self.merge_fd(h_fdmerge)
        t_fdmerge = t_data_fin
        t_fdmerge = F.normalize(t_fdmerge, 2, 1)
        t_fdmerge = self.merge_fd(t_fdmerge)

        return h_fdmerge, t_fdmerge, h_data_fin, h_data_desc, t_data_fin, t_data_desc


