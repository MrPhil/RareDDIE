import itertools
from collections import defaultdict
from operator import neg
import random
import math
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
os.chdir(sys.path[0])

df_drugs_smiles = pd.read_csv('toy_dataset/drug_smiles.csv')
DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}
DRUG_INDX_NAME_DICT = {indx: drug_id  for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}
drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]
drug_to_mol_graph = {id:Chem.MolFromSmiles(smiles.strip()) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])}

ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


fp = np.load('fp/features/morgan_toy_dataset.npz')
fp = fp['features']
fp = torch.tensor(np.where(np.isnan(fp), 0, fp), dtype=torch.float32)
desc = np.zeros([fp.size(0),200])
desc = torch.tensor(np.where(np.isnan(desc), 0, desc), dtype=torch.float32)


id_fp = {}
id_desc = {}
for i in range(len(DRUG_INDX_NAME_DICT)):
    id_fp[DRUG_INDX_NAME_DICT[i]] = fp[i, :]
for i in range(len(DRUG_INDX_NAME_DICT)):
    id_desc[DRUG_INDX_NAME_DICT[i]] = desc[i, :]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, 0, 0]
    bond_feats = bond_feats + one_of_k_encoding_unk(bond.GetIsConjugated(), [0, 1, "nonbond"]) + \
                                 one_of_k_encoding_unk(bond.IsInRing(), [0, 1, "nonbond"])
    bond_feats = bond_feats + one_of_k_encoding_unk(
                        str(bond.GetStereo()),
                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "nonbond"])
    results = np.array(bond_feats).astype(np.float32)
    return torch.from_numpy(results)

def get_mol_edge_list_and_feat_mtx(mol_graph):
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort()
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    e_features = [bond_features(b) for b in mol_graph.GetBonds()]
    if len(e_features):
        e_features = torch.stack(e_features) 
        e_features = torch.cat([e_features,e_features])
    else:
        undirected_edge_list = torch.LongTensor([(0, 0),(0, 0)])
        e_features = torch.tensor(np.zeros((2, 17))).float()

    return undirected_edge_list.T, n_features, e_features

def get_bipartite_graph(mol_graph_1,mol_graph_2):
    x1 = np.arange(0,len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0,len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    
    return edge_list

MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) 
                                for drug_id, mol in drug_id_mol_graph_tup}

MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])

def drug_structure_construct_peb(batch):
    def __create_graph_data(id):
        edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
        n_features = MOL_EDGE_LIST_FEAT_MTX[id][1]
        edge_attr = MOL_EDGE_LIST_FEAT_MTX[id][2]
        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_attr)

    def __create_graph_data_edge(id):
        edge_attr = MOL_EDGE_LIST_FEAT_MTX[id][2]

        return Data(x=edge_attr, edge_index=torch.LongTensor())

    pos_rels = []

    pos_h_samples = []
    pos_h_edge_samples = []
    pos_h_fp = []
    pos_h_desc = []

    pos_t_samples = []
    pos_t_edge_samples = []
    pos_t_fp = []
    pos_t_desc = []

    for h, t, r in batch:
        pos_rels.append(r)
        pos_h_samples.append(__create_graph_data(h))
        pos_h_fp.append(id_fp[h])
        pos_h_desc.append(id_desc[h])
        pos_t_samples.append(__create_graph_data(t))
        pos_t_fp.append(id_fp[t])
        pos_t_desc.append(id_desc[t])
        pos_h_edge_samples.append(__create_graph_data_edge(h))
        pos_t_edge_samples.append(__create_graph_data_edge(t))

    pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)
    pos_h_samples = Batch.from_data_list(pos_h_samples)
    pos_h_fp = torch.stack(pos_h_fp)
    pos_h_desc = torch.stack(pos_h_desc)

    pos_t_samples = Batch.from_data_list(pos_t_samples)
    pos_t_fp = torch.stack(pos_t_fp)
    pos_t_desc = torch.stack(pos_t_desc)

    pos_h_edge_samples = Batch.from_data_list(pos_h_edge_samples)
    pos_t_edge_samples = Batch.from_data_list(pos_t_edge_samples)

    pos_tri = (pos_h_samples, pos_h_fp, pos_h_desc, pos_t_samples, pos_t_fp, pos_t_desc, pos_rels, pos_h_edge_samples,
               pos_t_edge_samples)

    return pos_tri


class DrugDataset(Dataset):
    def __init__(self, tri_list, disjoint_split=True, shuffle=False):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        '''
        self.tri_list = []
        for h, t, r, *_ in tri_list:
            if ((h in MOL_EDGE_LIST_FEAT_MTX) and (t in MOL_EDGE_LIST_FEAT_MTX)):
                self.tri_list.append((h, t, r))
        d1, d2, *_ = zip(*self.tri_list)
        self.drug_ids = np.array(list(set(d1 + d2)))
        self.drug_ids = np.array([id for id in self.drug_ids if id in MOL_EDGE_LIST_FEAT_MTX])
        if shuffle:
            random.shuffle(self.tri_list)

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []

        pos_h_samples = []
        pos_h_edge_samples = []
        pos_h_fp = []
        pos_h_desc = []

        pos_t_samples = []
        pos_t_edge_samples = []
        pos_t_fp = []
        pos_t_desc = []


        for h, t, r in batch:
            pos_rels.append(r)
            pos_h_samples.append(self.__create_graph_data(h))
            pos_h_fp.append(id_fp[h])
            pos_h_desc.append(id_desc[h])
            pos_t_samples.append(self.__create_graph_data(t))
            pos_t_fp.append(id_fp[t])
            pos_t_desc.append(id_desc[t])
            pos_h_edge_samples.append(self.__create_graph_data_edge(h))
            pos_t_edge_samples.append(self.__create_graph_data_edge(t))


        pos_rels = torch.LongTensor(pos_rels).unsqueeze(0)
        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_h_fp = torch.stack(pos_h_fp)
        pos_h_desc = torch.stack(pos_h_desc)

        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_t_fp = torch.stack(pos_t_fp)
        pos_t_desc = torch.stack(pos_t_desc)

        pos_h_edge_samples = Batch.from_data_list(pos_h_edge_samples)
        pos_t_edge_samples = Batch.from_data_list(pos_t_edge_samples)

        pos_tri = (
        pos_h_samples, pos_h_fp, pos_h_desc, pos_t_samples, pos_t_fp, pos_t_desc, pos_rels, pos_h_edge_samples,
        pos_t_edge_samples)

        return pos_tri

    def __create_graph_data(self, id):
        edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
        n_features = MOL_EDGE_LIST_FEAT_MTX[id][1]
        edge_attr = MOL_EDGE_LIST_FEAT_MTX[id][2]
        return Data(x=n_features, edge_index=edge_index, edge_attr=edge_attr)

    def __create_graph_data_edge(self, id):
        edge_attr = MOL_EDGE_LIST_FEAT_MTX[id][2]

        return Data(x=edge_attr, edge_index=torch.LongTensor())

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

