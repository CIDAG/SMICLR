import numpy as np
import torch
import random


from rdkit import Chem
from torch_geometric.utils import remove_self_loops


class Complete(object):

    def __init__(
        self,
        augmentation=False,
        node_drop=False,
        subgraph=False,
        edge_perturbation=False,
        attribute_masking=False,
        masking_pos=False,
        enumeration=False,
        smiles_noise=False,
        aug_ratio=.2,
        xyz_pertub=.04,
        max_len=38
    ):

        self.augmentation = augmentation
        self.aug_ratio = aug_ratio 

        # Graph
        self.node_drop = node_drop
        self.subgraph = subgraph
        self.edge_perturbation = edge_perturbation
        self.attribute_masking = attribute_masking
        self.masking_pos = masking_pos
                
        self.r1 = -xyz_pertub
        self.r2 = xyz_pertub


        # SMILES
        self.enumeration = enumeration
        self.smiles_noise = smiles_noise

        self.max_len = max_len
        self.char_indices = {'C': 0, '5': 1, 'n': 2, 'o': 3, '(': 4, 'N': 5,
                             '4': 6, ')': 7, '@': 8, '2': 9, 'H': 10, '#': 11,
                             'O': 12, '1': 13, ']': 14, '[': 15, 'F': 16, '3': 17,
                             'c': 18, '=': 19, '-': 20, '+': 21, ' ': 22}

    def __call__(self, data):

        smi = data.smi

        if self.augmentation:

            if self.enumeration:
                mol = Chem.MolFromSmiles(smi)
                num, atoms_list = range(mol.GetNumAtoms()), mol.GetNumAtoms()

                smi = Chem.MolToSmiles(
                  Chem.RenumberAtoms(mol, random.sample(num, atoms_list)),
                  canonical=False, isomericSmiles=True
                )
            
            smi = self.pad_smile(smi, 'right')
            smi = np.array([self.char_indices[s] for s in smi])

            if self.smiles_noise:
                smi = self.add_noise_smiles(smi) 

            if self.node_drop:
                self.drop_nodes(data)
            if self.subgraph:
                self._subgraph(data)
            if self.edge_perturbation:
                self.permute_edges(data)
            if self.attribute_masking:
                self.mask_nodes(data)
            if self.masking_pos:
                self.mask_pos(data)

        else:

            smi = self.pad_smile(smi, 'right')
            smi = np.array([self.char_indices[s] for s in smi])

        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        data.smi = torch.from_numpy(smi)

        return data


    def pad_smile(self, string, padding='right'):
        if len(string) <= self.max_len:
            if padding == 'right':
                return string + " " * (self.max_len - len(string))
            elif padding == 'left':
                return " " * (self.max_len - len(string)) + string
            elif padding == 'none':
                return string
        else:
          raise ValueError('len(SMILES) > max_len')


    def add_noise_smiles(self, smi):

        noise_num = int(len(smi)  * self.aug_ratio)
        idx_perm = np.random.permutation(len(smi))

        idx_noise = idx_perm[:noise_num]
        smi[idx_noise] = self.char_indices[' ']

        return smi


    def drop_nodes(self, data):

        # Code from: https://github.com/Shen-Lab/GraphCL/blob/d857849d51bb168568267e07007c0b0c8bb6d869/transferLearning_MoleculeNet_PPI/chem/loader.py#L261

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * self.aug_ratio)

        idx_perm = np.random.permutation(node_num)

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

        edge_index = data.edge_index.numpy()

        edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
            
        try:
            data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
            data.x = data.x[idx_nondrop]
            data.z = data.z[idx_nondrop]
            data.pos = data.pos[idx_nondrop]
            data.edge_attr = data.edge_attr[edge_mask]
        except:
            data = data


    def _subgraph(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * self.aug_ratio)

        edge_index = data.edge_index.numpy()

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh)) 
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

        edge_index = data.edge_index.numpy()
        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
            data.x = data.x[idx_nondrop]
            data.z = data.z[idx_nondrop]
            data.pos = data.pos[idx_nondrop]
            data.edge_attr = data.edge_attr[edge_mask]
        except:
            data = data


    def permute_edges(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * self.aug_ratio)
        edge_index = data.edge_index.numpy()
        
        idx_remove = np.random.choice(edge_num, (permute_num), replace=False)
        edge_to_remove = edge_index[:, idx_remove]
        edge_to_remove = np.unique(np.hstack((edge_to_remove, edge_to_remove[[1,0]])), axis=1).transpose()
        edge_mask = np.array([n for n in range(edge_num) if not (edge_index[:, n] == edge_to_remove).all(1).any()])
        try:
            data.edge_index = torch.tensor(edge_index[:, edge_mask])
            data.edge_attr = data.edge_attr[edge_mask]
        except:
            data = data

    def mask_nodes(self, data):

        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * self.aug_ratio)
        x = data.x.clone().detach()

        token = x.mean(dim=0)
        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        x[idx_mask] = token
        data.x = x


    def mask_pos(self, data):

        device = data.pos.device
        data.pos = data.pos + ((self.r1 - self.r2) * torch.rand(data.pos.shape[0], data.pos.shape[1], device=device) + self.r2)


    def get_char_indices(self):
        return self.char_indices
