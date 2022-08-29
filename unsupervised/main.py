# ===================================================
# IMPORTING PACKAGES
# ===================================================

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch


from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
from MUTAG import MUTAG
from SMICLR import *
from torch_geometric.data import DataLoader


# ===================================================
# BASIC FUNCTIONS
# ===================================================

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
  
    max_len = 90  # mudar aqui
    char_indices = {
        'c': 0, '1': 1, '2': 2, '(': 3, ')': 4, '3': 5, '[': 6, 'N': 7, '+': 8, ']': 9, '=': 10, 
        'O': 11, '-': 12, 'n': 13, '4': 14, 'F': 15, 'C': 16, '#': 17, 'I': 18, '5': 19, '6': 20,
        '7': 21, '&': 22, '*': 23, 'o': 24, ' ': 25
    }
    nchars = len(char_indices)
    accuracies = {'val':[], 'test':[]}
    log_interval = 1
    args = arg_parse()
    
    seed_everything(args.seed)
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    bidirectional = args.bidirectional
    num_gc_layers = args.num_gc_layers
    num_layers = args.num_layers
    temperature = args.temperature
    dataset_path = args.dataset_path

    ratio = args.ratio
    
    if args.node_drop:
        graph_aug = 'dnodes'
    elif args.subgraph:
        graph_aug = 'subgraph'
    elif args.edge_pertubation:
        graph_aug = 'pedges'
    elif args.attribute_masking:
        graph_aug = 'mask_nodes'
    else:
        graph_aug = 'none'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Treinando com {} ...'.format(device))
    
    dataset = MUTAG(dataset_path, graph_aug=graph_aug, ratio=ratio).shuffle()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    dataset1 = MUTAG(dataset_path).shuffle()
    dataloader1 = DataLoader(dataset1, batch_size=batch_size)
    dataset.get_num_feature()
    
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1


    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('num_gc_layers: {}'.format(num_gc_layers))
    print('================')

    model = Net(
        dataset.num_features, 64, vocab_size=nchars, max_len=max_len,
        padding_idx=char_indices[' '], embedding_dim=32,
        num_layers=num_layers, bidirectional=bidirectional, num_gc_layers=num_gc_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb, y = model.MPNN.get_embeddings(dataloader1)
    print('===== Before training =====')
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)

    for epoch in range(1, epochs):
        loss_all = 0
        
        model.train()
        
        for data_aug in dataloader:

            node_num, _ = data_aug.x.size()
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]

            data_aug.batch = data_aug.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data = data_aug.to(device)
        
            out_1, out_2 = model(data)
            loss = nt_xent_loss(out_1, out_2, temperature)
            loss_all += loss.item()  * data.num_graphs
            loss.backward()
            optimizer.step()

        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.MPNN.get_embeddings(dataloader1)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('MUTAG: {},{},{},{}\n'.format(epochs, log_interval, lr, s))
