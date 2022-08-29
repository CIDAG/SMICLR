# ===================================================
# CONTRASTIVE MODEL
# ===================================================

import numpy as np
import math
import torch
import torch.nn.functional as F


from torch import nn
from torch_geometric.nn import GINConv, global_add_pool



class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn_ = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                nn_ = nn.Sequential(nn.Linear(num_features, dim), nn.ReLU(), nn.Linear(dim, dim))
            conv = GINConv(nn_)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)


    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x # , torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class SMILESEncoder(nn.Module):
  
    def __init__(
        self, vocab_size, max_len, padding_idx, embedding_dim=64,
        dim=128, num_layers=1, bidirectional=False
    ):
        super(SMILESEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
          nn.Embedding(
              num_embeddings=self.vocab_size,
              embedding_dim=self.embedding_dim,
              padding_idx=self.padding_idx
          ),
          nn.LSTM(
              self.embedding_dim,
              self.dim,
              self.num_layers,
              batch_first=True,
              bidirectional=self.bidirectional
          )
        )

    def forward(self, data):
        x = data.smi.view(-1, self.max_len)
        feat, (_, _) = self.encoder(x)
        return feat[:,-1]


class Net(nn.Module):
    def __init__(self, num_features, dim, vocab_size, max_len, padding_idx,
        embedding_dim=64, num_layers=1, bidirectional=False, num_gc_layers=5):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers
        self.num_features = num_features

        # Encoders
        self.MPNN = Encoder(self.num_features, self.dim, num_gc_layers)
        self.SMIEnc = SMILESEncoder(self.vocab_size, self.max_len, self.padding_idx,
            self.embedding_dim, num_gc_layers * self.dim, self.num_layers, self.bidirectional)

        # Projection head
        self.g = nn.Sequential(
            nn.Linear(num_gc_layers * self.dim, 4 * self.dim),
            nn.BatchNorm1d(4 * self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.dim, 6 * self.dim, bias=False)
        )

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, data):
        enc1 = self.MPNN(data.x, data.edge_index, data.batch)
        enc2 = self.SMIEnc(data)
        return F.normalize(self.g(enc1), dim=1), F.normalize(self.g(enc2), dim=1)

def nt_xent_loss(out_1,out_2, temperature, eps=1e-6):
  
    out = torch.cat([out_1, out_2], dim=0)
      
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)
      
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    
    return -torch.log(pos / (neg + eps)).mean()
