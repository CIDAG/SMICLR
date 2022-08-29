# ===================================================
# CONTRASTIVE MODEL
# ===================================================

import math
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import NNConv, Set2Set


class MPNNEncoder(nn.Module):
    def __init__(self, num_features, dim):
        super(MPNNEncoder, self).__init__()

        self.lin0 = nn.Linear(num_features, dim)

        mlp = nn.Sequential(nn.Linear(5, 128), nn.ReLU(),
                            nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, mlp, aggr='mean', root_weight=False)
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        return self.set2set(out, data.batch)


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
        return feat[:, -1]


class Net(nn.Module):
    def __init__(self, num_features, dim, vocab_size, max_len, padding_idx,
                 embedding_dim=64, num_layers=1, bidirectional=False):
        super(Net, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.dim = dim
        self.num_layers = num_layers
        self.num_features = num_features

        # Base encoders
        self.MPNN = MPNNEncoder(self.num_features, self.dim)
        self.SMIEnc = SMILESEncoder(
            self.vocab_size, self.max_len, self.padding_idx,
            self.embedding_dim, 2 * self.dim, 
            self.num_layers, self.bidirectional
        )

        # Projection head
        self.g = torch.nn.Sequential(
            nn.Linear(2 * self.dim, 4 * self.dim),
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
        enc1 = self.MPNN(data)
        enc2 = self.SMIEnc(data)
        return F.normalize(self.g(enc1), dim=1), F.normalize(self.g(enc2), dim=1)


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):

    out = torch.cat([out_1, out_2], dim=0)

    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(
        math.e**(1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    return -torch.log(pos / (neg + eps)).mean()


def test(model, loader, batch_size, temperature, device, std):

    model.eval()
    error = 0
    total_num = 0

    for data in loader:
        data = data.to(device)

        out_1, out_2 = model(data)

        loss = nt_xent_loss(out_1, out_2, temperature)

        error += loss.item() * batch_size
        total_num += batch_size

    return error / total_num


def train(model, unsup_loader, optimizer, batch_size, temperature, device, output):
    model.train()
    loss_all = 0
    total_num = 0

    for unsup in unsup_loader:

        unsup = unsup.to(device)

        optimizer.zero_grad()

        out_1, out_2 = model(unsup)

        loss = nt_xent_loss(out_1, out_2, temperature)
        loss.backward()

        total_num += batch_size
        loss_all += loss.item() * batch_size
        optimizer.step()

    torch.save(model.state_dict(), '{}/model_UNSUP.pth'.format(output))
    return loss_all / total_num
