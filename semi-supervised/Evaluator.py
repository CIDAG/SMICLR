# ===================================================
# SUPERVISED
# ===================================================
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


class Net(nn.Module):
    def __init__(self, num_features, dim):
        super(Net, self).__init__()
        self.dim = dim
        self.num_features = num_features

        # Encoders
        self.MPNN = MPNNEncoder(self.num_features, self.dim)

        self.fc1 = torch.nn.Linear(2 * self.dim, self.dim)
        self.fc2 = torch.nn.Linear(self.dim, 1)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        out = self.MPNN(data)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view(-1)


def test(model, loader, batch_size, temperature, device, std):

    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        error += (pred * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def train(model, loader, optimizer, batch_size, temperature, device, output):

    model.train()
    loss_all = 0

    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()

        loss = F.mse_loss(model(data), data.y)

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    torch.save(model.state_dict(), '{}/model_SUP.pth'.format(output))
    return loss_all / len(loader.dataset)
