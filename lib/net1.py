# lib/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

class GNN_Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(3, 16); self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16,32); self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = GCNConv(32,48); self.bn3 = nn.BatchNorm1d(48)
        self.conv4 = GCNConv(48,64); self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = GCNConv(64,96); self.bn5 = nn.BatchNorm1d(96)

        self.fc1 = nn.Linear(96,128); self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,96); self.bn7 = nn.BatchNorm1d(96)

        self.head_pose  = nn.Linear(96, 2)
        self.head_depth = nn.Linear(96, 1)
        self.head_cls   = nn.Linear(96, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = F.leaky_relu(self.bn4(self.conv4(x, edge_index)))
        x = F.leaky_relu(self.bn5(self.conv5(x, edge_index)))
        x = scatter_mean(x, batch, dim=0)
        x = F.leaky_relu(self.bn6(self.fc1(x)))
        feat = F.leaky_relu(self.bn7(self.fc2(x)))
        return {
            'pose':   self.head_pose(feat),
            'depth':  self.head_depth(feat),
            'logits': self.head_cls(feat),
            'feat':   feat
        }
