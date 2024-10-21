from functools import partial
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from MSVGAE.MSVGAE_Encoder import GAT_Encoder, GCN_Encoder

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

class GCNEncoder(torch.nn.Module):
    def __init__(self, num_layers,norm,feat_drop,in_dim,out_dim,num_hidden,activation,encoding):
        super().__init__()
        self.num_layers = num_layers
        norm_func = create_norm(norm)
        self.activation_func = create_activation(activation)
        self.dropout = feat_drop
        self.gcn_layers = nn.ModuleList()

        # build gcn layers
        if self.num_layers == 1:
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
        else:
            self.gcn_layers.append(GCNConv(in_dim, num_hidden))
            for l in range(1, self.num_layers - 1):
                self.gcn_layers.append(GCNConv(num_hidden, num_hidden))
            self.gcn_layers.append(GCNConv(num_hidden, out_dim))

        # build norms layers
        if norm_func is not None:
            self.norm_layers = nn.ModuleList(
                [norm_func(num_hidden) for _ in range(self.num_layers - 1)])
            if not encoding:
                self.norm_layers.append(norm_func(out_dim))
        else:
            self.norm_layers = None

        self.head = nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        hidden_list = []
        for l in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gcn_layers[l](x, edge_index, edge_weight)
            x = self.activation_func(x)
            if self.norm_layers is not None and l != self.num_layers - 1:
                x = self.norm_layers[l](x)
            hidden_list.append(x)
        if return_hidden:
            return self.head(x), hidden_list
        else:
            return self.head(x)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
