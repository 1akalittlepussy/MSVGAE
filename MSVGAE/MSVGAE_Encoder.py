# -*- coding: utf-8 -*-
# @Author  : sw t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # 属性定义：定义了输入特征in_feature、输出特征out_feature两个输入，以及权重weight和偏移向量bias两个参数
    def __init__(self, in_features, out_features, activation=F.relu, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # 由于weight可训练，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化，为了让每次训练产生的初始参数尽可能相同，便于实验结果复现，可以设置固定的随机数生成种子
    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数，size(1)为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform()方法将随机生成下一个实数，它在[x,y]范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 定义本层的前向传播，采用A*X*W的计算方法，A表示邻接矩阵，X表示特征矩阵，W表示权重
    def forward(self, input, adj):
        # input是X，即特征矩阵，adj为邻接矩阵，即A
        # print (input.size())
        # print (self.weight.size())
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return self.activation(output + self.bias)
        else:
            return self.activation(output)
    # 类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”
    # 如果对该方法进行重写，可以为其制作自定义的自我描述信息
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GAT_Encoder(nn.Module):
    def __init__(self, num_heads, in_channels, hidden_dims, latent_dim, dropout):
        super(GAT_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        #self.hidden_layer_1 = GATConv(
        #    in_channels=in_channels, out_channels=hidden_dims[0],
        #    heads=self.num_heads['first'],
        #    concat=True)

        self.hidden_layer_1 = GraphConvolution(in_channels, hidden_dims[0])

        #in_dim2 = hidden_dims[0] * self.num_heads['first'] + in_channels * 2
        # in_dim2 = hidden_dims[0] * self.num_heads['first']
        #in_dim2 = hidden_dims[0] * self.num_heads['first']

        #self.hidden_layer_2 = GATConv(
        #    in_channels=in_dim2, out_channels=hidden_dims[1],
       #     heads=self.num_heads['second'],
       #     concat=True)

        self.hidden_layer_2 = GraphConvolution(hidden_dims[0], hidden_dims[1])

        #in_dim_final = hidden_dims[-1] * self.num_heads['second'] + in_channels
        in_dim_final = hidden_dims[-1] * self.num_heads['second']
        # in_dim_final = hidden_dims[-1] * self.num_heads['second']
        #self.out_mean_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
        #                              heads=self.num_heads['mean'], concat=False,dropout=0.2)
        #self.out_logstd_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
        #                                heads=self.num_heads['std'], concat=False,dropout=0.2)
        self.out_mean_layer = GraphConvolution(hidden_dims[1], self.latent_dim, activation=lambda x: x)
        self.out_logstd_layer = GraphConvolution(hidden_dims[1], self.latent_dim, activation=lambda x: x)

        #self.out_mean_layer=GraphConvolution(in_dim_final, hidden_dims[1])
        #self.out_logstd_layer=GraphConvolution(in_channels, hidden_dims[1])
        self.bn1 = nn.BatchNorm1d(in_channels)

    def forward(self, x, edge_index):
        x = self.bn1(x)
        #print(x)
        #edge_index, _ = dense_to_sparse(torch.from_numpy(edge_index.astype(np.int16)))
        hidden_out1 = self.hidden_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        #print(hidden_out1)
        # add Gaussian noise being the same shape as x and concat
        #hidden_out1 = torch.cat([x, torch.randn_like(x), hidden_out1], dim=1)
        hidden_out2 = self.hidden_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        #hidden_out2 = F.dropout(hidden_out2, training=self.training)
        last_out = hidden_out2
        #print(last_out)
        # concat x with last_out
        #last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)
        #print(z_mean, z_logstd)
        return z_mean, z_logstd

class GCN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, latent_dim):
        super(GCN_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # initialize GCN layer
        self.hidden_layer_1 = GCNConv(in_channels=in_channels, out_channels=hidden_dims[0])
        self.hidden_layer_2 = GCNConv(in_channels=hidden_dims[0] + 2 * in_channels, out_channels=hidden_dims[1])
        self.out_mean_layer = GCNConv(in_channels=hidden_dims[-1] + in_channels, out_channels=latent_dim[0])
        self.out_logstd_layer = GCNConv(in_channels=hidden_dims[-1] + in_channels, out_channels=latent_dim[1])

    def forward(self, x, edge_index):
        x = self.bn1(x)
        edge_index, _ = dense_to_sparse(torch.from_numpy(edge_index.astype(np.int16)))
        hidden_out1 = self.hidden_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        # add Gaussian noise being the same shape as x and concat
        hidden_out1 = torch.cat([x, torch.randn_like(x), hidden_out1], dim=1)
        hidden_out2 = self.hidden_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        last_out = hidden_out2
        # concat x with last_out
        last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd