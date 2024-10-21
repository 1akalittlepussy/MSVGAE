import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


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
