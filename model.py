import torch
import numpy as np
import args
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import code_utils
import torch_geometric.nn as tnn
# 定义了一个VGAE网络
class VGAE(nn.Module):
    def __init__(self, dropout=0.):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(args.input_dim, args.hidden1_dim)
        self.gcn_mean = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.gcn_logstddev = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.dropout = dropout
        self.fea_weight = nn.Parameter(torch.randn(args.hidden2_dim, args.input_dim))
        self.bn1 = nn.BatchNorm1d(args.input_dim)

    # forward是前向传播，最后得到的传播方式为：gc1->dropout->sampled_z->decode
    def forward(self, x, adj):
        # 维度是[90,16]
        x=self.bn1(x)
        hidden = self.gc1(x, adj)
        # 维度是[90, 16]
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        # 均值,维度是[90,8]
        self.mean = self.gcn_mean(hidden, adj)
        # 方差,维度是[90,8]
        self.logstd = self.gcn_logstddev(hidden, adj)
        # 高斯噪声生成,维度是[90,8]
        gaussian_noise = torch.randn(hidden.size(0), args.hidden2_dim).cuda()
        # 利用均值、方差和高斯噪声生成假潜在空间Z,*表示矩阵对应元素相乘
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        # decoder生成假邻接矩阵
        A_pred = dot_product_decode(sampled_z)
        X_pred = torch.mm(sampled_z, self.fea_weight)
        X_pred = F.leaky_relu(X_pred, inplace=0.2)
        return A_pred,X_pred

class AGE(nn.Module):
    def __init__(self, dropout=0.):
        super(AGE, self).__init__()
        self.gc1 = GraphConvolution(args.input_dim, args.hidden1_dim)
        self.gc2 = GraphConvolution(args.hidden1_dim, args.hidden2_dim)
        self.gcn_mean = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.gcn_logstddev = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.dropout = dropout
        self.fea_weight = nn.Parameter(torch.randn(args.hidden2_dim, args.input_dim))
        self.bn1 = nn.BatchNorm1d(args.input_dim)

    def forward(self, x, adj):
        x = self.bn1(x)
        hidden = self.gc1(x, adj)
        output = self.gc2(hidden, adj)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        #self.mean = self.gcn_mean(hidden, adj)
        #self.logstd = self.gcn_logstddev(hidden, adj)
        #gaussian_noise = torch.randn(hidden.size(0), args.hidden2_dim)
        #sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        A_pred = dot_product_decode(output)
        # X=leaky_relu(Z*W)
        X_pred = torch.mm(output, self.fea_weight)
        X_pred = F.leaky_relu(X_pred, inplace=0.2)
        return A_pred, X_pred

class DEcoder(nn.Module):
    def __init__(self, dropout=0.):
        super(DEcoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        # BN标准化处理，输入特征矩阵为conv1的out_channel
        self.bn1 = nn.BatchNorm1d(args.input_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 每一个残差结构中主分支第二个卷积层，输入特征矩阵为bn1的out_channel，该卷积层步长均为1，不使用偏置
        self.conv2 = nn.Conv1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=1)
        # BN标准化处理，输入特征矩阵为conv2的out_channel

        self.deconv1= nn.ConvTranspose1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        self.deconv2= nn.ConvTranspose1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        self.deconv3= nn.ConvTranspose1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(args.input_dim)
        # 下采样方法，即侧分支为虚线
        self.downsample=None
        self.dropout=dropout
        self.fc11 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        self.fc12 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

    def forward(self, x, adj):
        m, n = np.shape(x)
        m = int(m / 62)
        # 将输入特征矩阵赋值给identity（副分支的输出值）
        x = x.reshape([m, 62, 62])
        identity = x
        # 如果需要下采样方法，将输入特征矩阵经过下采样函数再赋值给identity
        if self.downsample is not None:
            identity = self.downsample(x)
        # 主分支的传播过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.conv1(identity)
        identity = self.bn1(identity)
        identity = self.conv2(identity)
        identity = self.bn2(identity)
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc11(w))
        w = F.sigmoid(self.fc12(w))
        out = out * w
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        z=out

        out = self.deconv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        identity = self.deconv1(identity)
        identity = self.bn1(identity)
        identity = self.deconv2(identity)
        identity = self.bn2(identity)
        out += identity

        z = z.reshape(m * 62, 60)

        return z,out

class GANCMLAE(nn.Module):
    def __init__(self, dropout=0.):
        super(GANCMLAE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        # BN标准化处理，输入特征矩阵为conv1的out_channel
        self.bn1 = nn.BatchNorm1d(args.input_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 每一个残差结构中主分支第二个卷积层，输入特征矩阵为bn1的out_channel，该卷积层步长均为1，不使用偏置
        self.conv2 = nn.Conv1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=1)
        # BN标准化处理，输入特征矩阵为conv2的out_channel

        self.deconv1= nn.ConvTranspose1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        self.deconv2= nn.ConvTranspose1d(in_channels=args.input_dim,out_channels=args.input_dim, kernel_size=2)
        self.deconv3= nn.ConvTranspose1d(in_channels=args.input_dim, out_channels=args.input_dim, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(args.input_dim)
        # 下采样方法，即侧分支为虚线
        self.downsample=None
        self.dropout=dropout
        self.fc11 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        self.fc12 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)

    def forward(self, x, adj):
        m, n = np.shape(x)
        m = int(m / 62)
        # 将输入特征矩阵赋值给identity（副分支的输出值）
        x = x.reshape([m, 62, 62])
        identity = x
        # 如果需要下采样方法，将输入特征矩阵经过下采样函数再赋值给identity
        if self.downsample is not None:
            identity = self.downsample(x)
        # 主分支的传播过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.conv1(identity)
        identity = self.bn1(identity)
        identity = self.conv2(identity)
        identity = self.bn2(identity)
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc11(w))
        w = F.sigmoid(self.fc12(w))
        out = out * w
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        out = self.conv3(out)
        out = self.bn2(out)
        identity = self.conv3(identity)
        identity = self.bn2(identity)
        out += identity

        z=out

        out = self.deconv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deconv2(out)
        out = self.bn2(out)
        identity = self.deconv1(identity)
        identity = self.bn1(identity)
        identity = self.deconv2(identity)
        identity = self.bn2(identity)
        out += identity

        z = z.reshape(m * 62, 60)

        return z,out

class MGCN(nn.Module):
    def __init__(self, in_size, nb_class, hidden_dim, hidden_dim2, hidden_dim3, dropout=0.1, nb_layers=4):
        super(MGCN, self).__init__()
        self.features = in_size
        self.hidden_dim = hidden_dim
        self.hidden_dim2= hidden_dim2
        self.hidden_dim3= hidden_dim3
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout = dropout

        self.conv1 = GraphConvolution(self.features, self.hidden_dim)
        self.conv2 = GraphConvolution(self.features, self.hidden_dim2)
        self.conv3 = GraphConvolution(self.features, self.hidden_dim3)

        self.conv4 = GraphConvolution(self.hidden_dim, self.features)
        self.conv5 = GraphConvolution(self.hidden_dim2, self.features)
        self.conv6 = GraphConvolution(self.hidden_dim3, self.features)

        #self.convs = torch.nn.ModuleList()
        #for i in range(self.num_layers - 1):
        #    self.convs.append(GraphConvolution(self.hidden_dim+self.hidden_dim2+self.hidden_dim3,self.hidden_dim+self.hidden_dim2+self.hidden_dim3))

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        self.bn1 = nn.BatchNorm1d(in_size)
        self.bn2 = nn.BatchNorm1d(in_size)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def forward(self, x, adj):
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn1(x)
        x1 = F.relu(self.conv1(x, adj))
        x1 = F.relu(self.conv4(x1, adj))
        x2 = F.relu(self.conv2(x, adj))
        x2 = F.relu(self.conv5(x2, adj))
        x3 = F.relu(self.conv3(x, adj))
        x3 = F.relu(self.conv6(x3, adj))
        x = x1+x2+x3
        x = self.bn2(x)
        x = code_utils.partialCorrelationMatrix(x.cpu().detach().numpy())
        # thresh_val = code_utils.get_thresh_val(df2)
        # fake_A = code_utils.convert_binary_by_thresh_val(df2, thresh_val)
        thresh_val = code_utils.get_thresh_val(x)
        adj = code_utils.convert_binary_by_thresh_val(x, thresh_val)
        x = torch.from_numpy(x).cuda().float()
        adj = torch.from_numpy(adj).cuda().float()
        #for conv in self.convs:
        #    x = F.relu(conv(x, adj))
        #x = self.bn3(x)
        #x = global_add_pool(x, batch)
        #x = self.fc_forward(x)
        return x, adj

# output = adj * x * w， adj=inputs[0],x=inputs[1]
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs):
        adj = inputs[0]
        x = inputs[1]
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

# decoder部分, A_pred =z * z^T
def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

# Xavier初始化方法，均匀分布 W~U[-sqrt(6/(x+y)),sqrt(6/(x+y))]
# 更适合有ReLU的网络训练
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #self.discriminator = nn.Sequential(
        #    nn.Linear(8100, 500),
        #    nn.LeakyReLU(True),
        #    nn.Linear(500, 100),
        #    nn.LeakyReLU(True),
        #    nn.Linear(100, 1),
        #)

        self.discriminator = nn.Sequential(
            nn.Linear(60, 30),
            nn.LeakyReLU(True),
            nn.Linear(30, 20),
            nn.LeakyReLU(True),
            nn.Linear(20, 1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(62, 1),
        )

    def forward(self, x):
        output = self.discriminator(x)
        output=self.mlp(output.reshape(-1))
        return output


class GCNDiscriminator(nn.Module):
    def __init__(self, dropout=0.):
        super(GCNDiscriminator, self).__init__()
        self.gc1 = GraphConvolution(args.input_dim, args.hidden1_dim)
        self.gc2 = GraphConvolution(args.hidden1_dim, args.hidden2_dim)
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(args.node_num * args.hidden2_dim, 1),
        )
        self.bn1 = nn.BatchNorm1d(args.input_dim)
    def forward(self, x, adj):
        x=self.bn1(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.mlp(x.reshape(-1))
        return x

class GlobalSum1D(nn.Module):
    def __init__(self):
        super(GlobalSum1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1)