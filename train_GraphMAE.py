import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
# from torch_geometric.data import DataLoader, Data
from torch.utils.data import Subset
from torch_geometric.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from input_data import load_data, loadFCSCData, loadASD, load_multi_site_MDD
from preprocessing import *
import args
from model import VGAE, Discriminator,GCNDiscriminator,AGE
from graphmae.GraphMAE import GraphMAE
from Graph_StyleGAN import StyledGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import warnings
from graphmae.load_data import FSDataset
from graphmae.GraphMAE_Encoder import GCNEncoder
import argparse
import scipy.sparse as sp
from GraphVAE import GraphVAE
from torch_geometric.utils import dense_to_sparse,to_scipy_sparse_matrix
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Load zhongda or xinxiang data
# sc_features, sc_adj, sc_adj_label,
# fc_adj, fc_adj_label, fc_features, label_name, label_n = loadFCSCData(r'E:\F\data\xx&zd')
# sc_features, sc_adj, sc_adj_label, fc_adj, fc_adj_label, fc_features, label_name, label_n = loadFCSCData('data_fMRI_DTI/xinxiang_data_fmri_dti')
def load_args():
    parser = argparse.ArgumentParser(
        description='graphmae baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_rate', type=float, default=0.75)
    parser.add_argument('--drop_edge_rate', type=float, default=0.0)
    parser.add_argument('--concat_hidden', type=bool, default=False)
    parser.add_argument('--replace_rate', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_out_heads', type=int, default=8)
    parser.add_argument('--num_hidden', type=int, default=32)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--in_dim', type=int, default=62)
    #num_layers, norm, feat_drop, in_dim, out_dim, num_hidden, activation, encoding
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--norm', type=str, default='layernorm')
    parser.add_argument('--feat_drop', type=int, default=0.2)
    #parser.add_argument('--in_dim', type=int, default=62)
    parser.add_argument('--out_dim', type=int, default=16)
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--encoding', type=str, default=True)
    #parser.add_argument('--num_hidden', type=int, default=62)
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    #args.batch_norm = not args.layer_norm

    args.save_logs = False
    return args
# load ASD data
#fc_adj, fc_adj_label, fc_features, label_name, label_n = loadASD('E:\F\data\ADHD_useful')
data = FSDataset(r'F:\SC_pytorch\HCP\male')

train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def get_recall(adj_pred,adj_label):
    adj_pred=adj_pred.cpu()
    adj_label=adj_label.cpu()
    adj_pred_vector=(adj_pred > 0.5).view(-1).long()
    adj_label_vector=adj_label.to_dense().view(-1).long()
    precision_recall_report=classification_report(adj_label_vector, adj_pred_vector)
    warnings.filterwarnings("ignore")
    return precision_recall_report

# fc_adj, fc_adj_label, fc_features, label_name, label_n
def set_kfold(adj, adj_label, features, label_name, label_n, fold_id, random_seed = 125):
    total_data = len(adj)
    inst = KFold(n_splits=args.kfold, shuffle=True, random_state=random_seed)
    KFolds = list(inst.split(np.arange(total_data)))
    training_idx, test_idx = KFolds[fold_id]
    num_training = training_idx.size
    num_test = test_idx.size
    test_adj = Subset(adj, test_idx)
    test_adj_label = Subset(adj_label, test_idx)
    test_features = Subset(features, test_idx)
    test_label_name = Subset(label_name, test_idx)
    test_label_n = Subset(label_n, test_idx)

    # training_set = Subset(dataset, training_idx)
    # test_set = Subset(dataset, test_idx)
    # train_data = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    # test_data = DataLoader(test_set, batch_size=min(num_test, args.batch_size), shuffle=False)  # num_test
    return num_training, num_test, test_adj, test_adj_label, test_features, test_label_name, test_label_n

def train_VAE():
    global args
    args = load_args()
     #for sub in range(len(label_n)):
    #    adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
    #                                             torch.FloatTensor(adj_norm[sub][1]),
    #                                             torch.Size(adj_norm[sub][2])).cuda()
    #    adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
    #                                              torch.FloatTensor(adj_label[sub][1]),
    #                                              torch.Size(adj_label[sub][2])).cuda()
    #    features[sub] = torch.FloatTensor(features[sub]).cuda()
    # init model and optimizer
    #print()
    encoder = GCNEncoder(
        num_layers=args.num_layers, norm=args.norm, feat_drop=args.feat_drop, in_dim=args.in_dim,out_dim=args.out_dim,num_hidden=args.num_hidden,activation=args.activation,encoding=args.encoding
    )
    decoder = GCNEncoder(
        num_layers=1, norm=args.norm, feat_drop=args.feat_drop, in_dim=args.out_dim,out_dim=args.in_dim,num_hidden=args.num_hidden,activation=args.activation,encoding=False)
    model = GraphMAE(encoder=encoder,decoder=decoder,mask_rate=args.mask_rate,drop_edge_rate=args.drop_edge_rate,num_hidden=args.out_dim,concat_hidden=args.concat_hidden,replace_rate=args.replace_rate,num_heads=args.num_heads,num_out_heads=args.num_out_heads,in_dim=args.in_dim,out_dim=args.out_dim)
    for param in model.parameters():
        print(type(param), param.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.5, 0.999))
    # 十折划分，去掉分类时用到的测试集部分
    #optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #dataset = [adj_norm, adj_label, features, label_name, label_n]
    #if use_cuda:
    #    data = data.cuda()

    # train model
    loss_array = []
    for epoch in range(101):
        t = time.time()
        train_acc = 0
        loss_list = []
        #label_n=data.y
         # for sub in range(len(label_n)):
        for i, data in enumerate(train_loader):
            #range_1 = range(len(label_n))

            # 设置交叉熵权重
            pos_weight = 4
            # norm = sc_adj[sub].shape[0] * sc_adj[sub].shape[0] / float(
            #     (sc_adj[sub].shape[0] * sc_adj[sub].shape[0] - sc_adj[sub].sum()) * 2)

            #weight_mask = adj_label[sub].to_dense().view(-1) == 1
            #weight_tensor = torch.ones(weight_mask.size(0))
            #weight_tensor[weight_mask] = pos_weight

            A_pred,X_pred = model(data)
            fake_A = A_pred.cpu().detach().numpy()
            fake_X = X_pred.cpu().detach().numpy()
            np.savez('F:/SC_pytorch/HCP/GraphMAE/' + '/fake_sMRI_1' + str(i), fake_A=fake_A, fake_X=fake_X)
            optimizer.zero_grad()
            adj = data.edge_index
            #adj = adj.numpy()
            adj = to_scipy_sparse_matrix(adj)
            adj_dense=adj.todense()
            adj_dense=torch.from_numpy(adj_dense)
            adj_label = adj_dense.detach().numpy()
            adj_label = sp.eye(adj_label.shape[0], adj_label.shape[1])
            adj_label = sparse_to_tuple(sp.csr_matrix(adj_label / 1.0))
            adj_label= torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                      torch.FloatTensor(adj_label[1]),
                                                      torch.Size(adj_label[2]))
            #adj_dense=to_scipy_sparse_matrix(data.edge_index)
            # loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label[sub].to_dense().view(-1), weight=weight_tensor)
            #loss = F.mse_loss(A_pred, adj_label[sub].to_dense())
            loss = F.mse_loss(A_pred, adj_dense)
            #if args.model == 'VGAE':
            #    kl_divergence = 0.5 / A_pred.size(0) * (
            #            1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
            #    loss -= kl_divergence
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            #adj_label = adj_label.detach().numpy()
            #adj_label = sp.eye(adj_label.shape[0], adj_label.shape[1])
            #adj_label = sparse_to_tuple(sp.csr_matrix(adj_label / 1.0))
            train_acc += get_acc(A_pred, adj_label)
            #recall_precision=get_recall(A_pred,adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / i),
              "time=", "{:.5f}".format(time.time() - t))
        # print( recall_precision)
        loss_array.append(np.mean(loss_list))
#    x = np.linspace(1, 100, 100)
#    plt.plot(x, np.array(loss_array))
#    plt.legend(['loss'], loc='upper right')
#    plt.show()



# graph_gan()
train_VAE()
#graph_wgan()
#train_VAE()
#train_GAE()