import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
# from torch_geometric.data import DataLoader, Data
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
from utils_1 import to_gpu, Corpus, batchify, generate_walks
import seaborn as sns
import matplotlib.pyplot as plt
from input_data import load_data, loadFCSCData, loadASD, load_multi_site_MDD
from preprocessing import *
import args
from model import VGAE, Discriminator, GCNDiscriminator, AGE, MGCN, GANCMLAE,DEcoder
#from Graph_StyleGAN import StyledGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import warnings
from load_data import FSDataset
from GraphVAE import GraphVAE
from torch.autograd import Variable
import torch.optim as optim
# for parsing input command
import argparse

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Load zhongda or xinxiang data
# sc_features, sc_adj, sc_adj_label,
# fc_adj, fc_adj_label, fc_features, label_name, label_n = loadFCSCData(r'E:\F\data\xx&zd')
# sc_features, sc_adj, sc_adj_label, fc_adj, fc_adj_label, fc_features, label_name, label_n = loadFCSCData('data_fMRI_DTI/xinxiang_data_fmri_dti')

# load ASD data
#fc_adj, fc_adj_label, fc_features, label_name, label_n = loadASD('   E:\F\data\ADHD_useful')
fc_adj, fc_adj_label, fc_features, label_name, label_n, edge_index = FSDataset(r'F:/SC_pytorch/HCP/classify')


# load multi site MDD
# fc_adj, fc_adj_label, fc_features, label_name, label_n, center_name_list = load_multi_site_MDD('E:\F\data\ROISignals_FunImgARCWF_AAL')

adj = []
adj_norm = []
# adj_label = sc_adj_label
adj_label = fc_adj_label
# num_nodes = sc_adj[0].shape[0]
num_nodes = fc_adj[0].shape[0]
features = []
center_name = []
#parser = argparse.ArgumentParser(description='NetRA')
#parser.add_argument('--cuda', action='store_true',
#                    help='use CUDA')  # use CUDA for training
#args = parser.parse_args()

# Some preprocessing
# sc数据
# for i in range(len(sc_adj)):
#     adj_norm.append(preprocess_graph(sc_adj[i]))
#     features.append(sc_features[i])
#     adj_label[i] += sp.eye(adj_label[i].shape[0])
#     adj_label[i] = sparse_to_tuple(sp.csr_matrix(adj_label[i] / 1.0))

# fc数据
for i in range(len(fc_adj)):
    adj_norm.append(preprocess_graph_wgan(fc_adj[i]))
    features.append(fc_features[i])
    adj_label[i] += sp.eye(adj_label[i].shape[0], adj_label[i].shape[1])
    adj_label[i] = sparse_to_tuple(sp.csr_matrix(adj_label[i] / 1.0))

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

def train_GAE():
    for sub in range(len(label_n)):
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()
    # init model and optimizer
    #print()
    model = AGE().cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # 十折划分，去掉分类时用到的测试集部分
    #dataset = [adj_norm, adj_label, features, label_name, label_n]

    # train model
    loss_array = []
    for epoch in range(101):
        t = time.time()
        train_acc = 0
        loss_list = []
        for sub in range(len(label_n)):
            #range_1 = range(len(label_n))

            # 设置交叉熵权重
            pos_weight = 4
            # norm = sc_adj[sub].shape[0] * sc_adj[sub].shape[0] / float(
            #     (sc_adj[sub].shape[0] * sc_adj[sub].shape[0] - sc_adj[sub].sum()) * 2)

            weight_mask = adj_label[sub].to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight

            A_pred,X_pred = model(features[sub], adj_norm[sub])
            fake_A = A_pred.cpu().detach().numpy()
            fake_X = X_pred.cpu().detach().numpy()
            np.savez('F:/SC_pytorch/ZDXX/VGAE' + '/fake_sMRI_'+str(label_n[sub])+str(sub), fake_A=fake_A, fake_X=fake_X)
            optimizer.zero_grad()
            adj_dense=adj_label[sub].to_dense()
            # loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label[sub].to_dense().view(-1), weight=weight_tensor)
            #loss = F.mse_loss(A_pred, adj_label[sub].to_dense())
            loss = F.mse_loss(A_pred, adj_dense)
            if args.model == 'VGAE':
                kl_divergence = 0.5 / A_pred.size(0) * (
                        1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
                loss -= kl_divergence
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            adj=adj_label[sub]
            train_acc += get_acc(A_pred, adj_label[sub])
            recall_precision=get_recall(A_pred,adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
              "time=", "{:.5f}".format(time.time() - t))
        # print( recall_precision)
        loss_array.append(np.mean(loss_list))
    x = np.linspace(1, 100, 100)
    plt.plot(x, np.array(loss_array))
    plt.legend(['loss'], loc='upper right')
    plt.show()

def train_VAE():
    for sub in range(len(label_n)):
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()
    # init model and optimizer
    #print()
    model = VGAE(dropout=0.0).cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # 十折划分，去掉分类时用到的测试集部分
    #dataset = [adj_norm, adj_label, features, label_name, label_n]

    # train model
    loss_array = []
    for epoch in range(151):
        t = time.time()
        train_acc = 0
        loss_list = []
        for sub in range(len(label_n)):
            #range_1 = range(len(label_n))

            # 设置交叉熵权重
            pos_weight = 4
            # norm = sc_adj[sub].shape[0] * sc_adj[sub].shape[0] / float(
            #     (sc_adj[sub].shape[0] * sc_adj[sub].shape[0] - sc_adj[sub].sum()) * 2)

            weight_mask = adj_label[sub].to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight

            A_pred, X_pred = model(features[sub], adj_norm[sub])
            fake_A = A_pred.cpu().detach().numpy()
            fake_X = X_pred.cpu().detach().numpy()
            np.savez('F:/SC_pytorch/HCP/VGAE' + '/fake_sMRI_' +str(label_n[sub]) + str(sub), fake_A=fake_A, fake_X=fake_X)
            optimizer.zero_grad()
            adj_dense=adj_label[sub].to_dense()
            # loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label[sub].to_dense().view(-1), weight=weight_tensor)
            #loss = F.mse_loss(A_pred, adj_label[sub].to_dense())
            loss = F.mse_loss(A_pred, adj_dense)
            if args.model == 'VGAE':
                kl_divergence = 0.5 / A_pred.size(0) * (
                        1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()
                loss -= kl_divergence
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            adj=adj_label[sub]
            train_acc += get_acc(A_pred, adj_label[sub])
            recall_precision=get_recall(A_pred,adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
              "time=", "{:.5f}".format(time.time() - t))
        # print( recall_precision)
        loss_array.append(np.mean(loss_list))
    x = np.linspace(1, 100, 100)
    plt.plot(x, np.array(loss_array))
    plt.legend(['loss'], loc='upper right')
    plt.show()

    #test_acc = 0
    #total_acc = 0
    #for sub in range(len(label_name)):
        #for k in range(1):
            #test_pred = model(features[sub], adj_norm[sub])
            #data = test_pred.cpu().detach().numpy()
            #row, col = np.diag_indices_from(data)
            #data[row, col] = 0
            # fake = (data > 0.5).astype(int)
            # 生成二值图
            # fake = binaryzation(0.2, data)
            # 生成权值图
            #fake = data

            #real = adj_label[sub].cpu().to_dense().numpy()
            #real[row, col] =0
            # 画热力图
            # sns.set()
            # heatmap = sns.heatmap(fake, vmin=0.0, vmax=1.0, cmap='YlGnBu')
            # heat = heatmap.get_figure()
            # heat.savefig("result/predict/FCpre_" + str(sub) + ".png")
            # plt.show()

            # heatmap = sns.heatmap(real, vmin=0.0, vmax=1.0, cmap='YlGnBu')
            # heat = heatmap.get_figure()
            # heat.savefig("result/label/FClabel_" + str(sub) + ".png")
            # plt.show()
            # np.save('fake/xinxiang/fMRI/fake_DTI_' + str(k+1) + label_name[sub], data)

            # ASD
            # np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ASD/fake_ASD_VGAE_101/' + label_n[sub] + '/fake_fMRI_' + str(k + 1) + label_name[sub], fake)
            # ADHD
            #np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ADHD/fake_ADHD_VGAE_100/' + label_n[sub] + '/fake_fMRI_' + str(k + 1) + label_name[sub], fake)
            #np.savez('F:/FCSC_pytorch/ZDXX62pkl/classify/' + '/fake_sMRI_2' + str(sub), fake_A=fake_A, fake_X=fake_X)

            #test_acc = ((data.reshape(-1) == real.reshape(-1)).sum()) / 8100
            #total_acc += test_acc
            # recall_precision = get_recall(data, real)
            #print("Test accuracy: " + str(test_acc))
            # print(recall_precision)
    #print("Test average accuracy: " + str(total_acc / (len(label_name))))

def graph_gan():
    for sub in range(len(label_name)):
        # nn.init.kaiming_normal()
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()

    G = VGAE().cuda()
    D = Discriminator().cuda()

    loss_func = nn.BCEWithLogitsLoss()

    real_label = torch.tensor([1.]).cuda()
    fake_label = torch.tensor([0.]).cuda()
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

    G_loss_array = []
    D_loss_array = []

    G_loss_list = []
    D_loss_list = []

    print("Starting Training Loop...")

    for epoch in range(101):
        t = time.time()
        train_acc = 0
        for sub in range(len(label_name)):
            # 训练判别器
            optimizerD.zero_grad()
            # real_out 表示真实样本数据输入discriminator得到的分类结果
            real_out = D(adj_label[sub].to_dense().view(-1))
            D_loss_real = loss_func(real_out, real_label)
            D_loss_real.backward()

            output_G = G(features[sub], adj_norm[sub])
            # fake_out 表示生成样本数据输入discriminator得到的分类结果
            fake_out = D(output_G.view(-1))
            D_loss_fake = loss_func(fake_out, fake_label)
            D_loss_fake.backward(retain_graph=True)

            D_loss = D_loss_real + D_loss_fake

            # D_loss.backward()
            optimizerD.step()

            # 训练生成器
            # fake = G(features[sub], adj_norm[sub])
            # 此处的D和上面不同的点在于，已经经过了一步优化
            output_D = D(output_G.view(-1))
            G_loss = loss_func(output_D, real_label)

            # loss = F.mse_loss(A_pred.view(-1), adj_label_sub.to_dense().view(-1))
            if args.model == 'VGAE':
                kl_divergence = 0.5 / output_G.size(0) * (
                        1 + 2 * G.logstd - G.mean ** 2 - torch.exp(G.logstd) ** 2).sum(1).mean()
                G_loss -= kl_divergence

            D_loss_list.append(D_loss.item())
            G_loss_list.append(G_loss.item())
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()
            train_acc += get_acc(output_G, adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "D_loss=", "{:.5f}".format(np.mean(D_loss_list)),
              "G_loss=", "{:.5f}".format(np.mean(G_loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
              "time=", "{:.5f}".format(time.time() - t))
        G_loss_array.append(np.mean(G_loss_list))
        D_loss_array.append(np.mean(D_loss_list))

    x = np.linspace(1, 101, 101)
    plt.plot(x, np.array(G_loss_array))
    plt.plot(x, np.array(D_loss_array))
    plt.legend(['G_loss', 'D_loss'], loc='upper right')
    plt.show()

    test_acc = 0
    total_acc = 0
    # for fold in range(args.kfold):
    #     num_training, num_test, fold_adj, fold_adj_label, fold_features, fold_label_name, fold_label_n = set_kfold(adj_norm, adj_label, features, label_name, label_n, fold)
    #     for sub in range(len(fold_label_name)):
    #         for k in range(1):
    #             test_pred = G(fold_features[sub], fold_adj[sub])
    #             data = test_pred.cpu().detach().numpy()
    #             row, col = np.diag_indices_from(data)
    #             data[row, col] = 0
    #             # fake = (data > 0.5).astype(int)
    #             # 生成二值图
    #             # fake = binaryzation(0.2, data)
    #             # 生成权值图
    #             fake = data
    #
    #             real = adj_label[sub].cpu().to_dense().numpy()
    #             real[row, col] = 0
    #             # xinxiang
    #             # np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_GAN_xxzd_degree/'+ str(fold) + '/' + fold_label_n[sub] +'/fake_fMRI_' + str(k+1) + label_name[sub], fake)
    #             # ASD
    #             np.save(
    #                 'E:/F/code/GraphGAN/FCSC_pytorch/fake_ASD/fake_ASD_VAEGAN_101/' + fold_label_n[sub] + '/fake_fMRI_' + str(
    #                     k + 1) + fold_label_name[sub], fake)
    #             test_acc = ((fake.reshape(-1) == real.reshape(-1)).sum()) / 8100
    #             total_acc += test_acc
    #             print("Test accuracy: " + str(test_acc))
    #     print("Test average accuracy: " + str(total_acc / (len(label_name) * 1)))

    # 不按十折划分
    # for sub in range(len(label_name)):
    #     for k in range(1):
    #         test_pred = G(features[sub], adj_norm[sub])
    #         data = test_pred.cpu().detach().numpy()
    #         row, col = np.diag_indices_from(data)
    #         data[row, col] = 0
    #         # fake = (data > 0.5).astype(int)
    #         # 生成二值图
    #         # fake = binaryzation(0.2, data)
    #         # 生成权值图
    #         fake = data
    #         real = adj_label[sub].cpu().to_dense().numpy()
    #         real[row, col] = 0
    #         # 画热力图
    #         # sns.set()
    #         # heatmap = sns.heatmap(fake, vmin=0.0, vmax=1.0, cmap='YlGnBu')
    #         # heat = heatmap.get_figure()
    #         # heat.savefig("result/predict/FCpre_" + str(sub) + ".png")
    #         # plt.show()
    #
    #         # heatmap = sns.heatmap(real, vmin=0.0, vmax=1.0, cmap='YlGnBu')
    #         # heat = heatmap.get_figure()
    #         # heat.savefig("result/label/FClabel_" + str(sub) + ".png")
    #         # plt.show()
    #         # xinxiang
    #         # np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_GAN_xx_spectral/fMRI/'+label_n[sub]+'/fake_fMRI_' + str(k+1) + label_name[sub], fake)
    #
    #         # ASD
    #         # np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ASD/fake_ASD_CGAN_101/' + label_n[sub] + '/fake_fMRI_' + str(k + 1) + label_name[sub], fake)
    #         # ADHD
    #         np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ADHD/fake_ADHD_CGAN_101/'+label_n[sub]+'/fake_fMRI_' + str(k+1) + label_name[sub], fake)
    #
    #         # multi site MDD
    #         # np.save('E:/F/code/GraphGAN/FCSC_pytorch/test/'+center_name_list[sub]+ '/' + label_n[sub]+'/fake_fMRI_' + str(k+1) + label_name[sub], fake)
    #         test_acc = ((fake.reshape(-1) == real.reshape(-1)).sum()) / 8100
    #         total_acc += test_acc
    #         print("Test accuracy: " + str(test_acc))
    # print("Test average accuracy: " + str(total_acc / (len(label_name)*1)))

def graph_wgan():
    for sub in range(len(label_n)):
        # nn.init.kaiming_normal()
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()

    G = VGAE().cuda()
    #G= StyledGenerator().cuda()
    #G=AGE().cuda()
    D = GCNDiscriminator().cuda()

    optimizerD = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

    G_loss_array = []
    D_loss_array = []

    G_loss_list = []
    D_loss_list = []

    print("Starting Training Loop...")

    #for fold in range(args.kfold):
    # num_training, num_test, fold_adj, fold_adj_label, fold_features, fold_label_name, fold_label_n = set_kfold(adj_norm, adj_label, features, label_name, label_n, fold)
    for epoch in range(300):
        t = time.time()
        train_acc = 0
        label_length=len(label_name)
        for sub in range(len(label_n)):
            #range_1=range(len(label_n))
         #for k in range(1):
            #print(sub)
            # 训练判别器
            optimizerD.zero_grad()
            # real_out 表示真实样本数据输入discriminator得到的分类结果
            #real_out = D(adj_label[sub].to_dense().view(-1))
            real_out = D(features[sub], adj_norm[sub])
            #output_G, = G(features[sub], adj_norm[sub])
            output_A,output_X = G(features[sub], adj_norm[sub])
            # fake_out 表示生成样本数据输入discriminator得到的分类结果

            #fake_out = D(output_G.view(-1))
            fake_out = D(output_X,output_A)
            D_loss = -torch.mean(real_out) + torch.mean(fake_out)

            D_loss.backward(retain_graph=True)
            optimizerD.step()

            # WGAN修改部分
            # Clip weight of the discriminator
            for p in D.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # 训练生成器
            # Train the generator every critic_num iterations
            optimizerG.zero_grad()
            if sub % args.critic_num == 1:
                # 二次做真实数据embedding及假数据生成是因为fake_sample_z的梯度在鉴别器参数更新时已经被使用过
                #output_G = G(features[sub], adj_norm[sub])
                output_A, output_X = G(features[sub], adj_norm[sub])

                data_A = output_A.cpu().detach().numpy()
                row, col = np.diag_indices_from(data_A)
                data_A[row, col] = 0
                fake_A = (data_A > 0.5).astype(int)
                fake_A = binaryzation(0.2, data_A)
                fake_A = data_A
                sub_name=sub

                fake_X=output_X.cpu().detach().numpy()

                np.savez('F:/SC_pytorch/ZDXX/classify/test'+ '/fake_sMRI_'+str(label_n[sub])+str(sub), fake_A=fake_A,fake_X=fake_X)
                #np.save('F:/FCSC_pytorch/HCP62ROI/female1/' + '/fake_fMRI_' + str(sub) , fake_X)

                G_loss = -torch.mean(D(output_A, output_X))
                #G_loss = -torch.mean(D(output_G.view(-1)))
                if args.model == 'VGAE':
                    kl_divergence = 0.5 / output_A.size(0) * (
                            1 + 2 * G.logstd - G.mean ** 2 - torch.exp(G.logstd) ** 2).sum(1).mean()
                    G_loss -= kl_divergence
                G_loss.backward(retain_graph=True)
                optimizerG.step()

                D_loss_list.append(D_loss.item())
                G_loss_list.append(G_loss.item())

            train_acc += get_acc(output_A, adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "D_loss=", "{:.5f}".format(np.mean(D_loss_list)),
              "G_loss=", "{:.5f}".format(np.mean(G_loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
              "time=", "{:.5f}".format(time.time() - t))
        G_loss_array.append(np.mean(G_loss_list))
        D_loss_array.append(np.mean(D_loss_list))

    x = np.linspace(1, 101, 101)
    plt.plot(x, np.array(G_loss_array))
    plt.plot(x, np.array(D_loss_array))
    plt.legend(['G_loss', 'D_loss'], loc='upper right')
    plt.show()

    test_acc = 0
    total_acc = 0

    # 不按十折划分
    # for sub in range(len(label_name)):
    #     for k in range(1):
    #         test_pred = G(features[sub], adj_norm[sub])
    #         data = test_pred.cpu().detach().numpy()
    #         row, col = np.diag_indices_from(data)
    #         data[row, col] = 0
    #         # fake = (data > 0.5).astype(int)
    #         # 生成二值图
    #         # fake = binaryzation(0.2, data)
    #         # 生成权值图
    #         fake = data
    #         real = adj_label[sub].cpu().to_dense().numpy()
    #         real[row, col] = 0
    #
    #         # ASD
    #         np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ASD/fake_ASD_WGAN_101/' + label_n[sub] + '/fake_fMRI_' + str(k + 1) + label_name[sub], fake)
    #         # ADHD
    #         # np.save('E:/F/code/GraphGAN/FCSC_pytorch/fake_ADHD/fake_ADHD_WGAN_101/'+label_n[sub]+'/fake_fMRI_' + str(k+1) + label_name[sub], fake)
    #
    #         test_acc = ((fake.reshape(-1) == real.reshape(-1)).sum()) / 8100
    #         total_acc += test_acc
    #         print("Test accuracy: " + str(test_acc))
    # print("Test average accuracy: " + str(total_acc / (len(label_name)*1)))

def MGCN_GAN():
    for sub in range(len(label_n)):
        # nn.init.kaiming_normal()
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()

    hidden_dim=74
    hidden_dim2=148
    hidden_dim3=296
    dropout=0.1
    nb_layers=2
    #G = MGCN().cuda()
    G = MGCN(in_size=62, nb_class=2,hidden_dim=hidden_dim,hidden_dim2=hidden_dim2,hidden_dim3=hidden_dim3, dropout=dropout,nb_layers=nb_layers).cuda()
    D = GCNDiscriminator().cuda()

    loss_func = nn.BCEWithLogitsLoss()

    real_label = torch.tensor([1.]).cuda()
    fake_label = torch.tensor([0.]).cuda()
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))

    G_loss_array = []
    D_loss_array = []

    G_loss_list = []
    D_loss_list = []

    print("Starting Training Loop...")

    for epoch in range(301):
        t = time.time()
        train_acc = 0
        for sub in range(len(label_n)):
            # 训练判别器
            optimizerD.zero_grad()
            # real_out 表示真实样本数据输入discriminator得到的分类结果
            real_out = D(features[sub], adj_norm[sub])
            D_loss_real = loss_func(real_out, real_label)
            D_loss_real.backward()

            output_G,output_A = G(features[sub], adj_norm[sub])

            fake_A = output_A.cpu().detach().numpy()
            fake_X = output_G.cpu().detach().numpy()

            #print(str(label_n[sub]))

            #np.savez('F:/SC_pytorch/HCP/MGCN_GAN' + '/fake_sMRI_' + str(label_n[sub]) + str(sub), fake_A=fake_A, fake_X=fake_X)
            # fake_out 表示生成样本数据输入discriminator得到的分类结果
            fake_out = D(output_G,output_A)
            D_loss_fake = loss_func(fake_out, fake_label)
            D_loss_fake.backward(retain_graph=True)

            D_loss = D_loss_real + D_loss_fake

            # D_loss.backward()
            optimizerD.step()

            # 训练生成器
            # fake = G(features[sub], adj_norm[sub])
            # 此处的D和上面不同的点在于，已经经过了一步优化
            output_D = D(output_G,output_A)
            G_loss = loss_func(output_D, real_label)

            # loss = F.mse_loss(A_pred.view(-1), adj_label_sub.to_dense().view(-1))
            if args.model == 'VGAE':
                kl_divergence = 0.5 / output_G.size(0) * (
                        1 + 2 * G.logstd - G.mean ** 2 - torch.exp(G.logstd) ** 2).sum(1).mean()
                G_loss -= kl_divergence

            D_loss_list.append(D_loss.item())
            G_loss_list.append(G_loss.item())
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()
            train_acc += get_acc(output_G, adj_label[sub])
        print("Epoch:", '%04d' % (epoch + 1), "D_loss=", "{:.5f}".format(np.mean(D_loss_list)),
              "G_loss=", "{:.5f}".format(np.mean(G_loss_list)),
              "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
              "time=", "{:.5f}".format(time.time() - t))
        G_loss_array.append(np.mean(G_loss_list))
        D_loss_array.append(np.mean(D_loss_list))

    x = np.linspace(1, 301, 301)
    plt.plot(x, np.array(G_loss_array))
    plt.plot(x, np.array(D_loss_array))
    plt.legend(['G_loss', 'D_loss'], loc='upper right')
    plt.show()


def train_GANCMLAE():
    """
    Training WGAN discriminator
    :param batch: training batch data
    :return: discriminator part error
    """
    # clamp parameters to a cube
    # WGAN Weight clipping
    for sub in range(len(label_n)):
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()
    # init model and optimizer
    #print()
    model = GANCMLAE().cuda()
    decoder = DEcoder().cuda()
    #G = MGCN(in_size=62, nb_class=2,hidden_dim=hidden_dim,hidden_dim2=hidden_dim2,hidden_dim3=hidden_dim3, dropout=dropout,nb_layers=nb_layers).cuda()
    D = Discriminator().cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    optimizerde = Adam(decoder.parameters(), lr=args.learning_rate)
    # 十折划分，去掉分类时用到的测试集部分
    #dataset = [adj_norm, adj_label, features, label_name, label_n]
    loss_func = nn.BCEWithLogitsLoss()

    real_label = torch.tensor([1.]).cuda()
    fake_label = torch.tensor([0.]).cuda()
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.999))
    #optimizerG = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999))


    # train model
    loss_array = []
    for epoch in range(101):
        t = time.time()
        train_acc = 0
        loss_list = []
        for sub in range(len(label_n)):
            #range_1 = range(len(label_n))

            # 设置交叉熵权重
            pos_weight = 4
            # norm = sc_adj[sub].shape[0] * sc_adj[sub].shape[0] / float(
            #     (sc_adj[sub].shape[0] * sc_adj[sub].shape[0] - sc_adj[sub].sum()) * 2)

            weight_mask = adj_label[sub].to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight
            z, out = model(features[sub], adj_norm[sub])
            #fake_A = A_pred.cpu().detach().numpy()
            fake_X = out.cpu().detach().numpy().squeeze(0)
            feature=features[sub].cpu().detach().numpy()
            np.savez('F:/SC_pytorch/ZDXX/GAE/' + '/fake_sMRI_'+str(label_n[sub])+str(sub),fake_X=fake_X)
            optimizer.zero_grad()
            ae_loss=torch.nn.L1Loss()
            fake_X=torch.from_numpy(fake_X)
            feature = torch.from_numpy(feature)
            loss = ae_loss(fake_X,feature)
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss)

            optimizerD.zero_grad()
            # real_out 表示真实样本数据输入discriminator得到的分类结果
            real_out = D(z)
            D_loss_real = loss_func(real_out, real_label)
           # D_loss_real.backward()
            #optimizerD.step()

            gen_size = 62
            z_size=60
            noise = to_gpu(args.cuda,
                           Variable(torch.ones(gen_size, z_size)))
            noise.data.normal_(0, 1)
            errD_fake = D(noise)
            D_loss_fake = loss_func(errD_fake, fake_label)
            D_loss = D_loss_fake + D_loss_real
            D_loss.backward()
            optimizerD.step()
            print(D_loss)

            z1, out = decoder(out.squeeze(0), adj_norm[sub])
            optimizerde.zero_grad()
            de_loss=torch.nn.MSELoss()
            #z1=torch.from_numpy(z1)
            #z=torch.from_numpy(z)
            deloss = de_loss(z,z1)
            #deloss.requires_grad_(True)
            deloss.backward()
            #deloss.backward()
            optimizerde.step()

            print(deloss)

        #print("Epoch:", '%04d' % (epoch + 1), "ae_loss=", "{:.5f}".format(np.mean(loss_list)),
        #      "train_acc=", "{:.5f}".format(train_acc / len(label_name)),
        #      "time=", "{:.5f}".format(time.time() - t))
        # print( recall_precision)
        #loss_array.append(np.mean(loss_list))

#graph_gan()
#train_VAE()
MGCN_GAN()
#train_GANCMLAE()
#graph_wgan()
#train_VAE()
#train_GAE()