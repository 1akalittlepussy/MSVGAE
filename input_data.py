'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import os

import pickle as pkl
import networkx as nx
import scipy.sparse as sp

import torch
import copy
from scipy import io
from glob import glob
from preprocessing import preprocess_graph, binaryzation
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def loadFCSCData(path):
    label_list = os.listdir(path)
    # label_list.sort()
    # SC_adj_dir = "DTI_connectivity_count.mat"
    # SC_feature_dir = "region_features_norm.mat"
    FC_dir = "RegionSeries.mat"
    feature = []
    adj = []
    label = []
    label_n = []
    fc_adj = []
    fc_features = []
    sc_adj_label = []
    fc_adj_label = []
    threshold = 0.2
    for label_files in label_list:
        list = os.listdir(os.path.join(path, label_files))
        # list.sort()
        for files in list:
            # DTI数据读取
            # subj_sc_adj_dir = os.path.join(path, label_files, files, SC_adj_dir)
            # subj_sc_adj_data = io.loadmat(subj_sc_adj_dir)
            # print("reading data " + subj_sc_adj_dir)
            # subj_mat_sc_adj = subj_sc_adj_data['connectivity']
            # subj_mat_sc_adj = binaryzation(threshold, subj_mat_sc_adj)
            # sc_adj_label.append(subj_mat_sc_adj)
            # # subj_mat_sc_adj = preprocess_graph(subj_mat_sc_adj)
            # # 稀疏矩阵存储
            # adj.append(sp.csr_matrix(subj_mat_sc_adj))
            # subj_sc_feature_dir = os.path.join(path, label_files, files, SC_feature_dir)
            # subj_sc_feature_data = io.loadmat(subj_sc_feature_dir)
            # print("reading data " + subj_sc_feature_dir)
            # subj_mat_sc_feature = subj_sc_feature_data['region_features']
            # feature.append(subj_mat_sc_feature)

            # fMRI数据读取
            subj_fc_dir = os.path.join(path, label_files, files, FC_dir)
            subj_fc_data = io.loadmat(subj_fc_dir)
            print("reading data " + subj_fc_dir)
            subj_mat_fc = subj_fc_data['RegionSeries']
            subj_mat_fc_adj = np.corrcoef(np.transpose(subj_mat_fc))
            subj_mat_fc_adj = binaryzation(threshold, subj_mat_fc_adj)
            degree = comput_degree(subj_mat_fc_adj)
            # 作为损失函数的对比数据（真实数据）
            fc_adj_label.append(sp.csr_matrix(subj_mat_fc_adj))
            # subj_mat_fc_adj = preprocess_graph(subj_mat_fc_adj)
            # 作为GAN的输入数据，数据本身和fc_adj_label没有区别
            fc_adj.append(sp.csr_matrix(subj_mat_fc_adj))
            subj_mat_fc_list = subj_mat_fc.reshape((-1))
            # 归一化处理
            subj_mat_fc_new = (subj_mat_fc - min(subj_mat_fc_list)) / (max(subj_mat_fc_list) - min(subj_mat_fc_list))
            # fc_features.append(np.transpose(subj_mat_fc_new))
            fc_features.append(degree)
            label_n.append(label_files)
            label.append(label_files+"_"+files)
    # feature, adj, sc_adj_label,
    return fc_adj, fc_adj_label, fc_features, label, label_n

def compute_adj(feature_matrix):
    fc_adj=np.zeros((90,90))
    # 转置成90*172
    np.transpose(feature_matrix)
    # print(feature_matrix.shape)
    for i in range(90):
        for j in range(90):
            # 皮尔森相关系数计算
            adj=np.corrcoef(feature_matrix[i], feature_matrix[j])
            # print(adj)
            fc_adj[i][j]=adj[0,1]
            fc_adj[j][i]=adj[1,0]
    return fc_adj

def loadASD(path):
    # label_list = os.listdir(path)
    # label_list = ['HC', 'ASD']
    label_list = ['HC', 'ADHD']
    # label用来标注文件名称，主要用来做数量标记
    label_n = []
    label_name = []
    fc_adj = []
    fc_features = []
    fc_adj_label = []
    threshold = 0.2
    node_num = 90
    fc = []
    for i in range(len(label_list)):
        list_path = os.path.join(path, label_list[i])
        print(list_path)
        lists = os.listdir(list_path)
        # lists.sort()
        for files in lists:
            print(files)
            if "ABIDE" in path:
                subj_fc_dir = os.path.join(list_path, files)
                region_series = np.loadtxt(subj_fc_dir)[:176, :90]
            else:
                subj_dir = os.path.join(list_path, files)
                print(subj_dir)
                FC_dir = glob(os.path.join(subj_dir, 'sfnwmrda*' + '_rest_1_aal_TCs.1D'))
                dir = FC_dir[0]
                region_series = np.loadtxt(fname=dir, skiprows=1, usecols=np.arange(2, 118))[:, :90]

            # files_path = os.path.join(list_path, files)
            # region_series = np.loadtxt(files_path)[0:176, :90]
            # print("region_series:",region_series.shape)
            # subj_fc_adj = np.corrcoef(np.transpose(region_series))

            subj_mat_fc_list = region_series.reshape((-1))
            subj_mat_fc_new = (region_series - min(subj_mat_fc_list)) / (max(subj_mat_fc_list) - min(subj_mat_fc_list))
            w_edgeWeight = np.corrcoef(np.transpose(subj_mat_fc_new))


            node_feature = copy.deepcopy(w_edgeWeight)

            edgeWeight_list = w_edgeWeight.reshape((-1))
            thindex = int(threshold * edgeWeight_list.shape[0])
            thremax = edgeWeight_list[edgeWeight_list.argsort()[-1 * thindex]]
            w_edgeWeight[w_edgeWeight < thremax] = 0
            w_edgeWeight[w_edgeWeight >= thremax] = 1

            # subj_fc_adj = compute_adj(region_series)
            # subj_mat_fc_adj = binaryzation(threshold, subj_fc_adj)
            # degree = comput_degree(subj_mat_fc_adj)
            # 作为损失函数的对比数据（真实数据）
            # csr_fc_adj = sp.csr_matrix(subj_mat_fc_adj)
            fc_adj_label.append(w_edgeWeight)

            # subj_mat_fc_adj = preprocess_graph(subj_mat_fc_adj)
            # 作为GAN的输入数据，数据本身和fc_adj_label没有区别
            fc_adj.append(w_edgeWeight)

            # subj_mat_fc_list = region_series.reshape((-1))
            # 归一化处理
            # subj_mat_fc_new = (region_series - min(subj_mat_fc_list)) / (max(subj_mat_fc_list) - min(subj_mat_fc_list))

            # temp = np.isnan(subj_mat_fc_new)
            # temp_1 = np.isnan(fc_adj)

            # fc_features.append(np.transpose(subj_mat_fc_new))
            # if i == 0:
            #     y = torch.zeros(90, 1)
            # else:
            #     y = torch.ones(90, 1)
            # node_feature = np.concatenate((node_feature, y), axis=1)

            fc_features.append(node_feature)
            label_n.append(label_list[i])
            label_name.append(files)
            # data = Data(x=torch.from_numpy(degree).float(), edge_index=fc_adj, y=torch.tensor(fc_adj_label))
            # data.label_n = label_n
            # fc.append(data)
    return fc_adj, fc_adj_label, fc_features, label_name, label_n
    # return fc

def load_multi_site_MDD(path):
    center_list = os.listdir(path)
    label = []
    label_n = []
    center_name_list = []
    fc_adj = []
    fc_features = []
    fc_adj_label = []
    threshold = 0.2
    for center in center_list:
        label_path = os.path.join(path, center)
        label_list = os.listdir(label_path)
        for label_files in label_list:
            list_path = os.path.join(label_path, label_files)
            print(list_path)
            lists = os.listdir(list_path)
            # lists.sort()
            for files in lists:
                print(files)
                files_path = os.path.join(list_path, files)
                region_series_array = io.loadmat(files_path)['ROISignals_AAL']
                region_series = region_series_array[0:170, :90]
                # print("region_series:",region_series.shape)
                # subj_fc_adj = np.corrcoef(np.transpose(region_series))
                subj_fc_adj = compute_adj(region_series)
                subj_mat_fc_adj = binaryzation(threshold, subj_fc_adj)
                # 作为损失函数的对比数据（真实数据）
                csr_fc_adj = sp.csr_matrix(subj_mat_fc_adj)
                fc_adj_label.append(csr_fc_adj)
                # subj_mat_fc_adj = preprocess_graph(subj_mat_fc_adj)
                # 作为GAN的输入数据，数据本身和fc_adj_label没有区别
                fc_adj.append(csr_fc_adj)
                subj_mat_fc_list = region_series.reshape((-1))
                # 归一化处理
                subj_mat_fc_new = (region_series - min(subj_mat_fc_list)) / (
                            max(subj_mat_fc_list) - min(subj_mat_fc_list))

                # temp = np.isnan(subj_mat_fc_new)
                # temp_1 = np.isnan(fc_adj)
                fc_features.append(np.transpose(subj_mat_fc_new))
                label_n.append(label_files)
                label.append(label_files + "_" + files)
                center_name_list.append(center)
    return fc_adj, fc_adj_label, fc_features, label, label_n, center_name_list

def comput_degree(adj_matrix):
    degree_matrix=np.zeros((90,90))
    colsum=adj_matrix.sum(axis=0)
    for j in range(adj_matrix.shape[0]):
        degree_matrix[j][j] = colsum[j]
    return degree_matrix