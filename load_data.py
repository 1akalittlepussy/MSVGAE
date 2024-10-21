import pickle
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import scipy.io as scio
from scipy import io
from torch.utils.data import Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from glob import glob
import re
import pandas as pd
import code_utils
import seaborn as sns
# from sympy import re
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
#import pingouin as pg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices


#class FSDataset(object):
#    def __init__(self, root_dir):
def FSDataset(root_dir):
        data_files = glob(os.path.join(root_dir,"**","*.pkl"))
        data_files.sort()
        #self.fc = []
        label_n = []
        label_name = []
        fc_adj = []
        fc_features = []
        fc_adj_label = []
        edge_index=[]
        for file in data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            #adj = data.adjacency_mat

           # degree = data.degree
           #  degree1 = np.array(degree).reshape((90,1))
           #  degree1 =degree1.tolist()
           # clustering=data.clustering
           # clustering1 = np.array(clustering).reshape((90,1))
           # clustering1 = clustering1.tolist()
           # nodal_efficiency=data.nodal_efficiency
           # nodal_efficiency1= np.array(nodal_efficiency).reshape((90,1))
           # nodal_efficiency1 = nodal_efficiency1.tolist()

            #feature = data.source_mat.T
            #feature = np.hstack((feature,degree1))
            #feature =np.hstack((feature,clustering1))
            #feature =np.hstack((feature,nodal_efficiency1))

            feature = data.source_mat.T
            #feature =  np.delete(feature,1,axis = 1)
            #feature = np.delete(feature, 0, axis=1)
            #feature = np.corrcoef(feature)
            feature = code_utils.partialCorrelationMatrix(feature)
            fc_features.append(feature)

            #feature = code_utils.partialCorrelationMatrix(feature)

            #scaler = StandardScaler()
            #res_2 = scaler.fit_transform(feature)
            #res_2 = torch.tensor(res_2, dtype=torch.float).cuda()

            #偏相关
            #df = pd.DataFrame(feature)
            #df2= pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            #df2 = np.array(df2.pcorr())
            #sns.heatmap(df2, cmap='Blues', annot=False)
            #plt.matshow(df2)
            #plt.savefig('test7.png')

            #spearman
            #df = pd.DataFrame(feature)
            #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            #df2 = np.array(df.corr('spearman'))

            thresh_val=code_utils.get_thresh_val(feature)
            adj=code_utils.convert_binary_by_thresh_val(feature,thresh_val)
            fc_adj.append(adj)
            fc_adj_label.append(adj)

            #scaler = StandardScaler()
            #res_2 = scaler.fit_transform(feature)
            #res_2 = torch.tensor(res_2, dtype=torch.float).cuda()

            #thresh_val=code_utils.get_thresh_val(df2)
            #adj=code_utils.convert_binary_by_thresh_val(df2,thresh_val)

            #adj=code_utils.knn_generate_graph(data.source_mat)

            #feature = np.hstack((feature, feature1))
            # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
            #label = int(re.match(r".*ROISignals_.+-(\d)-.+?\.pkl", file).group(1)) - 1


            #label=int(re.match(r".*cortical.+\d+", file).group(1))-1
            #label = 0 if file.split('/')[-3] == "female" else 1

            fc_edge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            edge_index.append(fc_edge_index)
            label = int(re.match(r".*cortical_.+-(\d)-.+?\.pkl", file).group(1)) - 1
            #label_n=torch.as_tensor(label).long()
            label_n.append(label)
            label_name = []
            label_name.append(file)
            #self.fc.append(Data(
            #    x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
            #    ))

        return fc_adj, fc_adj_label, fc_features, label_name, label_n, edge_index
