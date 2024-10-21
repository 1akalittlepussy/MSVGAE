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
import pingouin as pg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_thresh_val(np_mat, thresh_persent=0.30):
    '''
        input:  np_mat-nparray格式矩阵
                thresh_persent-需要计算的阈值百分比

        output: np_mat中第thresh_persent小的数值
    '''
    # 打平到一维矩阵
    all_nums = np_mat.flatten()

    # 排序
    sorted_all_nums = np.sort(all_nums)
    # print(sorted_all_nums)
    # 20%的二值化位置
    index = int(np.size(sorted_all_nums) * (1 - thresh_persent))

    # 防止极端情况index越界
    if index < 0:
        return sorted_all_nums[0]
    # 防止极端情况index越界
    if index >= np.size(sorted_all_nums):
        return sorted_all_nums[-1]

    return sorted_all_nums[index]


def convert_binary_by_thresh_val(image_matrix, thresh_val: int):
    '''
        input:  image_matrix-nparray格式矩阵
                thresh_val-阈值

        output: image_matrix二值化处理
    '''
    upper_limit = 1
    lower_limit = -1

    temp_conv = np.where((image_matrix >= thresh_val), upper_limit, lower_limit)
    final_conv = np.where((temp_conv == -1), 0, temp_conv)
    np.fill_diagonal(final_conv, 0)
    # final_conv_numpy = final_conv.cpu().detach().numpy()
    # scipy.io.savemat('conv.mat', {'data': final_conv_numpy})

    return final_conv

def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices


class FSDataset(object):
    def __init__(self, root_dir, folds, seed):

        data_files = glob(os.path.join(root_dir,"**","*.pkl"))
        data_files.sort()
        self.fc = []
        for file in data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)

            feature = data.source_mat.T
            #feature = np.corrcoef(feature)
            #df = pd.DataFrame(feature)
            #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
            #df2 = np.array(df2.pcorr())
            feature = code_utils.partialCorrelationMatrix(feature)

            thresh_val=code_utils.get_thresh_val(feature)
            adj=code_utils.convert_binary_by_thresh_val(feature,thresh_val)

            #thresh_val=code_utils.get_thresh_val(df2)
            #adj=code_utils.convert_binary_by_thresh_val(df2,thresh_val)


            label =int(re.match(r".*cortical_.+-(\d)-.+?\.pkl", file).group(1)) - 1
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            self.fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
            ))

        data_files_fake = glob(os.path.join(root_dir, "*.npz"))
        data_files_fake.sort()
        for file in data_files_fake:
            with open(file, "rb") as f:
               file_r = np.load(f)
               file_r.files
               fake_X=file_r['fake_X']
               #fake_A=file_r['fake_A']
               #fake_X= np.corrcoef(fake_X)
               #df = pd.DataFrame(fake_X)
               #df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
               #df2 = np.array(df2.pcorr())
               #fake_X = code_utils.partialCorrelationMatrix(fake_X)
               #thresh_val = code_utils.get_thresh_val(df2)
               #fake_A = code_utils.convert_binary_by_thresh_val(df2, thresh_val)
               thresh_val = code_utils.get_thresh_val(fake_X)
               fake_A = code_utils.convert_binary_by_thresh_val(fake_X, thresh_val)
               fcedge_index, _ = dense_to_sparse(torch.from_numpy(fake_A.astype(np.int16)))
               lst = file.split("/")[4].split("\\")[1]
               lst=int(lst[10])
               if lst==0:
                   label=0
               else:
                   label=1
               self.fc.append(Data(
                   x=torch.from_numpy(fake_X).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
               ))

        #fc=self.fc
        #print(fc)
        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)

    def kfold_split(self, test_index):
        assert test_index < self.k_fold
        # valid_index = (test_index + 1) % self.k_fold
        valid_index = test_index
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.fc))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.fc, train_split.tolist())
        valid_subset = Subset(self.fc, valid_split.tolist())
        test_subset = Subset(self.fc, test_split.tolist())

        return train_subset, valid_subset, test_subset, train_split,valid_split,test_split

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.fc)

if __name__ == "__main__":

   #data = io.loadmat(r"..\data_graph\MDD\ROISignals_FunImgARCWF\ROISignals_S1-1-0005\ROISignals_S1-1-0005.mat")
   data = io.loadmat(r"D:\Project\python\graphmix\data_graph\MDD\data_90_230\len_90\ROISignals_S5-1-0001.mat")
   print(len(data['ROISignals'][0]),len(data['ROISignals']))