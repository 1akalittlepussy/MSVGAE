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


def FSDataset(root_dir):
    #def __init__(self, root_dir):

        data_files = glob(os.path.join(root_dir,"**","*.pkl"))
        data_files.sort()
        fc = []
        for file in data_files:
            with open(file, "rb") as f:
                data = pickle.load(f)
            feature = data.source_mat.T
            feature = code_utils.partialCorrelationMatrix(feature)

            thresh_val=code_utils.get_thresh_val(feature)
            adj=code_utils.convert_binary_by_thresh_val(feature,thresh_val)

            label =int(re.match(r".*cortical_.+-(\d)-.+?\.pkl", file).group(1)) - 1
            #label=int(re.match(r".*cortical.+\d+", file).group(1))-1
            #label = 0 if file.split('/')[-3] == "female" else 1
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
            ))

        return fc
