
#from cv2 import mean
import numpy as np
import scipy.io
#from sympy import re
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
# from sympy import re
from matplotlib import pyplot as plt


def load_data(file_path, key_name_in_mat):
    data = scipy.io.loadmat(file_path)  # 读取mat文件

    return data[key_name_in_mat]
    
def get_thresh_val(np_mat, thresh_persent=0.36):
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
    index = int(np.size(sorted_all_nums) * (1-thresh_persent))
    
    # 防止极端情况index越界
    if index < 0:
        return sorted_all_nums[0]
    # 防止极端情况index越界
    if index >= np.size(sorted_all_nums):
        return sorted_all_nums[-1]

    return sorted_all_nums[index]



def convert_binary_by_thresh_val(image_matrix,thresh_val:int):
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
    #final_conv_numpy = final_conv.cpu().detach().numpy()
    #scipy.io.savemat('conv.mat', {'data': final_conv_numpy})

    return final_conv

def knn_generate_graph(src_mat, n_neighbors=10):
    '''
        - src_mat: array-like of shape (n_features, n_samples)
        - k: Number of neighbors for each sample.
    '''

    if n_neighbors >= src_mat.shape[1]:
        raise RuntimeError("knn 选取k数大于节点数量")

    feature = np.corrcoef(src_mat.T)

    A = kneighbors_graph(feature, n_neighbors, mode='connectivity', include_self=False)

    return A.toarray()

def partialCorrelationMatrix(data):
    """
    偏相关系数矩阵

    参数
    --------------
    :param data: 输入样本数据
        维度：n*m，n(变量数)，m(样本数)
    :return: 偏相关系数矩阵
        维度：n*n
    """
    data = np.array(data)
    cov = np.cov(data)
    #sns.heatmap(cov, cmap='Blues', annot=False)
    #plt.matshow(cov)
    #plt.savefig('test3.png')
    # 相关系数矩阵

    #inverse_cov = np.linalg.inv(cov)  # 相关系数矩阵的逆
    #partial_cov = np.zeros(inverse_cov.shape)  # 偏相关系数矩阵
    #for i in range(partial_cov.shape[0]):
    #    for j in range(i, partial_cov.shape[1]):
    #        partial_cov[i, j] = -inverse_cov[i, j] / (np.sqrt(inverse_cov[i, i]) * np.sqrt(inverse_cov[j, j]))
    #        partial_cov[j, i] = partial_cov[i, j]
    return cov
