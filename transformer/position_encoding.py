import os
import pickle
import re
import torch
import torch.nn as nn
#from torch_geometric.graphgym.config import cfg
#from torch_geometric.graphgym.register import register_node_encoder
import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm


class PositionEncoding(object):
    def __init__(self, savepath=None, zero_diag=False):
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                pe = saved_pos_enc[i]
            if self.zero_diag:
                pe = pe.clone()
                pe.diagonal()[:] = 0
            
            dataset.pe_list.append(pe)

        self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        pass


class DiffusionEncoding(PositionEncoding):
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = expm(-self.beta * L)
        return torch.from_numpy(L.toarray())


class PStepRWEncoding(PositionEncoding):
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        L = sp.identity(L.shape[0], dtype=L.dtype) - self.beta * L
        tmp = L
        for _ in range(self.p - 1):
            tmp = tmp.dot(L)
        return torch.from_numpy(tmp.toarray())


class AdjEncoding(PositionEncoding):
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization

    def compute_pe(self, graph):
        return to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)

class FullEncoding(PositionEncoding):
    def __init__(self, savepath, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
        def __init__(self, dim, use_edge_attr=False, normalization=None):
            """
            normalization: for Laplacian None. sym or rw
            """
            self.pos_enc_dim = dim
            self.use_edge_attr = use_edge_attr

        def compute_pe(self, graph):
            W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
            W = W0
            vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
            vector[:, 0] = torch.from_numpy(W0.diagonal())
            for i in range(self.pos_enc_dim - 1):
                W = W.dot(W0)
                vector[:, i + 1] = torch.from_numpy(W.diagonal())
            return vector.float()

        def apply_to(self, dataset):
            dataset.lap_pe_list = []
            for i, g in enumerate(dataset):
                pe = self.compute_pe(g)
                dataset.lap_pe_list.append(pe)
            return dataset


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
        edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     device=edge_index.device)
        num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

        return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)



POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
}
