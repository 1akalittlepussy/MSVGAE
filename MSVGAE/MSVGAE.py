# -*- coding: utf-8 -*-
# @Author  : sw t
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import os
import pickle

EPS = 1e-15
MAX_LOGSTD = 10


# based on torch.nn.module class in torch

class MSVGAE(torch.nn.Module):
    def __init__(self, encoder_gat1, encoder_gat2, line_decoder_hid_dim=128):
        super(MSVGAE, self).__init__()

        # initialize parameter
        self.mu_gat2 = self.logstd_gat2 = None
        self.mu_gat1 = self.logstd_gat1 = None
        #self.mu_gat3 = self.logstd_gat3 = None
        # encoder
        self.encoder_gat1 = encoder_gat1
        self.encoder_gat2 = encoder_gat2
        #self.encoder_gat3 = encoder_gat3
        # use inner product decoder by default
        #self.decoder = InnerProductDecoder()
        self.dc = InnerProductDecoder(dropout=0.1, act=lambda x: x)
        # liner decoder
        self.liner_decoder = Sequential(
            Linear(in_features=self.encoder_gat1.latent_dim * 2, out_features=line_decoder_hid_dim),
            BatchNorm1d(line_decoder_hid_dim),
            ReLU(),
            Dropout(0.4),
            Linear(in_features=line_decoder_hid_dim, out_features=self.encoder_gat1.in_channels),
        )
        self.fea_weight = nn.Parameter(torch.randn(96, 62))
        self.fea_weight2 = nn.Parameter(torch.randn(64, 62))

    def encode(self, *args, **kwargs):
        """ encode """
        # GAT encoder
        self.mu_gat2, self.logstd_gat2 = self.encoder_gat2(*args, **kwargs)
        # GCN encoder
        self.mu_gat1, self.logstd_gat1 = self.encoder_gat1(*args, **kwargs)
        #self.mu_gat3, self.logstd_gat3 = self.encoder_gat3(*args, **kwargs)
        # fix range
        self.logstd_gat2 = self.logstd_gat2.clamp(max=MAX_LOGSTD)
        self.logstd_gat1 = self.logstd_gat1.clamp(max=MAX_LOGSTD)
        #self.logstd_gat3 = self.logstd_gat3.clamp(max=MAX_LOGSTD)
        # reparameter

        z_gat2 = self.reparametrize(self.mu_gat2, self.logstd_gat2)
        z_gat1 = self.reparametrize(self.mu_gat1, self.logstd_gat1)

        #z_gat3 = self.reparametrize(self.mu_gat3, self.logstd_gat3)
        z = torch.cat([z_gat1, z_gat2], dim=1)
        #print(z)
        #if self.logstd_gat2.requires_grad:
        #    self.logstd_gat2.register_hook(self.store_grad_norm)
        #if self.logstd_gat1.requires_grad:
        #    self.logstd_gat1.register_hook(self.store_grad_norm)

        #if z_gat2.requires_grad:
        #    z_gat2.register_hook(self.store_grad_norm)
        #if z_gat1.requires_grad:
        #    z_gat1.register_hook(self.store_grad_norm)

        X_pred = torch.mm(z, self.fea_weight)
        X_pred= F.leaky_relu(X_pred, inplace=0.2)

        #X_pred2 = torch.mm(z_gat2, self.fea_weight2)
        #X_pred2 = F.leaky_relu(X_pred2, inplace=0.2)
        #z = torch.cat([z_gat1, z_gat2], dim=1)
        return z, X_pred, self.mu_gat1, self.logstd_gat1, self.mu_gat2, self.logstd_gat2

    def reparametrize(self, mu, log_std):
        if self.training:
            return mu + torch.randn_like(log_std) * torch.exp(log_std)
        else:
            return mu

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """

        loss_kl = 0.0
        loss_kl = -0.5 * torch.mean(torch.sum(1 + 2 * self.logstd_gat2 - self.mu_gat2 ** 2 - self.logstd_gat2.exp()**2, dim=1))
        loss_kl += -0.5 * torch.mean(
            torch.sum(1 + 2 * self.logstd_gat1 - self.mu_gat1 ** 2 - self.logstd_gat1.exp() ** 2, dim=1))
        return loss_kl / 2

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

## Absolute position encoding

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        #z = F.dropout(z, self.dropout, training=self.training)
        #adj = self.act(torch.mm(z, z.t()))
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

class MLP_G(nn.Module):
    """MLP Generator"""

    def __init__(self, ninput, noutput, dropout=0.3):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.fc1 = nn.Linear(self.ninput, self.ninput)
        self.fc2 = nn.Linear(self.ninput, self.ninput // 2)
        self.fc3 = nn.Linear(self.ninput // 2, self.noutput)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout)
        x = self.fc3(x)
        #z = F.dropout(z, self.dropout, training=self.training)
        #adj = self.act(torch.mm(z, z.t()))
        #A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return x


class MLP_D(nn.Module):
    """MLP Generator"""

    def __init__(self, ninput, noutput, dropout=0.3):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.fc1 = nn.Linear(self.ninput, self.ninput)
        self.fc2 = nn.Linear(self.ninput, self.ninput // 2)
        self.fc3 = nn.Linear(self.ninput // 2, self.noutput)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout)
        x = self.fc3(x)
        #z = F.dropout(z, self.dropout, training=self.training)
        #adj = self.act(torch.mm(z, z.t()))
        #A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return x

## Absolute position encoding

class AdjEncoding(nn.Module):
    def __init__(self, normalization=None, zero_diag=False):
        """
        normalization: for Laplacian None. sym or rw
        """
        super(AdjEncoding,self).__init__( )
        self.normalization = normalization

    def compute_pe(self, edge_index):
        return to_dense_adj(edge_index)

class LapEncoding(nn.Module):
    def __init__(self, dim,d_model, use_edge_attr=False, normalization=None):
        super(LapEncoding, self).__init__()
        """
        normalization: for Laplacian None. sym or rw
        """
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr
        #self.d_model=d_model
        self.embedding_lap_pos_enc = nn.Linear(dim, d_model)

    def compute_pe(self, edge_index,edge_attr):
        edge_attr = edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = get_laplacian(
            edge_index, edge_attr, normalization=self.normalization)
        L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        lap_pos_enc=torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1]).float()
        return self.embedding_lap_pos_enc(lap_pos_enc)

    #def apply_to(self, dataset):
    #    dataset.lap_pe_list = []
    #    for i, g in enumerate(dataset):
    #        pe = self.compute_pe(g)
    #        dataset.lap_pe_list.append(pe)

    #    return dataset