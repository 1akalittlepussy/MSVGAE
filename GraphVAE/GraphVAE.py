import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphVAE.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden1_dim, hidden2_dim, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden1_dim, dropout, F.relu)
        self.gc2 = GraphConvolution(hidden1_dim, hidden2_dim, dropout, lambda x: x)
        self.gc3 = GraphConvolution(hidden1_dim, hidden2_dim, dropout, lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.bn1 = nn.BatchNorm1d(input_feat_dim)
        self.fea_weight = nn.Parameter(torch.randn(hidden2_dim, input_feat_dim))

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, x, adj):
        x = self.bn1(x)
        mu, logvar = self.encode(x, adj)
        if logvar.requires_grad:
            logvar.register_hook(self.store_grad_norm)
        z = self.reparameterize(mu, logvar)
        A_pred=self.dc(z)
        X_pred = torch.mm(z, self.fea_weight)
        X_pred = F.leaky_relu(X_pred, inplace=0.2)
        return A_pred,X_pred, mu, logvar, z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
