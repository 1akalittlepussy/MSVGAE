import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from functools import partial
#from model_sub_v1 import GAT, GCN, GIN
from torch_geometric.data import Data
import copy

class GraphMAE(nn.Module):
    def __init__(self, encoder,decoder,mask_rate,drop_edge_rate,num_hidden,concat_hidden,replace_rate,num_heads,num_out_heads,in_dim,out_dim):
        super(GraphMAE, self).__init__()

        self.mask_rate=mask_rate
        self.drop_edge_rate=drop_edge_rate
        self.num_hidden=num_hidden
        self.concat_hidden=concat_hidden
        self.replace_rate=replace_rate
        self.num_heads=num_heads
        self.num_out_heads=num_out_heads
        self.in_dim=in_dim
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_dim))
        self.mask_token_rate = 1 - self.replace_rate
        self.encoder_1 = encoder
        self.decoder_1 = decoder
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden
        self.encoder_to_decoder = nn.Linear(out_dim, out_dim, bias=False)
        self.fea_weight = nn.Parameter(torch.randn(out_dim, in_dim))

        #self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        #self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def encoding_mask_noise(self, g ,mask_rate=0.3):
        x=g.x
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)


    def forward(self, data):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(data, self.mask_rate)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder_1(x=use_x, edge_index=use_g.edge_index, return_hidden=True)
        rep = self.encoder_to_decoder(enc_rep)
        rep[mask_nodes] = 0
        #rep = rep.detach().numpy()
        recon = self.decoder_1(x=rep, edge_index=pre_use_g.edge_index)
        X_pred = torch.mm(enc_rep, self.fea_weight)
        X_pred = F.leaky_relu(X_pred, inplace=0.2)

        return recon, X_pred

    def __repr__(self):
        return self.__class__.__name__
