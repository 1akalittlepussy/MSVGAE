import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import GraphConvolution
from utils_1 import to_gpu
import json
import os
import args
import numpy as np

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred

class AGE(nn.Module):
    def __init__(self, dropout=0.):
        super(AGE, self).__init__()
        self.gc1 = GraphConvolution(args.input_dim, args.hidden1_dim)
        self.gcn_mean = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.gcn_logstddev = GraphConvolution(args.hidden1_dim, args.hidden2_dim, activation=lambda x: x)
        self.dropout = dropout
        self.fea_weight = nn.Parameter(torch.randn(args.hidden2_dim, args.input_dim))

    def forward(self, x, adj):
        hidden = self.gc1(x, adj)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(hidden.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        A_pred = dot_product_decode(sampled_z)
        # X=leaky_relu(Z*W)
        X_pred = torch.mm(sampled_z, self.fea_weight)
        X_pred = F.leaky_relu(X_pred, inplace=0.2)
        return A_pred, X_pred

class MLP_D(nn.Module):
    """
    Discriminator Class using MLP
    """
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        """
            parse network structure
        """
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        #layer_sizes = layers
        self.layers = []

        """
            padding different layers, one layer includes liner part, batchNormalization part and activation part
        """
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                #self.layers.append(bn)
                #self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        """
            padding output layer
        """
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x)
        #x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass



class GraphTransformer(nn.Module):
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalSum1D()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )
        self.bn1 = nn.BatchNorm1d(in_size)

    def forward(self, x, masks, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = self.bn1(x)
        #x = x.permute(1, 0, 2)

        #scaler = StandardScaler()
        #res_2 = torch.from_numpy(scaler.fit_transform(x.cpu()))
        #res_2 = torch.tensor(res_2,dtype=torch.float).cuda()
        output = self.embedding(x)
        output=torch.unsqueeze(output, 1)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)
        output=self.classifier(output)
        output = torch.mean(output)
        #output = torch.squeeze(output,1)
        # we only do mean pooling for now.
        return output

    def init_weights(self):
        """
        Initialize model parameters
        :return: no return
        """
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass



class MLP_G(nn.Module):
    """
        Generator Class using MLP
    """
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        """
            parse network structure
        """
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        """
            padding different layers, one layer includes liner part, batchNormalization part and activation part
        """
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        """
            padding output layer
        """
        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        """
        Initialize model parameters
        :return: no return
        """
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class GraphTransformer_G(nn.Module):
    def __init__(self, in_size, nb_class, d_model, nb_heads,
                 dim_feedforward=2048, dropout=0.1, nb_layers=4,
                 lap_pos_enc=False,lap_pos_enc_dim=0):
        super(GraphTransformer_G, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)

        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        #if lap_pos_enc and lap_pos_enc_dim > 0:
            # We embed the pos. encoding in a higher dim space
            # as Bresson et al. and add it to the features.
        self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalSum1D()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
        )
        self.bn1 = nn.BatchNorm1d(in_size)

    def forward(self, x, masks, x_lap_pos_enc=None, degree=None):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        x = self.bn1(x)
        #x = x.permute(1, 0, 2)

        #scaler = StandardScaler()
        #res_2 = torch.from_numpy(scaler.fit_transform(x.cpu()))
        #res_2 = torch.tensor(res_2,dtype=torch.float).cuda()
        output = self.embedding(x)
        output=torch.unsqueeze(output, 1)
        #if self.lap_pos_enc and x_lap_pos_enc is not None:
        #    x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
        #    x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
        #    output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        #output = self.pooling(output, masks)
        #output=self.classifier(output)
        #output = torch.mean(output)
        output = torch.squeeze(output,dim=0)
        # we only do mean pooling for now.
        return output

    def init_weights(self):
        """
        Initialize model parameters
        :return: no return
        """
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

class Seq2Seq(nn.Module):
    """
        LSTM autoEncoding
    """
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.emb_nhidden = None

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        """
        Initializing model weights
        :return:
        """
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        """
        Initializing hidden nodes
        :param bsz: batch size
        :return:
        """
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        """
        Initializing state
        :param bsz: batch size
        :return:
        """
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

#return embedding of current batch

    def return_emb(self):
        return self.emb_nhidden

    def embed_dictionary(self, dic):
        embeddings = self.embedding(dic)
        return embeddings

    def embed_after_LSTM(self, dic, length):
        embeddings = self.forward(dic, length, noise = False)
        return self.emb_nhidden

    def forward(self, indices, lengths, noise, encode_only=False):
        """
        Given nodes, feedforward to get the embedding codes of nodes
        :param indices: node id's
        :param lengths: length of walk
        :param noise:
        :param encode_only:
        :return:
        """
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)
        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        hidden = hidden / torch.norm(hidden, p=2, dim=1, keepdim=True)

        # norms = torch.norm(hidden, 2, 1)
        # hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        # ADD this can be used as embedding
        self.emb_nhidden = output.contiguous().view(-1, self.nhidden)

        return decoded


def load_models(load_path):
    """
    Load models save in args.json file and word vocabulary in vocab.json
    :param load_path: fold to store args.json and vocab.json files
    :return: model hyper-parameters, word and index, AE model, GAN-generator model, GAN-discriminator model
    """
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc

class GlobalSum1D(nn.Module):
    def __init__(self):
        super(GlobalSum1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1)