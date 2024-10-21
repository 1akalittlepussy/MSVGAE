
import argparse
import os
import time
import math
import numpy as np
import random

# for logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from load_data import FSDataset
from preprocessing import *
from torch.optim import Adam

# import utils for data preparation and algorithmic models
from utils_1 import to_gpu, Corpus, batchify, generate_walks
from NetRA import Seq2Seq, MLP_D, MLP_G,AGE,GraphTransformer,GraphTransformer_G
from GraphVAE.GraphVAE import GCNModelVAE
from MSVGAE.MSVGAE import MSVGAE, LapEncoding, AdjEncoding, MLP_G, MLP_D
from MSVGAE.MSVGAE_Encoder import GAT_Encoder, GCN_Encoder
# import visualization module
from viz_karate import viz

EMBED_SEGMENT = 4000

# import networkx package for graph data input parse, need to install networkx by: pip install networkx
import networkx as nx

# import scipy package for geting laplacian of graph, need to install networkx by: pip install scipy
from scipy.sparse import csgraph

"""Parameters to parse
        Path Arguments: The input and output directory
        Data Processing Arguments: data preprocessing for generating ``walks'' from the graph
        Model Arguments: parameters for the model
        Training Arguments, Evaluation Arguments, and others like 
"""
parser = argparse.ArgumentParser(description='NetRA')
# Path Arguments
#parser.add_argument('--data_path', type=str, default='../data/karate.adjlist',
#                    help='location of the data corpus')  # location of the graph with linked list format
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')  # location of output embeddings in different epochs

# Data Processing Arguments
parser.add_argument('--maxlen', type=int, default=100,
                    help='maximum sentence length')  # the parameter is for random walk to generating walks,
# this is the upbound of the walk length, in the code we generate walks with the same length


# Model Arguments
################### important hyper-parameters ################################
parser.add_argument('--hidden1_dim', type=int, default=96,
                    help='number of hidden dim1 per layer')
parser.add_argument('--hidden2_dim', type=int, default=64,
                    help='number of hidden dim2 per layer')
parser.add_argument('--nhidden', type=int, default=2,
                    help='number of hidden units per layer')  # dimension of embedding vectors, since we want to visualize to 2-dimensional
parser.add_argument('--emsize', type=int, default=30,
                    help='size of word embeddings')  # large graph 100-300, this is the size of input after original one hot embedding's mapping

################### typically below are set to default ones ###################
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')  # number of stacked LSTM for autoencoding
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')  # stard deviation of noise for autoencoder
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')  # decay rate for exponentially decaying noise on autoencoder
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='32-32',
                    help='generator architecture (MLP)')  # specify the MLP structure of generator in GAN;
# for example, 300-300 means two layers, each layer includes 300 nodes
parser.add_argument('--arch_d', type=str, default='32-32',
                    help='critic/discriminator architecture (MLP)')  # specify the MLP structure of discriminator in GAN;
# for example, 300-300 means two layers, each layer includes 300 nodes
parser.add_argument('--z_size', type=int, default=32,
                    help='dimension of random noise z to feed into generator')  # random noise to be feed into the generator
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')  # specify the temperature of softmax, \tau
parser.add_argument('--enc_grad_norm', type=bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_toenc', type=float, default=-0.01,
                    help='weight factor passing gradient from gan to encoder')  # weight factor passing from gradient of GAN to encoder, thi is used by grad_hook
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')  # dropout to prevent overfitting, by default, there is no dropout

# Training Arguments
################### important hyper-parameters ################################
parser.add_argument('--epochs', type=int, default=50,
                    help='maximum number of epochs')  # epochs for training, usually small graph 50, large graph 100
parser.add_argument('--walk_length', type=int, default=20,
                    help='length of walk sampled from the graph')  # the length of walk sampled rooted from each node
parser.add_argument('--numWalks_per_node', type=int, default=30,
                    help='number of walks sampled for each node')  # number of walks sampled for each node
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')  # batch size for training
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')  # in each epoch, number of iterations for training autoencoder
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')  # in each epoch, number of iterations for training discriminator
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')  # in each epoch, number of iterations for training generator

parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6-10-20-30-40',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')  # in different epochs, dynamically increase the GAN iterations,
# for example, 2-4-6 means, 2 epochs then increase one, 4 epochs then increase again

################### typically below are set to default ones ###################
parser.add_argument('--min_epochs', type=int, default=6,
                    help="minimum number of epochs to train for")  # minimum nuber of epochs for training
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")  # if conduct early stopping
parser.add_argument('--lr_ae', type=float, default=0.001,
                    help='autoencoder learning rate')  # learning rate for AE, because it is using SDG, by default it is 1
parser.add_argument('--lr_gan_g', type=float, default=0.001,
                    help='generator learning rate')  # learning rate for generator, because it is using ADM, by default it is a smaller one
parser.add_argument('--lr_gan_d', type=float, default=0.001,
                    help='critic/discriminator learning rate')  # learning rate for discriminator, because it is using ADM, by default it is a smaller one
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')  # beta for adam
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')  # gradient clipping
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')  # WGAN clamp

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

parser.add_argument('--dim_hidden', type=int, default=8)
parser.add_argument('--dim_hidden_1', type=int, default=100)
# parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='adj')
#parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default='adj')
parser.add_argument('--lappe', action='store_true', help='use laplacian PE', default=False)
#parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')
#parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--nb_heads', type=int, default=4)
parser.add_argument('--nb_layers', type=int, default=3)
# Other
parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='random seed')
# random seeds for parameter initialization
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')  # use CUDA for training
parser.add_argument('--hidden_dims', type=int, nargs='*', default=[32,64])
parser.add_argument('--GAT', action='store_true', help='use GAT', default=True)
parser.add_argument('--GCN', action='store_true', help='use GCN', default=False)
parser.add_argument('--num_heads',
                    help='Number of attention heads for each layer. Input is a list that must match the total number of layers = num_hidden_layers + 2 in length.',
                    type=int, nargs='*', default=[1, 1, 1, 1])
parser.add_argument('--latent_dim', help='Latent dimension (output dimension for node embeddings).', default=[32,64],
                    type=int)

args = parser.parse_args()
print(vars(args))

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.outf)):
    os.makedirs('./output/{}'.format(args.outf))

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

#print(os.getcwd())

os.environ['CUDA_VISIBLE_DEVICES'] = ""

#fc_adj, fc_adj_label, fc_features, label_name,label_n = FSDataset(r'')F:\FCSC_pytorch\HCP62ROI\male
fc_adj, fc_adj_label, fc_features, label_name,label_n, edge_index = FSDataset(r'F:\SC_pytorch\ZDXX\classify\0')

adj = []
adj_norm = []
# adj_label = sc_adj_label
adj_label = fc_adj_label
# num_nodes = sc_adj[0].shape[0]
num_nodes = fc_adj[0].shape[0]
features = []
center_name = []

# fc数据
for i in range(len(fc_adj)):
    adj_norm.append(preprocess_graph_wgan(fc_adj[i]))
    features.append(fc_features[i])
    adj_label[i] += sp.eye(adj_label[i].shape[0], adj_label[i].shape[1])
    adj_label[i] = sparse_to_tuple(sp.csr_matrix(adj_label[i] / 1.0))
print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

#ntokens = len(corpus.dictionary.word2idx)
#autoencoder = Seq2Seq(emsize=args.emsize,
#                      nhidden=args.nhidden,
#                      ntokens=ntokens,
#                      nlayers=args.nlayers,
#                      noise_radius=args.noise_radius,
#                      hidden_init=args.hidden_init,
#                      dropout=args.dropout,
#                      gpu=args.cuda)

#autoencoder=AGE()

#if args.cuda:
#    autoencoder = autoencoder.cuda()
#    gan_gen = gan_gen.cuda()
#    gan_disc = gan_disc.cuda()
#    criterion_ce = criterion_ce.cuda()


###############################################################################
# Training code
###############################################################################



def unique(tensor1d):
    t, idx = np.unique(tensor1d.cpu().data.numpy(), return_index=True)
    return t, idx



def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD



#def grad_hook(grad):
    # Gradient norm: regularize to be same
    # code_grad_gan * code_grad_ae / norm(code_grad_gan)
#    if args.enc_grad_norm:
#        gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
#        normed_grad = grad * autoencoder.grad_norm / gan_norm
#    else:
#        normed_grad = grad

    # weight factor and sign flip
#    normed_grad *= -math.fabs(args.gan_toenc)
#    return normed_grad



print("Training...")
with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1  # start from 1, and will be dynamically increased

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one: object = to_gpu(args.cuda, torch.tensor(1, dtype=torch.float))
mone: int = one * -1



def MSVAGE_GAN():
    for sub in range(len(label_n)):
        # nn.init.kaiming_normal()
        adj_norm[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[sub][0].T),
                                                 torch.FloatTensor(adj_norm[sub][1]),
                                                 torch.Size(adj_norm[sub][2])).cuda()
        adj_label[sub] = torch.sparse.FloatTensor(torch.LongTensor(adj_label[sub][0].T),
                                                  torch.FloatTensor(adj_label[sub][1]),
                                                  torch.Size(adj_label[sub][2])).cuda()
        features[sub] = torch.FloatTensor(features[sub]).cuda()
    #test=adj_label[sub]
    feat_dim = 62
    in_size = 8
    num_heads = {}
    num_heads['first'] = 5
    num_heads['second'] = 5
    num_heads['mean'] = 5
    num_heads['std'] = 5
    # num_heads=[5, 5, 5, 5]
    # autoencoder = GCNModelVAE(feat_dim, args.hidden1_dim, args.hidden2_dim, args.dropout)
    if args.GAT:
        encoder_1 = GAT_Encoder(
            in_channels=feat_dim,
            num_heads=num_heads,
            hidden_dims=args.hidden_dims,
            dropout=0.4,
            latent_dim=32
        )
        encoder_2 = GAT_Encoder(
            in_channels=feat_dim,
            num_heads=num_heads,
            hidden_dims=args.hidden_dims,
            dropout=0.4,
            latent_dim=64
        )
        encoder_3 = GAT_Encoder(
            in_channels=feat_dim,
            num_heads=num_heads,
            hidden_dims=args.hidden_dims,
            dropout=0.4,
            latent_dim=128
        )
    if args.GCN:
        encoder_1 = GCN_Encoder(
            in_channels=feat_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim
        )
        encoder_2 = GCN_Encoder(
            in_channels=feat_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim
        )

    autoencoder = MSVGAE(encoder_gat1=encoder_1, encoder_gat2=encoder_2).cuda()
    # gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
    gan_gen_1 = MLP_G(ninput=args.z_size, noutput=args.hidden1_dim, layers=args.arch_g).cuda()
    gan_gen_2 = MLP_G(ninput=args.z_size, noutput=args.hidden2_dim, layers=args.arch_g).cuda()
    gan_disc1 = MLP_D(ninput=96, noutput=2, layers=args.arch_d).cuda()
    gan_disc2 = MLP_D(ninput=64, noutput=2, layers=args.arch_d).cuda()
    Lap_Encode=LapEncoding(dim=args.lap_dim,d_model=62,use_edge_attr=False, normalization=None)
    Adj_Encode = AdjEncoding(normalization=None, zero_diag=False)

    ## SGD, learning rate should be larger, like 1, Adam's learning rate should be smaller, like 0.001
    # optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
    optimizer_ae = Adam(autoencoder.parameters(), lr=args.learning_rate)
    #optimizer_gan_g = optim.Adam(gan_gen_1.parameters(),
    #                             lr=args.lr_gan_g,
    #                             betas=(args.beta1, 0.999))
    optimizer_gan_g = optim.Adam(gan_gen_1.parameters(),
                                 lr=args.lr_gan_g)
    #optimizer_gan_d1 = optim.Adam(gan_disc1.parameters(),
    #                              lr=args.lr_gan_d,
    #                              betas=(args.beta1, 0.999))
    optimizer_gan_d1 = optim.Adam(gan_disc1.parameters(),
                                  lr=args.lr_gan_d)
    optimizer_gan_d2 = optim.Adam(gan_disc2.parameters(),
                                  lr=args.lr_gan_d,
                                  betas=(args.beta1, 0.999))
    #### crossEntropy loss for discriminator
    criterion_ce = nn.CrossEntropyLoss()

    print("Starting Training Loop...")

    for epoch in range(301):
        t = time.time()
        train_acc = 0
        loss_list = []
        G_loss_list = []
        D_loss_list = []
        for sub in range(len(label_n)):
            loss_func = nn.BCEWithLogitsLoss()

            # real_label = torch.tensor([1],dtype=torch.float32).cuda()
            real_label = torch.ones(62, 2)
            fake_label = torch.zeros(62, 2)
            edge_attr=None
            lap_pos_encode= Lap_Encode.compute_pe(edge_index[sub],edge_attr).cuda()
            #adj_encode =  Adj_Encode.compute_pe(edge_index[sub]).squeeze(0).cuda()
            #lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
            #lap_pos_encoder.apply_to(train_dset)
            #lap_pos_encoder.apply_to(val_dset)

            # batch_size x nhidden
            #feature = torch.from_numpy(features[sub]).to(torch.float32)
            features[sub]=features[sub]+lap_pos_encode
            z, X_pred, mu1, log1, mu2, log2 = autoencoder.encode(features[sub], adj_norm[sub])
            # print(z)
            reconstructed_A = autoencoder.dc(z)
            A = reconstructed_A.cpu().detach().numpy()
            # print(A)
            X = X_pred.cpu().detach().numpy()
            np.savez('F:/SC_pytorch/ZDXX/classify/ablation/lap_PE' + '/fake_sMRI_' + str(label_n[sub]) + str(sub), fake_A=A,
                     fake_X=X)
            optimizer_ae.zero_grad()
            # mae_loss=torch.nn.L1Loss( )
            # loss = mae_loss(torch.tensor(A),torch.tensor(adj))

            # loss / backprop
            optimizer_gan_d1.zero_grad()

            errD_real = gan_disc1(z)
            # errD_real = torch.tensor([errD_real]).cuda()
            D_loss_real = loss_func(errD_real, real_label.cuda())
            #D_loss_real.backward(retain_graph=True)
            # print(errD_real)
            gen_size = 62
            #noise = to_gpu(args.cuda,
            #               Variable(torch.ones(gen_size, args.z_size)))
            #noise.data.normal_(0, 1)

            noise = torch.randn(gen_size, args.z_size).cuda()

            fake_hidden = gan_gen_1(noise)

            errD_fake = gan_disc1(fake_hidden)
            #print(errD_fake.cuda())
            D_loss_fake = loss_func(errD_fake, fake_label.cuda())
            #D_loss_fake.backward(retain_graph=True)
            # print(errD_fake)

            #D_loss = errD_real + errD_fake
            D_loss = (D_loss_fake + D_loss_real)/2
            #D_loss = -torch.mean(errD_real) + torch.mean(errD_fake)
            D_loss_list.append(D_loss.item())
            D_loss.backward(retain_graph=True)
            optimizer_gan_d1.step()
            torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

            optimizer_gan_g.zero_grad()
            G_fake = gan_disc1(fake_hidden.detach())
            G_loss = loss_func(G_fake, real_label.cuda())
            #G_loss = -torch.mean(G_fake)
            G_loss_list.append(G_loss.item())
            G_loss.backward(retain_graph=True)
            optimizer_gan_g.step()
            #errD = -(errD_real - errD_fake)

            #print(adj_norm[sub])
            adj_dense = adj_label[sub].to_dense()
            loss = F.mse_loss(reconstructed_A, adj_dense.cuda())

            kl_loss_1 = 0.5 / torch.tensor(A).size(0) * (
                    1 + 2 * autoencoder.logstd_gat1 - autoencoder.mu_gat1 ** 2 - torch.exp(
                autoencoder.logstd_gat1) ** 2).sum(1).mean()

            kl_loss_2 = 0.5 / torch.tensor(A).size(0) * (
                    1 + 2 * autoencoder.logstd_gat2 - autoencoder.mu_gat2 ** 2 - torch.exp(
                autoencoder.logstd_gat2) ** 2).sum(1).mean()

            # kl_loss_3 = 0.5 / torch.tensor(A).size(0) * (
            #            1 + 2 * autoencoder.logstd_gat3 - autoencoder.mu_gat3 ** 2 - torch.exp(autoencoder.logstd_gat3) ** 2).sum(1).mean()
             #loss2 -= kl_divergence

            kl_loss = (kl_loss_1 + kl_loss_2) / 2

            loss -= kl_loss
            loss_list.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer_ae.step()

        print("Epoch:", '%04d' % (epoch),
              "loss=", "{:.5f}".format(np.mean(loss_list)),
              "G_loss=", "{:.5f}".format(np.mean(G_loss_list)),
              "D_loss=", "{:.5f}".format(np.mean(D_loss_list)))

        #return errD, errD_real, errD_fake


MSVAGE_GAN()



