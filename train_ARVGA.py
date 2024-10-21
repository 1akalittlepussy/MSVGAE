# for parsing input command
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

# import utils for data preparation and algorithmic models
from utils_1 import to_gpu, Corpus, batchify, generate_walks
from NetRA import Seq2Seq, MLP_D, MLP_G,AGE,GraphTransformer,GraphTransformer_G
from GraphVAE.GraphVAE import GCNModelVAE
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
parser.add_argument('--hidden1_dim', type=int, default=16,
                    help='number of hidden dim1 per layer')
parser.add_argument('--hidden2_dim', type=int, default=32,
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
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')  # specify the MLP structure of generator in GAN;
# for example, 300-300 means two layers, each layer includes 300 nodes
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')  # specify the MLP structure of discriminator in GAN;
# for example, 300-300 means two layers, each layer includes 300 nodes
parser.add_argument('--z_size', type=int, default=100,
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
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')  # learning rate for generator, because it is using ADM, by default it is a smaller one
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
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
parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')
#parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--nb_heads', type=int, default=4)
parser.add_argument('--nb_layers', type=int, default=3)
# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')  # random seeds for parameter initialization
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')  # use CUDA for training

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

fc_adj, fc_adj_label, fc_features, label_name,label_n = FSDataset(r'F:\SC_pytorch\HCP\classify')
#fc_adj, fc_adj_label, fc_features, label_name,label_n = FSDataset(r'F:\FCSC_pytorch\HCP62ROI\male')
adj = []
adj_norm = []
# adj_label = sc_adj_label
adj_label = fc_adj_label
# num_nodes = sc_adj[0].shape[0]
num_nodes = fc_adj[0].shape[0]
features = []
center_name = []
adj_size=len(adj_label)
#adj_size=np.array(adj_label)
#adj_size=adj_size.size()[0]
L = [None]*adj_size

for i in range(len(fc_adj)):
    adj_norm.append(preprocess_graph(fc_adj[i]))
    features.append(fc_features[i])
    adj_label[i] += sp.eye(adj_label[i].shape[0], adj_label[i].shape[1])
    #walks = generate_walks(adj_label[i], args.numWalks_per_node, args.walk_length)
    L[i] = csgraph.laplacian(adj_label[i], normed=False)  # use csgraph package to calculate the laplacian
    #L[i] = np.array(L[i], np.float32)
    L[i] = Variable(torch.from_numpy(L[i]))
    L[i] = to_gpu(args.cuda, L[i])

    #adj_orig = fc_adj[i]
    #adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    #adj_orig.eliminate_zeros()

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    #adj = adj_train

    # Some preprocessing
    #adj_norm.append(preprocess_graph(fc_adj[i]))
    #adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    #adj_label = torch.FloatTensor(adj_label.toarray())
    #adj_label[i] = sparse_to_tuple(sp.csr_matrix(adj_label[i] / 1.0))
    #walks = generate_walks(A_nx, args.numWalks_per_node, args.walk_length)


# A_nx = nx.read_edgelist(args.data_path,nodetype = int, create_using=nx.DiGraph())  # if the graph is with sparse edge format, each line one edge, then use this
#A_nx = nx.read_adjlist(args.data_path, nodetype=int)  #稀疏矩阵

#A = nx.to_scipy_sparse_matrix(A_nx)  # transfer to sparse matrix format
#L = csgraph.laplacian(A, normed=False)  # use csgraph package to calculate the laplacian
#L = np.array(L.toarray(), np.float32)
#L = Variable(torch.from_numpy(L))
#L = to_gpu(args.cuda, L)

# generate walk for each node with given walk_length
#walks = generate_walks(A_nx, args.numWalks_per_node, args.walk_length)
# save randomly generated walks to file ../tmp/train.txt
#np.savetxt('./tmp/train.txt', walks, delimiter=" ", fmt="%s")

# create corpus
#corpus = Corpus('./tmp/',
#                maxlen=args.maxlen)

# dumping vocabulary
#with open('./output/{}/vocab.json'.format(args.outf), 'w') as f:
#    json.dump(corpus.dictionary.word2idx, f)

# save arguments to args.json and logs.txt
#ntokens = len(corpus.dictionary.word2idx)
#print("Vocabulary Size: {}".format(ntokens))
#args.ntokens = ntokens
#with open('./output/{}/args.json'.format(args.outf), 'w') as f:
#    json.dump(vars(args), f)
#with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
#    f.write(str(vars(args)))
#    f.write("\n\n")

# preparing batches for training
#train_data = batchify(corpus.train, args.batch_size, shuffle=True)

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
feat_dim=62
in_size=8
autoencoder = GCNModelVAE(feat_dim, args.hidden1_dim, args.hidden2_dim, args.dropout)
#gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_gen = MLP_G(ninput=args.z_size, noutput=args.hidden2_dim, layers=args.arch_g)
#gan_gen = GraphTransformer_G(in_size=args.dim_hidden_1,
#                         nb_class=1,
#                         d_model=args.dim_hidden,
#                         dim_feedforward=2 * args.dim_hidden,
#                         dropout=args.dropout,
#                         nb_heads=args.nb_heads,
#                         nb_layers=args.nb_layers,
#                         lap_pos_enc=args.lappe,
#                         lap_pos_enc_dim=args.lap_dim)
gan_disc = MLP_D(ninput=args.hidden2_dim, noutput=2, layers=args.arch_d)
#gan_disc = GraphTransformer(in_size=in_size,
#                         nb_class=1,
#                         d_model=args.dim_hidden,
#                         dim_feedforward=2 * args.dim_hidden,
#                         dropout=args.dropout,
#                         nb_heads=args.nb_heads,
#                         nb_layers=args.nb_layers,
#                         lap_pos_enc=args.lappe,
#                         lap_pos_enc_dim=args.lap_dim)


#### optimizing AE, GAN-generator, GAN-discriminator
## SGD, learning rate should be larger, like 1, Adam's learning rate should be smaller, like 0.001
optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))
#### crossEntropy loss for discriminator
criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    criterion_ce = criterion_ce.cuda()


###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    with open('./output/{}/autoencoder_model.pt'.format(args.outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('./output/{}/gan_gen_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('./output/{}/gan_disc_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)



def unique(tensor1d):
    t, idx = np.unique(tensor1d.cpu().data.numpy(), return_index=True)
    return t, idx


def train_ae(x, adj_norm, adj_label, adj, total_loss_ae, start_time, i,sub):
    """
    Training LSTM AE
    :param batch: one batch of data
    :param total_loss_ae: accumulated loss for LSTM AE so far
    :param start_time: for timming
    :param i: current iteration ID
    :return: accumulated total loss of LSTM AE part so far, and start time for timing
    """
    autoencoder.train()
    autoencoder.zero_grad()

    #source, target, lengths = batch
    #source = to_gpu(args.cuda, Variable(source))
    #target = to_gpu(args.cuda, Variable(target))

    # Create sentence length mask over padding
    #mask = target.gt(0)
    #masked_target = target.masked_select(mask)
    # examples x ntokens
    #output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

    # output: batch x seq_len x ntokens
    #output = autoencoder(source, lengths, noise=True)
    #output, output_X = autoencoder(features[i], adj_norm[i])

    n_nodes=62
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    #norm = adj[i].shape[0] * adj[i].shape[0] / float((adj[i].shape[0] * adj[i].pos_weight -0.8shape[0] - adj[i].sum()) * 2)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    #recovered, mu, logvar = autoencoder(features[i], adj_norm[i])
    x=torch.from_numpy(x).to(torch.float32)
    adj_label = torch.from_numpy(adj_label).to(torch.float32)
    pos_weight=torch.tensor(pos_weight)
    #norm=torch.Tensor(norm).to(torch.float32)
    #recovered, mu, logvar = autoencoder(features[i], adj_norm[i])
    recovered,X_pred, mu, logvar,z = autoencoder(x, adj_norm)
    A=recovered.detach().numpy()
    X=X_pred.detach().numpy()
    np.savez('F:/FCSC_pytorch/ZDXX62pkl/ARVGA/' + '/fake_sMRI_2' + str(sub), fake_A=A, fake_X=X)
    loss = loss_function(preds=recovered, labels=adj_label,
                         mu=mu, logvar=logvar, n_nodes=n_nodes,
                         norm=norm, pos_weight=pos_weight)

    loss.backward()
    cur_loss = loss.item()

    #optimizer.step()

    # output_size: batch_size, maxlen, self.ntokens
    #flattened_output = output.view(-1, ntokens)

    #emb_py = embed_dic_for_loss(corpus, args.nhidden)
   # embT = torch.transpose(emb_py, 0, 1)
    #embT = torch.mm(embT, L)

    #adj_loss = torch.trace(torch.mm(embT, emb_py)) / ntokens

    #masked_output = \
    #    flattened_output.masked_select(output_mask).view(-1, ntokens)
    #loss = criterion_ce(masked_output / args.temp, masked_target)

    #loss += adj_loss

    #loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    # This is the version of Wasserstein GAN, which has gradient clipping
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None

    ######################## store log periodically ############################
    #if i % args.log_interval == 0 and i > 0:
        # accuracy
    #    probs = F.softmax(masked_output)
    #    max_vals, max_indices = torch.max(probs, 1)
    #    accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

    #    cur_loss = total_loss_ae[0] / args.log_interval
    #    elapsed = time.time() - start_time
    #    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
    #          'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
    #          .format(epoch, i, len(train_data),
    #                  elapsed * 1000 / args.log_interval,
    #                  cur_loss, math.exp(cur_loss), accuracy))

    #    with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
    #        f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
    #                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
    #                format(epoch, i, len(train_data),
    #                       elapsed * 1000 / args.log_interval,
    #                       cur_loss, math.exp(cur_loss), accuracy))

        #total_loss_ae = 0
        #start_time = time.time()

    #return total_loss_ae, start_time
    return total_loss_ae

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

def train_gan_g():
    """
    Training WGAN generator network
    :return: error of generator part
    """
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    mask_1 = torch.zeros((1, 64), dtype=bool)
   #fake_hidden = gan_gen(noise,mask_1)
    fake_hidden = gan_gen(noise)
    #errG = gan_disc(fake_hidden,mask_1)
    errG = gan_disc(fake_hidden)

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    # Gradient norm: regularize to be same
    # code_grad_gan * code_grad_ae / norm(code_grad_gan)
    if args.enc_grad_norm:
        gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
        normed_grad = grad * autoencoder.grad_norm / gan_norm
    else:
        normed_grad = grad

    # weight factor and sign flip
    normed_grad *= -math.fabs(args.gan_toenc)
    return normed_grad


def train_gan_d(x, adj_norm, adj_label, adj, total_loss_ae, start_time, i,sub):
    """
    Training WGAN discriminator
    :param batch: training batch data
    :return: discriminator part error
    """
    # clamp parameters to a cube
    # WGAN Weight clipping
    for p in gan_disc.parameters():
        p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    #for p in gan_disc2.parameters():
    #    p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    loss_func = nn.BCEWithLogitsLoss()

    #real_label = torch.tensor([1],dtype=torch.float32).cuda()
    real_label = torch.ones(62,2)
    fake_label = torch.zeros(62,2)

    n_nodes=62
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    #norm = adj[i].shape[0] * adj[i].shape[0] / float((adj[i].shape[0] * adj[i].pos_weight -0.8shape[0] - adj[i].sum()) * 2)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    #recovered, mu, logvar = autoencoder(features[i], adj_norm[i])
    x=torch.from_numpy(x).to(torch.float32)
    adj_label = torch.from_numpy(adj_label).to(torch.float32)
    pos_weight=torch.tensor(pos_weight)
    #norm=torch.Tensor(norm).to(torch.float32)
    #recovered, mu, logvar = autoencoder(features[i], adj_norm[i])
    recovered,X_pred, mu, logvar,z = autoencoder(x, adj_norm)
    A=recovered.detach().numpy()
    X=X_pred.detach().numpy()
    np.savez('F:/SC_pytorch/HCP/ARVGA' + '/fake_sMRI_' + str(label_n[sub]) + str(sub), fake_A=A, fake_X=X)
    loss = loss_function(preds=recovered, labels=adj_label,
                         mu=mu, logvar=logvar, n_nodes=n_nodes,
                         norm=norm, pos_weight=pos_weight)

    loss.backward()

    #loss_func = nn.BCEWithLogitsLoss()

    #real_label = torch.tensor([1],dtype=torch.float32).cuda()
    #real_label = torch.ones(62,2)
    #fake_label = torch.zeros(62,2)

    # batch_size x nhidden
    #feature = torch.from_numpy(feature).to(torch.float32)
    #z, X_pred, mu1, log1, mu2, log2 = autoencoder.encode(feature, adj)
    #print(z)
    #reconstructed_A = autoencoder.dc(z)
    #A=reconstructed_A.detach().numpy()
    #X=X_pred.detach().numpy()
    #np.savez('F:/SC_pytorch/ZDXX/classify/0' + '/fake_sMRI_'+str(label_n[sub])+str(label_n[sub]) + str(sub), fake_A=A, fake_X=X)

    #loss = F.mse_loss(torch.tensor(A), torch.tensor(adj))

    #kl_loss_1 = 0.5 / torch.tensor(A).size(0) * (
    #            1 + 2 * autoencoder.logstd_gat1 - autoencoder.mu_gat1 ** 2 - torch.exp(autoencoder.logstd_gat1) ** 2).sum(1).mean()

    #kl_loss_2 = 0.5 / torch.tensor(A).size(0) * (
    #            1 + 2 * autoencoder.logstd_gat2 - autoencoder.mu_gat2 ** 2 - torch.exp(autoencoder.logstd_gat2) ** 2).sum(1).mean()
    #loss2 -= kl_divergence
    #kl_loss=(kl_loss_1+kl_loss_2)/2

    #loss -= kl_loss
    #loss.backward()
    optimizer_ae.step()
    # loss / backprop
    optimizer_gan_d.zero_grad()

    errD_real = gan_disc(z).detach()
    #errD_real = torch.tensor([errD_real]).cuda()
    D_loss_real = loss_func(errD_real, real_label)
    #print(errD_real)
    gen_size=62
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(gen_size, args.z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)

    errD_fake = gan_disc(fake_hidden)
    D_loss_fake = loss_func(errD_fake, fake_label)
    #print(errD_fake)

    #D_loss = errD_real + errD_fake
    D_loss=D_loss_fake+D_loss_real
    D_loss.backward()
    optimizer_gan_d.step()
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_gan_g.zero_grad()
    G_fake = gan_disc(fake_hidden.detach())
    G_loss = loss_func(G_fake, real_label)
    #G_loss = -torch.mean(G_fake)
    G_loss.backward()
    optimizer_gan_g.step()
    errD = -(errD_real - errD_fake)
    print("Epoch:", '%04d' % (epoch),
          "loss=", "{:.5f}".format(loss),
          "G_loss=", "{:.5f}".format(G_loss),
          "D_loss=", "{:.5f}".format(D_loss))

    return errD, errD_real, errD_fake


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

for epoch in range(1, args.epochs + 1):
    # embed_afterLSTM(corpus, args.nhidden)

    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1
    for sub in range(len(label_n)):

     #while niter < adj_size:

        """ 
            Iteratively conduct autoencoder training, then GAN regularization,
            The GAN part includes discriminator and generator iteratively.
        """

        # train autoencoder ----------------------------
        #for i in range(args.niters_ae):
            #if niter == len(train_data):
        #    if niter == adj_size:
        #        break  # end of epoch
            #total_loss_ae = train_ae(features[i], adj_norm[i],fc_adj[i],adj_label[i],total_loss_ae, start_time, niter)
        #total_loss_ae = train_ae(features[sub], adj_norm[sub], fc_adj[sub], adj_label[sub], total_loss_ae, start_time,
        #                             niter,sub)
                #train_ae(train_data[niter], total_loss_ae, start_time, niter)
        #    niter += 1

        #print(niter_gan)

        # train gan ----------------------------------
        #for k in range(niter_gan):

            # train discriminator/critic
        #    for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                #errD, errD_real, errD_fake = \
                #    train_gan_d(train_data[random.randint(0, len(train_data) - 1)])
        #errD, errD_real, errD_fake = \
        #            train_gan_d(features[sub],fc_adj[sub])
                #errD, errD_real, errD_fake = \
                #    train_gan_d(features[i],fc_adj[i])
            # train generator
        #    for i in range(args.niters_gan_g):
        #errG = train_gan_g()

        errD, errD_real, errD_fake = \
        train_gan_d(features[i], adj_norm[i],fc_adj[i],adj_label[i],total_loss_ae, start_time, niter,sub)
        """
            The codes here are for logging running status, not actually conduct the algorithm logic
        """
#        niter_global += 1
#        if niter_global % 100 == 0:
#            print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
#                  'Loss_D_fake: %.8f) Loss_G: %.8f'
#                  % (epoch, args.epochs, niter, len(train_data),
#                     errD.data[0], errD_real.data[0],
#                     errD_fake.data[0], errG.data[0]))
#            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
#                f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
#                        'Loss_D_fake: %.8f) Loss_G: %.8f\n'
#                        % (epoch, args.epochs, niter, len(train_data),
#                           errD.data[0], errD_real.data[0],
#                           errD_fake.data[0], errG.data[0]))

            # exponentially decaying noise on autoencoder
#            autoencoder.noise_radius = \
#                autoencoder.noise_radius * args.noise_anneal

    # embed_corpus(corpus, args.emsize)
#    embed_afterLSTM(corpus, args.nhidden)
    # save_model()

    # shuffle between epochs
#    train_data = batchify(corpus.train, args.batch_size, shuffle=True)

###############################################################################
# visualization
###############################################################################

# input membership(label) of each node
#membership_path = "../data/membership.txt"

# use last iteration output embeddings as code
#embedding_path = "./output/example/embed_afterLSTM_{}.txt".format(args.epochs)

# viz_tsne(membership_path, embedding_path)
#viz(membership_path, embedding_path)
