### CONFIGS ###
dataset = 'cora'
model = 'GAE'

# sc
# input_dim = 21

# fc
# input_dim = 230

# ASD
input_dim = 62

# multi site MDD
# input_dim = 170

hidden1_dim = 16
hidden2_dim = 8
use_feature = True
node_num = 62
batch_size = 1
kfold = 10

num_epoch = 200
learning_rate = 0.0001

clip_value = 0.01
critic_num = 5

cuda=True

#def mask_rate():
#    return None