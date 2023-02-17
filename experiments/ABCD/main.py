import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tensorboardX import SummaryWriter
import datetime
import os
import torch
import random
import yaml

import src.model as MODEL
import src.eval as EVAL
import src.util as UTIL
import src.data_preprocess as DATA_PREPROCESS
import src.train as TRAIN

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/real_data.yaml')
parser.add_argument('--dtype',
                    type=str,
                    default='float32')
parser.add_argument('--num_epoch',
                    type=int,
                    default=None)
parser.add_argument('--n_fold',
                    type=int,
                    default=None)
parser.add_argument('--seed_num_datasplit',
                    type=int,
                    default=None)
parser.add_argument('--seed_num_opt',
                    type=int,
                    default=None)
parser.add_argument('--gen_nodes',
                    type=int,
                    default=None)
parser.add_argument('--beta_kl_post_prior',
                    type=float,
                    default=None)
parser.add_argument('--beta_kl_encoded_prior',
                    type=float,
                    default=None)
parser.add_argument('--fix_alpha',
                    type=float,
                    default=None)
parser.add_argument('--n_blk',
                    type=int,
                    default=None)
parser.add_argument('--dim_z',
                    type=int,
                    default=None)

opts = parser.parse_args()

with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
if opts.dtype == 'float32':
    config['dtype'] = torch.float32
    dtype = config['dtype']
    
if opts.num_epoch is not None:
    config['num_epoch'] = opts.num_epoch
if opts.n_fold is not None:
    config['n_fold'] = opts.n_fold
if opts.seed_num_datasplit is not None:
    config['seed_num_datasplit'] = opts.seed_num_datasplit
if opts.seed_num_opt is not None:
    config['seed_num_opt'] = opts.seed_num_opt
if opts.gen_nodes is not None:
    config['gen_nodes'] = opts.gen_nodes    
if opts.beta_kl_post_prior is not None:
    config['beta_kl_post_prior'] = opts.beta_kl_post_prior
if opts.beta_kl_encoded_prior is not None:
    config['beta_kl_encoded_prior'] = opts.beta_kl_encoded_prior
if opts.fix_alpha is not None:
    config['fix_alpha'] = opts.fix_alpha  
if opts.n_blk is not None:
    config['n_blk'] = opts.n_blk 
if opts.dim_z is not None:
    config['dim_z'] = opts.dim_z  

dim_x = config['dim_x']
dim_z = config['dim_z']
num_pc = config['num_pc']
seed_num_datasplit = config['seed_num_datasplit']
seed_num_opt = config['seed_num_opt']
gen_nodes = config['gen_nodes']
n_blk = config['n_blk']
disc = config['disc']
recon_error = config['recon_error']
init_lr = config['init_lr']
weight_decay = config['weight_decay']
num_epoch = config['num_epoch']
batch_size = config['batch_size']
num_worker = config['num_worker']
beta_kl_post_prior = config['beta_kl_post_prior']
beta_kl_encoded_prior = config['beta_kl_encoded_prior']
u_names = config['u_names']
v_names = config['v_names']
dropna = config['dropna']
fix_alpha = config['fix_alpha']
Adam_beta1 = config['Adam_beta1']
Adam_beta2 = config['Adam_beta2']
prior_hidden_nodes = config['prior_hidden_nodes']
M = config['M']

if config['save_model_path'] is None:
    now = datetime.datetime.now()
    if config['fix_alpha'] is not None:
        config['save_model_path'] = ('results/fix_alpha=%.2f-seed_num_datasplit=%d-seed_num_opt=%d-recon_error=%r-gen_nodes=%d-dim_z=%d-M=%d-beta_kl_post_prior=%.6f-beta_kl_encoded_prior=%.6f-init_lr=%.5f-%d-%d-%d-%d-%d'
                              % (fix_alpha, seed_num_datasplit, seed_num_opt, recon_error, gen_nodes, dim_z, M, beta_kl_post_prior, beta_kl_encoded_prior, init_lr, now.month, now.day, now.hour, now.minute, now.second))
    else:
        config['save_model_path'] = ('results/cai_vae-seed_num_datasplit=%d-seed_num_opt=%d-recon_error=%r-gen_nodes=%d-dim_z=%d-M=%d-beta_kl_post_prior=%.6f-beta_kl_encoded_prior=%.6f-init_lr=%.5f-%d-%d-%d-%d-%d'
                              % (seed_num_datasplit, seed_num_opt, recon_error, gen_nodes, dim_z, M, beta_kl_post_prior, beta_kl_encoded_prior, init_lr, now.month, now.day, now.hour, now.minute, now.second))

if config['load_model_path'] is None:
    config['load_model_path'] = config['save_model_path']

print(config)

dim_u = len(u_names)
dim_v = len(v_names)

save_model_path = config['save_model_path']
load_model_path = config['load_model_path']
data_path = config['data_path']
n_fold = config['n_fold']

dataset_names = ['train', 'val', 'test']
    
random.seed(seed_num_datasplit)

df, x_true, u_true, loggers = DATA_PREPROCESS.preprocess(config)

n = x_true.shape[0]
idx = np.arange(n)
random.shuffle(idx)

split_idx = np.array_split(np.arange(n), n_fold)
rsquared_adjs_folds, pvalues_folds, explained_variance_ratios_folds = {}, {}, {}
for dataset_name in dataset_names:
    rsquared_adjs_folds[dataset_name] = {}
    pvalues_folds[dataset_name] = {}
    explained_variance_ratios_folds[dataset_name] = {}
    for latent_type in ['post', 'encoded']:
        rsquared_adjs_folds[dataset_name][latent_type] = np.zeros((n_fold, num_pc))
        pvalues_folds[dataset_name][latent_type] = np.zeros((n_fold, num_pc, dim_u + dim_v))
        explained_variance_ratios_folds[dataset_name][latent_type] = np.zeros((n_fold, dim_z))
    del(latent_type)
del(dataset_name)

for i in range(n_fold):
    print('[Val set: fold %d, Test set: fold %d]' % ((i-1) % n_fold, i % n_fold))
    val_idx, test_idx = split_idx[(i-1) % n_fold], split_idx[i % n_fold]
    train_idx = list(set(idx) - set(val_idx) - set(test_idx))
    TRAIN.train(x_true, u_true, train_idx, val_idx, test_idx, loggers, config, fold=i)
del(i)
