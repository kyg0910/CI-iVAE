import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import random
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
import progressbar
import tqdm

import ci_ivae_src.model as MODEL
import ci_ivae_src.util as UTIL

def model(dim_x, dim_u,
          dim_z=16, prior_node_list=[128, 128],
          encoder_node_list=[4096, 4096],
          decoder_node_list=[4096, 4096],
          decoder_final_activation='sigmoid'):
    '''
    dim_z: dimension of representations
    prior_node_list: list of number of nodes in layers in label prior networks
    encoder_node_list: list of number of nodes in layers in encoder networks
    decoder_node_list: list of number of nodes in layers in decoder networks
    decoder_final_activation: the last activation layer in decoder. Please choose 'sigmoid' or 'None' 
    '''
    
    prior = MODEL.Prior_conti(dim_z, dim_u, prior_node_list)
    encoder = MODEL.Encoder(dim_x, dim_z, encoder_node_list)
    decoder = MODEL.Decoder(dim_z, dim_x, decoder_node_list,
                            final_activation=decoder_final_activation)
    return [prior, encoder, decoder]

def fit(model, x_train, u_train, x_val, u_val,
        num_epoch=100, batch_size=256, num_worker=32, seed=0,
        beta=0.01, Adam_beta1=0.5, Adam_beta2=0.999, weight_decay=5e-6,
        init_lr=5e-5, lr_milestones=[25, 50, 75], lr_gamma=0.5,
        dtype=torch.float32, M=50, alpha_step=0.025,
        fix_alpha=None, result_path=None):
    '''
    num_epoch: the number of epoch
    batch_size: the number of samples in each mini-batch
    num_worker: the number of CPU cores
    seed: the random seed number
    beta: the coefficient of KL-penalty term in ELBOs
    Adam_beta1: beta1 for Adam optimizer
    Adam_beta2: beta2 for Adam optimizer
    weight_decay: the coefficient of the half of L2 penalty term
    init_lr: the initial learning rate
    lr_milestones: the epochs to reduce the learning rate
    lr_gamma: the multiplier for each time learning rate is reduced
    dtype: the data type
    M: the number of MC samples to approximate skew KL-divergences
    alpha_step: the distance between each grid points in finding samplewise optimal alpha
    fix_alpha: If it is None, the objective function is supremum of ELBO(\alpha) over \alpha. If it is a real number in [0, 1], the objective function is the ELBO(\alpha). For example, when it is 0.0, the objective is the ELBO of iVAEs.
    result_path: the directory where results are saved
    '''
    
    # declare basic variables
    prior, encoder, decoder = model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beta_kl_post_prior = beta
    beta_kl_encoded_prior = beta
    Adam_betas = (Adam_beta1, Adam_beta2)
    if result_path is None:
        now = datetime.datetime.now()
        result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(result_path, exist_ok=True)
    
    # lines for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # convert data to tensors
    x_train = torch.tensor(x_train, dtype=dtype)
    x_val = torch.tensor(x_val, dtype=dtype)
    u_train = torch.tensor(u_train, dtype=dtype)
    u_val = torch.tensor(u_val, dtype=dtype)

    # define optimizers and schedulers
    enc_optimizer = torch.optim.Adam(encoder.parameters(),
                                     betas=Adam_betas,
                                     lr=init_lr,
                                     weight_decay=weight_decay)
    gen_optimizer = torch.optim.Adam(list(decoder.parameters())
                                     +list(prior.parameters()),
                                     betas=Adam_betas,
                                     lr=init_lr,
                                     weight_decay=weight_decay)

    enc_scheduler = torch.optim.lr_scheduler.MultiStepLR(enc_optimizer,
                                                         milestones=lr_milestones,
                                                         gamma=lr_gamma)
    gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                         milestones=lr_milestones,
                                                         gamma=lr_gamma)

    # define training log    
    loss_names = ['loss', 'recon_loss_post', 'kl_post_prior',
                  'recon_loss_encoded', 'kl_encoded_prior', 'l2_penalty']
    logs = {}
    for datasetname in ['train', 'val']:
        logs[datasetname] = {}
        for loss_name in loss_names:
            logs[datasetname][loss_name] = []
        del(loss_name)
    del(datasetname)
    summary_stats = []
    
    # define data loader
    dataloader = {}
    dataloader['train'] = DataLoader(TensorDataset(x_train, u_train),
                                     batch_size=batch_size, num_workers=num_worker,
                                     shuffle=True, drop_last=True)
    dataloader['val'] = DataLoader(TensorDataset(x_val, u_val),
                                   batch_size=batch_size, num_workers=num_worker,
                                   shuffle=True, drop_last=True)
    
    # training part
    mse_criterion = torch.nn.MSELoss()
    if device == 'cuda':
        prior.cuda()
        encoder.cuda()
        decoder.cuda()
        mse_criterion.cuda()
    for epoch in range(1, num_epoch+1):
        num_batch = 0
        for x_batch, u_batch in tqdm.tqdm(dataloader['train'],
                                         desc='[Epoch %d/%d] Training' % (epoch, num_epoch)):
            num_batch += 1
            if device == 'cuda':
                x_batch, u_batch = x_batch.cuda(), u_batch.cuda()
            x_batch += torch.randn_like(x_batch)*1e-2

            prior.train()
            encoder.train()
            decoder.train()

            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # forward step
            lam_mean, lam_log_var = prior(u_batch)
            z_mean, z_log_var = encoder(x_batch)
            post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
            post_sample = UTIL.sampling(post_mean, post_log_var)
            encoded_sample = UTIL.sampling(z_mean, z_log_var)

            epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M))
            if device == 'cuda':
                post_sample = post_sample.cuda()
                encoded_sample = encoded_sample.cuda()
                epsilon = epsilon.cuda()

            fire_rate_post, obs_log_var = decoder(post_sample)
            fire_rate_encoded, _ = decoder(encoded_sample)

            # compute objective function
            obs_loglik_post = -torch.mean((fire_rate_post - x_batch)**2, dim=1)
            obs_loglik_encoded = -torch.mean((fire_rate_encoded - x_batch)**2, dim=1)

            kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
            kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)

            elbo_post = obs_loglik_post - beta_kl_post_prior*kl_post_prior
            elbo_encoded = obs_loglik_encoded - beta_kl_encoded_prior*kl_encoded_prior

            z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
            z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
            z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

            post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
            post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
            post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon

            log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
            log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)

            if fix_alpha is not None:
                if fix_alpha == 0.0:
                    loss = torch.mean(-elbo_post)
                elif fix_alpha == 1.0:
                    loss = torch.mean(-elbo_encoded)
                else:
                    ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                    ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                    skew_kl_post = torch.log(1.0/(fix_alpha*ratio_z_over_post_with_post_sample+(1.0-fix_alpha)))
                    skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                    skew_kl_encoded = torch.log(1.0/(fix_alpha+(1.0-fix_alpha)*ratio_post_over_z_with_z_sample))
                    skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                    loss = -fix_alpha*elbo_encoded-(1.0-fix_alpha)*elbo_post+fix_alpha*skew_kl_encoded+(1.0-fix_alpha)*skew_kl_post
            else:
                alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                loss = torch.zeros((elbo_post.shape[0], len(alpha_list)))
                i = 0
                for alpha in alpha_list:
                    ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                    ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                    skew_kl_post = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                    skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                    skew_kl_encoded = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                    skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                    loss[:, i] = -alpha*elbo_encoded-(1.0-alpha)*elbo_post+alpha*skew_kl_encoded+(1.0-alpha)*skew_kl_post
                    i += 1
                del(alpha, i)
                loss, _ = torch.min(loss, dim = 1)
            loss = torch.mean(loss)

            # backward step
            loss.backward()

            enc_optimizer.step()
            gen_optimizer.step()
        del(x_batch, u_batch)
        
        prior.eval()
        encoder.eval()
        decoder.eval()
        for datasetname in ['train', 'val']:
            loss_cumsum, sample_size = 0.0, 0
            obs_loglik_post_cumsum, kl_post_prior_cumsum = 0.0, 0.0
            obs_loglik_encoded_cumsum, kl_encoded_prior_cumsum = 0.0, 0.0
            for x_batch, u_batch in tqdm.tqdm(dataloader[datasetname],
                                         desc='[Epoch %d/%d] Computing loss terms on %s' % (epoch, num_epoch, datasetname)):
                if device == 'cuda':
                    x_batch, u_batch = x_batch.cuda(), u_batch.cuda()

                # forward step
                lam_mean, lam_log_var = prior(u_batch)
                z_mean, z_log_var = encoder(x_batch)
                post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                post_sample = UTIL.sampling(post_mean, post_log_var)
                encoded_sample = UTIL.sampling(z_mean, z_log_var)

                epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M))
                if device == 'cuda':
                    post_sample = post_sample.cuda()
                    encoded_sample = encoded_sample.cuda()
                    epsilon = epsilon.cuda()

                fire_rate_post, obs_log_var = decoder(post_sample)
                fire_rate_encoded, _ = decoder(encoded_sample)

                # compute objective function
                obs_loglik_post = -torch.mean((fire_rate_post - x_batch)**2, dim=1)
                obs_loglik_encoded = -torch.mean((fire_rate_encoded - x_batch)**2, dim=1)

                kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)

                elbo_pi_vae = obs_loglik_post - beta_kl_post_prior*kl_post_prior
                elbo_vae = obs_loglik_encoded - beta_kl_encoded_prior*kl_encoded_prior

                z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon

                log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)

                if fix_alpha is not None:
                    if fix_alpha == 0.0:
                        loss = torch.mean(-elbo_post)
                    elif fix_alpha == 1.0:
                        loss = torch.mean(-elbo_encoded)
                    else:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_post = torch.log(1.0/(fix_alpha*ratio_z_over_post_with_post_sample+(1.0-fix_alpha)))
                        skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                        skew_kl_encoded = torch.log(1.0/(fix_alpha+(1.0-fix_alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                        loss = -fix_alpha*elbo_encoded-(1.0-fix_alpha)*elbo_post+fix_alpha*skew_kl_encoded+(1.0-fix_alpha)*skew_kl_post
                else:
                    alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                    loss = torch.zeros((elbo_post.shape[0], len(alpha_list)))
                    i = 0
                    for alpha in alpha_list:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_post = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                        skew_kl_encoded = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                        loss[:, i] = -alpha*elbo_encoded-(1.0-alpha)*elbo_post+alpha*skew_kl_encoded+(1.0-alpha)*skew_kl_post
                        i += 1
                    del(alpha, i)
                    loss, _ = torch.min(loss, dim = 1)
                loss = torch.mean(loss)
                
                loss_cumsum += loss.item()*np.shape(x_batch)[0]
                obs_loglik_post_cumsum += torch.mean(obs_loglik_post).item()*np.shape(x_batch)[0]
                kl_post_prior_cumsum += torch.mean(kl_post_prior).item()*np.shape(x_batch)[0]
                obs_loglik_encoded_cumsum += torch.mean(obs_loglik_encoded).item()*np.shape(x_batch)[0]
                kl_encoded_prior_cumsum += torch.mean(kl_encoded_prior).item()*np.shape(x_batch)[0]
                sample_size += np.shape(x_batch)[0]
            del(x_batch, u_batch)

            l2_penalty = 0.0
            for networks in [prior, encoder, decoder]:
                for name, m in networks.named_parameters():
                    if 'weight' in name:
                        l2_penalty += 0.5*torch.sum(m**2)
            logs[datasetname]['loss'].append(loss_cumsum/sample_size)
            logs[datasetname]['recon_loss_post'].append(-obs_loglik_post_cumsum/sample_size)
            logs[datasetname]['kl_post_prior'].append(kl_post_prior_cumsum/sample_size)
            logs[datasetname]['recon_loss_encoded'].append(-obs_loglik_encoded_cumsum/sample_size)
            logs[datasetname]['kl_encoded_prior'].append(kl_encoded_prior_cumsum/sample_size)
            logs[datasetname]['l2_penalty'].append(l2_penalty.item())
        
        # save loss curves
        linestyles = ['solid', 'dashed']
        i = 0
        for dataset_name in ['train', 'val']:
            plt.plot(logs[dataset_name]['loss'][:], linestyle=linestyles[i],
                     label=dataset_name)
            i += 1
        del(i)
        if epoch == 1:
            plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('%s/loss_curves.pdf' % (result_path), dpi=600)
        
        # update models and logs if the best validation loss is updated
        current_val_loss = logs['val']['loss'][-1]
        best_val_loss = current_val_loss if epoch == 1 else np.minimum(best_val_loss, current_val_loss)
        if best_val_loss == current_val_loss:
            # update model and logs
            best_val_epoch = epoch
            os.makedirs('%s/' % result_path, exist_ok=True)
            torch.save({'prior': prior,
                        'encoder': encoder,
                        'decoder': decoder,
                        'logs': logs,
                        'num_epoch': num_epoch,
                        'batch_size': batch_size,
                        'num_worker': num_worker,
                        'seed': seed,
                        'beta': beta,
                        'Adam_beta1': Adam_beta1,
                        'Adam_beta2': Adam_beta2,
                        'weight_decay': weight_decay,
                        'init_lr': init_lr,
                        'lr_milestones': lr_milestones,
                        'lr_gamma': lr_gamma,
                        'dtype': dtype,
                        'M': M,
                        'alpha_step': alpha_step,
                        'fix_alpha': fix_alpha,
                        'result_path': result_path},
                       '%s/model.pth' % result_path)
        if epoch == num_epoch:
            # update logs
            saved_model = torch.load('%s/model.pth' % result_path)
            saved_model['logs'] = logs
            torch.save(saved_model, '%s/model.pth' % result_path)
            del(saved_model)

        current_summary_stats_row = {}

        current_summary_stats_row['epoch'] = epoch
        current_summary_stats_row['best_val_epoch'] = best_val_epoch
        current_summary_stats_row['train_loss'] = logs['train']['loss'][-1]
        current_summary_stats_row['val_loss'] = logs['val']['loss'][-1]
        current_summary_stats_row['train_recon_loss_post'] = logs['train']['recon_loss_post'][-1]
        current_summary_stats_row['val_recon_loss_post'] = logs['val']['recon_loss_post'][-1]
        current_summary_stats_row['train_kl_post_prior'] = logs['train']['kl_post_prior'][-1]
        current_summary_stats_row['val_kl_post_prior'] = logs['val']['kl_post_prior'][-1]
        current_summary_stats_row['train_recon_loss_encoded'] = logs['train']['recon_loss_encoded'][-1]
        current_summary_stats_row['val_recon_loss_encoded'] = logs['val']['recon_loss_encoded'][-1]
        current_summary_stats_row['train_kl_encoded_prior'] = logs['train']['kl_encoded_prior'][-1]
        current_summary_stats_row['val_kl_encoded_prior'] = logs['val']['kl_encoded_prior'][-1]
        current_summary_stats_row['l2_penalty'] = logs['train']['l2_penalty'][-1]
        
        summary_stats.append(current_summary_stats_row)
        pd.DataFrame(summary_stats).to_csv('%s/summary_stats.csv' % result_path, index=False)
    del(epoch)    
    
    return None

def extract_feature(result_path, x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved_model = torch.load('%s/model.pth' % result_path)
    prior, encoder, decoder = saved_model['prior'], saved_model['encoder'], saved_model['decoder']
    prior.eval(); encoder.eval(); decoder.eval()
    
    if device == 'cuda':
        z_mean, z_log_var = encoder(x.cuda())
    elif device == 'cpu':
        z_mean, z_log_var = encoder(x)
    z_sample = UTIL.sampling(z_mean, z_log_var)
    return z_sample

def generate_z(result_path, u):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved_model = torch.load('%s/model.pth' % result_path)
    prior, encoder, decoder = saved_model['prior'], saved_model['encoder'], saved_model['decoder']
    prior.eval(); encoder.eval(); decoder.eval()
    
    u = u.cuda() if device == 'cuda' else u
    z_mean, z_log_var = prior(u.cuda())
    return z_mean, z_log_var
    
    
