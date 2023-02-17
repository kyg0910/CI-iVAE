import torch
import random
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from time import sleep
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from imblearn.over_sampling import RandomOverSampler, SMOTE

import src.data_preprocess as DATA_PREPROCESS
import src.model as MODEL
import src.util as UTIL

def _swap(x1, x2, swap_idx):
    x1[swap_idx] = x2[swap_idx]
    return x1

def train(x_true, u_true, train_idx, val_idx, test_idx, transgender_idx, loggers, config, fold):
    dtype = config['dtype']
    dim_z = config['dim_z']
    u_names = config['u_names']
    dropna = config['dropna']
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
    save_model_path = config['save_model_path']
    load_model_path = config['load_model_path']
    data_path = config['data_path']
    fix_alpha = config['fix_alpha']
    M = config['M']
    alpha_step = config['alpha_step']
    Adam_beta1 = config['Adam_beta1']
    Adam_beta2 = config['Adam_beta2']
    prior_hidden_nodes = config['prior_hidden_nodes']
    
    Adam_betas = (Adam_beta1, Adam_beta2)
    
    mse_criterion = torch.nn.MSELoss()
    
    random.seed(seed_num_opt)
    
    dim_x = np.shape(x_true)[1]
    
    swap_rate = 0.05
    
    dataset_idx = {}
    dataset_idx['train'], dataset_idx['val'], dataset_idx['test'] = train_idx, val_idx, test_idx
    
    x_train, u_train = x_true[dataset_idx['train']], u_true[dataset_idx['train']]
    x_val, u_val = x_true[dataset_idx['val']], u_true[dataset_idx['val']]
    x_test = x_true[list(set(dataset_idx['test'])-set(transgender_idx))]
    u_test = u_true[list(set(dataset_idx['test'])-set(transgender_idx))]
    
    dim_u = np.shape(u_train)[1]
    dataset_names = ['train', 'val', 'test']
    latent_name_list = ["Posterior [q(z|x,u)]", "Encoder [q_{phi}(z|x)]", "Prior [p_{T, lambda}(z|u)]"]
    
    if disc:
        prior = MODEL.Prior_disc(dim_z, dim_u)
    else:
        prior = MODEL.Prior_conti(dim_z, dim_u, prior_hidden_nodes)
        
    prior_optimizer = torch.optim.Adam(prior.parameters(), betas=Adam_betas, lr=init_lr, weight_decay=weight_decay)
    encoder = MODEL.Encoder(dim_x, gen_nodes, dim_z)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), betas=Adam_betas, lr=init_lr, weight_decay=weight_decay)
    if n_blk is not None:
        decoder = MODEL.Decoder_nflow(dim_z, n_blk, dim_x, gen_nodes)
    else:
        decoder = MODEL.Decoder(dim_z, dim_x, gen_nodes)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), betas=Adam_betas, lr=init_lr, weight_decay=weight_decay)
    
    prior_scheduler = torch.optim.lr_scheduler.MultiStepLR(prior_optimizer, milestones=[25, 50, 100], gamma=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[25, 50, 100], gamma=0.5)
    decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_optimizer, milestones=[25, 50, 100], gamma=0.5)

    mse_criterion.cuda()
    
    prior.cuda()
    encoder.cuda()
    decoder.cuda()

    loss_names = ['elbo_pi_vae', 'elbo_vae', 'obs_loglik_post', 'obs_loglik_encoded', 'kl_post_prior', 'kl_encoded_prior', 'loss']
    
    logs = {}
    for dataset_name in dataset_names:
        logs[dataset_name] = {}
        for loss_name in loss_names:
            logs[dataset_name][loss_name] = []
        del(loss_name)
    del(dataset_name)

    for epoch in range(1, num_epoch+1):
        # Ref: https://arxiv.org/abs/1511.06349
        kl_annealing = epoch/(0.1*num_epoch) if epoch < (0.1*num_epoch) else 1.0
        
        ros = RandomOverSampler(sampling_strategy='minority', random_state=epoch)
        smote = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=epoch)
        
        x_oversampled_train, u_oversampled_train = x_train, u_train
        x_oversampled_val, u_oversampled_val = x_val, u_val
        x_oversampled_test, u_oversampled_test = x_test, u_test
        #
        if epoch == 1:
            x_oversampled_train = torch.tensor(x_oversampled_train, dtype=dtype)
            u_oversampled_train = torch.tensor(u_oversampled_train, dtype=dtype)
            x_oversampled_val = torch.tensor(x_oversampled_val, dtype=dtype)
            u_oversampled_val = torch.tensor(u_oversampled_val, dtype=dtype)
            x_oversampled_test = torch.tensor(x_oversampled_test, dtype=dtype)
            u_oversampled_test = torch.tensor(u_oversampled_test, dtype=dtype)

            x_train, u_train = torch.tensor(x_train, dtype=dtype), torch.tensor(u_train, dtype=dtype)
            x_val, u_val = torch.tensor(x_val, dtype=dtype), torch.tensor(u_val, dtype=dtype)
            x_test, u_test = torch.tensor(x_test, dtype=dtype), torch.tensor(u_test, dtype=dtype)
        else:
            x_oversampled_train = x_oversampled_train.clone().detach()
            u_oversampled_train = u_oversampled_train.clone().detach()
            x_oversampled_val = x_oversampled_val.clone().detach()
            u_oversampled_val = u_oversampled_val.clone().detach()
            x_oversampled_test = x_oversampled_test.clone().detach()
            u_oversampled_test = u_oversampled_test.clone().detach()

            x_train, u_train = x_train.clone().detach(), u_train.clone().detach()
            x_val, u_val = x_val.clone().detach(), u_val.clone().detach()
            x_test, u_test = x_test.clone().detach(), u_test.clone().detach()
        #
        train_dataset = TensorDataset(x_oversampled_train, u_oversampled_train)
        val_dataset = TensorDataset(x_oversampled_val, u_oversampled_val)
        test_dataset = TensorDataset(x_oversampled_test, u_oversampled_test)
        
        dataset_loaders = {}
        dataset_loaders['train'] = DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_worker,
                                              shuffle=True,
                                              drop_last=True)
        dataset_loaders['val'] = DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_worker,
                                            shuffle=True,
                                            drop_last=True)
        dataset_loaders['test'] = DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_worker,
                                             shuffle=True,
                                             drop_last=True)
        
        prior.train()
        encoder.train()
        decoder.train()
        
        num_pos = np.sum(u_oversampled_train.numpy())
        num_neg = np.sum(1.0-u_oversampled_train.numpy())
        with tqdm(dataset_loaders['train'], unit="batch") as tepoch:
            for x_batch, u_batch in tepoch:
                prior_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                
                tepoch.set_description(f"Epoch {epoch}")                
                x_batch += torch.randn_like(x_batch)*1e-2
                
                # swap noise
                row_idx = np.random.choice(np.shape(x_oversampled_train)[0], batch_size)
                num_swap = int(dim_x*swap_rate)
                swap_idx = np.array(list(map(lambda i:
                                             np.random.choice(dim_x, num_swap, replace=False), range(batch_size))))
                x_batch_oversampled = x_oversampled_train[row_idx]
                x_batch = torch.reshape(torch.cat(list(map(lambda i: _swap(x_batch[i], x_batch_oversampled[i],
                                                                           swap_idx[i]),
                                                           range(batch_size)))),
                                        (batch_size, -1))
                x_batch, u_batch = x_batch.cuda(), u_batch.cuda()
                
                prior.train()
                prior.zero_grad()
                encoder.train()
                encoder.zero_grad()
                decoder.train()
                decoder.zero_grad()

                # forward step
                lam_mean, lam_log_var = prior(u_batch)
                z_mean, z_log_var = encoder(x_batch)
                post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                post_sample = UTIL.sampling(post_mean, post_log_var)
                encoded_sample = UTIL.sampling(z_mean, z_log_var)
                
                post_sample = post_sample.cuda()
                encoded_sample = encoded_sample.cuda()
                
                fire_rate_post, obs_log_var = decoder(post_sample)
                fire_rate_encoded, _ = decoder(encoded_sample)
                
                if recon_error:
                    #obs_loglik_post = -torch.mean(torch.abs(fire_rate_post - x_batch), dim=1)
                    #obs_loglik_encoded = -torch.mean(torch.abs(fire_rate_encoded - x_batch), dim=1)
                    obs_loglik_post = -torch.mean((fire_rate_post - x_batch)**2, dim=1)
                    obs_loglik_encoded = -torch.mean((fire_rate_encoded - x_batch)**2, dim=1)
                else:
                    obs_loglik_post = -torch.sum(torch.square(fire_rate_post - x_batch)/(2*torch.exp(obs_log_var))+(obs_log_var/2), dim=1)
                    obs_loglik_encoded = -torch.sum(torch.square(fire_rate_encoded - x_batch)/(2*torch.exp(obs_log_var))+(obs_log_var/2), dim=1)                        

                kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
                
                elbo_pi_vae = obs_loglik_post - kl_annealing*beta_kl_post_prior*kl_post_prior
                elbo_vae = obs_loglik_encoded - kl_annealing*beta_kl_encoded_prior*kl_encoded_prior

                epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).cuda()
                z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).cuda()
                post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon

                log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                
                if fix_alpha is not None:
                    if fix_alpha == 0.0:
                        loss = torch.mean(-(1.0-fix_alpha)*elbo_pi_vae)
                    elif fix_alpha == 1.0:
                        loss = torch.mean(-fix_alpha*elbo_vae)
                    else:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_pi_vae = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_pi_vae = torch.abs(torch.mean(skew_kl_pi_vae, dim=-1))
                        skew_kl_vae = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_vae = torch.abs(torch.mean(skew_kl_vae, dim=-1))
                        kl_loss = fix_alpha*skew_kl_vae+(1.0-fix_alpha)*skew_kl_pi_vae
                        loss = -fix_alpha*elbo_vae-(1.0-fix_alpha)*elbo_pi_vae+kl_loss
                        loss = torch.mean(loss)
                else:
                    alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                    loss = torch.zeros((elbo_pi_vae.shape[0], len(alpha_list)))
                    i = 0
                    for alpha in alpha_list:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_pi_vae = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_pi_vae = torch.abs(torch.mean(skew_kl_pi_vae, dim=-1))
                        skew_kl_vae = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_vae = torch.abs(torch.mean(skew_kl_vae, dim=-1))
                        loss[:, i] = -alpha*elbo_vae-(1.0-alpha)*elbo_pi_vae+alpha*skew_kl_vae+(1.0-alpha)*skew_kl_pi_vae
                        i += 1
                    del(i)
                    loss, _ = torch.min(loss, dim = 1)
                    loss = torch.mean(loss)

                # backward step
                loss.backward()
                
                prior_optimizer.step()
                encoder_optimizer.step()
                decoder_optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)
            del(x_batch, u_batch)

            prior.eval()
            encoder.eval()
            decoder.eval()
            for dataset_name in dataset_names:
                elbo_pi_vae_cumsum, elbo_vae_cumsum, obs_loglik_post_cumsum, obs_loglik_encoded_cumsum, kl_post_prior_cumsum, kl_encoded_prior_cumsum, loss_cumsum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                sample_size = 0
                
                for x_batch, u_batch in dataset_loaders[dataset_name]:
                    # swap noise
                    if dataset_name == 'train':
                        row_idx = np.random.choice(np.shape(x_oversampled_train)[0], batch_size)
                        x_batch_oversampled = x_oversampled_train[row_idx]
                    elif dataset_name == 'val':
                        row_idx = np.random.choice(np.shape(x_oversampled_val)[0], batch_size)
                        x_batch_oversampled = x_oversampled_val[row_idx]
                    elif dataset_name == 'test':
                        row_idx = np.random.choice(np.shape(x_oversampled_test)[0], batch_size)
                        x_batch_oversampled = x_oversampled_test[row_idx]
                    
                    num_swap = int(dim_x*swap_rate)
                    swap_idx = np.array(list(map(lambda i:
                                                 np.random.choice(dim_x, num_swap, replace=False), range(batch_size))))
                    if dataset_name == 'train':
                        x_batch = torch.reshape(torch.cat(list(map(lambda i: _swap(x_batch[i], x_batch_oversampled[i],
                                                                                   swap_idx[i]),
                                                                   range(batch_size)))),
                                                (batch_size, -1))
                    x_batch, u_batch = x_batch.cuda(), u_batch.cuda()

                    # forward step
                    lam_mean, lam_log_var = prior(u_batch)
                    z_mean, z_log_var = encoder(x_batch)
                    post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                    post_sample = UTIL.sampling(post_mean, post_log_var)
                    encoded_sample = UTIL.sampling(z_mean, z_log_var)

                    post_sample = post_sample.cuda()
                    encoded_sample = encoded_sample.cuda()
                    
                    fire_rate_post, obs_log_var = decoder(post_sample)
                    fire_rate_encoded, _ = decoder(encoded_sample)
                    
                    if recon_error:
                        obs_loglik_post = -torch.mean((fire_rate_post - x_batch)**2, dim=1)
                        obs_loglik_encoded = -torch.mean(torch.square(fire_rate_encoded - x_batch), dim=1)
                    else:
                        obs_loglik_post = -torch.sum(torch.square(fire_rate_post - x_batch)/(2*torch.exp(obs_log_var))+(obs_log_var/2), dim=1)
                        obs_loglik_encoded = -torch.sum(torch.square(fire_rate_encoded - x_batch)/(2*torch.exp(obs_log_var))+(obs_log_var/2), dim=1)

                    kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                    kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)

                    elbo_pi_vae = obs_loglik_post - beta_kl_post_prior * kl_post_prior
                    elbo_vae = obs_loglik_encoded - beta_kl_encoded_prior * kl_encoded_prior

                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).cuda()
                    z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                    z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                    z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).cuda()
                    post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                    post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                    post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon
                    
                    x_batch = x_batch.detach().cpu().numpy()
                    u_batch = u_batch.detach().cpu().numpy()
                    lam_mean = lam_mean.detach().cpu().numpy()
                    lam_log_var = lam_log_var.detach().cpu().numpy()
                    z_mean = z_mean.detach().cpu().numpy()
                    z_log_var = z_log_var.detach().cpu().numpy()
                    post_mean = post_mean.detach().cpu().numpy()
                    post_log_var = post_log_var.detach().cpu().numpy()
                    post_sample = post_sample.detach().cpu().numpy()
                    encoded_sample = encoded_sample.detach().cpu().numpy()
                    fire_rate_post = fire_rate_post.detach().cpu().numpy()
                    obs_log_var = obs_log_var.detach().cpu().numpy()
                    fire_rate_encoded = fire_rate_encoded.detach().cpu().numpy()
                    obs_loglik_post = obs_loglik_post.detach().cpu().numpy()
                    obs_loglik_encoded = obs_loglik_encoded.detach().cpu().numpy()
                    epsilon = epsilon.detach().cpu().numpy()
                    z_mean_tiled = z_mean_tiled.detach().cpu().numpy()
                    z_log_var_tiled = z_log_var_tiled.detach().cpu().numpy()
                    z_sample_tiled = z_sample_tiled.detach().cpu().numpy()
                    post_mean_tiled = post_mean_tiled.detach().cpu().numpy()
                    post_log_var_tiled = post_log_var_tiled.detach().cpu().numpy()
                    post_sample_tiled = post_sample_tiled.detach().cpu().numpy()
                    elbo_pi_vae = elbo_pi_vae.detach().cpu().numpy()
                    elbo_vae = elbo_vae.detach().cpu().numpy()
                    kl_post_prior = kl_post_prior.detach().cpu().numpy()
                    kl_encoded_prior = kl_encoded_prior.detach().cpu().numpy()
                    
                    log_z_density_with_post_sample = -np.sum((post_sample_tiled - z_mean_tiled)**2/(2*np.exp(z_log_var_tiled))+(z_log_var_tiled/2), axis=1)
                    log_post_density_with_post_sample = -np.sum((post_sample_tiled - post_mean_tiled)**2/(2*np.exp(post_log_var_tiled))+(post_log_var_tiled/2), axis=1)
                    log_z_density_with_z_sample = -np.sum((z_sample_tiled - z_mean_tiled)**2/(2*np.exp(z_log_var_tiled))+(z_log_var_tiled/2), axis=1)
                    log_post_density_with_z_sample = -np.sum((z_sample_tiled - post_mean_tiled)**2/(2*np.exp(post_log_var_tiled))+(post_log_var_tiled/2), axis=1)

                    if fix_alpha is not None:
                        if fix_alpha == 0.0:
                            loss = np.mean(-(1.0-fix_alpha)*elbo_pi_vae)
                        elif fix_alpha == 1.0:
                            loss = np.mean(-fix_alpha*elbo_vae)
                        else:
                            ratio_z_over_post_with_post_sample = np.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                            ratio_post_over_z_with_z_sample = np.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                            skew_kl_pi_vae = np.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                            skew_kl_pi_vae = np.abs(np.mean(skew_kl_pi_vae, axis=-1))
                            skew_kl_vae = np.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                            skew_kl_vae = np.abs(np.mean(skew_kl_vae, axis=-1))
                            kl_loss = fix_alpha*skew_kl_vae+(1.0-fix_alpha)*skew_kl_pi_vae
                            loss = -fix_alpha*elbo_vae-(1.0-fix_alpha)*elbo_pi_vae+kl_loss
                            loss = np.mean(loss)
                    else:
                        alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                        loss = np.zeros((elbo_pi_vae.shape[0], len(alpha_list)))
                        i = 0
                        for alpha in alpha_list:
                            ratio_z_over_post_with_post_sample = np.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                            ratio_post_over_z_with_z_sample = np.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                            skew_kl_pi_vae = np.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                            skew_kl_pi_vae = np.abs(np.mean(skew_kl_pi_vae, axis=-1))
                            skew_kl_vae = np.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                            skew_kl_vae = np.abs(np.mean(skew_kl_vae, axis=-1))
                            loss[:, i] = -alpha*elbo_vae-(1.0-alpha)*elbo_pi_vae+alpha*skew_kl_vae+(1.0-alpha)*skew_kl_pi_vae
                            i += 1
                        del(i)
                        loss = np.min(loss, axis = 1)
                        loss = np.mean(loss)
                    elbo_pi_vae_cumsum += np.mean(elbo_pi_vae)*np.shape(x_batch)[0]
                    elbo_vae_cumsum += np.mean(elbo_vae)*np.shape(x_batch)[0]
                    obs_loglik_post_cumsum += np.mean(obs_loglik_post)*np.shape(x_batch)[0]
                    obs_loglik_encoded_cumsum += np.mean(obs_loglik_encoded)*np.shape(x_batch)[0]
                    kl_post_prior_cumsum += np.mean(kl_post_prior)*np.shape(x_batch)[0]
                    kl_encoded_prior_cumsum += np.mean(kl_encoded_prior)*np.shape(x_batch)[0]
                    loss_cumsum += loss*np.shape(x_batch)[0]
                    sample_size += np.shape(x_batch)[0]
                
                l2_regularization = 0
                for network_parameters in [prior.parameters(), 
                                          encoder.parameters(),
                                          decoder.parameters()]:
                    for p in network_parameters:
                        l2_regularization += (weight_decay/2.0)*((p**2).sum())
                
                logs[dataset_name]['elbo_pi_vae'].append(elbo_pi_vae_cumsum.item()/sample_size)
                logs[dataset_name]['elbo_vae'].append(elbo_vae_cumsum.item()/sample_size)
                logs[dataset_name]['obs_loglik_post'].append(obs_loglik_post_cumsum.item()/sample_size)
                logs[dataset_name]['obs_loglik_encoded'].append(obs_loglik_encoded_cumsum.item()/sample_size)
                logs[dataset_name]['kl_post_prior'].append(kl_post_prior_cumsum.item()/sample_size)
                logs[dataset_name]['kl_encoded_prior'].append(kl_encoded_prior_cumsum.item()/sample_size)
                logs[dataset_name]['loss'].append(loss_cumsum.item()/sample_size + l2_regularization.item())
                
        for loss_name in loss_names:
            loggers[fold][dataset_name].add_scalar(loss_name, logs[dataset_name][loss_name][-1], epoch+1)
        del(loss_name)
        
        current_val_loss = logs['val']['loss'][-1]
        best_val_loss = current_val_loss if epoch == 1 else np.minimum(best_val_loss, current_val_loss)
        
        print('train_obs_loglik_encoded: ', logs['train']['obs_loglik_encoded'][-1])
        print('train_kl_post_prior: ', logs['train']['kl_post_prior'][-1])
        print('train_kl_encoded_prior: ', logs['train']['kl_encoded_prior'][-1])
        print('train_loss: ', logs['train']['loss'][-1])
        print('val_obs_loglik_encoded: ', logs['val']['obs_loglik_encoded'][-1])
        print('val_kl_post_prior: ', logs['val']['kl_post_prior'][-1])
        print('val_kl_encoded_prior: ', logs['val']['kl_encoded_prior'][-1])
        print('val_loss: ', logs['val']['loss'][-1])
        print('test_obs_loglik_encoded: ', logs['test']['obs_loglik_encoded'][-1])
        print('test_kl_post_prior: ', logs['test']['kl_post_prior'][-1])
        print('test_kl_encoded_prior: ', logs['test']['kl_encoded_prior'][-1])
        print('test_loss: ', logs['test']['loss'][-1])
        
        if best_val_loss == current_val_loss:
            # update model and logs
            print('best_val_loss: ', best_val_loss)
            os.makedirs('%s/%d/' % (save_model_path, fold), exist_ok=True)
            torch.save({
                'prior': prior,
                'encoder': encoder,
                'decoder': decoder,
                'logs': logs,
                'dataset_idx': dataset_idx,
                'config': config},
                '%s/%d/model.pth' % (save_model_path, fold))
        if epoch == num_epoch:
            del(prior, encoder, decoder, dataset_idx, config)
            # update logs
            saved_model = torch.load('%s/%d/model.pth' % (save_model_path, fold))
            saved_model['logs'] = logs
            torch.save(saved_model, '%s/%d/model.pth' % (save_model_path, fold))
            del(saved_model)
            
        prior_scheduler.step()
        encoder_scheduler.step()
        decoder_scheduler.step()
    del(epoch)
    
    return None
    
    
