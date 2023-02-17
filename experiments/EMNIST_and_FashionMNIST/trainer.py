import torch
import torch.nn as nn
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from collections import OrderedDict
from scipy import linalg
from scipy.stats import chi2, mode
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataloader, make_dataloader_emnist, make_dataloader_fashionmnist
from plot import plot_samples, plot_spectrum, plot_variation_along_dims, plot_t_sne

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import model

mse_criterion = torch.nn.MSELoss()
bce_criterion = torch.nn.BCELoss(reduction='none')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    mse_criterion.cuda()
    bce_criterion.cuda()

class GIN(nn.Module):
    def __init__(self, dataset, n_epochs, epochs_per_line, lr, lr_schedule, batch_size, incompressible_flow, empirical_vars, data_root_dir='./', n_classes=None, n_data_points=None, init_identity=True, seed=0):
        super().__init__()
        
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.epochs_per_line = epochs_per_line
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.incompressible_flow = bool(incompressible_flow)
        self.empirical_vars = bool(empirical_vars)
        self.init_identity = bool(init_identity)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = str(int(time()))
        self.seed = seed
        
        if self.dataset in ['EMNIST', 'FashionMNIST']:
            if not init_identity:
                raise RuntimeError('init_identity=False not implemented for %s experiments' % self.dataset)
            self.save_dir = os.path.join('./results/GIN/%s/seed=%d' % (self.dataset, self.seed), self.timestamp)
            self.data_root_dir = data_root_dir
            self.net = construct_net_gin(coupling_block='gin' if self.incompressible_flow else 'glow')
            if self.dataset == 'EMNIST':
                self.n_classes = 10
                self.n_dims = 28*28
                self.train_loader, self.val_loader = make_dataloader_emnist(batch_size=self.batch_size, train=True, root_dir=self.data_root_dir)
                self.test_loader  = make_dataloader_emnist(batch_size=self.batch_size, train=False, root_dir=self.data_root_dir)
            elif self.dataset == 'FashionMNIST':
                self.n_classes = 10
                self.n_dims = 28*28
                self.train_loader, self.val_loader = make_dataloader_fashionmnist(batch_size=self.batch_size, train=True, root_dir=self.data_root_dir)
                self.test_loader  = make_dataloader_fashionmnist(batch_size=self.batch_size, train=False, root_dir=self.data_root_dir)
            self.dim_u = self.n_classes
            print('number of parameters in networks:', count_parameters(self.net))
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
        
        if not empirical_vars:
            self.mu = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
            self.log_sig = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
            # initialize these parameters to reasonable values
            self.set_mu_sig(init=True)
            
        self.to(self.device)
            
    def forward(self, x, rev=False):
        x, logdet_J  = self.net(x, rev=rev)
        return x, logdet_J 
    
    def train_model(self):
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
            f.write(f'incompressible_flow {self.incompressible_flow}\n')
            f.write(f'empirical_vars {self.empirical_vars}\n')
            f.write(f'init_identity {self.init_identity}\n')
        os.makedirs(os.path.join(self.save_dir, 'model_save'))
        os.makedirs(os.path.join(self.save_dir, 'score_save'))
        os.makedirs(os.path.join(self.save_dir, 'figures'))
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.train()
        self.to(self.device)
        print('  time     epoch    iteration         loss       last checkpoint')
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.best_val_epoch = 1
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_schedule)
        losses = []
        t0 = time()
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target==i).item() for i in range(self.n_classes)]) < 2:
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                z, logdet_J = self.net(data)          # latent space variable
                if self.empirical_vars:
                    # we only need to calculate the std
                    sig = torch.stack([z[target==i].std(0, unbiased=False) for i in range(self.n_classes)])
                    # negative log-likelihood for gaussian in latent space
                    loss = 0.5 + sig[target].log().mean(1) + 0.5*np.log(2*np.pi)
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                loss -= logdet_J / self.n_dims
                loss = loss.mean()
                self.print_loss(loss.item(), batch_idx, epoch, t0)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
            
            if (epoch+1)%self.epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, epoch, t0, new_line=True)
                losses = []
            sched.step()
            
            val_loss, n_val = 0, 0
            for batch_idx, (data, target) in enumerate(self.val_loader):
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target==i).item() for i in range(self.n_classes)]) < 2:
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                z, logdet_J = self.net(data)          # latent space variable
                if self.empirical_vars:
                    # we only need to calculate the std
                    sig = torch.stack([z[target==i].std(0, unbiased=False) for i in range(self.n_classes)])
                    # negative log-likelihood for gaussian in latent space
                    loss = 0.5 + sig[target].log().mean(1) + 0.5*np.log(2*np.pi)
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                loss -= logdet_J / self.n_dims
                loss = loss.sum()
                val_loss += loss.item()
                n_val += np.shape(data)[0]
            val_loss /= n_val
            if epoch == 0:
                best_val_loss = val_loss
            else:
                best_val_loss = np.minimum(best_val_loss, val_loss)
                
            if best_val_loss == val_loss:
                print('Best validation loss occurs. Save current model.')
                self.best_val_epoch = epoch + 1
                self.save(os.path.join(self.save_dir, 'model_save', f'{epoch+1:03d}.pt'))
                
            if (epoch+1) == self.n_epochs:
                self.load(os.path.join(self.save_dir, 'model_save', f'{self.best_val_epoch:03d}.pt'))
                self.make_plots()
                score_df = {}
                score_df['seed'] = self.seed
                score_df['val_loss'] = best_val_loss
                score_df['ssw_over_sst'] = self.ssw_over_sst
                score_df = pd.DataFrame(score_df, index=[0])
                score_df.to_csv(os.path.join(self.save_dir, 'score_save', f'{epoch+1:03d}.csv'), index=False)
    
    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = self.best_val_epoch
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')
    
    def save(self, fname):
        state_dict = OrderedDict((k,v) for k,v in self.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)
    
    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])
    
    def make_plots(self):
        if self.dataset in ['EMNIST', 'FashionMNIST']:
            os.makedirs(os.path.join(self.save_dir, 'figures', f'epoch_{self.epoch+1:03d}'))
            self.set_mu_sig()
            sig_rms = np.sqrt(np.mean((self.sig**2).detach().cpu().numpy(), axis=0))
            plot_samples(self, n_rows=20, dataset=self.dataset)
            plot_spectrum(self, sig_rms, dims_to_plot=self.n_dims, dataset=self.dataset)
            plot_spectrum(self, sig_rms, dims_to_plot=64, dataset=self.dataset)
            n_dims_to_plot = 40
            top_sig_dims = np.flip(np.argsort(sig_rms))
            dims_to_plot = top_sig_dims[:n_dims_to_plot]
            plot_variation_along_dims(self, dims_to_plot)
            plot_t_sne(self, dataset=self.dataset)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
    
    def set_mu_sig(self, init=False, n_batches=40):
        examples = iter(self.test_loader)
        n_batches = min(n_batches, len(examples))
        self.lam_mean, self.z_mean, self.target = [], [], []
        target = []
        for _ in range(n_batches):
            data, targ = next(examples)
            data += torch.randn_like(data)*1e-2
            self.eval()
            self.z_mean.append((self(data.to(self.device))[0]).detach().cpu().numpy())
            self.target.append(targ)
        
        self.z_mean = np.concatenate(self.z_mean, 0)
        self.target = np.concatenate(self.target, 0)
        
        self.mu = torch.tensor([self.z_mean[self.target == i].mean(0) for i in range(10)]).to(self.device)
        self.sig = torch.tensor([self.z_mean[self.target == i].std(0) for i in range(10)]).to(self.device)  
        
        self.lam_mean = self.mu[self.target].detach().cpu().numpy()
        
        sst = np.sum((self.z_mean - np.mean(self.z_mean, axis=0))**2)
        ssw = np.sum([np.sum((self.z_mean[self.target==i]-np.mean(self.z_mean[self.target==i], axis=0))**2) for i in range(len(np.unique(self.target)))])
        print('ssw/sst: %.3f' % (ssw/sst))
        self.ssw_over_sst = ssw/sst
        
    def calculate_knn_accuracy(self, k=5, test=True):
        model_dir = './results/GIN/%s/seed=%d' % (self.dataset, self.seed)
        model_dir = os.path.join(model_dir, os.listdir(model_dir)[0], 'model_save')
        model_dir = os.path.join(model_dir, np.sort(os.listdir(model_dir))[-1])
        self.load(model_dir)
        
        latent_train = []; latent_test = []
        target_train = []; target_test = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = nn.functional.one_hot(target, num_classes=self.dim_u)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)

            z_mean, _ = self(data.to(self.device))
            latent_train.append(z_mean.cpu().detach().numpy())
            target_train.append(target.cpu().numpy())
        latent_train = np.concatenate(latent_train)
        target_train = np.concatenate(target_train)
            
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = nn.functional.one_hot(target, num_classes=self.dim_u)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)

            z_mean, _ = self(data.to(self.device))
            latent_test.append(z_mean.cpu().detach().numpy())
            target_test.append(target.cpu().numpy())
        latent_test = np.concatenate(latent_test)
        target_test = np.concatenate(target_test)
        
        if test:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(latent_train, target_train)
            
            target_pred = np.array(np.argmax(neigh.predict(latent_test), axis=1))
            target_test = np.array(np.argmax(target_test, axis=1))
            accuracy = np.mean(np.equal(target_pred, target_test))
        else:
            neigh = KNeighborsClassifier(n_neighbors=k+1)
            neigh.fit(latent_train, target_train)
            neighbor_indices = neigh.kneighbors(latent_train, return_distance=False)
            neighbor_indices = neighbor_indices[:, 1:]
            
            target_train = np.array(np.argmax(target_train, axis=1))
            target_pred = mode(target_train[neighbor_indices], axis=1)[0][:, 0]
            accuracy = np.mean(np.equal(target_pred, target_train))
        print('accuracy: ', accuracy)
                
class VAE(nn.Module):
    def __init__(self, dataset, n_epochs, epochs_per_line, lr, lr_schedule, batch_size, data_root_dir='./', n_classes=None, n_data_points=None, dim_z=64, nf=64, intermediate_nodes=256, beta=1.0, method='iVAE', seed=0, kl_annealing=False, aggressive_post=False, alpha_step=0.025):
        super().__init__()
        
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.epochs_per_line = epochs_per_line
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.dim_z = dim_z
        self.nf = nf
        self.intermediate_nodes = intermediate_nodes
        self.beta = beta
        self.method = method
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = str(int(time()))
        self.seed = seed
        self.kl_annealing = kl_annealing
        self.aggressive_post = aggressive_post
        self.alpha_step = alpha_step
        
        self.aggressive_flag = self.aggressive_post
        self.max_sub_iter = 50
        
        if self.dataset in ['EMNIST', 'FashionMNIST']:
            self.save_dir = os.path.join('./results/%s/%s/%d/kl_annealing=%r-aggressive_post=%r-dim_z=%d-nf=%d-intermediate_nodes=%d-beta=%.6f-lr=%.6f-timestamp=%s' % (self.method, self.dataset, self.seed, self.kl_annealing, self.aggressive_post, self.dim_z, self.nf, self.intermediate_nodes, self.beta, self.lr, self.timestamp))
            self.data_root_dir = data_root_dir
            if self.dataset == 'EMNIST':
                self.n_classes = 10
                self.n_dims = 28*28
                self.nc = 1
                self.train_loader, self.val_loader = make_dataloader_emnist(batch_size=self.batch_size, train=True, root_dir=self.data_root_dir)
                self.test_loader  = make_dataloader_emnist(batch_size=self.batch_size, train=False, root_dir=self.data_root_dir)
            elif self.dataset == 'FashionMNIST':
                self.n_classes = 10
                self.n_dims = 28*28
                self.nc = 1
                self.train_loader, self.val_loader = make_dataloader_fashionmnist(batch_size=self.batch_size, train=True, root_dir=self.data_root_dir)
                self.test_loader  = make_dataloader_fashionmnist(batch_size=self.batch_size, train=False, root_dir=self.data_root_dir)
            self.dim_u = self.n_classes
            self.encoder = model.Encoder(self.dim_z, nc=self.nc, nf=self.nf, intermediate_nodes=self.intermediate_nodes, device=self.device)
            self.decoder = model.Decoder(self.dim_z, nc=self.nc, nf=self.nf, intermediate_nodes=self.intermediate_nodes, device=self.device)
            self.label_prior = model.Label_Prior(self.dim_z, self.dim_u, hidden_nodes=256, device=self.device)
            
            print(self.encoder)
            print('number of parameters in encoder:', count_parameters(self.encoder))
            print(self.decoder)
            print('number of parameters in decoder:', count_parameters(self.decoder))
            print(self.label_prior)
            print('number of parameters in label prior:', count_parameters(self.label_prior))
            if self.method == 'IDVAE':
                self.label_decoder = model.Label_Decoder(self.dim_u, self.dim_z, hidden_nodes=256, device=self.device)
                print('number of parameters in label decoder:', count_parameters(self.label_decoder))
            self.decoder.save_dir = self.save_dir
            self.decoder.dim_z = self.dim_z
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
            
        self.to(self.device)
            
    def forward(self, x, rev=False):
        mean, log_var = self.encoder(x)
        return mean, log_var
    
    def calc_mi(self, data, target):
        """
        E_u I(x, z|u) = E_(x,u)E_{q_{\phi}(z|x,u)}log(q_{\phi}(z|x,u))
                        - E_(x,u)E_{q_{\phi}(z|x,u)}log(q_{\phi}(z|u))
        """
        lam_mean, lam_log_var = self.label_prior(target)
        z_mean, z_log_var = self.encoder(data)                
        post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
        
        n = z_mean.size()[0]
        
        # E_{q_{\phi}(z|x,u)}log(q_{\phi}(z|x,u)) = -0.5*dim_z*log(2*\pi) - 0.5*(1+post_log_var).sum(-1)
        neg_entropy = (-0.5*self.dim_z*math.log(2*math.pi)-0.5*(1+post_log_var).sum(-1)).mean()
        
        post_sample = sampling(post_mean, post_log_var)
        post_mean, post_log_var = post_mean.unsqueeze(0), post_log_var.unsqueeze(0)
        post_var = post_log_var.exp()
        dev = post_sample - post_mean
        log_density = (-0.5*((dev**2)/post_var).sum(dim=-1)-
                       0.5*(self.dim_z*math.log(2*math.pi)+post_log_var.sum(-1)))
        
        # log q_{\phi}(z|u): aggregate posterior
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(n)
        
        return (neg_entropy - log_qz.mean(-1)).item()
    
    def log_sum_exp(self, value, dim=None, keepdim=False):
        # ref: https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/utils.py
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)
    
    def train_model(self):
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
            f.write(f'dim_z {self.dim_z}\n')
        os.makedirs(os.path.join(self.save_dir, 'model_save'))
        os.makedirs(os.path.join(self.save_dir, 'score_save'))
        os.makedirs(os.path.join(self.save_dir, 'figures'))
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.train()
        self.to(self.device)
        print('  time     epoch    iteration         loss       last checkpoint')
        if self.aggressive_post:
            optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                                 self.lr)
            optimizer_decoder = torch.optim.Adam(self.decoder.parameters(),
                                                 self.lr)
            optimizer_label_prior = torch.optim.Adam(self.label_prior.parameters(),
                                                     self.lr)
            sched_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_encoder, self.lr_schedule)
            sched_decoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_decoder, self.lr_schedule)
            sched_label_prior = torch.optim.lr_scheduler.MultiStepLR(optimizer_label_prior, self.lr_schedule)
        else:
            optimizer = torch.optim.Adam(self.parameters(), self.lr)
            sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_schedule)
        losses = []
        self.best_val_epoch = 1
        t0 = time()
        
        mi_not_improved =0
        num_batch_used = 0
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.decoder.epoch = epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                num_batch_used += 1
                if (self.kl_annealing & (num_batch_used <= (10*len(self.train_loader)))):
                    kl_annealing_coeff = 0.1+(1.0-0.1)*num_batch_used/(10*len(self.train_loader))
                else:
                    kl_annealing_coeff = 1.0
                
                if self.aggressive_post:
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    optimizer_label_prior.zero_grad()
                else:
                    optimizer.zero_grad()
                
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                target = nn.functional.one_hot(target, num_classes=self.dim_u)
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                
                # ref: https://github.com/jxhe/vae-lagging-encoder/blob/master/image.py
                num_sub_iter = 1
                sub_num_examples = 0
                sub_pre_loss = 1e4
                sub_cur_loss = 0
                while (self.method == 'iVAE') and self.aggressive_flag and num_sub_iter < self.max_sub_iter:
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    optimizer_label_prior.zero_grad()
                    
                    sub_num_examples += np.shape(data)[0]
                    
                    lam_mean, lam_log_var = self.label_prior(target)
                    z_mean, z_log_var = self.encoder(data)                
                    post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)

                    post_sample = sampling(post_mean, post_log_var)
                    recon_data_post = self.decoder(post_sample)

                    obs_loglik_post = -torch.mean((recon_data_post - data)**2, dim=[1, 2, 3])
                    kl_post_prior = kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                    elbo_iVAE = obs_loglik_post - kl_annealing_coeff*self.beta*kl_post_prior
                    
                    loss = -elbo_iVAE
                    loss = torch.mean(loss)
                    loss.backward()
                    optimizer_encoder.step()
                    if num_sub_iter % 10 == 0:
                        sub_cur_loss = sub_cur_loss / sub_num_examples
                        if sub_pre_loss - sub_cur_loss < 0:
                            break
                        sub_pre_loss = sub_cur_loss
                        sub_cur_loss = sub_num_examples = 0

                    num_sub_iter += 1
                
                lam_mean, lam_log_var = self.label_prior(target)
                z_mean, z_log_var = self.encoder(data)                
                post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                
                post_sample = sampling(post_mean, post_log_var)
                z_sample = sampling(z_mean, z_log_var)
                recon_data_post = self.decoder(post_sample)
                obs_loglik_post = -torch.mean((recon_data_post - data)**2, dim=[1, 2, 3])
                kl_post_prior = kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                elbo_iVAE = obs_loglik_post - kl_annealing_coeff*self.beta*kl_post_prior
                
                if self.method == 'iVAE':
                    loss = -elbo_iVAE
                elif self.method == 'IDVAE':
                    u_sample = sampling(lam_mean, lam_log_var)
                    recon_u = self.label_decoder(u_sample)
                    obs_loglik_cond = -torch.mean((recon_u - target)**2, dim=[1])
                    kl_cond = kl_criterion(lam_mean, lam_log_var,
                                           torch.zeros_like(lam_mean),
                                           torch.ones_like(lam_log_var))
                    elbo_cond = obs_loglik_cond - kl_annealing_coeff*self.beta*kl_cond
                    loss = -elbo_iVAE-elbo_cond
                elif self.method == 'CI-iVAE':
                    recon_data_encoded = self.decoder(z_sample)
                    obs_loglik_encoded = -torch.mean((recon_data_encoded - data)**2, dim=[1, 2, 3])
                    kl_encoded_prior = kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
                    elbo_VAE_with_label_prior = obs_loglik_encoded - kl_annealing_coeff*self.beta*kl_encoded_prior
                    
                    M = 100
                    
                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).to(self.device)
                    z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                    z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                    z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).to(self.device)
                    post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                    post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                    post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon
                    
                    log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                    log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                    log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                    log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                    
                    alpha_list = self.alpha_step + np.arange(0.0, 1.0-self.alpha_step,
                                                             self.alpha_step)
                    loss = torch.zeros((elbo_iVAE.shape[0], len(alpha_list)+2))
                    loss[:, 0], loss[:, -1] = -elbo_iVAE, -elbo_VAE_with_label_prior
                    i = 1
                    for alpha in alpha_list:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_iVAE = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_iVAE = torch.abs(torch.mean(skew_kl_iVAE, dim=-1))
                        skew_kl_VAE_with_label_prior = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_VAE_with_label_prior = torch.abs(torch.mean(skew_kl_VAE_with_label_prior, dim=-1))
                        loss[:, i] = -alpha*elbo_VAE_with_label_prior-(1.0-alpha)*elbo_iVAE+alpha*skew_kl_VAE_with_label_prior+(1.0-alpha)*skew_kl_iVAE
                        i += 1
                    del(i)
                    loss, _ = torch.min(loss, dim = 1)
                    
                loss = torch.mean(loss)
                
                self.print_loss(loss.item(), batch_idx, epoch, t0)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                if self.aggressive_post:
                    if not self.aggressive_flag:
                        optimizer_encoder.step()
                    optimizer_decoder.step()
                    optimizer_label_prior.step()
                else:
                    optimizer.step()
                
            if (epoch+1)%self.epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, epoch, t0, new_line=True)
                obs_loglik_posts, kl_post_priors, losses = [], [], []
            
            if self.aggressive_post:
                sched_encoder.step()
                sched_decoder.step()
                sched_label_prior.step()
            else:
                sched.step()
            
            val_loss = 0; n_val = 0; val_mi = 0
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                target = nn.functional.one_hot(target, num_classes=self.dim_u)
                target = torch.tensor(target, dtype=torch.float32, device=self.device)
                
                if self.aggressive_post:
                    val_mi += self.calc_mi(data, target)
                
                lam_mean, lam_log_var = self.label_prior(target)
                z_mean, z_log_var = self.encoder(data)                
                post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                
                post_sample = sampling(post_mean, post_log_var)
                z_sample = sampling(z_mean, z_log_var)
                recon_data_post = self.decoder(post_sample)
                obs_loglik_post = -torch.mean((recon_data_post - data)**2, dim=[1, 2, 3])
                kl_post_prior = kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                elbo_iVAE = obs_loglik_post - self.beta*kl_post_prior
                
                if self.method == 'iVAE':
                    loss = -elbo_iVAE
                elif self.method == 'IDVAE':
                    u_sample = sampling(lam_mean, lam_log_var)
                    recon_u = self.label_decoder(u_sample)
                    obs_loglik_cond = -torch.mean((recon_u - target)**2, dim=[1])
                    kl_cond = kl_criterion(lam_mean, lam_log_var,
                                           torch.zeros_like(lam_mean),
                                           torch.ones_like(lam_log_var))
                    elbo_cond = obs_loglik_cond - kl_annealing_coeff*self.beta*kl_cond
                    loss = -elbo_iVAE-elbo_cond
                elif self.method == 'CI-iVAE':
                    recon_data_encoded = self.decoder(z_sample)
                    obs_loglik_encoded = -torch.mean((recon_data_encoded - data)**2, dim=[1, 2, 3])
                    kl_encoded_prior = kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
                    elbo_VAE_with_label_prior = obs_loglik_encoded - self.beta*kl_encoded_prior
                
                    M = 100
                    
                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).to(self.device)
                    z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                    z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                    z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                    epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M)).to(self.device)
                    post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                    post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                    post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon
                    
                    log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                    log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                    log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                    log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                    
                    alpha_list = self.alpha_step + np.arange(0.0, 1.0-self.alpha_step,
                                                             self.alpha_step)
                    loss = torch.zeros((elbo_iVAE.shape[0], len(alpha_list)+2))
                    loss[:, 0], loss[:, -1] = -elbo_iVAE, -elbo_VAE_with_label_prior
                    i = 1
                    for alpha in alpha_list:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_iVAE = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_iVAE = torch.abs(torch.mean(skew_kl_iVAE, dim=-1))
                        skew_kl_VAE_with_label_prior = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_VAE_with_label_prior = torch.abs(torch.mean(skew_kl_VAE_with_label_prior, dim=-1))
                        loss[:, i] = -alpha*elbo_VAE_with_label_prior-(1.0-alpha)*elbo_iVAE+alpha*skew_kl_VAE_with_label_prior+(1.0-alpha)*skew_kl_iVAE
                        i += 1
                    del(i)
                    loss, _ = torch.min(loss, dim = 1)
                    
                loss = torch.sum(loss)
                val_loss += loss.item()
                n_val += np.shape(data)[0]
            val_loss /= n_val
            val_mi /= n_val
            
            if epoch == 0:
                best_val_loss = val_loss
                best_val_mi = val_mi
            else:
                best_val_loss = np.minimum(best_val_loss, val_loss)
                if self.aggressive_flag:
                    if val_mi - best_val_mi < 0:
                        mi_not_improved += 1
                        if mi_not_improved == 5:
                            self.aggressive_flag = False
                            print("STOP BURNING")
                    else:
                        best_val_mi = val_mi
            
            if best_val_loss == val_loss:
                print('Best validation loss occurs. Save current model.')
                self.best_val_epoch = epoch + 1
                self.save(os.path.join(self.save_dir, 'model_save', f'{epoch+1:03d}.pt'))
                
            if (epoch+1) == self.n_epochs:
                self.load(os.path.join(self.save_dir, 'model_save', f'{self.best_val_epoch:03d}.pt'))
                self.make_plots()
                score_df = {}
                score_df['seed'] = self.seed
                score_df['val_loss'] = best_val_loss
                score_df['ssw_over_sst'] = self.ssw_over_sst
                score_df = pd.DataFrame(score_df, index=[0])
                score_df.to_csv(os.path.join(self.save_dir, 'score_save', f'{epoch+1:03d}.csv'), index=False)
    
    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = self.best_val_epoch
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')
    
    def save(self, fname):
        state_dict = OrderedDict((k,v) for k,v in self.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)
    
    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])
    
    def make_plots(self):
        if self.dataset in ['EMNIST', 'FashionMNIST']:
            os.makedirs(os.path.join(self.save_dir, 'figures', f'epoch_{self.epoch+1:03d}'))
            self.set_mu_sig()
            sig_rms = np.sqrt(np.mean((self.decoder.sig**2).detach().cpu().numpy(), axis=0))
            plot_samples(self.decoder, n_rows=20, dataset=self.dataset, method=self.method)
            plot_spectrum(self, sig_rms, dims_to_plot=self.dim_z, dataset=self.dataset)
            n_dims_to_plot = 40
            top_sig_dims = np.flip(np.argsort(sig_rms))
            dims_to_plot = top_sig_dims[:n_dims_to_plot]
            plot_variation_along_dims(self.decoder, dims_to_plot, method=self.method)
            plot_t_sne(self, dataset=self.dataset)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
    
    def set_mu_sig(self, n_batches=40):        
        examples = iter(self.test_loader)
        n_batches = min(n_batches, len(examples))
        self.lam_mean, self.lam_log_var, self.z_mean, self.z_log_var, self.target = [], [], [], [], []
        for _ in range(n_batches):
            data, targ = next(examples)
            data += torch.randn_like(data)*1e-2
            onehot_targ = nn.functional.one_hot(targ, num_classes=self.dim_u)
            onehot_targ = torch.tensor(onehot_targ, dtype=torch.float32, device=self.device)
            self.eval()
            
            lam_mean, lam_log_var = self.label_prior(onehot_targ)
            z_mean, z_log_var = self.encoder(data.to(self.device))
            lam_mean, lam_log_var, z_mean, z_log_var = lam_mean.detach().cpu().numpy(), lam_log_var.detach().cpu().numpy(), z_mean.detach().cpu().numpy(), z_log_var.detach().cpu().numpy()
            
            self.lam_mean.append(lam_mean)
            self.lam_log_var.append(lam_log_var)
            self.z_mean.append(z_mean)
            self.z_log_var.append(z_log_var)
            self.target.append(targ)
        
        self.lam_mean = np.concatenate(self.lam_mean, 0)
        self.lam_log_var = np.concatenate(self.lam_log_var, 0)
        self.z_mean = np.concatenate(self.z_mean, 0)
        self.z_log_var = np.concatenate(self.z_log_var, 0)
        self.target = np.concatenate(self.target, 0)
        
        self.decoder.mu = torch.tensor([self.lam_mean[self.target == i].mean(0) for i in range(self.dim_u)]).to(self.device)
        self.decoder.sig = torch.tensor([np.exp(0.5*self.lam_log_var[self.target == i]).mean(0) for i in range(self.dim_u)]).to(self.device)
        
        sst = np.sum((self.z_mean - np.mean(self.z_mean, axis=0))**2)
        ssw = np.sum([np.sum((self.z_mean[self.target==i]-np.mean(self.z_mean[self.target==i], axis=0))**2) for i in range(self.dim_u)])
        print('ssw/sst: %.3f' % (ssw/sst))
        self.ssw_over_sst = ssw/sst
    
    def calculate_knn_accuracy(self, k=5, test=True):
        model_dir = './results/%s/%s/%d' % (self.method, self.dataset, self.seed)
        model_dir = os.path.join(model_dir, os.listdir(model_dir)[0], 'model_save')
        model_dir = os.path.join(model_dir, np.sort(os.listdir(model_dir))[-1])
        self.load(model_dir)
        
        latent_train = []; latent_test = []
        target_train = []; target_test = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = nn.functional.one_hot(target, num_classes=self.dim_u)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)

            z_mean, _ = self.encoder(data)
            latent_train.append(z_mean.cpu().detach().numpy())
            target_train.append(target.cpu().numpy())
        latent_train = np.concatenate(latent_train)
        target_train = np.concatenate(target_train)
            
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data = data.to(self.device)
            target = nn.functional.one_hot(target, num_classes=self.dim_u)
            target = torch.tensor(target, dtype=torch.float32, device=self.device)

            z_mean, _ = self.encoder(data)
            latent_test.append(z_mean.cpu().detach().numpy())
            target_test.append(target.cpu().numpy())
        latent_test = np.concatenate(latent_test)
        target_test = np.concatenate(target_test)
        
        if test:
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(latent_train, target_train)
            
            target_pred = np.array(np.argmax(neigh.predict(latent_test), axis=1))
            target_test = np.array(np.argmax(target_test, axis=1))
            accuracy = np.mean(np.equal(target_pred, target_test))
        else:
            neigh = KNeighborsClassifier(n_neighbors=k+1)
            neigh.fit(latent_train, target_train)
            neighbor_indices = neigh.kneighbors(latent_train, return_distance=False)
            neighbor_indices = neighbor_indices[:, 1:]
            
            target_train = np.array(np.argmax(target_train, axis=1))
            target_pred = mode(target_train[neighbor_indices], axis=1)[0][:, 0]
            accuracy = np.mean(np.equal(target_pred, target_train))
        print('accuracy: ', accuracy)
        
    def make_plots_with_saved_model(self, n_batches=32):
        model_dir = './results/%s/%s/%d' % (self.method, self.dataset, self.seed)
        model_dir = os.path.join(model_dir, os.listdir(model_dir)[0], 'model_save')
        model_dir = os.path.join(model_dir, np.sort(os.listdir(model_dir))[-1])
        self.load(model_dir)
        
        self.set_mu_sig()
        sig_rms = np.sqrt(np.mean((self.decoder.sig**2).detach().cpu().numpy(), axis=0))
        plot_samples(self.decoder, n_rows=20, dataset=self.dataset, method=self.method)
        plot_spectrum(self, sig_rms, dims_to_plot=self.dim_z, dataset=self.dataset)
        n_dims_to_plot = 40
        top_sig_dims = np.flip(np.argsort(sig_rms))
        dims_to_plot = top_sig_dims[:n_dims_to_plot]
        plot_variation_along_dims(self.decoder, dims_to_plot, method=self.method)
        plot_t_sne(self, dataset=self.dataset)

def subnet_fc_gray(c_in, c_out):
    width = 392
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv1_gray(c_in, c_out):
    width = 16
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv2_gray(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_fc_color(c_in, c_out):
    width = 1536
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv1_color(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv2_color(c_in, c_out):
    width = 64
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def construct_net_gin(coupling_block):
    width, height, channel = 28, 28, 1
    
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(channel, width, height, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv1_gray, 'clamp':2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv2_gray, 'clamp':2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_fc_gray, 'clamp':2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)

def compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var):
    # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
    post_mean = (z_mean/(1+torch.exp(z_log_var-lam_log_var))) + (lam_mean/(1+torch.exp(lam_log_var-z_log_var)));
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var));
    
    return post_mean, post_log_var

def sampling(mean, log_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        epsilon = torch.randn(mean.shape).cuda()
        return mean + torch.exp(0.5 * log_var).cuda() * epsilon
    elif device == 'cpu':
        epsilon = torch.randn(mean.shape)
        return mean + torch.exp(0.5 * log_var) * epsilon

# Ref: https://github.com/edenton/svg/blob/master/train_svg_lp.py#L131-#L138
def kl_criterion(mu1, log_var1, mu2, log_var2, reduce_mean=False):
    sigma1 = log_var1.mul(0.5).exp() 
    sigma2 = log_var2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(log_var1) + (mu1 - mu2)**2)/(2*torch.exp(log_var2)) - 1/2
    if reduce_mean:
        return torch.mean(kld, dim=-1)
    else:
        return torch.sum(kld, dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

