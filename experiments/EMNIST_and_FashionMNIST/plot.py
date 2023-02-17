import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import os
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import seaborn as sns

def plot_samples(model, n_rows, temp=1, dataset='EMNIST', method='GIN'):
    """
    Plots sampled digits. Each row contains all 10 digits with a consistent style
    """
    width, height, channel = 28, 28, 1
        
    if method == 'GIN':
        dims_to_sample=torch.arange(width*height*channel)
    elif method in ['iVAE', 'IDVAE', 'CI-iVAE']:
        dims_to_sample=torch.arange(model.dim_z)
        
    model.eval()
    fig = plt.figure(figsize=(10, n_rows))
    n_dims_to_sample = len(dims_to_sample)
    style_sample = torch.zeros(n_rows, n_dims_to_sample)
    style_sample[:,dims_to_sample] = torch.randn(n_rows, n_dims_to_sample)*temp
    style_sample = style_sample.to(model.device)
    # style sample: (n_rows, n_dims)
    # mu,sig: (n_classes, n_dims)
    # latent: (n_rows, n_classes, n_dims)
    latent = style_sample.unsqueeze(1)*model.sig.unsqueeze(0) + model.mu.unsqueeze(0)
    latent.detach_()
    # data: (n_rows, n_classes, width, height)
    if method == 'GIN':
        data = (model(latent.view(-1, n_dims_to_sample), rev=True)[0]).detach().cpu().numpy().reshape(n_rows, 10, channel, height, width)
    elif method in ['iVAE', 'IDVAE', 'CI-iVAE']:
        data = (model(latent.view(-1, n_dims_to_sample))).detach().cpu().numpy().reshape(n_rows, 10, channel, height, width)
        
    im = data[:, :, 0, :, :].transpose(0, 2, 1, 3).reshape(n_rows*height, 10*width)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'samples.png'), bbox_inches='tight', pad_inches=0.5)
    plt.close()


def plot_variation_along_dims(model, dims_to_plot, method='GIN'):
    """
    Makes a plot for each of the given latent space dimensions. Each column contains all 10 digits
    with a consistent style. Each row shows the effect of varying the latent space value of the 
    chosen dimension from -2 to +2 standard deviations while keeping the latent space
    values of all other dimensions constant at the mean value. The rightmost column shows a heatmap
    of the absolute pixel difference between the column corresponding to -1 std and +1 std
    """
    os.makedirs(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots'))
    max_std = 2
    n_cols = 9
    width, height, channel = 28, 28, 1
    
    model.eval()
    for i, dim in enumerate(dims_to_plot):
        fig = plt.figure(figsize=(n_cols+1, 10))
        if method == 'GIN':
            style = torch.zeros(n_cols, width*height*channel)
        elif method in ['iVAE', 'IDVAE', 'SparseVAE', 'CI-iVAE']:
            style = torch.zeros(n_cols, model.dim_z)
        style[:, dim] = torch.linspace(-max_std, max_std, n_cols)
        style = style.to(model.device)
        # style: (n_cols, n_dims)
        # mu,sig: (n_classes, n_dims)
        # latent: (n_classes, n_cols, n_dims)
        latent = style.unsqueeze(0)*model.sig.unsqueeze(1) + model.mu.unsqueeze(1)
        latent.detach_()
        if method == 'GIN':
            data = (model(latent.view(-1, channel*width*height), rev=True)[0]).detach().cpu().numpy().reshape(10, n_cols, channel, width, height)
        elif method in ['iVAE', 'IDVAE', 'SparseVAE', 'CI-iVAE']:
            data = (model(latent.view(-1, model.dim_z))).detach().cpu().numpy().reshape(10, n_cols, channel, width, height)
            
        im = data[:, :, 0, :, :].transpose(0, 2, 1, 3).reshape(10*width, n_cols*height)
        # images at +1 and -1 std
        im_p1 = im[:, height*2:height*3]
        im_m1 = im[:, height*6:height*7]
        # full image with spacing between the two parts
        im = np.concatenate([im, np.ones((10*width, 3)), np.abs(im_p1-im_m1)], axis=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots', f'variable_{i+1:03d}.png'), bbox_inches='tight', pad_inches=0.5)
        plt.close()


def plot_spectrum(model, sig_rms, dims_to_plot, dataset='EMNIST'):
    fig = plt.figure(figsize=(12, 6))
    plt.semilogy(np.flip(np.sort(sig_rms))[:dims_to_plot], 'k')
    plt.xlabel('Latent dimension (sorted)')
    plt.ylabel('Standard deviation (RMS across classes)')
    plt.title('Spectrum on %s' % dataset)
    plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'spectrum_dims_to_plot=%d.png' % dims_to_plot))
    plt.close()
    
def plot_t_sne(model, dataset='EMNIST'):
    if dataset in ['EMNIST']:
        colors = {0:'Digit 0', 1:'Digit 1', 2:'Digit 2', 3:'Digit 3', 4:'Digit 4', 5:'Digit 5', 6:'Digit 6', 7:'Digit 7', 8:'Digit 8', 9:'Digit 9'}
    elif dataset == 'FashionMNIST':
        colors = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in colors.items()]
    
    tsne_results = np.concatenate((model.z_mean, np.array([model.lam_mean[model.target==i][0] for i in range(len(np.unique(model.target)))])))
    N = np.shape(tsne_results)[0]
    tsne_results = TSNE(n_components=2, learning_rate=max(N / 12.0 / 4, 50), init='pca', random_state=0).fit_transform(tsne_results)
    tsne_results_z_df = pd.DataFrame({'encoded (z|x)_1': tsne_results[:np.shape(model.z_mean)[0], 0],
                                      'encoded (z|x)_2': tsne_results[:np.shape(model.z_mean)[0], 1],
                                      'label': model.target})
    tsne_results_z_df['label'] = tsne_results_z_df['label'].map(colors)
    
    plt.figure(figsize=(24, 12))
    
    ax = plt.subplot(1, 2, 1)
    sns.scatterplot(x='encoded (z|x)_1', y='encoded (z|x)_2', hue='label', style='label', data=tsne_results_z_df, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('1st t-SNE Component')
    ax.set_ylabel('2nd t-SNE Component')
    ax.set_title('encoded (z|x) (SSW/SST on Z space: %.4f)' % model.ssw_over_sst)
    ax.legend(title='Class Label', labels=np.unique(tsne_results_z_df['label']))
    
    tsne_results_lam_df = pd.DataFrame({'label prior (z|u)_1': tsne_results[np.shape(model.z_mean)[0]:, 0],
                                        'label prior (z|u)_2': tsne_results[np.shape(model.z_mean)[0]:, 1],
                                        'label': np.arange(len(np.unique(model.target)))})
    tsne_results_lam_df['label'] = tsne_results_lam_df['label'].map(colors)
    
    ax = plt.subplot(1, 2, 2)
    sns.scatterplot(x='label prior (z|u)_1', y='label prior (z|u)_2', hue='label', style='label', data=tsne_results_lam_df, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('1st t-SNE Component')
    ax.set_ylabel('2nd t-SNE Component')
    ax.set_title('label prior (z|u)')
    ax.legend(title='Class Label', labels=np.unique(tsne_results_lam_df['label']))
    
    plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 't-sne.png'))
    plt.close()      
