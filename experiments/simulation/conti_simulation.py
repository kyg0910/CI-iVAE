import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
import random
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from sklearn import linear_model

import src.pi_vae as pi_vae
import src.util as util

os.makedirs('./data/sim/', exist_ok=True)
os.environ['PYTHONHASHSEED'] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess = tf.Session(graph=tf.get_default_graph(), config=config)
set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--length', default=30000, type=int)
parser.add_argument('--n_dim', default=100, type=int)
parser.add_argument('--dim_z', default=2, type=int)
parser.add_argument('--seed_num_dataset_list', nargs="+", type=int)
parser.add_argument('--seed_size_opt', default=10, type=int)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--mdl', default='gaussian', type=str)
parser.add_argument('--latent_type', default='sine', type=str)
parser.add_argument('--gen_nodes', default='60', type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--n_hidden_nodes_in_prior', default=20, type=int)
parser.add_argument('--beta_1', default=0.9, type=float)
parser.add_argument('--noise_structure', default='gaussian', type=str)
parser.add_argument('--fix_alpha', default=None, type=float)
parser.add_argument('--M', default=20, type=int)
parser.add_argument('--alpha_step', default=0.05, type=float)
opt = parser.parse_args()

seed_num_dataset_list = opt.seed_num_dataset_list
for seed_num_dataset in seed_num_dataset_list:
    length = opt.length;
    n_dim = opt.n_dim
    dim_z = opt.dim_z
    seed_size_opt = opt.seed_size_opt
    num_epoch = opt.num_epoch
    mdl = opt.mdl
    latent_type = opt.latent_type
    gen_nodes = opt.gen_nodes
    learning_rate = opt.learning_rate
    n_hidden_nodes_in_prior = opt.n_hidden_nodes_in_prior
    beta_1 = opt.beta_1
    noise_structure = opt.noise_structure
    fix_alpha = opt.fix_alpha
    M = opt.M
    alpha_step = opt.alpha_step

    now = datetime.datetime.now()
    if fix_alpha is not None:
        result_folder_path = './results/fix_alpha=%.2f-length=%d-latent_type=%s-seed_num_dataset=%d-seed_size_opt=%d-%d-%d-%d-%d-%d' % (fix_alpha, length, latent_type, seed_num_dataset, seed_size_opt, now.month, now.day, now.hour, now.minute, now.second)
    else:
        result_folder_path = './results/sup_alpha-length=%d-latent_type=%s-seed_num_dataset=%d-seed_size_opt=%d-%d-%d-%d-%d-%d' % (length, latent_type, seed_num_dataset, seed_size_opt, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(result_folder_path)

    sys.stdout=open(result_folder_path + "/log.txt", "w")
    print(opt)
    
    z_true, u_true, mean_true, lam_true, mu_true, var_true = pi_vae.simulate_cont_data_diff_var(length, n_dim, seed_num_dataset, latent_type);
    
    if noise_structure == 'poisson':
        x_true = np.random.poisson(lam_true);
    elif noise_structure == 'gaussian':
        x_true = np.random.normal(mean_true, 1.0, np.shape(mean_true))
        
    if fix_alpha is not None:
        np.savez('./data/sim/fix_alpha=%.2f-noise_structure=%s-mdl=%s-latent_type=%s-seed_num_dataset=%d-sim_100d_poisson_cont_label.npz' % (fix_alpha, noise_structure, mdl, latent_type, seed_num_dataset), u=u_true, z=z_true, x=x_true, lam=lam_true, mean=mean_true, mu=mu_true, var=var_true);

        dat = np.load('./data/sim/fix_alpha=%.2f-noise_structure=%s-mdl=%s-latent_type=%s-seed_num_dataset=%d-sim_100d_poisson_cont_label.npz' % (fix_alpha, noise_structure, mdl, latent_type, seed_num_dataset));
    else:
        np.savez('./data/sim/sup_alpha-noise_structure=%s-mdl=%s-latent_type=%s-seed_num_dataset=%d-sim_100d_poisson_cont_label.npz' % (noise_structure, mdl, latent_type, seed_num_dataset), u=u_true, z=z_true, x=x_true, lam=lam_true, mean=mean_true, mu=mu_true, var=var_true);

        dat = np.load('./data/sim/sup_alpha-noise_structure=%s-mdl=%s-latent_type=%s-seed_num_dataset=%d-sim_100d_poisson_cont_label.npz' % (noise_structure, mdl, latent_type, seed_num_dataset));
    u_true = dat['u'];
    z_true = dat['z'];
    x_true = dat['x'];
    mu_true = dat['mu']
    var_true = dat['var']

    batch_size = 300
    train_prop, val_prop, test_prop = 0.8, 0.1, 0.1
    num_batch_train, num_batch_val, num_batch_test = int(train_prop*length/batch_size), int(val_prop*length/batch_size), int(test_prop*length/batch_size)
    
    x_all = x_true.reshape(int(length/batch_size), batch_size, -1);
    z_all = z_true.reshape(int(length/batch_size), batch_size, -1);
    u_all = u_true.reshape(int(length/batch_size), batch_size, -1)

    x_train = x_all[:num_batch_train];
    z_train = z_all[:num_batch_train];
    u_train = u_all[:num_batch_train];

    x_valid = x_all[num_batch_train:(num_batch_train+num_batch_val)];
    z_valid = z_all[num_batch_train:(num_batch_train+num_batch_val)];
    u_valid = u_all[num_batch_train:(num_batch_train+num_batch_val)];

    x_test = x_all[(num_batch_train+num_batch_val):];
    z_test = z_all[(num_batch_train+num_batch_val):];
    u_test = u_all[(num_batch_train+num_batch_val):];

    summary_stats = []
    summary_stats_headings = ['seed_num_dataset', 'seed_num_opt', 'beta_kl_prior_post(original)', 'beta_kl_encoded_post(alt)', 'validation_loss', 'MSE_post', 'MSE_encoded', 'MSE_prior']
    for seed_num_opt in range(seed_size_opt):
        # For reproducibility
        random.seed(seed_num_opt)
        np.random.seed(seed_num_opt)
        tf.set_random_seed(seed_num_opt)
        tf.random.set_random_seed(seed_num_opt)
        
        os.makedirs(result_folder_path+'/%d' % seed_num_opt)
        model_chk_path = result_folder_path + '/%d/model.h5' % seed_num_opt
        vae = pi_vae.vae_mdl(dim_x=x_all[0].shape[-1],
                           dim_z=dim_z,
                           dim_u=u_all[0].shape[-1],
                           gen_nodes=gen_nodes, n_blk=2,
                             mdl=mdl, disc=False,
                             learning_rate=learning_rate,
                             latent_type = latent_type,
                             beta_1 = beta_1,
                            fix_alpha = fix_alpha,
                            M = M,
                            n_hidden_nodes_in_prior = n_hidden_nodes_in_prior)

        mcp = ModelCheckpoint(model_chk_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

        s_n = vae.fit_generator(pi_vae.custom_data_generator(x_train, z_train[:, :, :2], u_train),
                      steps_per_epoch=len(x_train), epochs=num_epoch,
                      verbose=1,
                      validation_data = pi_vae.custom_data_generator(x_valid, z_valid[:, :, :2], u_valid),
                      validation_steps = len(x_valid), callbacks=[mcp]);

        plt.plot(s_n.history['val_loss'][:])
        plt.title('Validation loss curve')
        plt.savefig(result_folder_path+'/%d/validation_loss_curve.png' % seed_num_opt)
        plt.clf()
        plt.cla()
        
        # visualize representations
        outputs_valid = vae.predict_generator(pi_vae.custom_data_generator(x_valid, z_valid[:, :, :2], u_valid), steps = len(x_valid));
        outputs = vae.predict_generator(pi_vae.custom_data_generator(x_test, z_test[:, :, :2], u_test), steps = len(x_test));
        if mdl == 'gaussian':
            post_mean_valid, post_log_var_valid, post_sample_valid, fire_rate_post_valid, lam_mean_valid, lam_log_var_valid, z_mean_valid, z_log_var_valid, obs_log_var_valid, z_sample_valid, fire_rate_z_valid, alpha_output_valid, fire_rate_gt_valid = outputs_valid
            post_mean, post_log_var, post_sample, fire_rate_post, lam_mean, lam_log_var, z_mean, z_log_var, obs_log_var, z_sample, fire_rate_z, alpha_output, fire_rate_gt = outputs
        elif mdl == 'poisson':
            post_mean, post_log_var, post_sample, fire_rate_post, lam_mean, lam_log_var, z_mean, z_log_var, z_sample, fire_rate_z, alpha_output, fire_rate_gt = outputs
        
        # plot representation: before affine transformation
        length = 30;
        c_vec = plt.cm.viridis(np.linspace(0,1,length))
        bins = np.linspace(-0.5*np.pi, 0.5*np.pi, length);
        centers = (bins[1:]+bins[:-1])/2;
        disc_loc = np.digitize(u_true[:,0], centers);
        c_all = c_vec[disc_loc];

        fsz = 14;
        n_train = np.shape(x_train)[0]*np.shape(x_train)[1]
        n_valid = np.shape(x_valid)[0]*np.shape(x_valid)[1]
        n_test = np.shape(x_test)[0]*np.shape(x_test)[1]
        epsilon = np.random.normal(0, 1, size=np.shape(post_mean))
        
        z_latents = [z_true[-n_test:, :2], post_mean+np.random.normal(0, 1, size=np.shape(post_mean))*np.exp(0.5*post_log_var),
                     z_mean+np.random.normal(0, 1, size=np.shape(post_mean))*np.exp(0.5*z_log_var),
                     lam_mean+np.random.normal(0, 1, size=np.shape(post_mean))*np.exp(0.5*lam_log_var)]
        mu_latents = [mu_true[-n_test:], post_mean, z_mean, lam_mean]
        var_latents = [var_true[-n_test:], post_log_var, z_log_var, lam_log_var]
        
        reg_gt_to_prior = linear_model.Ridge(alpha=.0).fit(mu_true[n_train:(n_train+n_valid):], lam_mean_valid)
        outputs = vae.predict_generator(pi_vae.custom_data_generator(x_test, reg_gt_to_prior.predict(z_true[-n_test:, :2]).reshape(int(n_test/batch_size), batch_size, -1), u_test), steps = len(x_test))
        post_mean, post_log_var, post_sample, fire_rate_post, lam_mean, lam_log_var, z_mean, z_log_var, obs_log_var, z_sample, fire_rate_z, alpha_output, fire_rate_gt = outputs
        
        plt.figure(figsize=(16,8));
        plt.subplots_adjust(top=0.90)
        i = 0
        for latents in [z_latents, mu_latents]:
            j = 0
            for latent in latents:
                ax = plt.subplot(2, len(latents), i*len(latents)+j+1)
                plt.scatter(latent[:,0], latent[:,1], c=c_all[-n_test:], s=1, alpha=0.5);
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.setp(ax.get_xticklabels(), fontsize=fsz);
                plt.setp(ax.get_yticklabels(), fontsize=fsz);
                j += 1
            i += 1
            
        plt.suptitle('[(Before) GT - post(z|x,u) - encoded(z|x) - prior(z|u)]', y=0.98)
        plt.savefig(result_folder_path+'/%d/vis_latent_before_affine_trans.png' % seed_num_opt)
        plt.clf()
        plt.cla()

        # plot representation: after affine transformation
        mse = []
        plt.figure(figsize=(16,8));
        plt.subplots_adjust(top=0.90)
        
        reg = {}
        reg['1'] = linear_model.Ridge(alpha=.0).fit(post_mean_valid, mu_true[n_train:(n_train+n_valid):])
        reg['2'] = linear_model.Ridge(alpha=.0).fit(z_mean_valid, mu_true[n_train:(n_train+n_valid):])
        reg['3'] = linear_model.Ridge(alpha=.0).fit(lam_mean_valid, mu_true[n_train:(n_train+n_valid):])
        
        i = 0
        for latents in [z_latents, mu_latents]:
            j = 0
            for latent in latents:
                ax = plt.subplot(2, len(latents), i*len(latents)+j+1) 
                if j != 0:
                    latent_transformed = reg['%d' % j].predict(latent)
                    mse.append(np.mean(np.square(latent_transformed - latents[0][:, :2])))
                    plt.scatter(latent_transformed[:,0], latent_transformed[:,1], c=c_all[-n_test:], s=1, alpha=0.5);
                else:
                    plt.scatter(latent[:,0], latent[:,1], c=c_all[-n_test:], s=1, alpha=0.5);
                    
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.setp(ax.get_xticklabels(), fontsize=fsz);
                plt.setp(ax.get_yticklabels(), fontsize=fsz);
                j += 1
            i += 1
        mse_mu_post, mse_mu_encoded, mse_mu_prior = mse[3:6]       
        
        plt.suptitle('[(After) GT - post(z|x,u) - encoded(z|x) - prior(z|u)]', y=0.98)
        plt.savefig(result_folder_path+'/%d/vis_latent_after_affine_trans.png' % seed_num_opt)
        plt.clf()
        plt.cla()

        # calculate log density
        x_input = np.reshape(x_test, (-1, x_all[0].shape[-1]))
        if mdl == 'gaussian':
            post_mean, post_log_var, post_sample, fire_rate_post, lam_mean, lam_log_var, z_mean, z_log_var, obs_log_var, z_sample, fire_rate_z, alpha_output, fire_rate_gt = outputs
            obs_log_var = np.repeat(obs_log_var, batch_size, axis=0)
        elif mdl == 'poisson':
            post_mean, post_log_var, post_sample, fire_rate_post, lam_mean, lam_log_var, z_mean, z_log_var, z_sample, fire_rate_z, alpha_output, fire_rate_gt = outputs
            
        epsilon = np.random.normal(size=(np.shape(lam_mean)[0], np.shape(lam_mean)[1], M))
        lam_mean_tiled = np.repeat(np.expand_dims(lam_mean, axis=2), M, axis=2)
        lam_log_var_tiled = np.repeat(np.expand_dims(lam_log_var, axis=2), M, axis=2)
        lam_sample_tiled = lam_mean_tiled + np.exp(0.5 * lam_log_var_tiled) * epsilon
        
        density = np.zeros(np.shape(lam_mean)[0])
        for m in range(M):
            _, mu_x_given_z = pi_vae.true_mapping_from_z_to_x(lam_sample_tiled[:, :, m], n_dim)
            density += np.prod(np.exp(-0.5*(x_input - mu_x_given_z)**2), axis=1)
        density /= M
        log_density = np.mean(np.log(density))
        
        current_row = {}
        
        if fix_alpha is None:
            current_row['method'] = 'CI-iVAE'
        elif fix_alpha == 0.0:
            current_row['method'] = 'iVAE'
        elif fix_alpha == 1.0:
            current_row['method'] = 'VAE with label prior'
            
        current_row['seed_num_dataset'] = seed_num_dataset
        current_row['seed_num_opt'] = seed_num_opt
        current_row['latent_type'] = latent_type
        current_row['validation_loss'] = np.min(s_n.history['val_loss'])
        current_row['test_loss'] = vae.evaluate(pi_vae.custom_data_generator(x_test, z_test[:, :, :2], u_test), steps = len(x_test))
        current_row['MSE_mu_post'] = mse_mu_post
        current_row['MSE_mu_encoded'] = mse_mu_encoded
        current_row['MSE_mu_prior'] = mse_mu_prior
        current_row['log_density'] = log_density
        current_row['fix_alpha'] = fix_alpha
        current_row['noise_structure'] = noise_structure

        summary_stats.append(current_row)

        del(vae)

    summary_stats = pd.DataFrame(summary_stats)
    summary_stats.to_csv('%s/summary_stats.csv' % result_folder_path, index=False)
    sys.stdout.close()
