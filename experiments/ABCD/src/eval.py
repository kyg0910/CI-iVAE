#import statsmodels.api as sm
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn import svm

import src.util as UTIL

def evaluation_OLS(x_true, u_true, load_model_path, df, fold, v_names = None):
    saved_model = torch.load('%s/%d/model.pth' % (load_model_path, fold))
    
    prior = saved_model['prior']
    encoder = saved_model['encoder']
    decoder = saved_model['decoder']
    config = saved_model['config']
    dataset_idx = saved_model['dataset_idx']
    
    dtype = config['dtype']
    dim_z = config['dim_z']
    num_pc = config['num_pc']
    u_names = config['u_names']
    #v_names = config['v_names']
    if v_names is None:
        v_names = config['v_names']
    
    dataset_names = ['train', 'val', 'test']
    dim_u = len(u_names)
    dim_v = len(v_names)
    x, u, v, eps, recon_x, mse_list, log_var_list = {}, {}, {}, {}, {}, {}, {}
    lam_mean, lam_log_var, z_mean, z_log_var, post_mean, post_log_var = {}, {}, {}, {}, {}, {}
    for dataset_name in dataset_names:
        x[dataset_name] = torch.tensor(x_true[dataset_idx[dataset_name]], dtype=dtype)
        u[dataset_name] = torch.tensor(u_true[dataset_idx[dataset_name]], dtype=dtype)
        v[dataset_name] = np.zeros((len(dataset_idx[dataset_name]), len(v_names)))
        recon_x[dataset_name] = torch.zeros_like(x[dataset_name])
        
        i = 0
        for i in range(len(v_names)):
            try:
                v[dataset_name][:, i] = np.array(df[v_names[i]], dtype='float32')[dataset_idx[dataset_name]]
            except:
                v[dataset_name][:, i] = np.array(df[v_names[i]].astype('category').cat.codes.to_numpy(), dtype='float32')[dataset_idx[dataset_name]]
            i += 1
        del(i)
        mse_list[dataset_name], log_var_list[dataset_name] = {}, {}
        '''
        x[dataset_name] = x[dataset_name][:200]
        u[dataset_name] = u[dataset_name][:200]
        v[dataset_name] = v[dataset_name][:200]
        '''
        lam_mean[dataset_name], lam_log_var[dataset_name] = prior(u[dataset_name].cuda())
        z_mean[dataset_name], z_log_var[dataset_name] = encoder(x[dataset_name].cuda())
        post_mean[dataset_name], post_log_var[dataset_name] = UTIL.compute_posterior(z_mean[dataset_name], z_log_var[dataset_name], lam_mean[dataset_name], lam_log_var[dataset_name])
        recon_x[dataset_name], _ = decoder(z_mean[dataset_name])
        eps[dataset_name] = x[dataset_name].cuda() - recon_x[dataset_name]
        
        x[dataset_name] = x[dataset_name].cpu().detach().numpy()
        u[dataset_name] = u[dataset_name].cpu().detach().numpy()
        eps[dataset_name] = eps[dataset_name].cpu().detach().numpy()
        recon_x[dataset_name] = recon_x[dataset_name].cpu().detach().numpy()
        lam_mean[dataset_name] = lam_mean[dataset_name].cpu().detach().numpy()
        lam_log_var[dataset_name] = lam_log_var[dataset_name].cpu().detach().numpy()
        z_mean[dataset_name] = z_mean[dataset_name].cpu().detach().numpy()
        z_log_var[dataset_name] = z_log_var[dataset_name].cpu().detach().numpy()
        post_mean[dataset_name] = post_mean[dataset_name].cpu().detach().numpy()
        post_log_var[dataset_name] = post_log_var[dataset_name].cpu().detach().numpy()
    del(dataset_name)

    latent_name_list = ["Posterior [q(z|x,u)]", "Encoder [q_{phi}(z|x)]", "Prior [p_{T, lambda}(z|u)]"]
    mean_list = {"Posterior [q(z|x,u)]": post_mean,
                "Encoder [q_{phi}(z|x)]": z_mean,
                "Prior [p_{T, lambda}(z|u)]": lam_mean}
    log_var_list = {"Posterior [q(z|x,u)]": post_log_var,
                    "Encoder [q_{phi}(z|x)]": z_log_var,
                    "Prior [p_{T, lambda}(z|u)]": lam_log_var}
    
    input_prediction = {}
    for dataset_name in dataset_names:
        mse_list[dataset_name]['recon_error'] = np.mean((x[dataset_name]-recon_x[dataset_name])**2)
        input_prediction[dataset_name] = {}
        input_prediction[dataset_name]['u'] = u[dataset_name]
        input_prediction[dataset_name]['x'] = x[dataset_name]
        input_prediction[dataset_name]['z'] = np.concatenate((mean_list['Encoder [q_{phi}(z|x)]'][dataset_name],
                                                             log_var_list['Encoder [q_{phi}(z|x)]'][dataset_name]),
                                                            axis = 1)  
        input_prediction[dataset_name]['epsilon'] = eps[dataset_name]
        input_prediction[dataset_name]['(z, epsilon)'] = np.concatenate((input_prediction[dataset_name]['z'],
                                                                         input_prediction[dataset_name]['epsilon']),
                                                                        axis = 1)
        input_prediction[dataset_name]['(u, x)'] = np.concatenate((input_prediction[dataset_name]['u'],
                                                                   input_prediction[dataset_name]['x']),
                                                                  axis = 1)
        input_prediction[dataset_name]['(u, z)'] = np.concatenate((input_prediction[dataset_name]['u'],
                                                                   input_prediction[dataset_name]['z']),
                                                                  axis = 1)
        input_prediction[dataset_name]['(u, epsilon)'] = np.concatenate((input_prediction[dataset_name]['u'],
                                                                         input_prediction[dataset_name]['epsilon']),
                                                                        axis = 1)
        input_prediction[dataset_name]['(u, z, epsilon)'] = np.concatenate((input_prediction[dataset_name]['u'],
                                                                            input_prediction[dataset_name]['z'],
                                                                            input_prediction[dataset_name]['epsilon']),
                                                                           axis = 1)
    del(dataset_name)
    
    svm_list, svr_list = {}, {}
    for dataset_name in dataset_names:        
        input_keys = ['x', 'z', 'epsilon', '(z, epsilon)']
        j = 0
        for i in range(dim_u):            
            length = 9 if u_names[i] in ['Puberty'] else 1
            class_weight = 'balanced' if u_names[i] in ['Puberty'] else None
            for input_key in input_keys:
                try:
                    if dataset_name == 'train':
                        if u_names[i] in ['Puberty']:
                            svm_list['%s on %s' % (input_key, u_names[i])] = svm.SVC(class_weight=class_weight).fit(input_prediction['train'][input_key],
                                                                                                      np.argmax(u['train'][:, j:(j+length)], axis=1))
                        else:
                            svm_list['%s on %s' % (input_key, u_names[i])] = svm.SVC(class_weight=class_weight).fit(input_prediction['train'][input_key],
                                                                                                      u['train'][:, j])
                    if u_names[i] in ['Puberty']:
                        f1 = f1_score(np.argmax(u[dataset_name][:, j:(j+length)], axis=1),
                                      svm_list['%s on %s' % (input_key, u_names[i])].predict(input_prediction[dataset_name][input_key]), average='macro')
                        mse_list[dataset_name]['F1 score: %s on %s' % (input_key, u_names[i])] = f1
                    else:
                        err = np.mean(svm_list['%s on %s' % (input_key, u_names[i])].predict(input_prediction[dataset_name][input_key])
                                        !=u[dataset_name][:, j])
                        mse_list[dataset_name]['Error rate: %s on %s' % (input_key, u_names[i])] = err
                except:
                    if dataset_name == 'train':
                        svr_list['%s on %s' % (input_key, u_names[i])] = svm.SVR().fit(input_prediction['train'][input_key], u['train'][:, j])
                    mse = np.mean((svr_list['%s on %s' % (input_key, u_names[i])].predict(input_prediction[dataset_name][input_key])-u[dataset_name][:, j])**2)
                    mse_list[dataset_name]['MSE: %s on %s' % (input_key, u_names[i])] = mse
            j += length
        del(i, j, length)
        
        input_keys = ['u', 'x', 'z', 'epsilon', '(z, epsilon)',
                  '(u, x)', '(u, z)', '(u, epsilon)', '(u, z, epsilon)']
        
        for i in range(dim_v):
            nan_idx = np.isnan(v[dataset_name][:, i])
            for input_key in input_keys:
                try:
                    if dataset_name == 'train':
                        if v_names[i] in ['ksads_14_853_p']:
                            obs_idx = np.where(v[dataset_name][~nan_idx, i]<=1)[0]
                            svm_list['%s on %s' % (input_key, v_names[i])] = svm.SVC(class_weight='balanced').fit(input_prediction['train'][input_key][~nan_idx][obs_idx],
                                                                                                      v['train'][~nan_idx, i][obs_idx])
                        elif v_names[i] in ['nihtbx_flanker_fc', 'nihtbx_totalcomp_fc', 'pea_wiscv_tss',
                                'tfmri_sst_all_beh_crgo_rt', 'tfmri_nb_all_beh_ctotal_rate']:
                            svm_list['%s on %s' % (input_key, v_names[i])] = svm.SVC(class_weight=class_weight).fit(input_prediction['train'][input_key][~nan_idx],
                                                                                                      v['train'][~nan_idx, i])
                    if v_names[i] in ['ksads_14_853_p']:
                        obs_idx = np.where(v[dataset_name][~nan_idx, i]<=1)[0]
                        f1 = f1_score(v[dataset_name][~nan_idx, i][obs_idx],
                                      svm_list['%s on %s' % (input_key, v_names[i])].predict(input_prediction[dataset_name][input_key][~nan_idx][obs_idx]), average='macro')
                        mse_list[dataset_name]['F1 score: %s on %s' % (input_key, v_names[i])] = f1
                    else:
                        err = np.mean(svm_list['%s on %s' % (input_key, v_names[i])].predict(input_prediction[dataset_name][input_key][~nan_idx])
                                        !=v[dataset_name][~nan_idx, i])
                        mse_list[dataset_name]['Error rate: %s on %s' % (input_key, v_names[i])] = err
                except:
                    if dataset_name == 'train':
                        svr_list['%s on %s' % (input_key, v_names[i])] = svm.SVR().fit(input_prediction['train'][input_key][~nan_idx], v['train'][~nan_idx, i])
                    mse = np.mean((svr_list['%s on %s' % (input_key, v_names[i])].predict(input_prediction[dataset_name][input_key][~nan_idx])-v[dataset_name][~nan_idx, i])**2)
                    mse_list[dataset_name]['MSE: %s on %s' % (input_key, v_names[i])] = mse
        del(i)
    del(dataset_name)
    
    return mse_list
'''
def evaluation_OLS_u_on_z(x_true, u_true, load_model_path, df, use_v, reduction_method, fold):
    saved_model = torch.load('%s/%d/model.pth' % (load_model_path, fold))
    
    prior = saved_model['prior']
    encoder = saved_model['encoder']
    decoder = saved_model['decoder']
    config = saved_model['config']
    dataset_idx = saved_model['dataset_idx']
    
    dtype = config['dtype']
    dim_z = config['dim_z']
    num_pc = config['num_pc']
    u_names = config['u_names']
    v_names = config['v_names']
    
    dataset_names = ['train', 'val', 'test']
    dim_u = len(u_names)
    x, u, v = {}, {}, {}
    lam_mean, lam_log_var, z_mean, z_log_var, post_mean, post_log_var = {}, {}, {}, {}, {}, {}
    for dataset_name in dataset_names:
        x[dataset_name] = torch.tensor(x_true[dataset_idx[dataset_name]], dtype=dtype)
        u[dataset_name] = torch.tensor(u_true[dataset_idx[dataset_name]], dtype=dtype)
        
        lam_mean[dataset_name], lam_log_var[dataset_name] = prior(u[dataset_name].cuda())
        z_mean[dataset_name], z_log_var[dataset_name] = encoder(x[dataset_name].cuda())
        post_mean[dataset_name], post_log_var[dataset_name] = UTIL.compute_posterior(z_mean[dataset_name], z_log_var[dataset_name], lam_mean[dataset_name], lam_log_var[dataset_name])
        
        lam_mean[dataset_name] = lam_mean[dataset_name].cpu().detach().numpy()
        lam_log_var[dataset_name] = lam_log_var[dataset_name].cpu().detach().numpy()
        z_mean[dataset_name] = z_mean[dataset_name].cpu().detach().numpy()
        z_log_var[dataset_name] = z_log_var[dataset_name].cpu().detach().numpy()
        post_mean[dataset_name] = post_mean[dataset_name].cpu().detach().numpy()
        post_log_var[dataset_name] = post_log_var[dataset_name].cpu().detach().numpy()
        
        v[dataset_name] = np.zeros((len(dataset_idx[dataset_name]), len(v_names)))
        i = 0
        for i in range(len(v_names)):
            try:
                v[dataset_name][:, i] = np.array(df[v_names[i]], dtype='float32')[dataset_idx[dataset_name]]
            except:
                v[dataset_name][:, i] = np.array(df[v_names[i]].astype('category').cat.codes.to_numpy(), dtype='float32')[dataset_idx[dataset_name]]
            i += 1
        del(i)
    del(dataset_name)

    prior = prior.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    
    rsquared_adjs, pvalues, explained_variance_ratios = {}, {}, {}
    for dataset_name in dataset_names:
        rsquared_adjs[dataset_name] = {}
        pvalues[dataset_name] = {}
        explained_variance_ratios[dataset_name] = {}
        for latent_type in ['post', 'encoded']:
            rsquared_adjs[dataset_name][latent_type] = []
            pvalues[dataset_name][latent_type] = []
        del(latent_type)
    del(dataset_name)
    
    for dataset_name in dataset_names:
        for latent_type in ['post', 'encoded']:
            if reduction_method == 'pca':
                pca = PCA(n_components=dim_z)
            elif reduction_method == 'pls':
                pls = PLSRegression(n_components=dim_z)
                
            if latent_type == 'post':
                z = post_mean[dataset_name]
                if dim_u == 1:
                    u[dataset_name] = u[dataset_name].detach().numpy().reshape(-1, 1)
                else:
                    u[dataset_name] = u[dataset_name].detach().numpy()
                if use_v:
                    X = np.concatenate((u[dataset_name], v[dataset_name]), axis = 1)
                else:
                    X = u[dataset_name]
                scaler = StandardScaler().fit(X)
                normalizedX = scaler.transform(X)
            elif latent_type == 'encoded':
                z = z_mean[dataset_name]
            
            if reduction_method == 'pca':
                pca.fit(z)
                z = pca.transform(z)
            elif reduction_method == 'pls':
                pls.fit(z, u[dataset_name])
                z = pls.transform(z)

            for i in range(num_pc):
                Y = z[:, i]
                scaler = StandardScaler().fit(Y.reshape(-1, 1))
                normalizedY = scaler.transform(Y.reshape(-1, 1))
                
                model = sm.OLS(normalizedY, normalizedX)
                results = model.fit()

                rsquared_adjs[dataset_name][latent_type].append(results.rsquared_adj)
                pvalues[dataset_name][latent_type].append(results.pvalues)
            del(i)
            if reduction_method == 'pca':
                explained_variance_ratios[dataset_name][latent_type] = pca.explained_variance_ratio_
            elif reduction_method == 'pls':
                explained_variance_ratios[dataset_name][latent_type] = np.var(z, axis=0)/np.sum(np.var(pls.x_scores_, axis=0))
        del(latent_type)
    del(dataset_name)

    return rsquared_adjs, pvalues, explained_variance_ratios
'''