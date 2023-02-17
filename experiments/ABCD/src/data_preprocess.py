import torch
import random
import math
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
from sklearn import preprocessing

def preprocess(config, data_path = None, make_loggers=True, output_headings = False):
    if data_path is None:
        data_path = config['data_path']
    save_model_path =  config['save_model_path']
    n_fold = config['n_fold']
    u_names = config['u_names']
    dropna = config['dropna']
    
    dim_u = len(u_names)
    cat_names = ['sex']
    
    # load dataset
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)
    
    log_dir = save_model_path + '/logs'
    loggers = {}

    if make_loggers:
        dataset_names = ['train', 'val', 'test']
        for i in range(n_fold):
            loggers[i] = {}
            for dataset_name in dataset_names:
                loggers[i][dataset_name] = SummaryWriter('%s/%d/%s' % (log_dir, i, dataset_name))
        
    headings = np.array(list(df))
    subheadings_connectivity = headings[:np.where(headings=="subjectkey")[0][0]]
    x_true = np.array(df[subheadings_connectivity])
    for i in range(dim_u):
        if u_names[i] in cat_names:
            lb = preprocessing.LabelBinarizer()
            current_u = lb.fit_transform(np.array(df[u_names[i]].astype('category').cat.codes.to_numpy(),
                                                  dtype='float32'))
        else:
            current_u = np.array(df[u_names[i]], dtype='float32')
            current_u = (current_u-np.mean(current_u, axis=0))/np.std(current_u, axis=0)
            
        if len(np.shape(current_u)) == 1:
            current_u = current_u[:, np.newaxis]
        u_true = current_u if i ==0 else np.concatenate((u_true, current_u), axis=1)
        
    x_true = (x_true-np.quantile(x_true, 0.5, axis=0))/(np.quantile(x_true, 0.75, axis=0)-np.quantile(x_true, 0.25, axis=0))
    
    if output_headings:
        return df, x_true, u_true, loggers, brain_headings
    else:
        return df, x_true, u_true, loggers

def split_train_val_test(x_true, u_true, train_idx, val_idx, test_idx, config):
    dtype = config['dtype']
    
    x_train, u_train = torch.tensor(x_true[train_idx], dtype=dtype), torch.tensor(u_true[train_idx], dtype=dtype)
    x_val, u_val = torch.tensor(x_true[val_idx], dtype=dtype), torch.tensor(u_true[val_idx], dtype=dtype)
    x_test, u_test = torch.tensor(x_true[test_idx], dtype=dtype), torch.tensor(u_true[test_idx], dtype=dtype)
    
    return x_train, u_train, x_val, u_val, x_test, u_test