import torch
import torch.nn as nn
import random
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=1e-6)

class Prior_conti(nn.Module):
    def __init__(self, dim_z, dim_u, hidden_nodes):
        super(Prior_conti, self).__init__()
        self.dim_z, self.dim_u, self.hidden_nodes = dim_z, dim_u, hidden_nodes
        
        # input dimension is dim_u
        self.mu_net = nn.Sequential(
                        nn.Linear(self.dim_u, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Linear(self.hidden_nodes, self.dim_z)
                        )
        self.log_var_net = nn.Sequential(
                        nn.Linear(self.dim_u, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Linear(self.hidden_nodes, self.dim_z)
                        )
        
    def forward(self, u_input):
        mu, log_var = self.mu_net(u_input), self.log_var_net(u_input)
        return mu, log_var

class Encoder(nn.Module):
    def __init__(self, dim_x, gen_nodes, dim_z):
        super(Encoder, self).__init__()
        self.dim_x, self.gen_nodes, self.dim_z = dim_x, gen_nodes, dim_z
        
        # input dimension is dim_x
        self.main = nn.Sequential(
                        nn.Linear(self.dim_x, self.gen_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Linear(self.gen_nodes, self.gen_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
        
        # input dimension is gen_nodes
        self.mu_net = nn.Linear(self.gen_nodes, self.dim_z)
        self.log_var_net = nn.Linear(self.gen_nodes, self.dim_z)
    
    def forward(self, x_input):
        h = self.main(x_input)
        mu, log_var = self.mu_net(h), self.log_var_net(h)
        return mu, log_var
        
class Decoder(nn.Module):
    def __init__(self, dim_z, dim_x, gen_nodes=None):
        super(Decoder, self).__init__()
        self.dim_z, self.dim_x, self.gen_nodes = dim_z, dim_x, gen_nodes
        
        # input dimension is dim_x
        self.main = nn.Sequential(
                        nn.Linear(self.dim_z, self.gen_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.BatchNorm1d(self.gen_nodes, track_running_stats=True),
                        nn.Linear(self.gen_nodes, self.gen_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.BatchNorm1d(self.gen_nodes, track_running_stats=True),
                        )
        
        # input dimension is gen_nodes
        self.mu_net = nn.Sequential(
                        nn.Linear(self.gen_nodes, self.dim_x),
                        #nn.BatchNorm1d(self.dim_x),
                        #nn.Tanh(),
                        )
        self.obs_log_var_net = nn.Linear(1, self.dim_x)
        
    def forward(self, z_input):
        # first nflow layer
        h = self.main(z_input)
        o = self.mu_net(h)
        
        device = z_input.get_device()
        if device == -1:
            one_tensor = torch.ones((1,1))
            obs_log_var = self.obs_log_var_net(one_tensor)
            return o, obs_log_var
        else:
            one_tensor = torch.ones((1,1)).cuda()
            obs_log_var = self.obs_log_var_net(one_tensor)
            return o, obs_log_var
        
