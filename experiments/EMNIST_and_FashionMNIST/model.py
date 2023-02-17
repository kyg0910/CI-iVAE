import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F

class conv_downsample(nn.Module):
    def __init__(self, nin, nout):
        super(conv_downsample, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class conv_upsample(nn.Module):
    def __init__(self, nin, nout):
        super(conv_upsample, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
    
class Encoder(nn.Module):
    def __init__(self, dim_z, nc=1, nf=64, intermediate_nodes=256, device='gpu'):
        super(Encoder, self).__init__()
        self.dim_z = dim_z
        self.nf = nf
        self.intermediate_nodes = intermediate_nodes
        
        width, height, channel = 28, 28, 1

        # input size is (nc) x H x W
        self.c1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.Conv2d(nc, self.nf, 3, 1, 1),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # input size is (nf) x (H) x (W)
        self.c2 = conv_downsample(self.nf, self.nf * 2)
        # input size is (nf*2) x (H/2) x (W/2)
        self.c3 = conv_downsample(self.nf * 2, self.nf * 4)
        # input size is (nf*4) x (H/4) x (W/4)
        self.c4 = nn.Sequential(
                nn.Conv2d(self.nf*4, self.intermediate_nodes, int(width/4), 1, 0),
                nn.BatchNorm2d(self.intermediate_nodes),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # input size is (intermediate_nodes) x (1) x (1)
        self.mean_net = nn.Conv2d(self.intermediate_nodes, self.dim_z, 1, 1, 0)
        self.log_var_net = nn.Conv2d(self.intermediate_nodes, self.dim_z, 1, 1, 0)
        
        self.device = device
    
    def reparameterize(self, mean, log_var):
        log_var = log_var.mul(0.5).exp_()
        eps = Variable(log_var.data.new(log_var.size()).normal_())
        return eps.mul(log_var).add_(mean)
    
    def forward(self, x, return_mean_and_log_var = True):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        mean, log_var = self.mean_net(h4).view(-1, self.dim_z), self.log_var_net(h4).view(-1, self.dim_z)
            
        if return_mean_and_log_var:
            return mean, log_var
        else:
            z = self.reparameterize(mean, log_var)
            return z
    
class Decoder(nn.Module):
    def __init__(self, dim_z, nc=1, nf=64, intermediate_nodes=256, device='gpu'):
        super(Decoder, self).__init__()
        self.dim_z = dim_z
        self.nf = nf
        self.intermediate_nodes = intermediate_nodes
        
        width, height, channel = 28, 28, 1

        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dim_z, self.intermediate_nodes, 1, 1, 0),
            nn.BatchNorm2d(self.intermediate_nodes),
            nn.LeakyReLU(0.2, inplace=True)
            )
        # input size is (nf*4) x (1) x (1)
        self.upc2 = nn.Sequential(
                nn.ConvTranspose2d(self.intermediate_nodes, self.nf*4, int(width/4), 1, 0),
                nn.BatchNorm2d(self.nf*4),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # input size is (nf*4) x (H/4) x (W/4)
        self.upc3 = conv_upsample(self.nf*4, self.nf*2)
        # input size is (nf*2) x (H/2) x (W/2)
        self.upc4 = conv_upsample(self.nf*2, self.nf)
        # input size is (nf) x (H) x (W)
        self.final = nn.Sequential(
                nn.ConvTranspose2d(self.nf, nc, 3, 1, 1),
                nn.Sigmoid()
                # state size. (nc) x 28 x 28
                )
        
        self.device = device

    def forward(self, z):
        d1 = self.upc1(z.view(-1, self.dim_z, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.final(d4)
        return output
    
class Label_Prior(nn.Module):
    def __init__(self, dim_z, dim_u, hidden_nodes, device='gpu'):
        super(Label_Prior, self).__init__()
        self.dim_z, self.dim_u, self.hidden_nodes = dim_z, dim_u, hidden_nodes
        
        # input dimension is dim_u
        self.main = nn.Sequential(
                        nn.Linear(self.dim_u, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Linear(self.hidden_nodes, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True)
                        )
        
        # input dimension is 20
        self.mean_net = nn.Linear(self.hidden_nodes, self.dim_z)
        self.log_var_net = nn.Linear(self.hidden_nodes, self.dim_z)
        
        self.device = device
        
    def forward(self, u_input):
        h = self.main(u_input)
        mean, log_var = self.mean_net(h), self.log_var_net(h)
        return mean, log_var
    
class Label_Decoder(nn.Module):
    def __init__(self, dim_u, dim_z, hidden_nodes, device='gpu'):
        super(Label_Decoder, self).__init__()
        self.dim_u, self.dim_z, self.hidden_nodes = dim_u, dim_z, hidden_nodes
        
        # input dimension is dim_u
        self.main = nn.Sequential(
                        nn.Linear(self.dim_z, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Linear(self.hidden_nodes, self.hidden_nodes),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True)
                        )
        
        # input dimension is 20
        self.out_net = nn.Linear(self.hidden_nodes, self.dim_u)
        
        self.device = device
        
    def forward(self, u_input):
        h = self.main(u_input)
        o = self.out_net(h)
        return o

class SparseDecoder(nn.Module):
    def __init__(self, dim_z, nc=1, nf=64, intermediate_nodes=256, device='gpu'):
        super(SparseDecoder, self).__init__()
        self.dim_z = dim_z
        self.nc = nc
        self.nf = nf
        self.intermediate_nodes = intermediate_nodes
        self.device = device
        
        self.sigma_prior_df = 3
        self.sigma_prior_scale = 1
        self.width, self.height, self.channel = 28, 28, 1
        self.input_dim = self.width * self.height * self.channel
        self.lambda0, self.lambda1 = 10.0, 0.1
        self.a, self.b = 1, self.input_dim
        
        self.p_star = nn.Parameter(0.5*torch.ones(self.input_dim, self.dim_z,
                                                  dtype=torch.float, device=device), requires_grad=False)
        self.thetas = nn.Parameter(torch.rand(self.dim_z), requires_grad=False)
        self.W = nn.Parameter(torch.randn(self.input_dim, self.dim_z), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.randn(self.input_dim))
        
        self.upc1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dim_z, self.intermediate_nodes, 1, 1, 0),
            nn.BatchNorm2d(self.intermediate_nodes),
            nn.LeakyReLU(0.2, inplace=True)
            )
        # input size is (nf*4) x (1) x (1)
        self.upc2 = nn.Sequential(
                nn.ConvTranspose2d(self.intermediate_nodes, self.nf*4, int(self.width/4), 1, 0),
                nn.BatchNorm2d(self.nf*4),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # input size is (nf*4) x (H/4) x (W/4)
        self.upc3 = conv_upsample(self.nf*4, self.nf*2)
        # input size is (nf*2) x (H/2) x (W/2)
        self.upc4 = conv_upsample(self.nf*2, self.nf)
        # input size is (nf) x (H) x (W)
        self.final = nn.Sequential(
                nn.ConvTranspose2d(self.nf, nc, 3, 1, 1),
                nn.Sigmoid()
                # state size. (nc) x 28 x 28
                )
        
    def forward(self, z, training=False):
        output = torch.zeros(z.shape[0], self.input_dim, device=self.device)
        for j in range(self.input_dim):
            masked_input = torch.mul(z.view(-1, self.dim_z), self.W[j, :]).view(-1, self.dim_z, 1, 1)
            output[:, j] = self.final(self.upc4(self.upc3(self.upc2(self.upc1(z.view(-1, self.dim_z, 1, 1)))))).view(-1, self.input_dim)[:, j]
        output = output.view(-1, self.channel, self.width, self.height)
        return output
    
'''
class SparseDecoder(nn.Module):
    def __init__(self, dim_z, ssl_intermediate_nodes=2048, device='gpu'):
        super(SparseDecoder, self).__init__()
        self.dim_z = dim_z
        self.ssl_intermediate_nodes = ssl_intermediate_nodes
        self.device = device
        
        self.sigma_prior_df = 3
        self.sigma_prior_scale = 1
        self.width, self.height, self.channel = 28, 28, 1
        self.input_dim = self.width * self.height * self.channel
        self.lambda0, self.lambda1 = 10.0, 0.1
        self.a, self.b = 1, self.input_dim
        
        self.p_star = nn.Parameter(0.5*torch.ones(self.input_dim, self.dim_z,
                                                  dtype=torch.float, device=device), requires_grad=False)
        self.thetas = nn.Parameter(torch.rand(self.dim_z), requires_grad=False)
        self.W = nn.Parameter(torch.randn(self.input_dim, self.dim_z), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.randn(self.input_dim))
        
        self.generator = nn.Sequential(nn.Linear(self.dim_z, self.ssl_intermediate_nodes, bias=False),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Linear(self.ssl_intermediate_nodes, self.ssl_intermediate_nodes),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Linear(self.ssl_intermediate_nodes, self.ssl_intermediate_nodes),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.column_means = nn.ModuleList([nn.Linear(self.ssl_intermediate_nodes, 1)
                                           for i in range(self.input_dim)])
        
    def forward(self, z):
        z = z.view(-1, self.dim_z)
        output = torch.zeros(z.shape[0], self.input_dim, device=self.device)
        for j in range(self.input_dim):
            masked_input = torch.mul(z, self.W[j, :])
            output[:, j] = torch.sigmoid(self.column_means[j](self.generator(masked_input)).squeeze())
        
        output = output.view(-1, self.channel, self.width, self.height)
        return output
'''