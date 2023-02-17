import torch
import numpy as np

def compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var):
    # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
    post_mean = (z_mean/(1+torch.exp(z_log_var-lam_log_var))) + (lam_mean/(1+torch.exp(lam_log_var-z_log_var)));
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var));
    
    return post_mean, post_log_var

def sampling(mean, log_var):
    device = mean.get_device()
    if device == -1:
        epsilon = torch.randn(mean.shape)
        return mean + torch.exp(0.5 * log_var) * epsilon
    else:
        epsilon = torch.randn(mean.shape).cuda()
        return mean + torch.exp(0.5 * log_var).cuda() * epsilon

# Ref: https://github.com/edenton/svg/blob/master/train_svg_lp.py#L131-#L138
def kl_criterion(mu1, log_var1, mu2, log_var2):
    sigma1 = log_var1.mul(0.5).exp() 
    sigma2 = log_var2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(log_var1) + (mu1 - mu2)**2)/(2*torch.exp(log_var2)) - 1/2
    return torch.mean(kld, dim=-1)
