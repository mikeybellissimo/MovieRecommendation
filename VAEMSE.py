from torch import nn 
import torch

# This is a standard VAE but using MSE as the reconstruction term. 

#Actually just rename this out and make loss function a parameter
class VAEMSE(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.encoder = nn.Sequential(
            #Encoder
            nn.Linear(1682, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )
        self.decoder = nn.Sequential(
            #Decoder
            nn.Linear(512,1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024,1682),
            nn.Tanh()
        )
        # distribution parameters
        self.fc_mu = nn.Linear(1024, 512)
        self.fc_var = nn.Linear(1024, 512)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        #for evaluation purposes
        self.test_mse = 1000
        self.mse_loss_fcn = nn.MSELoss()
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing data under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=1)
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        
        #perform the kernel trick to allow for backprop through sampling
        
        epsilon = torch.distributions.Normal(0, 1).rsample()
        z = mu + epsilon * std
        # decoded
        ratings = self.decoder(z) * 2
        return ratings, z, mu, std
    
    def vae_loss(self, x_hat, x, z, mu, std):
        # reconstruction loss. This is now MSE instead of gaussian likelihood
        #indies = x.nonzero().split(1, dim=1)
        
        #recon_loss = ((x[indies] - x_hat[indies]) ** 2).sum() 
        recon_loss = self.mse_loss_fcn(x_hat, x) * 10000#self.gaussian_likelihood(x_hat, self.log_scale, x)
        
        # kl
        kl = self.kl_divergence(z, mu, std)
        # elbo
        elbo = (kl + recon_loss).mean()

        return elbo
