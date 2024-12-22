import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
from torch.autograd import Variable
import torch.optim as optim

# --------------------------
# NETWORK BUILDING BLOCK
# --------------------------

class GaussianLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.main_transform = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, out_features)
        self.log_std_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        hidden = self.main_transform(x)
        z_mean = self.mean_layer(hidden)
        z_log_std = self.log_std_layer(hidden)
        z_std = torch.exp(z_log_std)
        return z_mean, z_std



class IWAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, feature_dim):
        super().__init__()
        
        # Encoder layers
        self.encode_gauss_1 = GaussianLayer(in_features=feature_dim, hidden_dim=hidden_dim, out_features=hidden_dim)
        self.encode_gauss_2 = GaussianLayer(in_features=hidden_dim, hidden_dim=latent_dim, out_features=latent_dim)
        
        # Decoder layers
        self.decode_gauss_1 = GaussianLayer(in_features=latent_dim, hidden_dim=latent_dim, out_features=latent_dim)
        self.decode_sigmoid = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        # First layer
        mu_q1, std_q1 = self.encode_gauss_1(x)
        h1 = dist.Normal(mu_q1, std_q1).rsample()
        
        # Second layer
        mu_q2, std_q2 = self.encode_gauss_2(h1)
        h2 = dist.Normal(mu_q2, std_q2).rsample()
        
        return (h1, mu_q1, std_q1), (h2, mu_q2, std_q2)

    def decode(self, h2):
        # p(h1 | h2) from the top-down pass
        mu_p1, std_p1 = self.decode_gauss_1(h2)
        h1 = dist.Normal(mu_p1, std_p1).rsample()

        # p(x | h1)
        mu_p0 = self.decode_sigmoid(h1)
        return (h1, mu_p1, std_p1), mu_p0

    def forward(self, x):
        (h1_q, mu_q1, std_q1), (h2, mu_q2, std_q2) = self.encode(x)
        (h1_p, mu_p1, std_p1), mu_p0 = self.decode(h2)
        return ((h1_q, mu_q1, std_q1), (h2, mu_q2, std_q2)), (h1_p, mu_p1, std_p1, mu_p0)

    def log_likelihood(self, x):
        ((h1_q, mu_q1, std_q1), (h2, mu_q2, std_q2)), (h1_p, mu_p1, std_p1, mu_p0) = self.forward(x)

        # Q(h1 | x), Q(h2 | h1)
        log_qh1_given_x = dist.Normal(mu_q1, std_q1).log_prob(h1_q).sum(-1)
        log_qh2_given_h1 = dist.Normal(mu_q2, std_q2).log_prob(h2).sum(-1)


        # p(h2) = Normal(0,I)
        log_ph2 = dist.Normal(0, 1).log_prob(h2).sum(-1)
        # p(h1 | h2) = Normal(mu_p1, std_p1)
        log_ph1_given_h2 = dist.Normal(mu_p1, std_p1).log_prob(h1_p).sum(-1)
        # p(x | h1) = Bernoulli(mu_p0)
        log_px_given_h1 = dist.Bernoulli(mu_p0).log_prob(x).sum(-1)


        return log_ph2 + log_ph1_given_h2 + log_px_given_h1 - log_qh1_given_x - log_qh2_given_h1

    def compute_loss(self, x, k, model):


        if model == "IWAE":
            x = x.expand(k, x.size(0), 784)

            log_w = self.log_likelihood(x)

            # Stabilize for numerical safety
            log_w_stab = log_w - torch.max(log_w, dim=0)[0]
            w = torch.exp(log_w_stab)
            normalized_w = w / torch.sum(w, dim=0, keepdim=True)
            normalized_w = normalized_w.detach()  # no gradient through weights

            # Weighted average
            loss = -torch.mean(torch.sum(normalized_w * (log_w), dim=0))
        else:
            x = x.repeat(k, 1).expand(1, x.size(0) * k, 784)            
            log_w = self.log_likelihood(x)

            w = torch.exp(log_w)
            # IWAE loss: -E[log( avg(w) )]
            loss = -torch.mean(torch.log(torch.mean(w, dim=0)))

        return loss