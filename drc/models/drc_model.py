"""Deep Reinforcement Clustering"""
import numpy as np
import torch
import torch.nn as nn


class DRC(nn.Module):
    """Deep Reinforcement Clustering"""
    
    def __init__(self, autoencoder, n_clusters, latent_dim):
        super(DRC, self).__init__()
        
        self.autoencoder = autoencoder
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
    
    def forward(self, *args):
        z, x_recon = self.autoencoder(*args)
        return z, x_recon
    
    def cauchy_similarity(self, z, kappa=1.0):
        """Cauchy similarity"""
        z_expanded = z.unsqueeze(1)
        centers_expanded = self.cluster_centers.unsqueeze(0)
        distances_sq = torch.sum((z_expanded - centers_expanded) ** 2, dim=2)
        similarities = (1.0 / np.pi) * (kappa / (distances_sq + kappa**2))
        return similarities
    
    def decision_probability(self, z, kappa=1.0):
        """Decision probabilities"""
        similarities = self.cauchy_similarity(z, kappa)
        probs = torch.sigmoid(similarities)
        return probs
    
    def compute_loss(self, x, z, x_recon, gamma=0.01, v=100.0):
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        
        probs = self.decision_probability(z) 
        max_probs, _ = torch.max(probs, dim=1)
        p_random = torch.rand(max_probs.size(), device=z.device)
        y_ij = (max_probs > p_random).float().detach()
        
        rewards = (v * (2 * y_ij - 1)).detach()
        
        log_prob = y_ij * torch.log(max_probs) + \
                   (1 - y_ij) * torch.log(1 - max_probs)
        
        rc_loss = -gamma * torch.mean(rewards * log_prob)
        
        total_loss = recon_loss + rc_loss
        
        return total_loss, recon_loss, rc_loss
    
    def get_cluster_assignments(self, z):
        """Get cluster assignments - Inference"""
        probs = self.decision_probability(z)
        return torch.argmax(probs, dim=1)