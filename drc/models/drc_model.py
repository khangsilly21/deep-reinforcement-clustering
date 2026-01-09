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
    
    def get_cluster_assignments(self, z):
        """Get cluster assignments - Inference"""
        probs = self.decision_probability(z)
        return torch.argmax(probs, dim=1)