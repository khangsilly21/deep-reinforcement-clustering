
"""
Graph AutoEncoder for DRC on Citeseer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer
    GCN Layer: H' = σ(D^{-1/2} A D^{-1/2} H W)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphAutoEncoder(nn.Module):
    """
    Graph AutoEncoder for Citeseer/Cora datasets
    
    Encoder: 
        Input (3703 for Citeseer) → GCN(512) → GCN(256) → Linear(latent_dim)
    
    Decoder:
        latent_dim → Linear(256) → Linear(512) → Linear(3703)
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=16, dropout=0.1):
        super(GraphAutoEncoder, self).__init__()
        
        self.dropout = dropout
        
        # encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(GraphConvolution(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Final encoding layer 
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder_gcns = nn.ModuleList(encoder_layers[:-1])  # GCN layers
        self.encoder_final = encoder_layers[-1]  # Final linear layer
        
        # decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x, adj):
        h = x
        
        for gcn in self.encoder_gcns:
            h = gcn(h, adj)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.encoder_final(h)
        
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, adj):
        z = self.encode(x, adj)
        x_recon = self.decode(z)
        return z, x_recon