"""Graph AutoEncoder - Graph dataset"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """GCN layer"""
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) + self.bias
        return output


class GraphAutoEncoder(nn.Module):
    """Graph AutoEncoder"""
    
    def __init__(self, input_dim=3703, hidden_dims=[1024, 512, 256], 
                 latent_dim=16, dropout=0.5):
        super(GraphAutoEncoder, self).__init__()
        
        self.dropout = dropout
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(GraphConvolution(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.encoder_final = GraphConvolution(prev_dim, latent_dim)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            self.decoder_layers.append(GraphConvolution(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.decoder_final = GraphConvolution(prev_dim, input_dim)
    
    def forward(self, x, adj):
        # Encode
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        z = self.encoder_final(h, adj)
        
        # Decode
        h = z
        for layer in self.decoder_layers:
            h = F.relu(layer(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        x_recon = self.decoder_final(h, adj)
        
        return z, x_recon