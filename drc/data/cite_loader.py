"""CITE (Citeseer) data loader"""
import os

import numpy as np
import torch


def load_cite_simple(path='./data/citeseer'):
    """Load CITE dataset from .npz file"""
    data = {}
    
    feature_path = os.path.join(path, 'citeseer_feat.npy')
    adj_path = os.path.join(path, 'citeseer_adj.npy')
    label_path = os.path.join(path, 'citeseer_label.npy')
    
    if not os.path.exists(path) or not os.path.exists(feature_path) \
       or not os.path.exists(adj_path) or not os.path.exists(label_path):
        raise FileNotFoundError(
            f"{path} not found. Download citeseer from: "
            "https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering"
        )
        
    data['features'] = np.load(feature_path)
    data['adj'] = np.load(adj_path)
    data['labels'] = np.load(label_path)
    
    if hasattr(data['features'], 'toarray'):
        features = torch.FloatTensor(data['features'].toarray())
    else:
        features = torch.FloatTensor(data['features'])
    
    if hasattr(data['adj'], 'toarray'):
        adj = torch.FloatTensor(data['adj'].toarray())
    else:
        adj = torch.FloatTensor(data['adj'])
    
    labels = torch.LongTensor(data['labels'])
    
    print(f"CITE: Nodes={features.shape[0]}, Features={features.shape[1]}, "
          f"Edges={int(adj.sum().item()/2)}, Classes={labels.max().item()+1}")
    
    return features, adj, labels


def preprocess_cite_data(features, adj, labels):
    """Preprocess CITE data"""
    
    # Normalize 
    rowsum = features.sum(1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    
    # Normalize adjacency: add self-loops + D^{-1/2} A D^{-1/2}
    adj = adj + torch.eye(adj.size(0))
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return features, adj_norm, labels