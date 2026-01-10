"""Clustering evaluation metrics"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_accuracy(y_true, y_pred):
    """Clustering accuracy with Hungarian algorithm"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate_clustering(y_true, y_pred):
    """Evaluate with ACC, NMI, ARI"""
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return {'ACC': acc, 'NMI': nmi, 'ARI': ari}


# Image Datasets Specific Functions
def predict_clusters(model, data_loader, device):
    """Predict cluster assignments for dataset"""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(data.size(0), -1).to(device)
            z, _ = model(data)
            preds = model.get_cluster_assignments(z)
            
            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
    
    return np.concatenate(all_labels), np.concatenate(all_preds)


def evaluate_model(model, data_loader, device):
    """Evaluate model on dataset"""
    y_true, y_pred = predict_clusters(model, data_loader, device)
    return evaluate_clustering(y_true, y_pred)

# Graph Datasets Specific Functions