"""MNIST Training Script"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

from configs.mnist_config import MNISTConfig
from drc.data.mnist_loader import load_mnist
from drc.models.autoencoder import AutoEncoder
from drc.models.drc_model import DRC
from drc.utils.metrics import evaluate_model
from drc.utils.visualization import visualize_training_results


def pretrain(model, loader, device, epochs, lr):
    print("\n=== Pretraining ===")
    optimizer = optim.Adam(model.autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            z, x_recon = model.autoencoder(data)
            loss = criterion(x_recon, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    print("Pretraining done!")


def init_centers(model, loader, device):
    print("\n=== Initializing Centers ===")
    model.eval()
    all_z = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            z, _ = model(data)
            all_z.append(z.cpu())
    
    all_z = torch.cat(all_z, dim=0).numpy()
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=20, random_state=42)
    kmeans.fit(all_z)
    model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    print("Centers initialized!")


def train_drc(model, train_loader, val_loader, device, epochs, lr, gamma, v):
    print("\n=== Training DRC ===")
    
    optimizer_network = optim.Adam(model.autoencoder.parameters(), lr=lr)
    optimizer_centers = optim.SGD([model.cluster_centers], lr=lr*10, momentum=0.9)
    criterion = nn.MSELoss()
    
    history = {'loss': [], 'recon_loss': [], 'rc_loss': [], 'val_acc': [], 'val_nmi': [], 'val_ari': []}
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_rc = 0
        
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            
            optimizer_network.zero_grad()
            optimizer_centers.zero_grad()
            
            z, x_recon = model(data)
            loss, L_rec, L_rc = model.compute_loss(data, z, x_recon, gamma, v)
            loss.backward()
            optimizer_centers.step()
            optimizer_network.step()
            
            total_loss += loss.item()
            total_recon += L_rec.item()
            total_rc += L_rc.item()
        
        history['loss'].append(total_loss / len(train_loader))
        history['recon_loss'].append(total_recon / len(train_loader))
        history['rc_loss'].append(total_rc / len(train_loader))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate_model(model, val_loader, device)
            history['val_acc'].append(metrics['ACC'])
            history['val_nmi'].append(metrics['NMI'])
            history['val_ari'].append(metrics['ARI'])
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f}, "
                  f"ACC: {metrics['ACC']:.4f}, NMI: {metrics['NMI']:.4f}")
            
            if metrics['ACC'] > best_acc:
                best_acc = metrics['ACC']
    
    return history


def main():
    config = MNISTConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print(f"Device: {device}")
    print(f"Config: {config.INPUT_DIM} -> {config.HIDDEN_DIMS} -> {config.LATENT_DIM}")
    
    save_dir = Path(config.SAVE_DIR) / config.DATASET_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, test_loader, full_train_loader = load_mnist(config)
    
    autoencoder = AutoEncoder(config.INPUT_DIM, config.HIDDEN_DIMS, config.LATENT_DIM)
    model = DRC(autoencoder, config.N_CLUSTERS, config.LATENT_DIM).to(device)
    
    pretrain(model, full_train_loader, device, config.PRETRAIN_EPOCHS, config.PRETRAIN_LR)
    init_centers(model, full_train_loader, device)
    history = train_drc(model, train_loader, val_loader, device, config.TRAIN_EPOCHS, 
                       config.LEARNING_RATE, config.GAMMA, config.V)
    
    print("\n=== Test Evaluation ===")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"ACC: {test_metrics['ACC']:.4f}, NMI: {test_metrics['NMI']:.4f}, ARI: {test_metrics['ARI']:.4f}")
    
    visualize_training_results(history, test_metrics, save_dir)
    torch.save(model.state_dict(), save_dir / 'model.pth')
    
    return model, history, test_metrics


if __name__ == "__main__":
    model, history, metrics = main()