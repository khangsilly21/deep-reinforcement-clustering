"""MNIST data loader"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_mnist(config):
    """Load MNIST with train/val/test splits"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    
    train_dataset = datasets.MNIST(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )
    
    # Train/val split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    full_train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    print(f"MNIST: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_train_loader