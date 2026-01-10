"""Configuration for CITE dataset"""
class CITEConfig:
    DATASET_NAME = 'CITE'
    
    # Graph 
    N_NODES = 3312
    N_FEATURES = 3703
    N_CLUSTERS = 6
    
    # Model architecture (Graph Autoencoder)
    INPUT_DIM = 3703
    HIDDEN_DIMS = [1024, 512, 256]
    LATENT_DIM = 16
    
    # Training parameters
    BATCH_SIZE = None  # Full
    PRETRAIN_EPOCHS = 200
    TRAIN_EPOCHS = 400
    LEARNING_RATE = 0.001
    PRETRAIN_LR = 0.001
    
    # DRC parameters
    GAMMA = 0.01
    V = 100.0
    
    # Graph-specific
    DROPOUT = 0.5
    
    # Data
    DATA_PATH = './data'
    
    # Misc
    RANDOM_SEED = 42
    DEVICE = 'cuda'
    SAVE_DIR = './results'