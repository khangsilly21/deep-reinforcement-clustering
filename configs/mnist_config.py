"""Config for MNIST dataset"""
class MNISTConfig:
    DATASET_NAME = 'MNIST'
    
    # Model architecture (IDEC)
    INPUT_DIM = 784
    HIDDEN_DIMS = [500, 500, 2000]
    LATENT_DIM = 10
    N_CLUSTERS = 10
    
    # Training parameters
    BATCH_SIZE = 256
    PRETRAIN_EPOCHS = 50
    TRAIN_EPOCHS = 200
    LEARNING_RATE = 0.0001
    PRETRAIN_LR = 0.001
    
    # DRC parameters
    GAMMA = 0.01
    V = 100.0
    
    # Data
    DATA_PATH = './data'
    
    # Misc
    RANDOM_SEED = 42
    DEVICE = 'cuda'
    SAVE_DIR = './results'