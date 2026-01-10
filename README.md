# Deep Reinforcement Clustering (DRC)

PyTorch implementation of **"Deep Reinforcement Clustering"** (IEEE Transactions on Multimedia, 2023) by Li et al.

Implementation for **MNIST** (image dataset) and **CITE** (citation graph dataset).

---

## 🎯 Overview

**Deep Reinforcement Clustering (DRC)** reformulates clustering as a Markov Decision Process, learning an adaptive partition policy through reinforcement learning instead of using pre-defined strategies.

### Key Features

- **Adaptive Clustering**: Learns clustering policy through environment interaction
- **Cauchy Similarity**: Heavy-tailed distribution for accurate structure measurement
- **Bernoulli Action**: Balances exploration and exploitation
- **Two Architectures**:
  - Standard autoencoder (IDEC) for images
  - Graph autoencoder (GCN) for citation networks

### Paper Citation

```bibtex
@article{li2023deep,
  title={Deep Reinforcement Clustering},
  author={Li, Peng and Gao, Jing and Zhang, Jianing and Jin, Shan and Chen, Zhikui},
  journal={IEEE Transactions on Multimedia},
  volume={25},
  pages={8183--8193},
  year={2023},
  publisher={IEEE}
}
```

---

## 🔧 Installation

### Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib networkx tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 🚀 Quick Start

### MNIST 

```bash
python experiments/train_mnist.py
```

**Expected output:**
```
Device: cuda
MNIST: Train=54000, Val=6000, Test=10000
=== Pretraining ===
...
=== Training DRC ===
...
=== Test Evaluation ===
ACC: 0.XXX, NMI: 0.XXX, ARI: 0.XXX
```

### CITE (Download citeseer dataset yourself) - In Progress

1. **Download dataset:**
   - Get `citeseer.npz` from [Awesome-Deep-Graph-Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering)
   - Place it in folder drc/data
2. **Train:**
```bash
python experiments/train_cite.py
```

**Expected output:**
```
Device: cuda
CITE: Nodes=3312, Features=3703, Edges=4732, Classes=6
=== Pretraining ===
...
=== Training DRC ===
...
=== Final Evaluation ===
ACC: 0.XXX, NMI: 0.XXX, ARI: 0.XXX
```

---

## ⚙️ Configuration

### MNIST Configuration

Edit `configs/mnist_config.py`:

```python
class MNISTConfig:
    # Model architecture
    INPUT_DIM = 784
    HIDDEN_DIMS = [500, 500, 2000]
    LATENT_DIM = 10
    N_CLUSTERS = 10
    
    # Training
    BATCH_SIZE = 256
    PRETRAIN_EPOCHS = 50
    TRAIN_EPOCHS = 200
    LEARNING_RATE = 0.0001
    
    # DRC parameters
    GAMMA = 0.01      # Sample-average weight
    V = 100.0         # Reward constant
```

### CITE Configuration

Edit `configs/cite_config.py`:

```python
class CITEConfig:
    # Model architecture
    INPUT_DIM = 3703
    HIDDEN_DIMS = [1024, 512, 256]
    LATENT_DIM = 16
    N_CLUSTERS = 6
    
    # Training
    BATCH_SIZE = None  # Full graph
    PRETRAIN_EPOCHS = 200
    TRAIN_EPOCHS = 400
    LEARNING_RATE = 0.001
    
    # DRC parameters
    GAMMA = 0.01
    V = 100.0
    DROPOUT = 0.5     # For GCN layers
```

---

## 🎓 Training

### MNIST Training Process

```
1. Load MNIST dataset
   ├── Flatten images to 784-dim
   └── Split: train (54K), val (6K), test (10K)

2. Create model
   ├── AutoEncoder: 784→500→500→2000→10
   └── DRC with 10 cluster centers

3. Pretrain autoencoder (50 epochs)
   └── Minimize reconstruction loss

4. Initialize cluster centers
   └── K-means on latent embeddings

5. Train DRC (200 epochs)
   ├── Reconstruction loss
   ├── Reinforcement clustering loss
   └── Update network + cluster centers

6. Evaluate on test set
```


### CITE Training Process -- In Progress (Planning flow)

```
1. Load CITE graph
   ├── Node features: 3703-dim
   ├── Adjacency matrix: 3312×3312
   └── Normalize features and adjacency

2. Create model
   ├── GraphAutoEncoder: 3703→1024→512→256→16
   └── DRC with 6 cluster centers

3. Pretrain graph autoencoder (200 epochs)
   └── Minimize reconstruction loss 

4. Initialize cluster centers
   └── K-means on graph embeddings

5. Train DRC (400 epochs)
   ├── Process entire graph (no batching)
   ├── Reconstruction loss
   ├── Reinforcement clustering loss
   └── Update network + cluster centers

6. Evaluate on all nodes
```



---


## 🔬 Advanced Usage

### Using Trained Models

```python
import torch
from drc.models.autoencoder import AutoEncoder
from drc.models.drc_model import DRC

# Load trained model
autoencoder = AutoEncoder(784, [500, 500, 2000], 10)
model = DRC(autoencoder, n_clusters=10, latent_dim=10)
model.load_state_dict(torch.load('results/MNIST/model.pth'))
model.eval()

# Predict clusters
with torch.no_grad():
    z, _ = model(data)
    clusters = model.get_cluster_assignments(z)
```

### Modifying Hyperparameters

```python
# Edit config file or pass directly
config = MNISTConfig()
config.LEARNING_RATE = 0.0002
config.GAMMA = 0.02
config.V = 150.0
```

### Custom Dataset

To add a new dataset:

1. Create config file: `configs/my_dataset_config.py`
2. Create data loader: `drc/data/my_dataset_loader.py`
3. Create training script: `experiments/train_my_dataset.py`
4. Use appropriate autoencoder (standard or graph)

---

## 📖 References

1. Li et al., "Deep Reinforcement Clustering", IEEE TMM 2023
2. Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016 (DEC)
3. Guo et al., "Improved Deep Embedded Clustering with Local Structure Preservation", IJCAI 2017 (IDEC)
4. Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017 (GCN)

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- Based on the DRC paper (IEEE TMM 2023)
- IDEC architecture for image datasets
- GCN architecture for graph datasets
- MNIST dataset from torchvision
- CITE dataset from Awesome-Deep-Graph-Clustering

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---
