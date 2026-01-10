"""Visualization utilities"""
from pathlib import Path

import matplotlib.pyplot as plt


def visualize_training_results(history, test_metrics, save_dir, save_name='training_results.png'):
    """Visualize training curves and metrics"""
    
    fig = plt.figure(figsize=(15, 5))
    
    # Loss curves
    ax1 = plt.subplot(1, 3, 1)
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], label='Total Loss', linewidth=2)
    ax1.plot(epochs, history['recon_loss'], label='Recon Loss', linewidth=2)
    ax1.plot(epochs, history['rc_loss'], label='RC Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation metrics
    ax2 = plt.subplot(1, 3, 2)
    val_epochs = [1] + list(range(10, len(history['loss']) + 1, 10))
    ax2.plot(val_epochs, history['val_acc'], label='ACC', linewidth=2, marker='o')
    ax2.plot(val_epochs, history['val_nmi'], label='NMI', linewidth=2, marker='s')
    ax2.plot(val_epochs, history['val_ari'], label='ARI', linewidth=2, marker='^')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Test metrics
    ax3 = plt.subplot(1, 3, 3)
    names = list(test_metrics.keys())
    values = list(test_metrics.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax3.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Score')
    ax3.set_title('Test Performance')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to '{save_path}'")
    plt.close()