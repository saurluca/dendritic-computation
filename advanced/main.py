import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from modules import DendriticLayer, register_dendritic_hooks
from data import load_mnist_data, get_sample_image
from utils import (
    calculate_eigenvalues,
    plot_dendritic_weights_single_image,
    plot_dendritic_weights_full_model,
)
from training import compare_models


def create_models(in_dim, n_classes, device):
    """
    Create the three models for comparison.

    Args:
        in_dim: Input dimension
        n_classes: Number of output classes
        device: Device to place models on

    Returns:
        tuple: (model_1, model_2, model_3, model_names)
    """
    # Model configurations
    n_dendrite_inputs = 32
    n_dendrites = 23
    n_neurons = 10
    hidden_dim = 10

    # Model 1: Synaptic Resampling Dendritic Model
    model_1 = nn.Sequential(
        DendriticLayer(
            in_dim=in_dim,
            n_neurons=n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
            synaptic_resampling=True,
            percentage_resample=0.1,
            steps_to_resample=128,
        ),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(n_neurons, n_classes),
    ).to(device)

    # Register hooks for synaptic resampling
    register_dendritic_hooks(model_1[0])

    # Model 2: Base Dendritic Model (no resampling)
    model_2 = nn.Sequential(
        DendriticLayer(
            in_dim=in_dim,
            n_neurons=n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
            synaptic_resampling=False,
        ),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(n_neurons, n_classes),
    ).to(device)

    # Model 3: Vanilla ANN
    model_3 = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    model_names = ["Synaptic Resampling", "Base Dendritic", "Vanilla ANN"]

    return (model_1, model_2, model_3), model_names



# Set random seed for reproducibility
torch.manual_seed(1287311233)
np.random.seed(1287311233)

# Configuration
dataset = "fashion-mnist"  # "mnist" or "fashion-mnist"
n_epochs = 20
lr = 0.001
batch_size = 128
in_dim = 28 * 28
n_classes = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create models
print("\nCreating models...")
models, model_names = create_models(in_dim, n_classes, device)
model_1, model_2, model_3 = models

# Calculate eigenvalues before training
print("\n" + "=" * 50)
print("EIGENVALUES BEFORE TRAINING")
print("=" * 50)
for model, name in zip(models, model_names):
    calculate_eigenvalues(model, name)

# Create optimizers
optimizers = [
    optim.Adam(model_1.parameters(), lr=lr),
    optim.Adam(model_2.parameters(), lr=lr),
    optim.Adam(model_3.parameters(), lr=lr),
]

# Prepare models config for compare_models
models_config = [
    (model_1, optimizers[0], model_names[0]),
    (model_2, optimizers[1], model_names[1]),
    (model_3, optimizers[2], model_names[2]),
]

# Train and compare all models
results = compare_models(
    models_config=models_config,
    dataset=dataset,
    n_epochs=n_epochs,
    batch_size=batch_size,
    device=device,
    plot_results=True,
    verbose=True
)
