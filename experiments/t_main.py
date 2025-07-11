# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from t_modules import DendriticLayer, VisionTransformer
from t_training import train_models_comparative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset = "fashion-mnist"  # "mnist", "fashion-mnist", or "cifar10"
n_epochs = 15
batch_size = 256
n_classes = 10
    
# Get dataset-specific parameters
if dataset == "mnist" or dataset == "fashion-mnist":
    img_size = 28
    in_dim = 28 * 28
    in_channels = 1
elif dataset == "cifar10":
    img_size = 32
    in_dim = 32 * 32 * 3
    in_channels = 3

# Dendritic model config
n_dendrite_inputs = 64
n_neurons = 512
# ViT config
patch_size = 4  # 8x8 patches for 32x32 images
embed_dim = 192
depth = 4
n_heads = 8

print(f"Creating models for {dataset.upper()} dataset...")

# 1. Dendritic Neural Network (Simplified)
model_1 = nn.Sequential(
    DendriticLayer(
        in_dim=in_dim,
        n_neurons=n_neurons,
        n_dendrite_inputs=n_dendrite_inputs,
        synaptic_resampling=False,
        percentage_resample=0.3,
        steps_to_resample=100,
    ),
    nn.LeakyReLU(),
    nn.Linear(n_neurons, n_classes),
).to(device)


model_2 = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    n_classes=n_classes,
    embed_dim=embed_dim,
    depth=depth,
    n_heads=n_heads,
    dropout=0.1,
    use_dendritic=True,
).to(device)

# 2. Standard Vision Transformer (FF layers)
model_3 = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    n_classes=n_classes,
    embed_dim=embed_dim,
    depth=depth,
    n_heads=n_heads,
    dropout=0.1,
    use_dendritic=False,
).to(device)

optimiser_1 = optim.Adam(model_1.parameters(), lr=0.001)
optimiser_2 = optim.Adam(model_2.parameters(), lr=0.001)
optimiser_3 = optim.Adam(model_3.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Configure models for comparative training
models_config = [
    [model_1, optimiser_1, "Dendritic Net"],
    # [model_2, optimiser_2, "ViT-Dendritic"],
    # [model_3, optimiser_3, "ViT"],
]

# Use the new comparative training function
results = train_models_comparative(
    models_config=models_config,
    dataset=dataset,
    criterion=criterion,
    n_epochs=n_epochs,
    batch_size=batch_size,
    verbose=True,
)