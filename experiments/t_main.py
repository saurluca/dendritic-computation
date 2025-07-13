# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from t_modules import DropLinear, DendriticLayer
from t_training import train_models_comparative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
dataset = "fashion-mnist"  # "mnist", "fashion-mnist", or "cifar10"
n_epochs = 20
batch_size = 128
n_classes = 10

# Get dataset-specific parameters
if dataset == "mnist" or dataset == "fashion-mnist":
    img_size = 28
    in_dim = 28 * 28
    in_channels = 1
    n_samples = 60000
elif dataset == "cifar10":
    img_size = 32
    in_dim = 32 * 32 * 3
    in_channels = 3
    n_samples = 60000

# drop linear model config
hidden_dim = 32
drop_distribution = "exponential"
steps_to_resample = 32
undo_last_mask = False

training_steps = n_samples // batch_size * n_epochs

# Dendritic model config
n_dendrite_inputs = 32
output_dim = 128

# ViT config
patch_size = 4  # 8x8 patches for 32x32 images
embed_dim = 192
depth = 4
n_heads = 8

# 1. Dendritic Neural Network (Simplified)
model_1 = nn.Sequential(
    DropLinear(
        in_dim,
        hidden_dim,
        drop_distribution=drop_distribution,
        steps_to_resample=steps_to_resample,
        target_params=8096,
        training_steps=training_steps,
    ),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LeakyReLU(),
    nn.Linear(hidden_dim, n_classes),
).to(device)

model_2 = nn.Sequential(
    nn.Linear(in_dim, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 10),
    nn.LeakyReLU(),
    nn.Linear(10, n_classes),
).to(device)

model_3 = nn.Sequential(
    DendriticLayer(
        in_dim=in_dim,
        output_dim=4096,
        n_dendrite_inputs=128,
        synaptic_resampling=False,
        percentage_resample=0.05,
        steps_to_resample=128,
    ),
    nn.LeakyReLU(),
    DendriticLayer(
        in_dim=4096,
        output_dim=128,
        n_dendrite_inputs=512,
        synaptic_resampling=False,
    ),
    nn.LeakyReLU(),
    nn.Linear(128, n_classes),
).to(device)

model_4 = nn.Sequential(
    DendriticLayer(
        in_dim=in_dim,
        output_dim=output_dim,
        n_dendrite_inputs=n_dendrite_inputs,
        synaptic_resampling=True,
        percentage_resample=0.33,
        steps_to_resample=64,
    ),
    nn.LeakyReLU(),
    DendriticLayer(
        in_dim=output_dim,
        output_dim=output_dim,
        n_dendrite_inputs=n_dendrite_inputs,
        synaptic_resampling=False,
    ),
    nn.LeakyReLU(),
    nn.Linear(output_dim, n_classes),
).to(device)

print(model_1)


optimiser_1 = optim.AdamW(model_1.parameters(), lr=0.001, weight_decay=0.001)
optimiser_2 = optim.AdamW(model_2.parameters(), lr=0.001, weight_decay=0.001)
optimiser_3 = optim.AdamW(model_3.parameters(), lr=0.001, weight_decay=0.001)
optimiser_4 = optim.AdamW(model_4.parameters(), lr=0.001, weight_decay=0.001)

criterion = nn.CrossEntropyLoss()

# Configure models for comparative training
models_config = [
    # [model_1, optimiser_1, "DropLinear"],
    [model_3, optimiser_3, "Dendritic"],
    # [model_4, optimiser_4, "Dendritic w/ resampling"],
    # [model_2, optimiser_2, "Linear"],
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

print(model_1[0].num_active_params())

print(model_1[0].num_params())

# %%

print(model_1[4].num_active_params())

print(model_1[0].num_params())
print(model_1.num_active_params())

# old

# model_2 = nn.Sequential(
#     DendriticLayer(
#         in_dim=in_dim,
#         output_dim=output_dim,
#         n_dendrite_inputs=n_dendrite_inputs,
#         synaptic_resampling=False,
#         percentage_resample=0.2,
#         steps_to_resample=128,
#     ),
#     nn.BatchNorm1d(output_dim),
#     nn.LeakyReLU(),
#     DendriticLayer(
#         in_dim=output_dim,
#         output_dim=output_dim,
#         n_dendrite_inputs=n_dendrite_inputs,
#         synaptic_resampling=False,
#         percentage_resample=0.3,
#         steps_to_resample=128,
#     ),
#     nn.BatchNorm1d(output_dim),
#     nn.LeakyReLU(),
#     nn.Linear(output_dim, n_classes),
# ).to(device)

# # 2. Standard Vision Transformer (FF layers)
# model_3 = VisionTransformer(
#     img_size=img_size,
#     patch_size=patch_size,
#     in_channels=in_channels,
#     n_classes=n_classes,
#     embed_dim=embed_dim,
#     depth=depth,
#     n_heads=n_heads,
#     dropout=0.1,
#     use_dendritic=False,
# ).to(device)
