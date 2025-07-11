# %%
try:
    import cupy as cp
    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")
from modules import (
    Adam,
    CrossEntropy,
    LeakyReLU,
    Sequential,
    DendriticLayer,
    LinearLayer,
    BatchNormalization,
    Dropout,
)
from training import train_models

# for repoducability
cp.random.seed(1223)

# data config
dataset = "cifar10"  # "mnist", "fashion-mnist", "cifar10"
subset_size = None
data_augmentation = True

# config
n_epochs = 50 # 15 MNIST, 20 Fashion-MNIST
lr = 0.002  # 0.003
weight_decay = 0.01  # 0.001
batch_size = 256
n_classes = 10

if dataset in ["mnist", "fashion-mnist"]:
    in_dim = 28 * 28  # Image dimensions (28x28 MNIST, 32x32x3 CIFAR-10)
elif dataset == "cifar10":
    in_dim = 32 * 32 * 3
else:
    raise ValueError(f"Invalid dataset: {dataset}")

# dendriticmodel config
n_dendrite_inputs = 64  # 31 / 128
n_dendrites = 512  # 23 / 6
n_neurons = 64 # 10 / 10
strategy = "random"  # ["random", "local-receptive-fields", "fully-connected"]

# model to compare 
model_1 = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_dendrites=n_dendrites,
            n_dendrite_inputs=n_dendrite_inputs,
            soma_enabled=False,
            synaptic_resampling=True,
            percentage_resample=0.2,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        DendriticLayer(
            n_dendrites,
            n_dendrites=n_dendrites,
            n_dendrite_inputs=n_dendrite_inputs,
            soma_enabled=False,
            synaptic_resampling=False,
            percentage_resample=0.2,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        LinearLayer(n_dendrites, n_classes),
    ]
)
# baseline dANN
model_2 = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_dendrites=n_dendrites,
            n_dendrite_inputs=n_dendrite_inputs,
            soma_enabled=False,
            synaptic_resampling=True,
            percentage_resample=0.2,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        DendriticLayer(
            n_dendrites,
            n_dendrites=n_dendrites,
            n_dendrite_inputs=n_dendrite_inputs,
            soma_enabled=False,
            synaptic_resampling=True,
            percentage_resample=0.2,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        LinearLayer(n_dendrites, n_classes),
    ]
)
# baseline vANN
model_3 = Sequential(
    [
        LinearLayer(in_dim, 23, bias=False),
        LeakyReLU(),
        LinearLayer(23, 15, bias=False),
        LeakyReLU(),        
        LinearLayer(15, n_classes),
    ]
)



criterion = CrossEntropy()
optimiser = Adam(model_1.params(), criterion, lr=lr, weight_decay=weight_decay)
b_optimiser = Adam(model_2.params(), criterion, lr=lr, weight_decay=weight_decay)
v_optimiser = Adam(model_3.params(), criterion, lr=lr, weight_decay=weight_decay)

models_config = [
    [model_1, optimiser, "dANN"],
    # [model_2, b_optimiser, "dANN, b-norm"],
    [model_3, v_optimiser, "vANN"],
]

results=train_models(
    models_config,
    dataset,
    criterion,
    n_epochs,
    data_augmentation=data_augmentation,
    batch_size=batch_size,
)

