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

from matplotlib import pyplot as plt
from modules import (
    Adam,
    CrossEntropy,
    LeakyReLU,
    Sequential,
    DendriticLayer,
    LinearLayer,
)
from utils import load_mnist_data, load_cifar10_data
from training import (
    train_models,
    plot_dendritic_weights,
    plot_dendritic_weights_single_image,
    print_network_entropy,
    plot_dendritic_weights_full_model,
)

# for repoducability
cp.random.seed(1223)

# data config
dataset = "fashion-mnist"  # "mnist", "fashion-mnist", "cifar10"
subset_size = None

# config
n_epochs = 20  # 15 MNIST, 20 Fashion-MNIST
lr = 0.0005  # 0.003
v_lr = 0.0005  # 0.015 - SGD
b_lr = 0.0005  # 0.015 - SGD
weight_decay = 0.01  # 0.001
batch_size = 256
n_classes = 10

if dataset in ["mnist", "fashion-mnist"]:
    in_dim = 28 * 28  # Image dimensions (28x28 MNIST, 32x32x3 CIFAR-10)
elif dataset == "cifar10":
    in_dim = 32 * 32 * 3
else:
    raise ValueError(f"Invalid dataset: {dataset}")

# cifar10: 32 / 18 / 49, equal to input / 10 / 10

# dendriticmodel config
n_dendrite_inputs = 16  # 31 / 128
n_dendrites = 32  # 23 / 6
n_dendrites_without_soma = 200 # 57 / 106
n_neurons = 10 # 10 / 10
strategy = "random"  # ["random", "local-receptive-fields", "fully-connected"]

if dataset in ["mnist", "fashion-mnist"]:
    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset=dataset, subset_size=subset_size
    )
elif dataset == "cifar10":
    X_train, y_train, X_test, y_test = load_cifar10_data(subset_size=subset_size)


model = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_neurons,
            n_dendrite_inputs=45,
            n_dendrites=285,
            strategy=strategy,
            soma_enabled=False,
            synaptic_resampling=False,
            percentage_resample=0.25,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        LinearLayer(285, n_classes),
        # LeakyReLU(),
        # LinearLayer(n_classes, n_classes),
    ]
)
# baseline dendritic model
b_model = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_neurons,
            n_dendrite_inputs=45,
            n_dendrites=285,
            strategy=strategy,
            soma_enabled=False,
            synaptic_resampling=True,
            percentage_resample=0.25,
            steps_to_resample=128,
        ),
        LeakyReLU(),
        LinearLayer(285, n_classes),
        # LeakyReLU(),
        # LinearLayer(n_classes, n_classes),
    ]
)
# vanilla model
v_model = Sequential(
    [
        LinearLayer(in_dim, 20),
        LeakyReLU(),
        LinearLayer(20, 10),
        LeakyReLU(),
        LinearLayer(10, n_classes),
    ]
)

criterion = CrossEntropy()
optimiser = Adam(model.params(), criterion, lr=lr, weight_decay=weight_decay)
b_optimiser = Adam(b_model.params(), criterion, lr=b_lr, weight_decay=weight_decay)
v_optimiser = Adam(v_model.params(), criterion, lr=v_lr, weight_decay=weight_decay)

models_config = [
    [model, optimiser, "dANN, no soma, without ReLU"],
    [b_model, b_optimiser, "dANN, no soma, with ReLU"],
    [v_model, v_optimiser, "vANN"],
]

results=train_models(
    models_config,
    X_train,
    y_train,
    X_test,
    y_test,
    criterion,
    n_epochs=n_epochs,
    batch_size=batch_size,
)

# # Visualize the weights of the first neuron in the dendritic model
# # print("\nVisualizing dendritic weights for the first neuron of the dendritic model...")
# # # plot_dendritic_weights(model, X_test[0], neuron_idx=0)
# # plot_dendritic_weights_single_image(model, X_test[0], neuron_idx=0)
# # print("Vanilla model")
# # for i in range(10):
#     # print(i)
#     # plot_dendritic_weights_single_image(model, X_test[0], neuron_idx=i)

# # %%

# plot_dendritic_weights_full_model(model, X_test[0])
# # %%
# idx = 1
# sample_1 = X_test[idx]
# true_label = cp.argmax(y_test[idx])
# predictions = model(sample_1)

# predicted_label = cp.argmax(predictions)
# print(f"raw prediction: {predictions}")
# print(f"sample 1 predicted label: {predicted_label}, true label: {true_label}")

# plot_dendritic_weights(model, X_test[idx], neuron_idx=true_label)
# plot_dendritic_weights_single_image(model, X_test[idx], neuron_idx=true_label)


# if __name__ == "__main__":
# main()
