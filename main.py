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
from modules import Adam, CrossEntropy, LeakyReLU, Sequential, DendriticLayer, LinearLayer
from utils import load_mnist_data, load_cifar10_data
from training import (
    compare_models,
    plot_dendritic_weights,
    plot_dendritic_weights_single_image,
    print_network_entropy,
    train,
    train_one_model,
    plot_dendritic_weights_full_model,
)


# for repoducability
cp.random.seed(12123)

# data config
dataset = "cifar10"  # "mnist", "fashion-mnist", "cifar10"
subset_size = None

# config
n_epochs = 20  # 15 MNIST, 20 Fashion-MNIST
lr = 0.001  # 0.003
v_lr = 0.001  # 0.015 - SGD
b_lr = 0.001  # 0.015 - SGD
weight_decay = 0.001  # 0.001
batch_size = 128
n_classes = 10

print("RUN 50 epochs, 0.001 lr, 128 batch size")

if dataset in ["mnist", "fashion-mnist"]:
    in_dim = 28 * 28  # Image dimensions (28x28 MNIST, 32x32x3 CIFAR-10)
elif dataset == "cifar10":
    in_dim = 32 * 32 * 3
else:
    raise ValueError(f"Invalid dataset: {dataset}")

# cifar10: 32 / 18 / 49, equal to input / 10 / 10

# dendriticmodel config
n_dendrite_inputs = 32  # 31
n_dendrites = 32  # 23
n_neurons = 256  # 10
strategy = "random"  # ["random", "local-receptive-fields", "fully-connected"]

if dataset in ["mnist", "fashion-mnist"]:
    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset=dataset, subset_size=subset_size
    )
elif dataset == "cifar10":
    X_train, y_train, X_test, y_test = load_cifar10_data(subset_size=subset_size)

print("Preparing model...")
criterion = CrossEntropy()
model = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
            strategy=strategy,
            synaptic_resampling=True,
            percentage_resample=0.25,
            steps_to_resample=128,
            scaling_resampling_percentage=False,
            dynamic_steps_size=False,
        ),
        LeakyReLU(),
        LinearLayer(n_neurons, n_classes),
    ]
)
optimiser = Adam(
    model.params(), criterion, lr=lr, weight_decay=weight_decay, grad_clip=0.1
)

# train_one_model(
#     X_train,
#     y_train,
#     X_test,
#     y_test,
#     model,
#     criterion,
#     optimiser,
#     n_epochs=n_epochs,
#     batch_size=batch_size,
# )

# # plot_dendritic_weights_full_model(model, X_test[0])
# # for i in range(10):
#     # plot_dendritic_weights_single_image(model, X_test[0], neuron_idx=i)

sum_of_synapses = cp.count_nonzero(model.layers[0].dendrite_W)
print(f"sum of synapses: {sum_of_synapses}")


# baseline dendritic model
b_criterion = CrossEntropy()
b_model = Sequential(
    [
        DendriticLayer(
            in_dim,
            n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
            strategy=strategy,
            synaptic_resampling=False,
        ),
        LeakyReLU(),
        LinearLayer(n_neurons, n_classes),
    ]
)
b_optimiser = Adam(b_model.params(), b_criterion, lr=b_lr, weight_decay=weight_decay)

# vanilla model
v_criterion = CrossEntropy()
v_model = Sequential(
    [
        LinearLayer(in_dim, 10),
        LeakyReLU(),
        LinearLayer(10, 10),
        LeakyReLU(),
        LinearLayer(10, n_classes),
    ]
)
v_optimiser = Adam(v_model.params(), v_criterion, lr=v_lr, weight_decay=weight_decay)

print(f"number of model_1 params: {model.num_params()}")
print(f"number of model_2 params: {b_model.num_params()}")
print(f"number of model_3 params: {v_model.num_params()}")


compare_models(
    model,
    b_model,
    v_model,
    optimiser,
    b_optimiser,
    v_optimiser,
    X_train,
    y_train,
    X_test,
    y_test,
    criterion,
    n_epochs=n_epochs,
    batch_size=batch_size,
    model_name_1="Synaptic Resampling",
    model_name_2="Base Dendritic",
    model_name_3="Vanilla ANN",
    track_variance=False,
)

# # print("Dendritic model")
# # print_network_entropy(model)
# # print("Vanilla model")
# # print_network_entropy(v_model)

# # print("\n\n")
# # print(f"number of mask updates: {model.layers[0].num_mask_updates}")
# # print(f"number of mask updates baseline model: {v_model.layers[0].num_mask_updates}")
# # print("\n\n")

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
