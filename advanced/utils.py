import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def calculate_eigenvalues(model, model_name):
    """
    Calculate eigenvalues of weight matrices in dendritic layers and linear layers.

    Args:
        model: PyTorch model containing DendriticLayer and/or Linear layers
        model_name (str): Name of the model for display purposes
    """
    print(f"\n=== Eigenvalues for {model_name} ===")

    for i, layer in enumerate(model.modules()):
        # Skip the top-level Sequential container
        if isinstance(layer, nn.Sequential):
            continue

        if hasattr(layer, "dendrite_W"):
            # DendriticLayer
            # Convert to numpy for eigenvalue calculation
            dendrite_W = layer.dendrite_W.detach().cpu().numpy()
            soma_W = layer.soma_W.detach().cpu().numpy()

            # Calculate eigenvalues of covariance matrices
            # For dendrite_W: (n_soma_connections, in_dim) -> compute W @ W.T
            dendrite_cov = dendrite_W @ dendrite_W.T
            dendrite_eigenvals = np.linalg.eigvals(dendrite_cov)

            # For soma_W: (n_neurons, n_soma_connections) -> compute W @ W.T
            soma_cov = soma_W @ soma_W.T
            soma_eigenvals = np.linalg.eigvals(soma_cov)

            # Sort eigenvalues in descending order
            dendrite_eigenvals = np.sort(np.real(dendrite_eigenvals))[::-1]
            soma_eigenvals = np.sort(np.real(soma_eigenvals))[::-1]

            print(f"  Layer {i} (DendriticLayer):")
            print(f"    Dendrite weights eigenvalues (top 5): {dendrite_eigenvals[:5]}")
            print(
                f"    Dendrite weights eigenvalues (bottom 5): {dendrite_eigenvals[-5:]}"
            )
            print(f"    Soma weights eigenvalues (top 5): {soma_eigenvals[:5]}")
            print(f"    Soma weights eigenvalues (bottom 5): {soma_eigenvals[-5:]}")
            print(f"    Dendrite spectral norm: {dendrite_eigenvals[0]:.6f}")
            print(f"    Soma spectral norm: {soma_eigenvals[0]:.6f}")

        elif isinstance(layer, nn.Linear):
            # Linear layer
            # Convert to numpy for eigenvalue calculation
            W = layer.weight.detach().cpu().numpy()

            # Calculate eigenvalues of covariance matrix
            # For W: (out_dim, in_dim) -> compute W @ W.T
            W_cov = W @ W.T
            W_eigenvals = np.linalg.eigvals(W_cov)

            # Sort eigenvalues in descending order
            W_eigenvals = np.sort(np.real(W_eigenvals))[::-1]

            print(f"  Layer {i} (Linear):")
            print(f"    Weight matrix shape: {W.shape}")
            print(f"    Weight eigenvalues (top 5): {W_eigenvals[:5]}")
            print(f"    Weight eigenvalues (bottom 5): {W_eigenvals[-5:]}")
            print(f"    Spectral norm: {W_eigenvals[0]:.6f}")


def plot_dendritic_weights_single_image(
    model, input_image, neuron_idx=0, image_shape=(28, 28)
):
    """
    Plots the aggregated magnitude of all dendritic weights of a single neuron on one image.
    Color indicates the sum of magnitudes at each location.

    Args:
        model: PyTorch model containing DendriticLayer
        input_image: Input image tensor (flattened)
        neuron_idx (int): Index of neuron to visualize
        image_shape (tuple): Shape to reshape image for display
    """

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Find the first DendriticLayer
    dendritic_layer = None
    for layer in model.modules():
        if hasattr(layer, "dendrite_W"):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    if not (0 <= neuron_idx < dendritic_layer.n_neurons):
        print(
            f"Invalid neuron_idx. Must be between 0 and {dendritic_layer.n_neurons - 1}."
        )
        return

    # Get the weights and mask for the specified neuron's dendrites
    start_idx = neuron_idx * dendritic_layer.n_dendrites
    end_idx = start_idx + dendritic_layer.n_dendrites

    dendrite_weights = to_numpy(dendritic_layer.dendrite_W[start_idx:end_idx])
    dendrite_mask = to_numpy(dendritic_layer.dendrite_mask[start_idx:end_idx])

    masked_weights = dendrite_weights * dendrite_mask
    input_image_np = to_numpy(input_image)

    # Calculate and sum magnitudes
    magnitudes = np.abs(masked_weights)
    summed_magnitudes = np.sum(magnitudes, axis=0)

    # Reshape for plotting
    summed_magnitudes_2d = summed_magnitudes.reshape(image_shape)

    # Plot background image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(input_image_np.reshape(image_shape), cmap="gray", interpolation="nearest")

    # Mask zeros for the overlay
    heatmap_masked = np.ma.masked_where(summed_magnitudes_2d == 0, summed_magnitudes_2d)

    # Plot heatmap of magnitudes
    im = ax.imshow(heatmap_masked, cmap="viridis", alpha=0.6, interpolation="nearest")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Sum of Weight Magnitudes")

    ax.set_title(f"Aggregated Dendritic Weight Magnitudes for Neuron {neuron_idx}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_dendritic_weights_full_model(model, image_shape=(28, 28)):
    """
    Plots the aggregated magnitude of all dendritic weights across all neurons in the model.
    Shows the combined weight pattern without any background image.

    Args:
        model: PyTorch model containing DendriticLayer
        image_shape (tuple): Shape to reshape weights for display
    """

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # Find the DendriticLayer
    dendritic_layer = None
    for layer in model.modules():
        if hasattr(layer, "dendrite_W"):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    n_neurons = dendritic_layer.n_neurons
    n_dendrites = dendritic_layer.n_dendrites

    print(f"Visualizing {n_neurons} neurons, {n_dendrites} dendrites each")

    # Get all dendritic weights and masks
    dendrite_weights = to_numpy(dendritic_layer.dendrite_W)
    dendrite_mask = to_numpy(dendritic_layer.dendrite_mask)

    # Apply mask to get only active weights
    masked_weights = dendrite_weights * dendrite_mask

    # Calculate magnitudes and sum across all dendrites and neurons
    magnitudes = np.abs(masked_weights)
    total_magnitudes = np.sum(magnitudes, axis=0)

    # Reshape for plotting
    total_magnitudes_2d = total_magnitudes.reshape(image_shape)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot heatmap of all dendritic weights
    im = ax.imshow(total_magnitudes_2d, cmap="viridis", interpolation="nearest")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Sum of All Dendritic Weight Magnitudes")

    ax.set_title(
        f"All Dendritic Weights Aggregated\nTotal Neurons: {n_neurons}, Dendrites per Neuron: {n_dendrites}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Also create a subplot showing individual neuron contributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Dendritic Weight Analysis - Full Model", fontsize=16)

    # Plot 1: All weights aggregated
    im1 = axes[0, 0].imshow(
        total_magnitudes_2d, cmap="viridis", interpolation="nearest"
    )
    axes[0, 0].set_title("All Dendritic Weights")
    axes[0, 0].axis("off")
    fig.colorbar(im1, ax=axes[0, 0], shrink=0.7)

    # Plot 2: Weight distribution histogram
    axes[0, 1].hist(
        magnitudes[magnitudes > 0], bins=50, alpha=0.7, color="blue", edgecolor="black"
    )
    axes[0, 1].set_title("Weight Magnitude Distribution")
    axes[0, 1].set_xlabel("Weight Magnitude")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Spatial activity map (sum of absolute weights at each location)
    spatial_activity = np.sum(np.abs(masked_weights), axis=0).reshape(image_shape)
    im3 = axes[1, 0].imshow(spatial_activity, cmap="plasma", interpolation="nearest")
    axes[1, 0].set_title("Spatial Activity Map")
    axes[1, 0].axis("off")
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.7)

    # Plot 4: Active connections map (count of non-zero weights at each location)
    active_connections = np.sum(masked_weights != 0, axis=0).reshape(image_shape)
    im4 = axes[1, 1].imshow(active_connections, cmap="hot", interpolation="nearest")
    axes[1, 1].set_title("Active Connections Count")
    axes[1, 1].axis("off")
    fig.colorbar(im4, ax=axes[1, 1], shrink=0.5)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\n=== Full Model Dendritic Statistics ===")
    print(f"Total parameters: {dendrite_weights.size}")
    print(f"Active parameters: {np.sum(dendrite_mask)}")
    print(f"Sparsity: {1 - np.sum(dendrite_mask) / dendrite_mask.size:.4f}")
    print(f"Mean active weight magnitude: {np.mean(magnitudes[magnitudes > 0]):.6f}")
    print(f"Max weight magnitude: {np.max(magnitudes):.6f}")
    print(f"Min non-zero weight magnitude: {np.min(magnitudes[magnitudes > 0]):.6f}")


def count_parameters(model):
    """Count the number of parameters in a PyTorch model"""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    # For DendriticLayer, also count effective parameters (considering sparsity)
    effective_params = 0
    for module in model.modules():
        if hasattr(module, "num_params"):
            effective_params += module.num_params()
        elif isinstance(module, nn.Linear):
            effective_params += module.weight.numel() + module.bias.numel()

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    if effective_params != total_params:
        print(f"Effective parameters (accounting for sparsity): {effective_params}")

    return total_params, trainable_params, effective_params


def plot_training_curves(
    train_losses, train_accuracies, test_losses, test_accuracies, model_names
):
    """
    Plot training curves for multiple models.

    Args:
        train_losses: List of training loss lists for each model
        train_accuracies: List of training accuracy lists for each model
        test_losses: List of test loss lists for each model
        test_accuracies: List of test accuracy lists for each model
        model_names: List of model names for legend
    """
    colors = ["green", "blue", "red", "orange", "purple"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracies
    for i, (train_acc, test_acc, name) in enumerate(
        zip(train_accuracies, test_accuracies, model_names)
    ):
        color = colors[i % len(colors)]
        ax1.plot(train_acc, label=f"{name} Train", color=color, linestyle="--")
        ax1.plot(test_acc, label=f"{name} Test", color=color)

    ax1.set_title("Accuracy over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    for i, (train_loss, test_loss, name) in enumerate(
        zip(train_losses, test_losses, model_names)
    ):
        color = colors[i % len(colors)]
        ax2.plot(train_loss, label=f"{name} Train", color=color, linestyle="--")
        ax2.plot(test_loss, label=f"{name} Test", color=color)

    ax2.set_title("Loss over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_final_results(
    train_losses, train_accuracies, test_losses, test_accuracies, model_names
):
    """Print final training results for all models"""
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    for i, name in enumerate(model_names):
        print(f"\n{name}:")
        print(f"  Final train loss: {train_losses[i][-1]:.4f}")
        print(f"  Final test loss: {test_losses[i][-1]:.4f}")
        print(f"  Final train accuracy: {train_accuracies[i][-1] * 100:.1f}%")
        print(f"  Final test accuracy: {test_accuracies[i][-1] * 100:.1f}%")

    # Compare models
    print("\nComparison:")
    train_loss_str = " vs ".join([f"{loss[-1]:.4f}" for loss in train_losses])
    test_loss_str = " vs ".join([f"{loss[-1]:.4f}" for loss in test_losses])
    train_acc_str = " vs ".join([f"{acc[-1] * 100:.1f}%" for acc in train_accuracies])
    test_acc_str = " vs ".join([f"{acc[-1] * 100:.1f}%" for acc in test_accuracies])

    print(f"  Train loss: {train_loss_str}")
    print(f"  Test loss: {test_loss_str}")
    print(f"  Train accuracy: {train_acc_str}")
    print(f"  Test accuracy: {test_acc_str}")
