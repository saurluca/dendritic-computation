try:
    import cupy as cp

    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")

from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from utils import load_mnist_data, load_cifar10_data


def create_batches(X, y, batch_size=128, shuffle=True, drop_last=True):
    n_samples = len(X)
    # shuffle data
    if shuffle:
        indices = cp.arange(n_samples)
        cp.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    for i in range(0, n_samples, batch_size):
        if drop_last and i + batch_size > n_samples:
            break
        X_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield X_batch, y_batch


def train(
    X_train,
    y_train,
    X_test,
    y_test,
    model,
    criterion,
    optimiser,
    n_epochs=2,
    batch_size=128,
):
    train_losses = []
    accuracy = []
    test_losses = []
    test_accuracy = []
    n_samples = len(X_train)
    num_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
    total_batches = n_epochs * num_batches_per_epoch

    with tqdm(total=total_batches, desc="Training ") as pbar:
        for epoch in range(n_epochs):
            train_loss = 0.0
            correct_pred = 0.0
            for batch_idx, (X, target) in enumerate(
                create_batches(X_train, y_train, batch_size, shuffle=True)
            ):
                # forward pass
                pred = model(X)
                batch_loss = criterion(pred, target)
                train_loss += batch_loss
                # if most likely prediction equals target add to correct predictions
                batch_correct = cp.sum(
                    cp.argmax(pred, axis=1) == cp.argmax(target, axis=1)
                )
                correct_pred += batch_correct

                # backward pass
                optimiser.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimiser.step()

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch + 1}/{n_epochs}",
                        "Batch": f"{batch_idx + 1}/{num_batches_per_epoch}",
                        "Loss": f"{float(batch_loss):.4f}",
                    }
                )
                pbar.update(1)
            # evaluate on test set
            epoch_test_loss, epoch_test_accuracy = evaluate(
                X_test, y_test, model, criterion
            )
            normalised_train_loss = train_loss / num_batches_per_epoch
            train_losses.append(
                float(normalised_train_loss)
            )  # Convert to float for plotting
            epoch_accuracy = correct_pred / n_samples
            accuracy.append(float(epoch_accuracy))  # Convert to float for plotting
            test_losses.append(float(epoch_test_loss))
            test_accuracy.append(float(epoch_test_accuracy))
    return train_losses, accuracy, test_losses, test_accuracy


def evaluate(
    X_test,
    y_test,
    model,
    criterion,
    batch_size=1024,
):
    n_samples = len(X_test)
    test_loss = 0.0
    correct_pred = 0.0
    num_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
    for X, target in create_batches(
        X_test, y_test, batch_size, shuffle=False, drop_last=False
    ):
        # forward pass
        pred = model(X)
        batch_loss = criterion(pred, target)
        test_loss += batch_loss
        # if most likely prediction eqauls target add to correct predictions
        batch_correct = cp.sum(cp.argmax(pred, axis=1) == cp.argmax(target, axis=1))
        correct_pred += batch_correct
    normalised_test_loss = test_loss / num_batches_per_epoch
    accuracy = correct_pred / n_samples
    return float(normalised_test_loss), float(accuracy)


def train_models(
    models_config,
    dataset,
    criterion,
    n_epochs,
    batch_size=256,
    subset_size=None,
    verbose=False,
    data_augmentation=False,
):
    """
    Train one or multiple models dynamically.

    Args:
        models_config: List of [model, optimizer, name] tuples
        X_train, y_train: Training data
        X_test, y_test: Test data
        criterion: Loss function
        n_epochs: Number of epochs
        batch_size: Batch size

    Example:
        models_config = [
            [model1, optimizer1, "Synaptic Resampling"],
            [model2, optimizer2, "Base Dendritic"],
            [model3, optimizer3, "Vanilla ANN"]
        ]
    """

    num_models = len(models_config)
    results = []

    # Define colors for plotting
    colors = [
        "green",
        "blue",
        "red",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # print parameters of each model
    for i, (model, optimizer, name) in enumerate(models_config):
        print(f"Number of params: {model.num_params(verbose=verbose)} of {name} model")

    if dataset in ["mnist", "fashion-mnist"]:
        X_train, y_train, X_test, y_test = load_mnist_data(
            dataset=dataset, subset_size=subset_size
        )
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(
            subset_size=subset_size, data_augmentation=data_augmentation
        )

    # Train each model
    for i, (model, optimizer, name) in enumerate(models_config):
        print(f"\nTraining {name} model...")

        train_losses, train_accuracy, test_losses, test_accuracy = train(
            X_train,
            y_train,
            X_test,
            y_test,
            model,
            criterion,
            optimizer,
            n_epochs,
            batch_size,
        )

        # Print mask updates if available
        if (
            hasattr(model, "layers")
            and len(model.layers) > 0
            and hasattr(model.layers[0], "num_mask_updates")
        ):
            print(f"Number of mask updates: {model.layers[0].num_mask_updates}")

        # Print results for this model
        print(f"Train loss {name} model: {round(train_losses[-1], 4)}")
        print(f"Train accuracy {name} model: {round(train_accuracy[-1] * 100, 1)}%")
        print(f"Test accuracy {name} model: {round(test_accuracy[-1] * 100, 1)}%")

        results.append(
            {
                "name": name,
                "train_losses": train_losses,
                "train_accuracy": train_accuracy,
                "test_losses": test_losses,
                "test_accuracy": test_accuracy,
                "color": colors[i % len(colors)],
            }
        )

    # Plot results
    if num_models == 1:
        # Single model plotting (similar to train_one_model)
        result = results[0]

        # Plot losses
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(result["train_losses"], label="Train Loss", color="blue")
        plt.plot(result["test_losses"], label="Test Loss", color="red")
        plt.title(f"Loss - {result['name']}")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(result["train_accuracy"], label="Train Accuracy", color="blue")
        plt.plot(result["test_accuracy"], label="Test Accuracy", color="red")
        plt.title(f"Accuracy - {result['name']}")
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Multiple models comparison plotting

        # Plot accuracy comparison
        plt.figure(figsize=(12, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        for result in results:
            plt.plot(
                result["train_accuracy"],
                label=f"{result['name']} Train",
                color=result["color"],
                linestyle="--",
            )
            plt.plot(
                result["test_accuracy"],
                label=f"{result['name']} Test",
                color=result["color"],
            )
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        for result in results:
            plt.plot(
                result["train_losses"],
                label=f"{result['name']} Train",
                color=result["color"],
                linestyle="--",
            )
            plt.plot(
                result["test_losses"],
                label=f"{result['name']} Test",
                color=result["color"],
            )
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Print final comparison statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    if num_models == 1:
        result = results[0]
        print(f"Model: {result['name']}")
        print(f"Train loss: {result['train_losses'][-1]:.3f}")
        print(f"Test loss: {result['test_losses'][-1]:.3f}")
        print(f"Train accuracy: {result['train_accuracy'][-1]:.3f}")
        print(f"Test accuracy: {result['test_accuracy'][-1]:.3f}")
    else:
        # Print comparison table
        print(
            f"{'Model':<20} {'Train Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12}"
        )
        print("-" * 68)
        for result in results:
            print(
                f"{result['name']:<20} {result['train_losses'][-1]:<12.4f} "
                f"{result['test_losses'][-1]:<12.4f} {result['train_accuracy'][-1] * 100:<11.1f}% "
                f"{result['test_accuracy'][-1] * 100:<11.1f}%"
            )

    print("=" * 60)

    return results


def calculate_dendritic_spatial_entropy(
    dendrite_weights, dendrite_mask, image_shape=(28, 28)
):
    """
    Calculate the spatial entropy of dendritic weights.

    Args:
        dendrite_weights: Array of dendritic weights
        dendrite_mask: Mask indicating which weights are active
        image_shape: Shape to reshape the weights into

    Returns:
        tuple: (spatial_entropy, weight_value_entropy)
    """
    import numpy as np

    def to_numpy(arr):
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    dendrite_weights = to_numpy(dendrite_weights)
    dendrite_mask = to_numpy(dendrite_mask)

    # Apply mask to get only active weights
    masked_weights = dendrite_weights * dendrite_mask

    # 1. Spatial entropy: How spread out are the non-zero weights?
    # Sum across dendrites to get total activity at each spatial location
    spatial_activity = np.sum(np.abs(masked_weights), axis=0)
    spatial_activity_2d = spatial_activity.reshape(image_shape)

    # Normalize to create probability distribution
    total_activity = np.sum(spatial_activity_2d)
    if total_activity == 0:
        spatial_entropy = 0
    else:
        spatial_prob = spatial_activity_2d / total_activity
        # Remove zeros to avoid log(0)
        spatial_prob = spatial_prob[spatial_prob > 0]
        spatial_entropy = -np.sum(spatial_prob * np.log2(spatial_prob))

    # 2. Weight value entropy: How diverse are the weight values themselves?
    # Get all non-zero weights
    nonzero_weights = masked_weights[masked_weights != 0]
    if len(nonzero_weights) == 0:
        weight_value_entropy = 0
    else:
        # Create histogram of weight values
        hist, _ = np.histogram(nonzero_weights, bins=50, density=True)
        # Normalize to probabilities
        hist = hist / np.sum(hist)
        # Remove zeros
        hist = hist[hist > 0]
        weight_value_entropy = -np.sum(hist * np.log2(hist))

    return spatial_entropy, weight_value_entropy


def plot_dendritic_weights(model, input_image, neuron_idx=0, image_shape=(28, 28)):
    """
    Plots the weights of each dendrite of a specific neuron from a DendriticLayer
    over a given input image.

    Args:
        model (Sequential): The trained model containing a DendriticLayer.
        input_image (cp.ndarray): A single input image (flattened).
        neuron_idx (int): The index of the neuron to visualize.
        image_shape (tuple): The shape to reshape the image and weights into (e.g., (28, 28)).
    """
    import numpy as np

    def to_numpy(arr):
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # Find the DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
        if hasattr(layer, "dendrite_W"):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    # Check if neuron_idx is valid
    if not (0 <= neuron_idx < dendritic_layer.n_neurons):
        print(
            f"Invalid neuron_idx. Must be between 0 and {dendritic_layer.n_neurons - 1}."
        )
        return

    n_dendrites = dendritic_layer.n_dendrites

    # Get the weights and mask for the specified neuron's dendrites
    start_idx = neuron_idx * n_dendrites
    end_idx = start_idx + n_dendrites

    dendrite_weights = to_numpy(dendritic_layer.dendrite_W[start_idx:end_idx])
    dendrite_mask = to_numpy(dendritic_layer.dendrite_mask[start_idx:end_idx])

    # Calculate entropy
    spatial_entropy, weight_value_entropy = calculate_dendritic_spatial_entropy(
        dendrite_weights, dendrite_mask, image_shape
    )

    print(f"Neuron {neuron_idx} - Spatial Entropy: {spatial_entropy:.4f}")
    print(f"Neuron {neuron_idx} - Weight Value Entropy: {weight_value_entropy:.4f}")

    masked_weights = dendrite_weights * dendrite_mask
    magnitudes = np.abs(masked_weights)
    input_image_np = to_numpy(input_image)

    # Determine grid size for subplots
    n_cols = int(math.ceil(math.sqrt(n_dendrites)))
    n_rows = int(math.ceil(n_dendrites / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle(
        f"Dendritic Weight Magnitudes for Neuron {neuron_idx}\nSpatial Entropy: {spatial_entropy:.4f}, Weight Entropy: {weight_value_entropy:.4f}",
        fontsize=14,
    )

    axes = axes.flatten()

    # Find global min/max for consistent color scaling of weights
    vmax = magnitudes.max()
    if vmax == 0:
        vmax = 1.0
    vmin = 0

    for i in range(n_dendrites):
        ax = axes[i]

        image_2d = input_image_np.reshape(image_shape)
        magnitudes_2d = magnitudes[i].reshape(image_shape)

        ax.imshow(image_2d, cmap="gray", interpolation="nearest")

        magnitudes_masked = np.ma.masked_where(magnitudes_2d == 0, magnitudes_2d)

        im = ax.imshow(
            magnitudes_masked,
            cmap="viridis",
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(f"Dendrite {i + 1}")
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_dendrites, len(axes)):
        axes[i].axis("off")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Weight Magnitude")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_dendritic_weights_single_image(
    model, input_image, neuron_idx=0, image_shape=(28, 28)
):
    """
    Plots the aggregated magnitude of all dendritic weights of a single neuron on one image.
    Color indicates the sum of magnitudes at each location.
    """
    import numpy as np

    def to_numpy(arr):
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # Find the first DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
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

    # Calculate entropy
    spatial_entropy, weight_value_entropy = calculate_dendritic_spatial_entropy(
        dendrite_weights, dendrite_mask, image_shape
    )

    print(f"Neuron {neuron_idx} - Spatial Entropy: {spatial_entropy:.4f}")
    print(f"Neuron {neuron_idx} - Weight Value Entropy: {weight_value_entropy:.4f}")

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

    ax.set_title(
        f"Aggregated Dendritic Weight Magnitudes for Neuron {neuron_idx}\nSpatial Entropy: {spatial_entropy:.4f}, Weight Entropy: {weight_value_entropy:.4f}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_dendritic_weights_full_model(model, input_image, image_shape=(28, 28)):
    """
    Plots the aggregated magnitude of all dendritic weights across all neurons in the model.
    Shows the combined weight pattern without any background image.
    """
    import numpy as np

    def to_numpy(arr):
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # Find the DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
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

    # Calculate entropy for the entire network
    spatial_entropy, weight_value_entropy = calculate_dendritic_spatial_entropy(
        dendrite_weights, dendrite_mask, image_shape
    )

    print(f"Full Model - Spatial Entropy: {spatial_entropy:.4f}")
    print(f"Full Model - Weight Value Entropy: {weight_value_entropy:.4f}")

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
        f"All Dendritic Weights Aggregated\n"
        f"Spatial Entropy: {spatial_entropy:.4f}, Weight Entropy: {weight_value_entropy:.4f}\n"
        f"Total Neurons: {n_neurons}, Dendrites per Neuron: {n_dendrites}"
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


def print_network_entropy(model, image_shape=(28, 28)):
    """
    Calculate and print entropy values for all neurons in the dendritic layer.

    Args:
        model: The trained model containing a DendriticLayer
        image_shape: Shape to reshape the weights into (e.g., (28, 28))
    """
    import numpy as np

    # Find the DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
        if hasattr(layer, "dendrite_W"):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    n_neurons = dendritic_layer.n_neurons
    n_dendrites = dendritic_layer.n_dendrites

    spatial_entropies = []
    weight_entropies = []

    print("=== Network Entropy Analysis ===")
    print(f"Dendritic Layer: {n_neurons} neurons, {n_dendrites} dendrites each")
    print(f"{'Neuron':<6} {'Spatial Entropy':<15} {'Weight Entropy':<15}")

    for neuron_idx in range(n_neurons):
        # Get the weights and mask for the specified neuron's dendrites
        start_idx = neuron_idx * n_dendrites
        end_idx = start_idx + n_dendrites

        dendrite_weights = dendritic_layer.dendrite_W[start_idx:end_idx]
        dendrite_mask = dendritic_layer.dendrite_mask[start_idx:end_idx]

        # Calculate entropy for this neuron
        spatial_entropy, weight_value_entropy = calculate_dendritic_spatial_entropy(
            dendrite_weights, dendrite_mask, image_shape
        )

        spatial_entropies.append(spatial_entropy)
        weight_entropies.append(weight_value_entropy)

    # Calculate summary statistics
    spatial_entropies = np.array(spatial_entropies)
    weight_entropies = np.array(weight_entropies)

    print("-" * 40)
    print(
        f"{'Mean':<6} {np.mean(spatial_entropies):<15.4f} {np.mean(weight_entropies):<15.4f}"
    )
    print(
        f"{'Std':<6} {np.std(spatial_entropies):<15.4f} {np.std(weight_entropies):<15.4f}"
    )
    print(
        f"{'Min':<6} {np.min(spatial_entropies):<15.4f} {np.min(weight_entropies):<15.4f}"
    )
    print(
        f"{'Max':<6} {np.max(spatial_entropies):<15.4f} {np.max(weight_entropies):<15.4f}"
    )
    print(
        f"{'Range':<6} {np.max(spatial_entropies) - np.min(spatial_entropies):<15.4f} {np.max(weight_entropies) - np.min(weight_entropies):<15.4f}"
    )
    print("\n\n")
    return spatial_entropies, weight_entropies
