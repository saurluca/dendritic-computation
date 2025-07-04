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
    track_variance=False,
):
    train_losses = []
    accuracy = []
    test_losses = []
    test_accuracy = []
    n_samples = len(X_train)
    num_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
    total_batches = n_epochs * num_batches_per_epoch

    variance_of_grads = [] 
    variance_of_weights = []

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
                model_grads = model.backward(grad)
                optimiser.step()
                
                if track_variance:
                    variance_of_grads.append(cp.var(model_grads))
                    variance_of_weights.append(model.var_params())


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
    return train_losses, accuracy, test_losses, test_accuracy, variance_of_grads, variance_of_weights


def evaluate(
    X_test,
    y_test,
    model,
    criterion,
    batch_size=256,
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


def compare_models(
    model_1,
    model_2,
    optimiser_1,
    optimiser_2,
    X_train,
    y_train,
    X_test,
    y_test,
    criterion,
    n_epochs=10,
    batch_size=256,
    model_name_1="Dendritic",
    model_name_2="Vanilla",
    track_variance=False,
):
    print(f"Training {model_name_1} model...")
    train_losses_1, train_accuracy_1, test_losses_1, test_accuracy_1, variance_of_grads_1, variance_of_weights_1 = train(
        X_train,
        y_train,
        X_test,
        y_test,
        model_1,
        criterion,
        optimiser_1,
        n_epochs,
        batch_size,
        track_variance,
    )

    print(f"Training {model_name_2} model...")
    train_losses_2, train_accuracy_2, test_losses_2, test_accuracy_2, variance_of_grads_2, variance_of_weights_2 = train(
        X_train,
        y_train,
        X_test,
        y_test,
        model_2,
        criterion,
        optimiser_2,
        n_epochs,
        batch_size,
        track_variance,
    )

    # plot accuracy of vanilla model vs dendritic model
    plt.plot(
        train_accuracy_1, label=f"{model_name_1} Train", color="green", linestyle="--"
    )
    plt.plot(
        train_accuracy_2, label=f"{model_name_2} Train", color="blue", linestyle="--"
    )
    plt.plot(test_accuracy_1, label=f"{model_name_1} Test", color="green")
    plt.plot(test_accuracy_2, label=f"{model_name_2} Test", color="blue")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.show()

    # plot both models in comparison
    plt.plot(
        train_losses_1, label=f"{model_name_1} Train", color="green", linestyle="--"
    )
    plt.plot(
        train_losses_2, label=f"{model_name_2} Train", color="blue", linestyle="--"
    )
    plt.plot(test_losses_1, label=f"{model_name_1} Test", color="green")
    plt.plot(test_losses_2, label=f"{model_name_2} Test", color="blue")
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()
    
    # print final weight stats of dendritc layer
    weights_1 = cp.abs(model_1.params()[0].dendrite_W)
    weights_2 = cp.abs(model_2.params()[0].dendrite_W)
    
    print(f"weights_1: {weights_1.shape}")
    print(f"weights_2: {weights_2.shape}")
    print(f"mean weights_1: {cp.mean(weights_1)}")
    print(f"mean weights_2: {cp.mean(weights_2)}")
    print(f"std weights_1: {cp.std(weights_1)}")
    print(f"std weights_2: {cp.std(weights_2)}")
    
    
    # plot variance of grads
    if track_variance:
        # Convert CuPy arrays to NumPy arrays for plotting
        variance_grads_1_np = [float(x.get()) if hasattr(x, 'get') else float(x) for x in variance_of_grads_1]
        variance_grads_2_np = [float(x.get()) if hasattr(x, 'get') else float(x) for x in variance_of_grads_2]
        
        # Compute mean variance of gradients for each model
        mean_variance_grads_1 = sum(variance_grads_1_np) / len(variance_grads_1_np)
        mean_variance_grads_2 = sum(variance_grads_2_np) / len(variance_grads_2_np)
        
        # Handle variance_of_weights which is a list of lists (one list per batch, containing variances per layer)
        # Compute mean variance across all layers for each batch
        variance_weights_1_np = []
        for batch_variances in variance_of_weights_1:
            # batch_variances is a list of variances from each layer
            # Convert each variance to float and compute mean
            layer_variances = [float(var.get()) if hasattr(var, 'get') else float(var) for var in batch_variances]
            variance_weights_1_np.append(sum(layer_variances) / len(layer_variances))
        
        variance_weights_2_np = []
        for batch_variances in variance_of_weights_2:
            # batch_variances is a list of variances from each layer
            # Convert each variance to float and compute mean
            layer_variances = [float(var.get()) if hasattr(var, 'get') else float(var) for var in batch_variances]
            variance_weights_2_np.append(sum(layer_variances) / len(layer_variances))
        
        plt.plot([mean_variance_grads_1], label=f"{model_name_1} Mean Variance of Gradients", color="green", linestyle="--")
        plt.plot([mean_variance_grads_2], label=f"{model_name_2} Mean Variance of Gradients", color="blue", linestyle="--")
        plt.title("Mean Variance of Gradients over epochs")
        plt.legend()
        plt.show()
        
        plt.plot(variance_weights_1_np, label=f"{model_name_1} Variance of Weights", color="green", linestyle="--")
        plt.plot(variance_weights_2_np, label=f"{model_name_2} Variance of Weights", color="blue", linestyle="--")
        plt.title("Variance of Weights over epochs")
        plt.legend()
        plt.show()

    print(
        f"train loss {model_name_1} model {round(train_losses_1[-1], 4)} vs {model_name_2} {round(train_losses_2[-1], 4)}"
    )
    print(
        f"test loss {model_name_1} model {round(test_losses_1[-1], 4)} vs {model_name_2} {round(test_losses_2[-1], 4)}"
    )
    print(
        f"train accuracy {model_name_1} model {round(train_accuracy_1[-1] * 100, 1)}% vs {model_name_2} {round(train_accuracy_2[-1] * 100, 1)}%"
    )
    print(
        f"test accuracy {model_name_1} model {round(test_accuracy_1[-1] * 100, 1)}% vs {model_name_2} {round(test_accuracy_2[-1] * 100, 1)}%"
    )


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
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    # Find the DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
        if hasattr(layer, 'dendrite_W'):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    # Check if neuron_idx is valid
    if not (0 <= neuron_idx < dendritic_layer.n_neurons):
        print(f"Invalid neuron_idx. Must be between 0 and {dendritic_layer.n_neurons - 1}.")
        return

    n_dendrites = dendritic_layer.n_dendrites
    
    # Get the weights and mask for the specified neuron's dendrites
    start_idx = neuron_idx * n_dendrites
    end_idx = start_idx + n_dendrites
    
    dendrite_weights = to_numpy(dendritic_layer.dendrite_W[start_idx:end_idx])
    dendrite_mask = to_numpy(dendritic_layer.dendrite_mask[start_idx:end_idx])
    
    masked_weights = dendrite_weights * dendrite_mask
    magnitudes = np.abs(masked_weights)
    input_image_np = to_numpy(input_image)

    # Determine grid size for subplots
    n_cols = int(math.ceil(math.sqrt(n_dendrites)))
    n_rows = int(math.ceil(n_dendrites / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle(f"Dendritic Weight Magnitudes for Neuron {neuron_idx}", fontsize=16)
    
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
        
        ax.imshow(image_2d, cmap='gray', interpolation='nearest')
        
        magnitudes_masked = np.ma.masked_where(magnitudes_2d == 0, magnitudes_2d)
        
        im = ax.imshow(magnitudes_masked, cmap='viridis', alpha=0.6, vmin=vmin, vmax=vmax, interpolation='nearest')
        
        ax.set_title(f"Dendrite {i + 1}")
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_dendrites, len(axes)):
        axes[i].axis('off')

    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="Weight Magnitude")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_dendritic_weights_single_image(model, input_image, neuron_idx=0, image_shape=(28, 28)):
    """
    Plots the aggregated magnitude of all dendritic weights of a single neuron on one image.
    Color indicates the sum of magnitudes at each location.
    """
    import numpy as np

    def to_numpy(arr):
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    # Find the first DendriticLayer
    dendritic_layer = None
    for layer in model.layers:
        if hasattr(layer, 'dendrite_W'):
            dendritic_layer = layer
            break

    if dendritic_layer is None:
        print("No DendriticLayer found in the model.")
        return

    if not (0 <= neuron_idx < dendritic_layer.n_neurons):
        print(f"Invalid neuron_idx. Must be between 0 and {dendritic_layer.n_neurons - 1}.")
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
    ax.imshow(input_image_np.reshape(image_shape), cmap='gray', interpolation='nearest')

    # Mask zeros for the overlay
    heatmap_masked = np.ma.masked_where(summed_magnitudes_2d == 0, summed_magnitudes_2d)

    # Plot heatmap of magnitudes
    im = ax.imshow(heatmap_masked, cmap='viridis', alpha=0.6, interpolation='nearest')

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Sum of Weight Magnitudes")
    
    ax.set_title(f'Aggregated Dendritic Weight Magnitudes for Neuron {neuron_idx}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


