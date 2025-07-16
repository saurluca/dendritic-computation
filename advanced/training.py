import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools

from data import load_mnist_data, create_data_loaders
from utils import plot_training_curves, print_final_results, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        # Convert one-hot targets to class indices for CrossEntropyLoss
        target_labels = torch.argmax(target, dim=1)
        loss = criterion(output, target_labels)

        # Backward pass
        loss.backward()

        # Handle synaptic resampling for DendriticLayer if needed
        for module in model.modules():
            if hasattr(module, "_schedule_resampling") and module._schedule_resampling:
                module.resample_dendrites()
                module._schedule_resampling = False

        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        correct_predictions += (pred == target_labels).sum().item()
        total_samples += data.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test data.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            # Convert one-hot targets to class indices for CrossEntropyLoss
            target_labels = torch.argmax(target, dim=1)
            loss = criterion(output, target_labels)

            # Statistics
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct_predictions += (pred == target_labels).sum().item()
            total_samples += data.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    n_epochs,
    device,
    model_name,
    verbose=True,
):
    """
    Train a model for multiple epochs and track performance.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        n_epochs: Number of epochs
        device: Device to run on
        model_name: Name for display
        verbose: Whether to show progress bars

    Returns:
        tuple: (train_losses, train_accuracies, test_losses, test_accuracies)
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    if verbose:
        print(f"\nTraining {model_name} model...")

    progress_bar = tqdm(
        total=n_epochs, desc=f"Training {model_name}", disable=not verbose
    )

    for epoch in range(n_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Update progress bar
        if verbose:
            progress_bar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Train Acc": f"{train_acc:.3f}",
                    "Test Loss": f"{test_loss:.4f}",
                    "Test Acc": f"{test_acc:.3f}",
                }
            )
        progress_bar.update(1)

    progress_bar.close()

    if verbose:
        print(
            f"Final {model_name} - Train: {train_acc * 100:.1f}%, Test: {test_acc * 100:.1f}%"
        )

    return train_losses, train_accuracies, test_losses, test_accuracies


def compare_models(
    models_config,
    dataset="fashion-mnist",
    n_epochs=20,
    batch_size=128,
    device=None,
    subset_size=None,
    plot_results=True,
    verbose=True,
):
    """
    Train and compare multiple models.

    Args:
        models_config: List of [model, optimizer, name] tuples
        dataset: Dataset to use ("mnist" or "fashion-mnist")
        n_epochs: Number of epochs to train
        batch_size: Batch size for training
        device: Device to use (auto-detect if None)
        subset_size: Use subset of data if specified
        plot_results: Whether to plot comparison results
        verbose: Whether to show detailed output

    Returns:
        list: Results for each model with metrics and metadata

    Example:
        models_config = [
            [model1, optimizer1, "Synaptic Resampling"],
            [model2, optimizer2, "Base Dendritic"],
            [model3, optimizer3, "Vanilla ANN"]
        ]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")
        print(f"Comparing {len(models_config)} models on {dataset}")

    # Load data
    if verbose:
        print("Loading data...")
    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset=dataset, subset_size=subset_size
    )

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=batch_size
    )

    # Print model parameters
    if verbose:
        print("\nModel parameters:")
        for model, optimizer, name in models_config:
            print(f"\n{name}:")
            count_parameters(model)

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

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Train each model
    results = []
    for i, (model, optimizer, name) in enumerate(models_config):
        train_losses, train_accs, test_losses, test_accs = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            n_epochs,
            device,
            name,
            verbose=verbose,
        )

        results.append(
            {
                "name": name,
                "model": model,
                "train_losses": train_losses,
                "train_accuracies": train_accs,
                "test_losses": test_losses,
                "test_accuracies": test_accs,
                "color": colors[i % len(colors)],
                "final_train_acc": train_accs[-1],
                "final_test_acc": test_accs[-1],
                "final_train_loss": train_losses[-1],
                "final_test_loss": test_losses[-1],
            }
        )

    # Plot results if requested
    if plot_results:
        model_names = [r["name"] for r in results]
        train_losses_all = [r["train_losses"] for r in results]
        train_accs_all = [r["train_accuracies"] for r in results]
        test_losses_all = [r["test_losses"] for r in results]
        test_accs_all = [r["test_accuracies"] for r in results]

        plot_training_curves(
            train_losses_all,
            train_accs_all,
            test_losses_all,
            test_accs_all,
            model_names,
        )

        # Plot additional analysis for dendritic models
        dendritic_results = [r for r in results if hasattr(r["model"][0], "dendrite_W")]
        if len(dendritic_results) > 1:
            plot_dendritic_comparison(dendritic_results)

    # Print final results
    if verbose:
        print_final_results(
            [r["train_losses"] for r in results],
            [r["train_accuracies"] for r in results],
            [r["test_losses"] for r in results],
            [r["test_accuracies"] for r in results],
            [r["name"] for r in results],
        )

    return results


def plot_dendritic_comparison(dendritic_results):
    """Plot comparison specific to dendritic models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot weight distribution histogram
    for result in dendritic_results:
        model = result["model"]
        name = result["name"]
        color = result["color"]

        # Get dendritic layer (assuming it's the first layer)
        dendritic_layer = model[0]
        dendrite_weights = dendritic_layer.dendrite_W.detach().cpu().numpy()
        dendrite_mask = dendritic_layer.dendrite_mask.detach().cpu().numpy()
        masked_weights = dendrite_weights * dendrite_mask

        magnitudes = np.abs(masked_weights)
        active_magnitudes = magnitudes[magnitudes > 0]

        ax1.hist(
            active_magnitudes,
            bins=50,
            alpha=0.6,
            color=color,
            edgecolor="black",
            label=name,
            density=True,
        )

    ax1.set_title("Weight Magnitude Distribution")
    ax1.set_xlabel("Weight Magnitude")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot sparsity comparison
    sparsity_data = []
    model_names = []
    colors_list = []

    for result in dendritic_results:
        model = result["model"]
        name = result["name"]
        color = result["color"]

        dendritic_layer = model[0]
        dendrite_mask = dendritic_layer.dendrite_mask.detach().cpu().numpy()

        total_params = dendrite_mask.size
        active_params = np.sum(dendrite_mask)
        sparsity = 1 - (active_params / total_params)

        sparsity_data.append(sparsity)
        model_names.append(name)
        colors_list.append(color)

    bars = ax2.bar(
        model_names, sparsity_data, color=colors_list, alpha=0.7, edgecolor="black"
    )
    ax2.set_title("Model Sparsity")
    ax2.set_ylabel("Sparsity (1 - active_params/total_params)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, sparsity in zip(bars, sparsity_data):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{sparsity:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def hyperparameter_tuning(
    model_factory,
    param_grid,
    dataset="fashion-mnist",
    n_epochs=10,
    batch_size=128,
    device=None,
    subset_size=None,
    metric="test_accuracy",
    verbose=True,
):
    """
    Perform hyperparameter tuning using grid search.

    Args:
        model_factory: Function that takes hyperparameters and returns (model, optimizer)
        param_grid: Dictionary of parameter names to lists of values to try
        dataset: Dataset to use
        n_epochs: Number of epochs for each trial
        batch_size: Batch size for training
        device: Device to use
        subset_size: Use subset of data if specified
        metric: Metric to optimize ('test_accuracy', 'test_loss', 'train_accuracy', 'train_loss')
        verbose: Whether to show detailed output

    Returns:
        dict: Best parameters and results

    Example:
        def model_factory(lr, hidden_dim):
            model = nn.Sequential(
                nn.Linear(784, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 10)
            )
            optimizer = optim.Adam(model.parameters(), lr=lr)
            return model, optimizer

        param_grid = {
            'lr': [0.001, 0.01, 0.1],
            'hidden_dim': [32, 64, 128]
        }

        best_params = hyperparameter_tuning(model_factory, param_grid)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data once
    if verbose:
        print("Loading data...")
    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset=dataset, subset_size=subset_size
    )
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=batch_size
    )

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations...")
        print(f"Parameters: {param_names}")

    # Track results
    results = []
    best_score = float("-inf") if "accuracy" in metric else float("inf")
    best_params = None
    best_result = None

    criterion = nn.CrossEntropyLoss()

    # Test each parameter combination
    for i, param_combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))

        if verbose:
            print(f"\nTrial {i + 1}/{len(param_combinations)}: {params}")

        try:
            # Create model and optimizer with current parameters
            model, optimizer = model_factory(**params)
            model = model.to(device)

            # Train model
            train_losses, train_accs, test_losses, test_accs = train_model(
                model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                n_epochs,
                device,
                f"Trial {i + 1}",
                verbose=False,
            )

            # Extract final metric value
            final_metrics = {
                "test_accuracy": test_accs[-1],
                "test_loss": test_losses[-1],
                "train_accuracy": train_accs[-1],
                "train_loss": train_losses[-1],
            }

            score = final_metrics[metric]

            result = {
                "params": params,
                "score": score,
                "train_losses": train_losses,
                "train_accuracies": train_accs,
                "test_losses": test_losses,
                "test_accuracies": test_accs,
                "final_metrics": final_metrics,
            }

            results.append(result)

            # Check if this is the best so far
            is_better = (
                (score > best_score) if "accuracy" in metric else (score < best_score)
            )
            if is_better:
                best_score = score
                best_params = params.copy()
                best_result = result

            if verbose:
                print(f"  {metric}: {score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Failed: {str(e)}")
            continue

    # Plot results
    if len(results) > 0:
        plot_hyperparameter_results(results, param_names, metric)

    if verbose:
        print(f"\n{'=' * 50}")
        print("HYPERPARAMETER TUNING RESULTS")
        print(f"{'=' * 50}")
        print(f"Best {metric}: {best_score:.4f}")
        print(f"Best parameters: {best_params}")

        if best_result:
            print("\nFull metrics for best model:")
            for metric_name, value in best_result["final_metrics"].items():
                print(f"  {metric_name}: {value:.4f}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_result": best_result,
        "all_results": results,
        "optimization_metric": metric,
    }


def plot_hyperparameter_results(results, param_names, metric):
    """Plot hyperparameter tuning results"""
    if len(param_names) == 1:
        # 1D parameter sweep
        param_name = param_names[0]
        param_values = [r["params"][param_name] for r in results]
        scores = [r["score"] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, scores, "bo-")
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.title(f"Hyperparameter Tuning: {param_name} vs {metric}")
        plt.grid(True, alpha=0.3)
        plt.show()

    elif len(param_names) == 2:
        # 2D parameter sweep - heatmap
        param1_name, param2_name = param_names

        # Get unique values for each parameter
        param1_values = sorted(list(set(r["params"][param1_name] for r in results)))
        param2_values = sorted(list(set(r["params"][param2_name] for r in results)))

        # Create score matrix
        score_matrix = np.full((len(param2_values), len(param1_values)), np.nan)

        for result in results:
            i = param2_values.index(result["params"][param2_name])
            j = param1_values.index(result["params"][param1_name])
            score_matrix[i, j] = result["score"]

        plt.figure(figsize=(10, 8))
        im = plt.imshow(score_matrix, cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(im, label=metric)

        plt.xticks(range(len(param1_values)), param1_values)
        plt.yticks(range(len(param2_values)), param2_values)
        plt.xlabel(param1_name)
        plt.ylabel(param2_name)
        plt.title(f"Hyperparameter Tuning: {metric}")

        # Add text annotations
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                if not np.isnan(score_matrix[i, j]):
                    plt.text(
                        j,
                        i,
                        f"{score_matrix[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.show()

    else:
        # Multiple parameters - show best vs parameter values
        fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 4))
        if len(param_names) == 1:
            axes = [axes]

        for i, param_name in enumerate(param_names):
            param_values = [r["params"][param_name] for r in results]
            scores = [r["score"] for r in results]

            axes[i].scatter(param_values, scores, alpha=0.7)
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f"{param_name} vs {metric}")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
