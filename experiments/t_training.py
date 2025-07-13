import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from t_data import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _needs_flattening(model):
    """Check if a model needs flattened input (contains Linear or DropLinear layers)"""
    if isinstance(model, torch.nn.Sequential):
        # Check if first layer is Linear or DropLinear
        first_layer = model[0]
        return isinstance(
            first_layer, (torch.nn.Linear, torch.nn.modules.module.Module)
        ) and (hasattr(first_layer, "in_features") or hasattr(first_layer, "in_dim"))
    return False


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    model_name="Model",
    n_epochs=20,
):
    """Train a neural network (either dendritic or ViT)"""
    # Check if this is a dendritic model
    is_dendritic = hasattr(model, "dendritic_layer")

    # Check if model needs flattened input (contains Linear or DropLinear layers)
    needs_flattening = is_dendritic or _needs_flattening(model)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    total_batches = len(train_loader) * n_epochs

    with tqdm(total=total_batches, desc=f"Training {model_name}") as pbar:
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                # Flatten images for models that need flattened input
                if needs_flattening:
                    data = data.view(data.size(0), -1)  # Flatten for linear layers

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()

                # Apply masks only for dendritic model
                if is_dendritic:
                    with torch.no_grad():
                        if (
                            model.dendritic_layer.dendrite_linear.weight.grad
                            is not None
                        ):
                            model.dendritic_layer.dendrite_linear.weight.grad *= (
                                model.dendritic_layer.dendrite_mask
                            )

                optimizer.step()

                # Apply masks to weights after optimizer step (dendritic only)
                if is_dendritic:
                    model.dendritic_layer._apply_masks()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch + 1}/{n_epochs}",
                        "Batch": f"{batch_idx + 1}/{len(train_loader)}",
                        "Loss": f"{loss.item():.4f}",
                    }
                )
                pbar.update(1)

            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_acc = correct / total

            # Evaluate on test set
            test_loss, test_acc = evaluate_model(
                model, test_loader, criterion, needs_flattening
            )

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            if epoch % 2 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                )

    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluate_model(model, test_loader, criterion, needs_flattening=None):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # Auto-detect if flattening is needed if not provided
    if needs_flattening is None:
        is_dendritic = hasattr(model, "dendritic_layer")
        needs_flattening = is_dendritic or _needs_flattening(model)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Flatten images for models that need flattened input
            if needs_flattening:
                data = data.view(data.size(0), -1)  # Flatten for linear layers

            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total

    return test_loss, test_acc


def plot_weight_distributions(results_dict):
    """Plot weight distributions for the first layer of all trained models"""

    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 5))
    if len(results_dict) == 1:
        axes = [axes]

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, (model_name, result) in enumerate(results_dict.items()):
        model = result["model"]
        ax = axes[i]

        # Get the first layer
        if isinstance(model, torch.nn.Sequential):
            first_layer = model[0]
        else:
            first_layer = model

        # Extract weights based on layer type
        if hasattr(first_layer, "dendrite_mask"):  # DendriticLayer
            weights = first_layer.dendrite_linear.weight.data.cpu().numpy()
            mask = first_layer.dendrite_mask.cpu().numpy()

            # Only plot active weights (where mask is True)
            active_weights = []
            for i_out in range(weights.shape[0]):
                for i_in in range(weights.shape[1]):
                    if mask[i_out, i_in]:  # dendrite_mask is 2D (output_dim, in_dim)
                        active_weights.append(weights[i_out, i_in])

            active_weights = np.array(active_weights)

            ax.hist(
                active_weights,
                bins=50,
                alpha=0.7,
                density=True,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

            n_active = len(active_weights)
            n_total = weights.size
            ax.set_title(
                f"{model_name} - Active Weights\n({n_active}/{n_total} = {n_active / n_total:.1%})"
            )

        elif hasattr(first_layer, "mask"):  # DropLinear layer
            weights = first_layer.weight.data.cpu().numpy()
            mask = first_layer.mask.cpu().numpy()

            # Only plot active weights (where mask is True)
            active_weights = []
            for i_out in range(weights.shape[0]):
                for i_in in range(weights.shape[1]):
                    if mask[i_in]:  # mask is 1D for input features
                        active_weights.append(weights[i_out, i_in])

            active_weights = np.array(active_weights)

            ax.hist(
                active_weights,
                bins=50,
                alpha=0.7,
                density=True,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

            n_active = len(active_weights)
            n_total = weights.size
            ax.set_title(
                f"{model_name} - Active Weights\n({n_active}/{n_total} = {n_active / n_total:.1%})"
            )

        else:  # Regular Linear layer
            weights = first_layer.weight.data.cpu().numpy().flatten()

            ax.hist(
                weights,
                bins=50,
                alpha=0.7,
                density=True,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

            ax.set_title(f"{model_name} - All Weights\n({len(weights)} weights)")

        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

        # Set consistent scales for comparison
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(0, 5)

        # Add statistics text
        if hasattr(first_layer, "dendrite_mask") or hasattr(first_layer, "mask"):
            mean_val = np.mean(active_weights)
            std_val = np.std(active_weights)
        else:
            mean_val = np.mean(weights)
            std_val = np.std(weights)

        ax.text(
            0.02,
            0.98,
            f"μ = {mean_val:.3f}\nσ = {std_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig("weight_distributions.png", dpi=300, bbox_inches="tight")


def plot_results(results_dict):
    """Plot training results for multiple models"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot losses
    for i, (
        model_name,
        (train_losses, train_accuracies, test_losses, test_accuracies),
    ) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax1.plot(
            train_losses,
            label=f"{model_name} Train",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax1.plot(test_losses, label=f"{model_name} Test", color=color)

    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    for i, (
        model_name,
        (train_losses, train_accuracies, test_losses, test_accuracies),
    ) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax2.plot(
            train_accuracies,
            label=f"{model_name} Train",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax2.plot(test_accuracies, label=f"{model_name} Test", color=color)

    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{model_name}_results.png")


def count_active_parameters(model):
    """Count active parameters in a model, considering dendritic layers"""
    total_params = 0

    # Handle Sequential models with DendriticLayers
    if isinstance(model, torch.nn.Sequential):
        for layer in model:
            if hasattr(layer, "num_active_params"):
                # DropLinear with active parameter counting
                total_params += int(layer.num_active_params())
            elif hasattr(layer, "num_params"):
                # DendriticLayer with custom parameter counting
                total_params += layer.num_params()
            else:
                # Standard layer
                total_params += sum(p.numel() for p in layer.parameters())
    # Handle models with custom num_params method (like VisionTransformer)
    elif hasattr(model, "num_params"):
        total_params = model.num_params()
    else:
        # Standard model - count all parameters
        total_params = sum(p.numel() for p in model.parameters())

    return total_params


def train_models_comparative(
    models_config,
    criterion,
    dataset="mnist",
    n_epochs=8,
    batch_size=256,
    verbose=True,
):
    """
    Train multiple models comparatively and plot results.

    Args:
        models_config: List of [model, optimizer, name] tuples
        dataset: Dataset to use ("mnist", "fashion-mnist", or "cifar10")
        n_epochs: Number of epochs to train
        batch_size: Batch size for training
        verbose: Whether to print detailed information

    Example:
        models_config = [
            [dendritic_model, optim.Adam(dendritic_model.parameters(), lr=0.002), "Dendritic"],
            [vit_model, optim.Adam(vit_model.parameters(), lr=0.001), "ViT"],
            [ff_model, optim.Adam(ff_model.parameters(), lr=0.002), "Standard"]
        ]
    """

    # Define colors for plotting
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Load dataset
    train_loader, test_loader, input_dim, num_classes = load_dataset(
        dataset, batch_size
    )

    # Print model parameters if verbose
    if verbose:
        print(f"Dataset: {dataset.upper()}")
        print(f"Training {len(models_config)} models for {n_epochs} epochs")
        print("-" * 60)
        for i, config in enumerate(models_config):
            model, optimizer, name = config[:3]  # Handle both 3 and 4 element configs
            params = count_active_parameters(model)
            print(f"{name}: {params:,} parameters")
        print("-" * 60)

    # Train each model and collect results
    results = {}

    for i, config in enumerate(models_config):
        model, optimizer, name = config

        print(f"\nTraining {name} model...")

        # Train the model
        train_losses, train_accuracies, test_losses, test_accuracies = train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=name,
            n_epochs=n_epochs,
            criterion=criterion,
        )

        # Store results
        results[name] = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_losses": test_losses,
            "test_accuracies": test_accuracies,
            "color": colors[i % len(colors)],
            "model": model,
        }

        # Print individual results
        print(f"{name} Results:")
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        print(f"  Final Train Accuracy: {train_accuracies[-1] * 100:.1f}%")
        print(f"  Final Test Loss: {test_losses[-1]:.4f}")
        print(f"  Final Test Accuracy: {test_accuracies[-1] * 100:.1f}%")

        # Print mask updates if available (for dendritic models)
        if hasattr(model, "dendritic_layer") and hasattr(
            model.dendritic_layer, "num_mask_updates"
        ):
            print(f"  Mask Updates: {model.dendritic_layer.num_mask_updates}")
        elif hasattr(model, "blocks"):  # For ViT with dendritic layers
            total_updates = 0
            for block in model.blocks:
                if hasattr(block.mlp, "__getitem__") and hasattr(
                    block.mlp[0], "num_mask_updates"
                ):
                    total_updates += block.mlp[0].num_mask_updates
            if total_updates > 0:
                print(f"  Total Mask Updates: {total_updates}")

    # Create comparative plots
    print("\n" + "=" * 60)
    print("CREATING COMPARATIVE PLOTS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Model Comparison on {dataset.upper()}", fontsize=16)

    # Plot 1: Test Accuracy Comparison
    ax1 = axes[0, 0]
    for name, result in results.items():
        ax1.plot(
            result["test_accuracies"],
            label=f"{name}",
            color=result["color"],
            linewidth=2,
        )
    ax1.set_title("Test Accuracy Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Loss Comparison
    ax2 = axes[0, 1]
    for name, result in results.items():
        ax2.plot(
            result["test_losses"], label=f"{name}", color=result["color"], linewidth=2
        )
    ax2.set_title("Test Loss Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Train vs Test Accuracy
    ax3 = axes[1, 0]
    for name, result in results.items():
        ax3.plot(
            result["train_accuracies"],
            label=f"{name} Train",
            color=result["color"],
            linestyle="--",
            alpha=0.7,
        )
        ax3.plot(
            result["test_accuracies"],
            label=f"{name} Test",
            color=result["color"],
            linewidth=2,
        )
    ax3.set_title("Train vs Test Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Train vs Test Loss
    ax4 = axes[1, 1]
    for name, result in results.items():
        ax4.plot(
            result["train_losses"],
            label=f"{name} Train",
            color=result["color"],
            linestyle="--",
            alpha=0.7,
        )
        ax4.plot(
            result["test_losses"],
            label=f"{name} Test",
            color=result["color"],
            linewidth=2,
        )
    ax4.set_title("Train vs Test Loss")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final comparison table
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)

    # Create comparison table
    print(
        f"{'Model':<20} {'Train Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Parameters':<12}"
    )
    print("-" * 80)

    for name, result in results.items():
        model = result["model"]
        params = count_active_parameters(model)

        print(
            f"{name:<20} {result['train_losses'][-1]:<12.4f} "
            f"{result['test_losses'][-1]:<12.4f} {result['train_accuracies'][-1] * 100:<11.1f}% "
            f"{result['test_accuracies'][-1] * 100:<11.1f}% {params:<12,}"
        )

    print("Final number of parameters:")
    print(f"Training {len(models_config)} models for {n_epochs} epochs")
    print("-" * 60)
    for i, config in enumerate(models_config):
        model, optimizer, name = config[:3]  # Handle both 3 and 4 element configs
        params = count_active_parameters(model)
        print(f"{name}: {params:,} parameters")
    print("-" * 60)

    # Plot weight distributions for first layer of all models
    print("\n" + "=" * 60)
    print("WEIGHT DISTRIBUTIONS (First Layer)")
    print("=" * 60)
    plot_weight_distributions(results)
