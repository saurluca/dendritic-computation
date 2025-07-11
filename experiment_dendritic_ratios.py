# %%
"""
Dendritic Layer Ratio Experiment

This experiment systematically explores optimal ratios for dendritic layer parameters
on Fashion-MNIST dataset.

Key Configuration:
- Modify `n_experiments` in the main section to control how many different
  ratio configurations to test (default: 12)
- The system automatically generates appropriate ratios based on this number:
  * 1 experiment: Uses single reasonable default
  * ≤9 experiments: Selects from predefined good ratios
  * >9 experiments: Generates log-spaced ratios for comprehensive coverage

Architecture: DendriticLayer (soma disabled) → LeakyReLU → LinearLayer
"""

import json
from datetime import datetime

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
import matplotlib.patches as mpatches
from modules import (
    Adam,
    CrossEntropy,
    LeakyReLU,
    Sequential,
    DendriticLayer,
    LinearLayer,
)
from utils import load_mnist_data
from training import train_models


def calculate_total_model_params(n_dendrites, n_dendrite_inputs, n_classes=10):
    """Calculate total number of parameters in the full model (dendritic layer + linear layer)"""
    # Dendritic layer parameters:
    # dendrite_W: n_dendrites * n_dendrite_inputs (masked)
    # dendrite_b: n_dendrites
    dendrite_params = n_dendrites * (n_dendrite_inputs + 1)

    # Linear layer parameters:
    # W: n_dendrites * n_classes
    # b: n_classes
    linear_params = n_dendrites * n_classes + n_classes

    total_params = dendrite_params + linear_params
    return total_params


def find_valid_combinations(
    target_params=16000, in_dim=784, tolerance=0.05, n_experiments=9
):
    """Find valid combinations of n_dendrites and n_dendrite_inputs for target parameter count

    Args:
        target_params: Target number of total model parameters
        in_dim: Input dimension
        tolerance: Tolerance for parameter count matching
        n_experiments: Number of experiments to run (controls number of ratios tested)
    """
    valid_combinations = []

    # Generate ratios dynamically based on n_experiments
    # Use log-spaced ratios between 1/100 and 1/2 for good coverage
    if n_experiments == 1:
        dendrite_input_ratios = [1 / 8]  # Single reasonable default
    elif n_experiments <= 9:
        # For small numbers, use the predefined good ratios
        all_ratios = [
            1 / 100,
            1 / 50,
            1 / 25,
            1 / 10,
            1 / 8,
            1 / 5,
            1 / 4,
            1 / 3,
            1 / 2,
        ]
        # Select evenly spaced ratios from the list
        indices = cp.linspace(0, len(all_ratios) - 1, n_experiments, dtype=int)
        dendrite_input_ratios = [all_ratios[i] for i in indices]
    else:
        # For larger numbers, generate log-spaced ratios
        import numpy as np

        # Generate log-spaced ratios between 1/100 and 1/2
        log_ratios = np.logspace(np.log10(1 / 100), np.log10(1 / 2), n_experiments)
        dendrite_input_ratios = log_ratios.tolist()

    print(
        f"Testing {n_experiments} experiments with ratios: {[f'{r:.4f}' for r in dendrite_input_ratios]}"
    )

    for ratio in dendrite_input_ratios:
        n_dendrite_inputs = max(1, int(in_dim * ratio))
        if n_dendrite_inputs > in_dim:
            continue

        # Calculate required n_dendrites for target parameters
        # For the full model: target_params = dendrite_params + linear_params
        # dendrite_params = n_dendrites * (n_dendrite_inputs + 1)
        # linear_params = n_dendrites * n_classes + n_classes
        # total = n_dendrites * (n_dendrite_inputs + 1 + n_classes) + n_classes
        n_classes = 10
        n_dendrites = (target_params - n_classes) // (n_dendrite_inputs + 1 + n_classes)

        if n_dendrites <= 0:
            continue

        actual_params = calculate_total_model_params(
            n_dendrites, n_dendrite_inputs, n_classes
        )
        param_error = abs(actual_params - target_params) / target_params

        if param_error <= tolerance:
            valid_combinations.append(
                {
                    "n_dendrite_inputs": n_dendrite_inputs,
                    "n_dendrites": n_dendrites,
                    "actual_params": actual_params,
                    "dendrite_input_ratio": n_dendrite_inputs / in_dim,
                    "dendrite_ratio": n_dendrites / in_dim,
                    "param_error": param_error,
                }
            )

    return valid_combinations


def create_model(
    n_dendrite_inputs, n_dendrites, n_classes, in_dim, synaptic_resampling=False
):
    """Create dendritic model without soma"""
    return Sequential(
        [
            DendriticLayer(
                in_dim,
                n_neurons=1,  # Not used when soma_enabled=False
                n_dendrite_inputs=n_dendrite_inputs,
                n_dendrites=n_dendrites,
                strategy="random",
                soma_enabled=False,
                synaptic_resampling=synaptic_resampling,
                percentage_resample=0.25,
                steps_to_resample=128,
            ),
            LeakyReLU(),
            LinearLayer(n_dendrites, n_classes),
        ]
    )


def run_experiment(
    target_params=16000,
    n_epochs=10,
    lr=0.0005,
    weight_decay=0.01,
    batch_size=256,
    subset_size=5000,
    test_synaptic_resampling=False,
    n_experiments=9,
):
    """Run the dendritic layer ratio experiment

    Args:
        target_params: Target number of total model parameters
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        batch_size: Batch size for training
        subset_size: Number of samples to use (None for full dataset)
        test_synaptic_resampling: Whether to test both with/without synaptic resampling
        n_experiments: Number of different ratio configurations to test
    """

    # Set seed for reproducibility
    cp.random.seed(1223)

    # Load Fashion-MNIST data
    print("Loading Fashion-MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset="fashion-mnist", subset_size=subset_size
    )

    in_dim = 28 * 28
    n_classes = 10

    # Find valid parameter combinations
    print(
        f"Finding valid combinations for {target_params} total model parameters (dendritic + linear layers)..."
    )
    combinations = find_valid_combinations(
        target_params, in_dim, n_experiments=n_experiments
    )

    if not combinations:
        print("No valid combinations found!")
        return None

    print(f"Found {len(combinations)} valid combinations")
    for i, combo in enumerate(combinations[:3]):  # Show first 3
        print(
            f"  {i + 1}: n_dendrite_inputs={combo['n_dendrite_inputs']}, "
            f"n_dendrites={combo['n_dendrites']}, "
            f"total_params={combo['actual_params']}"
        )

    # Run experiments
    results = []
    criterion = CrossEntropy()

    synaptic_options = [False, True] if test_synaptic_resampling else [False]

    for combo in combinations:
        for synaptic_resampling in synaptic_options:
            print(
                f"\nTesting: n_dendrite_inputs={combo['n_dendrite_inputs']}, "
                f"n_dendrites={combo['n_dendrites']}, "
                f"synaptic_resampling={synaptic_resampling}"
            )

            try:
                # Create model
                model = create_model(
                    combo["n_dendrite_inputs"],
                    combo["n_dendrites"],
                    n_classes,
                    in_dim,
                    synaptic_resampling,
                )

                # Create optimizer
                optimizer = Adam(
                    model.params(), criterion, lr=lr, weight_decay=weight_decay
                )

                # Train model
                model_config = [[model, optimizer, "Dendrite_Test"]]

                experiment_results = train_models(
                    model_config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    criterion,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                )

                # Extract results - FIX: use correct key names
                result_data = experiment_results[0]
                final_test_acc = result_data["test_accuracy"][
                    -1
                ]  # Changed from 'test_accuracies'
                final_train_acc = result_data["train_accuracy"][
                    -1
                ]  # Changed from 'train_accuracies'
                final_test_loss = result_data["test_losses"][-1]
                final_train_loss = result_data["train_losses"][-1]

                result = {
                    **combo,
                    "synaptic_resampling": synaptic_resampling,
                    "final_test_accuracy": float(final_test_acc),
                    "final_train_accuracy": float(final_train_acc),
                    "final_test_loss": float(final_test_loss),
                    "final_train_loss": float(final_train_loss),
                    "success": True,
                }

                print(f"  ✓ Success! Test accuracy: {final_test_acc:.4f}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                result = {
                    **combo,
                    "synaptic_resampling": synaptic_resampling,
                    "final_test_accuracy": 0.0,
                    "final_train_accuracy": 0.0,
                    "final_test_loss": float("inf"),
                    "final_train_loss": float("inf"),
                    "success": False,
                    "error": str(e),
                }

            results.append(result)

    return results


def print_best_ratios(results):
    """Print a clear summary of the best ratios found"""
    if not results:
        print("No results to summarize!")
        return None

    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        print("No successful experiments to summarize!")
        return None

    # Find best result
    best_result = max(successful_results, key=lambda x: x["final_test_accuracy"])

    print("\n" + "=" * 60)
    print("🏆 BEST CONFIGURATION FOUND")
    print("=" * 60)
    print(f"Best Test Accuracy: {best_result['final_test_accuracy']:.4f}")
    print(f"Total Model Parameters: {best_result['actual_params']}")

    # Calculate parameter breakdown
    n_dendrites = best_result["n_dendrites"]
    n_dendrite_inputs = best_result["n_dendrite_inputs"]
    n_classes = 10

    dendrite_params = n_dendrites * (n_dendrite_inputs + 1)
    linear_params = n_dendrites * n_classes + n_classes

    print(f"  • Dendritic Layer: {dendrite_params} params")
    print(f"    - Dendrite weights: {n_dendrites * n_dendrite_inputs}")
    print(f"    - Dendrite biases: {n_dendrites}")
    print(f"  • Linear Layer: {linear_params} params")
    print(f"    - Linear weights: {n_dendrites * n_classes}")
    print(f"    - Linear biases: {n_classes}")
    print()
    print("📊 OPTIMAL RATIOS:")
    print(
        f"  • n_dendrite_inputs/in_dim ratio: {best_result['dendrite_input_ratio']:.4f}"
    )
    print(
        f"    (n_dendrite_inputs = {best_result['n_dendrite_inputs']} out of {28 * 28} inputs)"
    )
    print(f"  • n_dendrites/in_dim ratio: {best_result['dendrite_ratio']:.4f}")
    print(f"    (n_dendrites = {best_result['n_dendrites']})")
    print(
        f"  • Synaptic resampling: {'ON' if best_result['synaptic_resampling'] else 'OFF'}"
    )
    print()
    print("📋 COPY-PASTE VALUES:")
    print(f"n_dendrite_inputs = {best_result['n_dendrite_inputs']}")
    print(f"n_dendrites = {best_result['n_dendrites']}")
    print(f"synaptic_resampling = {best_result['synaptic_resampling']}")
    print("=" * 60)

    return best_result


def plot_results(results):
    """Create plots showing how accuracy relates to dendritic parameters"""
    if not results:
        print("No results to plot!")
        return

    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        print("No successful results to plot!")
        return

    # Extract data for plotting
    n_dendrite_inputs = [r["n_dendrite_inputs"] for r in successful_results]
    n_dendrites = [r["n_dendrites"] for r in successful_results]
    accuracies = [r["final_test_accuracy"] for r in successful_results]
    dendrite_input_ratios = [r["dendrite_input_ratio"] for r in successful_results]
    dendrite_ratios = [r["dendrite_ratio"] for r in successful_results]
    synaptic_resampling = [r["synaptic_resampling"] for r in successful_results]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Dendritic Parameter Analysis - Accuracy vs Architecture",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: n_dendrites vs n_dendrite_inputs colored by accuracy
    scatter1 = axes[0, 0].scatter(
        n_dendrite_inputs,
        n_dendrites,
        c=accuracies,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    axes[0, 0].set_xlabel("n_dendrite_inputs")
    axes[0, 0].set_ylabel("n_dendrites")
    axes[0, 0].set_title("Parameter Space (colored by accuracy)")
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label("Test Accuracy")

    # Add text annotations for best points
    best_acc = max(accuracies)
    for i, (x, y, acc) in enumerate(zip(n_dendrite_inputs, n_dendrites, accuracies)):
        if acc == best_acc:
            axes[0, 0].annotate(
                f"Best\n{acc:.3f}",
                (x, y),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    # Plot 2: Accuracy vs n_dendrite_inputs
    colors2 = ["red" if sr else "blue" for sr in synaptic_resampling]
    axes[0, 1].scatter(
        n_dendrite_inputs, accuracies, c=colors2, s=100, alpha=0.7, edgecolors="black"
    )
    axes[0, 1].set_xlabel("n_dendrite_inputs")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Accuracy vs Dendrite Input Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Add trend line
    if len(n_dendrite_inputs) > 1:
        import numpy as np

        z = np.polyfit(n_dendrite_inputs, accuracies, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(n_dendrite_inputs), max(n_dendrite_inputs), 100)
        axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    # Plot 3: Accuracy vs n_dendrites
    axes[1, 0].scatter(
        n_dendrites, accuracies, c=colors2, s=100, alpha=0.7, edgecolors="black"
    )
    axes[1, 0].set_xlabel("n_dendrites")
    axes[1, 0].set_ylabel("Test Accuracy")
    axes[1, 0].set_title("Accuracy vs Dendrite Count")
    axes[1, 0].grid(True, alpha=0.3)

    # Add trend line
    if len(n_dendrites) > 1:
        z = np.polyfit(n_dendrites, accuracies, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(n_dendrites), max(n_dendrites), 100)
        axes[1, 0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    # Plot 4: Accuracy vs Input/Dendrite Ratios
    axes[1, 1].scatter(
        dendrite_input_ratios,
        accuracies,
        c=colors2,
        s=100,
        alpha=0.7,
        edgecolors="black",
        label="Input Ratio",
    )
    axes[1, 1].scatter(
        dendrite_ratios,
        accuracies,
        c=colors2,
        s=100,
        alpha=0.7,
        edgecolors="black",
        marker="^",
        label="Dendrite Ratio",
    )
    axes[1, 1].set_xlabel("Ratio (to input dimension)")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Accuracy vs Architecture Ratios")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Add legends for synaptic resampling
    if any(synaptic_resampling):
        blue_patch = mpatches.Patch(color="blue", label="No Resampling")
        red_patch = mpatches.Patch(color="red", label="With Resampling")
        axes[0, 1].legend(handles=[blue_patch, red_patch], loc="upper right")
        axes[1, 0].legend(handles=[blue_patch, red_patch], loc="upper right")
        axes[1, 1].legend(
            handles=[
                blue_patch,
                red_patch,
                mpatches.Patch(color="gray", label="○ Input Ratio"),
                mpatches.Patch(color="gray", label="△ Dendrite Ratio"),
            ],
            loc="upper right",
        )

    # Adjust layout and save
    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dendritic_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {filename}")

    # Show plot
    plt.show()

    # Print some insights
    print("\n📈 ANALYSIS INSIGHTS:")
    if len(successful_results) > 1:
        # Correlation analysis
        input_corr = np.corrcoef(n_dendrite_inputs, accuracies)[0, 1]
        dendrite_corr = np.corrcoef(n_dendrites, accuracies)[0, 1]
        input_ratio_corr = np.corrcoef(dendrite_input_ratios, accuracies)[0, 1]

        print(f"  • n_dendrite_inputs correlation with accuracy: {input_corr:.3f}")
        print(f"  • n_dendrites correlation with accuracy: {dendrite_corr:.3f}")
        print(f"  • Input ratio correlation with accuracy: {input_ratio_corr:.3f}")

        # Find trends
        if input_corr > 0.3:
            print("  → More dendrite inputs tend to improve accuracy")
        elif input_corr < -0.3:
            print("  → Fewer dendrite inputs tend to improve accuracy")

        if dendrite_corr > 0.3:
            print("  → More dendrites tend to improve accuracy")
        elif dendrite_corr < -0.3:
            print("  → Fewer dendrites tend to improve accuracy")

        # Parameter efficiency
        best_result = max(successful_results, key=lambda x: x["final_test_accuracy"])
        print(
            f"  • Best configuration uses {best_result['n_dendrite_inputs']} inputs per dendrite"
        )
        print(
            f"  • Best configuration has {best_result['n_dendrites']} total dendrites"
        )
        print(
            f"  • This represents {best_result['dendrite_input_ratio'] * 100:.1f}% input connectivity"
        )

    return filename


def save_results(results, filename=None):
    """Save results to JSON file"""
    if filename is None:
        filename = (
            f"dendritic_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    # Convert numpy types to Python types for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if hasattr(value, "item"):  # numpy scalar
                json_result[key] = value.item()
            else:
                json_result[key] = value
        json_results.append(json_result)

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {filename}")
    return filename


if __name__ == "__main__":
    print("🧪 Dendritic Layer Ratio Experiment")
    print("=" * 50)

    # Experiment configuration
    n_experiments = 30  # Number of different ratio configurations to test

    print(f"Configuration: Running {n_experiments} different ratio experiments")

    # Run experiment
    results = run_experiment(
        target_params=8000,
        n_epochs=10,
        lr=0.0005,
        weight_decay=0.01,
        batch_size=256,
        subset_size=None,
        test_synaptic_resampling=True,
        n_experiments=n_experiments,
    )

    if results:
        # Show results
        successful_results = [r for r in results if r["success"]]
        print(
            f"\nResults: {len(successful_results)}/{len(results)} experiments successful"
        )

        if successful_results:
            print_best_ratios(results)

            # Create analysis plots
            print("\n📊 Creating analysis plots...")
            plot_results(results)
        else:
            print("All experiments failed!")
            for i, r in enumerate(results):
                if not r["success"]:
                    print(f"  Experiment {i + 1}: {r.get('error', 'Unknown error')}")

        # Save results
        save_results(results)
        print("\nExperiment completed!")
    else:
        print("Experiment failed to produce results.")
