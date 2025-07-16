"""
Example demonstrating hyperparameter tuning functionality.

This example shows how to:
1. Define a model factory function
2. Specify hyperparameter grid
3. Run hyperparameter tuning
4. Analyze results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from modules import DendriticLayer, register_dendritic_hooks
from training import hyperparameter_tuning, compare_models


def dendritic_model_factory(lr, n_dendrite_inputs, n_dendrites, percentage_resample):
    """
    Factory function for creating dendritic models with different hyperparameters.

    Args:
        lr: Learning rate
        n_dendrite_inputs: Number of inputs per dendrite
        n_dendrites: Number of dendrites per neuron
        percentage_resample: Percentage of connections to resample

    Returns:
        tuple: (model, optimizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        DendriticLayer(
            in_dim=784,  # MNIST/Fashion-MNIST input size
            n_neurons=10,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites,
            synaptic_resampling=True,
            percentage_resample=percentage_resample,
            steps_to_resample=128,
        ),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(10, 10),  # 10 classes
    ).to(device)

    # Register hooks for synaptic resampling
    register_dendritic_hooks(model[0])

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def vanilla_model_factory(lr, hidden_dim1, hidden_dim2):
    """
    Factory function for creating vanilla neural networks.

    Args:
        lr: Learning rate
        hidden_dim1: Size of first hidden layer
        hidden_dim2: Size of second hidden layer

    Returns:
        tuple: (model, optimizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(784, hidden_dim1),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(hidden_dim2, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def example_dendritic_hyperparameter_tuning():
    """Example of hyperparameter tuning for dendritic models"""
    print("=" * 60)
    print("DENDRITIC MODEL HYPERPARAMETER TUNING")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define hyperparameter grid
    param_grid = {
        "lr": [0.001, 0.01],
        "n_dendrite_inputs": [16, 32],
        "n_dendrites": [15, 23],
        "percentage_resample": [0.1, 0.25],
    }

    # Run hyperparameter tuning
    results = hyperparameter_tuning(
        model_factory=dendritic_model_factory,
        param_grid=param_grid,
        dataset="fashion-mnist",
        n_epochs=5,  # Use fewer epochs for faster tuning
        batch_size=128,
        subset_size=10000,  # Use subset for faster tuning
        metric="test_accuracy",
        verbose=True,
    )

    print("\nBest hyperparameters found:")
    print(f"Parameters: {results['best_params']}")
    print(f"Best {results['optimization_metric']}: {results['best_score']:.4f}")

    return results


def example_vanilla_hyperparameter_tuning():
    """Example of hyperparameter tuning for vanilla neural networks"""
    print("=" * 60)
    print("VANILLA MODEL HYPERPARAMETER TUNING")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define hyperparameter grid
    param_grid = {
        "lr": [0.001, 0.01, 0.1],
        "hidden_dim1": [32, 64, 128],
        "hidden_dim2": [16, 32, 64],
    }

    # Run hyperparameter tuning
    results = hyperparameter_tuning(
        model_factory=vanilla_model_factory,
        param_grid=param_grid,
        dataset="fashion-mnist",
        n_epochs=5,  # Use fewer epochs for faster tuning
        batch_size=128,
        subset_size=10000,  # Use subset for faster tuning
        metric="test_accuracy",
        verbose=True,
    )

    print("\nBest hyperparameters found:")
    print(f"Parameters: {results['best_params']}")
    print(f"Best {results['optimization_metric']}: {results['best_score']:.4f}")

    return results


def example_compare_best_models():
    """Example of comparing models with their best hyperparameters"""
    print("=" * 60)
    print("COMPARING MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models with manually selected "good" hyperparameters
    # (In practice, you'd use the results from hyperparameter tuning)

    # Best dendritic model (example parameters)
    model1, optimizer1 = dendritic_model_factory(
        lr=0.001, n_dendrite_inputs=32, n_dendrites=23, percentage_resample=0.1
    )

    # Best vanilla model (example parameters)
    model2, optimizer2 = vanilla_model_factory(lr=0.001, hidden_dim1=64, hidden_dim2=32)

    # Baseline dendritic model without resampling
    model3 = nn.Sequential(
        DendriticLayer(
            in_dim=784,
            n_neurons=10,
            n_dendrite_inputs=32,
            n_dendrites=23,
            synaptic_resampling=False,  # No resampling
        ),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Linear(10, 10),
    ).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

    # Compare models
    models_config = [
        (model1, optimizer1, "Tuned Dendritic (Resampling)"),
        (model2, optimizer2, "Tuned Vanilla ANN"),
        (model3, optimizer3, "Baseline Dendritic (No Resampling)"),
    ]

    results = compare_models(
        models_config=models_config,
        dataset="fashion-mnist",
        n_epochs=10,
        batch_size=128,
        subset_size=20000,  # Use larger subset for final comparison
        plot_results=True,
        verbose=True,
    )

    return results


def main():
    """Run all examples"""
    # Example 1: Dendritic model hyperparameter tuning
    dendritic_results = example_dendritic_hyperparameter_tuning()

    print("\n" + "=" * 80 + "\n")

    # Example 2: Vanilla model hyperparameter tuning
    vanilla_results = example_vanilla_hyperparameter_tuning()

    print("\n" + "=" * 80 + "\n")

    # Example 3: Compare best models
    comparison_results = example_compare_best_models()

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING EXAMPLES COMPLETED")
    print("=" * 80)

    return {
        "dendritic_tuning": dendritic_results,
        "vanilla_tuning": vanilla_results,
        "final_comparison": comparison_results,
    }


if __name__ == "__main__":
    results = main()
