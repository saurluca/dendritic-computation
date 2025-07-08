import itertools
import json
import time
import os

try:
    import cupy as cp

    cp.cuda.Device(0).compute_capability
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception):
    import numpy as cp

    print("CuPy not available, using NumPy (CPU)")

from modules import Adam, CrossEntropy, LeakyReLU, Sequential
from main import DendriticLayer, LinearLayer

from training import train
from utils import load_mnist_data, load_cifar10_data


def grid_search(
    param_grid,
    dataset="fashion-mnist",
    subset_size=None,
    n_epochs=15,
    lr=0.002,
    weight_decay=0.01,
    batch_size=256,
    results_file="grid_search_results_small.json",
):
    """
    Performs a grid search over the specified hyperparameter grid.
    Results are appended to existing results file if it exists.

    Args:
        param_grid (dict): A dictionary where keys are hyperparameter names and values are lists of values to try.
        dataset (str): The dataset to use ('mnist', 'fashion-mnist', 'cifar10').
        subset_size (int, optional): The number of samples to use from the dataset. Defaults to None (full dataset).
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        batch_size (int): Batch size.
        results_file (str): Path to the results file.
    """

    # Load existing results if file exists
    existing_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
            print(
                f"Loaded {len(existing_results)} existing results from {results_file}"
            )
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"No existing results found or file corrupted, starting fresh")
            existing_results = []
    else:
        print(f"No existing results file found, starting fresh")

    # Load data
    if dataset in ["mnist", "fashion-mnist"]:
        X_train, y_train, X_test, y_test = load_mnist_data(
            dataset=dataset, subset_size=subset_size
        )
        in_dim = 28 * 28
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(subset_size=subset_size)
        in_dim = 32 * 32 * 3
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    keys, values = zip(*param_grid.items())
    new_results = []

    total_combinations = len(list(itertools.product(*values)))
    print(f"Starting grid search for {total_combinations} combinations...")
    print(f"Results will be appended to existing {len(existing_results)} results")

    for i, v in enumerate(itertools.product(*values)):
        params = dict(zip(keys, v))
        print(f"Testing params ({i + 1}/{total_combinations}): {params}")

        start_time = time.time()

        # Use seed from parameters for reproducibility
        seed = params.get("seed", int(time.time() * 1000000) % 2**32)
        cp.random.seed(seed)

        model = Sequential(
            [
                DendriticLayer(
                    in_dim=in_dim,
                    n_neurons=params["n_neurons"],
                    n_dendrite_inputs=params["n_dendrite_inputs"],
                    n_dendrites=params["n_dendrites"],
                    strategy="random",
                    synaptic_resampling=True,
                    percentage_resample=params["percentage_resample"],
                    steps_to_resample=params["steps_to_resample"],
                ),
                LeakyReLU(),
                LinearLayer(params["n_neurons"], n_classes),
            ]
        )

        criterion = CrossEntropy()
        optimiser = Adam(model.params(), criterion, lr=lr, weight_decay=weight_decay)

        train_losses, train_accuracy, test_losses, test_accuracy, _ = train(
            X_train,
            y_train,
            X_test,
            y_test,
            model,
            criterion,
            optimiser,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

        end_time = time.time()

        result = {
            "params": params,  # This now includes the seed
            "train_losses": [float(l) for l in train_losses],
            "train_accuracy": [float(a) for a in train_accuracy],
            "test_losses": [float(l) for l in test_losses],
            "test_accuracy": [float(a) for a in test_accuracy],
            "training_time": end_time - start_time,
            "final_test_accuracy": float(test_accuracy[-1]),
            "seed": seed,  # Also store seed separately for easy access
        }
        new_results.append(result)

        # Combine existing and new results and save
        all_results = existing_results + new_results
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)

        print(
            f"  -> Accuracy: {float(test_accuracy[-1]):.4f}, Time: {end_time - start_time:.2f}s"
        )

    print(f"Grid search finished. Added {len(new_results)} new results.")
    print(
        f"Total results in {results_file}: {len(existing_results) + len(new_results)}"
    )
    return existing_results + new_results


if __name__ == "__main__":
    # Base parameter grid
    base_param_grid = {
        # 'steps_to_resample': [4, 8, 16, 32, 64, 128, 256, 512],
        # 'percentage_resample': [0.05, 0.1, 0.25, 0.5, 0.9],
        # 'n_dendrite_inputs': [1, 2, 4, 8, 16, 32],
        # 'n_dendrites': [1, 2, 4, 8, 16, 32],
        "steps_to_resample": [128],
        "percentage_resample": [0.25],
        "n_dendrite_inputs": [16, 32, 64, 128, 256],
        "n_dendrites": [16, 32, 64, 128, 256, 512],
        "n_neurons": [10, 16, 32, 64, 128, 256, 512],
    }

    # Example 1: Single run with specific seed
    param_grid_with_seed = base_param_grid.copy()
    param_grid_with_seed["seed"] = [1229434093, 4562312312, 7899873570912395]

    print("Running grid search with seed=42...")
    grid_search(
        param_grid_with_seed,
        dataset="mnist",
        n_epochs=20,
        lr=0.002,
        weight_decay=0.01,
        batch_size=256,
    )

    # Example 2: Multiple runs with different seeds (uncomment to run)
    # for seed_value in [123, 456, 789]:
    #     param_grid_with_seed = base_param_grid.copy()
    #     param_grid_with_seed['seed'] = [seed_value]
    #     print(f"\nRunning grid search with seed={seed_value}...")
    #     grid_search(param_grid_with_seed, dataset="mnist", n_epochs=20, lr=0.002, weight_decay=0.01, batch_size=256)

    # Example 3: Multiple seeds in single run (will create many combinations)
    # param_grid_multi_seed = base_param_grid.copy()
    # param_grid_multi_seed['seed'] = [42, 123, 456]  # Multiple seeds
    # grid_search(param_grid_multi_seed, dataset="mnist", n_epochs=20, lr=0.002, weight_decay=0.01, batch_size=256)
