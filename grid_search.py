import itertools
import json
import time

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception):
    import numpy as cp
    print("CuPy not available, using NumPy (CPU)")

from modules import Adam, CrossEntropy, LeakyReLU, Sequential
from main import DendriticLayer, LinearLayer

from training import train
from utils import load_mnist_data, load_cifar10_data


def grid_search(param_grid, dataset="fashion-mnist", subset_size=None, n_epochs=15, lr=0.01, weight_decay=0.001, batch_size=256):
    """
    Performs a grid search over the specified hyperparameter grid.

    Args:
        param_grid (dict): A dictionary where keys are hyperparameter names and values are lists of values to try.
        dataset (str): The dataset to use ('mnist', 'fashion-mnist', 'cifar10').
        subset_size (int, optional): The number of samples to use from the dataset. Defaults to None (full dataset).
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        batch_size (int): Batch size.
    """
    
    # Load data
    if dataset in ["mnist", "fashion-mnist"]:
        X_train, y_train, X_test, y_test = load_mnist_data(dataset=dataset, subset_size=subset_size)
        in_dim = 28 * 28
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(subset_size=subset_size)
        in_dim = 32 * 32 * 3
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    keys, values = zip(*param_grid.items())
    results = []

    print(f"Starting grid search for {len(list(itertools.product(*values)))} combinations...")

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"Testing params: {params}")
        
        start_time = time.time()

        # for repoducability
        cp.random.seed(123125125)

        model = Sequential([
            DendriticLayer(
                in_dim=in_dim,
                n_neurons=params['n_neurons'],
                n_dendrite_inputs=params['n_dendrite_inputs'],
                n_dendrites=params['n_dendrites'],
                strategy="random",
                synaptic_resampling=True,
                percentage_resample=params['percentage_resample'],
                steps_to_resample=params['steps_to_resample'],
            ),
            LeakyReLU(),
            LinearLayer(params['n_neurons'], 10)
        ])
        
        criterion = CrossEntropy()
        optimiser = Adam(model.params(), criterion, lr=lr, weight_decay=weight_decay)

        train_losses, train_accuracy, test_losses, test_accuracy, _ = train(
            X_train, y_train, X_test, y_test, model, criterion, optimiser,
            n_epochs=n_epochs, batch_size=batch_size
        )

        end_time = time.time()

        result = {
            'params': params,
            'train_losses': [float(l) for l in train_losses],
            'train_accuracy': [float(a) for a in train_accuracy],
            'test_losses': [float(l) for l in test_losses],
            'test_accuracy': [float(a) for a in test_accuracy],
            'training_time': end_time - start_time,
            'final_test_accuracy': float(test_accuracy[-1])
        }
        results.append(result)
        
        # Save intermediate results
        with open('grid_search_results.json', 'w') as f:
            json.dump(results, f, indent=4)

    print("Grid search finished.")
    return results

if __name__ == '__main__':
    # param_grid = {
        'percentage_resample': [0.1, 0.2, 0.5, 0.9],
        # percentage_resample: [0.05, 0.1, 0.2, 0.4, 0.8] use next time
        'steps_to_resample': [50, 100, 200, 500, 1000],
        # 'steps_to_resample': [64, 128, 256, 512, 1024], use next time
        'n_dendrite_inputs': [8, 16, 32],
        'n_dendrites': [8, 16, 32],
        'n_neurons': [8, 16, 32, 128]
        # n_neurons': [8, 16, 32, 64, 128] use next time
    }

    grid_search(param_grid, dataset="fashion-mnist", n_epochs=20) # a small number of epochs for testing 