# %%
"""
Hyperparameter Search for Dendritic Neural Networks

This script performs hyperparameter optimization to find the best dendritic model
architecture while staying within a specified parameter budget.
"""

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp
    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")

import itertools
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import random
from dataclasses import dataclass

from modules import (
    Adam,
    CrossEntropy,
    LeakyReLU,
    Sequential,
    DendriticLayer,
    LinearLayer,
)
from utils import load_mnist_data, load_cifar10_data
from training import train


@dataclass
class SearchResult:
    """Container for hyperparameter search results"""
    params: Dict[str, Any]
    test_accuracy: float
    train_accuracy: float
    total_params: int
    train_time: float
    final_test_loss: float


class HyperparameterSearcher:
    """
    Hyperparameter search for dendritic neural networks with parameter budget constraints.
    """
    
    def __init__(
        self,
        dataset: str = "fashion-mnist",
        subset_size: Optional[int] = None,
        max_params: int = 8000,
        min_param_ratio: float = 0.8,
        n_epochs: int = 15,
        lr: float = 0.0005,
        weight_decay: float = 0.01,
        batch_size: int = 256,
        n_classes: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the hyperparameter searcher.
        
        Args:
            dataset: Dataset to use ("mnist", "fashion-mnist", "cifar10")
            subset_size: Optional subset size for faster testing
            max_params: Maximum number of parameters allowed
            min_param_ratio: Minimum ratio of max_params that model should use (default: 0.8 = 80%)
            n_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            batch_size: Training batch size
            n_classes: Number of output classes
            verbose: Whether to print detailed information
        """
        self.dataset = dataset
        self.subset_size = subset_size
        self.max_params = max_params
        self.min_params = int(max_params * min_param_ratio)
        self.min_param_ratio = min_param_ratio
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.verbose = verbose
        
        # Load data
        self._load_data()
        
        # Results storage
        self.results: List[SearchResult] = []
        
    def _load_data(self):
        """Load the specified dataset"""
        if self.dataset in ["mnist", "fashion-mnist"]:
            self.X_train, self.y_train, self.X_test, self.y_test = load_mnist_data(
                dataset=self.dataset, subset_size=self.subset_size
            )
            self.in_dim = 28 * 28
        elif self.dataset == "cifar10":
            self.X_train, self.y_train, self.X_test, self.y_test = load_cifar10_data(
                subset_size=self.subset_size
            )
            self.in_dim = 32 * 32 * 3
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")
        
        if self.verbose:
            print(f"Loaded {self.dataset} dataset:")
            print(f"  Train samples: {len(self.X_train)}")
            print(f"  Test samples: {len(self.X_test)}")
            print(f"  Input dimension: {self.in_dim}")
            print(f"Parameter budget: {self.min_params}-{self.max_params} ({self.min_param_ratio*100:.0f}%-100%)")
    
    def calculate_dendritic_params(
        self,
        in_dim: int,
        n_neurons: int,
        n_dendrites: int,
        n_dendrite_inputs: int,
        soma_enabled: bool = True,
        additional_layers: Optional[List[Tuple[int, int]]] = None
    ) -> int:
        """
        Calculate the total number of parameters for a dendritic model architecture.
        
        Args:
            in_dim: Input dimension
            n_neurons: Number of neurons in dendritic layer
            n_dendrites: Number of dendrites per neuron
            n_dendrite_inputs: Number of inputs per dendrite
            soma_enabled: Whether soma is enabled
            additional_layers: List of (in_dim, out_dim) for additional linear layers
            
        Returns:
            Total number of parameters
        """
        if soma_enabled:
            # With soma: full dendritic layer
            n_soma_connections = n_dendrites * n_neurons
            dendrite_params = n_dendrite_inputs * n_soma_connections + n_soma_connections  # weights + biases
            soma_params = n_dendrites * n_neurons + n_neurons  # connections + biases
            dendritic_layer_params = dendrite_params + soma_params
            output_dim = n_neurons
        else:
            # Without soma: dendrites are direct outputs
            dendrite_params = n_dendrite_inputs * n_dendrites + n_dendrites  # weights + biases
            dendritic_layer_params = dendrite_params
            output_dim = n_dendrites
        
        total_params = dendritic_layer_params
        
        # Add additional layers if specified
        if additional_layers:
            current_dim = output_dim
            for layer_in, layer_out in additional_layers:
                if layer_in == -1:  # Use previous layer's output
                    layer_in = current_dim
                total_params += layer_in * layer_out + layer_out  # weights + biases
                current_dim = layer_out
        else:
            # Default: add output layer
            total_params += output_dim * self.n_classes + self.n_classes
        
        return total_params
    
    @staticmethod
    def estimate_params(
        in_dim: int,
        n_neurons: int, 
        n_dendrites: int,
        n_dendrite_inputs: int,
        soma_enabled: bool = True,
        n_classes: int = 10,
        architecture: str = 'dendritic_simple',
        hidden_dim: Optional[int] = None
    ) -> int:
        """
        Static method to estimate parameters before creating a searcher.
        Useful for planning parameter budgets.
        
        Args:
            in_dim: Input dimension
            n_neurons: Number of neurons 
            n_dendrites: Number of dendrites
            n_dendrite_inputs: Inputs per dendrite
            soma_enabled: Whether soma is enabled
            n_classes: Number of output classes
            architecture: Architecture type
            hidden_dim: Hidden layer dimension (for dendritic_with_hidden)
            
        Returns:
            Estimated parameter count
        """
        if architecture == 'dendritic_simple':
            if soma_enabled:
                dendrite_params = n_dendrite_inputs * n_dendrites * n_neurons + n_dendrites * n_neurons
                soma_params = n_dendrites * n_neurons + n_neurons  
                output_params = n_neurons * n_classes + n_classes
                return dendrite_params + soma_params + output_params
            else:
                dendrite_params = n_dendrite_inputs * n_dendrites + n_dendrites
                output_params = n_dendrites * n_classes + n_classes
                return dendrite_params + output_params
                
        elif architecture == 'soma_disabled':
            dendrite_params = n_dendrite_inputs * n_dendrites + n_dendrites
            hidden_params = n_dendrites * n_neurons + n_neurons
            output_params = n_neurons * n_classes + n_classes
            return dendrite_params + hidden_params + output_params
            
        elif architecture == 'dendritic_with_hidden':
            if hidden_dim is None:
                hidden_dim = n_neurons
            if soma_enabled:
                dendrite_params = n_dendrite_inputs * n_dendrites * n_neurons + n_dendrites * n_neurons
                soma_params = n_dendrites * n_neurons + n_neurons
                hidden_params = n_neurons * hidden_dim + hidden_dim
                output_params = hidden_dim * n_classes + n_classes
                return dendrite_params + soma_params + hidden_params + output_params
            else:
                dendrite_params = n_dendrite_inputs * n_dendrites + n_dendrites
                hidden_params = n_dendrites * hidden_dim + hidden_dim
                output_params = hidden_dim * n_classes + n_classes
                return dendrite_params + hidden_params + output_params
        
        return 0
    
    def create_model(self, params: Dict[str, Any]) -> Tuple[Sequential, int]:
        """
        Create a model with the given hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Tuple of (model, total_parameters)
        """
        # Extract parameters with defaults
        n_neurons = params.get('n_neurons', 10)
        n_dendrites = params.get('n_dendrites', 6)
        n_dendrite_inputs = params.get('n_dendrite_inputs', 128)
        soma_enabled = params.get('soma_enabled', True)
        strategy = params.get('strategy', 'random')
        synaptic_resampling = params.get('synaptic_resampling', True)
        percentage_resample = params.get('percentage_resample', 0.25)
        steps_to_resample = params.get('steps_to_resample', 128)
        architecture = params.get('architecture', 'dendritic_simple')
        
        if architecture == 'dendritic_simple':
            # Simple dendritic model with one output layer
            layers = [
                DendriticLayer(
                    self.in_dim,
                    n_neurons,
                    n_dendrite_inputs=n_dendrite_inputs,
                    n_dendrites=n_dendrites,
                    strategy=strategy,
                    soma_enabled=soma_enabled,
                    synaptic_resampling=synaptic_resampling,
                    percentage_resample=percentage_resample,
                    steps_to_resample=steps_to_resample,
                ),
                LinearLayer(n_neurons if soma_enabled else n_dendrites, self.n_classes),
            ]
            
        elif architecture == 'dendritic_with_hidden':
            # Dendritic model with additional hidden layer
            hidden_dim = params.get('hidden_dim', n_neurons)
            layers = [
                DendriticLayer(
                    self.in_dim,
                    n_neurons,
                    n_dendrite_inputs=n_dendrite_inputs,
                    n_dendrites=n_dendrites,
                    strategy=strategy,
                    soma_enabled=soma_enabled,
                    synaptic_resampling=synaptic_resampling,
                    percentage_resample=percentage_resample,
                    steps_to_resample=steps_to_resample,
                ),
                LeakyReLU(),
                LinearLayer(n_neurons if soma_enabled else n_dendrites, hidden_dim),
                LeakyReLU(),
                LinearLayer(hidden_dim, self.n_classes),
            ]
            
        elif architecture == 'soma_disabled':
            # Dendritic model without soma (like in main.py)
            layers = [
                DendriticLayer(
                    self.in_dim,
                    n_neurons,
                    n_dendrite_inputs=n_dendrite_inputs,
                    n_dendrites=n_dendrites,
                    strategy=strategy,
                    soma_enabled=False,
                    synaptic_resampling=synaptic_resampling,
                    percentage_resample=percentage_resample,
                    steps_to_resample=steps_to_resample,
                ),
                LinearLayer(n_dendrites, self.n_classes),
                LeakyReLU(),
                LinearLayer(n_neurons, self.n_classes),
            ]
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        model = Sequential(layers)
        total_params = model.num_params()
        
        return model, total_params
    
    def is_valid_config(self, params: Dict[str, Any]) -> bool:
        """
        Check if a parameter configuration is valid (within parameter budget and uses enough capacity).
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            model, total_params = self.create_model(params)
            
            # Check if within budget
            if total_params > self.max_params:
                if self.verbose:
                    print(f"Config exceeds budget: {total_params} > {self.max_params}")
                return False
            
            # Check if uses at least minimum ratio of available parameters
            if total_params < self.min_params:
                if self.verbose:
                    print(f"Config uses too few parameters: {total_params} < {self.min_params} ({self.min_param_ratio*100:.0f}% of budget)")
                return False
                
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Invalid config {params}: {e}")
            return False
    
    def train_and_evaluate(self, params: Dict[str, Any]) -> Optional[SearchResult]:
        """
        Train and evaluate a model with given hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            SearchResult if training successful, None otherwise
        """
        if not self.is_valid_config(params):
            return None
        
        try:
            # Set random seed for reproducibility
            seed = params.get('seed', int(time.time() * 1000000) % 2**32)
            cp.random.seed(seed)
            
            start_time = time.time()
            
            # Create model
            model, total_params = self.create_model(params)
            
            if self.verbose:
                print(f"  Model parameters: {total_params}/{self.max_params}")
            
            # Create optimizer and criterion
            criterion = CrossEntropy()
            optimizer = Adam(model.params(), criterion, lr=self.lr, weight_decay=self.weight_decay)
            
            # Train model
            train_losses, train_accuracies, test_losses, test_accuracies = train(
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                model,
                criterion,
                optimizer,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
            )
            
            train_time = time.time() - start_time
            
            # Create result
            result = SearchResult(
                params=params.copy(),
                test_accuracy=test_accuracies[-1],
                train_accuracy=train_accuracies[-1],
                total_params=total_params,
                train_time=train_time,
                final_test_loss=test_losses[-1]
            )
            
            if self.verbose:
                print(f"  Test accuracy: {result.test_accuracy:.3f}")
                print(f"  Train time: {result.train_time:.1f}s")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"  Training failed: {e}")
            return None
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        max_trials: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            max_trials: Maximum number of trials to run (None for all combinations)
            
        Returns:
            List of SearchResult objects
        """
        keys, values = zip(*param_grid.items())
        combinations = list(itertools.product(*values))
        
        if max_trials and len(combinations) > max_trials:
            combinations = random.sample(combinations, max_trials)
        
        total_combinations = len(combinations)
        valid_configs = 0
        
        if self.verbose:
            print(f"Starting grid search with {total_combinations} combinations")
            print(f"Parameter budget: {self.min_params}-{self.max_params} parameters ({self.min_param_ratio*100:.0f}%-100%)")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            
            if self.verbose:
                print(f"\nTrial {i+1}/{total_combinations}: {params}")
            
            result = self.train_and_evaluate(params)
            if result:
                self.results.append(result)
                valid_configs += 1
        
        if self.verbose:
            print(f"\nGrid search completed: {valid_configs}/{total_combinations} valid configurations")
        
        return self.results
    
    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_trials: int = 50
    ) -> List[SearchResult]:
        """
        Perform random search over parameter space.
        
        Args:
            param_distributions: Dictionary mapping parameter names to distributions
            n_trials: Number of random trials to perform
            
        Returns:
            List of SearchResult objects
        """
        if self.verbose:
            print(f"Starting random search with {n_trials} trials")
            print(f"Parameter budget: {self.min_params}-{self.max_params} parameters ({self.min_param_ratio*100:.0f}%-100%)")
        
        valid_configs = 0
        
        for i in range(n_trials):
            # Sample parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[param_name] = random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Assume (min, max) for integers
                    params[param_name] = random.randint(distribution[0], distribution[1])
                elif isinstance(distribution, dict):
                    if distribution['type'] == 'uniform':
                        params[param_name] = random.uniform(distribution['low'], distribution['high'])
                    elif distribution['type'] == 'choice':
                        params[param_name] = random.choice(distribution['values'])
                    elif distribution['type'] == 'randint':
                        params[param_name] = random.randint(distribution['low'], distribution['high'])
                else:
                    params[param_name] = distribution
            
            if self.verbose:
                print(f"\nTrial {i+1}/{n_trials}: {params}")
            
            result = self.train_and_evaluate(params)
            if result:
                self.results.append(result)
                valid_configs += 1
        
        if self.verbose:
            print(f"\nRandom search completed: {valid_configs}/{n_trials} valid configurations")
        
        return self.results
    
    def get_best_results(self, n: int = 5) -> List[SearchResult]:
        """Get the top n results sorted by test accuracy"""
        return sorted(self.results, key=lambda x: x.test_accuracy, reverse=True)[:n]
    
    def print_results(self, n: int = 5):
        """Print the top n results"""
        best_results = self.get_best_results(n)
        
        print(f"\n{'='*80}")
        print(f"TOP {min(n, len(best_results))} RESULTS")
        print(f"{'='*80}")
        
        for i, result in enumerate(best_results, 1):
            print(f"\nRank {i}:")
            print(f"  Test Accuracy: {result.test_accuracy:.4f}")
            print(f"  Train Accuracy: {result.train_accuracy:.4f}")
            print(f"  Parameters: {result.total_params}/{self.max_params}")
            print(f"  Train Time: {result.train_time:.1f}s")
            print(f"  Configuration: {result.params}")
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'params': result.params,
                'test_accuracy': float(result.test_accuracy),
                'train_accuracy': float(result.train_accuracy),
                'total_params': int(result.total_params),
                'train_time': float(result.train_time),
                'final_test_loss': float(result.final_test_loss)
            })
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {filename}")


def main():
    """Example usage of the hyperparameter searcher"""
    
    # Example parameter estimation before creating searcher
    print("PARAMETER ESTIMATION EXAMPLE:")
    estimated = HyperparameterSearcher.estimate_params(
        in_dim=784, n_neurons=10, n_dendrites=6, n_dendrite_inputs=128,
        soma_enabled=True, architecture='dendritic_simple'
    )
    print(f"Estimated parameters for sample config: {estimated}")
    
    # Configuration - models must use at least 80% of parameter budget
    searcher = HyperparameterSearcher(
        dataset="fashion-mnist",
        max_params=8000,
        min_param_ratio=0.8,  # Require at least 80% of budget
        n_epochs=15,
        lr=0.0005,
        weight_decay=0.01,
        batch_size=256,
        verbose=True
    )
    
    # Example 1: Grid search for simple dendritic architecture
    print("EXAMPLE 1: Grid Search for Simple Dendritic Architecture")
    param_grid = {
        'n_neurons': [8, 10, 12],
        'n_dendrites': [4, 6, 8],
        'n_dendrite_inputs': [64, 96, 128],
        'soma_enabled': [True],
        'architecture': ['dendritic_simple'],
        'synaptic_resampling': [True, False],
        'percentage_resample': [0.15, 0.25, 0.35],
        'seed': [42, 123, 456]  # Multiple seeds for robustness
    }
    
    searcher.grid_search(param_grid, max_trials=20)
    searcher.print_results(5)
    searcher.save_results("grid_search_results.json")
    
    # Reset for next search
    searcher.results = []
    
    # Example 2: Random search exploring larger space
    print("\n\nEXAMPLE 2: Random Search for Architecture Optimization")
    param_distributions = {
        'n_neurons': (5, 15),
        'n_dendrites': (3, 10),
        'n_dendrite_inputs': (32, 200),
        'soma_enabled': [True, False],
        'architecture': ['dendritic_simple', 'soma_disabled'],
        'synaptic_resampling': [True, False],
        'percentage_resample': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
        'steps_to_resample': {'type': 'choice', 'values': [64, 128, 256]},
        'seed': {'type': 'randint', 'low': 1, 'high': 10000}
    }
    
    searcher.random_search(param_distributions, n_trials=30)
    searcher.print_results(5)
    searcher.save_results("random_search_results.json")


if __name__ == "__main__":
    main() 