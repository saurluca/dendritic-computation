# Dendritic Computation PyTorch Implementation

This project implements biologically-inspired dendritic neural networks with synaptic resampling in PyTorch, converted from the original NumPy/CuPy implementation.

## Project Structure

```
advanced/
├── modules.py                    # DendriticLayer implementation
├── data.py                      # Data loading utilities
├── utils.py                     # Visualization and analysis utilities
├── training.py                  # Training functions and model comparison
├── main.py                      # Main experiment script
├── example_hyperparameter_tuning.py  # Hyperparameter tuning examples
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Features

### Core Components

1. **DendriticLayer** (`modules.py`): 
   - Biologically-inspired sparse dendritic neural network layer
   - Synaptic resampling for improved learning and generalization
   - Configurable dendrite structure and sparsity

2. **Training Framework** (`training.py`):
   - Efficient training and evaluation functions
   - Model comparison utilities
   - Hyperparameter tuning with grid search

3. **Data Handling** (`data.py`):
   - MNIST and Fashion-MNIST dataset loading
   - Data preprocessing and normalization
   - PyTorch DataLoader integration

4. **Visualization** (`utils.py`):
   - Dendritic weight visualization
   - Training curve plotting
   - Eigenvalue analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Model Comparison

```python
from modules import DendriticLayer, register_dendritic_hooks
from training import compare_models
import torch
import torch.nn as nn
import torch.optim as optim

# Create models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dendritic model with resampling
model1 = nn.Sequential(
    DendriticLayer(784, 10, 32, 23, synaptic_resampling=True),
    nn.LeakyReLU(0.1),
    nn.Linear(10, 10)
).to(device)
register_dendritic_hooks(model1[0])

# Vanilla neural network
model2 = nn.Sequential(
    nn.Linear(784, 64),
    nn.LeakyReLU(0.1),
    nn.Linear(64, 10)
).to(device)

# Prepare configuration
models_config = [
    (model1, optim.Adam(model1.parameters(), lr=0.001), "Dendritic"),
    (model2, optim.Adam(model2.parameters(), lr=0.001), "Vanilla")
]

# Train and compare
results = compare_models(
    models_config=models_config,
    dataset="fashion-mnist",
    n_epochs=20,
    batch_size=128
)
```

### Hyperparameter Tuning

```python
from training import hyperparameter_tuning

def model_factory(lr, n_dendrite_inputs, n_dendrites):
    model = nn.Sequential(
        DendriticLayer(784, 10, n_dendrite_inputs, n_dendrites),
        nn.LeakyReLU(0.1),
        nn.Linear(10, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

param_grid = {
    'lr': [0.001, 0.01],
    'n_dendrite_inputs': [16, 32, 48],
    'n_dendrites': [15, 23, 31]
}

best_params = hyperparameter_tuning(
    model_factory=model_factory,
    param_grid=param_grid,
    dataset="fashion-mnist",
    n_epochs=10,
    metric='test_accuracy'
)
```

### Quick Start

Run the main experiment:
```bash
cd advanced/
python main.py
```

Run hyperparameter tuning examples:
```bash
python example_hyperparameter_tuning.py
```

## Key Parameters

### DendriticLayer Parameters

- `in_dim`: Input dimension (e.g., 784 for MNIST)
- `n_neurons`: Number of neurons in the layer
- `n_dendrite_inputs`: Number of inputs each dendrite receives
- `n_dendrites`: Number of dendrites per neuron
- `synaptic_resampling`: Enable/disable synaptic resampling
- `percentage_resample`: Percentage of connections to resample (0.0-1.0)
- `steps_to_resample`: Number of backward passes between resampling

### Training Parameters

- `n_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `dataset`: "mnist" or "fashion-mnist"
- `subset_size`: Use subset of data (for faster experimentation)

## Functions Reference

### Training Functions (`training.py`)

- `train_epoch()`: Train model for one epoch
- `evaluate()`: Evaluate model on test data
- `train_model()`: Complete training loop with metrics tracking
- `compare_models()`: Train and compare multiple models
- `hyperparameter_tuning()`: Grid search hyperparameter optimization

### Utility Functions (`utils.py`)

- `calculate_eigenvalues()`: Analyze weight matrix eigenvalues
- `plot_dendritic_weights_single_image()`: Visualize single neuron weights
- `plot_dendritic_weights_full_model()`: Visualize all dendritic weights
- `count_parameters()`: Count model parameters
- `plot_training_curves()`: Plot training/validation curves

### Data Functions (`data.py`)

- `load_mnist_data()`: Load and preprocess MNIST/Fashion-MNIST
- `create_data_loaders()`: Create PyTorch DataLoaders
- `get_sample_image()`: Get sample image for visualization

## Model Architecture

The DendriticLayer implements a biologically-inspired architecture:

1. **Dendrites**: Each neuron has multiple dendrites that receive sparse inputs
2. **Soma**: Aggregates dendrite outputs through learned weights
3. **Sparsity**: Each dendrite connects to only a subset of inputs
4. **Resampling**: Weak connections are periodically replaced with new random connections

This architecture mimics biological neural plasticity and can improve generalization.

## Performance Notes

- Use CUDA when available for faster training
- Consider using subset_size for quick experimentation
- Hyperparameter tuning can be time-consuming; start with small grids
- The synaptic resampling adds computational overhead but can improve performance

## Citation

This implementation is based on research in dendritic computation and synaptic plasticity. If you use this code, please cite the relevant papers and acknowledge the biological inspiration.

## License

[Add your license information here] 