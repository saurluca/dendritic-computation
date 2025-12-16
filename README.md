# Dendritic Computation: Biologically-Inspired Neural Networks with Structural Plasticity

A parameter-efficient neural network implementation inspired by dendritic computation in biological neurons based on the work from Chavlis & Poirazi, 2025 [2]. This project explores how sparse dendritic architectures with synaptic resampling can achieve competitive performance with significantly fewer parameters than traditional deep learning models. 

## 🧠 Overview

This project implements **dendritic neural networks** that mimic the computational properties of biological dendrites - the branched structures that receive and process inputs in real neurons. By incorporating structural plasticity through **synaptic resampling**, these networks can dynamically reorganize their connectivity during training, leading to improved learning efficiency and generalization with minimal parameters.

### Key Innovation: Synaptic Resampling

Traditional neural networks have fixed connectivity that only changes through weight updates. Our dendritic networks implement **structural plasticity** by periodically replacing weak synaptic connections with new random connections during training. This biological mechanism:

- 🎯 **Improves exploration** of the solution space
- 🔄 **Prevents overfitting** by maintaining diversity in connectivity
- 📉 **Reduces parameters** while maintaining or improving accuracy
- 🧬 **Mimics biological learning** observed in real neural systems

## ✨ Key Features

- **Sparse Dendritic Architecture**: Each dendrite connects to only a subset of inputs, dramatically reducing parameters
- **Synaptic Resampling**: Dynamic connection reorganization during training for improved learning
- **Multiple Connectivity Strategies**: Random, local receptive fields, and fully-connected modes
- **GPU Acceleration**: Support for both CuPy (GPU) and NumPy (CPU) backends
- **Parameter-Efficient**: Achieves competitive accuracy with 10-100x fewer parameters
- **Comprehensive Tooling**: Hyperparameter search, visualization, and model comparison utilities

## 📊 Performance Results

### MNIST Classification

| Accuracy | Parameters | Configuration | Reduction vs Baseline |
|----------|-----------|---------------|----------------------|
| 81% | 180 | 18 synapses / 10 dendrites | ~100x fewer params |
| 90% | 600 | 60 synapses / 10 dendrites | ~50x fewer params |
| 94.7% | 2,890 | 16 inputs/dendrite, 16 dendrites, 10 neurons | ~20x fewer params |
| **96.4%** | **~8,000** | With synaptic resampling (50 epochs) | **10x fewer params** |

### Fashion-MNIST Classification

- **80% accuracy** with only **110 parameters**
- **87%+ accuracy** with ~8,000 parameters using synaptic resampling

### Key Findings

**50 Epochs MNIST Training (with synaptic resampling):**
- Train loss: **0.0768** (vs 0.2162 baseline)
- Test loss: **0.1175** (vs 0.2338 baseline)  
- Train accuracy: **97.4%** (vs 93.5% baseline)
- Test accuracy: **96.4%** (vs 93.0% baseline)

## 🏗️ Architecture

### Dendritic Layer Structure

```
Input (784 features)
    ↓
[Dendrite 1] ← sparse subset (e.g., 16 inputs)
[Dendrite 2] ← sparse subset (16 inputs)
[Dendrite 3] ← sparse subset (16 inputs)
    ...
[Dendrite N] ← sparse subset (16 inputs)
    ↓
Dendrite Activations (LeakyReLU)
    ↓
Soma Integration (optional)
    ↓
Output Layer
```

Each dendrite:
1. Receives inputs from a **sparse subset** of features
2. Applies learned **weights and bias**
3. Passes through **non-linear activation**
4. Can be **resampled** during training if connection is weak

### Model Configurations

**1. Simple Dendritic Model**
```python
DendriticLayer(in_dim=784, n_neurons=10, n_dendrites=16, n_dendrite_inputs=16)
→ Output
```

**2. Dendritic with Linear Output**
```python
DendriticLayer(in_dim=784, n_dendrites=25, n_dendrite_inputs=64, soma_enabled=False)
→ LeakyReLU
→ LinearLayer(25, 10)
```

**3. Deep Dendritic Model**
```python
DendriticLayer(...)
→ LeakyReLU
→ LinearLayer(hidden_dim)
→ LeakyReLU  
→ LinearLayer(n_classes)
```

## 🚀 Installation

### Requirements

- Python ≥3.12
- CUDA-capable GPU (optional, for acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/saurluca/dendritic-computation.git
cd dendritic-computation

# Install dependencies (using uv)
pip install uv
uv sync

# Or install directly with pip
pip install cupy-cuda12x matplotlib pandas torch torchvision scikit-learn seaborn tqdm
```

### Dependencies

- **cupy-cuda12x**: GPU acceleration (falls back to NumPy if unavailable)
- **torch/torchvision**: PyTorch implementation (in `advanced/` directory)
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Metrics and utilities

## 📖 Usage

### Basic Training Example

```python
from modules import DendriticLayer, LinearLayer, LeakyReLU, Sequential, Adam, CrossEntropy
from utils import load_mnist_data
from training import train_models

# Load data
X_train, y_train, X_test, y_test = load_mnist_data(dataset="mnist")

# Create dendritic model
model = Sequential([
    DendriticLayer(
        in_dim=784,
        n_neurons=10,
        n_dendrites=16,
        n_dendrite_inputs=16,
        synaptic_resampling=True,  # Enable structural plasticity
        percentage_resample=0.25,   # Resample 25% of weakest connections
        steps_to_resample=128       # Every 128 training steps
    ),
    LeakyReLU(alpha=0.1),
    LinearLayer(10, 10)
])

# Train
optimizer = Adam(model.layers, lr=0.0005, weight_decay=0.01)
criterion = CrossEntropy()

results = train_models(
    models=[model],
    optimizers=[optimizer],
    criterion=criterion,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_epochs=20,
    batch_size=256
)
```

### Running Experiments

```bash
# Run ratio optimization experiment
python experiment_dendritic_ratios.py

# Run hyperparameter search
cd hp_search
python hyperparameter_examples.py

# Run PyTorch version
cd advanced
python main.py
```

## 🔬 Key Experimental Findings

### 1. LeakyReLU is Critical
**LeakyReLU activation is essential** for dendritic models to outperform vanilla networks. Without it, performance degrades significantly.

### 2. Data Scale Matters
- With small datasets (10² samples): Vanilla models perform better
- With larger datasets (10³+ samples): Dendritic models excel, especially with LeakyReLU

### 3. Training Robustness
- **Dendritic models** are robust to lack of batch shuffling
- **Vanilla models** lose significant performance (92% → 83% accuracy) without shuffling

### 4. Connectivity Strategies
- **Random connectivity**: Best overall performance
- **Local receptive fields**: Requires more dendrites (≥32) to be effective
- **Fully-connected**: Defeats the purpose of sparsity

### 5. Synaptic Resampling Benefits
- Models can train for **longer without overfitting**
- **Both train and test metrics** continue improving
- Maintains performance with **fewer parameters**

### 6. Performance vs Parameters Trade-off

![Weight Distributions](experiments/weight_distributions.png)

*Weight distributions showing the sparse connectivity patterns in dendritic layers*

## 📁 Project Structure

```
dendritic-computation/
├── modules.py                    # Core dendritic layer (NumPy/CuPy)
├── training.py                   # Training utilities
├── utils.py                      # Data loading and visualization
├── main.py                       # Basic experiments
├── experiment_dendritic_ratios.py # Ratio optimization experiments
│
├── advanced/                     # PyTorch implementation
│   ├── modules.py               # DendriticLayer in PyTorch
│   ├── training.py              # PyTorch training functions
│   ├── data.py                  # DataLoader utilities
│   ├── utils.py                 # Visualization tools
│   └── main.py                  # PyTorch experiments
│
├── hp_search/                    # Hyperparameter optimization
│   ├── hyperparameter_search.py # Search framework
│   ├── hyperparameter_examples.py # Example searches
│   └── HYPERPARAMETER_SEARCH_README.md
│
├── experiments/                  # Experimental scripts
│   └── t_modules.py             # Transformer experiments
│
└── notebooks/                    # Jupyter notebooks
```

## 🎛️ Key Parameters

### Dendritic Layer Configuration

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_dendrites` | Number of dendrites per neuron | 4-32 |
| `n_dendrite_inputs` | Inputs per dendrite (sparsity) | 8-128 |
| `n_neurons` | Number of neurons/output units | 10-50 |
| `soma_enabled` | Enable soma aggregation layer | True/False |
| `strategy` | Connectivity: "random", "local-receptive-fields" | "random" |

### Synaptic Resampling

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `synaptic_resampling` | Enable structural plasticity | True/False |
| `percentage_resample` | Fraction of connections to replace | 0.05-0.5 |
| `steps_to_resample` | Training steps between resampling | 64-256 |
| `probabilistic_resampling` | Weight-based resampling probability | True/False |

### Training

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `lr` | Learning rate | 0.0005-0.003 |
| `weight_decay` | L2 regularization | 0.001-0.01 |
| `batch_size` | Training batch size | 128-512 |
| `n_epochs` | Training epochs | 10-50 |

## 🔧 Advanced Features

### Hyperparameter Search

The project includes sophisticated hyperparameter search tools with budget constraints:

```python
from hp_search.hyperparameter_search import HyperparameterSearcher

searcher = HyperparameterSearcher(
    dataset="fashion-mnist",
    max_params=8000,  # Parameter budget
    n_epochs=15
)

param_grid = {
    'n_neurons': [8, 10, 12],
    'n_dendrites': [4, 6, 8],
    'n_dendrite_inputs': [64, 96, 128],
    'synaptic_resampling': [True, False]
}

searcher.grid_search(param_grid, max_trials=20)
searcher.print_results(5)
```

See [`hp_search/HYPERPARAMETER_SEARCH_README.md`](hp_search/HYPERPARAMETER_SEARCH_README.md) for detailed documentation.

### Visualization Tools

```python
from utils import plot_dendritic_weights_full_model, calculate_eigenvalues

# Visualize learned dendritic weights
plot_dendritic_weights_full_model(model)

# Analyze weight matrix eigenvalues
eigenvalues = calculate_eigenvalues(model.layers[0])
```

### Model Comparison

```python
from training import train_models

# Compare dendritic vs vanilla models
results = train_models(
    models=[dendritic_model, vanilla_model],
    optimizers=[optimizer1, optimizer2],
    criterion=criterion,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    model_names=["Dendritic", "Vanilla"]
)
```

## 🐛 Implementation Notes

### GPU Acceleration

The code automatically detects and uses CuPy for GPU acceleration:

```python
try:
    import cupy as cp
    print("Using CuPy (GPU acceleration)")
except:
    import numpy as cp
    print("Using NumPy (CPU)")
```

### Performance Optimization

- **CSR Matrix Representation**: ~50% speedup for sparse operations
- **Batch Processing**: Efficient mini-batch training
- **Vectorized Operations**: NumPy/CuPy for fast computation

Note: Dendritic models are currently slower to train than vanilla models due to sparse connectivity overhead, but this is offset by requiring far fewer parameters.

## 📚 References

This work is based on and inspired by:

1. **London, M., & Häusser, M. (2005)**  
   *Dendritic computation*  
   Annual Review of Neuroscience, 28, 503-532.  
   [DOI: 10.1146/annurev.neuro.28.061604.135703](https://doi.org/10.1146/annurev.neuro.28.061604.135703)

2. **Chavlis, S. and Poirazi, P. (2025)**  
   *Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning*  
   Nature Communications, 16(1), p.943.  
   [DOI: 10.1038/s41467-024-55957-7](https://doi.org/10.1038/s41467-024-55957-7)
]

**Keywords**: dendritic computation, structural plasticity, synaptic resampling, parameter-efficient learning, sparse neural networks, biologically-inspired AI

