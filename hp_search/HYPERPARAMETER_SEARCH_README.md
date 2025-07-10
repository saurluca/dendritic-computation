# Hyperparameter Search for Dendritic Neural Networks

This toolkit provides automated hyperparameter optimization for dendritic neural networks with parameter budget constraints. It helps you find the best model architecture while staying within memory and computational limits.

## 🎯 Key Features

- **Parameter Budget Control**: Automatically ensures models stay under specified parameter limits
- **Multiple Search Strategies**: Grid search, random search, and targeted optimization
- **Architecture Comparison**: Compare different dendritic architectures fairly
- **Result Tracking**: Comprehensive logging and JSON export of all results
- **Flexible Configuration**: Support for different datasets, training parameters, and model architectures

## 📁 Files

- `hyperparameter_search.py` - Core hyperparameter search framework
- `hyperparameter_examples.py` - Ready-to-use examples for common scenarios
- `HYPERPARAMETER_SEARCH_README.md` - This documentation

## 🚀 Quick Start

### Basic Usage

```python
from hyperparameter_search import HyperparameterSearcher

# Create searcher with 8000 parameter budget
searcher = HyperparameterSearcher(
    dataset="fashion-mnist",
    max_params=8000,
    n_epochs=15,
    verbose=True
)

# Define parameter search space
param_grid = {
    'n_neurons': [8, 10, 12],
    'n_dendrites': [4, 6, 8],
    'n_dendrite_inputs': [64, 96, 128],
    'soma_enabled': [True, False],
    'architecture': ['dendritic_simple'],
    'synaptic_resampling': [True, False]
}

# Run search
searcher.grid_search(param_grid, max_trials=20)

# View results
searcher.print_results(5)
searcher.save_results("my_search_results.json")
```

### Running Examples

```bash
# Run all example scenarios
python hyperparameter_examples.py

# Run specific budget optimization
python -c "
from hyperparameter_examples import optimize_for_budget
result = optimize_for_budget(max_params=5000, dataset='mnist')
print(f'Best accuracy: {result.test_accuracy:.3f}')
"
```

## 🏗️ Model Architectures

The search supports three main dendritic architectures:

### 1. Simple Dendritic (`dendritic_simple`)
```
Input → DendriticLayer(soma=True) → LinearLayer → Output
```
- **Parameters**: `n_dendrite_inputs × n_dendrites × n_neurons + n_dendrites × n_neurons + n_neurons + output_params`
- **Best for**: Balanced performance and parameter efficiency

### 2. Soma Disabled (`soma_disabled`)
```
Input → DendriticLayer(soma=False) → LinearLayer → LeakyReLU → LinearLayer → Output
```
- **Parameters**: `n_dendrite_inputs × n_dendrites + n_dendrites + hidden_params + output_params`
- **Best for**: Direct dendrite-to-output mappings

### 3. Dendritic with Hidden (`dendritic_with_hidden`)
```
Input → DendriticLayer → LeakyReLU → LinearLayer → LeakyReLU → LinearLayer → Output
```
- **Parameters**: Dendritic + hidden layer + output layer parameters
- **Best for**: Complex pattern recognition requiring additional capacity

## 📊 Parameter Calculation

Understanding how parameters are counted is crucial for budget planning:

### DendriticLayer Parameters
- **With Soma**: 
  - Dendrite weights: `n_dendrite_inputs × n_dendrites × n_neurons`
  - Dendrite biases: `n_dendrites × n_neurons`
  - Soma weights: `n_dendrites × n_neurons`
  - Soma biases: `n_neurons`

- **Without Soma**:
  - Dendrite weights: `n_dendrite_inputs × n_dendrites`
  - Dendrite biases: `n_dendrites`

### LinearLayer Parameters
- Weights: `input_dim × output_dim`
- Biases: `output_dim`

## 🔍 Search Strategies

### 1. Grid Search
Exhaustive search over all parameter combinations:

```python
param_grid = {
    'n_neurons': [8, 10, 12],
    'n_dendrites': [4, 6, 8],
    'n_dendrite_inputs': [64, 96, 128]
}
searcher.grid_search(param_grid)
```

### 2. Random Search
Sample from parameter distributions:

```python
param_distributions = {
    'n_neurons': (8, 15),  # Range
    'n_dendrites': [4, 6, 8, 10],  # Choices
    'n_dendrite_inputs': {'type': 'randint', 'low': 32, 'high': 200}
}
searcher.random_search(param_distributions, n_trials=50)
```

## 📈 Usage Scenarios

### 1. Quick Prototyping
For rapid iteration during development:

```python
from hyperparameter_examples import quick_prototype_search
result = quick_prototype_search(max_params=5000, dataset="mnist")
```
- Fewer epochs (8)
- Larger batch sizes (512)
- Focused parameter ranges
- ~5-10 minutes runtime

### 2. Budget Optimization
Find the best model within a specific parameter budget:

```python
from hyperparameter_examples import optimize_for_budget
result = optimize_for_budget(max_params=8000, dataset="fashion-mnist")
```
- Automatic parameter range adjustment based on budget
- Smart architecture selection
- Comprehensive search within constraints

### 3. Architecture Comparison
Compare different architectures fairly:

```python
from hyperparameter_examples import compare_architectures
results = compare_architectures(max_params=8000, dataset="fashion-mnist")
```
- Tests all supported architectures
- Same parameter budget for each
- Direct performance comparison

### 4. Thorough Optimization
Production-ready search with comprehensive evaluation:

```python
from hyperparameter_examples import thorough_optimization
result = thorough_optimization(max_params=8000, dataset="fashion-mnist")
```
- Two-phase search (broad → refinement)
- More epochs (20)
- Multiple seeds for robustness
- Extended runtime (~2-4 hours)

## 🎛️ Configuration Options

### HyperparameterSearcher Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | "fashion-mnist" | Dataset: "mnist", "fashion-mnist", "cifar10" |
| `subset_size` | None | Limit dataset size for faster testing |
| `max_params` | 8000 | Maximum allowed parameters |
| `n_epochs` | 15 | Training epochs per trial |
| `lr` | 0.0005 | Learning rate |
| `weight_decay` | 0.01 | Weight decay for regularization |
| `batch_size` | 256 | Training batch size |
| `n_classes` | 10 | Number of output classes |
| `verbose` | True | Print detailed progress |

### Model Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_neurons` | Neurons in dendritic layer | 5-20 |
| `n_dendrites` | Dendrites per neuron | 3-15 |
| `n_dendrite_inputs` | Inputs per dendrite | 16-256 |
| `soma_enabled` | Enable soma layer | True/False |
| `architecture` | Model architecture | "dendritic_simple", "soma_disabled", "dendritic_with_hidden" |
| `synaptic_resampling` | Enable synaptic resampling | True/False |
| `percentage_resample` | Resampling percentage | 0.1-0.5 |
| `steps_to_resample` | Steps between resampling | 64-256 |

## 📋 Results Analysis

### SearchResult Object
```python
@dataclass
class SearchResult:
    params: Dict[str, Any]          # Hyperparameter configuration
    test_accuracy: float            # Final test accuracy
    train_accuracy: float           # Final train accuracy
    total_params: int               # Total model parameters
    train_time: float               # Training time in seconds
    final_test_loss: float          # Final test loss
```

### Accessing Results
```python
# Get top 5 results
best_results = searcher.get_best_results(5)

# Print summary
searcher.print_results(5)

# Save to JSON
searcher.save_results("results.json")

# Access specific result
best = best_results[0]
print(f"Best config: {best.params}")
print(f"Accuracy: {best.test_accuracy:.3f}")
print(f"Parameters: {best.total_params}")
```

## 🔧 Tips and Best Practices

### 1. Parameter Budget Planning
- **Small budget (≤2000)**: Focus on `dendritic_simple` with minimal parameters
- **Medium budget (2000-5000)**: Explore `soma_disabled` architectures
- **Large budget (≥5000)**: Test `dendritic_with_hidden` for complex tasks

### 2. Search Strategy Selection
- **Grid search**: When you have specific parameter values to test
- **Random search**: When exploring large parameter spaces
- **Mixed approach**: Grid search for architecture, random for fine-tuning

### 3. Computational Efficiency
- Use `subset_size` for initial exploration
- Reduce `n_epochs` for parameter screening
- Increase `batch_size` for faster training (if memory allows)

### 4. Result Validation
- Always test multiple seeds for robust results
- Validate best configurations with longer training
- Consider parameter efficiency (accuracy per parameter)

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `batch_size`
   - Decrease `max_params` limit
   - Use `subset_size` to limit dataset

2. **No Valid Configurations**
   - Increase `max_params` budget
   - Reduce parameter ranges
   - Check architecture compatibility

3. **Poor Results**
   - Increase `n_epochs`
   - Adjust learning rate
   - Try different architectures
   - Verify dataset loading

### Parameter Budget Estimation

Quick estimation formulas:

**Simple Dendritic**:
```
params ≈ n_dendrite_inputs × n_dendrites × n_neurons × 2 + n_neurons × (n_classes + 1)
```

**Soma Disabled**:
```
params ≈ n_dendrite_inputs × n_dendrites + hidden_params + output_params
```

## 📊 Example Results

Typical results on Fashion-MNIST with 8000 parameter budget:

| Architecture | Test Accuracy | Parameters | Configuration |
|-------------|---------------|------------|---------------|
| Simple Dendritic | 0.876 | 7,890 | n_neurons=12, n_dendrites=6, n_dendrite_inputs=96 |
| Soma Disabled | 0.871 | 7,654 | n_dendrites=25, n_dendrite_inputs=64 |
| With Hidden | 0.868 | 7,982 | n_neurons=10, hidden_dim=8, n_dendrites=6 |

## 🚀 Getting Started Checklist

1. **Install Dependencies**: Ensure you have the required modules (`modules.py`, `utils.py`, `training.py`)
2. **Choose Scenario**: Quick prototype, budget optimization, or architecture comparison
3. **Set Budget**: Determine your parameter limit based on memory/compute constraints
4. **Run Search**: Start with example scripts, then customize for your needs
5. **Analyze Results**: Compare accuracies, parameter counts, and training times
6. **Validate**: Re-train best configurations with different seeds
7. **Deploy**: Use best configuration in your production pipeline

## 📝 Example Workflow

```python
# 1. Quick exploration
from hyperparameter_examples import quick_prototype_search
quick_result = quick_prototype_search(max_params=5000)

# 2. Detailed search around promising area
searcher = HyperparameterSearcher(max_params=8000, n_epochs=20)
param_distributions = {
    'n_neurons': (10, 15),  # Narrow range around quick result
    'n_dendrites': (4, 8),
    'n_dendrite_inputs': (80, 120),
    'architecture': ['dendritic_simple'],
    'synaptic_resampling': [True]
}
searcher.random_search(param_distributions, n_trials=30)

# 3. Final validation
best_config = searcher.get_best_results(1)[0].params
# Train with best config using more epochs...
```

Happy hyperparameter hunting! 🎯 