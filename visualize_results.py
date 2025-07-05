# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(filename='grid_search_results.json'):
    """Load grid search results from JSON file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def calculate_model_params(n_neurons, n_dendrite_inputs, n_dendrites, input_dim=784):
    """Calculate approximate number of parameters in the model."""
    # DendriticLayer parameters
    # Each dendrite has n_dendrite_inputs connections from input
    # Each neuron has n_dendrites dendrites
    dendritic_params = n_neurons * n_dendrites * n_dendrite_inputs
    
    # Linear layer parameters (output layer)
    linear_params = n_neurons * 10 + 10  # weights + biases
    
    return dendritic_params + linear_params

def create_results_dir():
    """Create results directory if it doesn't exist."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    return results_dir

def plot_steps_to_resample_performance(results, results_dir):
    """Plot performance changes for different steps_to_resample values."""
    # Group results by steps_to_resample
    steps_performance = defaultdict(list)
    
    for result in results:
        steps = result['params']['steps_to_resample']
        steps_performance[steps].append(result['final_test_accuracy'])
    
    # Calculate statistics
    steps_list = sorted(steps_performance.keys())
    means = [np.mean(steps_performance[s]) for s in steps_list]
    stds = [np.std(steps_performance[s]) for s in steps_list]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(steps_list, means, yerr=stds, marker='o', capsize=5, capthick=2)
    plt.xlabel('Steps to Resample')
    plt.ylabel('Test Accuracy')
    plt.title('Performance vs. Steps to Resample')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'steps_to_resample_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_vs_params(results, results_dir):
    """Plot performance vs model complexity for different hyperparameters."""
    # Calculate model parameters and performance for each result
    data = []
    for result in results:
        params = result['params']
        n_params = calculate_model_params(
            params['n_neurons'], 
            params['n_dendrite_inputs'], 
            params['n_dendrites']
        )
        
        data.append({
            'n_params': n_params,
            'accuracy': result['final_test_accuracy'],
            'n_neurons': params['n_neurons'],
            'n_dendrite_inputs': params['n_dendrite_inputs'],
            'n_dendrites': params['n_dendrites']
        })
    
    df = pd.DataFrame(data)
    
    # Create subplots for different hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance vs total parameters
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['n_params'], df['accuracy'], 
                         c=df['n_neurons'], cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance vs Model Size (colored by n_neurons)')
    plt.colorbar(scatter, ax=ax1, label='n_neurons')
    
    # Performance vs n_neurons
    ax2 = axes[0, 1]
    neurons_perf = df.groupby('n_neurons')['accuracy'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(neurons_perf['n_neurons'], neurons_perf['mean'], 
                yerr=neurons_perf['std'], marker='o', capsize=5)
    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Performance vs Number of Neurons')
    ax2.grid(True, alpha=0.3)
    
    # Performance vs n_dendrite_inputs
    ax3 = axes[1, 0]
    inputs_perf = df.groupby('n_dendrite_inputs')['accuracy'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(inputs_perf['n_dendrite_inputs'], inputs_perf['mean'], 
                yerr=inputs_perf['std'], marker='s', capsize=5)
    ax3.set_xlabel('Number of Dendrite Inputs')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Performance vs Dendrite Inputs')
    ax3.grid(True, alpha=0.3)
    
    # Performance vs n_dendrites
    ax4 = axes[1, 1]
    dendrites_perf = df.groupby('n_dendrites')['accuracy'].agg(['mean', 'std']).reset_index()
    ax4.errorbar(dendrites_perf['n_dendrites'], dendrites_perf['mean'], 
                yerr=dendrites_perf['std'], marker='^', capsize=5)
    ax4.set_xlabel('Number of Dendrites')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Performance vs Number of Dendrites')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_vs_params.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_resampling_analysis(results, results_dir):
    """Plot resampling frequency vs percentage analysis."""
    # Create a 2D heatmap of performance vs resampling parameters
    data = []
    for result in results:
        params = result['params']
        data.append({
            'percentage_resample': params['percentage_resample'],
            'steps_to_resample': params['steps_to_resample'],
            'accuracy': result['final_test_accuracy']
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table for heatmap
    pivot_table = df.pivot_table(
        values='accuracy', 
        index='percentage_resample', 
        columns='steps_to_resample', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis', 
                cbar_kws={'label': 'Test Accuracy'})
    plt.title('Resampling Strategy Performance Heatmap')
    plt.xlabel('Steps to Resample')
    plt.ylabel('Percentage to Resample')
    plt.tight_layout()
    plt.savefig(results_dir / 'resampling_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Percentage vs accuracy
    perc_perf = df.groupby('percentage_resample')['accuracy'].agg(['mean', 'std']).reset_index()
    ax1.errorbar(perc_perf['percentage_resample'], perc_perf['mean'], 
                yerr=perc_perf['std'], marker='o', capsize=5)
    ax1.set_xlabel('Percentage Resample')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance vs Resampling Percentage')
    ax1.grid(True, alpha=0.3)
    
    # Steps vs accuracy
    steps_perf = df.groupby('steps_to_resample')['accuracy'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(steps_perf['steps_to_resample'], steps_perf['mean'], 
                yerr=steps_perf['std'], marker='s', capsize=5)
    ax2.set_xlabel('Steps to Resample')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Performance vs Resampling Steps')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'resampling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_time_vs_params(results, results_dir):
    """Plot training time vs number of parameters."""
    data = []
    for result in results:
        params = result['params']
        n_params = calculate_model_params(
            params['n_neurons'], 
            params['n_dendrite_inputs'], 
            params['n_dendrites']
        )
        
        data.append({
            'n_params': n_params,
            'training_time': result['training_time'],
            'n_neurons': params['n_neurons'],
            'accuracy': result['final_test_accuracy']
        })
    
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training time vs parameters
    scatter1 = ax1.scatter(df['n_params'], df['training_time'], 
                          c=df['n_neurons'], cmap='plasma', alpha=0.7, s=50)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time vs Model Size')
    plt.colorbar(scatter1, ax=ax1, label='n_neurons')
    
    # Training time vs accuracy (efficiency plot)
    scatter2 = ax2.scatter(df['training_time'], df['accuracy'], 
                          c=df['n_params'], cmap='coolwarm', alpha=0.7, s=50)
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Training Efficiency: Accuracy vs Time')
    plt.colorbar(scatter2, ax=ax2, label='n_params')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_best_configurations(results, results_dir):
    """Plot analysis of best performing configurations."""
    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x['final_test_accuracy'], reverse=True)
    top_10 = sorted_results[:10]
    
    # Extract parameters for top configurations
    config_data = []
    for i, result in enumerate(top_10):
        params = result['params']
        config_data.append({
            'rank': i + 1,
            'accuracy': result['final_test_accuracy'],
            'n_neurons': params['n_neurons'],
            'n_dendrite_inputs': params['n_dendrite_inputs'],
            'n_dendrites': params['n_dendrites'],
            'percentage_resample': params['percentage_resample'],
            'steps_to_resample': params['steps_to_resample'],
            'training_time': result['training_time']
        })
    
    df = pd.DataFrame(config_data)
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top 10 accuracies
    ax1 = axes[0, 0]
    bars = ax1.bar(df['rank'], df['accuracy'], color='skyblue', alpha=0.7)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Top 10 Configurations by Accuracy')
    ax1.set_xticks(df['rank'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, df['accuracy']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Parameter distribution in top configurations
    param_cols = ['n_neurons', 'n_dendrite_inputs', 'n_dendrites', 
                  'percentage_resample', 'steps_to_resample']
    
    for i, param in enumerate(param_cols):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        values = df[param].values
        ax.scatter(range(1, 11), values, alpha=0.7, s=60)
        ax.set_xlabel('Rank')
        ax.set_ylabel(param.replace('_', ' ').title())
        ax.set_title(f'{param.replace("_", " ").title()} in Top 10')
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'best_configurations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("TOP 10 CONFIGURATIONS SUMMARY")
    print("="*50)
    for i, config in enumerate(config_data):
        print(f"\nRank {i+1}: Accuracy = {config['accuracy']:.4f}")
        print(f"  n_neurons: {config['n_neurons']}")
        print(f"  n_dendrite_inputs: {config['n_dendrite_inputs']}")
        print(f"  n_dendrites: {config['n_dendrites']}")
        print(f"  percentage_resample: {config['percentage_resample']}")
        print(f"  steps_to_resample: {config['steps_to_resample']}")
        print(f"  training_time: {config['training_time']:.2f}s")

def main():
    """Main function to run all visualizations."""
    print("Loading grid search results...")
    results = load_results()
    print(f"Loaded {len(results)} results")
    
    # Create results directory
    results_dir = create_results_dir()
    print(f"Created results directory: {results_dir}")
    
    # Generate all plots
    print("\n1. Plotting steps_to_resample performance...")
    plot_steps_to_resample_performance(results, results_dir)
    
    print("\n2. Plotting performance vs parameters...")
    plot_performance_vs_params(results, results_dir)
    
    print("\n3. Plotting resampling analysis...")
    plot_resampling_analysis(results, results_dir)
    
    print("\n4. Plotting training time analysis...")
    plot_training_time_vs_params(results, results_dir)
    
    print("\n5. Plotting best configurations...")
    plot_best_configurations(results, results_dir)
    
    print(f"\nAll plots saved to: {results_dir}")
    print("Visualization complete!")

if __name__ == "__main__":
    main() 