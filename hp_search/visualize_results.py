# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_results(filename="grid_search_results_small.json"):
    """Load grid search results from JSON file."""
    with open(filename, "r") as f:
        results = json.load(f)
    return results


def calculate_model_params(n_dendrite_inputs, n_dendrites, n_neurons=10, input_dim=784):
    """Calculate approximate number of parameters in the model."""
    # DendriticLayer parameters
    # Each dendrite has n_dendrite_inputs connections from input
    # Each neuron has n_dendrites dendrites
    # n_neurons is always 10 in this case
    dendritic_params = n_neurons * n_dendrites * n_dendrite_inputs
    return dendritic_params


def create_results_dir():
    """Create results directory if it doesn't exist."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def plot_steps_to_resample_performance(grouped_results, results_dir):
    """Plot performance changes for different steps_to_resample values."""
    # Group results by steps_to_resample
    steps_performance = defaultdict(list)
    steps_std = defaultdict(list)

    for result in grouped_results:
        steps = result["params"]["steps_to_resample"]
        steps_performance[steps].append(result["accuracy_mean"])
        steps_std[steps].append(result["accuracy_std"])

    # Calculate statistics across different configurations with same steps_to_resample
    steps_list = sorted(steps_performance.keys())
    means = [np.mean(steps_performance[s]) for s in steps_list]
    # Combine standard deviations properly (this is the std of the means plus the mean of the stds)
    stds = [
        np.sqrt(
            np.mean([std**2 for std in steps_std[s]]) + np.var(steps_performance[s])
        )
        for s in steps_list
    ]

    plt.figure(figsize=(10, 6))
    plt.errorbar(steps_list, means, yerr=stds, marker="o", capsize=5, capthick=2)
    plt.xlabel("Steps to Resample")
    plt.ylabel("Test Accuracy (Mean ± Std)")
    plt.title(
        "Performance vs. Steps to Resample\n(Error bars show variance across seeds and configurations)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        results_dir / "steps_to_resample_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_performance_vs_params(grouped_results, results_dir):
    """Plot performance vs model complexity for different hyperparameters."""
    # Calculate model parameters and performance for each result
    data = []
    for result in grouped_results:
        params = result["params"]
        n_params = calculate_model_params(
            params["n_dendrite_inputs"], params["n_dendrites"]
        )

        data.append(
            {
                "n_params": n_params,
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_dendrite_inputs": params["n_dendrite_inputs"],
                "n_dendrites": params["n_dendrites"],
                "n_runs": result["n_runs"],
            }
        )

    df = pd.DataFrame(data)

    # Create subplots for different hyperparameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Performance vs total parameters
    ax1 = axes[0]
    scatter = ax1.scatter(
        df["n_params"],
        df["accuracy"],
        c=df["n_dendrites"],
        cmap="viridis",
        alpha=0.7,
        s=50,
    )
    # Add error bars
    ax1.errorbar(
        df["n_params"],
        df["accuracy"],
        yerr=df["accuracy_std"],
        fmt="none",
        alpha=0.3,
        capsize=2,
    )
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Test Accuracy (Mean ± Std)")
    ax1.set_title("Performance vs Model Size (colored by n_dendrites)")
    plt.colorbar(scatter, ax=ax1, label="n_dendrites")

    # Performance vs n_dendrite_inputs
    ax2 = axes[1]
    inputs_perf = (
        df.groupby("n_dendrite_inputs")
        .agg(
            {
                "accuracy": "mean",
                "accuracy_std": lambda x: np.sqrt(np.mean(x**2)),  # RMS of std errors
            }
        )
        .reset_index()
    )
    ax2.errorbar(
        inputs_perf["n_dendrite_inputs"],
        inputs_perf["accuracy"],
        yerr=inputs_perf["accuracy_std"],
        marker="s",
        capsize=5,
    )
    ax2.set_xlabel("Number of Dendrite Inputs")
    ax2.set_ylabel("Test Accuracy (Mean ± Std)")
    ax2.set_title("Performance vs Dendrite Inputs")
    ax2.grid(True, alpha=0.3)

    # Performance vs n_dendrites
    ax3 = axes[2]
    dendrites_perf = (
        df.groupby("n_dendrites")
        .agg(
            {
                "accuracy": "mean",
                "accuracy_std": lambda x: np.sqrt(np.mean(x**2)),  # RMS of std errors
            }
        )
        .reset_index()
    )
    ax3.errorbar(
        dendrites_perf["n_dendrites"],
        dendrites_perf["accuracy"],
        yerr=dendrites_perf["accuracy_std"],
        marker="^",
        capsize=5,
    )
    ax3.set_xlabel("Number of Dendrites")
    ax3.set_ylabel("Test Accuracy (Mean ± Std)")
    ax3.set_title("Performance vs Number of Dendrites")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "performance_vs_params.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_resampling_analysis(grouped_results, results_dir):
    """Plot resampling frequency vs percentage analysis."""
    # Create a 2D heatmap of performance vs resampling parameters
    data = []
    for result in grouped_results:
        params = result["params"]
        data.append(
            {
                "percentage_resample": params["percentage_resample"],
                "steps_to_resample": params["steps_to_resample"],
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_runs": result["n_runs"],
            }
        )

    df = pd.DataFrame(data)

    # Create pivot table for heatmap (using mean accuracy)
    pivot_table = df.pivot_table(
        values="accuracy",
        index="percentage_resample",
        columns="steps_to_resample",
        aggfunc="mean",
    )

    # Create pivot table for standard deviations
    pivot_std_table = df.pivot_table(
        values="accuracy_std",
        index="percentage_resample",
        columns="steps_to_resample",
        aggfunc=lambda x: np.sqrt(np.mean(x**2)),  # RMS of std errors
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap of mean accuracy
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={"label": "Test Accuracy (Mean)"},
        ax=ax1,
    )
    ax1.set_title("Resampling Strategy Performance Heatmap (Mean Accuracy)")
    ax1.set_xlabel("Steps to Resample")
    ax1.set_ylabel("Percentage to Resample")

    # Heatmap of standard deviations
    sns.heatmap(
        pivot_std_table,
        annot=True,
        fmt=".4f",
        cmap="Reds",
        cbar_kws={"label": "Test Accuracy (Std)"},
        ax=ax2,
    )
    ax2.set_title("Resampling Strategy Uncertainty Heatmap (Std Across Seeds)")
    ax2.set_xlabel("Steps to Resample")
    ax2.set_ylabel("Percentage to Resample")

    plt.tight_layout()
    plt.savefig(results_dir / "resampling_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Additional scatter plots with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Percentage vs accuracy
    perc_perf = (
        df.groupby("percentage_resample")
        .agg({"accuracy": "mean", "accuracy_std": lambda x: np.sqrt(np.mean(x**2))})
        .reset_index()
    )
    ax1.errorbar(
        perc_perf["percentage_resample"],
        perc_perf["accuracy"],
        yerr=perc_perf["accuracy_std"],
        marker="o",
        capsize=5,
    )
    ax1.set_xlabel("Percentage Resample")
    ax1.set_ylabel("Test Accuracy (Mean ± Std)")
    ax1.set_title("Performance vs Resampling Percentage")
    ax1.grid(True, alpha=0.3)

    # Steps vs accuracy
    steps_perf = (
        df.groupby("steps_to_resample")
        .agg({"accuracy": "mean", "accuracy_std": lambda x: np.sqrt(np.mean(x**2))})
        .reset_index()
    )
    ax2.errorbar(
        steps_perf["steps_to_resample"],
        steps_perf["accuracy"],
        yerr=steps_perf["accuracy_std"],
        marker="s",
        capsize=5,
    )
    ax2.set_xlabel("Steps to Resample")
    ax2.set_ylabel("Test Accuracy (Mean ± Std)")
    ax2.set_title("Performance vs Resampling Steps")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "resampling_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_time_vs_params(grouped_results, results_dir):
    """Plot training time vs number of parameters."""
    data = []
    for result in grouped_results:
        params = result["params"]
        n_params = calculate_model_params(
            params["n_dendrite_inputs"], params["n_dendrites"]
        )

        data.append(
            {
                "n_params": n_params,
                "training_time": result["training_time_mean"],
                "training_time_std": result["training_time_std"],
                "n_dendrites": params["n_dendrites"],
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_runs": result["n_runs"],
            }
        )

    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training time vs parameters
    scatter1 = ax1.scatter(
        df["n_params"],
        df["training_time"],
        c=df["n_dendrites"],
        cmap="plasma",
        alpha=0.7,
        s=50,
    )
    # Add error bars for training time
    ax1.errorbar(
        df["n_params"],
        df["training_time"],
        yerr=df["training_time_std"],
        fmt="none",
        alpha=0.3,
        capsize=2,
    )
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Training Time (Mean ± Std, seconds)")
    ax1.set_title("Training Time vs Model Size")
    plt.colorbar(scatter1, ax=ax1, label="n_dendrites")

    # Training time vs accuracy (efficiency plot)
    scatter2 = ax2.scatter(
        df["training_time"],
        df["accuracy"],
        c=df["n_params"],
        cmap="coolwarm",
        alpha=0.7,
        s=50,
    )
    # Add error bars for both accuracy and training time
    ax2.errorbar(
        df["training_time"],
        df["accuracy"],
        xerr=df["training_time_std"],
        yerr=df["accuracy_std"],
        fmt="none",
        alpha=0.3,
        capsize=2,
    )
    ax2.set_xlabel("Training Time (Mean ± Std, seconds)")
    ax2.set_ylabel("Test Accuracy (Mean ± Std)")
    ax2.set_title("Training Efficiency: Accuracy vs Time")
    plt.colorbar(scatter2, ax=ax2, label="n_params")

    plt.tight_layout()
    plt.savefig(
        results_dir / "training_time_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_best_configurations(grouped_results, results_dir):
    """Plot analysis of best performing configurations."""
    # Sort results by mean accuracy
    sorted_results = sorted(
        grouped_results, key=lambda x: x["accuracy_mean"], reverse=True
    )
    top_10 = sorted_results[:10]

    # Extract parameters for top configurations
    config_data = []
    for i, result in enumerate(top_10):
        params = result["params"]
        n_params = calculate_model_params(
            params["n_dendrite_inputs"], params["n_dendrites"]
        )
        config_data.append(
            {
                "rank": i + 1,
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_params": n_params,
                "n_dendrite_inputs": params["n_dendrite_inputs"],
                "n_dendrites": params["n_dendrites"],
                "percentage_resample": params["percentage_resample"],
                "steps_to_resample": params["steps_to_resample"],
                "training_time": result["training_time_mean"],
                "training_time_std": result["training_time_std"],
                "n_runs": result["n_runs"],
                "seeds": result["seeds"],
            }
        )

    df = pd.DataFrame(config_data)

    # Create a comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top 10 accuracies with error bars
    ax1 = axes[0, 0]
    bars = ax1.bar(
        df["rank"],
        df["accuracy"],
        color="skyblue",
        alpha=0.7,
        yerr=df["accuracy_std"],
        capsize=5,
    )
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("Test Accuracy (Mean ± Std)")
    ax1.set_title("Top 10 Configurations by Accuracy")
    ax1.set_xticks(df["rank"])

    # Add value labels on bars
    for bar, acc, std in zip(bars, df["accuracy"], df["accuracy_std"]):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.001,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Parameter distribution in top configurations
    param_cols = [
        "n_dendrite_inputs",
        "n_dendrites",
        "percentage_resample",
        "steps_to_resample",
    ]

    for i, param in enumerate(param_cols):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        values = df[param].values
        ax.scatter(range(1, 11), values, alpha=0.7, s=60)
        ax.set_xlabel("Rank")
        ax.set_ylabel(param.replace("_", " ").title())
        ax.set_title(f"{param.replace('_', ' ').title()} in Top 10")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)

    # Hide the last subplot since we only have 4 parameters now
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(results_dir / "best_configurations.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS SUMMARY (AVERAGED ACROSS SEEDS)")
    print("=" * 70)
    for i, config in enumerate(config_data):
        print(
            f"\nRank {i + 1}: Accuracy = {config['accuracy']:.4f} ± {config['accuracy_std']:.4f}"
        )
        print(f"  Parameters: {config['n_params']:,}")
        print(f"  n_dendrite_inputs: {config['n_dendrite_inputs']}")
        print(f"  n_dendrites: {config['n_dendrites']}")
        print(f"  percentage_resample: {config['percentage_resample']}")
        print(f"  steps_to_resample: {config['steps_to_resample']}")
        print(
            f"  training_time: {config['training_time']:.2f} ± {config['training_time_std']:.2f}s"
        )
        print(f"  n_runs: {config['n_runs']} (seeds: {config['seeds']})")


def plot_parameter_efficiency(grouped_results, results_dir):
    """Plot parameter efficiency: accuracy vs number of parameters."""
    data = []
    for result in grouped_results:
        params = result["params"]
        n_params = calculate_model_params(
            params["n_dendrite_inputs"], params["n_dendrites"]
        )

        data.append(
            {
                "n_params": n_params,
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_dendrite_inputs": params["n_dendrite_inputs"],
                "n_dendrites": params["n_dendrites"],
                "percentage_resample": params["percentage_resample"],
                "steps_to_resample": params["steps_to_resample"],
                "n_runs": result["n_runs"],
            }
        )

    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Parameter efficiency scatter plot
    scatter1 = ax1.scatter(
        df["n_params"],
        df["accuracy"],
        c=df["n_dendrites"],
        cmap="viridis",
        alpha=0.7,
        s=50,
    )
    # Add error bars
    ax1.errorbar(
        df["n_params"],
        df["accuracy"],
        yerr=df["accuracy_std"],
        fmt="none",
        alpha=0.3,
        capsize=2,
    )
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Test Accuracy (Mean ± Std)")
    ax1.set_title("Parameter Efficiency: Accuracy vs Model Size")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label="n_dendrites")

    # Add efficiency frontier (Pareto front) - use mean accuracy for frontier
    df_sorted = df.sort_values("n_params")
    pareto_points = []
    max_acc_so_far = 0

    for _, row in df_sorted.iterrows():
        if row["accuracy"] > max_acc_so_far:
            pareto_points.append((row["n_params"], row["accuracy"]))
            max_acc_so_far = row["accuracy"]

    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        ax1.plot(
            pareto_x,
            pareto_y,
            "r-",
            linewidth=2,
            alpha=0.7,
            label="Efficiency Frontier",
        )
        ax1.legend()

    # Parameter efficiency ratio (accuracy per parameter)
    df["efficiency"] = df["accuracy"] / df["n_params"]
    df["efficiency_std"] = (
        df["accuracy_std"] / df["n_params"]
    )  # Approximate std for efficiency
    scatter2 = ax2.scatter(
        df["n_params"],
        df["efficiency"],
        c=df["accuracy"],
        cmap="plasma",
        alpha=0.7,
        s=50,
    )
    # Add error bars for efficiency
    ax2.errorbar(
        df["n_params"],
        df["efficiency"],
        yerr=df["efficiency_std"],
        fmt="none",
        alpha=0.3,
        capsize=2,
    )
    ax2.set_xlabel("Number of Parameters")
    ax2.set_ylabel("Accuracy per Parameter (Mean ± Std)")
    ax2.set_title("Parameter Efficiency Ratio")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label="Test Accuracy (Mean)")

    plt.tight_layout()
    plt.savefig(results_dir / "parameter_efficiency.png", dpi=300, bbox_inches="tight")
    plt.show()

    return df


def analyze_accuracy_thresholds(grouped_results):
    """Find smallest models that reach specific accuracy thresholds."""
    data = []
    for result in grouped_results:
        params = result["params"]
        n_params = calculate_model_params(
            params["n_dendrite_inputs"], params["n_dendrites"]
        )

        data.append(
            {
                "n_params": n_params,
                "accuracy": result["accuracy_mean"],
                "accuracy_std": result["accuracy_std"],
                "n_dendrite_inputs": params["n_dendrite_inputs"],
                "n_dendrites": params["n_dendrites"],
                "percentage_resample": params["percentage_resample"],
                "steps_to_resample": params["steps_to_resample"],
                "training_time": result["training_time_mean"],
                "training_time_std": result["training_time_std"],
                "n_runs": result["n_runs"],
                "seeds": result["seeds"],
            }
        )

    df = pd.DataFrame(data)

    thresholds = [0.80, 0.90, 0.95]

    print("\n" + "=" * 80)
    print("SMALLEST MODELS REACHING ACCURACY THRESHOLDS (AVERAGED ACROSS SEEDS)")
    print("=" * 80)

    for threshold in thresholds:
        # Filter models that reach the threshold (using mean - std to be conservative)
        conservative_accuracy = df["accuracy"] - df["accuracy_std"]
        threshold_models = df[conservative_accuracy >= threshold]

        if len(threshold_models) > 0:
            # Find the model with minimum parameters
            smallest_model = threshold_models.loc[threshold_models["n_params"].idxmin()]

            print(
                f"\nSmallest model reaching {threshold * 100:.0f}% accuracy (conservative estimate):"
            )
            print(
                f"  Accuracy: {smallest_model['accuracy']:.4f} ± {smallest_model['accuracy_std']:.4f}"
            )
            print(
                f"  Conservative estimate: {smallest_model['accuracy'] - smallest_model['accuracy_std']:.4f}"
            )
            print(f"  Parameters: {smallest_model['n_params']:,}")
            print(f"  n_dendrite_inputs: {smallest_model['n_dendrite_inputs']}")
            print(f"  n_dendrites: {smallest_model['n_dendrites']}")
            print(f"  percentage_resample: {smallest_model['percentage_resample']}")
            print(f"  steps_to_resample: {smallest_model['steps_to_resample']}")
            print(
                f"  training_time: {smallest_model['training_time']:.2f} ± {smallest_model['training_time_std']:.2f}s"
            )
            print(
                f"  n_runs: {smallest_model['n_runs']} (seeds: {smallest_model['seeds']})"
            )
        else:
            print(
                f"\nNo models reached {threshold * 100:.0f}% accuracy (conservative estimate)"
            )

        # Also show optimistic estimate (using just mean accuracy)
        optimistic_models = df[df["accuracy"] >= threshold]
        if len(optimistic_models) > 0:
            smallest_optimistic = optimistic_models.loc[
                optimistic_models["n_params"].idxmin()
            ]
            if (
                smallest_optimistic["n_params"] != smallest_model["n_params"]
                if len(threshold_models) > 0
                else True
            ):
                print(f"\n  (Optimistic estimate - using mean accuracy only):")
                print(
                    f"    Accuracy: {smallest_optimistic['accuracy']:.4f} ± {smallest_optimistic['accuracy_std']:.4f}"
                )
                print(f"    Parameters: {smallest_optimistic['n_params']:,}")
                print(
                    f"    Configuration: {smallest_optimistic['n_dendrite_inputs']} inputs, {smallest_optimistic['n_dendrites']} dendrites"
                )


def group_results_by_config(results):
    """Group results by hyperparameter configuration (excluding seed) and calculate statistics."""
    from collections import defaultdict

    # Group by all parameters except seed
    config_groups = defaultdict(list)

    for result in results:
        params = result["params"].copy()
        # Remove seed from params for grouping
        params.pop("seed", None)

        # Create a hashable key from the remaining parameters
        config_key = tuple(sorted(params.items()))
        config_groups[config_key].append(result)

    # Calculate statistics for each configuration
    grouped_results = []
    for config_key, config_results in config_groups.items():
        params = dict(config_key)

        # Extract all accuracy values for this configuration
        accuracies = [r["final_test_accuracy"] for r in config_results]
        training_times = [r["training_time"] for r in config_results]
        seeds = [
            r.get("seed", r["params"].get("seed", "unknown")) for r in config_results
        ]

        grouped_result = {
            "params": params,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_values": accuracies,
            "training_time_mean": np.mean(training_times),
            "training_time_std": np.std(training_times),
            "n_runs": len(config_results),
            "seeds": seeds,
            "final_test_accuracy": np.mean(
                accuracies
            ),  # For compatibility with existing code
        }
        grouped_results.append(grouped_result)

    return grouped_results


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
    grouped_results = group_results_by_config(results)
    plot_steps_to_resample_performance(grouped_results, results_dir)

    print("\n2. Plotting performance vs parameters...")
    plot_performance_vs_params(grouped_results, results_dir)

    print("\n3. Plotting resampling analysis...")
    plot_resampling_analysis(grouped_results, results_dir)

    print("\n4. Plotting training time analysis...")
    plot_training_time_vs_params(grouped_results, results_dir)

    print("\n5. Plotting parameter efficiency...")
    plot_parameter_efficiency(grouped_results, results_dir)

    print("\n6. Plotting best configurations...")
    plot_best_configurations(grouped_results, results_dir)

    print("\n7. Analyzing accuracy thresholds...")
    analyze_accuracy_thresholds(grouped_results)

    print(f"\nAll plots saved to: {results_dir}")
    print("Visualization complete!")


if __name__ == "__main__":
    main()
