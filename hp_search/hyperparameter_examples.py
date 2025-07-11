# %%
"""
Practical Examples for Hyperparameter Search

This script provides ready-to-use examples for different scenarios:
- Budget-constrained optimization
- Architecture comparison
- Quick prototyping vs thorough search
"""

from hyperparameter_search import HyperparameterSearcher
import time


def optimize_for_budget(max_params=8000, dataset="fashion-mnist", n_trials=50):
    """
    Find the best model architecture within a specific parameter budget.

    Args:
        max_params: Maximum number of parameters allowed
        dataset: Dataset to use for optimization
        n_trials: Number of trials to run
    """
    print(f"🎯 OPTIMIZING FOR BUDGET: {max_params} parameters")
    print(f"Dataset: {dataset}")
    print("=" * 60)

    searcher = HyperparameterSearcher(
        dataset=dataset,
        max_params=max_params,
        min_param_ratio=0.8,  # Require at least 80% of budget for efficiency
        n_epochs=10,  # Reduced for faster search
        verbose=True,
    )

    # Smart parameter ranges based on budget
    if max_params <= 2000:
        # Very small budget - focus on minimal architectures
        param_distributions = {
            "n_neurons": (5, 10),
            "n_dendrites": (2, 6),
            "n_dendrite_inputs": (16, 64),
            "soma_enabled": [True, False],
            "architecture": ["dendritic_simple"],
            "synaptic_resampling": [True, False],
            "percentage_resample": {"type": "uniform", "low": 0.1, "high": 0.4},
            "seed": {"type": "randint", "low": 1, "high": 10000},
        }
    elif max_params <= 5000:
        # Medium budget - explore different architectures
        param_distributions = {
            "n_neurons": (8, 15),
            "n_dendrites": (4, 10),
            "n_dendrite_inputs": (32, 128),
            "soma_enabled": [True, False],
            "architecture": ["dendritic_simple", "soma_disabled"],
            "synaptic_resampling": [True, False],
            "percentage_resample": {"type": "uniform", "low": 0.15, "high": 0.35},
            "seed": {"type": "randint", "low": 1, "high": 10000},
        }
    else:
        # Large budget - full exploration
        param_distributions = {
            "n_neurons": (10, 20),
            "n_dendrites": (6, 15),
            "n_dendrite_inputs": (64, 256),
            "soma_enabled": [True, False],
            "architecture": [
                "dendritic_simple",
                "dendritic_with_hidden",
                "soma_disabled",
            ],
            "synaptic_resampling": [True, False],
            "percentage_resample": {"type": "uniform", "low": 0.1, "high": 0.5},
            "hidden_dim": {"type": "randint", "low": 5, "high": 15},
            "seed": {"type": "randint", "low": 1, "high": 10000},
        }

    # Run search
    start_time = time.time()
    searcher.random_search(param_distributions, n_trials=n_trials)
    search_time = time.time() - start_time

    # Results
    searcher.print_results(5)
    print(f"\n⏱️  Total search time: {search_time:.1f}s")

    # Save results with budget in filename
    filename = f"budget_{max_params}_{dataset}_results.json"
    searcher.save_results(filename)

    return searcher.get_best_results(1)[0] if searcher.results else None


def compare_architectures(max_params=8000, dataset="fashion-mnist"):
    """
    Compare different dendritic architectures under the same parameter budget.
    """
    print("🏗️  ARCHITECTURE COMPARISON")
    print(f"Budget: {max_params} parameters, Dataset: {dataset}")
    print("=" * 60)

    architectures = [
        ("dendritic_simple", "Simple Dendritic (soma + output)"),
        ("soma_disabled", "Soma Disabled (dendrites → hidden → output)"),
        ("dendritic_with_hidden", "Dendritic with Hidden Layer"),
    ]

    results_summary = []

    for arch_name, arch_desc in architectures:
        print(f"\n🔬 Testing: {arch_desc}")
        print("-" * 40)

        searcher = HyperparameterSearcher(
            dataset=dataset,
            max_params=max_params,
            min_param_ratio=0.75,  # Slightly lower for architecture comparison
            n_epochs=12,
            verbose=False,  # Reduce verbosity for comparison
        )

        # Architecture-specific parameter ranges
        if arch_name == "dendritic_simple":
            param_grid = {
                "n_neurons": [8, 10, 12, 15],
                "n_dendrites": [4, 6, 8],
                "n_dendrite_inputs": [64, 96, 128],
                "soma_enabled": [True],
                "architecture": [arch_name],
                "synaptic_resampling": [True, False],
                "seed": [42, 123, 456],
            }
        elif arch_name == "soma_disabled":
            param_grid = {
                "n_neurons": [8, 10, 12],
                "n_dendrites": [
                    15,
                    20,
                    25,
                    30,
                ],  # More dendrites since they're direct outputs
                "n_dendrite_inputs": [32, 64, 96],
                "soma_enabled": [False],
                "architecture": [arch_name],
                "synaptic_resampling": [True, False],
                "seed": [42, 123, 456],
            }
        else:  # dendritic_with_hidden
            param_grid = {
                "n_neurons": [8, 10, 12],
                "n_dendrites": [4, 6, 8],
                "n_dendrite_inputs": [48, 64, 96],
                "hidden_dim": [6, 8, 10],
                "soma_enabled": [True],
                "architecture": [arch_name],
                "synaptic_resampling": [True],
                "seed": [42, 123, 456],
            }

        # Run grid search
        searcher.grid_search(param_grid, max_trials=15)

        if searcher.results:
            best = searcher.get_best_results(1)[0]
            results_summary.append(
                {
                    "architecture": arch_desc,
                    "test_accuracy": best.test_accuracy,
                    "params": best.total_params,
                    "config": best.params,
                }
            )
            print(
                f"✅ Best accuracy: {best.test_accuracy:.3f} ({best.total_params} params)"
            )
        else:
            print("❌ No valid configurations found within budget")

    # Final comparison
    print("\n📊 ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 60)
    results_summary.sort(key=lambda x: x["test_accuracy"], reverse=True)

    for i, result in enumerate(results_summary, 1):
        print(f"{i}. {result['architecture']}")
        print(f"   Accuracy: {result['test_accuracy']:.3f}")
        print(f"   Parameters: {result['params']}/{max_params}")
        print(f"   Config: {result['config']}")
        print()

    return results_summary


def quick_prototype_search(max_params=5000, dataset="mnist"):
    """
    Quick search for prototyping - fewer epochs, focused parameter ranges.
    """
    print("⚡ QUICK PROTOTYPE SEARCH")
    print(f"Budget: {max_params} parameters, Dataset: {dataset}")
    print("=" * 60)

    searcher = HyperparameterSearcher(
        dataset=dataset,
        max_params=max_params,
        min_param_ratio=0.8,  # Require good parameter utilization
        n_epochs=8,  # Faster training
        batch_size=512,  # Larger batches for speed
        verbose=True,
    )

    # Focused search space for quick results
    param_grid = {
        "n_neurons": [8, 10, 12],
        "n_dendrites": [4, 6, 8],
        "n_dendrite_inputs": [64, 96, 128],
        "soma_enabled": [True, False],
        "architecture": ["dendritic_simple"],
        "synaptic_resampling": [True],
        "percentage_resample": [0.25],
        "seed": [42],
    }

    start_time = time.time()
    searcher.grid_search(param_grid, max_trials=12)
    search_time = time.time() - start_time

    searcher.print_results(3)
    print(f"\n⚡ Quick search completed in {search_time:.1f}s")

    return searcher.get_best_results(1)[0] if searcher.results else None


def thorough_optimization(max_params=8000, dataset="fashion-mnist"):
    """
    Thorough search for final model - more epochs, comprehensive parameter space.
    """
    print("🔬 THOROUGH OPTIMIZATION")
    print(f"Budget: {max_params} parameters, Dataset: {dataset}")
    print("=" * 60)

    searcher = HyperparameterSearcher(
        dataset=dataset,
        max_params=max_params,
        min_param_ratio=0.85,  # Higher requirement for thorough optimization
        n_epochs=20,  # More epochs for better evaluation
        verbose=True,
    )

    # First phase: broad random search
    print("Phase 1: Broad Random Search")
    param_distributions = {
        "n_neurons": (5, 20),
        "n_dendrites": (3, 15),
        "n_dendrite_inputs": (32, 256),
        "soma_enabled": [True, False],
        "architecture": ["dendritic_simple", "soma_disabled", "dendritic_with_hidden"],
        "synaptic_resampling": [True, False],
        "percentage_resample": {"type": "uniform", "low": 0.1, "high": 0.5},
        "steps_to_resample": {"type": "choice", "values": [64, 128, 256]},
        "hidden_dim": {"type": "randint", "low": 5, "high": 15},
        "seed": {"type": "randint", "low": 1, "high": 10000},
    }

    searcher.random_search(param_distributions, n_trials=40)

    # Get top configurations for phase 2
    top_configs = searcher.get_best_results(5)

    print(f"\nPhase 2: Refinement around top {len(top_configs)} configurations")

    # Phase 2: refine around best configurations
    for i, config in enumerate(top_configs[:3]):  # Only refine top 3
        print(f"\nRefining configuration {i + 1}:")
        base_params = config.params

        # Create variations around the best config
        variations = []
        for seed in [42, 123, 456, 789, 999]:  # Multiple seeds
            variation = base_params.copy()
            variation["seed"] = seed

            # Small perturbations
            if "n_dendrite_inputs" in variation:
                original = variation["n_dendrite_inputs"]
                for delta in [-16, -8, 0, 8, 16]:
                    if original + delta > 16:  # Minimum reasonable value
                        new_variation = variation.copy()
                        new_variation["n_dendrite_inputs"] = original + delta
                        variations.append(new_variation)

        # Test variations
        for variation in variations[:10]:  # Limit variations
            result = searcher.train_and_evaluate(variation)
            if result:
                searcher.results.append(result)

    searcher.print_results(5)
    searcher.save_results("thorough_optimization_results.json")

    return searcher.get_best_results(1)[0] if searcher.results else None


def find_pareto_optimal(dataset="fashion-mnist", max_budget=10000):
    """
    Find Pareto optimal solutions: best accuracy vs parameter count trade-offs.
    """
    print("📈 PARETO OPTIMIZATION")
    print(f"Dataset: {dataset}, Max budget: {max_budget}")
    print("=" * 60)

    budgets = [2000, 4000, 6000, 8000, max_budget]
    pareto_results = []

    for budget in budgets:
        print(f"\n💰 Budget: {budget} parameters")
        result = optimize_for_budget(budget, dataset, n_trials=20)
        if result:
            pareto_results.append(
                {
                    "budget": budget,
                    "accuracy": result.test_accuracy,
                    "actual_params": result.total_params,
                    "config": result.params,
                }
            )

    # Display Pareto front
    print("\n📊 PARETO FRONT: Accuracy vs Parameters")
    print("=" * 60)
    for result in pareto_results:
        efficiency = result["accuracy"] / (
            result["actual_params"] / 1000
        )  # Accuracy per 1K params
        print(
            f"Budget: {result['budget']:5d} | "
            f"Params: {result['actual_params']:4d} | "
            f"Accuracy: {result['accuracy']:.3f} | "
            f"Efficiency: {efficiency:.3f}"
        )

    return pareto_results


def main():
    """Run example scenarios"""

    print("🚀 HYPERPARAMETER SEARCH EXAMPLES")
    print("=" * 80)

    # Example 1: Quick prototype
    print("\n1️⃣  QUICK PROTOTYPE (for rapid iteration)")
    quick_result = quick_prototype_search(max_params=5000, dataset="mnist")

    if quick_result:
        print(
            f"✅ Quick prototype found: {quick_result.test_accuracy:.3f} accuracy with {quick_result.total_params} params"
        )

    # Example 2: Budget optimization
    print("\n\n2️⃣  BUDGET OPTIMIZATION")
    budget_result = optimize_for_budget(
        max_params=8000, dataset="fashion-mnist", n_trials=25
    )

    if budget_result:
        print(
            f"✅ Best model: {budget_result.test_accuracy:.3f} accuracy with {budget_result.total_params} params"
        )

    # Example 3: Architecture comparison
    print("\n\n3️⃣  ARCHITECTURE COMPARISON")
    arch_results = compare_architectures(max_params=8000, dataset="fashion-mnist")

    if arch_results:
        best_arch = arch_results[0]
        print(
            f"✅ Best architecture: {best_arch['architecture']} with {best_arch['test_accuracy']:.3f} accuracy"
        )

    print(
        "\n🎉 All examples completed! Check the generated JSON files for detailed results."
    )


if __name__ == "__main__":
    main()
