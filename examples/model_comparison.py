"""
Model comparison example.

This script demonstrates how to compare multiple GHSOM models
with different hyperparameters and visualize the comparison.

Note: Requires pandas to be installed.
Install with: pip install ghsom-toolkits[pandas]
"""

import os
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "../example_outputs"


def main():
    """Train multiple GHSOM models and compare them."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create sample data
    np.random.seed(42)
    data = np.random.rand(200, 10)

    # Train models with different hyperparameters
    print("Training Model 1 (Loose clustering: t1=0.5, t2=0.05)...")
    result1 = GHSOM(
        input_dataset=data,
        t1=0.5,
        t2=0.05,
        learning_rate=0.1,
        gaussian_sigma=1.0,
        decay=0.9,
    ).train(epochs_number=50, n_workers=1)
    model1 = result1.child_map if hasattr(result1, 'child_map') else result1
    print(f"  ✓ Root map: {model1.map_shape()}")

    print("\nTraining Model 2 (Medium clustering: t1=0.3, t2=0.03)...")
    result2 = GHSOM(
        input_dataset=data,
        t1=0.3,
        t2=0.03,
        learning_rate=0.1,
        gaussian_sigma=1.0,
        decay=0.9,
    ).train(epochs_number=50, n_workers=1)
    model2 = result2.child_map if hasattr(result2, 'child_map') else result2
    print(f"  ✓ Root map: {model2.map_shape()}")

    print("\nTraining Model 3 (Tight clustering: t1=0.2, t2=0.02)...")
    result3 = GHSOM(
        input_dataset=data,
        t1=0.2,
        t2=0.02,
        learning_rate=0.1,
        gaussian_sigma=1.0,
        decay=0.9,
    ).train(epochs_number=50, n_workers=1)
    model3 = result3.child_map if hasattr(result3, 'child_map') else result3
    print(f"  ✓ Root map: {model3.map_shape()}")

    # Compare models
    print("\n" + "=" * 60)
    print("COMPARING MODELS")
    print("=" * 60)

    try:
        from ghsom_toolkits.analysis import compare_models, plot_comparison

        # Create comparison table
        models = [model1, model2, model3]
        model_names = ["Loose (t1=0.5)", "Medium (t1=0.3)", "Tight (t1=0.2)"]

        comparison = compare_models(
            models,
            data,
            model_names=model_names,
            metrics=["num_nodes", "num_clusters", "depth", "qe", "avg_cluster_size"],
        )

        print("\nComparison Table:")
        print(comparison.to_string())

        # Create visualizations
        print("\nGenerating comparison plots...")

        # Bar chart
        fig1 = plot_comparison(
            comparison,
            plot_type="bar",
            figsize=(14, 6),
            save_path=os.path.join(OUTPUT_DIR, "comparison_bar.png"),
        )
        plt.close(fig1)
        print("  ✓ Saved bar chart to comparison_bar.png")

        # Radar chart
        fig2 = plot_comparison(
            comparison,
            plot_type="radar",
            figsize=(10, 8),
            save_path=os.path.join(OUTPUT_DIR, "comparison_radar.png"),
        )
        plt.close(fig2)
        print("  ✓ Saved radar chart to comparison_radar.png")

        # Heatmap
        fig3 = plot_comparison(
            comparison,
            plot_type="heatmap",
            figsize=(10, 6),
            save_path=os.path.join(OUTPUT_DIR, "comparison_heatmap.png"),
        )
        plt.close(fig3)
        print("  ✓ Saved heatmap to comparison_heatmap.png")

        print("\n✨ Model comparison completed!")
        print(f"\nGenerated files in {OUTPUT_DIR}:")
        print("  - comparison_bar.png")
        print("  - comparison_radar.png")
        print("  - comparison_heatmap.png")

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use model comparison, install optional dependencies:")
        print("  pip install ghsom-toolkits[pandas]")
        print("\nOr install all optional dependencies:")
        print("  pip install ghsom-toolkits[all]")


if __name__ == "__main__":
    main()
