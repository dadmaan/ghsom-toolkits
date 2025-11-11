"""
Heatmap visualization example.

This script demonstrates how to use the heatmap visualization features
to explore neuron weights, activation patterns, and U-Matrix.
"""

import os
import pickle
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.plotting.heatmaps import (
    plot_activation_map,
    plot_umatrix,
    plot_weight_heatmap,
)
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "../example_outputs"
PRETRAINED_MODEL = "../../ghsom-py/experiment_output/ghsom_model.pkl"


def main():
    """Train a GHSOM model and create heatmap visualizations."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to load pre-trained model, fallback to training new one
    if os.path.exists(PRETRAINED_MODEL):
        print(f"Loading pre-trained model from: {PRETRAINED_MODEL}")
        with open(PRETRAINED_MODEL, 'rb') as f:
            loaded = pickle.load(f)
        # Handle if the pickle contains a Neuron with child_map
        result = loaded
        print(f"✓ Model loaded successfully!")
        # Create compatible sample data
        np.random.seed(42)
        data = np.random.rand(200, 10)
    else:
        print(f"Pre-trained model not found. Training new model...")
        # Create sample data
        np.random.seed(42)
        data = np.random.rand(100, 10)

        print("Training GHSOM model...")
        ghsom = GHSOM(
            input_dataset=data,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            gaussian_sigma=1.0,
            decay=0.9,
        )

        result = ghsom.train(epochs_number=50, n_workers=1)
        # Extract the root GSOM from the result
        # Model will be adapted below
        print(f"✓ Training complete!")

    # Adapt model to ghsom-toolkits interface
    model = adapt_model(result)
    print(f"✓ Model adapted! Level: {model.level}, Map: {model.rows}x{model.columns}")

    # 1. Plot weight heatmap for all neurons
    print("\n1. Creating weight heatmap...")
    fig1 = plot_weight_heatmap(
        model,
        figsize=(12, 8),
        cmap="viridis",
        save_path=os.path.join(OUTPUT_DIR, "weight_heatmap_all.png"),
    )
    plt.close(fig1)
    print("  ✓ Saved to weight_heatmap_all.png")

    # 2. Plot weight heatmap for a specific neuron
    print("\n2. Creating single neuron weight heatmap...")
    fig2 = plot_weight_heatmap(
        model,
        neuron_position=(0, 0),
        figsize=(10, 4),
        cmap="coolwarm",
        save_path=os.path.join(OUTPUT_DIR, "weight_heatmap_single.png"),
    )
    plt.close(fig2)
    print("  ✓ Saved to weight_heatmap_single.png")

    # 3. Plot activation maps for sample data
    print("\n3. Creating activation maps...")
    fig3 = plot_activation_map(
        model,
        data,
        sample_indices=[0, 1, 2, 3],
        cmap="YlOrRd",
        figsize=(12, 8),
        save_path=os.path.join(OUTPUT_DIR, "activation_maps.png"),
    )
    plt.close(fig3)
    print("  ✓ Saved to activation_maps.png")

    # 4. Plot U-Matrix
    print("\n4. Creating U-Matrix visualization...")
    fig4 = plot_umatrix(
        model,
        cmap="gray_r",
        figsize=(8, 6),
        save_path=os.path.join(OUTPUT_DIR, "umatrix.png"),
    )
    plt.close(fig4)
    print("  ✓ Saved to umatrix.png")

    print("\n✨ All heatmap visualizations completed!")
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("  - weight_heatmap_all.png")
    print("  - weight_heatmap_single.png")
    print("  - activation_maps.png")
    print("  - umatrix.png")


if __name__ == "__main__":
    main()
