"""
Cluster analysis example.

This script demonstrates advanced cluster analysis features including
distribution plots, quality metrics, and growth timeline visualization.

Note: Requires scikit-learn for quality metrics.
Install with: pip install ghsom-toolkits[analysis]
"""

import os
import pickle
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table
from ghsom_toolkits.plotting.clusters import (
    plot_cluster_distribution,
    plot_cluster_quality,
    plot_growth_timeline,
)
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "../example_outputs"
PRETRAINED_MODEL = "../../ghsom-py/experiment_output/ghsom_model.pkl"


def main():
    """Train a GHSOM model and perform cluster analysis."""
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
        # Create sample data with distinct clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 10) + np.array([0] * 10)
        cluster2 = np.random.randn(50, 10) + np.array([5] * 10)
        cluster3 = np.random.randn(50, 10) + np.array([-5] * 10)
        data = np.vstack([cluster1, cluster2, cluster3])

        print("Training GHSOM model on clustered data...")
        ghsom = GHSOM(
            input_dataset=data,
            t1=0.4,
            t2=0.04,
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

    # 1. Plot cluster distribution
    print("\n1. Analyzing cluster size distribution...")
    fig1 = plot_cluster_distribution(
        model,
        figsize=(12, 6),
        save_path=os.path.join(OUTPUT_DIR, "cluster_distribution.png"),
    )
    plt.close(fig1)
    print("  ✓ Saved to cluster_distribution.png")

    # 2. Plot cluster quality metrics (if sklearn available)
    print("\n2. Computing cluster quality metrics...")
    try:
        # Assign samples to clusters
        cluster_labels = []
        for sample in data:
            # Find BMU
            current = model
            while True:
                min_dist = float("inf")
                bmu_pos = (0, 0)

                for r in range(current.rows):
                    for c in range(current.columns):
                        dist = np.linalg.norm(sample - current.map[r][c].weights)
                        if dist < min_dist:
                            min_dist = dist
                            bmu_pos = (r, c)

                # Check if BMU has children
                bmu_neuron = current.map[bmu_pos[0]][bmu_pos[1]]
                if bmu_neuron.children:
                    current = bmu_neuron.children[0]
                else:
                    # Reached leaf - create unique ID
                    cluster_id = id(bmu_neuron)
                    cluster_labels.append(cluster_id)
                    break

        # Convert to sequential IDs
        unique_ids = list(set(cluster_labels))
        cluster_labels = np.array([unique_ids.index(cid) for cid in cluster_labels])

        fig2 = plot_cluster_quality(
            data,
            cluster_labels,
            figsize=(14, 5),
            save_path=os.path.join(OUTPUT_DIR, "cluster_quality.png"),
        )
        plt.close(fig2)
        print("  ✓ Saved to cluster_quality.png")

    except ImportError as e:
        print(f"  ⚠ Skipping quality metrics: {e}")
        print("  Install with: pip install ghsom-toolkits[analysis]")

    # 3. Plot growth timeline (with simulated history)
    print("\n3. Creating growth timeline...")
    # Simulate training history
    training_history = [
        {"epoch": 0, "num_nodes": 4, "depth": 1, "qe": 0.8, "te": 0.15},
        {"epoch": 10, "num_nodes": 8, "depth": 2, "qe": 0.6, "te": 0.12},
        {"epoch": 20, "num_nodes": 12, "depth": 2, "qe": 0.45, "te": 0.10},
        {"epoch": 30, "num_nodes": 16, "depth": 3, "qe": 0.35, "te": 0.08},
        {"epoch": 40, "num_nodes": 18, "depth": 3, "qe": 0.28, "te": 0.06},
        {"epoch": 50, "num_nodes": 20, "depth": 3, "qe": 0.25, "te": 0.05},
    ]

    fig3 = plot_growth_timeline(
        training_history,
        figsize=(12, 8),
        save_path=os.path.join(OUTPUT_DIR, "growth_timeline.png"),
    )
    plt.close(fig3)
    print("  ✓ Saved to growth_timeline.png")

    print("\n✨ Cluster analysis completed!")
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("  - cluster_distribution.png")
    print("  - cluster_quality.png (if sklearn available)")
    print("  - growth_timeline.png")


if __name__ == "__main__":
    main()
