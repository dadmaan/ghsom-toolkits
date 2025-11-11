"""
Basic GHSOM visualization example.

This script demonstrates how to use ghsom-toolkits to visualize
a trained GHSOM model using the compatibility adapter.
"""

import os
import pickle
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits import visualize_ghsom_hierarchy
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# Configuration
OUTPUT_DIR = "../example_outputs"
PRETRAINED_MODEL = "../../ghsom-py/experiment_output/ghsom_model.pkl"


def main():
    """Train a simple GHSOM and visualize it."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to load pre-trained model, fallback to training new one
    if os.path.exists(PRETRAINED_MODEL):
        print(f"Loading pre-trained model from: {PRETRAINED_MODEL}")
        with open(PRETRAINED_MODEL, 'rb') as f:
            result = pickle.load(f)
        print(f"✓ Model loaded successfully!")
        # Create compatible sample data for the loaded model
        np.random.seed(42)
        data = np.random.rand(200, 10)  # Match the pre-trained model's dataset shape
    else:
        print(f"Pre-trained model not found. Training new model...")
        # Create sample data
        np.random.seed(42)
        data = np.random.rand(100, 10)

        print("Training GHSOM model...")
        ghsom = GHSOM(
            input_dataset=data,
            t1=0.5,  # Growth threshold
            t2=0.05,  # Stopping criterion
            learning_rate=0.1,
            gaussian_sigma=1.0,
            decay=0.9,
        )

        result = ghsom.train(epochs_number=50, n_workers=1)
        print(f"✓ Training complete!")

    # Adapt model to ghsom-toolkits interface
    print("Adapting model for visualization...")
    model = adapt_model(result)
    print(f"✓ Model adapted! Level: {model.level}, Map: {model.rows}x{model.columns}")

    # Build lookup table for visualization
    lookup = build_lookup_table(model)
    print(f"Total nodes in hierarchy: {len(lookup)}")

    # Visualize the hierarchy
    print("Generating visualization...")
    output_file = os.path.join(OUTPUT_DIR, "ghsom_hierarchy_example.png")
    visualize_ghsom_hierarchy(
        node=model, lookup_table=lookup, filename=output_file
    )

    print(f"✓ Visualization saved to: {output_file}")


if __name__ == "__main__":
    main()
