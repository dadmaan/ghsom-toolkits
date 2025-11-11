"""
Interactive dashboard example.

This script demonstrates how to launch an interactive Dash dashboard
for exploring GHSOM models.

Note: Requires plotly and dash to be installed.
Install with: pip install ghsom-toolkits[interactive]
"""

import os
import pickle
import numpy as np
from ghsom import GHSOM
from ghsom_toolkits.adapters import adapt_model, build_lookup_table

# Configuration
OUTPUT_DIR = "../example_outputs"
PRETRAINED_MODEL = "../../ghsom-py/experiment_output/ghsom_model.pkl"



def main():
    """Train a GHSOM model and launch interactive dashboard."""
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



    # Build lookup table
    lookup = build_lookup_table(model)
    print(f"Total nodes in hierarchy: {len(lookup)}")

    # Launch dashboard
    print("\n" + "=" * 60)
    print("LAUNCHING INTERACTIVE DASHBOARD")
    print("=" * 60)

    try:
        from ghsom_toolkits.interactive import launch_dashboard

        launch_dashboard(
            node=model,
            lookup_table=lookup,
            data=data,
            port=8050,
            debug=False,
        )
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use the interactive dashboard, install optional dependencies:")
        print("  pip install ghsom-toolkits[interactive]")
        print("\nOr install all optional dependencies:")
        print("  pip install ghsom-toolkits[all]")


if __name__ == "__main__":
    main()
