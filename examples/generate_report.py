"""
Report generation example.

This script demonstrates how to generate a comprehensive HTML report
for a trained GHSOM model.

Note: Requires jinja2 to be installed.
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
    """Train a GHSOM model and generate a comprehensive report."""
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
        data = np.random.rand(150, 10)

        print("Training GHSOM model...")
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



    # Build lookup table
    lookup = build_lookup_table(model)
    print(f"Total nodes in hierarchy: {len(lookup)}")

    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE HTML REPORT")
    print("=" * 60)

    try:
        from ghsom_toolkits.analysis import generate_report

        # Add custom information
        custom_info = """
        <h3>Experiment Details</h3>
        <ul>
            <li><strong>Dataset:</strong> Random synthetic data</li>
            <li><strong>Purpose:</strong> Demonstration of GHSOM capabilities</li>
            <li><strong>Training Time:</strong> ~5 seconds</li>
            <li><strong>Hardware:</strong> Standard CPU</li>
        </ul>
        <h3>Notes</h3>
        <p>This model was trained with balanced parameters to create a moderate hierarchy depth
        while maintaining reasonable cluster sizes.</p>
        """

        # Collect external plots from example outputs
        external_plots = {}

        # Check for available visualization files
        plot_files = {
            "Weight Heatmap - All Neurons": "weight_heatmap_all.png",
            "Weight Heatmap - Single Neuron": "weight_heatmap_single.png",
            "Activation Maps": "activation_maps.png",
            "U-Matrix (Distance Matrix)": "umatrix.png",
            "Cluster Quality Metrics": "cluster_quality.png",
            "Growth Timeline": "growth_timeline.png",
        }

        for plot_name, filename in plot_files.items():
            plot_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(plot_path):
                external_plots[plot_name] = plot_path
                print(f"  ✓ Found: {filename}")
            else:
                print(f"  ⊗ Not found: {filename} (will be skipped)")

        print(f"\nIncluding {len(external_plots)} external visualizations in report...")

        output_path = generate_report(
            model=model,
            data=data,
            output_path=os.path.join(OUTPUT_DIR, "ghsom_report.html"),
            lookup_table=lookup,
            include_plots=True,
            custom_info=custom_info,
            external_plots=external_plots,
        )

        print(f"\n✨ Report successfully generated!")
        print(f"\n📄 Report saved to: {output_path}")
        print("\nThe report includes:")
        print("  • Model information and metrics")
        print("  • Hierarchy structure visualization")
        print("  • Cluster size distribution")
        print(f"  • {len(external_plots)} additional visualizations")
        print("\nOpen the report in your browser to explore:")
        print(f"  file://{os.path.abspath(output_path)}")

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use report generation, install optional dependencies:")
        print("  pip install ghsom-toolkits[interactive]")
        print("\nOr install all optional dependencies:")
        print("  pip install ghsom-toolkits[all]")


if __name__ == "__main__":
    main()
