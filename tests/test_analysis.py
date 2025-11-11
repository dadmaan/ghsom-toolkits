"""Tests for analysis tools."""

import pytest
import numpy as np
from unittest.mock import Mock
import matplotlib.pyplot as plt


class TestCompareModels:
    """Test model comparison function."""

    def setup_method(self):
        """Set up test models."""
        # Create mock models
        self.model1 = Mock()
        self.model1.rows = 2
        self.model1.columns = 2
        self.model1.input_dataset_size = 100
        self.model1.t1 = 0.5
        self.model1.t2 = 0.05
        self.model1.map = [[Mock() for _ in range(2)] for _ in range(2)]

        # Make neurons with weights
        for r in range(2):
            for c in range(2):
                self.model1.map[r][c].weights = np.random.rand(5)
                self.model1.map[r][c].children = []

        # Add some children to create hierarchy
        child1 = Mock()
        child1.children = []
        child1.input_dataset_size = 50
        self.model1.children = [child1]

        self.model2 = Mock()
        self.model2.rows = 3
        self.model2.columns = 3
        self.model2.input_dataset_size = 100
        self.model2.t1 = 0.3
        self.model2.t2 = 0.03
        self.model2.map = [[Mock() for _ in range(3)] for _ in range(3)]

        for r in range(3):
            for c in range(3):
                self.model2.map[r][c].weights = np.random.rand(5)
                self.model2.map[r][c].children = []

        self.model2.children = []

        self.data = np.random.rand(100, 5)

    def test_compare_models_with_pandas(self):
        """Test model comparison when pandas is available."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import compare_models

        models = [self.model1, self.model2]
        comparison = compare_models(models, self.data)

        assert comparison is not None
        assert len(comparison) == 2  # Two models

    def test_compare_models_without_pandas(self, monkeypatch):
        """Test that missing pandas raises ImportError."""
        import ghsom_toolkits.analysis.comparisons as comp_module
        monkeypatch.setattr(comp_module, "PANDAS_AVAILABLE", False)

        from ghsom_toolkits.analysis.comparisons import compare_models

        with pytest.raises(ImportError, match="pandas is required"):
            compare_models([self.model1, self.model2], self.data)

    def test_compare_models_with_names(self):
        """Test model comparison with custom names."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import compare_models

        models = [self.model1, self.model2]
        names = ['Loose', 'Tight']
        comparison = compare_models(models, self.data, model_names=names)

        assert 'Loose' in comparison.index
        assert 'Tight' in comparison.index

    def test_compare_models_metric_selection(self):
        """Test model comparison with specific metrics."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import compare_models

        models = [self.model1, self.model2]
        metrics = ['num_nodes', 'depth']
        comparison = compare_models(models, self.data, metrics=metrics)

        assert 'num_nodes' in comparison.columns
        assert 'depth' in comparison.columns


class TestPlotComparison:
    """Test comparison plotting function."""

    def setup_method(self):
        """Set up test data."""
        try:
            import pandas as pd
            self.comparison_df = pd.DataFrame({
                'num_nodes': [10, 15, 20],
                'depth': [2, 3, 3],
                'qe': [0.5, 0.4, 0.3],
            }, index=['Model 1', 'Model 2', 'Model 3'])
        except ImportError:
            pytest.skip("pandas not available")

    def test_plot_comparison_bar(self):
        """Test bar plot comparison."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import plot_comparison

        fig = plot_comparison(self.comparison_df, plot_type='bar')

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison_radar(self):
        """Test radar plot comparison."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import plot_comparison

        fig = plot_comparison(self.comparison_df, plot_type='radar')

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison_heatmap(self):
        """Test heatmap comparison."""
        pytest.importorskip("pandas")
        pytest.importorskip("seaborn")

        from ghsom_toolkits.analysis.comparisons import plot_comparison

        fig = plot_comparison(self.comparison_df, plot_type='heatmap')

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison_invalid_type(self):
        """Test that invalid plot type raises ValueError."""
        pytest.importorskip("pandas")

        from ghsom_toolkits.analysis.comparisons import plot_comparison

        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_comparison(self.comparison_df, plot_type='invalid')


class TestGenerateReport:
    """Test report generation function."""

    def setup_method(self):
        """Set up test model."""
        self.model = Mock()
        self.model.rows = 2
        self.model.columns = 2
        self.model.input_dataset_size = 100
        self.model.level = 0
        self.model.t1 = 0.5
        self.model.t2 = 0.05
        self.model.learning_rate = 0.1
        self.model.gaussian_sigma = 1.0
        self.model.decay = 0.9
        self.model.map = [[Mock() for _ in range(2)] for _ in range(2)]
        self.model.children = []

        for r in range(2):
            for c in range(2):
                self.model.map[r][c].weights = np.random.rand(5)
                self.model.map[r][c].children = []

        self.data = np.random.rand(100, 5)

    def test_generate_report_with_jinja2(self, tmp_path):
        """Test report generation when jinja2 is available."""
        pytest.importorskip("jinja2")

        from ghsom_toolkits.analysis.reports import generate_report

        output_path = tmp_path / "test_report.html"
        result = generate_report(
            self.model,
            self.data,
            output_path=str(output_path),
            include_plots=False  # Skip plots for faster testing
        )

        assert output_path.exists()
        assert result == str(output_path)

        # Check HTML content
        content = output_path.read_text()
        assert "GHSOM Model Report" in content
        assert "Growth Threshold" in content

    def test_generate_report_without_jinja2(self, monkeypatch, tmp_path):
        """Test that missing jinja2 raises ImportError."""
        import ghsom_toolkits.analysis.reports as reports_module
        monkeypatch.setattr(reports_module, "JINJA2_AVAILABLE", False)

        from ghsom_toolkits.analysis.reports import generate_report

        with pytest.raises(ImportError, match="jinja2 is required"):
            generate_report(self.model, self.data, output_path=str(tmp_path / "report.html"))

    def test_generate_report_with_custom_info(self, tmp_path):
        """Test report generation with custom information."""
        pytest.importorskip("jinja2")

        from ghsom_toolkits.analysis.reports import generate_report

        output_path = tmp_path / "custom_report.html"
        custom_info = "<p>This is custom information</p>"

        generate_report(
            self.model,
            self.data,
            output_path=str(output_path),
            include_plots=False,
            custom_info=custom_info
        )

        content = output_path.read_text()
        assert "custom information" in content
