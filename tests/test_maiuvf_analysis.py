"""
Unit tests for MA-IUVF Analysis components

Tests the shared analyzer utilities and individual experiment components
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

# Test data paths
CHECKPOINT_PATH = "outputs/maiuvf_phase1/bspline_hash_dynamic_anchor1_metric0p01/run_000_samples300000_sigma0p01_epochs100/train/best.pt"
MESH_PATH = "data/models/stanford_bunny_procedural.obj"


class TestMAIUVFAnalyzer:
    """Test MAIUVFAnalyzer base class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        from src.analysis.maiuvf_analyzer import MAIUVFAnalyzer

        # Skip test if checkpoint not available
        if not Path(CHECKPOINT_PATH).exists():
            pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")

        return MAIUVFAnalyzer(CHECKPOINT_PATH, device='cpu')

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.model is not None
        assert analyzer.num_charts == 8
        assert analyzer.baker_metadata is not None

    def test_get_network_outputs(self, analyzer):
        """Test network output extraction"""
        # Create dummy positions
        positions = np.random.randn(10, 3).astype(np.float32)

        # Get outputs
        outputs = analyzer.get_network_outputs(positions, return_probs=True)

        # Check shapes
        assert outputs['logits'].shape == (10, 8)
        assert outputs['uv_preds'].shape == (10, 8, 2)
        assert outputs['probs'].shape == (10, 8)
        assert outputs['chart_ids'].shape == (10,)
        assert outputs['selected_uvs'].shape == (10, 2)

        # Check probabilities sum to 1
        probs_sum = outputs['probs'].sum(axis=1)
        assert np.allclose(probs_sum, 1.0, atol=1e-5)

    def test_compute_entropy(self, analyzer):
        """Test entropy computation"""
        # Create dummy probability distributions
        probs = np.random.rand(10, 8)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Compute entropy
        entropy = analyzer.compute_entropy(probs)

        # Check shape
        assert entropy.shape == (10,)

        # Check non-negative
        assert np.all(entropy >= 0)

        # Check uniform distribution has maximum entropy
        uniform_probs = np.ones((1, 8)) / 8
        uniform_entropy = analyzer.compute_entropy(uniform_probs)
        assert uniform_entropy[0] > 0

    def test_compute_correlation(self, analyzer):
        """Test correlation computation"""
        # Create correlated data
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1

        # Compute correlation
        r, p = analyzer.compute_correlation(x, y)

        # Check correlation is high
        assert r > 0.9
        assert p < 0.05


class TestJacobianComputation:
    """Test Jacobian computation utilities"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        from src.analysis.maiuvf_analyzer import MAIUVFAnalyzer

        if not Path(CHECKPOINT_PATH).exists():
            pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")

        return MAIUVFAnalyzer(CHECKPOINT_PATH, device='cpu')

    def test_compute_jacobians(self, analyzer):
        """Test Jacobian computation via autograd"""
        # Create dummy positions
        positions = np.random.randn(5, 3).astype(np.float32)
        chart_ids = np.random.randint(0, 8, 5)

        # Compute Jacobians
        jacobians = analyzer.compute_jacobians(positions, chart_ids, batch_size=5)

        # Check shape
        assert jacobians.shape == (5, 2, 3)

        # Check finite values
        assert np.all(np.isfinite(jacobians))

    def test_get_deformation_energy(self, analyzer):
        """Test deformation energy computation"""
        # Create dummy Jacobians
        jacobians = np.random.randn(10, 2, 3)

        # Compute deformation energy
        energy = analyzer.get_deformation_energy(jacobians)

        # Check shape
        assert energy.shape == (10,)

        # Check non-negative
        assert np.all(energy >= 0)

    def test_get_normal_derivative(self, analyzer):
        """Test normal derivative computation"""
        # Create dummy Jacobians and normals
        jacobians = np.random.randn(10, 2, 3)
        normals = np.random.randn(10, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Compute normal derivatives
        D_normal = analyzer.get_normal_derivative(jacobians, normals)

        # Check shape
        assert D_normal.shape == (10,)

        # Check non-negative
        assert np.all(D_normal >= 0)


class TestVisualizationUtils:
    """Test visualization utilities"""

    def test_setup_plot_style(self):
        """Test plot style setup"""
        from src.analysis.utils import setup_plot_style

        # Should not raise error
        setup_plot_style()

    def test_generate_statistics_table(self):
        """Test statistics generation"""
        from src.analysis.utils import generate_statistics_table

        # Create dummy data
        data = np.random.randn(100)

        # Generate statistics
        stats = generate_statistics_table(data)

        # Check keys
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'p50' in stats
        assert 'p90' in stats
        assert 'p95' in stats
        assert 'p99' in stats


class TestThicknessComputation:
    """Test thickness computation"""

    @pytest.fixture
    def mesh(self):
        """Create simple test mesh"""
        import trimesh

        # Create a simple cube
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 2, 6], [1, 6, 5]   # Right
        ])

        return trimesh.Trimesh(vertices=vertices, faces=faces)

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        from src.analysis.maiuvf_analyzer import MAIUVFAnalyzer

        if not Path(CHECKPOINT_PATH).exists():
            pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")

        return MAIUVFAnalyzer(CHECKPOINT_PATH, device='cpu')

    def test_compute_thickness(self, analyzer, mesh):
        """Test thickness computation"""
        # Test on a few vertices
        vertices = mesh.vertices[:4]
        normals = mesh.vertex_normals[:4]

        # Compute thickness
        thickness = analyzer.compute_thickness(vertices, normals, mesh)

        # Check shape
        assert thickness.shape == (4,)

        # Check positive values
        assert np.all(thickness > 0)


class TestAnalysisResult:
    """Test AnalysisResult dataclass"""

    def test_save_and_load(self):
        """Test saving and loading results"""
        from src.analysis.maiuvf_analyzer import AnalysisResult

        # Create dummy result
        data = {
            'x': np.array([1, 2, 3]),
            'y': np.array([4, 5, 6])
        }
        metadata = {'test': 'data'}
        figures = ['fig1', 'fig2']

        result = AnalysisResult(
            experiment_name='test',
            data=data,
            metadata=metadata,
            figures=figures
        )

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir)

            # Check files exist
            assert Path(tmpdir, 'data.csv').exists()
            assert Path(tmpdir, 'metadata.json').exists()

            # Load and check
            import pandas as pd
            loaded_data = pd.read_csv(Path(tmpdir, 'data.csv'))
            assert len(loaded_data) == 3


class TestIntegration:
    """Integration tests for complete experiment pipeline"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        from src.analysis.maiuvf_analyzer import MAIUVFAnalyzer

        if not Path(CHECKPOINT_PATH).exists():
            pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")

        return MAIUVFAnalyzer(CHECKPOINT_PATH, device='cpu')

    @pytest.fixture
    def mesh(self):
        """Create mesh instance"""
        import trimesh

        if not Path(MESH_PATH).exists():
            pytest.skip(f"Mesh not found: {MESH_PATH}")

        mesh = trimesh.load(MESH_PATH)
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        return mesh

    def test_experiment_4_pipeline(self, analyzer, mesh):
        """Test Experiment 4 complete pipeline"""
        from src.analysis.experiments import run_experiment4

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run experiment with small sample
            result = run_experiment4(
                analyzer=analyzer,
                mesh=mesh,
                num_samples=100,  # Small sample for testing
                output_dir=tmpdir
            )

            # Check result
            assert result.experiment_name == 'exp4_normal_noise'
            assert 'point_id' in result.data
            assert 'D_normal' in result.data
            assert len(result.figures) > 0

            # Check files created
            assert Path(tmpdir, 'data.csv').exists()
            assert Path(tmpdir, 'report.md').exists

    def test_create_output_dir(self, analyzer):
        """Test output directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = analyzer.create_output_dir("test_exp")

            assert output_dir.exists()
            assert output_dir.name == "test_exp"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
