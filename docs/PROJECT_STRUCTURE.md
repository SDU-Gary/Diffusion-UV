# Diffusion-UV Project Structure

## Overview

Diffusion-UV is a research-to-production project implementing **Metric-Aligned Implicit UV Fields (MA-IUVF)** for low-poly mesh coloring under shared texture constraints.

## Directory Structure

```
Diffusion-UV/
├── configs/                          # Configuration files (YAML)
│   ├── default.yaml                  # Default experiment configuration
│   ├── maiuvf_baseline.yaml         # MA-IUVF baseline configuration
│   ├── production.yaml                # Production settings
│   ├── gpu_training.yaml              # GPU training configuration
│   └── bunny_test.yaml                # Stanford Bunny test configuration
│
├── docs/                             # Documentation
│   ├── PROJECT_REPORT.md             # Academic project report
│   ├── PROJECT_STRUCTURE.md          # This file
│   ├── CLAUDE.md                     # Comprehensive project documentation
│   ├── START.md                      # Detailed technical design
│   └── ALSFD.md                      # ALSFD method documentation
│
├── scripts/                          # Training and evaluation scripts
│   ├── train_metric_aligned_iuv_field.py   # MA-IUVF training script
│   ├── run_maiuvf_experiment.py             # End-to-end experiment runner
│   ├── infer_metric_aligned_iuv.py          # MA-IUVF inference
│   ├── bake_metric_aligned_iuv_data.py     # Data baking
│   ├── render_metric_aligned_iuv_test.py   # Rendering validation
│   └── ...                           # (63 total script files)
│
├── src/                              # Source code
│   ├── models/                       # Neural network implementations
│   │   ├── metric_aligned_iuv_field.py    # Core MA-IUVF network
│   │   ├── encoders/
│   │   │   └── bspline_grid.py            # B-Spline hash grid encoder
│   │   ├── sdf_network.py                 # SDF network for geometry
│   │   ├── texture_sampler_field.py       # Texture sampler
│   │   ├── network_g.py                   # Geometry network (original)
│   │   ├── network_d.py                   # Diffusion network (original)
│   │   └── network_r.py                   # Reverse mapping (original)
│   │
│   ├── data/                         # Data loading and preprocessing
│   │   ├── obj_parser.py                # OBJ parser with face-corner UVs
│   │   ├── uv_chart_segmentation.py      # UV chart segmentation
│   │   ├── metric_aligned_iuv_baker.py    # Training data baker
│   │   └── gpu_dataset.py                # GPU sampling utilities
│   │
│   ├── training/                     # Training logic and losses
│   │   ├── metric_aligned_iuv_losses.py   # MA-IUVF loss functions
│   │   ├── sdf_losses.py                 # SDF-specific losses
│   │   └── __init__.py                    # Training orchestration
│   │
│   ├── inference/                    # Inference and rendering
│   │   ├── metric_aligned_iuv_inference.py # MA-IUVF inference engine
│   │   ├── offline_renderer.py            # CPU rasterizer
│   │   ├── opengl_renderer.py             # OpenGL renderer (experimental)
│   │   └── mesh_simplification.py         # Mesh simplification
│   │
│   ├── geometry/                     # Geometric processing
│   │   ├── alsfd_diffusion.py            # ALSFD diffusion
│   │   ├── alsfd_diffusion_fixed.py      # Fixed ALSFD implementation
│   │   ├── projection.py                 # Projection utilities
│   │   └── heat_method.py                # Geodesic distances
│   │
│   ├── analysis/                     # Analysis tools
│   │   └── maiuvf_analyzer.py            # MA-IUVF analysis
│   │
│   ├── ops/                          # Operations
│   │   └── bspline_ops.py               # B-Spline operations
│   │
│   └── utils/                        # Utilities
│       └── ...                         # Supporting utilities
│
├── tests/                            # Unit tests (49+ tests passing)
│   ├── test_metric_aligned_iuv_baker.py
│   ├── test_metric_aligned_iuv_training.py
│   ├── test_metric_aligned_iuv_inference.py
│   ├── test_multichart_training.py
│   ├── test_offline_renderer_sampling.py
│   └── ...                           # (15 total test files)
│
├── data/                             # Data assets
│   ├── models/                       # Test models
│   │   └── stanford_bunny_procedural.obj
│   └── textures/                     # Texture files
│       └── bunny_texture.png
│
├── outputs/                          # Generated outputs
│   ├── maiuvf_bunny/                # MA-IUVF experiment outputs
│   └── ...
│
├── docs/                             # (legacy docs)
├── cache/                            # Cached data and features
├── logs/                             # Training logs
│
├── README.md                         # Project overview (updated)
├── demo.py                           # Demo script
├── quickstart.py                     # Quick start script
└── requirements.txt                  # Python dependencies
```

## Key Files Reference

### Core Implementation

| File | Purpose | Status |
|------|---------|--------|
| `src/models/metric_aligned_iuv_field.py` | MA-IUVF network architecture | ✅ Complete |
| `src/training/metric_aligned_iuv_losses.py` | Loss functions | ✅ Complete |
| `src/inference/metric_aligned_iuv_inference.py` | Inference pipeline | ✅ Complete |
| `src/inference/offline_renderer.py` | CPU rasterizer | ✅ Complete |

### Data Pipeline

| File | Purpose | Status |
|------|---------|--------|
| `src/data/obj_parser.py` | OBJ parser with face-corner UVs | ⚠️ Referenced |
| `src/data/uv_chart_segmentation.py` | UV chart segmentation | ⚠️ Referenced |
| `src/data/metric_aligned_iuv_baker.py` | Training data baker | ⚠️ Referenced |

### Scripts

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/train_metric_aligned_iuv_field.py` | MA-IUVF training | ~1000 |
| `scripts/run_maiuvf_experiment.py` | End-to-end experiment | ~1000 |
| `scripts/infer_metric_aligned_iuv.py` | MA-IUVF inference | ~600 |

### Tests

| File | Purpose | Tests |
|------|---------|-------|
| `tests/test_metric_aligned_iuv_baker.py` | OBJ parsing & baking | 8 |
| `tests/test_metric_aligned_iuv_training.py` | Training pipeline | 12 |
| `tests/test_metric_aligned_iuv_inference.py` | Inference pipeline | 10 |
| `tests/test_uv_charts.py` | Chart segmentation | 6 |

## Module Dependencies

```
configs/
    ↓
scripts/
    ↓ src/
    ├── models/ (metric_aligned_iuv_field.py)
    ├── data/ (obj_parser, uv_chart_segmentation, baker)
    ├── training/ (losses)
    └── inference/ (inference, renderer)
```

## Data Flow

```
High-Poly Mesh (OBJ)
    ↓
OBJ Parser (face-corner UVs)
    ↓
UV Chart Segmentation
    ↓
Metric-Aligned Baker (Jacobian computation)
    ↓
Training Data (positions, UVs, chart IDs)
    ↓
MA-IUVF Training (metric alignment loss)
    ↓
Trained Model (checkpoint)
    ↓
Inference (UV prediction on low-poly)
    ↓
Rendering (texture mapping)
```

## Configuration System

### YAML Configuration

```yaml
# configs/maiuvf_baseline.yaml
data:
  high_mesh_path: data/models/stanford_bunny_procedural.obj
  texture_path: data/textures/bunny_texture.png
  num_samples: 100000
  chart_mode: uv_islands

model:
  num_charts: 8
  hidden_dim: 128
  num_layers: 3
  encoder_type: fourier

training:
  epochs: 100
  batch_size: 4096
  learning_rate: 0.0001
```

### CLI Override

```bash
python scripts/run_maiuvf_experiment.py \
    --config configs/maiuvf_baseline.yaml \
    --training.epochs 200 \
    --model.hidden_dim 256
```

## File Statistics

| Category | Count |
|----------|-------|
| Python files | 88 |
| Test files | 15 |
| Config files | 5 |
| Documentation files | 8 |
| Scripts | 63 |

## Key Implementations

### MA-IUVF Network (292 lines)

```python
# src/models/metric_aligned_iuv_field.py

class MetricAlignedIUVField(nn.Module):
    def __init__(self, num_charts=8, hidden_dim=128, ...):
        self.encoder = FourierPositionalEncoding(...)
        self.mlp = nn.Sequential(...)
        self.chart_head = nn.Linear(hidden_dim, num_charts)
        self.uv_head = nn.Linear(hidden_dim, num_charts * 2)

    def forward(self, positions):
        encoded = self.encoder(positions)
        features = self.mlp(encoded)
        chart_logits = self.chart_head(features)
        uv_preds = self.uv_head(features).view(-1, num_charts, 2)
        return MetricAlignedIUVOutput(chart_logits, uv_preds)
```

### Metric Alignment Loss (572 lines)

```python
# src/training/metric_aligned_iuv_losses.py

def compute_metric_aligned_iuv_loss(
    uv_preds, logits, positions, uv_gt, chart_ids, num_charts
):
    # 1. Gather UV predictions from selected chart
    selected_uv = gather_chart_uvs(uv_preds, chart_ids)

    # 2. Compute UV Jacobian
    jacobian = compute_uv_jacobian(selected_uv, positions)

    # 3. Compute losses
    metric_loss = compute_metric_alignment_loss(jacobian, j_gt)
    anchor_loss = compute_anchor_loss(selected_uv, uv_gt)
    chart_loss = compute_classification_loss(logits, chart_ids)

    return {
        'metric_alignment': metric_loss,
        'uv_anchor': anchor_loss,
        'chart_classification': chart_loss,
        'total': metric_loss + anchor_loss + chart_loss
    }
```

### CPU Rasterizer (562 lines)

```python
# src/inference/offline_renderer.py

class OfflineRenderer:
    def render_with_maiuvf(self, model, maiuvf_inference):
        # 1. Rasterize mesh to get pixel coordinates
        pixel_coords = self.rasterize()

        # 2. Collect valid pixels
        valid_pixels = pixel_coords[self.valid_mask]

        # 3. MA-IUVF prediction
        output = maiuvf_inference.predict(valid_pixels)

        # 4. Chart selection
        chart_ids = output.logits.argmax(axis=-1)
        selected_uvs = output.uv_preds[arange(N), chart_ids]

        # 5. Texture sampling
        colors = self._sample_texture(selected_uvs)

        return colors
```

## Running the Project

### Quick Start

```bash
# Run quick start demo
python quickstart.py --output-dir outputs/quickstart

# Run full demo
python demo.py --mode all

# Run specific demo
python demo.py --mode training
```

### Training

```bash
# Train MA-IUVF model
python scripts/train_metric_aligned_iuv_field.py \
    --high-mesh data/models/stanford_bunny_procedural.obj \
    --texture data/textures/bunny_texture.png \
    --output-dir outputs/maiuvf_bunny/train \
    --num-charts 8 \
    --epochs 100
```

### Inference

```bash
# Run inference on low-poly mesh
python scripts/infer_metric_aligned_iuv.py \
    --checkpoint outputs/maiuvf_bunny/best.pt \
    --input-mesh data/models/low_poly.obj \
    --output-dir outputs/maiuvf_bunny/inference
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_metric_aligned_iuv_training.py -v
```

## Architecture Phases

### Phase 1: MA-IUVF Core
- Multi-chart implicit UV field
- Fourier positional encoding
- Metric alignment loss
- CPU rasterizer

### Phase 2: Performance Optimization
- B-Spline hash grid encoder
- Multi-epoch training
- Chart classification improvement

### Phase 3: Full Integration
- Texture generation network
- Multi-chart optimization
- End-to-end pipeline

## Documentation

| File | Description |
|------|-------------|
| `README.md` | Project overview and quick start |
| `CLAUDE.md` | Comprehensive project documentation |
| `docs/PROJECT_REPORT.md` | Academic project report |
| `docs/PROJECT_STRUCTURE.md` | This file |
| `START.md` | Detailed technical design |

## Contributing

When adding new features:

1. **Model Changes**: Update `src/models/metric_aligned_iuv_field.py`
2. **Loss Changes**: Update `src/training/metric_aligned_iuv_losses.py`
3. **Data Changes**: Update `src/data/metric_aligned_iuv_baker.py`
4. **Training**: Use `scripts/train_metric_aligned_iuv_field.py`
5. **Validation**: Run unit tests in `tests/`

## Debugging

1. **OBJ Parsing**: Check `src/data/obj_parser.py` validation assertions
2. **Chart Segmentation**: Verify UV continuity tolerance
3. **Training**: Check loss weights and learning rates
4. **Rendering**: Use CPU rasterizer as fallback
