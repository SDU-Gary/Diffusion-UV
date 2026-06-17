#!/bin/bash

# Complete Inference Pipeline for Diffusion-UV
# This script runs the full inference process: mesh simplification, model inference, and visualization

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   Diffusion-UV Inference Pipeline${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Default paths
HIGH_MESH="data/models/stanford-bunny.obj"
CHECKPOINT="logs/bunny_production/checkpoints/phase3_epoch_100.pt"
OUTPUT_DIR="outputs/inference_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --high-mesh)
            HIGH_MESH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --face-ratio)
            FACE_RATIO="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
FACE_RATIO=${FACE_RATIO:-0.05}
DEVICE=${DEVICE:-cuda}

echo -e "${YELLOW}Configuration:${NC}"
echo "  High-poly mesh: $HIGH_MESH"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Face ratio: $FACE_RATIO"
echo "  Device: $DEVICE"
echo ""

# Check files exist
if [ ! -f "$HIGH_MESH" ]; then
    echo -e "${RED}Error: High-poly mesh not found: $HIGH_MESH${NC}"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
    echo -e "${YELLOW}Hint: Run training first to generate checkpoint${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ==============================================================================
# Step 1: Mesh Simplification
# ==============================================================================
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Step 1: Mesh Simplification${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

LOW_MESH="$OUTPUT_DIR/low_poly_bunny.obj"

python3 << EOF
import sys
sys.path.append('src')
from inference.mesh_simplification import MeshSimplifier
import logging

logging.basicConfig(level=logging.INFO)

print(f"Loading high-poly mesh: $HIGH_MESH")
simplifier = MeshSimplifier("$HIGH_MESH")

print(f"Simplifying to {int($FACE_RATIO * 100)}% faces...")
low_mesh = simplifier.simplify_by_ratio($FACE_RATIO, method="quadric")

print(f"Saving to: $LOW_MESH")
low_mesh.export("$LOW_MESH")

# Print stats
stats = simplifier.compare_meshes(low_mesh)
print(f"\nSimplification Statistics:")
print(f"  Vertices: {stats['high_vertices']} -> {stats['low_vertices']} ({stats['vertex_ratio']:.1%})")
print(f"  Faces: {stats['high_faces']} -> {stats['low_faces']} ({stats['face_ratio']:.1%})")
print(f"  Compression: {stats['compression']:.1f}%")
EOF

echo ""
echo -e "${GREEN}✓ Mesh simplification complete${NC}"
echo ""

# ==============================================================================
# Step 2: Model Inference
# ==============================================================================
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Step 2: Model Inference${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

COLORED_MESH="$OUTPUT_DIR/colored_bunny.obj"

python3 scripts/inference.py \
    "$LOW_MESH" \
    "$COLORED_MESH" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --batch-size 8192

echo ""
echo -e "${GREEN}✓ Model inference complete${NC}"
echo ""

# ==============================================================================
# Step 3: Comparison (if high mesh has colors)
# ==============================================================================
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Step 3: Quality Assessment${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Check if high mesh has vertex colors
HIGH_MESH_HAS_COLORS=$(python3 << EOF
import trimesh
try:
    mesh = trimesh.load("$HIGH_MESH")
    has_colors = hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
    print(1 if has_colors else 0)
except:
    print(0)
EOF
)

if [ "$HIGH_MESH_HAS_COLORS" = "1" ]; then
    echo "High-poly mesh has vertex colors, computing comparison metrics..."

    python3 scripts/inference.py \
        "$LOW_MESH" \
        "$COLORED_MESH" \
        --checkpoint "$CHECKPOINT" \
        --compare "$HIGH_MESH" \
        --device "$DEVICE"
else
    echo "High-poly mesh has no vertex colors, skipping comparison"
    echo "(This is expected for procedurally textured meshes)"
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Inference Pipeline Complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

echo -e "${YELLOW}Generated Files:${NC}"
echo "  📊 Low-poly mesh: $LOW_MESH"
echo "  🎨 Colored mesh: $COLORED_MESH"
echo ""

# Get file sizes
LOW_SIZE=$(du -h "$LOW_MESH" | cut -f1)
COLORED_SIZE=$(du -h "$COLORED_MESH" | cut -f1)

echo -e "${YELLOW}File Sizes:${NC}"
echo "  Low-poly: $LOW_SIZE"
echo "  Colored: $COLORED_SIZE"
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. View in desktop 3D viewer:"
echo -e "     ${GREEN}python3 scripts/viewer_3d.py '$COLORED_MESH'${NC}"
echo ""
echo "  2. Alternative: Open in 3D modeling software:"
echo -e "     ${GREEN}blender '$COLORED_MESH'${NC}"
echo ""
echo "  3. Viewer controls:"
echo "     - Left click + drag: Rotate"
echo "     - Shift + Left click: Pan"
echo "     - Scroll: Zoom"
echo "     - 'w': Toggle wireframe"
echo "     - 'r': Reset camera"
echo "     - 'q': Quit"
echo ""

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}   Inference pipeline completed successfully! 🎉${NC}"
echo -e "${BLUE}==================================================${NC}"
