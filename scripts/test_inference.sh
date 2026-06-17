#!/bin/bash

# Test script for inference system
# This script verifies that all components are working correctly

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   Inference System Test${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Test counter
passed=0
failed=0

# Test function
test_component() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}Testing: $name${NC}"

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED: $name${NC}"
        ((passed++))
        return 0
    else
        echo -e "${RED}✗ FAILED: $name${NC}"
        ((failed++))
        return 1
    fi
}

# ==============================================================================
# Test 1: Python Dependencies
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 1: Python Dependencies${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

test_component "Python 3" "python3 --version"
test_component "trimesh" "python3 -c 'import trimesh'"
test_component "torch" "python3 -c 'import torch'"
test_component "numpy" "python3 -c 'import numpy'"

echo ""

# ==============================================================================
# Test 2: Source Code Import
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 2: Source Code Import${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

test_component "Inference module" "python3 -c 'import sys; sys.path.append(\"src\"); from inference import MeshSimplifier'"
test_component "Simplification class" "python3 -c 'import sys; sys.path.append(\"src\"); from inference.mesh_simplification import MeshSimplifier'"

echo ""

# ==============================================================================
# Test 3: Data Files
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 3: Data Files${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

test_component "High-poly mesh exists" "[ -f 'data/models/stanford-bunny.obj' ]"
test_component "Checkpoint exists" "[ -f 'logs/bunny_gpu_test/checkpoints/final.pt' ]"

echo ""

# ==============================================================================
# Test 4: Mesh Simplification
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 4: Mesh Simplification${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Test mesh simplification with a small ratio
echo -e "${YELLOW}Testing mesh simplification (1% ratio)...${NC}"

python3 << EOF
import sys
sys.path.append('src')
from inference.mesh_simplification import MeshSimplifier

try:
    simplifier = MeshSimplifier('data/models/stanford-bunny.obj')
    low_mesh = simplifier.simplify_by_ratio(0.01)

    # Check result
    assert low_mesh is not None, "Simplification returned None"
    assert len(low_mesh.vertices) > 0, "No vertices in simplified mesh"
    assert len(low_mesh.faces) > 0, "No faces in simplified mesh"

    # Check compression
    face_ratio = len(low_mesh.faces) / len(simplifier.high_mesh.faces)
    assert face_ratio < 0.02, f"Face ratio too high: {face_ratio}"

    print(f"✓ Simplification successful")
    print(f"  Vertices: {len(simplifier.high_mesh.vertices)} -> {len(low_mesh.vertices)}")
    print(f"  Faces: {len(simplifier.high_mesh.faces)} -> {len(low_mesh.faces)}")
    print(f"  Compression: {(1-face_ratio)*100:.1f}%")

except Exception as e:
    print(f"✗ Simplification failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: Mesh simplification${NC}"
    ((passed++))
else
    echo -e "${RED}✗ FAILED: Mesh simplification${NC}"
    ((failed++))
fi

echo ""

# ==============================================================================
# Test 5: Inference Engine
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 5: Inference Engine${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Create a test low-poly mesh first
echo -e "${YELLOW}Creating test low-poly mesh...${NC}"

python3 << EOF
import sys
sys.path.append('src')
from inference.mesh_simplification import create_low_poly_mesh

create_low_poly_mesh(
    'data/models/stanford-bunny.obj',
    '/tmp/test_low_bunny.obj',
    face_ratio=0.05
)
print("✓ Test mesh created")
EOF

echo ""
echo -e "${YELLOW}Testing inference engine initialization...${NC}"

python3 << EOF
import sys
sys.path.append('src')
sys.path.append('scripts')
from inference import InferenceEngine

try:
    engine = InferenceEngine(
        checkpoint_path='logs/bunny_gpu_test/checkpoints/final.pt',
        device='cpu'  # Use CPU for testing
    )

    assert engine.model_g is not None, "Network G not loaded"
    assert engine.device is not None, "Device not set"

    print("✓ Inference engine initialized")
    print(f"  Device: {engine.device}")
    print(f"  Network G loaded: ✓")
    print(f"  Network D loaded: {'✓' if engine.model_d else 'N/A'}")

except Exception as e:
    print(f"✗ Inference engine initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED: Inference engine initialization${NC}"
    ((passed++))
else
    echo -e "${RED}✗ FAILED: Inference engine initialization${NC}"
    ((failed++))
fi

echo ""

# ==============================================================================
# Test 6: Output Files
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test 6: Output Files${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Test output directory creation
mkdir -p outputs/test_inference

test_component "Output directory creation" "[ -d 'outputs/test_inference' ]"
test_component "Viewer HTML exists" "[ -f 'viewer/index.html' ]"

echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

total=$((passed + failed))
echo -e "Total tests: $total"
echo -e "${GREEN}Passed: $passed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}Failed: $failed${NC}"
    echo ""
    echo -e "${RED}Some tests failed. Please check the error messages above.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo -e "${GREEN}The inference system is ready to use!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Run the full pipeline:"
    echo -e "     ${GREEN}./scripts/run_inference_pipeline.sh${NC}"
    echo ""
    echo "  2. Or run individual tests:"
    echo "     - Test mesh simplification"
    echo "     - Test model inference"
    echo "     - Test web viewer"
fi

echo ""
echo -e "${BLUE}==================================================${NC}"
