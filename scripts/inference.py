"""
High-Performance Inference Script for Implicit Texture Field

This script performs inference on low-poly meshes using trained Network G and D.
It processes vertices in batches for optimal GPU utilization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from pathlib import Path
import argparse
import yaml
import time
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import our models
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import NetworkG, NetworkD, create_network_g_from_config, create_network_d_from_config
from src.interfaces import ConditionVector
from src.data.sampling import GeometryFeatureExtractor
from src.inference.mesh_simplification import MeshSimplifier

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine for implicit texture field.

    Process vertices in batches with GPU acceleration.

    Pipeline: 减面 → 推理 → 保存
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        target_faces: Optional[int] = None,
        face_ratio: Optional[float] = None,
        simplify: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Optional path to training config
            device: Device to run inference on
            target_faces: Target face count after simplification
            face_ratio: Face ratio to keep (0.0-1.0)
            simplify: Whether to perform mesh simplification
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.simplify = simplify
        self.target_faces = target_faces
        self.face_ratio = face_ratio

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config if provided
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default config from checkpoint
            self.config = checkpoint.get('config', self._get_default_config())

        # Initialize networks
        self.model_g = self._load_network_g(checkpoint)
        self.model_d = self._load_network_d(checkpoint)

        logger.info(f"Inference engine initialized on {self.device}")

    def _load_network_g(self, checkpoint: Dict) -> NetworkG:
        """Load Network G from checkpoint."""
        # Get network_g config
        if hasattr(self.config, 'network_g'):
            network_g_config = self.config.network_g
        else:
            network_g_config = self.config.get('network_g', self._get_default_config()['network_g'])

        model_g = create_network_g_from_config(network_g_config)

        # Load state dict - checkpoint uses 'network_g_state' not 'network_g'
        state_key = 'network_g_state' if 'network_g_state' in checkpoint else 'network_g'
        if state_key in checkpoint and checkpoint[state_key] is not None:
            model_g.load_state_dict(checkpoint[state_key])
            logger.info(f"Loaded Network G from {state_key}")
        else:
            logger.warning(f"Network G state not found in checkpoint")

        model_g.to(self.device)
        model_g.eval()
        return model_g

    def _load_network_d(self, checkpoint: Dict) -> Optional[NetworkD]:
        """Load Network D from checkpoint."""
        # Check for network_d state
        state_key = 'network_d_state' if 'network_d_state' in checkpoint else 'network_d'

        if state_key not in checkpoint or checkpoint[state_key] is None:
            logger.warning("Network D not found in checkpoint, using G-only mode")
            return None

        # Get network_d config
        if hasattr(self.config, 'network_d'):
            network_d_config = self.config.network_d
        else:
            network_d_config = self.config.get('network_d', self._get_default_config()['network_d'])

        model_d = create_network_d_from_config(network_d_config)
        model_d.load_state_dict(checkpoint[state_key])
        model_d.to(self.device)
        model_d.eval()
        logger.info(f"Loaded Network D from {state_key}")
        return model_d

    def _get_default_config(self) -> Dict:
        """Get default configuration if not in checkpoint."""
        return {
            'network_g': {
                'hidden_dim': 256,
                'num_layers': 8,
                'positional_encoding_freqs': 6,
                'skip_connection_layer': 4,
                'include_raw_input': True,
                'sdf_output_range': 1.0,
            },
            'network_d': {
                'condition_dim': 42,
                'hidden_channels': 128,
                'num_res_blocks': 4,
                'num_diffusion_steps': 1000,
            }
        }

    @torch.no_grad()
    def infer_vertex_colors(
        self,
        vertices: np.ndarray,
        mesh_data: Optional[trimesh.Trimesh] = None,
        batch_size: int = 8192,
        return_intermediates: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Infer colors for mesh vertices.

        Args:
            vertices: (N, 3) vertex positions
            mesh_data: Optional mesh for feature extraction
            batch_size: Batch size for inference
            return_intermediates: Whether to return intermediate results

        Returns:
            (N, 3) predicted colors and dictionary of intermediates
        """
        # Normalize vertices to [-1, 1]
        vertices_normalized = self._normalize_vertices(vertices)

        # Convert to tensor
        vertices_tensor = torch.from_numpy(vertices_normalized).float().to(self.device)

        # Process in batches
        all_colors = []
        all_sdf = []
        all_base_colors = []
        all_diffusion_colors = []

        num_batches = (len(vertices_tensor) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(vertices_tensor))

            batch_vertices = vertices_tensor[start_idx:end_idx]

            # Network G forward pass
            g_output = self.model_g(batch_vertices)
            sdf = g_output.sdf
            base_color = g_output.color_base

            if self.model_d is not None:
                # Match the current training-time condition construction:
                # [G base color (3), G sdf (1), zero padding for the remaining
                # geometry/global features (38)]. Full geometry features can be
                # wired here once D is trained with them.
                batch_count = len(batch_vertices)
                condition = ConditionVector(
                    color_base=base_color,
                    sdf=sdf.view(-1, 1),
                    curvature=torch.zeros(batch_count, 2, device=self.device),
                    normal=torch.zeros(batch_count, 3, device=self.device),
                    boundary_distance=torch.zeros(batch_count, 1, device=self.device),
                    global_shape_code=torch.zeros(batch_count, 32, device=self.device),
                )

                num_steps = self._get_diffusion_inference_steps()
                final_color = self.model_d.sample(
                    condition,
                    num_steps=num_steps,
                    deterministic=True,
                )
                all_diffusion_colors.append(final_color.cpu().numpy())
            else:
                final_color = base_color

            all_colors.append(final_color.cpu().numpy())
            all_sdf.append(sdf.cpu().numpy())
            all_base_colors.append(base_color.cpu().numpy())

        # Concatenate results
        colors = np.concatenate(all_colors, axis=0)
        sdf_values = np.concatenate(all_sdf, axis=0)
        base_colors = np.concatenate(all_base_colors, axis=0)

        intermediates = {
            'sdf': sdf_values,
            'base_color': base_colors,
            'used_network_d': self.model_d is not None,
        }
        if all_diffusion_colors:
            intermediates['diffusion_color'] = np.concatenate(all_diffusion_colors, axis=0)

        if return_intermediates:
            return colors, intermediates
        else:
            return colors, None

    def _get_diffusion_inference_steps(self) -> int:
        """Return configured D inference steps with a conservative fallback."""
        if isinstance(self.config, dict):
            network_d_config = self.config.get('network_d', {})
            return int(network_d_config.get('inference_steps', 20))

        network_d_config = getattr(self.config, 'network_d', None)
        if network_d_config is not None:
            return int(getattr(network_d_config, 'inference_steps', 20))

        return 20

    def _normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Normalize vertices to [-1, 1] range."""
        # Compute bounding box
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)

        # Normalize
        normalized = 2 * (vertices - vmin) / (vmax - vmin + 1e-8) - 1
        return normalized.astype(np.float32)

    def process_mesh(
        self,
        input_mesh_path: str,
        output_mesh_path: str,
        batch_size: int = 8192
    ) -> trimesh.Trimesh:
        """
        Process entire mesh and save colored result.

        Pipeline: 减面 → 推理 → 保存

        Args:
            input_mesh_path: Path to input mesh
            output_mesh_path: Path to save colored mesh
            batch_size: Batch size for inference

        Returns:
            Colored low-poly mesh
        """
        logger.info(f"Loading mesh from {input_mesh_path}")
        print(f"Loading mesh from {input_mesh_path}")
        original_mesh = trimesh.load(input_mesh_path)

        original_vertices = len(original_mesh.vertices)
        original_faces = len(original_mesh.faces)
        logger.info(f"原始mesh: {original_vertices} 顶点, {original_faces} 面")
        print(f"原始mesh: {original_vertices} 顶点, {original_faces} 面")

        # 步骤1: 减面
        if self.simplify:
            logger.info("步骤1: 减面...")
            print("步骤1: 减面...")

            low_mesh = self._simplify_mesh(original_mesh)

            simplified_vertices = len(low_mesh.vertices)
            simplified_faces = len(low_mesh.faces)
            reduction_ratio = (1 - simplified_faces / original_faces) * 100

            logger.info(f"减面后: {simplified_vertices} 顶点, {simplified_faces} 面 (减少 {reduction_ratio:.1f}%)")
            print(f"  减面后: {simplified_vertices} 顶点, {simplified_faces} 面 (减少 {reduction_ratio:.1f}%)")

            mesh = low_mesh
        else:
            logger.info("跳过减面，使用原始mesh")
            mesh = original_mesh

        # 步骤2: 推理
        logger.info(f"步骤2: 推理颜色...")
        print(f"步骤2: 推理 {len(mesh.vertices)} 个顶点的颜色...")
        if self.model_d is not None:
            logger.info(f"使用 Network D 扩散采样 ({self._get_diffusion_inference_steps()} steps)")
            print(f"  使用 Network D 扩散采样 ({self._get_diffusion_inference_steps()} steps)")
        else:
            logger.info("Network D 不可用，回退到 Network G base color")
            print("  Network D 不可用，回退到 Network G base color")
        start_time = time.time()

        colors, intermediates = self.infer_vertex_colors(
            mesh.vertices,
            mesh,
            batch_size=batch_size,
            return_intermediates=True
        )

        inference_time = time.time() - start_time

        # 步骤3: 保存
        logger.info(f"步骤3: 保存带颜色的低面数模型...")
        print(f"步骤3: 保存...")

        # Clip colors to valid range
        colors = np.clip(colors, 0.0, 1.0)

        # Convert to 0-255 range for OBJ format
        colors_uint8 = (colors * 255).astype(np.uint8)

        # Apply colors to mesh
        mesh.visual.vertex_colors = colors_uint8

        # Save result
        output_path = Path(output_mesh_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

        logger.info(f"已保存到: {output_path}")
        logger.info(f"推理时间: {inference_time:.2f}s ({len(mesh.vertices)/inference_time:.0f} 顶点/秒)")
        print(f"\n✓ 完成!")
        print(f"  原始: {original_vertices} 顶点, {original_faces} 面")
        if self.simplify:
            print(f"  低面: {simplified_vertices} 顶点, {simplified_faces} 面")
        print(f"  推理: {inference_time:.2f}s")
        print(f"  输出: {output_path}")

        return mesh

    def _simplify_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Simplify mesh to target face count or ratio.

        Args:
            mesh: Input mesh

        Returns:
            Simplified mesh
        """
        from src.inference.mesh_simplification import MeshSimplifier

        # 创建临时文件路径
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_mesh_path = temp_dir / "temp_mesh.obj"
        mesh.export(str(temp_mesh_path))

        # 创建simplifier
        simplifier = MeshSimplifier(str(temp_mesh_path))

        # 确定目标面数
        if self.target_faces:
            target = self.target_faces
            logger.info(f"减面到目标面数: {target}")
        elif self.face_ratio:
            target = int(len(mesh.faces) * self.face_ratio)
            logger.info(f"减面比例: {self.face_ratio:.1%} → {target} 面")
        else:
            # 默认减面到5%
            target = int(len(mesh.faces) * 0.05)
            logger.info(f"使用默认减面比例 5% → {target} 面")

        # 执行减面
        low_mesh = simplifier.simplify_by_count(target, method="quadric", aggression=10)

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return low_mesh

    def compare_with_highpoly(
        self,
        high_mesh_path: str,
        low_mesh_colored_path: str,
        num_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Compare colored low-poly with ground truth high-poly.

        Args:
            high_mesh_path: Path to original high-poly mesh
            low_mesh_colored_path: Path to colored low-poly mesh
            num_samples: Number of samples for comparison

        Returns:
            Dictionary of metrics
        """
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        except ImportError:
            logger.warning("scikit-image not available, skipping metrics")
            return {}

        # Load meshes
        high_mesh = trimesh.load(high_mesh_path)
        low_mesh = trimesh.load(low_mesh_colored_path)

        # Sample points on both meshes (simplified comparison)
        high_sample = high_mesh.vertices
        low_sample = low_mesh.vertices

        # Get colors
        high_colors = high_mesh.visual.vertex_colors if hasattr(high_mesh.visual, 'vertex_colors') else None
        low_colors = low_mesh.visual.vertex_colors

        if high_colors is None:
            logger.warning("High-poly mesh has no vertex colors, skipping comparison")
            return {}

        # Compute metrics (simplified - just compare color distributions)
        metrics = {}

        # PSNR (using colors directly)
        try:
            # Resize to match if needed
            min_len = min(len(high_colors), len(low_colors))
            high_colors_subset = high_colors[:min_len]
            low_colors_subset = low_colors[:min_len]

            psnr = peak_signal_noise_ratio(high_colors_subset, low_colors_subset)
            metrics['psnr'] = psnr

            ssim = structural_similarity(high_colors_subset, low_colors_subset, multichannel=True)
            metrics['ssim'] = ssim
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Inference for implicit texture field - 减面 → 推理 → 保存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 减面到5%面数（默认）
  python scripts/inference.py data/models/bunny.obj --checkpoint checkpoints/model.pt

  # 减面到1000面
  python scripts/inference.py data/models/bunny.obj --checkpoint checkpoints/model.pt --target-faces 1000

  # 减面到10%面数
  python scripts/inference.py data/models/bunny.obj --checkpoint checkpoints/model.pt --face-ratio 0.1

  # 不减面（对原始mesh推理）
  python scripts/inference.py data/models/bunny.obj --checkpoint checkpoints/model.pt --no-simplify
        """
    )
    parser.add_argument("input_mesh", help="Path to input mesh (high-poly)")
    parser.add_argument("output_mesh", nargs='?', help="Path to save colored mesh (optional, auto-detects from checkpoint)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", help="Path to training config")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for inference")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--compare", help="Path to high-poly mesh for comparison")

    # 减面参数
    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument("--target-faces", type=int, help="Target face count after simplification")
    simplify_group.add_argument("--face-ratio", type=float, help="Face ratio to keep (0.0-1.0, e.g. 0.1 = 10%%)")
    simplify_group.add_argument("--no-simplify", dest="simplify", action="store_false", help="Skip mesh simplification")
    parser.set_defaults(simplify=True)

    args = parser.parse_args()

    # 自动推断输出路径：如果未指定output_mesh，则使用checkpoint所在实验的inference文件夹
    if args.output_mesh is None:
        checkpoint_path = Path(args.checkpoint)
        # checkpoint路径格式：outputs/experiments/EXP_ID/checkpoints/checkpoint.pt
        # 推断实验文件夹
        parts = checkpoint_path.parts
        if 'experiments' in parts:
            experiments_idx = parts.index('experiments')
            # 获取实验ID
            if experiments_idx + 1 < len(parts):
                experiment_id = parts[experiments_idx + 1]
                experiment_dir = Path(*parts[:experiments_idx + 2])  # outputs/experiments/EXP_ID
                inference_dir = experiment_dir / "inference"
                inference_dir.mkdir(parents=True, exist_ok=True)

                # 生成输出文件名
                checkpoint_name = checkpoint_path.stem

                # 添加减面信息到文件名
                if args.simplify:
                    if args.target_faces:
                        suffix = f"_lowpoly_{args.target_faces}faces"
                    elif args.face_ratio:
                        suffix = f"_lowpoly_{int(args.face_ratio*100)}percent"
                    else:
                        suffix = "_lowpoly_default"
                else:
                    suffix = "_original"

                output_filename = f"colored_{checkpoint_name}{suffix}.obj"
                args.output_mesh = str(inference_dir / output_filename)

                logger.info(f"自动推断输出路径: {args.output_mesh}")
            else:
                raise ValueError("无法从checkpoint路径推断实验文件夹")
        else:
            raise ValueError("Checkpoint不在experiments文件夹中，无法自动推断输出路径")

    # Initialize inference engine
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        target_faces=args.target_faces,
        face_ratio=args.face_ratio,
        simplify=args.simplify
    )

    # Process mesh
    colored_mesh = engine.process_mesh(
        input_mesh_path=args.input_mesh,
        output_mesh_path=args.output_mesh,
        batch_size=args.batch_size
    )

    # Compare if requested
    if args.compare:
        logger.info("Comparing with high-poly mesh...")
        metrics = engine.compare_with_highpoly(args.compare, args.output_mesh)
        logger.info(f"Metrics: {metrics}")

    logger.info(f"Inference complete! 结果保存到: {args.output_mesh}")


if __name__ == "__main__":
    main()
