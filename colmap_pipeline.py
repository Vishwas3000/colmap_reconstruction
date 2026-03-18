"""
COLMAP Point Cloud Extraction Pipeline

Automates the full COLMAP Structure-from-Motion pipeline:
  Feature Extraction -> Feature Matching -> Sparse Reconstruction -> (Optional) Dense Reconstruction

Usage:
  python colmap_pipeline.py --image_dir images --workspace workspace
  python colmap_pipeline.py --image_dir images --workspace workspace --dense
  python colmap_pipeline.py --image_dir images --workspace workspace --visualize
  python colmap_pipeline.py --image_dir images --workspace workspace --backend cli
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# ─── Utility ──────────────────────────────────────────────────────

def check_colmap_installed():
    """Check if COLMAP CLI is available on PATH."""
    if shutil.which("colmap") is None:
        log.error(
            "COLMAP not found on PATH.\n"
            "  Download from: https://github.com/colmap/colmap/releases\n"
            "  Extract the zip and add the folder containing colmap.exe to your PATH."
        )
        sys.exit(1)


def run_colmap_cmd(args):
    """Run a COLMAP CLI command and stream output."""
    cmd = ["colmap"] + args
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        log.debug(result.stdout)
    if result.returncode != 0:
        log.error("COLMAP failed:\n%s", result.stderr)
        raise RuntimeError(f"COLMAP command failed: {args[0]}")
    return result.stdout


def count_images(image_dir):
    """Count supported image files."""
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    count = sum(1 for f in Path(image_dir).iterdir() if f.suffix.lower() in extensions)
    return count


def check_gpu_available():
    """Check if CUDA GPU is likely available (Windows/NVIDIA)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─── Backend: pycolmap ────────────────────────────────────────────

def run_with_pycolmap(image_dir, workspace, dense=False, use_gpu=True):
    """Run pipeline using pycolmap Python bindings."""
    try:
        import pycolmap
    except ImportError:
        log.error("pycolmap not installed. Run: pip install pycolmap")
        sys.exit(1)

    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Feature extraction
    log.info("Step 1/3: Extracting features...")
    pycolmap.extract_features(database_path, image_dir)

    # Step 2: Exhaustive matching
    log.info("Step 2/3: Matching features (exhaustive)...")
    pycolmap.match_exhaustive(database_path)

    # Step 3: Incremental mapping (sparse reconstruction)
    log.info("Step 3/3: Running sparse reconstruction...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not maps:
        log.error("Reconstruction failed — no models produced. Check image overlap.")
        sys.exit(1)

    maps[0].write(sparse_dir)
    log.info("Sparse reconstruction complete: %s", maps[0].summary())

    # Export sparse PLY
    sparse_ply = sparse_dir / "sparse.ply"
    run_colmap_cmd([
        "model_converter",
        "--input_path", str(sparse_dir / "0"),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ])
    log.info("Sparse point cloud saved to: %s", sparse_ply)

    # Dense reconstruction
    mvs_dir = None
    if dense:
        mvs_dir = workspace / "mvs"
        log.info("Step 4: Undistorting images...")
        pycolmap.undistort_images(mvs_dir, sparse_dir, image_dir)

        log.info("Step 5: PatchMatch stereo (requires CUDA)...")
        pycolmap.patch_match_stereo(mvs_dir)

        dense_ply = mvs_dir / "fused.ply"
        log.info("Step 6: Fusing into dense point cloud...")
        pycolmap.stereo_fusion(dense_ply, mvs_dir)
        log.info("Dense point cloud saved to: %s", dense_ply)

    return sparse_dir, mvs_dir


# ─── Backend: CLI subprocess ─────────────────────────────────────

def run_with_cli(image_dir, workspace, dense=False, use_gpu=True):
    """Run pipeline using COLMAP CLI commands via subprocess."""
    check_colmap_installed()

    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Feature extraction
    log.info("Step 1/3: Extracting features...")
    run_colmap_cmd([
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
    ])

    # Step 2: Exhaustive matching
    log.info("Step 2/3: Matching features (exhaustive)...")
    run_colmap_cmd([
        "exhaustive_matcher",
        "--database_path", str(database_path),
    ])

    # Step 3: Sparse reconstruction
    log.info("Step 3/3: Running sparse reconstruction...")
    run_colmap_cmd([
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ])

    # Check reconstruction output
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        log.error("Reconstruction failed — no model produced. Check image overlap.")
        sys.exit(1)

    # Export sparse PLY
    sparse_ply = sparse_dir / "sparse.ply"
    run_colmap_cmd([
        "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ])
    log.info("Sparse point cloud saved to: %s", sparse_ply)

    # Dense reconstruction
    mvs_dir = None
    if dense:
        if not use_gpu:
            log.error("Dense reconstruction requires CUDA GPU. Skipping.")
            return sparse_dir, None

        mvs_dir = workspace / "mvs"
        mvs_dir.mkdir(parents=True, exist_ok=True)

        log.info("Step 4: Undistorting images...")
        run_colmap_cmd([
            "image_undistorter",
            "--image_path", str(image_dir),
            "--input_path", str(model_dir),
            "--output_path", str(mvs_dir),
            "--output_type", "COLMAP",
        ])

        log.info("Step 5: PatchMatch stereo (requires CUDA)...")
        run_colmap_cmd([
            "patch_match_stereo",
            "--workspace_path", str(mvs_dir),
            "--PatchMatchStereo.geom_consistency", "true",
        ])

        dense_ply = mvs_dir / "fused.ply"
        log.info("Step 6: Fusing into dense point cloud...")
        run_colmap_cmd([
            "stereo_fusion",
            "--workspace_path", str(mvs_dir),
            "--output_path", str(dense_ply),
        ])
        log.info("Dense point cloud saved to: %s", dense_ply)

    return sparse_dir, mvs_dir


# ─── Backend: Automatic Reconstructor ────────────────────────────

def run_automatic(image_dir, workspace, dense=False, use_gpu=True):
    """Run COLMAP's all-in-one automatic_reconstructor."""
    check_colmap_installed()

    args = [
        "automatic_reconstructor",
        "--image_path", str(image_dir),
        "--workspace_path", str(workspace),
    ]
    if dense:
        args += ["--dense", "1"]
    else:
        args += ["--dense", "0"]

    log.info("Running automatic reconstructor...")
    run_colmap_cmd(args)

    sparse_dir = workspace / "sparse"
    mvs_dir = workspace / "dense" if dense else None

    # Export sparse PLY
    model_dir = sparse_dir / "0"
    if model_dir.exists():
        sparse_ply = sparse_dir / "sparse.ply"
        run_colmap_cmd([
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ])
        log.info("Sparse point cloud saved to: %s", sparse_ply)

    return sparse_dir, mvs_dir


# ─── Visualization ───────────────────────────────────────────────

def visualize_point_cloud(ply_path):
    """Open a PLY point cloud in an Open3D viewer window."""
    try:
        import open3d as o3d
    except ImportError:
        log.error("open3d not installed. Run: pip install open3d")
        return

    ply_path = Path(ply_path)
    if not ply_path.exists():
        log.error("PLY file not found: %s", ply_path)
        return

    log.info("Loading point cloud: %s", ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    log.info("Point cloud has %d points", len(pcd.points))

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Point Cloud — {ply_path.name}",
        width=1280,
        height=720,
    )


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COLMAP Point Cloud Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True,
        help="Path to folder containing input photos",
    )
    parser.add_argument(
        "--workspace", type=Path, required=True,
        help="Path to workspace folder for COLMAP outputs",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Run dense reconstruction (requires NVIDIA GPU with CUDA)",
    )
    parser.add_argument(
        "--backend", choices=["pycolmap", "cli", "auto"], default="cli",
        help="Pipeline backend: pycolmap (Python bindings), cli (subprocess), auto (automatic reconstructor). Default: cli",
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="Disable GPU acceleration (slower but works without CUDA)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open point cloud viewer after reconstruction",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate inputs
    if not args.image_dir.exists():
        log.error("Image directory does not exist: %s", args.image_dir)
        sys.exit(1)

    num_images = count_images(args.image_dir)
    if num_images == 0:
        log.error("No supported images found in: %s", args.image_dir)
        sys.exit(1)
    log.info("Found %d images in %s", num_images, args.image_dir)

    if num_images < 3:
        log.warning("Very few images (%d). COLMAP needs at least 3, ideally 20+.", num_images)

    # GPU detection
    use_gpu = not args.no_gpu
    if use_gpu:
        has_gpu = check_gpu_available()
        if has_gpu:
            log.info("NVIDIA GPU detected — GPU acceleration enabled.")
        else:
            log.warning("No NVIDIA GPU detected. Running on CPU (slower).")
            use_gpu = False

    if args.dense and not use_gpu:
        log.error("Dense reconstruction requires CUDA GPU. Use --no_gpu to skip, or remove --dense.")
        sys.exit(1)

    # Create workspace
    args.workspace.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    if args.backend == "pycolmap":
        sparse_dir, mvs_dir = run_with_pycolmap(
            args.image_dir, args.workspace, args.dense, use_gpu
        )
    elif args.backend == "cli":
        sparse_dir, mvs_dir = run_with_cli(
            args.image_dir, args.workspace, args.dense, use_gpu
        )
    elif args.backend == "auto":
        sparse_dir, mvs_dir = run_automatic(
            args.image_dir, args.workspace, args.dense, use_gpu
        )

    log.info("Pipeline complete!")

    # Visualize
    if args.visualize:
        if args.dense and mvs_dir:
            dense_ply = mvs_dir / "fused.ply"
            if dense_ply.exists():
                visualize_point_cloud(dense_ply)
            else:
                log.warning("Dense PLY not found, showing sparse instead.")
                visualize_point_cloud(sparse_dir / "sparse.ply")
        else:
            visualize_point_cloud(sparse_dir / "sparse.ply")


if __name__ == "__main__":
    main()
