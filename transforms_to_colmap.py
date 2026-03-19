"""
Convert Blender's transforms.json to COLMAP sparse model format.

This bypasses COLMAP's SfM pipeline entirely by using the known camera
intrinsics and extrinsics from Blender. Then runs dense reconstruction
to generate a dense point cloud.

Usage:
  python transforms_to_colmap.py --transforms transforms.json --image_dir images --workspace workspace
  python transforms_to_colmap.py --transforms transforms.json --image_dir images --workspace workspace --dense
"""

import argparse
import json
import logging
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ─── COLMAP binary file writers ──────────────────────────────────

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def write_cameras_bin(path, camera):
    """Write cameras.bin in COLMAP binary format."""
    # OPENCV model id = 4, has 8 params: fx, fy, cx, cy, k1, k2, p1, p2
    with open(path, "wb") as f:
        # Number of cameras
        f.write(struct.pack("<Q", 1))
        # Camera: id, model_id, width, height, params
        camera_id = 1
        model_id = 4  # OPENCV
        width = camera["w"]
        height = camera["h"]
        f.write(struct.pack("<I", camera_id))
        f.write(struct.pack("<i", model_id))
        f.write(struct.pack("<Q", width))
        f.write(struct.pack("<Q", height))
        # 8 params: fx, fy, cx, cy, k1, k2, p1, p2
        params = [
            camera["fl_x"], camera["fl_y"],
            camera["cx"], camera["cy"],
            camera.get("k1", 0.0), camera.get("k2", 0.0),
            camera.get("p1", 0.0), camera.get("p2", 0.0),
        ]
        for p in params:
            f.write(struct.pack("<d", p))


def write_images_bin(path, frames, image_dir):
    """Write images.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        # Number of images
        f.write(struct.pack("<Q", len(frames)))

        for idx, frame in enumerate(frames):
            image_id = idx + 1
            camera_id = 1

            # Get transform matrix (4x4, world-to-camera is inverse of Blender's camera-to-world)
            c2w = np.array(frame["transform_matrix"])
            # Convert from Blender coordinate system (Y-up) to COLMAP (Y-down, Z-forward)
            # Blender: +X right, +Y up, -Z forward
            # COLMAP:  +X right, -Y up, +Z forward
            # Flip Y and Z axes
            c2w[0:3, 1] *= -1  # flip Y
            c2w[0:3, 2] *= -1  # flip Z

            # COLMAP wants world-to-camera (w2c)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            quat = rotation_matrix_to_quaternion(R)  # w, x, y, z

            # Image name (just the filename, not the full path)
            file_path = frame["file_path"]
            image_name = Path(file_path).name
            # Check if file exists with this name
            if not (Path(image_dir) / image_name).exists():
                # Try with different extension
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = Path(file_path).stem + ext
                    if (Path(image_dir) / candidate).exists():
                        image_name = candidate
                        break

            # Write image entry
            f.write(struct.pack("<I", image_id))
            for q in quat:
                f.write(struct.pack("<d", q))
            for ti in t:
                f.write(struct.pack("<d", ti))
            f.write(struct.pack("<I", camera_id))
            # Image name as null-terminated string
            f.write(image_name.encode("utf-8"))
            f.write(b"\x00")
            # Number of 2D points (0 for now — dense reconstruction doesn't need them)
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path):
    """Write empty points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ─── Main pipeline ───────────────────────────────────────────────

def run_colmap_cmd(args):
    """Run a COLMAP CLI command."""
    cmd = ["colmap"] + args
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("COLMAP failed:\n%s", result.stderr)
        raise RuntimeError(f"COLMAP command failed: {args[0]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Convert Blender transforms.json to COLMAP and run dense reconstruction",
    )
    parser.add_argument(
        "--transforms", type=Path, required=True,
        help="Path to transforms.json from Blender",
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True,
        help="Path to folder containing rendered images",
    )
    parser.add_argument(
        "--workspace", type=Path, required=True,
        help="Path to workspace folder for outputs",
    )
    parser.add_argument(
        "--dense", action="store_true",
        help="Run dense reconstruction (requires COLMAP with CUDA)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load transforms.json
    log.info("Loading transforms from: %s", args.transforms)
    with open(args.transforms) as f:
        data = json.load(f)

    frames = data["frames"]
    log.info("Found %d frames", len(frames))
    log.info("Camera: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, %dx%d",
             data["fl_x"], data["fl_y"], data["cx"], data["cy"],
             data["w"], data["h"])

    # Create sparse model directory
    sparse_dir = args.workspace / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Write COLMAP binary model files
    log.info("Writing COLMAP sparse model...")
    write_cameras_bin(sparse_dir / "cameras.bin", data)
    write_images_bin(sparse_dir / "images.bin", frames, args.image_dir)
    write_points3d_bin(sparse_dir / "points3D.bin")
    log.info("Sparse model written to: %s", sparse_dir)

    # Export sparse PLY (will be empty since no 3D points yet)
    sparse_ply = args.workspace / "sparse" / "sparse.ply"
    try:
        run_colmap_cmd([
            "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ])
        log.info("Sparse PLY saved to: %s", sparse_ply)
    except RuntimeError:
        log.warning("Could not export sparse PLY (no 3D points yet, this is normal)")

    if args.dense:
        # Step 1: Undistort images
        mvs_dir = args.workspace / "mvs"
        log.info("Undistorting images...")
        run_colmap_cmd([
            "image_undistorter",
            "--image_path", str(args.image_dir),
            "--input_path", str(sparse_dir),
            "--output_path", str(mvs_dir),
            "--output_type", "COLMAP",
        ])

        # Step 2: PatchMatch stereo
        log.info("Running PatchMatch stereo (this may take a while)...")
        run_colmap_cmd([
            "patch_match_stereo",
            "--workspace_path", str(mvs_dir),
            "--PatchMatchStereo.geom_consistency", "true",
        ])

        # Step 3: Stereo fusion
        dense_ply = mvs_dir / "fused.ply"
        log.info("Fusing into dense point cloud...")
        run_colmap_cmd([
            "stereo_fusion",
            "--workspace_path", str(mvs_dir),
            "--output_path", str(dense_ply),
        ])
        log.info("Dense point cloud saved to: %s", dense_ply)

    log.info("Done!")


if __name__ == "__main__":
    main()
