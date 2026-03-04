#!/usr/bin/env python3
"""
Offline EE workspace convex hull computation for a fixed-base robot.

Samples joint configurations uniformly over URDF joint limits, runs FK,
builds a convex hull (H-representation) of EE positions in the base frame,
and saves it to a YAML file for use by WorkspaceHull at runtime.

Usage:
  python3 compute_workspace.py \
    --urdf /path/to/robot.urdf \
    --frame optimo_ee_link \
    --samples 100000 \
    --output ../config/optimo_workspace.yaml

Dependencies:
  pip install pin numpy scipy pyyaml
  (pin = pinocchio Python bindings)
"""

import argparse
import sys
import time
import numpy as np
import yaml

try:
    import pinocchio as pin
except ImportError:
    print("ERROR: pinocchio Python bindings not found. Install with: pip install pin")
    sys.exit(1)

try:
    from scipy.spatial import ConvexHull
except ImportError:
    print("ERROR: scipy not found. Install with: pip install scipy")
    sys.exit(1)


def resolve_urdf_path(urdf_path: str) -> str:
    """Resolve package:// or file:// URIs to absolute paths."""
    if urdf_path.startswith("package://"):
        # Try ament_index_python for ROS2 packages
        try:
            from ament_index_python.packages import get_package_share_directory
            rest = urdf_path[len("package://"):]
            pkg, *parts = rest.split("/")
            pkg_dir = get_package_share_directory(pkg)
            return pkg_dir + "/" + "/".join(parts)
        except Exception as e:
            print(f"WARNING: Could not resolve package:// via ament_index: {e}")
            print("         Pass an absolute path instead.")
            sys.exit(1)
    if urdf_path.startswith("file://"):
        return urdf_path[len("file://"):]
    return urdf_path


def sample_workspace(model, data, frame_id: int, n_samples: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Sample EE positions in base frame over joint limits."""
    q_lo = model.lowerPositionLimit
    q_hi = model.upperPositionLimit

    # Clamp to finite range (some limits may be ±1e300 for continuous joints)
    finite_mask = np.isfinite(q_lo) & np.isfinite(q_hi)
    q_lo = np.where(finite_mask, q_lo, -np.pi)
    q_hi = np.where(finite_mask, q_hi,  np.pi)

    points = np.empty((n_samples, 3))
    for i in range(n_samples):
        q = q_lo + rng.random(model.nq) * (q_hi - q_lo)
        pin.framesForwardKinematics(model, data, q)
        points[i] = data.oMf[frame_id].translation.copy()

    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute EE workspace convex hull (H-rep) from URDF.")
    parser.add_argument("--urdf",    required=True,
                        help="Path to URDF file (absolute, file://, or package://)")
    parser.add_argument("--frame",   required=True,
                        help="EE frame name as defined in the URDF")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Number of random joint configurations to sample (default: 100000)")
    parser.add_argument("--output",  default="optimo_workspace.yaml",
                        help="Output YAML file path (default: optimo_workspace.yaml)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    args = parser.parse_args()

    urdf_path = resolve_urdf_path(args.urdf)
    print(f"Loading URDF: {urdf_path}")
    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()

    frame_id = model.getFrameId(args.frame)
    if frame_id >= model.nframes:
        print(f"ERROR: Frame '{args.frame}' not found in URDF.")
        print("Available frames:")
        for f in model.frames:
            print(f"  {f.name}")
        sys.exit(1)

    print(f"  nq = {model.nq}, frame = '{args.frame}' (id={frame_id})")
    print(f"Sampling {args.samples:,} configurations...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    points = sample_workspace(model, data, frame_id, args.samples, rng)
    t1 = time.perf_counter()
    print(f"  Done in {t1 - t0:.1f}s  —  "
          f"EE range: x=[{points[:,0].min():.3f}, {points[:,0].max():.3f}]  "
          f"y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}]  "
          f"z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    print("Computing convex hull...")
    hull = ConvexHull(points)
    print(f"  {len(hull.equations)} faces, {len(hull.vertices)} vertices")

    # H-representation: for each face, normal.dot(x) + offset <= 0 (interior points).
    # hull.equations shape: (n_faces, 4), columns = [nx, ny, nz, d]
    normals = hull.equations[:, :3]   # outward unit normals
    offsets = hull.equations[:,  3]   # d such that n.dot(x) + d <= 0 inside

    output = {
        "metadata": {
            "urdf":    urdf_path,
            "frame":   args.frame,
            "samples": args.samples,
            "seed":    args.seed,
            "n_faces": int(len(hull.equations)),
        },
        "normals": normals.tolist(),
        "offsets": offsets.tolist(),
    }

    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
