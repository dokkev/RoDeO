#!/usr/bin/env python3
"""Run a simple Pinocchio RBD simulation and visualize it with Meshcat."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin


def _append_unique_dir(container: list[str], candidate: Path | str | None) -> None:
  if candidate is None:
    return
  path = Path(candidate).expanduser().resolve()
  if not path.is_dir():
    return
  path_str = str(path)
  if path_str not in container:
    container.append(path_str)


def _get_share_dir(package_name: str) -> Path | None:
  try:
    from ament_index_python.packages import (  # type: ignore
        PackageNotFoundError,
        get_package_share_directory,
    )
  except Exception:
    return None

  try:
    return Path(get_package_share_directory(package_name))
  except PackageNotFoundError:
    return None


def _resolve_paths(
    user_urdf: str | None, user_package_dirs: list[str]
) -> tuple[Path, list[str]]:
  script_file = Path(__file__).resolve()
  script_pkg_root = script_file.parents[1]
  rpc_example_share = _get_share_dir("rpc_example")
  optimo_description_share = _get_share_dir("optimo_description")

  if user_urdf:
    urdf_path = Path(user_urdf).expanduser().resolve()
  else:
    candidates = []
    if rpc_example_share is not None:
      candidates.append(rpc_example_share / "description/urdf/optimo.urdf")
    candidates.append(script_pkg_root / "description/urdf/optimo.urdf")
    urdf_path = next((p for p in candidates if p.is_file()), candidates[0])

  package_dirs: list[str] = []
  for user_dir in user_package_dirs:
    _append_unique_dir(package_dirs, user_dir)

  from_env = os.environ.get("PINOCCHIO_PACKAGE_DIRS", "")
  for env_dir in from_env.split(os.pathsep):
    if env_dir:
      _append_unique_dir(package_dirs, env_dir)

  if optimo_description_share is not None:
    _append_unique_dir(package_dirs, optimo_description_share.parent)
    _append_unique_dir(package_dirs, optimo_description_share)
  if rpc_example_share is not None:
    _append_unique_dir(package_dirs, rpc_example_share.parent)

  if len(script_file.parents) > 3:
    source_description = script_file.parents[3] / "description"
    _append_unique_dir(package_dirs, source_description)
    _append_unique_dir(package_dirs, source_description / "optimo_description")

  _append_unique_dir(package_dirs, urdf_path.parent)

  return urdf_path, package_dirs


def _make_visualizer(
    model: pin.Model,
    collision_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
    open_viewer: bool,
) -> object:
  try:
    from pinocchio.visualize import MeshcatVisualizer
  except Exception as exc:
    raise RuntimeError(
        "Pinocchio Meshcat visualizer is unavailable. Install meshcat "
        "with: pip install --user meshcat"
    ) from exc

  try:
    visualizer = MeshcatVisualizer(model, collision_model, visual_model)
  except Exception as exc:
    raise RuntimeError(
        "Failed to initialize Meshcat visualizer. "
        "Install meshcat with: pip install --user meshcat"
    ) from exc

  visualizer.initViewer(open=open_viewer)
  visualizer.loadViewerModel(rootNodeName="optimo_rbd")
  return visualizer


def _run_simulation(
    model: pin.Model,
    data: pin.Data,
    visualizer: object | None,
    dt: float,
    duration: float,
    torque_amplitude: float,
    excitation_hz: float,
    damping: float,
    max_accel: float,
    max_velocity: float,
    real_time: bool,
    print_every: int,
) -> int:
  q = pin.neutral(model)
  v = np.zeros(model.nv)
  tau = np.zeros(model.nv)

  actuated_start = 0
  actuated_dofs = max(model.nv - actuated_start, 0)
  phases = np.linspace(0.0, math.pi, num=max(actuated_dofs, 1), endpoint=False)

  num_steps = max(1, int(duration / dt))
  wall_start = time.perf_counter()

  for step in range(num_steps):
    t = step * dt
    tau.fill(0.0)

    if actuated_dofs > 0:
      tau_wave = np.sin(2.0 * math.pi * excitation_hz * t + phases[:actuated_dofs])
      tau[actuated_start:] = torque_amplitude * tau_wave - damping * v[actuated_start:]

    ddq = pin.aba(model, data, q, v, tau)
    if max_accel > 0.0:
      ddq = np.clip(ddq, -max_accel, max_accel)

    v += ddq * dt
    if max_velocity > 0.0:
      v = np.clip(v, -max_velocity, max_velocity)

    q = pin.integrate(model, q, v * dt)

    if visualizer is not None:
      visualizer.display(q)

    if print_every > 0 and step % print_every == 0:
      com = pin.centerOfMass(model, data, q, v)
      print(
          f"[step {step:6d}] t={t:7.3f}s | |v|={np.linalg.norm(v):7.4f} "
          f"| com=({com[0]: .3f}, {com[1]: .3f}, {com[2]: .3f})"
      )

    if real_time:
      target = wall_start + (step + 1) * dt
      remaining = target - time.perf_counter()
      if remaining > 0.0:
        time.sleep(remaining)

  print("Simulation finished.")
  return 0


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Pinocchio rigid-body dynamics simulation with Meshcat visualization."
  )
  parser.add_argument(
      "--urdf",
      default=None,
      help="URDF path. Default: rpc_example/description/urdf/optimo.urdf",
  )
  parser.add_argument(
      "--package-dir",
      action="append",
      default=[],
      help=(
          "Additional package root for package:// URIs. "
          "Can be repeated; also reads PINOCCHIO_PACKAGE_DIRS."
      ),
  )
  parser.add_argument("--dt", type=float, default=0.002, help="Simulation step [s].")
  parser.add_argument(
      "--duration", type=float, default=20.0, help="Simulation duration [s]."
  )
  parser.add_argument(
      "--torque-amplitude",
      type=float,
      default=0.3,
      help="Sinusoidal torque amplitude [Nm] for actuated joints.",
  )
  parser.add_argument(
      "--excitation-hz",
      type=float,
      default=0.25,
      help="Sinusoidal excitation frequency [Hz].",
  )
  parser.add_argument(
      "--damping",
      type=float,
      default=3.0,
      help="Velocity damping gain applied to actuated joints.",
  )
  parser.add_argument(
      "--armature",
      type=float,
      default=0.1,
      help="Diagonal armature regularization for ABA [kg*m^2].",
  )
  parser.add_argument(
      "--max-accel",
      type=float,
      default=50.0,
      help="Clamp joint acceleration to +/- value [rad/s^2]. <=0 disables.",
  )
  parser.add_argument(
      "--max-velocity",
      type=float,
      default=5.0,
      help="Clamp joint velocity to +/- value [rad/s]. <=0 disables.",
  )
  parser.add_argument(
      "--headless",
      action="store_true",
      help="Run dynamics loop without launching Meshcat.",
  )
  parser.add_argument(
      "--open-viewer",
      action="store_true",
      help="Open browser automatically when Meshcat starts.",
  )
  parser.add_argument(
      "--no-real-time",
      action="store_true",
      help="Run as fast as possible instead of wall-clock synchronized.",
  )
  parser.add_argument(
      "--print-every",
      type=int,
      default=250,
      help="Print status every N steps (<=0 disables logging).",
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  if args.dt <= 0.0:
    print("`--dt` must be positive.", file=sys.stderr)
    return 2
  if args.duration <= 0.0:
    print("`--duration` must be positive.", file=sys.stderr)
    return 2
  if args.armature < 0.0:
    print("`--armature` cannot be negative.", file=sys.stderr)
    return 2

  urdf_path, package_dirs = _resolve_paths(args.urdf, args.package_dir)
  if not urdf_path.is_file():
    print(f"URDF not found: {urdf_path}", file=sys.stderr)
    return 2

  print(f"URDF: {urdf_path}")
  print(f"Package dirs ({len(package_dirs)}):")
  for package_dir in package_dirs:
    print(f"  - {package_dir}")

  try:
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), package_dirs
    )
  except Exception as exc:
    print(f"Failed to build Pinocchio model from URDF: {exc}", file=sys.stderr)
    return 2

  if args.armature > 0.0:
    model.armature[:] = args.armature

  data = model.createData()
  visualizer = None
  if not args.headless:
    try:
      visualizer = _make_visualizer(
          model, collision_model, visual_model, args.open_viewer
      )
    except RuntimeError as exc:
      print(str(exc), file=sys.stderr)
      return 2

  try:
    return _run_simulation(
        model=model,
        data=data,
        visualizer=visualizer,
        dt=args.dt,
        duration=args.duration,
        torque_amplitude=args.torque_amplitude,
        excitation_hz=args.excitation_hz,
        damping=args.damping,
        max_accel=args.max_accel,
        max_velocity=args.max_velocity,
        real_time=not args.no_real_time,
        print_every=args.print_every,
    )
  except KeyboardInterrupt:
    print("\nInterrupted by user.")
    return 130
  except Exception as exc:
    print(f"Simulation failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
