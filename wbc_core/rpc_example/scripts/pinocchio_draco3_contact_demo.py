#!/usr/bin/env python3
"""Pinocchio Draco3 constrained-contact simulation demo.

This script does not perform collision detection. Contact is assumed active
for the specified contact frames and enforced via Pinocchio's
`constraintDynamics`.
"""

from __future__ import annotations

import argparse
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
  rpc_example_share = _get_share_dir("rpc_example")
  draco_description_share = _get_share_dir("draco_description")

  if user_urdf:
    urdf_path = Path(user_urdf).expanduser().resolve()
  else:
    candidates: list[Path] = []
    if draco_description_share is not None:
      candidates.append(
          draco_description_share / "urdf/draco_modified_rviz.urdf"
      )
    if len(script_file.parents) > 3:
      candidates.append(
          script_file.parents[3]
          / "description/draco_description/urdf/draco_modified_rviz.urdf"
      )
    urdf_path = next((p for p in candidates if p.is_file()), candidates[0])

  package_dirs: list[str] = []
  for user_dir in user_package_dirs:
    _append_unique_dir(package_dirs, user_dir)

  from_env = os.environ.get("PINOCCHIO_PACKAGE_DIRS", "")
  for env_dir in from_env.split(os.pathsep):
    if env_dir:
      _append_unique_dir(package_dirs, env_dir)

  if draco_description_share is not None:
    _append_unique_dir(package_dirs, draco_description_share.parent)
    _append_unique_dir(package_dirs, draco_description_share)

  if rpc_example_share is not None:
    _append_unique_dir(package_dirs, rpc_example_share.parent)

  if len(script_file.parents) > 3:
    _append_unique_dir(package_dirs, script_file.parents[3])

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

  visualizer = MeshcatVisualizer(model, collision_model, visual_model)
  visualizer.initViewer(open=open_viewer)
  visualizer.loadViewerModel(rootNodeName="draco3_contact_demo")
  return visualizer


def _find_contact_frames(model: pin.Model, names: list[str]) -> list[int]:
  frame_ids: list[int] = []
  for name in names:
    if not model.existFrame(name):
      candidates = [
          f.name for f in model.frames if "foot" in f.name.lower() or "contact" in f.name.lower()
      ]
      raise RuntimeError(
          f"Contact frame '{name}' not found. Candidate frames: {candidates[:20]}"
      )
    frame_ids.append(model.getFrameId(name))
  return frame_ids


def _build_contact_models(
    model: pin.Model,
    data: pin.Data,
    contact_names: list[str],
    contact_type: str,
    contact_kp: float,
    contact_kd: float,
) -> tuple[list[pin.RigidConstraintModel], list[pin.RigidConstraintData]]:
  pin.framesForwardKinematics(model, data, pin.neutral(model))
  frame_ids = _find_contact_frames(model, contact_names)

  ctype = (
      pin.ContactType.CONTACT_6D if contact_type == "6d" else pin.ContactType.CONTACT_3D
  )

  models: list[pin.RigidConstraintModel] = []
  for frame_id in frame_ids:
    joint_id = model.frames[frame_id].parentJoint
    joint_placement = model.frames[frame_id].placement
    world_contact = data.oMf[frame_id]
    cm = pin.RigidConstraintModel(
        ctype,
        model,
        joint_id,
        joint_placement,
        0,
        world_contact,
        pin.ReferenceFrame.LOCAL,
    )
    cm.corrector.Kp[:] = contact_kp
    cm.corrector.Kd[:] = contact_kd
    models.append(cm)

  datas = [cm.createData() for cm in models]
  return models, datas


def _format_vec3(vec: np.ndarray) -> str:
  return f"({vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f})"


def _run(
    model: pin.Model,
    data: pin.Data,
    visualizer: object | None,
    contact_models: list[pin.RigidConstraintModel],
    contact_datas: list[pin.RigidConstraintData],
    contact_names: list[str],
    dt: float,
    duration: float,
    base_height: float,
    hold_kp: float,
    hold_kd: float,
    max_accel: float,
    max_velocity: float,
    force_threshold: float,
    prox_abs: float,
    prox_rel: float,
    prox_iters: int,
    real_time: bool,
    print_every: int,
) -> int:
  if model.nv <= 6:
    raise RuntimeError("Expected floating-base model (nv > 6).")

  q = pin.neutral(model)
  q[2] = base_height
  v = np.zeros(model.nv)
  q_ref = q.copy()

  pin.forwardKinematics(model, data, q, v)
  pin.updateFramePlacements(model, data)

  # Lock contact target to the initial frame poses in world.
  for frame_name, cm in zip(contact_names, contact_models):
    frame_id = model.getFrameId(frame_name)
    cm.joint2_placement = data.oMf[frame_id]

  pin.initConstraintDynamics(model, data, contact_models)
  prox = pin.ProximalSettings(prox_abs, prox_rel, prox_iters)

  steps = max(1, int(duration / dt))
  wall_start = time.perf_counter()

  for step in range(steps):
    t = step * dt

    tau = np.zeros(model.nv)
    gravity = pin.computeGeneralizedGravity(model, data, q)

    # Floating-base is unactuated. Apply torque only to actuated joints.
    tau[6:] = gravity[6:]
    tau[6:] += hold_kp * (q_ref[7:] - q[7:]) - hold_kd * v[6:]

    qdd = pin.constraintDynamics(
        model, data, q, v, tau, contact_models, contact_datas, prox
    )

    if max_accel > 0.0:
      qdd = np.clip(qdd, -max_accel, max_accel)

    v = v + qdd * dt
    if max_velocity > 0.0:
      v = np.clip(v, -max_velocity, max_velocity)

    q = pin.integrate(model, q, v * dt)

    if visualizer is not None:
      visualizer.display(q)

    if print_every > 0 and (step % print_every) == 0:
      pin.centerOfMass(model, data, q, v)

      drift_trans = [np.linalg.norm(cd.c1Mc2.translation) for cd in contact_datas]
      drift_rot = [np.linalg.norm(pin.log3(cd.c1Mc2.rotation)) for cd in contact_datas]

      normals: list[float] = []
      active: list[bool] = []
      offset = 0
      for cm in contact_models:
        dim = cm.size()
        lam_seg = data.lambda_c[offset : offset + dim]
        force = pin.Force(lam_seg)
        normal = float(force.linear[2])
        normals.append(normal)
        active.append(normal > force_threshold)
        offset += dim

      status = ", ".join(
          f"{name}:fz={fz: .2f}N active={is_on}"
          for name, fz, is_on in zip(contact_names, normals, active)
      )
      print(
          f"[step {step:6d}] t={t:7.3f}s com={_format_vec3(data.com[0])} "
          f"drift_t_max={max(drift_trans):.3e} drift_r_max={max(drift_rot):.3e} "
          f"| {status}"
      )

    if real_time:
      target = wall_start + (step + 1) * dt
      remain = target - time.perf_counter()
      if remain > 0.0:
        time.sleep(remain)

  print("Simulation finished.")
  return 0


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Draco3 Pinocchio contact demo using constraintDynamics "
          "(assumed active contacts)."
      )
  )
  parser.add_argument("--urdf", default=None, help="URDF path.")
  parser.add_argument(
      "--package-dir",
      action="append",
      default=[],
      help="Additional package root for package:// URIs (repeatable).",
  )
  parser.add_argument(
      "--contacts",
      default="l_foot_contact,r_foot_contact",
      help="Comma-separated contact frame names.",
  )
  parser.add_argument(
      "--contact-type",
      choices=["3d", "6d"],
      default="6d",
      help="Constraint type per contact frame.",
  )
  parser.add_argument("--dt", type=float, default=0.001)
  parser.add_argument("--duration", type=float, default=6.0)
  parser.add_argument("--base-height", type=float, default=0.95)
  parser.add_argument("--hold-kp", type=float, default=40.0)
  parser.add_argument("--hold-kd", type=float, default=6.0)
  parser.add_argument("--contact-kp", type=float, default=120.0)
  parser.add_argument("--contact-kd", type=float, default=20.0)
  parser.add_argument("--max-accel", type=float, default=120.0)
  parser.add_argument("--max-velocity", type=float, default=12.0)
  parser.add_argument("--contact-force-threshold", type=float, default=20.0)
  parser.add_argument("--prox-abs", type=float, default=1e-8)
  parser.add_argument("--prox-rel", type=float, default=1e-8)
  parser.add_argument("--prox-iters", type=int, default=30)
  parser.add_argument("--print-every", type=int, default=250)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--open-viewer", action="store_true")
  parser.add_argument("--no-real-time", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  if args.dt <= 0.0 or args.duration <= 0.0:
    print("`--dt` and `--duration` must be positive.", file=sys.stderr)
    return 2

  contact_names = [name.strip() for name in args.contacts.split(",") if name.strip()]
  if not contact_names:
    print("At least one contact frame is required.", file=sys.stderr)
    return 2

  urdf_path, package_dirs = _resolve_paths(args.urdf, args.package_dir)
  if not urdf_path.is_file():
    print(f"URDF not found: {urdf_path}", file=sys.stderr)
    return 2

  print(f"URDF: {urdf_path}")
  print(f"Package dirs ({len(package_dirs)}):")
  for pdir in package_dirs:
    print(f"  - {pdir}")
  print(f"Contacts: {contact_names} ({args.contact_type.upper()})")

  try:
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), package_dirs, pin.JointModelFreeFlyer()
    )
  except Exception as exc:
    print(f"Failed to build model: {exc}", file=sys.stderr)
    return 2

  data = model.createData()
  q_seed = pin.neutral(model)
  q_seed[2] = args.base_height
  pin.forwardKinematics(model, data, q_seed)
  pin.updateFramePlacements(model, data)

  try:
    contact_models, contact_datas = _build_contact_models(
        model,
        data,
        contact_names,
        args.contact_type,
        args.contact_kp,
        args.contact_kd,
    )
  except Exception as exc:
    print(f"Failed to build contact models: {exc}", file=sys.stderr)
    return 2

  visualizer = None
  if not args.headless:
    try:
      visualizer = _make_visualizer(
          model, collision_model, visual_model, args.open_viewer
      )
    except Exception as exc:
      print(f"Viewer init failed: {exc}", file=sys.stderr)
      return 2

  try:
    return _run(
        model=model,
        data=data,
        visualizer=visualizer,
        contact_models=contact_models,
        contact_datas=contact_datas,
        contact_names=contact_names,
        dt=args.dt,
        duration=args.duration,
        base_height=args.base_height,
        hold_kp=args.hold_kp,
        hold_kd=args.hold_kd,
        max_accel=args.max_accel,
        max_velocity=args.max_velocity,
        force_threshold=args.contact_force_threshold,
        prox_abs=args.prox_abs,
        prox_rel=args.prox_rel,
        prox_iters=args.prox_iters,
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
