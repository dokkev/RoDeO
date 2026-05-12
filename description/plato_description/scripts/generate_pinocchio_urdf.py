#!/usr/bin/env python3
"""Generate a standalone Plato NariTouch URDF for Pinocchio."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root")
    parser.add_argument("urdf_output")
    parser.add_argument("--xacro", default="xacro")
    args = parser.parse_args()

    package_name = "plato_description"
    package_root = Path(args.package_root).resolve()
    urdf_output = Path(args.urdf_output).resolve()
    description_xacro = (
        package_root / "urdf" / "xacro" / "plato_naritouch_description.xacro"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prefix = tmp_path / "prefix"
        resource_index = (
            prefix / "share" / "ament_index" / "resource_index" / "packages"
        )
        resource_index.mkdir(parents=True)
        (resource_index / package_name).write_text("")
        (prefix / "share" / package_name).symlink_to(package_root)

        wrapper = tmp_path / "plato_naritouch_pinocchio.urdf.xacro"
        wrapper.write_text(
            f"""<?xml version="1.0"?>
<robot name="plato_naritouch_hand" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="{description_xacro}"/>
  <xacro:plato_naritouch_description connector_link="world" xyz="0 0 0" rpy="0 0 0"/>
</robot>
"""
        )

        env = os.environ.copy()
        env["AMENT_PREFIX_PATH"] = (
            str(prefix)
            if not env.get("AMENT_PREFIX_PATH")
            else str(prefix) + os.pathsep + env["AMENT_PREFIX_PATH"]
        )
        result = subprocess.run(
            [args.xacro, str(wrapper)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

    urdf = result.stdout
    urdf = urdf.replace(
        f"file://{prefix / 'share' / package_name}/",
        f"package://{package_name}/",
    )
    urdf = urdf.replace(f"file://{package_root}/", f"package://{package_name}/")
    urdf_output.parent.mkdir(parents=True, exist_ok=True)
    urdf_output.write_text(urdf)
    print(f"Generated {urdf_output}")


if __name__ == "__main__":
    main()
