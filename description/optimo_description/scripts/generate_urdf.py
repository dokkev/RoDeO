#!/usr/bin/env python3
"""Generate standalone optimo.urdf from xacro.

Usage: python3 generate_urdf.py <xacro_input> <urdf_output>

Post-processes xacro output to:
- Replace package:// mesh paths with relative paths
- Add MuJoCo <compiler meshdir> element
"""
import re
import subprocess
import sys


def main():
    xacro_input = sys.argv[1]
    urdf_output = sys.argv[2]

    result = subprocess.run(
        ["xacro", xacro_input], capture_output=True, text=True, check=True
    )
    urdf = result.stdout

    # Replace package:// mesh paths with relative (for Pinocchio / MuJoCo)
    urdf = urdf.replace("package://optimo_description/urdf/meshes/", "meshes/")

    # Add mujoco compiler element for MuJoCo URDF loading
    urdf = re.sub(
        r"(<robot[^>]*>)",
        r'\1\n  <mujoco>\n    <compiler meshdir="meshes/"/>\n  </mujoco>',
        urdf,
    )

    with open(urdf_output, "w") as f:
        f.write(urdf)

    print(f"Generated {urdf_output}")


if __name__ == "__main__":
    main()
