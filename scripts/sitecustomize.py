"""
Ensure UTF-8 (no BOM) + LF write policy also applies when running Python
scripts from inside the `scripts/` directory (so that Python's automatic
sitecustomize import finds the repository root module).
"""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Import the root-level sitecustomize to install the patches
import sitecustomize as _root_sitecustomize  # noqa: F401

