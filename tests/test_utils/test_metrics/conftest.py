"""Local imports for metric invariance helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow sibling helper modules to be imported without making ``tests`` a package.
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
