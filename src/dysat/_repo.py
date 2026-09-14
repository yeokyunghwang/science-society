"""Make `scisoc` importable no matter which directory the DySAT scripts run from.

The DySAT package keeps the upstream convention of being launched from its own
directory (`python train.py ...`), so it cannot rely on the repository being
installed. This shim puts `<repo>/src` on `sys.path` and re-exports the shared
path configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]          # <repo>/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scisoc.config import paths  # noqa: E402

__all__ = ["paths"]
