"""Single source of truth for every path used in this project.

Nothing in this repository hard-codes an absolute path. The repository root is
found by walking up from this file, and the two locations that genuinely live
outside the repository -- the raw subject-term dumps and (optionally) a large
scratch area for processed data -- are read from environment variables or from
a `.env` file at the repository root.

Environment variables (all optional)
------------------------------------
SCISOC_ROOT       repository root (default: inferred from this file)
SCISOC_RAW        directory holding news/paper `*_subject_by_year.pkl`
                  (default: <root>/data/raw)
SCISOC_DATA       processed-data root (default: <root>/data/processed)
SCISOC_RESULTS    results root        (default: <root>/results)

`.env` example (repository root, not tracked by git)::

    SCISOC_RAW=/Volumes/My Passport_2/Science-Society/data/processed

Usage::

    from scisoc.config import paths
    paths.networks / "news" / "adj_1995.npz"
    paths.ensure()                      # create the output directories
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Paths", "paths", "find_repo_root", "load_dotenv"]

# Markers that identify the repository root when walking up the tree.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until a root marker is found.

    Falls back to two levels above `src/scisoc/` so that the package still
    resolves when the repository is vendored without `.git`.
    """
    start = Path(start) if start is not None else Path(__file__).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    # src/scisoc/config.py -> src/scisoc -> src -> <root>
    return Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """Minimal `.env` reader: `KEY=value` lines, `#` comments, no interpolation.

    Existing environment variables always win, so an explicit `export` in the
    shell overrides the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    """Every directory the pipeline reads from or writes to."""

    root: Path

    # ---- inputs -------------------------------------------------------
    @property
    def raw(self) -> Path:
        """Raw subject-term tables (`news_subject_by_year.pkl`, `paper_...`)."""
        return _env_path("SCISOC_RAW", self.root / "data" / "raw")

    def raw_subjects(self, source: str) -> Path:
        return self.raw / f"{source}_subject_by_year.pkl"

    # ---- processed data ----------------------------------------------
    @property
    def data(self) -> Path:
        return _env_path("SCISOC_DATA", self.root / "data" / "processed")

    @property
    def networks(self) -> Path:
        """`networks/<source>/adj_<year>.npz` + `node_info.pkl` (notebook 01)."""
        return self.data / "networks"

    @property
    def backbone(self) -> Path:
        """Disparity backbone edge tables (notebook 03)."""
        return self.data / "backbone"

    @property
    def embeddings(self) -> Path:
        """DySAT output: `<source>[_<variant>]_E.npz` with E [N, T, F] and active [N, T]."""
        return self.data / "embeddings"

    # ---- results ------------------------------------------------------
    @property
    def results(self) -> Path:
        return _env_path("SCISOC_RESULTS", self.root / "results")

    @property
    def cp_results(self) -> Path:
        return self.results / "cp_results"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def dysat_logs(self) -> Path:
        """Training logs / checkpoints (not tracked)."""
        return self.results / "dysat"

    # ---- helpers ------------------------------------------------------
    def network_adj(self, source: str, year: int) -> Path:
        return self.networks / source / f"adj_{year}.npz"

    def node_info(self, source: str) -> Path:
        return self.networks / source / "node_info.pkl"

    def embedding_npz(self, source: str, variant: str = "gat") -> Path:
        """`<source>_E.npz` for GAT, `<source>_gatv2_E.npz` for GATv2.

        The plain name is kept for GAT so runs made before the variant flag existed
        still resolve.
        """
        stem = source if variant == "gat" else f"{source}_{variant}"
        return self.embeddings / f"{stem}_E.npz"

    def ensure(self) -> "Paths":
        """Create every output directory. Inputs are never created."""
        for d in (self.networks, self.backbone, self.embeddings,
                  self.cp_results, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        rows = [
            ("root", self.root),
            ("raw (input)", self.raw),
            ("networks", self.networks),
            ("backbone", self.backbone),
            ("embeddings", self.embeddings),
            ("cp_results", self.cp_results),
            ("figures", self.figures),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


_ROOT = _env_path("SCISOC_ROOT", find_repo_root())
load_dotenv(_ROOT / ".env")
# Re-read: `.env` may itself define SCISOC_ROOT.
_ROOT = _env_path("SCISOC_ROOT", _ROOT)

paths = Paths(root=_ROOT)
