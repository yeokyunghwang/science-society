"""Loaders shared by the notebooks and by the DySAT scripts.

Every function takes a `source` in {"news", "paper"} and resolves its own path
through `scisoc.config.paths`, so a notebook never spells a directory out.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from scisoc.config import paths

__all__ = [
    "load_node_info",
    "load_adj",
    "iter_adj",
    "load_embeddings",
    "active_mask",
]


def load_node_info(source: str, net_dir: Path | None = None) -> dict:
    """`node_info.pkl` written by notebook 01.

    Keys: vocab, years, concept_freq_year [T, V], concept_freq_total [V],
    concept_share [V], concept_share_all [V], n_docs_all [T], n_docs_ge2 [T],
    V_t [T], E_t [T].
    """
    net_dir = Path(net_dir) if net_dir is not None else paths.networks
    with open(net_dir / source / "node_info.pkl", "rb") as f:
        return pickle.load(f)


def load_adj(source: str, year: int, net_dir: Path | None = None):
    """One year's raw co-occurrence matrix (scipy csr, symmetric, no self-loops)."""
    from scipy.sparse import load_npz  # local import: notebooks that only need
    net_dir = Path(net_dir) if net_dir is not None else paths.networks  # paths skip scipy
    return load_npz(net_dir / source / f"adj_{year}.npz").tocsr()


def iter_adj(source: str, years, net_dir: Path | None = None):
    """Yield `(year, csr_matrix)` for each year, reading one file at a time."""
    for year in years:
        yield int(year), load_adj(source, int(year), net_dir)


def load_embeddings(source: str, path: Path | None = None):
    """Return `(E, active)` for one arena.

    Accepts either
      * the DySAT export `<source>_E.npz` (keys `E` [N, T, F], `active` [N, T]), or
      * a bare `.npy` array [N, T, F] -- then `active` is None and the caller
        falls back to `node_info["concept_freq_year"] > 0`.
    """
    path = Path(path) if path is not None else paths.embedding_npz(source)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            E = d["E"]
            act = d["active"].astype(bool) if "active" in d.files else None
        return E, act
    return np.load(path), None


def active_mask(source: str, info: dict | None = None, emb_active=None) -> np.ndarray:
    """Boolean [N, T] mask of concepts that occur at least once in a given year.

    Prefers the mask stored next to the embeddings (it is the one the model was
    trained with); falls back to the yearly document frequency from notebook 01.
    """
    if emb_active is not None:
        return np.asarray(emb_active, dtype=bool)
    info = info if info is not None else load_node_info(source)
    return np.asarray(info["concept_freq_year"]).T > 0      # [T, V] -> [V, T]
