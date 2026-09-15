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


def load_embeddings(source: str, path: Path | None = None, variant: str = "gat"):
    """Return `(E, active)` from the file src/dysat/train.py wrote for one arena.

    E       float32 [N, T, F]  -- one vector per concept per snapshot
    active  bool    [N, T]     -- concept has at least one edge in that snapshot;
                                  this is the candidate set the loss normalized over,
                                  so it is the candidate set for perplexity too.

    `variant` ("gat" | "gatv2") picks the file, `<source>_E.npz` or
    `<source>_gatv2_E.npz`; an explicit `path` wins. The file records which variant
    trained it, and a mismatch with `variant` is an error rather than a silent swap.

    Notebook 03 reads a T-snapshot embedding as the LAST T years of node_info.
    """
    explicit = path is not None
    path = Path(path) if explicit else paths.embedding_npz(source, variant)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run  python train.py --dataset {source} --years FIRST-LAST"
            + ("" if variant == "gat" else f" --attn_variant {variant}"))

    with np.load(path, allow_pickle=False) as d:
        missing = {"E", "active"} - set(d.files)
        if missing:
            raise KeyError(f"{path} lacks {sorted(missing)}; not a train.py export")
        E = d["E"]
        active = d["active"].astype(bool)
        stored = str(d["attn_variant"]) if "attn_variant" in d.files else None

    if E.ndim != 3 or active.shape != E.shape[:2]:
        raise ValueError(f"{path}: E is {E.shape}, active is {active.shape}; "
                         "expected E [N, T, F] and active [N, T]")
    # With the default path, the name promised a variant; make the file agree.
    # (Files from before the flag existed carry no record and are let through.)
    if not explicit and stored is not None and stored != variant:
        raise ValueError(f"{path} was trained with attn_variant={stored!r}, "
                         f"but variant={variant!r} was requested")
    return E, active


def active_mask(source: str, info: dict | None = None) -> np.ndarray:
    """Boolean [N, T] mask from node_info: concept appears in at least one document
    that year. NOT the same as the embedding's `active` (at least one edge): a concept
    that is the only subject of its documents has frequency > 0 but no edges. For
    anything scored against the model use the `active` that load_embeddings returns.
    """
    info = info if info is not None else load_node_info(source)
    return np.asarray(info["concept_freq_year"]).T > 0      # [T, V] -> [V, T]
