"""Conditional distributions and perplexity from a saved E.

    from export_probs import load_export, cond_prob, perplexity
    E, active = load_export('news')             # <embeddings>/news_E.npz
    p  = cond_prob(E, active, v, t)             # [N], sums to 1, 0 on inactive & self
    PP = perplexity(E, active, pairs, t)        # pairs: [(v, u), ...] from backbone paths

`p_t(u | v) = softmax_u <E[v,t], E[u,t]>` over the nodes active in year t,
excluding v itself -- exactly the quantity the training loss normalises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # <repo>/src -> scisoc
from scisoc.config import paths

__all__ = ["load_export", "cond_prob", "log_prob_pairs", "perplexity"]


def load_export(source: str, path: Path | None = None):
    """Return `(E, active)` from the npz written by train.py."""
    path = Path(path) if path is not None else paths.embedding_npz(source)
    with np.load(path, allow_pickle=False) as d:
        return d["E"], d["active"].astype(bool)


def cond_prob(E, active, v, t):
    """p_t(. | v) as a length-N vector. Returns all zeros if year t has no other active node."""
    logits = (E[:, t, :] @ E[v, t, :]).astype(np.float64)
    logits[v] = -np.inf
    logits[~active[:, t]] = -np.inf
    finite = np.isfinite(logits)
    if not finite.any():                      # v isolated / year empty -> no distribution
        return np.zeros_like(logits)
    logits -= logits[finite].max()
    p = np.exp(logits)
    return p / p.sum()


def log_prob_pairs(E, active, pairs, t):
    """log p_t(u | v) for each (v, u); for undirected use, average both directions."""
    out = np.empty(len(pairs))
    cache = {}
    for i, (v, u) in enumerate(pairs):
        if v not in cache:
            cache[v] = cond_prob(E, active, v, t)
        pu = cache[v][u]
        out[i] = np.log(pu) if pu > 0 else -np.inf
    return out


def perplexity(E, active, pairs, t, both_directions=True):
    """Perplexity over the given pairs in year t. NaN when no pair is scorable."""
    pairs = list(pairs)
    if not pairs:
        return float("nan")
    lp = log_prob_pairs(E, active, pairs, t)
    if both_directions:
        lp = 0.5 * (lp + log_prob_pairs(E, active, [(u, v) for v, u in pairs], t))
    lp = lp[np.isfinite(lp)]
    if lp.size == 0:
        return float("nan")
    return float(np.exp(-lp.mean()))
