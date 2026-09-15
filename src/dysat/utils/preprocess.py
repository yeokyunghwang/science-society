from __future__ import print_function
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # <repo>/src -> scisoc
from scisoc.config import paths

from tf_compat import tf

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)


# ---------------------------------------------------------------------------
# unchanged (except list(map) for py3)
# ---------------------------------------------------------------------------
def load_graphs(src, years):
    """Per-year adjacency matrices, read straight from `adj_<year>.npz`.

    CHANGE: the original unpickled a list of networkx graphs from
    `data/<dataset>/graphs.npz` and immediately converted them back with
    `nx.adjacency_matrix`. Both ends of that round trip were scipy sparse and the
    networkx objects themselves were never used (they fed the random walks and the
    link-prediction eval, both removed), so the conversion -- and the prepare_data.py
    step that produced graphs.npz -- are gone.

    Every snapshot is N x N on the same node set, so concepts that do not occur in a
    given year stay in the matrix as empty rows; train.py marks them inactive.

    Returns a list of csr matrices, one per year, self-loops removed.
    """
    src = Path(src)
    adjs, N = [], None
    for y in years:
        f = src / "adj_{}.npz".format(y)
        if not f.is_file():
            raise FileNotFoundError(
                "{} not found. Point --src at the directory holding adj_<year>.npz "
                "(notebook 01 writes them to <networks>/<source>/).".format(f))
        A = sp.load_npz(f).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        if N is None:
            N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError(
                "fixed vocabulary expected: {} is {} but the first snapshot was ({}, {})"
                .format(f, A.shape, N, N))
        adjs.append(A)
        print("{}  nodes {}  edges {}".format(y, N, A.nnz // 2))
    if not adjs:
        raise ValueError("no snapshots in range {}".format(list(years)))
    return adjs


# REMOVED: load_feats (features.npz was never produced; featureless=True is the only
#          wired-up path) and dataset_dir (there is no intermediate dataset dir now).


def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict).
    CHANGE: indices are put in canonical row-major order so that tf.sparse_softmax /
    the log-weight bias in the structural layer see values in the same order as indices."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        order = np.lexsort((mx.col, mx.row))                      # NEW: canonical order
        coords = np.vstack((mx.row[order], mx.col[order])).transpose()
        values = mx.data[order].astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation.
    CHANGE: no .todense() (29,312 x 29,312 dense = 6.9 GB)."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return None, sparse_to_tuple(features)


# ---------------------------------------------------------------------------
# REPLACES normalize_graph_gcn
# ---------------------------------------------------------------------------
def adj_with_selfloop_raw(adj, self_loop='mean'):
    """Raw edge weights + self-loop. NO GCN symmetric normalization.

    Rationale: the structural layer now uses  e_uv = LeakyReLU(f1_u + f2_v) + beta*log(A_uv).
    softmax over N_v makes any per-row scaling of A cancel, so no normalization is needed;
    D^-1/2 A D^-1/2 would inject a -0.5*log(deg_u) term that does not cancel (degree penalty).

    self_loop: 'mean' -> row mean of edge weights (log-bias = 0 relative to the average neighbour)
               'max'  -> row max
               'one'  -> 1.0
    Isolated nodes get self-loop weight 1.0 (only entry in their row; softmax gives 1).
    Returns tuple (indices, values, shape) in canonical order.
    """
    adj = sp.csr_matrix(adj, dtype=np.float64)
    adj.setdiag(0)
    adj.eliminate_zeros()
    deg = np.asarray((adj > 0).sum(1)).flatten()
    if self_loop == 'mean':
        rowsum = np.asarray(adj.sum(1)).flatten()
        self_w = np.where(deg > 0, rowsum / np.maximum(deg, 1), 1.0)
    elif self_loop == 'max':
        self_w = np.where(deg > 0, adj.max(axis=1).toarray().flatten(), 1.0)
    else:
        self_w = np.ones(adj.shape[0])
    adj_ = (adj + sp.diags(self_w)).tocoo()
    return sparse_to_tuple(adj_)


# ---------------------------------------------------------------------------
# NEW: edge split for validation loss (replaces get_evaluation_data)
# ---------------------------------------------------------------------------
def split_edges(adj, val_frac, seed=123):
    """Hold out a fraction of undirected edges from one snapshot.

    Returns
      adj_train : scipy csr, symmetric, held-out edges removed, no self loops
      val_pairs : int array [M, 2] of (v, u) with both directions of every held-out edge
    The encoder (structural attention) and positive sampling both use adj_train, so the
    validation loss asks whether the model generalizes to co-occurrences it never saw.
    """
    rng = np.random.RandomState(seed)
    adj = sp.csr_matrix(adj)
    adj.setdiag(0)
    adj.eliminate_zeros()
    triu = sp.triu(adj, k=1).tocoo()
    n_edges = triu.nnz
    n_val = int(round(val_frac * n_edges))
    if n_val == 0:
        return adj, np.zeros((0, 2), dtype=np.int64)
    perm = rng.permutation(n_edges)
    val_idx = perm[:n_val]
    vr, vc = triu.row[val_idx], triu.col[val_idx]
    # remove both directions
    mask = sp.csr_matrix((np.ones(2 * n_val), (np.concatenate([vr, vc]), np.concatenate([vc, vr]))),
                         shape=adj.shape)
    adj_train = adj - adj.multiply(mask)
    adj_train.eliminate_zeros()
    val_pairs = np.stack([np.concatenate([vr, vc]), np.concatenate([vc, vr])], axis=1).astype(np.int64)
    return adj_train.tocsr(), val_pairs


# ---------------------------------------------------------------------------
# REMOVED: get_context_pairs, get_context_pairs_incremental, get_evaluation_data,
#          create_data_splits, normalize_graph_gcn  (random walks / link prediction protocol)
# ---------------------------------------------------------------------------
