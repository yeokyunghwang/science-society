from __future__ import division
from __future__ import print_function

import numpy as np
from tf_compat import tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class NodeMinibatchIterator(object):
    """
    Iterates over nodes; for each node in the batch samples `max_positive` neighbours per snapshot
    with probability proportional to edge weight.  CHANGE from original: context pairs are no
    longer precomputed by random walks -- they are drawn from the (training) adjacency each step.

    adjs_train_csr -- list of scipy csr (training edges only, NO self loops) used for sampling
    adjs_feed      -- list of (indices, values, shape) tuples (with self loops) fed to the encoder
    features       -- list of (indices, values, shape) tuples
    placeholders   -- tensorflow placeholders
    num_time_steps -- number of snapshots (all are trained)
    batch_size     -- # nodes per batch
    """
    def __init__(self, adjs_train_csr, adjs_feed, features, placeholders, num_time_steps, batch_size=100):
        self.adjs_csr = adjs_train_csr
        self.adjs_feed = adjs_feed
        self.features = features
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.num_nodes = adjs_train_csr[0].shape[0]
        self.max_positive = FLAGS.max_positive                       # was FLAGS.neg_sample_size
        self.degs = self.construct_degs()
        self.active = [(d > 0).astype(np.float32) for d in self.degs]   # NEW: per-snapshot active mask [N]
        self.train_nodes = np.arange(self.num_nodes)                   # fixed vocabulary
        print("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ Node degrees (# neighbours) in each training snapshot."""
        return [np.asarray((A > 0).sum(1)).flatten() for A in self.adjs_csr]

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    # ------------------------------------------------------------------
    # NEW: positive sampling from adjacency rows
    # ------------------------------------------------------------------
    def sample_pairs(self, t, nodes):
        A = self.adjs_csr[t]
        node_1, node_2 = [], []
        for n in nodes:
            s, e = A.indptr[n], A.indptr[n + 1]
            if e == s:                        # isolated at t -> no loss contribution
                continue
            nbrs = A.indices[s:e]
            w = A.data[s:e]
            k = min(self.max_positive, e - s)
            chosen = np.random.choice(nbrs, size=k, replace=False, p=w / w.sum())
            node_1.extend([int(n)] * k)
            node_2.extend(chosen.tolist())
        return node_1, node_2

    def _time_range(self):
        min_t = 0
        if FLAGS.window > 0:
            min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        return min_t

    def base_feed_dict(self):
        """features, adjs, active masks -- identical every step."""
        min_t = self._time_range()
        fd = {}
        fd.update({self.placeholders['features'][t - min_t]: self.features[t] for t in range(min_t, self.num_time_steps)})
        fd.update({self.placeholders['adjs'][t - min_t]: self.adjs_feed[t] for t in range(min_t, self.num_time_steps)})
        fd.update({self.placeholders['active'][t - min_t]: self.active[t] for t in range(min_t, self.num_time_steps)})
        return fd

    def pairs_feed_dict(self, node_1_all, node_2_all):
        """Feed explicit (node_1, node_2) lists per snapshot (used for validation)."""
        min_t = self._time_range()
        fd = self.base_feed_dict()
        fd.update({self.placeholders['node_1'][t - min_t]: node_1_all[t] for t in range(min_t, self.num_time_steps)})
        fd.update({self.placeholders['node_2'][t - min_t]: node_2_all[t] for t in range(min_t, self.num_time_steps)})
        return fd

    def batch_feed_dict(self, batch_nodes):
        """ Feed dict with (a) sampled node pairs, (b) attribute matrices (c) snapshot adjs (d) active masks"""
        min_t = self._time_range()
        node_1_all, node_2_all = {}, {}
        for t in range(min_t, self.num_time_steps):
            node_1_all[t], node_2_all[t] = self.sample_pairs(t, batch_nodes)
        fd = self.pairs_feed_dict(node_1_all, node_2_all)
        fd.update({self.placeholders['batch_nodes']: np.array(batch_nodes).astype(np.int32)})
        return fd

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes)

    def shuffle(self):
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        self.train_nodes = np.arange(self.num_nodes)
        self.batch_num = 0
