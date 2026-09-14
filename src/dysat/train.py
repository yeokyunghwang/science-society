from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import scipy.sparse as sp

from _repo import paths
from flags import *
from tf_compat import tf
from models.DySAT.models import DySAT
from utils.minibatch import NodeMinibatchIterator
from utils.preprocess import load_graphs, preprocess_features, adj_with_selfloop_raw, split_edges

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

# ---------------------------------------------------------------------------
# directories / logging  (JSON flag override and csv result files removed)
# every path comes from scisoc.config.paths -- nothing is hard-coded
# ---------------------------------------------------------------------------
run_dir = paths.dysat_logs / "{}_{}".format(FLAGS.base_model, FLAGS.model)
LOG_DIR = run_dir / FLAGS.log_subdir
MODEL_DIR = run_dir / FLAGS.model_dir
SAVE_DIR = paths.embeddings                       # final E lands where notebook 03 reads it
for d in (run_dir, LOG_DIR, MODEL_DIR, SAVE_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)
today = datetime.today()
log_file = str(LOG_DIR / "{}_{}_{}_{}_{}.log".format(
    FLAGS.dataset.split("/")[0], today.year, today.month, today.day, FLAGS.time_steps))
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logging.info(FLAGS.flag_values_dict().items())
print("run dir  ", run_dir)
print("output   ", SAVE_DIR)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
num_time_steps = FLAGS.time_steps
graphs, adjs = load_graphs(FLAGS.dataset)
assert num_time_steps <= len(adjs), "time_steps ({}) > number of snapshots ({})".format(
    num_time_steps, len(adjs))                                  # was `< len+1` (t+1 prediction)
adjs = adjs[:num_time_steps]
N = adjs[0].shape[0]
for a in adjs:
    assert a.shape[0] == N, "fixed vocabulary expected: every snapshot must have the same node set"

# REMOVED: get_context_pairs (random walks)
# REMOVED: get_evaluation_data (link-prediction train/val/test split)
# REMOVED: overwrite of graphs[T-1] with edges of graphs[T-2]

# Edge split per snapshot -> training adjacency (encoder + positive sampling) and held-out pairs
adjs_train_csr, val_pairs = [], []
for t, a in enumerate(adjs):
    a_tr, vp = split_edges(a, FLAGS.val_frac, seed=FLAGS.seed + t)
    adjs_train_csr.append(a_tr)
    val_pairs.append(vp)
    print("t={:2d}  edges(train)={:9d}  edges(val)={:7d}".format(t, a_tr.nnz // 2, len(vp) // 2))

adjs_feed_train = [adj_with_selfloop_raw(a) for a in adjs_train_csr]   # was normalize_graph_gcn
adjs_feed_full = [adj_with_selfloop_raw(a) for a in adjs]              # for the final embeddings
active_full = np.stack([(np.asarray((sp.csr_matrix(a) > 0).sum(1)).flatten() > 0) for a in adjs], axis=1)  # [N, T]

# 1-hot features: build the identity ONCE and reuse the same tuple for every snapshot
if FLAGS.featureless:
    feat_tuple = preprocess_features(sp.identity(N, format='csr'))[1]
    feats_train = [feat_tuple] * num_time_steps
    num_features = N
else:
    raise NotImplementedError("only featureless=True is wired up")
num_features_nonzero = [x[1].shape[0] for x in feats_train]


def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for _ in
                 range(min_t, num_time_steps)],
        'active': [tf.placeholder(tf.float32, shape=(None,), name="active") for _ in         # NEW
                   range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders


print("Initializing session")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

placeholders = construct_placeholders(num_time_steps)

minibatchIterator = NodeMinibatchIterator(adjs_train_csr, adjs_feed_train, feats_train, placeholders,
                                          num_time_steps, batch_size=FLAGS.batch_size)
print("# training batches per epoch", minibatchIterator.num_training_batches())

model = DySAT(placeholders, num_features, num_features_nonzero, num_nodes=N)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
ckpt_path = str(MODEL_DIR / "best.ckpt")

# diagnostics ops
T = num_time_steps
attn0 = model.attn_wts_all[0]                                   # [h*N, T, T]
n_heads_t = model.temporal_head_config[0]
temporal_attn_mean = tf.reduce_mean(tf.reshape(attn0, [n_heads_t, -1, T, T]), axis=1)   # [h, T, T]
beta_var = model.structural_attention_layers[0].vars['beta']
pos_emb = model.temporal_attention_layers[0].vars['position_embeddings']

# fixed validation subset per snapshot (same pairs every epoch)
rng = np.random.RandomState(FLAGS.seed)
val_node_1, val_node_2 = {}, {}
n_val_total = 0
for t in range(num_time_steps):
    vp = val_pairs[t]
    if len(vp) > FLAGS.val_pairs_per_step:
        vp = vp[rng.choice(len(vp), FLAGS.val_pairs_per_step, replace=False)]
    val_node_1[t] = vp[:, 0].tolist()
    val_node_2[t] = vp[:, 1].tolist()
    n_val_total += len(vp)
assert n_val_total > 0, ("no validation pairs: val_frac={} is too small for this graph. "
                         "Raise --val_frac or lower --patience expectations.".format(FLAGS.val_frac))
val_feed = minibatchIterator.pairs_feed_dict(val_node_1, val_node_2)
val_feed.update({placeholders['spatial_drop']: 0.0, placeholders['temporal_drop']: 0.0})

# ---------------------------------------------------------------------------
# training loop with early stopping on validation loss
# ---------------------------------------------------------------------------
best_val, best_epoch, bad_epochs = np.inf, -1, 0
history = defaultdict(list)

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss, it, epoch_time = 0.0, 0, 0.0
    while not minibatchIterator.end():
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop,
                          placeholders['temporal_drop']: FLAGS.temporal_drop})
        t0 = time.time()
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                       feed_dict=feed_dict)
        epoch_time += time.time() - t0
        logging.info("Mini batch Iter: {} train_loss= {:.5f} graph_loss= {:.5f} reg_loss= {:.5f}".format(
            it, train_cost, graph_cost, reg_cost))
        epoch_loss += train_cost
        it += 1
    epoch_loss /= max(it, 1)
    history['train_loss'].append(epoch_loss)

    if (epoch + 1) % FLAGS.val_freq == 0:
        val_loss, val_per_t, beta_val = sess.run([model.graph_loss, model.graph_loss_per_t, beta_var],
                                                 feed_dict=val_feed)
        history['val_loss'].append(val_loss)
        history['val_loss_per_t'].append(val_per_t)
        msg = "Epoch {:3d}  train {:.4f}  val {:.4f}  (PP_val ~ {:.1f})  beta {:.3f}  time {:.1f}s".format(
            epoch, epoch_loss, val_loss, np.exp(val_loss), beta_val, epoch_time)
        print(msg)
        logging.info(msg)
        logging.info("val loss per t: " + " ".join("{:.3f}".format(x) for x in val_per_t))

        if val_loss < best_val - 1e-4:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            saver.save(sess, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= FLAGS.patience:
                print("Early stopping at epoch {} (best epoch {}, val {:.4f})".format(epoch, best_epoch, best_val))
                break

# ---------------------------------------------------------------------------
# final embeddings: restore best parameters, forward pass on the FULL adjacency (val edges included)
# ---------------------------------------------------------------------------
if best_epoch >= 0:
    saver.restore(sess, ckpt_path)
else:
    # validation never improved on the initial value -> no checkpoint was written.
    print("WARNING: validation loss never improved; exporting the last parameters instead.")
    logging.warning("no checkpoint written; exporting last parameters")

full_feed = {}
full_feed.update({placeholders['features'][t]: feats_train[t] for t in range(num_time_steps)})
full_feed.update({placeholders['adjs'][t]: adjs_feed_full[t] for t in range(num_time_steps)})
full_feed.update({placeholders['spatial_drop']: 0.0, placeholders['temporal_drop']: 0.0})

E, attn_mean, beta_val, P = sess.run([model.final_output_embeddings, temporal_attn_mean, beta_var, pos_emb],
                                     feed_dict=full_feed)                       # E: [N, T, F]  (was [:, T-2, :])
out_path = SAVE_DIR / "{}_E.npz".format(FLAGS.dataset.replace('/', '_'))
np.savez(out_path, E=E.astype(np.float32), active=active_full, beta=beta_val,
         temporal_attn_mean=attn_mean, position_embeddings=P,
         best_epoch=best_epoch, best_val_loss=best_val,
         train_loss=np.array(history['train_loss']),
         val_loss=np.array(history['val_loss']),
         val_loss_per_t=np.array(history['val_loss_per_t']))
print("Saved", out_path, "E shape", E.shape, "beta", beta_val, "best epoch", best_epoch)
