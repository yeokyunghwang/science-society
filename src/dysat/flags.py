from tf_compat import tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# ---- unchanged from original ----
flags.DEFINE_string('base_model', 'DySAT', 'Base model string.')
flags.DEFINE_string('model', 'default', 'Model string.')
flags.DEFINE_string('dataset', 'news', 'Arena name: names the output <embeddings>/<dataset>_E.npz '
                    'and, unless --src is given, the input dir <networks>/<dataset>/.')
flags.DEFINE_string('src', None, 'Dir holding adj_<year>.npz. Default: <networks>/<dataset>/.')
flags.DEFINE_string('years', '1990-2023', 'Snapshot range FIRST-LAST. Sets the number of time steps.')
flags.DEFINE_integer('GPU_ID', 0, 'GPU_ID')
flags.DEFINE_boolean('featureless', True, 'Use 1-hot instead of features')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency (epochs)')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate for self-attention model.')
flags.DEFINE_float('spatial_drop', 0.1, 'attn Dropout (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_boolean('use_residual', False, 'Residual connections')
flags.DEFINE_string('structural_layer_config', '128', 'Encoder layer config: # units in each GAT layer')
flags.DEFINE_string('temporal_head_config', '16', 'Encoder layer config: # attention heads in each temporal layer')
flags.DEFINE_string('temporal_layer_config', '128', 'Encoder layer config: # units in each temporal layer')
flags.DEFINE_boolean('position_ffn', True, 'Use position wise feedforward')
flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
flags.DEFINE_integer('seed', 7, 'Random seed')
flags.DEFINE_string('log_subdir', "log", 'Log dir inside the run dir (renamed: log_dir collides with absl)')
flags.DEFINE_string('model_dir', "model", 'Checkpoint dir inside the run dir')
flags.DEFINE_integer('window', -1, 'Window for temporal attention (default : -1 => full)')

# ---- changed defaults ----
flags.DEFINE_integer('epochs', 200, 'Max number of epochs (early stopping on val loss).')             # was 1
flags.DEFINE_integer('batch_size', 512, 'Batch size (# nodes)')                                       # unchanged value
flags.DEFINE_string('structural_head_config', '8', '# attention heads in each GAT layer')            # was '16'
# ---- attention variant ----
flags.DEFINE_string('attn_variant', 'gat',
                    "Structural attention scoring: 'gat' (Velickovic et al. 2018, static) or "
                    "'gatv2' (Brody et al. 2022, dynamic). Names the output "
                    "<embeddings>/<dataset>_E.npz vs <dataset>_gatv2_E.npz.")

flags.DEFINE_float('temporal_drop', 0.0, 'Dropout on temporal attention weights. NOTE: original '
                   'used tf.layers.dropout(training=False) => never applied; 0.0 reproduces that.')  # was 0.5

# ---- new ----
flags.DEFINE_integer('max_positive', 1, 'Neighbour samples per node per snapshot (was tied to neg_sample_size).')
flags.DEFINE_float('val_frac', 0.05, 'Fraction of edges per snapshot held out for validation loss.')
flags.DEFINE_integer('val_pairs_per_step', 512, 'Held-out pairs per snapshot used to compute validation loss.')
flags.DEFINE_integer('patience', 20, 'Early stopping patience (epochs without val-loss improvement).')
flags.DEFINE_float('beta_init', 1.0, 'Init of edge-weight log-bias coefficient beta in structural attention.')
flags.DEFINE_boolean('binary_adj', False, 'If True, ignore edge weights in structural attention (beta fixed at 0).')

# ---- removed ----
# neg_sample_size, neg_weight, walk_len, test_freq, csv_dir
# time_steps: derived from --years; ALL snapshots are trained (no held-out future step)
# save_dir: the final embeddings now go to scisoc.config.paths.embeddings,
# the one place notebook 03 reads them from. Run artefacts (log/, model/)
# live under paths.dysat_logs/<base_model>_<model>/.
