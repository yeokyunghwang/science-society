from tf_compat import tf
from models.DySAT.inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def _dense1(x, out_dim, name, use_bias=True, activation=None):
    """Position-wise linear map on the last axis. Replaces tf.layers.conv1d(kernel_size=1),
    which no longer exists in TF>=2.16 compat.v1. Same init (glorot_uniform kernel, zero bias).
    Variable reuse follows the enclosing tf.variable_scope."""
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(name + '/kernel', shape=[in_dim, out_dim], dtype=tf.float32,
                        initializer=tf.glorot_uniform_initializer())
    y = tf.tensordot(x, W, axes=[[x.get_shape().ndims - 1], [0]])
    if use_bias:
        b = tf.get_variable(name + '/bias', shape=[out_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        y = y + b
    if activation is not None:
        y = activation(y)
    return y


class Layer(object):
    """Base layer class. (unchanged)"""

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class TemporalAttentionLayer(Layer):
    """ Temporal self-attention over snapshots. UNCHANGED in behaviour except:
        - tf.contrib initializer / lower-triangular op replaced by tf.compat.v1 equivalents
        - tf.layers.dropout (which had training=False => never applied) replaced by tf.nn.dropout
          that honours the temporal_drop placeholder (default 0.0 reproduces the original)."""
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim

        xavier_init = tf.glorot_uniform_initializer()                     # was tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name + '_vars'):
            if use_position_embedding:
                self.vars['position_embeddings'] = tf.get_variable('position_embeddings',
                                                                   dtype=tf.float32,
                                                                   shape=[self.num_time_steps, input_dim],
                                                                   initializer=xavier_init)  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable('Q_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)
            self.vars['K_embedding_weights'] = tf.get_variable('K_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)
            self.vars['V_embedding_weights'] = tf.get_variable('V_embedding_weights', dtype=tf.float32,
                                                               shape=[input_dim, input_dim], initializer=xavier_init)

    def __call__(self, inputs):
        """ In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]."""
        # 1: Add position embeddings to input
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs)[0], 1])
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'],
                                                          position_inputs)  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]

        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)        # (original scales by sqrt(T), kept as is)

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
        tril = tf.linalg.band_part(diag_val, -1, 0)   # was tf.contrib.linalg.LinearOperatorLowerTriangular(...).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = tf.nn.dropout(outputs, rate=self.attn_drop)   # was tf.layers.dropout(..., training=False) => no-op
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]

        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)

        # Optional: Feedforward and residual
        if FLAGS.position_ffn:
            outputs = self.feedforward(outputs)

        if self.residual:
            outputs += temporal_inputs

        return outputs

    def feedforward(self, inputs, reuse=None):
        """Point-wise feed forward net:  ReLU(W x + b) + x   (unchanged behaviour)."""
        with tf.variable_scope(self.name + '_vars', reuse=reuse):
            inputs = tf.reshape(inputs, [-1, self.num_time_steps, self.input_dim])
            outputs = _dense1(inputs, self.input_dim, name='ffn', use_bias=True, activation=tf.nn.relu)
            outputs += inputs
        return outputs


class StructuralAttentionLayer(Layer):
    """ GAT layer applied per snapshot with shared parameters.
        CHANGE (modification 1): edge weight enters as an additive log-bias outside the
        nonlinearity,  e_uv = LeakyReLU(f1_u + f2_v) + beta * log(A_uv),  instead of the
        original  e_uv = LeakyReLU(A_uv * (f1_u + f2_v)).  After softmax this gives
        alpha_uv ∝ A_uv^beta * exp(LeakyReLU(.)), which is invariant to rescaling all weights."""
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, **kwargs):
        super(StructuralAttentionLayer, self).__init__(**kwargs)
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.sparse_inputs = sparse_inputs

        # NEW: one scalar beta per layer (shared across heads and snapshots)
        with tf.variable_scope(self.name + '_vars'):
            if FLAGS.binary_adj:
                self.vars['beta'] = tf.constant(0.0, dtype=tf.float32, name='beta_fixed')
            else:
                self.vars['beta'] = tf.get_variable('beta', shape=[], dtype=tf.float32,
                                                    initializer=tf.constant_initializer(FLAGS.beta_init))

        if self.logging:
            self._log_vars()
        self.n_calls = 0

    def _call(self, inputs):
        self.n_calls += 1
        x = inputs[0]
        adj = inputs[1]
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(self.sp_attn_head(x, adj_mat=adj, in_sz=self.input_dim,
                                                out_sz=self.output_dim // self.n_heads, activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual,
                                                layer_str="l_{}_h_{}".format(self.name, j),
                                                sparse_inputs=self.sparse_inputs,
                                                reuse_scope=reuse_scope))

        h = tf.concat(attentions, axis=-1)
        return h

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return tf.maximum(alpha * features, features)          # was math_ops.maximum

    def sp_attn_head(self, seq, in_sz, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
                     layer_str="", sparse_inputs=False, reuse_scope=None):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""

        with tf.variable_scope('struct_attn', reuse=reuse_scope):
            if sparse_inputs:
                weight_var = tf.get_variable("layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz],
                                             dtype=tf.float32)
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [1, N, F]
            else:
                seq_fts = _dense1(seq, out_sz, name='layer_' + str(layer_str) + '_weight_transform', use_bias=False)

            # Additive self-attention.
            f_1 = _dense1(seq_fts, 1, name='layer_' + str(layer_str) + '_a1')     # was tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = _dense1(seq_fts, 1, name='layer_' + str(layer_str) + '_a2')
            f_1 = tf.reshape(f_1, [-1])  # [N]
            f_2 = tf.reshape(f_2, [-1])  # [N]

            # ---- modification 1 -------------------------------------------------
            # original:
            #   logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))
            #   leaky   = LeakyReLU(logits.values)                       # A_uv multiplied INSIDE
            # new: gather f1[row] + f2[col] per stored edge so values align with adj_mat.values,
            #      apply LeakyReLU, then add beta * log(A_uv) OUTSIDE the nonlinearity.
            adj_mat = tf.sparse_reorder(adj_mat)                          # canonical order
            rows = adj_mat.indices[:, 0]
            cols = adj_mat.indices[:, 1]
            score = tf.gather(f_1, rows) + tf.gather(f_2, cols)           # [E]
            score = self.leaky_relu(score)
            score = score + self.vars['beta'] * tf.log(adj_mat.values + 1e-12)
            leaky_relu = tf.SparseTensor(indices=adj_mat.indices, values=score, dense_shape=adj_mat.dense_shape)
            # ---------------------------------------------------------------------
            coefficients = tf.sparse_softmax(leaky_relu)  # [N, N] (sparse), softmax over cols within each row

            # dropout: `if coef_drop != 0.0` on a placeholder is not allowed in graph mode; rate=0 is a no-op.
            coefficients = tf.SparseTensor(indices=coefficients.indices,
                                           values=tf.nn.dropout(coefficients.values, rate=coef_drop),
                                           dense_shape=coefficients.dense_shape)  # [N, N] (sparse)
            seq_fts = tf.nn.dropout(seq_fts, rate=in_drop)  # [1, N, D]

            seq_fts = tf.reshape(seq_fts, [-1, out_sz])
            values = tf.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            values = tf.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

            if residual:
                residual_wt = tf.get_variable("layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz],
                                              dtype=tf.float32)
                if sparse_inputs:
                    ret = ret + tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, residual_wt), axis=0)
                else:
                    ret = ret + _dense1(seq, out_sz, name='layer_' + str(layer_str) + '_residual_weight', use_bias=False)
            return activation(ret)
