from models.DySAT.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


class DySAT(Model):
    def _accuracy(self):
        pass

    def __init__(self, placeholders, num_features, num_features_nonzero, num_nodes, **kwargs):
        super(DySAT, self).__init__(**kwargs)
        self.attn_wts_all = []
        self.temporal_attention_layers = []
        self.structural_attention_layers = []
        self.placeholders = placeholders
        if FLAGS.window < 0:
            self.num_time_steps = len(placeholders['features'])
        else:
            self.num_time_steps = min(len(placeholders['features']), FLAGS.window + 1)
        self.num_time_steps_train = self.num_time_steps          # was num_time_steps - 1 (last step held out for prediction)
        self.num_features = num_features
        self.num_features_nonzero = num_features_nonzero
        self.num_nodes = num_nodes                                # NEW (replaces `degrees`, which fed the negative sampler)
        self.structural_head_config = list(map(int, FLAGS.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, FLAGS.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, FLAGS.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, FLAGS.temporal_layer_config.split(",")))
        self._build()

    def _build(self):
        # REMOVED: proximity_neg_samples / tf.nn.fixed_unigram_candidate_sampler (no negative sampling)

        # Build actual model.
        self.final_output_embeddings = self.build_net(self.structural_head_config, self.structural_layer_config,
                                                      self.temporal_head_config,
                                                      self.temporal_layer_config,
                                                      self.placeholders['spatial_drop'],
                                                      self.placeholders['temporal_drop'],
                                                      self.placeholders['adjs'])
        self._loss()
        self.init_optimizer()

    def build_net(self, attn_head_config, attn_layer_config, temporal_head_config, temporal_layer_config,
                  spatial_drop, temporal_drop, adjs):
        """UNCHANGED."""
        input_dim = self.num_features
        sparse_inputs = True

        # 1: Structural Attention Layers
        for i in range(0, len(attn_layer_config)):
            if i > 0:
                input_dim = attn_layer_config[i - 1]
                sparse_inputs = False
            self.structural_attention_layers.append(StructuralAttentionLayer(input_dim=input_dim,
                                                                             output_dim=attn_layer_config[i],
                                                                             n_heads=attn_head_config[i],
                                                                             attn_drop=spatial_drop,
                                                                             ffd_drop=spatial_drop,
                                                                             act=tf.nn.elu,
                                                                             sparse_inputs=sparse_inputs,
                                                                             residual=False))
        # 2: Temporal Attention Layers
        input_dim = attn_layer_config[-1]
        for i in range(0, len(temporal_layer_config)):
            if i > 0:
                input_dim = temporal_layer_config[i - 1]
            temporal_layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=temporal_head_config[i],
                                                    attn_drop=temporal_drop, num_time_steps=self.num_time_steps,
                                                    residual=False)
            self.temporal_attention_layers.append(temporal_layer)

        # 3: Structural Attention forward
        input_list = self.placeholders['features']  # List of t feature matrices. [N x F]
        for layer in self.structural_attention_layers:
            attn_outputs = []
            for t in range(0, self.num_time_steps):
                out = layer([input_list[t], adjs[t]])
                attn_outputs.append(out)  # A list of [1x Ni x F]
            input_list = list(attn_outputs)

        # 4: Pack embeddings across snapshots. (zero padding is a no-op with a fixed vocabulary)
        for t in range(0, self.num_time_steps):
            zero_padding = tf.zeros(
                [1, tf.shape(attn_outputs[-1])[1] - tf.shape(attn_outputs[t])[1], attn_layer_config[-1]])
            attn_outputs[t] = tf.concat([attn_outputs[t], zero_padding], axis=1)

        structural_outputs = tf.transpose(tf.concat(attn_outputs, axis=0), [1, 0, 2])  # [N, T, F]
        structural_outputs = tf.reshape(structural_outputs,
                                        [-1, self.num_time_steps, attn_layer_config[-1]])  # [N, T, F]

        # 5: Temporal Attention forward
        temporal_inputs = structural_outputs
        for temporal_layer in self.temporal_attention_layers:
            outputs = temporal_layer(temporal_inputs)  # [N, T, F]
            temporal_inputs = outputs
            self.attn_wts_all.append(temporal_layer.attn_wts_all)
        return outputs

    def _loss(self):
        """modification 2: full-softmax cross-entropy over all active nodes of the snapshot.

        original (per t):
            pos = <e_v, e_u>,  neg = <e_v, e_u'> for 10 sampled u' ~ deg^0.75
            loss = BCE(pos, 1) + w_n * BCE(-neg, 1)                       # two sigmoid terms, not normalized
        new (per t):
            logits = E[v, t, :] . E[:, t, :]^T                            # [B, N]
            logits[v] = -inf ; logits[inactive at t] = -inf
            loss = -log softmax(logits)[u]                                 # normalized over active nodes
        """
        self.graph_loss = tf.constant(0.0)
        self.graph_loss_per_t = []
        E = self.final_output_embeddings                                   # [N, T, F]
        for t in range(self.num_time_steps_train):
            E_t = E[:, t, :]                                               # [N, F]
            node_1 = self.placeholders['node_1'][t]                        # [B]
            node_2 = self.placeholders['node_2'][t]                        # [B]
            q = tf.nn.embedding_lookup(E_t, node_1)                        # [B, F]
            logits = tf.matmul(q, E_t, transpose_b=True)                   # [B, N]
            # mask self (tied embeddings => <e_v,e_v> is always the max) and inactive nodes
            self_mask = tf.one_hot(node_1, self.num_nodes, on_value=-1e9, off_value=0.0, dtype=tf.float32)
            inactive = (1.0 - self.placeholders['active'][t]) * -1e9       # [N]
            logits = logits + self_mask + inactive[None, :]
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=node_2, logits=logits)   # [B]
            n = tf.maximum(tf.cast(tf.shape(ce)[0], tf.float32), 1.0)    # guard empty batch at t
            loss_t = tf.reduce_sum(ce) / n
            self.graph_loss_per_t.append(loss_t)
            self.graph_loss += loss_t
        self.graph_loss = self.graph_loss / float(self.num_time_steps_train)   # mean over snapshots (orig: sum)

        self.reg_loss = tf.constant(0.0)
        reg_vars = [v for v in tf.trainable_variables() if "struct_attn" in v.name and "bias" not in v.name]
        if len(reg_vars) > 0:
            self.reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in reg_vars]) * FLAGS.weight_decay
        self.loss = self.graph_loss + self.reg_loss

    def init_optimizer(self):
        """UNCHANGED."""
        trainable_params = tf.trainable_variables()
        actual_loss = self.loss
        gradients = tf.gradients(actual_loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))
