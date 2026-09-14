"""TF1-style API on either TF 1.x or TF 2.x (compat.v1). Import `tf` from here everywhere."""
import tensorflow as _tf

if int(_tf.__version__.split('.')[0]) >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    tf = _tf
