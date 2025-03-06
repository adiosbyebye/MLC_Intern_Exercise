import tensorflow as tf
import pickle

def L2_loss(a, b):
    """
    Computes the L2 loss (mean squared error) between two tensors.
    """
    return tf.reduce_mean(tf.square(a - b))

def choose_nonlinearity(name):
    if name == 'tanh':
        return tf.math.tanh
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'sigmoid':
        return tf.math.sigmoid
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'selu':
        return tf.nn.selu
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'swish':
        return lambda x: x * tf.math.sigmoid(x)
    elif name == 'sine':
        return tf.math.sin
    else:
        raise ValueError("Nonlinearity not recognized")