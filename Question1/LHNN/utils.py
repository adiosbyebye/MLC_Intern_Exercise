import tensorflow as tf
import pickle


def lfrog(fun, y0, t, dt, *args, **kwargs):
    k1 = fun(y0, t - dt, *args, **kwargs)
    k2 = fun(y0, t + dt, *args, **kwargs)
    dy = (k2 - k1) / (2 * dt)
    return dy

def L2_loss(u, v):
    return tf.reduce_mean(tf.square(u - v))

def to_pickle(thing, path):
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(path):
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

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