import tensorflow as tf
from utils import choose_nonlinearity


class MLP(tf.keras.Model):
    """
    MLP model to approximate the Hamiltonian H_\phi(\theta, u, rho, p).
    Input dimension: 2*(d + D)
    Output dimension: 1 by default, representing a scalar Hamiltonian.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine'):
        super(MLP, self).__init__()
        self.nonlinearity = choose_nonlinearity(nonlinearity)
        self.linear1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='orthogonal')
        self.linear2 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='orthogonal')
        self.linear3 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='orthogonal')
        self.linear4 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='orthogonal')
        self.linear5 = tf.keras.layers.Dense(output_dim, use_bias=True, kernel_initializer='orthogonal')

    def call(self, x):
        """
        Forward pass of the MLP. Applies four hidden layers with the specified
        nonlinearity, then projects to output_dim.
        """
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        h = self.nonlinearity(self.linear4(h))
        out = self.linear5(h)
        return out