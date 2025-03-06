import tensorflow as tf
from utils import choose_nonlinearity


class MLP(tf.keras.Model):
    '''
    MLP for L-HNN:
    Inputs: {q, p} in R^(input_dim)
    Outputs: λ in R^(latent_dim)
    The Hamiltonian is then computed as sum(λ_i).
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine'):
        super(MLP, self).__init__()
        self.nonlinearity = choose_nonlinearity(nonlinearity)

        self.linear1 = tf.keras.layers.Dense(hidden_dim,
                                             kernel_initializer=tf.keras.initializers.Orthogonal(),
                                             input_shape=(input_dim,))
        self.linear2 = tf.keras.layers.Dense(hidden_dim,
                                             kernel_initializer=tf.keras.initializers.Orthogonal())
        self.linear3 = tf.keras.layers.Dense(hidden_dim,
                                             kernel_initializer=tf.keras.initializers.Orthogonal())
        self.linear4 = tf.keras.layers.Dense(hidden_dim,
                                             kernel_initializer=tf.keras.initializers.Orthogonal())
        # The final layer outputs λ, a vector in R^(latent_dim)
        self.linear5 = tf.keras.layers.Dense(output_dim, use_bias=True,
                                             kernel_initializer=tf.keras.initializers.Orthogonal())

    def call(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        h = self.nonlinearity(self.linear4(h))
        λ = self.linear5(h)  # shape: (batch_size, latent_dim)
        return λ