import tensorflow as tf
import sys

class HNN(tf.keras.Model):
    """
    Hamiltonian Neural Network for an extended PM-HMC state space:
    coords = [theta (dim_theta), u (dim_u), rho (dim_theta), p (dim_u)].
    """
    def __init__(self, input_dim, mlp_model, d, D):
        """
        Args:
            input_dim: 2*(d + D), dimension of coords
            mlp_model: an MLP instance that returns a scalar
            d: dim_theta
            D: dim_u
        """
        super(HNN, self).__init__()
        self.input_dim = input_dim
        self.mlp_model = mlp_model
        self.d = d
        self.D = D

    def call(self, x):
        """
        Forward pass to compute the Hamiltonian.
        """
        return self.mlp_model(x)

    def time_derivative(self, x, t=None):
        """
        Computes dx/dt from the learned Hamiltonian.
        Parses (theta, u, rho, p) and applies Hamilton's equations:
            dot(theta) =  dH/d(rho)
            dot(u)     =  dH/d(p)
            dot(rho)   = -dH/d(theta)
            dot(p)     = -dH/d(u)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            H_vals = self.call(x)
            H_vals = tf.reshape(H_vals, [-1])

        dH = tape.gradient(H_vals, x)

        d = self.d
        D = self.D
        dH_dtheta = dH[:, 0:d]
        dH_du     = dH[:, d:d+D]
        dH_drho   = dH[:, d+D:d+D+d]
        dH_dp     = dH[:, d+D+d:]

        dot_theta = dH_drho
        dot_u     = dH_dp
        dot_rho   = -dH_dtheta
        dot_p     = -dH_du

        dxdt = tf.concat([dot_theta, dot_u, dot_rho, dot_p], axis=1)
        return dxdt