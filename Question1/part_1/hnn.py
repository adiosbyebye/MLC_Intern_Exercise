import tensorflow as tf
import sys

class HNN(tf.keras.Model):
    '''
    Latent Hamiltonian Neural Network (L-HNN):
    The network returns a set of latent variables λ (a vector), from which we compute
    the Hamiltonian H_θ = sum_i λ_i for each data sample.
    Gradients are computed with respect to q, p (contained in x).
    '''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.input_dim = input_dim
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Symplectic structure matrix
        self.field_type = field_type

    def call(self, x):
        # The differentiable model (MLP) outputs λ ∈ R^(latent_dim)
        # Shape: (batch_size, latent_dim)
        λ = self.differentiable_model(x)
        return λ

    def time_derivative(self, x, t=None):
        '''
        Compute dx/dt given x:
        1. Compute λ = MLP(x).
        2. Compute H_θ = sum(λ_i) over the latent dimension for each sample.
        3. Compute dH/dx = gradient of H_θ w.r.t. x.
        4. Use dx/dt = M * dH/dx.
        '''
        # Just ensure x is a Tensor/Variable from outside code.
        if tf.reduce_any(tf.math.is_nan(x)):
            print("x contains NaN. Exiting program.")
            sys.exit()

        with tf.GradientTape() as tape:
            tape.watch(x)
            λ = self.call(x)              # shape: (batch, latent_dim)
            Hθ = tf.reduce_sum(λ, axis=1) # shape: (batch,)

        if tf.reduce_any(tf.math.is_nan(λ)):
            print("λ contains NaN. Exiting program.")
            sys.exit()
        # Compute the gradient of H_θ w.r.t x.
        # This gives dH/dx for each sample in the batch.
        dH = tape.gradient(Hθ, x)  # shape: (batch, input_dim)

        # For Hamiltonian systems: dx/dt = M * dH (M is the symplectic form).
        dxdt = tf.matmul(dH, tf.transpose(self.M))
        return dxdt

    def permutation_tensor(self, n):
        # Construct the symplectic form for canonical coords:
        half = n // 2
        top = tf.concat([tf.zeros((half, half)), tf.eye(half)], axis=1)
        bottom = tf.concat([-tf.eye(half), tf.zeros((half, half))], axis=1)
        M = tf.concat([top, bottom], axis=0)
        return M