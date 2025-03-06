import unittest
import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import yaml
import sys

###############################################################################
# Data generation for logistic GLMM
###############################################################################
def generate_glmm_data_mixture(
    N=50,
    T=6,
    d_beta=8,
    D=4,
    w_true=0.5,
    mu1_true=None,
    mu2_true=None,
    lambda1_true=2.0,
    lambda2_true=3.0,
    beta_true=None,
    seed=42
):
    """
    Generates synthetic data for a logistic GLMM, drawing random effects
    from a two-component Gaussian mixture:
    b_i ~ w_true * N(mu1_true, (1/lambda1_true)I)
          + (1 - w_true) * N(mu2_true, (1/lambda2_true)I).

    Returns:
        X: shape (N, T, d_beta)
        Z: shape (N, T, D)
        Y: shape (N, T)
        b_true: shape (N, D)
        beta_true: shape (d_beta,)
    """

    # Seed for reproducibility
    np.random.seed(seed)

    # Default means if not provided
    if mu1_true is None:
        mu1_true = np.array([0.0, 0.0, 0.0, 0.0])
    if mu2_true is None:
        mu2_true = np.array([1.0, -1.0, 2.0, 0.5])

    # Default fixed effects if not provided
    if beta_true is None:
        beta_true = np.random.randn(d_beta)

    # Generate covariates
    X = np.random.randn(N, T, d_beta)

    # Random effects covariates
    Z = X[:, :, :D]

    # Mixture-based random effects
    b_true = np.zeros((N, D))
    for i in range(N):
        if np.random.rand() < w_true:
            e1 = np.random.randn(D) / np.sqrt(lambda1_true)
            b_true[i, :] = mu1_true + e1
        else:
            e2 = np.random.randn(D) / np.sqrt(lambda2_true)
            b_true[i, :] = mu2_true + e2

    # Linear predictor
    eta = np.einsum("ntd,d->nt", X, beta_true) + np.einsum("ntD,nD->nt", Z, b_true)

    # Logistic transform
    nu = 1.0 / (1.0 + np.exp(-eta))
    Y = np.random.binomial(1, nu)

    return X, Z, Y, b_true, beta_true

###############################################################################
# Config loading
###############################################################################
def load_config(config_path="./config_training_data.yaml"):
    """
    Loads configuration parameters from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

###############################################################################
# Configuration container
###############################################################################
class Config:
    """
    Stores configuration parameters as object attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Load user configuration
args = Config(**load_config())

###############################################################################
# Generate GLMM data with mixture logic
###############################################################################
X_data, Z_data, Y_data, b_true_data, beta_true_data = generate_glmm_data_mixture(
    N=args.N,
    T=args.T,
    d_beta=8,
    D=args.dim_per_subject,
    seed=args.seed
)

N_global = X_data.shape[0]
T_global = X_data.shape[1]
d_global = 8
D_subj   = args.dim_per_subject

###############################################################################
# Pseudo-marginal potential for the 2-component mixture
###############################################################################
def pm_potential_only(theta_u):
    """
    Computes the negative log posterior for the pseudo-marginal approach.
    Parses the extended parameter vector theta_u into:
      theta: beta(8), mu1(4), mu2(4), loglambda1(1), loglambda2(1), logitw(1)
      u:     S*N_global*5 draws for mixture assignments and offsets
    Returns the negative log posterior at (theta, u).
    """

    d = args.dim_theta
    S = args.S
    coords = tf.reshape(theta_u, [-1])
    theta = coords[:d]
    u     = coords[d:]

    # Extract parameters
    beta       = theta[0:8]
    mu1        = theta[8:12]
    mu2        = theta[12:16]
    loglambda1 = theta[16]
    loglambda2 = theta[17]
    logitw     = theta[18]

    # Convert parameters to real scale
    lambda1 = tf.exp(loglambda1)
    lambda2 = tf.exp(loglambda2)
    w       = tf.sigmoid(logitw)

    # Negative log-prior for theta ~ Normal(0, I)
    neg_log_prior_theta = 0.5 * tf.reduce_sum(theta**2)

    # Reshape latent variables u
    u_reshaped = tf.reshape(u, [S, N_global, 5])

    # Single-draw negative log-likelihood
    def single_draw_nll(u_s):
        u_unif = u_s[:, 0]
        u_norm = u_s[:, 1:5]

        mask = (u_unif < w)
        scale1 = tf.exp(-0.5*loglambda1)
        scale2 = tf.exp(-0.5*loglambda2)

        e1 = scale1 * u_norm
        e2 = scale2 * u_norm

        b_s = tf.where(
            tf.expand_dims(mask, axis=1),
            mu1 + e1,
            mu2 + e2
        )

        linear_term = (
            tf.einsum("ntd,d->nt", X_data, beta)
            + tf.einsum("ntD,nD->nt", Z_data, b_s)
        )
        nu_ = tf.sigmoid(linear_term)

        nll_ = -tf.reduce_sum(
            Y_data*tf.math.log(nu_+1e-9) + (1.-Y_data)*tf.math.log(1.-nu_+1e-9)
        )
        return tf.cast(nll_, tf.float32)

    nll_per_draw = tf.map_fn(single_draw_nll, u_reshaped, fn_output_signature=tf.float32)
    like_per_draw = tf.exp(-nll_per_draw)
    like_mean = tf.reduce_mean(like_per_draw)

    neg_log_data = -tf.math.log(like_mean + 1e-12)
    return neg_log_prior_theta + neg_log_data

###############################################################################
# Extended Hamiltonian function
###############################################################################
def pm_functions(coords):
    """
    Computes the extended Hamiltonian H(theta, u, rho, p).
    coords has layout: [theta(19), u(dim_u), rho(19), p(dim_u)].
    Returns H = potential + kinetic.
    """

    coords = tf.reshape(coords, [-1])
    d = args.dim_theta
    dim_u = args.dim_u

    # Split coordinates into position and momentum
    theta = coords[:d]
    u     = coords[d : d+dim_u]
    rho   = coords[d+dim_u : d+dim_u+d]
    p     = coords[d+dim_u+d : ]

    # Potential term from pseudo-marginal approach
    pot = pm_potential_only(tf.concat([theta, u], axis=0))

    # Kinetic term with 0.5||rho||^2 + 0.5||p||^2 + 0.5||u||^2
    kin = 0.5*tf.reduce_sum(rho**2) + 0.5*tf.reduce_sum(p**2) + 0.5*tf.reduce_sum(u**2)
    return pot + kin

###############################################################################
# Hamiltonian dynamics
###############################################################################
def pm_dynamics_fn(t, coords):
    """
    Computes time derivatives of the coordinates under Hamiltonian dynamics.
    Returns d/dt of [theta, u, rho, p].
    """

    coords_tf = tf.convert_to_tensor(coords, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(coords_tf)
        H_val = pm_functions(coords_tf)
    dH = tape.gradient(H_val, coords_tf)

    d = args.dim_theta
    dim_u = args.dim_u

    dH_dtheta = dH[:d]
    dH_du     = dH[d : d+dim_u]
    dH_drho   = dH[d+dim_u : d+dim_u+d]
    dH_dp     = dH[d+dim_u+d : ]

    dot_theta = dH_drho
    dot_u     = dH_dp
    dot_rho   = -dH_dtheta
    dot_p     = -dH_du

    return np.concatenate([dot_theta, dot_u, dot_rho, dot_p], axis=0)

###############################################################################
# Leapfrog integrator
###############################################################################
def pm_leapfrog_integration(dynamics_fn, t_span, y0, n_steps):
    """
    Performs leapfrog integration over the interval [t_span[0], t_span[1]]
    using n_steps sub-steps. Returns the trajectory of states.
    """

    dt = (t_span[1] - t_span[0]) / n_steps
    traj = []
    y = y0.copy()
    traj.append(y.copy())
    t = t_span[0]

    for step in range(n_steps):
        dcoords = dynamics_fn(t, y)

        d = args.dim_theta
        dim_u = args.dim_u
        pos_dim = d + dim_u

        # Split position and momentum
        pos = y[:pos_dim]
        mom = y[pos_dim:]
        dot_pos = dcoords[:pos_dim]
        dot_mom = dcoords[pos_dim:]

        # Half-step momentum
        mom_half = mom + 0.5*dt*dot_mom

        # Drift in position
        y_mid = np.concatenate([pos, mom_half], axis=0)
        dcoords_mid = dynamics_fn(t + 0.5*dt, y_mid)
        dot_pos_mid = dcoords_mid[:pos_dim]
        pos_new = pos + dt*dot_pos_mid

        # Final half-step in momentum
        y_mid2 = np.concatenate([pos_new, mom_half], axis=0)
        dcoords_mid2 = dynamics_fn(t + dt, y_mid2)
        dot_mom_mid2 = dcoords_mid2[pos_dim:]
        mom_new = mom_half + 0.5*dt*dot_mom_mid2

        # Update full state
        y = np.concatenate([pos_new, mom_new], axis=0)
        traj.append(y.copy())
        t += dt

    return np.array(traj)

###############################################################################
# Single-trajectory generation for HNN data
###############################################################################
def get_pm_trajectory_unified(y0=None):
    """
    Generates a single trajectory of states by applying leapfrog integrator
    to the pseudo-marginal Hamiltonian system.
    """

    t_span = [0., args.len_sample]
    n_steps = args.n_steps
    if y0 is None:
        total_dim = 2*(args.dim_theta + args.dim_u)
        y0 = np.random.randn(total_dim)

    traj = pm_leapfrog_integration(pm_dynamics_fn, t_span, y0, n_steps)
    dydt = np.array([pm_dynamics_fn(None, traj[i]) for i in range(traj.shape[0])])
    return traj, dydt

###############################################################################
# Utilities for saving/loading
###############################################################################
def to_pickle(obj, path):
    """
    Saves an object to a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def from_pickle(path):
    """
    Loads an object from a pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

###############################################################################
# Main data generation routine (with acceptance) for HNN training
###############################################################################
def get_dataset(seed=0):
    """
    Generates or loads a dataset of (coords, dcoords) from pseudo-marginal HMC
    short trajectories with Metropolis acceptance. Shuffles and splits data into
    training/testing sets, then saves the results to a pickle file.
    """

    np.random.seed(seed)
    samples = args.num_samples
    test_split = 1.0 - args.test_fraction

    if args.should_load:
        path = f"{args.load_dir}/{args.load_file_name}.pkl"
        data = from_pickle(path)
        print("Successfully loaded data")
    else:
        data = {}
        xs, dxs = [], []
        y_init = np.random.randn(2*(args.dim_theta + args.dim_u))
        print("Generating PM-HMC samples (importance-sampling) for HNN training ...")

        for s in range(samples):
            print(f"Sample {s + 1}/{samples}")
            lp_traj, dydt = get_pm_trajectory_unified(y0=y_init)
            xs.append(lp_traj)
            dxs.append(dydt)

            initial_state = lp_traj[0,:]
            final_state   = lp_traj[-1,:]
            H_init  = pm_functions(tf.convert_to_tensor(initial_state, dtype=tf.float32))
            H_final = pm_functions(tf.convert_to_tensor(final_state,   dtype=tf.float32))

            accept_prob = np.exp(H_init - H_final)
            if np.random.rand() < accept_prob:
                accepted_state = final_state
            else:
                accepted_state = initial_state

            d_ = args.dim_theta
            dim_u_ = args.dim_u
            new_state = np.zeros_like(accepted_state)
            new_state[:(d_ + dim_u_)] = accepted_state[:(d_ + dim_u_)]
            new_state[(d_ + dim_u_):] = np.random.randn(d_ + dim_u_)

            y_init = new_state

            data['coords']  = np.concatenate(xs, axis=0)
            data['dcoords'] = np.concatenate(dxs, axis=0)

            N_data = data['coords'].shape[0]
            perm = np.random.permutation(N_data)
            split_ix = int(N_data * test_split)

            train_ix = perm[:split_ix]
            test_ix  = perm[split_ix:]

            split_data = {}
            split_data['coords']      = data['coords'][train_ix]
            split_data['dcoords']     = data['dcoords'][train_ix]
            split_data['test_coords'] = data['coords'][test_ix]
            split_data['test_dcoords']= data['dcoords'][test_ix]

            data = split_data
            path = f"{args.dist_name}.pkl"
            to_pickle(data, path)

    return data

###############################################################################
# Unit tests for code functionality
###############################################################################
class TestFunctions(unittest.TestCase):
    def test_generate_glmm_data_mixture_shapes(self):
        """
        Tests whether generate_glmm_data_mixture returns outputs with
        expected shapes.
        """
        X, Z, Y, b_true, beta_true = generate_glmm_data_mixture(
            N=10, T=5, d_beta=3, D=2, w_true=0.5, seed=123
        )
        self.assertEqual(X.shape, (10, 5, 3))
        self.assertEqual(Z.shape, (10, 5, 2))
        self.assertEqual(Y.shape, (10, 5))
        self.assertEqual(b_true.shape, (10, 2))
        self.assertEqual(beta_true.shape, (3,))

    def test_pm_potential_only_output(self):
        """
        Tests whether pm_potential_only runs without error
        and returns a valid tensor.
        """
        d = args.dim_theta
        S = args.S
        dim_u = 5 * S * N_global
        total_dim = d + dim_u
        theta_u = tf.ones([total_dim], dtype=tf.float32)
        result = pm_potential_only(theta_u)
        self.assertTrue(isinstance(result, tf.Tensor))

    def test_pm_functions_output(self):
        """
        Tests whether pm_functions runs without error
        and returns a valid tensor.
        """
        d = args.dim_theta
        dim_u = args.dim_u
        coords = tf.ones([d + dim_u + d + dim_u], dtype=tf.float32)
        result = pm_functions(coords)
        self.assertTrue(isinstance(result, tf.Tensor))

###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    # If an argument "test" is passed, run the tests.
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        data = get_dataset(seed=args.seed)
        coords_train = data['coords']
        dcoords_train = data['dcoords']
        coords_test  = data['test_coords']
        dcoords_test = data['test_dcoords']

        print(f"Train coords shape: {coords_train.shape}")
        print(f"Train dcoords shape: {dcoords_train.shape}")
        print(f"Test coords shape:  {coords_test.shape}")
        print(f"Test dcoords shape: {dcoords_test.shape}")
        print("Done data generation and plotting.")
